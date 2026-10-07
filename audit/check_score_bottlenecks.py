"""Bounded mechanism checks, synthetic CPU tensors and read-only epoch420 JSONL."""
from pathlib import Path
import sys
import json
import hashlib
import importlib.metadata
from types import SimpleNamespace
from unittest.mock import patch
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import numpy as np
import tensorflow as tf
from test_fixes import config
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_mask_quality import component_quality_targets
from cytonucs_inference import CytoNucsPredictor
from cytonucs_refinement_audit import RefinementStageAudit
import cytonucs_inference as inference


def prompt_exposure():
    cfg=config(predicted_prompt_fraction=1.,predicted_prompt_sample_peaks=True)
    shape=(8,32,32);x=np.full((1,)+shape+(5,),.123,np.float32)
    prob=np.zeros((1,8,16,16,1),np.float32);prob[0,3,6,6,0]=.9
    pred={h+'_prob':tf.constant(prob) for h in ('nucleus','cytoplasm')}
    pred.update({h+'_dist':tf.ones((1,8,16,16,12))*4 for h in ('nucleus','cytoplasm')})
    target=np.zeros(shape,np.int32);target[2:5,8:18,8:18]=1
    stub=SimpleNamespace(config=cfg,prompt_detector=lambda x,training:pred,
                         optimizer=SimpleNamespace(iterations=tf.Variable(0,dtype=tf.int64)))
    result={}
    for paired in (False,True):
        y={}
        for h in ('nucleus','cytoplasm'):
            y[h+'_target']=tf.constant(np.repeat(target[None],2,axis=0),tf.float32)
            y[h+'_instances']=tf.constant(target[None])
            y[h+'_z_prompt_pair']=tf.constant([paired])
        out=CytoNucsTrainer._predicted_prompts(stub,tf.constant(x),y).numpy()
        result[str(paired)]=sum(not np.array_equal(out[...,i],x[...,i]) for i in range(1,5))
    assert result=={'False':4,'True':0}
    return result


def roi_limit():
    cfg=config(nucleus_grid=(1,1,1),cytoplasm_grid=(1,1,1),
               refinement_patch_size=[0,32,32],refinement_adaptive_roi=True)
    shape=(8,32,96);gt=np.zeros(shape,np.int32);gt[:,6:26,8:88]=1
    image=(gt>0).astype(np.float32)[...,None];ctr=np.array([[4,16,48]])
    anchors=np.zeros(shape,np.int32);anchors[tuple(ctr[0])]=1
    def refine(inputs):
        binary=inputs[0][...,0]
        logits=tf.stack([10*(1-binary),10*binary,tf.zeros_like(binary)],axis=-1)
        return logits,logits
    result={}
    for full_x in (False,True):
        cfg.refinement_patch_size=[0,32,0 if full_x else 32]
        observers={'cytoplasm':RefinementStageAudit(gt)}
        with patch.object(inference,'_refinement_runners',return_value=(lambda x:[x]*7,refine)), \
             patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda x:refine(x)[1]):
            n,c,p=inference.predict_instances_prompted(SimpleNamespace(),cfg,image,ctr,np.ones(1),
                nuc_dists=np.full((1,12),2.),nuc_labels=anchors,stage_audit=observers,verbose=False)
        report=observers['cytoplasm'].report()
        result[str(full_x)]=dict(roi=report['rois'][0]['shape'],
            best_iou=report['stages']['before_nms']['best_iou_per_gt']['1'],
            matches=report['stages']['final']['oracle_one_to_one_tp'])
    assert abs(result['False']['best_iou']-.4)<1e-6
    assert result['True']['best_iou']==1.
    return result


def cropped_quality():
    shape=(8,32,32);axes=np.indices(shape,dtype=np.float32)
    prompt=np.exp(-.5*sum(((axes[a]-[4,16,16][a])/[2,8,8][a])**2 for a in range(3)))
    target=np.ones(shape,bool)
    quality=component_quality_targets(np.ones((1,)+shape+(1,),np.float32),
        target[None],prompt[None],(1,1,1),.45)[0]
    # The crop occupies one quarter of the full object's XY extent.
    full=np.ones((8,64,64),bool);pred=np.zeros_like(full);pred[:,16:48,16:48]=1
    iou=np.count_nonzero(pred&full)/np.count_nonzero(pred|full)
    assert quality==1. and iou==.25
    return dict(perfect_crop_quality=float(quality),full_object_iou=iou)


def proposal_limits():
    cfg=config(proposal_mode='peaks',median_max_dist_nucleus=54.,rays_anisotropy_nucleus=(19,1,1))
    shape=(8,32,32);prob=np.zeros((8,16,16),np.float32)
    prob[3,1,7]=.9;prob[3,7,7]=.9;prob[3,7,9]=.8
    dist=np.ones(prob.shape+(12,),np.float32)
    def run(border,spacing):
        cfg.nms_border_exclude_xy=border;cfg.nms_candidate_spacing=spacing
        stub=SimpleNamespace(config=cfg,ndim=3,rays_nucleus=cfg.rays_nucleus,
            rays_cytoplasm=cfg.rays_cytoplasm,_current_type='nucleus')
        CytoNucsPredictor._stardist_postprocess(stub,prob,dist,.4,.3)
        return np.asarray(stub._last_coords).astype(int).tolist()
    # Use a larger probability map so geometry window5 isn't clipped to3.
    prob=np.pad(prob,((0,0),(0,16),(0,16)));dist=np.ones(prob.shape+(12,),np.float32)
    result=dict(default=run(2,.12),no_border=run(0,.12),smaller_window=run(0,.01))
    assert [3,2,14] not in result['default'] and [3,2,14] in result['no_border']
    assert [3,14,18] not in result['no_border'] and [3,14,18] in result['smaller_window']
    return result


def observed():
    path=ROOT/'fixes_v2.8/epoch_0420_refinement.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result={}
    for head in ('nucleus','cytoplasm'):
        reasons=Counter();windows=Counter();rois=Counter();border_above=0;lost_selected=0
        for r in rows:
            a=r['stage_audit'][head];p=a['proposals'];reasons.update(p['reason_counts'])
            windows[str(p['window'])]+=1
            for roi in a['rois']:rois[str(roi['shape'])]+=1
            border_above+=sum(v['reason']=='border_only' and v['max_prob_grid']>p['threshold']
                              for v in p['per_gt'].values())
            if head=='nucleus':
                lost_selected+=sum(v['reason']=='selected' and a['stages']['final']['best_iou_per_gt'][g]<.5
                                   for g,v in p['per_gt'].items())
        result[head]=dict(proposal_reasons=dict(reasons),windows=dict(windows),roi_shapes=dict(rois),
                          border_only_above_threshold=border_above,selected_without_final_iou=lost_selected if head=='nucleus' else None)
    h=json.loads((ROOT/'fixes_v2.8/history.json').read_text())
    result['detector_prompt_upper_bounds']={}
    for head in ('nucleus','cytoplasm'):
        fractions=h['train_'+head+'_z_pair_fraction'][410:420]
        mean=float(np.mean(fractions))
        result['detector_prompt_upper_bounds'][head]=dict(mean_pair_fraction=mean,
            eligible_expected_fraction_upper_bound=.25*(1-mean),
            scope='K=2; upper bound before eligible-peak filtering, not measured replacement frequency')
    result['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


if __name__=='__main__':
    result=dict(prompt_protection=prompt_exposure(),cell_roi_oracle=roi_limit(),
                cropped_quality=cropped_quality(),proposal_limits=proposal_limits(),observed420=observed(),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
                packages={x:importlib.metadata.version(x) for x in ('tensorflow','keras','numpy','scipy','stardist')}, seed=42,
                scope='Synthetic mechanism checks and read-only logs. No trained-model A/B or measured JTPR gain.')
    with (ROOT/'audit/score_bottlenecks.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
