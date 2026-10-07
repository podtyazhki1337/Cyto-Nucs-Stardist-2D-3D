"""Reproduce structural limits, independent of checkpoint quality.

The refinement oracle below returns perfect foreground inside its available
window. This deliberately separates window/assembly limits from learned errors.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ast
import json
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
import cytonucs_inference as inference
from cytonucs_config import CytoNucsConfig
from cytonucs_losses import CytoNucsLoss
from cytonucs_postprocess import assemble_masks, mask_candidate, candidate_iou


def run():
    output = ROOT/'audit/architecture_limits.json'
    if output.exists():
        raise FileExistsError(output)
    cfg = CytoNucsConfig(ndim=3, n_rays=12, n_channel_in=1,
        train_patch_size=(16,32,32), nucleus_grid=(1,2,2), cytoplasm_grid=(1,2,2),
        unet_n_depth=2, train_batch_size=1, prompts_per_crop=1,
        median_max_dist_nucleus=10., median_max_dist_cytoplasm=10.)
    shape=(40,32,32)
    image=np.zeros(shape+(1,),np.float32)
    labels=np.zeros(shape,np.int32); labels[20,16,16]=1
    foreground_calls=[]
    def features(x):
        foreground_calls.append(tuple(x.shape))
        return [x]*7
    def logits(prompt):
        sh=tf.shape(prompt)
        spatial=(int(sh[1]),int(sh[2])//2,int(sh[3])//2)
        a=np.zeros((1,*spatial,3),np.float32);a[...,1]=12.
        return tf.constant(a)
    def refine(inputs):
        a=logits(inputs[-1]);return a,a
    with patch.object(inference,'_refinement_runners',return_value=(features,refine)), \
         patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda inputs:logits(inputs[-1])):
        n,c,_=inference.predict_instances_prompted(SimpleNamespace(),cfg,image,
            np.array([[20,16,16]]),np.array([.9]),nuc_labels=labels,
            cyto_labels=np.zeros_like(labels),verbose=False)
        extent=np.ptp(np.argwhere(n>0),axis=0)+1
        assert tuple(extent)==cfg.train_patch_size
        assert len(np.unique(n[n>0]))==1
        n0,c0,_=inference.predict_instances_prompted(SimpleNamespace(),cfg,image,
            None,None,nuc_labels=np.zeros_like(labels),cyto_labels=np.zeros_like(labels),verbose=False)
        assert not n0.any() and not c0.any()
    # A perfect large mask plus a confident internal fragment survive IoU NMS.
    shape2=(8,16,16)
    large=np.zeros(shape2,bool);large[:,2:14,2:14]=True
    small=np.zeros(shape2,bool);small[2:6,6:10,6:10]=True
    ca=mask_candidate(np.full(shape2,.8,np.float32),large,(0,0,0))
    cb=mask_candidate(np.full(shape2,.95,np.float32),small,(0,0,0))
    out,stats=assemble_masks([ca,cb],shape2,.5)
    assert stats['written']==2
    assert np.all(out[small]==out[3,7,7]) and np.all(out[small]>0)
    assert out[0,3,3]!=out[3,7,7]
    # Public configuration toggle exists, but the loss never calls consistency.
    cfg0=CytoNucsConfig(ndim=3,n_rays=12,n_channel_in=1,train_patch_size=(4,8,8),
        nucleus_grid=(1,1,1),cytoplasm_grid=(1,1,1),prompts_per_crop=1,
        lambda_nuc_mask=0,lambda_cyto_mask=0,lambda_wbr=0,lambda_containment=0,
        enable_consistency=False,lambda_consistency=0)
    cfg1=CytoNucsConfig(ndim=3,n_rays=12,n_channel_in=1,train_patch_size=(4,8,8),
        nucleus_grid=(1,1,1),cytoplasm_grid=(1,1,1),prompts_per_crop=1,
        lambda_nuc_mask=0,lambda_cyto_mask=0,lambda_wbr=0,lambda_containment=0,
        enable_consistency=True,lambda_consistency=100)
    for conf in (cfg0,cfg1):
        conf.median_max_dist_nucleus=4.
        conf.median_max_dist_cytoplasm=4.
    ys={};yp={}
    for head in ('nucleus','cytoplasm'):
        ys[head+'_instances']=tf.ones((1,4,8,8),tf.int32)
        ys[head+'_prob']=tf.ones((1,4,8,8,1))*.7
        ys[head+'_dist']=tf.ones((1,4,8,8,12))*4
        yp[head+'_prob']=tf.ones((1,4,8,8,1))*.3
        yp[head+'_dist']=tf.ones((1,4,8,8,12))*2
    loss0=float(CytoNucsLoss(cfg0)(ys,yp,current_epoch=100))
    loss1=float(CytoNucsLoss(cfg1)(ys,yp,current_epoch=100))
    assert loss0==loss1
    mt=ast.parse((ROOT/'cytonucs_model.py').read_text(encoding='utf-8'))
    shared_refs=[n.lineno for n in ast.walk(mt) if isinstance(n,ast.Attribute) and n.attr=='shared_encoder']
    shared_getattrs=[n.lineno for n in ast.walk(mt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                    and n.func.id=='getattr' and len(n.args)>1 and isinstance(n.args[1],ast.Constant)
                    and n.args[1].value=='shared_encoder']
    assert not shared_refs and not shared_getattrs
    result={'scope':'synthetic structural counterexamples; no user checkpoint, no JTPR prediction',
        'versions':{'tensorflow':tf.__version__,'numpy':np.__version__},
        'window_limit':{'full_volume_shape':shape,'patch':cfg.train_patch_size,
            'oracle_output_extent':extent.tolist(),'feature_input_shapes':foreground_calls,
            'upper_iou_for_uniform_full_volume_object':float(np.prod(extent)/np.prod(shape))},
        'zero_proposals':{'nucleus_instances':int(n0.max()),'cell_instances':int(c0.max())},
        'nested_masks':{'large_voxels':int(large.sum()),'small_voxels':int(small.sum()),
            'candidate_iou':candidate_iou(ca,cb),'small_containment':1.,'nms_threshold':.5,
            'stats':stats,'large_remainder_voxels':int(np.count_nonzero(out==out[0,3,3]))},
        'consistency_toggle':{'off_loss':loss0,'on_weight100_loss':loss1,'has_effect':loss0!=loss1},
        'shared_encoder_read_by_model':False,'assertions':'passed',
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ['cytonucs_inference.py','cytonucs_postprocess.py','cytonucs_losses.py','cytonucs_model.py',
             'cytonucs_config.py','audit/check_architecture_limits.py']}}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
