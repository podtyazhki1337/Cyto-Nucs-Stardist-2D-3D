import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config
from cytonucs_refinement_audit import RefinementStageAudit
from cytonucs_postprocess import assemble_masks, mask_candidate, upsample_grid, prompt_component
from cytonucs_model import build_cytonucs_model
from cytonucs_geometry import architecture_plan, pad_to_multiple
import cytonucs_inference as inference


class StageTests(unittest.TestCase):
    def test_matching_handles_sparse_ids_and_overlapping_candidates(self):
        gt=np.zeros((2,4,8),np.int32);gt[:,:,:4]=1000000;gt[:,:,4:]=7
        audit=RefinementStageAudit(gt)
        merged=mask_candidate(np.ones(gt.shape),gt>0,(0,0,0))
        audit.observe('before_nms',candidates=[merged])
        r=audit.report()['stages']['before_nms']
        self.assertEqual(r['gt_covered_at_iou'],2)
        self.assertEqual(r['oracle_one_to_one_tp'],1)
        self.assertEqual(r['best_iou_per_gt'],{'7':.5,'1000000':.5})
        audit.observe('after_nms',candidates=[merged,merged])
        self.assertEqual(audit.report()['stages']['after_nms']['oracle_one_to_one_tp'],2)

    def test_audit_localizes_nms_and_competition_damage_without_changing_result(self):
        gt=np.ones((2,10,10),np.int32)
        full=mask_candidate(np.full(gt.shape,.8),gt>0,(0,0,0))
        stripe=np.zeros_like(gt,bool);stripe[:,:,4:6]=True
        cut=mask_candidate(np.full(gt.shape,.95),stripe,(0,0,0))
        audit=RefinementStageAudit(gt)
        old,stats=assemble_masks([full,cut],gt.shape,dedup_iou=.5,overlap_threshold=.85)
        out,measured=assemble_masks([full,cut],gt.shape,dedup_iou=.5,overlap_threshold=.85,observer=audit)
        np.testing.assert_array_equal(old,out);self.assertEqual(stats,measured)
        r=audit.report()
        self.assertEqual(r['stages']['before_nms']['oracle_one_to_one_tp'],1)
        self.assertEqual(r['stages']['competition']['oracle_one_to_one_tp'],1)
        self.assertEqual(r['stages']['final']['oracle_one_to_one_tp'],0)
        self.assertEqual(r['transitions']['competition->final']['lost_gt_ids'],['1'])
        wrong=mask_candidate(np.full(gt.shape,.9),np.indices(gt.shape)[2]<6,(0,0,0))
        # Higher-scoring 60% mask suppresses exact candidate at IoU=.6.
        a=RefinementStageAudit(gt,threshold=.75)
        assemble_masks([full,wrong],gt.shape,dedup_iou=.5,observer=a)
        self.assertEqual(a.report()['transitions']['before_nms->after_nms']['lost_gt_ids'],['1'])

    def test_upsample_preserves_axis_order_and_grid_origin_in_3d(self):
        grid=(2,3,4);z,y,x=np.indices((4,5,6))
        field=(100*z*2+10*y*3+x*4).astype(float)
        result=upsample_grid(field,(7,13,21),grid)
        zz,yy,xx=np.indices(result.shape)
        np.testing.assert_allclose(result,100*zz+10*yy+xx,atol=1e-7)

    def test_component_is_truly_3d_not_independent_slices(self):
        mask=np.zeros((9,13,17),bool)
        for z in range(9):mask[z,2+z,3+z]=True
        mask[:,1,15]=True # disconnected distractor through Z
        selected=prompt_component(mask,(4,6,7),(1,1,1))
        self.assertEqual(int(selected.sum()),9)
        self.assertTrue(selected[0,2,3] and selected[8,10,11])
        self.assertFalse(selected[:,1,15].any())

    def test_full_z_odd_roi_global_coordinates_padding_and_observer_noninterference(self):
        cfg=config(nucleus_grid=(1,1,1),cytoplasm_grid=(1,1,1),refinement_patch_size=[0,31,35])
        shape=(23,55,79);gt=np.zeros(shape,np.int32);gt[1:22,29:38,47:56]=1
        image=(gt>0).astype(np.float32)[...,None]
        center=np.array([[11,33,51]]);anchors=np.zeros(shape,np.int32);anchors[tuple(center[0])]=1
        def features(x):return [x]*7
        def refine(inputs):
            binary=inputs[0][...,0]
            logits=tf.stack([10*(1-binary),10*binary,tf.zeros_like(binary)],axis=-1)
            return logits,logits
        def call(audits):
            with patch.object(inference,'_refinement_runners',return_value=(features,refine)), \
                 patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda x:refine(x)[1]):
                return inference.predict_instances_prompted(SimpleNamespace(),cfg,image,center,np.ones(1),
                    nuc_labels=anchors,verbose=False,stage_audit=audits)
        baseline=call(None);audits={h:RefinementStageAudit(gt) for h in ('nucleus','cytoplasm')}
        measured=call(audits)
        for a,b in zip(baseline[:2],measured[:2]):np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(b,gt)
        for a in audits.values():
            r=a.report();roi=r['rois'][0]
            self.assertEqual(roi['origin'],[0,18,34]);self.assertEqual(roi['shape'],[23,31,35])
            self.assertEqual(roi['center_global'],[11.,33.,51.])
            self.assertTrue(all(p>=s for p,s in zip(roi['padded_shape'],roi['shape'])))
            self.assertTrue(all(v['oracle_one_to_one_tp']==1 for v in r['stages'].values()))

    def test_actual_3d_runtime_equals_joint_forward_with_long_padded_input(self):
        cfg=config(refinement_depth=1,mask_quality_head=True,mask_quality_pooling='mask')
        model=build_cytonucs_model(cfg)
        image=np.random.default_rng(4).random((13,35,41,1)).astype(np.float32)
        image=pad_to_multiple(image,architecture_plan(cfg)['input_multiple'])
        axes=np.indices(image.shape[:3]);p=np.exp(-sum(((axes[a]-[9,21,30][a])/[2,3,4][a])**2 for a in range(3))/2)[...,None].astype(np.float32)
        x=tf.constant(np.concatenate([image,p,p,p,p],axis=-1)[None])
        joint=model.keras_model(x)
        features=list(model.refinement_features_model(x[...,:1]))
        runtime=model.refinement_with_quality_model(features+[x[...,1:2],x[...,2:3]])
        for actual,key in zip(runtime,['nucleus_mask_logit','cytoplasm_mask_logit','nucleus_quality','cytoplasm_quality']):
            np.testing.assert_allclose(actual,joint[key][:1],rtol=1e-5,atol=1e-5)

    def test_empty_prediction_audit_has_all_gt_failures(self):
        a=RefinementStageAudit(np.ones((2,3,4),np.int32))
        labels,stats=assemble_masks([],a.gt.shape,observer=a)
        self.assertEqual(labels.max(),0)
        self.assertTrue(all(s['oracle_one_to_one_tp']==0 for s in a.report()['stages'].values()))

    def test_probe_cli_and_validation_jsonl_on_fixed_checkpoint(self):
        import json,tempfile,tifffile
        from pathlib import Path
        from test_fixes import ROOT
        from audit.refinement_stage_probe import main
        from cytonucs_trainer import DiceCallback
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            folder=Path(tmp)
            for sub in ('images','nucleus_masks','cytoplasm_masks','assignments'):(folder/'val'/sub).mkdir(parents=True)
            gt=np.zeros((9,31,29),np.uint16);gt[2:7,9:20,9:20]=1
            image=(gt>0).astype(np.float32)
            for sub,array in [('images',image),('nucleus_masks',gt),('cytoplasm_masks',gt)]:
                tifffile.imwrite(folder/'val'/sub/'synthetic.tiff',array,photometric='minisblack')
            (folder/'val/assignments/synthetic.json').write_text('{"1":1}')
            raw=json.loads((ROOT/'config_cytonucs.json').read_text())
            raw['model'].update(n_rays=12,n_channel_in=1,use_gpu=False,unet_n_depth=2,
                unet_n_filter_base=2,net_conv_after_unet=4,mask_refinement_filters=4)
            raw['training']['train_patch_size']=[8,32,32]
            raw['data']['dataset_dir']=str(folder)
            raw['inference'].update(prob_thresh_nucleus=1.,prob_thresh_cytoplasm=1.)
            raw['evaluation']['refinement_stage_audit']=True
            (folder/'config.json').write_text(json.dumps(raw))
            cfg=config(refinement_stage_audit=True,prob_thresh_nucleus=1.,prob_thresh_cytoplasm=1.)
            model=build_cytonucs_model(cfg,basedir=tmp)
            (folder/'geometry.json').write_text(json.dumps(dict(n_rays=12,
                rays_anisotropy_nucleus=list(cfg.rays_anisotropy_nucleus),rays_anisotropy_cytoplasm=list(cfg.rays_anisotropy_cytoplasm),
                median_max_dist_nucleus=6.,median_max_dist_cytoplasm=10.)))
            weights=folder/'weights_epoch_075.weights.h5';model.save_weights(weights)
            out=folder/'probe.json'
            args=['probe','--config',str(folder/'config.json'),'--checkpoint',str(weights),'--output',str(out),
                  '--compare-assembly','--gt-prompts','--files','synthetic.tiff']
            with patch('sys.argv',args):main()
            report=json.loads(out.read_text());self.assertEqual(report['status'],'complete')
            result=report['results'][0]
            self.assertEqual(result['stage_audit']['nucleus']['stages']['final']['gt_count'],1)
            self.assertEqual(result['stage_audit']['nucleus']['stages']['final']['candidates'],0)
            self.assertTrue(result['gt_assignment_source'].endswith('synthetic.json'))
            self.assertEqual(set(result['assembly_comparison']),{'voxel','object','object_quality'})
            self.assertEqual(result['gt_prompt_oracle']['stage_audit']['nucleus']['stages']['final']['gt_count'],1)
            with patch('sys.argv',args),self.assertRaises(SystemExit):main()
            # Built-in validation writes the same data, without stale prior-volume counters.
            gen=SimpleNamespace(image_files=['synthetic.tiff'],data_pool=[dict(image=image[...,None],
                nucleus_mask=gt,cytoplasm_mask=gt,assignments={1:1},filename='synthetic.tiff')])
            callback=DiceCallback(model,gen,eval_every=1,save_best=False,prob_thresh_nucleus=1.,prob_thresh_cytoplasm=1.)
            with patch.object(callback,'_save_epoch_visualization_full'):
                callback.on_epoch_end(74,{})
            row=json.loads((folder/model.name/'visualizations/epoch_0075_refinement.jsonl').read_text())
            self.assertEqual(row['stage_audit']['cytoplasm']['stages']['final']['gt_count'],1)
            self.assertEqual(row['refinement']['cytoplasm']['attempted'],0)


if __name__=='__main__':unittest.main()
