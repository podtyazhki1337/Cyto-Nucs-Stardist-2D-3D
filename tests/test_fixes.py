"""Regression tests: real TF/StarDist on tiny synthetic data, no user dataset."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
from cytonucs_config import CytoNucsConfig
from cytonucs_geometry import architecture_plan, fixed_window_plan, pad_to_multiple
from cytonucs_postprocess import mask_candidate, assemble_masks, assign_final_masks
from cytonucs_inference import compute_jtpr, match_instances, CytoNucsPredictor, predict_instances_prompted, _refinement_runners, _group_prompt_jobs, _cytoplasm_refinement_runner
from cytonucs_model import build_cytonucs_model
from cytonucs_losses import CytoNucsLoss, compute_stardist_gt
from cytonucs_data_generator import CytoNucsDataGenerator, cytonucs_augmenter_3d
from cytonucs_trainer import CytoNucsTrainer, DiceCallback, _Prefetcher


def config(**kwargs):
    params = dict(ndim=3, n_rays=12, n_channel_in=1, train_patch_size=(8,32,32),
                  nucleus_grid=(1,2,2), cytoplasm_grid=(1,2,2), anisotropy=(3,1,1),
                  unet_n_depth=2, unet_n_filter_base=2, net_conv_after_unet=4,
                  mask_refinement_filters=4, train_batch_size=2, prompts_per_crop=2,
                  median_max_dist_nucleus=6., median_max_dist_cytoplasm=10.,
                  lambda_nuc_mask=0.4, lambda_cyto_mask=0.4, w_containment_distance=0.,
                  w_center_separation=0., seed=42, zoom_range=None)
    params.update(kwargs)
    return CytoNucsConfig(**params)


class ResumeTests(unittest.TestCase):
    def test_epoch_resolution_never_guesses_from_best_name(self):
        from cytonucs_runtime import resolve_resume_epoch
        self.assertEqual(resolve_resume_epoch('weights_epoch_020.weights.h5'),20)
        self.assertEqual(resolve_resume_epoch('weights_best.weights.h5',20),20)
        self.assertEqual(resolve_resume_epoch('weights_epoch_020.weights.h5',17),17)
        for name in ('weights_best.weights.h5','weights_best_val_loss.weights.h5'):
            with self.assertRaisesRegex(ValueError,'epoch unknown'):
                resolve_resume_epoch(name)
        with self.assertRaises(ValueError): resolve_resume_epoch('weights_best.weights.h5',-1)

    def test_resume_schedule_uses_only_one_step_source(self):
        import ast
        from cytonucs_runtime import resolve_resume_epoch
        tree=ast.parse((ROOT/'train_cytonucs.py').read_text(encoding='utf-8'))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='WarmupExponentialDecay')
        scope={'tf':tf}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'schedule-test','exec'),scope)
        Schedule=scope['WarmupExponentialDecay']
        uninterrupted=Schedule(.0003,300,5000,.85,min_lr=1e-7,step_offset=0)
        for restored in (False,True):
            epoch=resolve_resume_epoch('weights_epoch_020.weights.h5',20 if restored else None)
            resumed=Schedule(.0003,300,5000,.85,min_lr=1e-7,step_offset=0 if restored else epoch*100)
            for extra in (0,1,100):
                step=2000+extra if restored else extra
                np.testing.assert_allclose(resumed(step).numpy(),uninterrupted(2000+extra).numpy(),rtol=1e-7)


class ProbabilityAuditTests(unittest.TestCase):
    def test_candidate_coverage_is_not_instance_recall(self):
        from audit.diagnose_probability import summarize
        labels=np.zeros((2,8,8),np.int32)
        labels[:,2:6,2:6]=1; labels[0,0,0]=2
        prob=np.zeros((2,4,4),np.float32)
        prob[0,1,1]=.45; prob[1,2,2]=.2
        stats=summarize(prob,labels,(1,2,2),[.4,.5],border=1)
        self.assertEqual(stats['gt_count'],2)
        self.assertEqual(stats['gt_with_grid_voxels_inside_border'],1)
        self.assertEqual([x['gt_with_candidate'] for x in stats['thresholds']],[1,0])


class ConfigContractTests(unittest.TestCase):
    def load(self, requested):
        import json
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            path=Path(tmp)/'config.json'
            requested['output']['basedir']=str(Path(tmp)/'models')
            path.write_text(json.dumps(requested))
            return CytoNucsConfig.from_json(path)

    def test_shipped_json_reaches_config(self):
        import json
        requested=json.loads((ROOT/'config_cytonucs.json').read_text())
        cfg=self.load(requested)
        for section, settings in CytoNucsConfig.FIXES_SETTINGS.items():
            for key in settings:
                self.assertIn(key,requested[section],f'Missing explicit {section}.{key}')
                self.assertEqual(getattr(cfg,key),requested[section][key],key)
        # Check all flattened sections, not only newly introduced parameters.
        for section in ('model','training','loss_weights','loss_params','evaluation','augmentation','inference'):
            for key,value in requested[section].items():
                if key in ('enable','checkpoint_path'): continue
                actual=getattr(cfg,key)
                if isinstance(actual,tuple): actual=list(actual)
                self.assertEqual(actual,value,f'{section}.{key}')
        self.assertEqual(cfg.train_learning_rate,requested['training']['initial_learning_rate'])
        self.assertEqual(cfg.train_steps_per_epoch,requested['training']['steps_per_epoch'])
        self.assertIn('image_batch_size=1',repr(cfg))
        self.assertIn('prompts_per_crop=2',repr(cfg))

    def test_overrides_not_silently_replaced_by_defaults(self):
        import json
        requested=json.loads((ROOT/'config_cytonucs.json').read_text())
        for section, settings in CytoNucsConfig.FIXES_SETTINGS.items():
            for key,default in settings.items():
                if key=='refinement_mode': value='mask'
                elif key=='proposal_mode': value='peaks'
                elif key=='mask_quality_pooling': value='mask'
                elif key=='refinement_pool': value='auto'
                elif key=='mask_quality_target': value='component'
                elif key=='mask_overlap_loss': value='lovasz'
                # Needs mask_quality_min='auto' with object_quality assembly; flipped in tests/test_v3.py.
                elif key=='object_losses_min_area_from_gate': value=False
                elif key.startswith('mask_assembly_'): value='object'
                elif key=='unet_pool_schedule': value=[[1,2,2],[2,2,2],[2,2,2],[1,2,2]]
                elif key=='unet_kernel_schedule': value=[[1,3,3]]*5
                elif key=='refinement_patch_size': value=[0,448,448]
                elif key=='dist_target_stride': value=[1,2,2]
                elif isinstance(default,bool): value=not default
                elif key=='tiling_fixed_axes': value=[0]
                elif key=='levels': value=['nucleus']
                elif key in ('mask_thresh_nucleus','mask_thresh_cytoplasm'): value=.63
                elif key=='prompts_per_crop': value=3
                elif key.startswith('proposal_peak_window_'): value=[3,3,3]
                elif key.startswith('proposal_border_'): value=0
                elif isinstance(default,int): value=default+1
                else: value=.17
                requested[section][key]=value
        requested['training']['gradient_clip_norm']=7.
        requested['loss_params']['w_center_separation']=0.   # required by dist_target_stride
        cfg=self.load(requested)
        for section,settings in CytoNucsConfig.FIXES_SETTINGS.items():
            for key in settings: self.assertEqual(getattr(cfg,key),requested[section][key],key)
        self.assertEqual(cfg.gradient_clip_norm,7.)

    def test_invalid_or_misplaced_settings_fail(self):
        import json
        for section,key,value in [('training','prompts_per_crop',0),
                                  ('augmentation','negative_prompt_prob',1.1),
                                  ('training','pair_min_mask_iou',.4)]:
            requested=json.loads((ROOT/'config_cytonucs.json').read_text())
            requested[section][key]=value
            with self.assertRaises(ValueError): self.load(requested)


class GeometryTests(unittest.TestCase):
    def test_architecture_and_odd_padding(self):
        c = config(train_patch_size=(16,448,448), unet_n_depth=4)
        plan = architecture_plan(c)
        self.assertEqual(plan['input_multiple'], (8,32,32))
        for z in (8,15,16,17,31,32):
            x = pad_to_multiple(np.zeros((z,33,35,1)), plan['input_multiple'])
            self.assertTrue(all(s % g == 0 for s,g in zip(x.shape,plan['input_multiple'])))

    def test_checkpoint_geometry_override(self):
        import json
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            root=Path(tmp); checkpoint=root/'weights.weights.h5'
            requested=json.loads((ROOT/'config_cytonucs.json').read_text())
            requested['output']['basedir']=str(root/'other')
            (root/'config.json').write_text(json.dumps(requested))
            geo=json.loads((ROOT/'geometry.json').read_text())
            geo['rays_anisotropy_nucleus']=[7,1,1]
            (root/'geometry.json').write_text(json.dumps(geo))
            cfg=CytoNucsConfig.from_json(root/'config.json',checkpoint_path=checkpoint)
            self.assertEqual(tuple(cfg.rays_anisotropy_nucleus),(7,1,1))

    def test_fixed_windows_cover_once(self):
        for n in (1,8,15,16,17,31,32,65):
            cov = np.zeros(n, int)
            for start,size,lo,hi in fixed_window_plan(n,16,8):
                self.assertEqual(start % 8,0)
                self.assertEqual(size % 8,0)
                self.assertGreaterEqual(lo,start)
                self.assertLessEqual(hi,start+size)
                cov[lo:hi] += 1
            np.testing.assert_array_equal(cov,1)


class InstanceTests(unittest.TestCase):
    def test_perfect_mask_keeps_iou_one(self):
        mask=np.zeros((2,3,4),bool); mask[:,1,1:3]=True
        out,_=assemble_masks([mask_candidate(mask.astype(float),mask,(0,0,0))],mask.shape)
        np.testing.assert_array_equal(out>0,mask)

    def test_competition_and_multinuclear(self):
        shape=(1,4,8); a=np.zeros(shape,bool); b=a.copy()
        a[:,:,:5]=True; b[:,:,3:]=True
        pa=np.where(a,.7,0); pb=np.where(b,.9,0)
        out,_=assemble_masks([mask_candidate(pa,a,(0,0,0)),mask_candidate(pb,b,(0,0,0))],shape,.9)
        self.assertEqual(len(np.unique(out)),2)
        self.assertTrue(np.all(out[:,:,3:5]==out[0,0,7]))
        nuclei=np.zeros(shape,np.int32); nuclei[0,1,4]=1; nuclei[0,2,6]=2
        pairs=assign_final_masks(nuclei,out)
        self.assertEqual(pairs[1],pairs[2])

    def test_ring_cell_prompt_evidence(self):
        n=np.zeros((3,7,7),np.int32); n[1,3,3]=1
        c=np.ones_like(n); c[n>0]=0
        self.assertEqual(assign_final_masks(n,c),{})
        evidence=[(mask_candidate((n>0).astype(float),n>0,(0,0,0)),
                   mask_candidate((c>0).astype(float),c>0,(0,0,0)))]
        self.assertEqual(assign_final_masks(n,c,evidence=evidence),{1:1})

    def test_half_iou_is_one_to_one(self):
        gt=np.array([[[1,1,2,2]]],np.int32)
        self.assertEqual(len(match_instances(np.ones_like(gt),gt,.5)),1)

    def test_jtpr_identity_empty_and_wrong_link(self):
        n=np.zeros((2,4,8),np.int32); c=n.copy()
        n[0,1,1]=1; n[0,1,6]=2; c[:,:,:4]=1; c[:,:,4:]=2
        for pred,expected in (({1:1,2:2},1),({},0),({1:2,2:1},0)):
            j,d=compute_jtpr(n,c,n,c,assignments_pred=pred,assignments_gt={1:1,2:2})
            self.assertEqual(j,expected)
            self.assertEqual(d['joint_tp']+sum(d[k] for k in ('joint_fail_nucleus','joint_fail_cell','joint_fail_link')),2)


class TensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(42)
        cls.cfg=config()
        cls.model=build_cytonucs_model(cls.cfg, name='synthetic_test', basedir=str(ROOT/'.cache'/'test_models'))

    def test_cached_refine_equals_full_model(self):
        x=np.random.default_rng(5).uniform(size=(1,8,32,32,5)).astype('float32')
        full=self.model.keras_model(x,training=False)
        feat=self.model.refinement_features_model(x[...,:1],training=False)
        logits=self.model.refinement_model(list(feat)+[tf.constant(x[...,1:2]),tf.constant(x[...,2:3])],training=False)
        for actual,key in zip(logits,('nucleus_mask_logit','cytoplasm_mask_logit')):
            np.testing.assert_allclose(actual.numpy(),full[key].numpy()[:1],rtol=1e-5,atol=1e-5)

    def test_compiled_refinement_matches_eager(self):
        x=tf.random.stateless_uniform((1,8,32,32,1),seed=(1,7))
        p=tf.ones_like(x)
        eager_features=self.model.refinement_features_model(x,training=False)
        eager=self.model.refinement_model(list(eager_features)+[p,p],training=False)
        feature_fn,refine_fn=_refinement_runners(self.model)
        actual=refine_fn(list(feature_fn(x))+[p,p])
        for a,b in zip(eager,actual): np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-4,atol=1e-5)
        self.assertIs(_refinement_runners(self.model)[0],feature_fn)

    def test_exact_window_grouping_preserves_job_identity(self):
        jobs=[('nucleus',(1,),None),('nucleus',(2,),None),('cytoplasm',(1,),None)]
        groups=list(_group_prompt_jobs(jobs,lambda ctr:ctr))
        self.assertEqual([[i for i,_ in g] for g in groups],[[0,2],[1]])
        restored=[None]*len(jobs)
        for group in groups:
            for i,job in group: restored[i]=job
        self.assertEqual(restored,jobs)

    def test_grouped_pipeline_keeps_masks_and_reduces_forwards(self):
        import io, contextlib, re
        from unittest.mock import patch
        x=np.random.default_rng(3).uniform(size=(8,48,48,1)).astype('float32')
        labels=np.zeros(x.shape[:3],np.int32)
        labels[2:6,4:9,4:9]=1; labels[2:6,38:43,38:43]=2
        ctr=np.array([[4,6,6],[4,40,40]])
        def call():
            log=io.StringIO()
            with contextlib.redirect_stdout(log):
                result=predict_instances_prompted(self.model,self.cfg,x,ctr,np.ones(2),ctr,np.ones(2),
                    nuc_labels=labels,cyto_labels=labels,mask_thresh=0.)
            return result,int(re.search(r'feature forwards=(\d+)',log.getvalue()).group(1))
        grouped,fast_count=call()
        with patch('cytonucs_inference._group_prompt_jobs',
                   side_effect=lambda jobs,origin_fn:[[(i,j)] for i,j in enumerate(jobs)]):
            serial,slow_count=call()
        np.testing.assert_array_equal(grouped[0],serial[0])
        np.testing.assert_array_equal(grouped[1],serial[1])
        self.assertEqual(grouped[2],serial[2])
        self.assertEqual((fast_count,slow_count),(2,4))

    def test_cell_only_head_matches_dual_and_uses_live_weights(self):
        x=tf.random.stateless_uniform((1,8,32,32,1),seed=(8,9))
        prompt=tf.ones_like(x)
        features=list(self.model.refinement_features_model(x,training=False))
        dual_args=features+[prompt,prompt]
        cell_args=[features[i] for i in (1,2,5,6)]+[prompt]
        fn=_cytoplasm_refinement_runner(self.model)
        expected=self.model.refinement_model(dual_args,training=False)[1]
        actual=fn(cell_args)
        np.testing.assert_allclose(actual.numpy(),expected.numpy(),rtol=1e-4,atol=1e-5)
        names=[layer.name for layer in self.model.cytoplasm_refinement_model.layers]
        self.assertNotIn('nucleus_mask_out',names)
        self.assertFalse(any('nuc_refine_conv' in name for name in names))
        bias=self.model.keras_model.get_layer('cytoplasm_mask_out').bias
        saved=bias.numpy().copy()
        try:
            bias.assign_add(tf.ones_like(bias)*0.1)
            changed=fn(cell_args)
            expected=self.model.refinement_model(dual_args,training=False)[1]
            np.testing.assert_allclose(changed.numpy(),expected.numpy(),rtol=1e-4,atol=1e-5)
            np.testing.assert_allclose(changed.numpy()-actual.numpy(),0.1,rtol=1e-4,atol=1e-5)
        finally: bias.assign(saved)

    def test_probability_only_model_matches_full(self):
        from audit.diagnose_probability import probability_model
        x=tf.ones((1,8,32,32,5))
        full=self.model.keras_model(x,training=False)
        fast=probability_model(self.model)
        actual=fast(x,training=False)
        for head in ('nucleus','cytoplasm'):
            np.testing.assert_allclose(actual[head],full[head+'_prob'],rtol=1e-5,atol=1e-6)
        self.assertFalse(any('refine_conv' in layer.name for layer in fast.layers))

    def test_probability_audit_cli_on_synthetic_tiffs(self):
        import json, tifffile
        from unittest.mock import patch
        from audit.diagnose_probability import main
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            folder=Path(tmp)
            for name in ('images','nucleus_masks','cytoplasm_masks'):
                (folder/'val'/name).mkdir(parents=True)
            shape=(8,32,32)
            mask=np.zeros(shape,np.uint16); mask[2:6,10:20,10:20]=1
            for i in range(3):
                for name,arr in [('images',np.ones(shape,np.float32)),('nucleus_masks',mask),('cytoplasm_masks',mask)]:
                    tifffile.imwrite(folder/'val'/name/f'{i}.tif',arr,photometric='minisblack')
            cfg=json.loads((ROOT/'config_cytonucs.json').read_text())
            cfg['model'].update(ndim=3,n_rays=12,n_channel_in=1,use_gpu=False,
                unet_n_depth=2,unet_n_filter_base=2,net_conv_after_unet=4,mask_refinement_filters=4)
            cfg['training']['train_patch_size']=[8,32,32]
            cfg['data']['dataset_dir']=str(folder)
            (folder/'config.json').write_text(json.dumps(cfg))
            (folder/'geometry.json').write_text(json.dumps(dict(n_rays=12,
                rays_anisotropy_nucleus=list(self.cfg.rays_anisotropy_nucleus),
                rays_anisotropy_cytoplasm=list(self.cfg.rays_anisotropy_cytoplasm),
                median_max_dist_nucleus=6.,median_max_dist_cytoplasm=10.)))
            checkpoint=folder/'weights_epoch_030.weights.h5'; self.model.save_weights(checkpoint)
            output=folder/'result.json'
            args=['audit/diagnose_probability.py','--config',str(folder/'config.json'),
                  '--checkpoint',str(checkpoint),'--output',str(output)]
            with patch('sys.argv',args): main()
            report=json.loads(output.read_text())
            self.assertEqual(report['status'],'complete')
            self.assertEqual(len(report['results']),3)
            self.assertEqual(report['results'][0]['nucleus']['gt_count'],1)
            with patch('sys.argv',args),self.assertRaises(SystemExit): main()

    def test_run_artifacts(self):
        import json
        import cytonucs_runtime as runtime
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            root=Path(tmp)
            config_path=root/'config.json'
            config_path.write_text(json.dumps(dict(output={'basedir':str(root)},
                experiment={'name':'logged','comet_api_key':'private-test-value'})))
            def main(path):
                runtime.save_effective_config(self.model)
                print('run test message')
            try:
                runtime.run_logged(config_path,main)
                folder=runtime.ACTIVE_RUN
                data=json.loads((folder/'run.json').read_text())
                self.assertEqual(data['status'],'complete')
                self.assertNotIn('private-test-value',(folder/'run.json').read_text())
                effective=json.loads((folder/'effective_config.json').read_text())
                self.assertEqual(len(effective['rays']['nucleus']['vertices']),12)
                self.assertIn('run test message',(folder/'console.log').read_text())
            finally: runtime.ACTIVE_RUN=None

    def test_odd_volume_detection(self):
        predictor=CytoNucsPredictor(self.model,self.cfg,prob_thresh_nucleus=1.,prob_thresh_cytoplasm=1.)
        x=np.zeros((7,31,29,1),np.float32)
        n,c,_=predictor.predict_instances(x,normalize_img=False)
        self.assertEqual(n.shape,x.shape[:3]); self.assertEqual(c.shape,n.shape)

    def test_prompted_and_tiled_runtime(self):
        x=np.zeros((9,31,29,1),np.float32)
        n=np.zeros(x.shape[:3],np.int32); n[3:6,12:17,12:17]=1
        result=predict_instances_prompted(self.model,self.cfg,x,np.array([[4,14,14]]),np.array([.9]),
                                         nuc_labels=n,cyto_labels=n,mask_thresh=0.)
        self.assertEqual(result[0].shape,n.shape)
        self.assertGreater(result[0].max(),0)
        predictor=CytoNucsPredictor(self.model,self.cfg,prob_thresh_nucleus=1.,prob_thresh_cytoplasm=1.)
        n,c,_=predictor._predict_with_tiling(x,(1,1,1),False)
        cached=dict(predictor._detection_forward_cache)
        n2,c2,_=predictor._predict_with_tiling(x,(1,1,1),False)
        for key,fn in cached.items(): self.assertIs(predictor._detection_forward_cache[key],fn)
        np.testing.assert_array_equal(n,n2); np.testing.assert_array_equal(c,c2)
        self.assertEqual(n.shape,x.shape[:3]); self.assertEqual(c.shape,n.shape)

    def test_gradient_and_optimizer_restore(self):
        n=np.zeros((8,32,32),np.int32); n[2:6,10:20,10:20]=1
        c=np.zeros_like(n); c[1:7,6:24,6:24]=1
        y={'nucleus_instances':n[None],'cytoplasm_instances':c[None],
           'nucleus_target':np.repeat((n>0)[None],2,axis=0).astype('float32'),
           'cytoplasm_target':np.repeat((c>0)[None],2,axis=0).astype('float32')}
        for head,mask in (('nucleus',n),('cytoplasm',c)):
            p,d=compute_stardist_gt(mask,self.cfg,getattr(self.cfg,head+'_grid'))
            y[head+'_prob']=p[None]; y[head+'_dist']=d[None].astype('float32')
        x=tf.ones((1,8,32,32,5))
        loss=CytoNucsLoss(self.cfg)
        trainer=CytoNucsTrainer(self.model,self.cfg,loss,0)
        result=trainer.train_step(x,{k:tf.convert_to_tensor(v) for k,v in y.items()})
        self.assertTrue(np.isfinite(float(result[0])))
        self.assertGreater(float(result[4]),0)
        graph=trainer.train_step.get_concrete_function(x,{k:tf.convert_to_tensor(v) for k,v in y.items()}).graph
        self.assertFalse(any(op.type in ('EagerPyFunc','PyFunc','PrintV2') for op in graph.get_operations()))
        trainer.state_epoch.assign(3)
        path=ROOT/'.cache'/'test_models'/'synthetic_test'/'test.weights.h5'
        trainer.save_checkpoint(path)
        steps=int(trainer.optimizer.iterations.numpy())
        trainer.optimizer.iterations.assign(0); trainer.state_epoch.assign(0)
        self.assertTrue(trainer.restore_training_state(path))
        self.assertEqual(int(trainer.optimizer.iterations.numpy()),steps)
        self.assertEqual(int(trainer.state_epoch.numpy()),3)

    def test_complete_training_epoch(self):
        n=np.zeros((8,32,32),np.int32); n[2:6,10:20,10:20]=1
        c=np.zeros_like(n); c[1:7,6:24,6:24]=1
        y={'nucleus_instances':n[None], 'cytoplasm_instances':c[None],
           'nucleus_target':np.repeat((n>0)[None],2,axis=0).astype('float32'),
           'cytoplasm_target':np.repeat((c>0)[None],2,axis=0).astype('float32')}
        for head,mask in (('nucleus',n),('cytoplasm',c)):
            prob,dist=compute_stardist_gt(mask,self.cfg,getattr(self.cfg,head+'_grid'))
            y[head+'_prob']=prob[None]; y[head+'_dist']=dist[None].astype('float32')
        x=np.ones((1,8,32,32,5),np.float32)
        class Gen:
            def __len__(self): return 1
            def __getitem__(self,idx): return x,y
            def set_epoch(self,epoch): self.epoch=epoch
        gen=Gen()
        trainer=CytoNucsTrainer(self.model,self.cfg,CytoNucsLoss(self.cfg),0)
        gen.image_files=['synthetic.tif']
        gen.data_pool=[dict(image=x[0,...,:1],nucleus_mask=n,cytoplasm_mask=c,
                            assignments={1:1},filename='synthetic.tif')]
        callback=DiceCallback(self.model,gen,eval_every=1,prob_thresh_nucleus=1.,prob_thresh_cytoplasm=1.)
        callback.trainer=trainer
        trainer.train(gen,gen,epochs=1,steps_per_epoch=1,callbacks=[callback])
        self.assertEqual(trainer.history['fullval_jtpr'],[0.])
        self.assertEqual(len(trainer.history['val_loss']),1)
        self.assertTrue(np.isfinite(trainer.history['val_nucleus_mask'][0]))
        path=trainer.checkpoint_dir/'weights_best_val_loss.weights.h5'
        fresh=build_cytonucs_model(self.cfg,name='fresh_restore',basedir=str(ROOT/'.cache'/'test_models'))
        other=CytoNucsTrainer(fresh,self.cfg,CytoNucsLoss(self.cfg),0)
        self.assertTrue(other.restore_training_state(path))
        for a,b in zip(trainer.optimizer.variables,other.optimizer.variables):
            np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-6,atol=1e-7)
        other.current_epoch_var.assign(trainer.current_epoch_var)
        yt={k:tf.convert_to_tensor(v) for k,v in y.items()}
        trainer.train_step(tf.constant(x),yt); other.train_step(tf.constant(x),yt)
        for a,b in zip(self.model.keras_model.weights,fresh.keras_model.weights):
            np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-5,atol=1e-6)

    def test_negative_prompt_ce_has_gradient(self):
        loss=CytoNucsLoss(self.cfg)
        z=tf.Variable(tf.zeros((1,2,4,4,3)))
        with tf.GradientTape() as tape:
            value=loss._refinement_ce_loss(z,tf.zeros((1,2,4,4,1)),tf.zeros((1,2,4,4,1)))
        self.assertGreater(float(value),0)
        self.assertGreater(float(tf.linalg.global_norm([tape.gradient(value,z)])),0)


class GeneratorTests(unittest.TestCase):
    def test_real_gt_reproducible_order_and_augmentation(self):
        import tifffile, json
        (ROOT/'.cache').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            root=Path(tmp)
            for folder in ('images','nucleus_masks','cytoplasm_masks','assignments'):
                (root/folder).mkdir()
            n=np.zeros((8,32,32),np.uint16); n[2:6,10:20,10:20]=1
            c=np.zeros_like(n); c[1:7,6:24,6:24]=1
            img=np.random.default_rng(0).uniform(size=n.shape).astype('float32')
            for folder,data in (('images',img),('nucleus_masks',n),('cytoplasm_masks',c)):
                tifffile.imwrite(root/folder/'sample.tif',data,photometric='minisblack')
            (root/'assignments'/'sample.json').write_text(json.dumps({'1':1}))
            cfg=config(train_steps_per_epoch=3,negative_prompt_prob=0.0)
            gen=CytoNucsDataGenerator(root,cfg,cfg.rays,augmenter=cytonucs_augmenter_3d)
            gen.set_epoch(2)
            a,ya=gen[0]; b,yb=gen[0]
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(ya['nucleus_dist'],yb['nucleus_dist'])
            self.assertEqual(a.shape,(1,8,32,32,5))
            self.assertEqual(ya['nucleus_target'].shape,(2,8,32,32))
            prefetch=_Prefetcher(gen,workers=2,depth=2,count=2)
            try:
                for i in range(2):
                    np.testing.assert_array_equal(prefetch.next()[0],gen[i][0])
            finally: prefetch.stop()


if __name__=='__main__':
    unittest.main(verbosity=2)
