"""Behavioral regressions for architecture v2 and legacy checkpoint compatibility."""
import json
import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from cytonucs_config import CytoNucsConfig
from cytonucs_model import build_cytonucs_model
from cytonucs_losses import CytoNucsLoss, compute_stardist_gt
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_postprocess import mask_candidate, assemble_masks
from cytonucs_data_generator import cytonucs_augmenter_2d, prepare_cytonucs_dataset
from cytonucs_runtime import read_model_image, resolve_resume_epoch
import cytonucs_inference as inference


def batch(cfg):
    shape=cfg.train_patch_size
    n=np.zeros(shape,np.int32);c=n.copy()
    n[2:6,12:20,12:20]=1;c[1:7,6:26,6:26]=1
    x=np.stack([.1+.4*c+.4*n,n.astype(float),n.astype(float)]*1,axis=-1)[None].astype('float32')
    x=np.concatenate([x]+[x[...,-2:] for _ in range(cfg.prompts_per_crop-1)],axis=-1)
    y={}
    for head,lab in [('nucleus',n),('cytoplasm',c)]:
        p,d=compute_stardist_gt(lab,cfg,getattr(cfg,head+'_grid'),
            rays=getattr(cfg,'rays_'+head),aniso=cfg.anisotropy)
        y[head+'_instances']=tf.constant(lab[None])
        y[head+'_prob']=tf.constant(p[None])
        y[head+'_dist']=tf.constant(d[None],tf.float32)
        y[head+'_target']=tf.constant(np.repeat(lab[None],cfg.prompts_per_crop,axis=0),tf.float32)
    return tf.constant(x),y


class ArchitectureV2Tests(unittest.TestCase):
    def test_v2_json_reaches_every_setting(self):
        raw=json.loads((ROOT/'config_cytonucs_v2.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            raw['output']['basedir']=tmp
            path=Path(tmp)/'config.json';path.write_text(json.dumps(raw))
            cfg=CytoNucsConfig.from_json(path)
            for section,defaults in cfg.FIXES_SETTINGS.items():
                for key in defaults:self.assertEqual(getattr(cfg,key),raw[section][key],key)
            self.assertEqual(cfg.refinement_depth,2)
            self.assertIsNone(raw['training']['resume_from_checkpoint'])

    def test_invalid_architecture_settings(self):
        for change in [dict(unet_pool_schedule=[[2,2,2]]),dict(unet_kernel_schedule=[[2,3,3]]*3),
                       dict(refinement_gradient_scale=1.5),dict(mask_quality_weight=.1),
                       dict(proposal_mode='peaks',refinement_mode='polyhedron'),
                       dict(refinement_patch_size=[-1,32,32])]:
            with self.assertRaises(ValueError):config(**change)

    def test_multiscale_train_and_runtime_quality_agree(self):
        tf.keras.utils.set_random_seed(45)
        cfg=config(refinement_depth=1,refinement_gradient_scale=.1,mask_quality_head=True)
        m=build_cytonucs_model(cfg)
        x,y=batch(cfg);full=m.keras_model(x)
        features=list(m.refinement_features_model(x[...,:1]))
        out=m.refinement_with_quality_model(features+[x[...,1:2],x[...,2:3]])
        for actual,key in zip(out,('nucleus_mask_logit','cytoplasm_mask_logit','nucleus_quality','cytoplasm_quality')):
            np.testing.assert_allclose(actual,full[key][:1],atol=1e-5,rtol=1e-5)
        cell=m.cytoplasm_refinement_model([features[i] for i in (1,2,5,6)]+[x[...,2:3]])
        np.testing.assert_allclose(cell[0],out[1],atol=1e-5)
        np.testing.assert_allclose(cell[1],out[3],atol=1e-5)

    def test_gradient_scale_opens_only_decoder_feature_path(self):
        for scale in (0.,.1):
            tf.keras.utils.set_random_seed(48)
            m=build_cytonucs_model(config(refinement_depth=1,refinement_gradient_scale=scale)).keras_model
            x,_=batch(config())
            with tf.GradientTape() as tape:
                out=m(x);loss=tf.reduce_mean(tf.square(out['nucleus_mask_logit']))
            variables=m.trainable_variables;grads=tape.gradient(loss,variables)
            enc=[g for g,v in zip(grads,variables) if 'enc_l' in v.path and g is not None]
            self.assertEqual(len(enc)>0,scale>0)
            if scale:self.assertGreater(float(tf.linalg.global_norm(enc)),0)
            self.assertTrue(all(g is None for g,v in zip(grads,variables)
                                if 'nucleus_prob_logit' in v.path or 'nucleus_dist/' in v.path))

    def test_nonshared_encoder_and_kernel_schedule_are_real(self):
        cfg=config(shared_encoder=False,unet_pool_schedule=[[1,2,2],[2,2,2]],
                   unet_kernel_schedule=[[1,3,3],[3,3,3],[3,3,3]],unet_activation='elu')
        m=build_cytonucs_model(cfg)
        self.assertEqual(m._architecture_plan['pools'],[(1,2,2),(2,2,2)])
        self.assertEqual(m.keras_model.get_layer('enc_l0_c1').kernel_size,(1,3,3))
        self.assertEqual(m.keras_model.get_layer('cell_enc_l0_c1').activation.__name__,'elu')
        x,_=batch(cfg);self.assertEqual(m.keras_model(x)['nucleus_prob'].shape,(1,8,16,16,1))

    def test_actual_pre_v2_weights_load_identically(self):
        original=ROOT/'audit/history/20260917T063359783906Z_fixes_v2/before/cytonucs_model.py'
        spec=importlib.util.spec_from_file_location('pre_v2_model',original)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cfg=config();old=module.build_cytonucs_model(cfg);new=build_cytonucs_model(cfg)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            weights=Path(tmp)/'old.weights.h5';old.keras_model.save_weights(weights)
            new.load_weights(weights)
            x,_=batch(cfg)
            for k,v in old.keras_model(x).items():np.testing.assert_allclose(v,new.keras_model(x)[k],atol=1e-6)
            changed=build_cytonucs_model(config(refinement_depth=1))
            with self.assertRaisesRegex(ValueError,'metadata missing'):changed.load_weights(weights)

    def test_train_step_predicted_prompts_losses_and_checkpoint(self):
        cfg=config(refinement_depth=1,refinement_gradient_scale=.1,mask_quality_head=True,
                   mask_quality_weight=.1,prompt_consistency_weight=.1,
                   predicted_prompt_fraction=1.,predicted_prompt_start_epoch=1,
                   enable_consistency=True,lambda_consistency=.1)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            m=build_cytonucs_model(cfg,basedir=tmp,name='v2')
            loss=CytoNucsLoss(cfg);trainer=CytoNucsTrainer(m,cfg,loss,0.)
            x,y=batch(cfg)
            mixed=trainer._predicted_prompts(x,y)
            self.assertTrue(np.isfinite(mixed.numpy()).all())
            self.assertFalse(np.allclose(mixed[...,1:],x[...,1:]))
            empty=dict(y);empty['nucleus_target']=tf.zeros_like(y['nucleus_target'])
            self.assertTrue(np.isfinite(trainer._predicted_prompts(x,empty)).all())
            # Multiple images and K-major targets must retain their batch shape.
            y2={key:tf.repeat(value,2,axis=0) for key,value in y.items()}
            self.assertEqual(trainer._predicted_prompts(tf.repeat(x,2,axis=0),y2).shape,(2,8,32,32,5))
            result=trainer.train_step(x,y)
            self.assertTrue(np.isfinite(float(result[0])))
            self.assertGreater(float(result[-1]['mask_quality']),0)
            trainer.state_epoch.assign(12)
            weights=Path(tmp)/'weights_best.weights.h5';trainer.save_checkpoint(weights)
            self.assertEqual(resolve_resume_epoch(weights),12)
            m.load_weights(weights)
            changed=build_cytonucs_model(config())
            with self.assertRaisesRegex(ValueError,'architecture differs'):changed.load_weights(weights)

    def test_prompt_consistency_only_compares_same_image_and_instance(self):
        loss=CytoNucsLoss(config(prompts_per_crop=2))
        target=tf.ones((4,2,2,2,1))
        pred=tf.Variable(tf.concat([tf.zeros((2,2,2,2,1)),tf.ones((2,2,2,2,1))],0))
        with tf.GradientTape() as tape:value=loss._prompt_consistency(pred,target)
        self.assertAlmostEqual(float(value),1.)
        self.assertGreater(float(tf.linalg.global_norm([tape.gradient(value,pred)])),0)
        # K-major B=2: compare sample0 with sample0, not sample0 with sample1.
        different=tf.concat([tf.ones((2,2,2,2,1)),tf.zeros((2,2,2,2,1))],0)
        self.assertEqual(float(loss._prompt_consistency(pred,different)),0.)

    def test_consistency_respects_edt_and_reverse_grids(self):
        cfg=config(nucleus_grid=(2,4,4),cytoplasm_grid=(1,2,2))
        loss=CytoNucsLoss(cfg)
        a=tf.ones((1,4,8,8,1));self.assertEqual(loss._match_to_cytoplasm(a).shape,(1,8,16,16,1))
        gt=tf.reshape(tf.linspace(.1,.9,64),(1,4,4,4,1))
        self.assertEqual(float(loss._consistency_loss(gt,gt)),0.)
        self.assertGreater(float(loss._consistency_loss(1-gt,gt)),0.)
        result=CytoNucsLoss(config(median_max_dist_nucleus=None))._stardist_loss(
            tf.ones((1,2,2,2,1))*.5,tf.ones((1,2,2,2,1))*.4,
            tf.ones((1,2,2,2,12)),tf.ones((1,2,2,2,12)),name='nucleus')
        self.assertTrue(np.isfinite(float(result['distance_rel'])))

    def test_nested_mask_removed_without_merging_disjoint_cells(self):
        shape=(8,24,24);big=np.zeros(shape,bool);big[:,2:14,2:14]=1
        small=np.zeros(shape,bool);small[2:6,6:10,6:10]=1
        other=np.zeros(shape,bool);other[:,16:22,16:22]=1
        candidates=[mask_candidate(np.full(shape,p,np.float32),mask,(0,0,0))
                    for mask,p in ((big,.8),(small,.95),(other,.9))]
        out,stats=assemble_masks(candidates,shape,.5,.95)
        self.assertEqual(stats['written'],2)
        self.assertEqual(len(np.unique(out[big])),1)
        self.assertNotEqual(out[3,7,7],out[3,18,18])

    def test_full_z_refinement_and_peaks_defer_polyhedral_nms(self):
        cfg=config(train_patch_size=(16,32,32),refinement_patch_size=[0,32,32],proposal_mode='peaks')
        shape=(40,32,32);image=np.zeros(shape+(1,),np.float32);labels=np.zeros(shape,np.int32)
        labels[20,16,16]=1
        def refine(inputs):
            sh=inputs[-1].shape;a=np.zeros((1,sh[1],sh[2]//2,sh[3]//2,3),np.float32);a[...,1]=10
            return tf.constant(a),tf.constant(a)
        with patch.object(inference,'_refinement_runners',return_value=(lambda x:[x]*7,refine)), \
             patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda x:refine(x)[1]):
            n,c,_=inference.predict_instances_prompted(SimpleNamespace(),cfg,image,np.array([[20,16,16]]),
                np.ones(1),nuc_labels=labels,cyto_labels=np.zeros_like(labels),verbose=False)
        self.assertEqual(int(np.ptp(np.argwhere(n)[:,0])+1),40)
        predictor=inference.CytoNucsPredictor(SimpleNamespace(),cfg)
        predictor._current_type='nucleus'
        prob=np.zeros((16,16,16),np.float32);prob[4,4,4]=.9;prob[12,12,12]=.8
        dist=np.full((16,16,16,12),100.,np.float32)
        with patch('stardist.nms.non_maximum_suppression_3d_sparse',side_effect=AssertionError('Must defer NMS')):
            anchors=predictor._stardist_postprocess(prob,dist,.5,.3)
        self.assertEqual(int(anchors.max()),2)

    def test_2d_augmentation_and_tiff_axes_dtype(self):
        import tifffile
        a=np.arange(32*32,dtype=np.float32).reshape(32,32);lab=(a>100).astype('int32')
        cfg=config(ndim=2,train_patch_size=(32,32),nucleus_grid=(2,2),cytoplasm_grid=(2,2),anisotropy=(1,1))
        r1=cytonucs_augmenter_2d(a,lab,lab,np.random.default_rng(42),cfg)
        r2=cytonucs_augmenter_2d(a,lab,lab,np.random.default_rng(42),cfg)
        for x,y in zip(r1,r2):np.testing.assert_array_equal(x,y)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            p=Path(tmp)/'rgb.tif';image=np.full((8,3,12,16),1024,np.uint16)
            tifffile.imwrite(p,image,metadata={'axes':'ZCYX'},photometric='minisblack')
            read=read_model_image(p,3,3)
            self.assertEqual(read.shape,(8,12,16,3));self.assertEqual(read.dtype,np.uint16)
            self.assertEqual(int(read.max()),1024)

    def test_preprocessing_preserves_uint16_rgb_and_instance_ids(self):
        import tifffile
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            root=Path(tmp)
            for folder in ('images','nuc','cyto'):(root/folder).mkdir()
            for i in range(3):
                rgb=np.full((4,8,8,3),1024+i,np.uint16)
                lab=np.zeros((4,8,8),np.uint32);lab[:,2:6,2:6]=70000+i
                tifffile.imwrite(root/'images'/f'{i}.tiff',rgb,photometric='rgb')
                for folder in ('nuc','cyto'):tifffile.imwrite(root/folder/f'{i}.tiff',lab,photometric='minisblack')
            prepare_cytonucs_dataset(root/'images',root/'nuc',root/'cyto',root/'out',
                                     compute_assignments=False,ndim=3)
            for split in ('train','val','test'):
                img_path=next((root/'out'/split/'images').glob('*.tiff'))
                img=tifffile.imread(img_path);mask=tifffile.imread(img_path.parent.parent/'nucleus_masks'/img_path.name)
                self.assertEqual(img.dtype,np.uint16);self.assertGreaterEqual(int(img.max()),1024)
                self.assertEqual(mask.dtype,np.uint32);self.assertGreaterEqual(int(mask.max()),70000)

    def test_threshold_search_uses_effective_head_overrides(self):
        cfg=config(mask_thresh_nucleus=.9,mask_thresh_cytoplasm=.8)
        predictor=inference.CytoNucsPredictor(SimpleNamespace(),cfg)
        gt=np.zeros((8,32,32),np.int32);gt[2:6,8:24,8:24]=1
        seen=[]
        def predict(img,normalize_img=False):
            seen.append((cfg.mask_thresh_nucleus,cfg.mask_thresh_cytoplasm))
            correct=cfg.mask_thresh_nucleus==.4 and cfg.mask_thresh_cytoplasm==.4
            predictor._prompt_assignments={1:1} if correct else {}
            result=gt if correct else np.zeros_like(gt)
            return result,result,{}
        with patch.object(predictor,'predict_instances',side_effect=predict):
            result=predictor.optimize_thresholds([np.zeros(gt.shape+(1,),np.float32)],[gt],[gt],
                prob_thresh_range=[.5],nms_thresh_range=[.3],mask_thresh_range=[.4],val_assignments=[{1:1}])
        self.assertIn((.9,.8),seen)
        self.assertEqual(result['mask_thresh_nucleus'],.4)
        self.assertEqual(result['mask_thresh_cytoplasm'],.4)
        self.assertEqual(result['jtpr'],1.)

    def test_consistency_toggle_changes_total_loss(self):
        cfg=config(enable_consistency=True,lambda_consistency=.5)
        m=build_cytonucs_model(cfg);x,y=batch(cfg);pred=m.keras_model(x)
        enabled=CytoNucsLoss(cfg);a=enabled(y,pred,current_epoch=100)
        disabled=CytoNucsLoss(config(enable_consistency=False));b=disabled(y,pred,current_epoch=100)
        self.assertGreater(float(enabled.last_components['consistency']),0.)
        self.assertAlmostEqual(float(a-b),float(enabled.last_components['consistency']),places=5)

    def test_quality_rejects_bad_masks_in_runtime(self):
        cfg=config(mask_quality_head=True,mask_quality_min=.25)
        shape=(8,32,32);labels=np.zeros(shape,np.int32);labels[4,16,16]=1
        def outputs(inputs):
            logits=np.zeros((1,8,16,16,3),np.float32);logits[...,1]=10
            return tf.constant(logits),tf.constant(logits),tf.constant([[.1]]),tf.constant([[.9]])
        with patch.object(inference,'_refinement_runners',return_value=(lambda x:[x]*7,outputs)), \
             patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda x:outputs(x)[1]):
            n,c,_=inference.predict_instances_prompted(SimpleNamespace(),cfg,np.zeros(shape+(1,),np.float32),
                np.array([[4,16,16]]),np.ones(1),nuc_labels=labels,cyto_labels=np.zeros_like(labels),verbose=False)
        self.assertEqual(int(n.max()),0)
        self.assertEqual(int(c.max()),1)


if __name__=='__main__':unittest.main()
