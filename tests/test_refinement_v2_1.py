"""Regression evidence for fixes after epoch70; no claims about real JTPR."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from cytonucs_config import CytoNucsConfig
from cytonucs_model import MaskQualityPool, build_cytonucs_model
from cytonucs_losses import CytoNucsLoss
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_postprocess import assemble_masks, mask_candidate
from cytonucs_data_generator import project_prompt_center
import cytonucs_inference as inference


class RefinementV21Tests(unittest.TestCase):
    def test_resume_preset_and_config_contract(self):
        raw = json.loads((ROOT/'config_cytonucs_v2_1.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            raw['output']['basedir'] = tmp
            self.assertTrue(raw['training']['resume_from_checkpoint'].endswith('weights_epoch_075.weights.h5'))
            raw['training']['resume_from_checkpoint']=str(Path(tmp)/'weights_epoch_070.weights.h5')
            (Path(tmp)/'geometry.json').write_bytes((ROOT/'fixes_v2/Epoch70/geometry.json').read_bytes())
            path=Path(tmp)/'config.json';path.write_text(json.dumps(raw))
            cfg=CytoNucsConfig.from_json(path)
            for section, fields in cfg.FIXES_SETTINGS.items():
                for key in fields:self.assertEqual(getattr(cfg,key),raw[section][key])
            self.assertTrue(raw['training']['resume_from_checkpoint'].endswith('weights_epoch_070.weights.h5'))
            self.assertEqual(cfg.mask_quality_pooling,'mask')
            self.assertFalse(cfg.mask_quality_use_scores)
            self.assertEqual(cfg.mask_quality_min,0)
        for invalid in [dict(mask_quality_pooling='bad'),dict(mask_ce_class_balance=1),
                        dict(mask_overlap_nucleus=1.1),dict(prompt_inside_target='true')]:
            with self.assertRaises(ValueError):config(**invalid)

    def test_quality_localization_and_stop_gradient(self):
        # Identical foreground, very different distant ROI context.
        h=np.zeros((1,2,4,8,2),np.float32);h[:,:,:,0,:]=3
        logits=np.zeros((1,2,4,8,3),np.float32);logits[...,0]=20
        logits[:,:,:,0,0]=0;logits[:,:,:,0,1]=20
        other=h.copy();other[:,:,:,1:,:]=100
        pool=MaskQualityPool()
        a=pool([tf.constant(h),tf.constant(logits)])
        b=pool([tf.constant(other),tf.constant(logits)])
        np.testing.assert_allclose(a,b,atol=1e-5)
        np.testing.assert_allclose(a,3,atol=1e-5)
        self.assertGreater(float(np.mean(other)-np.mean(h)),80)
        hv=tf.Variable(h);lv=tf.Variable(logits)
        with tf.GradientTape() as tape:loss=tf.reduce_sum(pool([hv,lv]))
        gh,gl=tape.gradient(loss,[hv,lv])
        self.assertIsNone(gl);self.assertGreater(float(tf.linalg.global_norm([gh])),0)
        empty=tf.constant(np.full_like(logits,-1e4));self.assertTrue(np.isfinite(pool([hv,empty])).all())

    def test_balanced_ce_preserves_small_instance_gradient(self):
        # Same class predictions; increase other-instance area by 100x.
        measured={}
        for balanced in (False,True):
            loss=CytoNucsLoss(config(mask_ce_class_balance=balanced))
            values=[]
            for length in (10,1000):
                z=tf.Variable(np.zeros((1,1,1,length,3),np.float32))
                tg=np.zeros((1,1,1,length,1),np.float32);tg[...,0,0]=1
                any_fg=np.ones_like(tg);any_fg[...,-1,0]=0
                with tf.GradientTape() as tape:value=loss._refinement_ce_loss(z,tg,any_fg)
                grad=tape.gradient(value,z).numpy()[0,0,0,0,1]
                self.assertTrue(np.isfinite(float(value)))
                values.append(abs(float(grad)))
            measured[balanced]=values
        self.assertAlmostEqual(measured[False][0]/measured[False][1],100,places=3)
        self.assertAlmostEqual(measured[True][0],measured[True][1],places=6)
        print('CE target gradients at 10/1000 voxels:',measured)

    def test_ce_empty_classes_are_finite(self):
        loss=CytoNucsLoss(config(mask_ce_class_balance=True))
        z=tf.Variable(tf.zeros((2,2,2,2,3)))
        for target,other in [(0,0),(0,1),(1,1)]:
            with tf.GradientTape() as tape:
                value=loss._refinement_ce_loss(z,tf.ones((2,2,2,2,1))*target,tf.ones((2,2,2,2,1))*other)
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(np.isfinite(tape.gradient(value,z)).all())

    def test_positive_centers_stay_on_owner(self):
        owner=np.zeros((4,12,12),bool);owner[:,2:10,2:10]=True;owner[:,4:8,4:8]=False
        for center in [(2,6,6),(-15,35,8),(2,2,2)]:
            result=project_prompt_center(center,owner,np.array([4,8,8]))
            self.assertTrue(owner[tuple(np.round(result).astype(int))])

    def test_peak_sampling_exposes_different_valid_hypotheses(self):
        cfg=config(predicted_prompt_fraction=1.,predicted_prompt_sample_peaks=True)
        x,y=batch(cfg);p=np.zeros((1,8,16,16,1),np.float32)
        p[0,3,7,7,0]=.9;p[0,5,9,9,0]=.8
        predictions={h+'_prob':tf.constant(p) for h in ('nucleus','cytoplasm')}
        predictions.update({h+'_dist':tf.ones((1,8,16,16,12))*4 for h in ('nucleus','cytoplasm')})
        stub=SimpleNamespace(config=cfg,prompt_detector=lambda x,training:predictions,
                             optimizer=SimpleNamespace(iterations=tf.Variable(0,dtype=tf.int64)))
        centers=[]
        for step in range(12):
            stub.optimizer.iterations.assign(step)
            out=CytoNucsTrainer._predicted_prompts(stub,x,y).numpy()
            repeat=CytoNucsTrainer._predicted_prompts(stub,x,y).numpy()
            np.testing.assert_array_equal(out,repeat)
            for channel in (1,3):
                center=np.unravel_index(np.argmax(out[0,...,channel]),out.shape[1:4])
                centers.append(center)
                self.assertIn(center,[(3,14,14),(5,18,18)])
        self.assertEqual(len(set(centers)),2)
        cfg.predicted_prompt_sample_peaks=False
        out=CytoNucsTrainer._predicted_prompts(stub,x,y).numpy()
        np.testing.assert_array_equal(out[...,1],out[...,3])

    def test_overlap_dedup_removes_fragment_keeps_neighbour(self):
        shape=(4,20,30)
        full=np.zeros(shape,bool);full[:,2:14,2:14]=True
        fragment=np.zeros(shape,bool);fragment[:,4:9,4:9]=True
        neighbour=np.zeros(shape,bool);neighbour[:,2:14,16:28]=True
        cs=[mask_candidate(np.where(a,q,0),a,(0,0,0)) for a,q in [(full,.9),(fragment,.8),(neighbour,.85)]]
        old,os=assemble_masks(cs,shape,dedup_iou=.8)
        fixed,fs=assemble_masks(cs,shape,dedup_iou=.5,overlap_threshold=.85)
        self.assertEqual(os['after_mask_nms'],3);self.assertEqual(fs['after_mask_nms'],2)
        self.assertEqual(int(fixed.max()),2)
        self.assertEqual(len(np.unique(fixed[full])),1)

    def test_diagnostics_separate_rejection_causes_and_bypass_quality(self):
        cfg=config(mask_quality_head=True,mask_quality_min=.25)
        shape=(8,32,32);labels=np.zeros(shape,np.int32);labels[4,16,16]=1
        for reason in ('accepted','quality_rejected','empty_threshold','missing_component','nonfinite'):
            z=np.zeros((1,8,16,16,3),np.float32);z[...,0]=10
            quality=.9
            if reason in ('accepted','quality_rejected'):z[...,1]=20
            elif reason=='missing_component':z[:,0,0,0,1]=20
            elif reason=='nonfinite':z[...,1]=np.nan
            if reason=='quality_rejected':quality=.1
            def outputs(inputs):return tf.constant(z),tf.constant(z),tf.constant([[quality]]),tf.constant([[quality]])
            model=SimpleNamespace()
            with patch.object(inference,'_refinement_runners',return_value=(lambda x:[x]*7,outputs)), \
                 patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda x:None):
                inference.predict_instances_prompted(model,cfg,np.zeros(shape+(1,),np.float32),
                    np.array([[4,16,16]]),np.ones(1),nuc_labels=labels,verbose=False)
                for diag in model.last_refinement_diagnostics.values():
                    self.assertEqual(diag[reason],1)
                    self.assertEqual(sum(diag[k] for k in ('accepted','quality_rejected','empty_threshold','missing_component','nonfinite')),diag['attempted'])
                if reason=='quality_rejected':
                    cfg.mask_quality_min=0;cfg.mask_quality_use_scores=False
                    n,c,_=inference.predict_instances_prompted(model,cfg,np.zeros(shape+(1,),np.float32),
                        np.array([[4,16,16]]),np.ones(1),nuc_labels=labels,verbose=False)
                    self.assertEqual(n.max(),1);self.assertEqual(c.max(),1)
                    cfg.mask_quality_min=.25

    def test_v2_weights_and_adam_restore_into_local_quality_pool(self):
        cfg=config(refinement_depth=1,mask_quality_head=True,mask_quality_weight=.1)
        old=build_cytonucs_model(cfg);x,y=batch(cfg)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            old.basedir=tmp
            trainer=CytoNucsTrainer(old,cfg,CytoNucsLoss(cfg),0.)
            trainer.train_step(x,y);trainer.state_epoch.assign(70)
            trainer.state_best_full.assign(.23);trainer.state_best_val.assign(6.)
            weights=Path(tmp)/'weights_epoch_070.weights.h5';trainer.save_checkpoint(weights)
            # Match the actual epoch70 sidecar schema (no pooling key).
            side=Path(str(weights)+'.architecture.json');sig=json.loads(side.read_text());sig.pop('mask_quality_pooling');side.write_text(json.dumps(sig))
            nc=config(refinement_depth=1,mask_quality_head=True,mask_quality_pooling='mask',
                      mask_quality_weight=.1,mask_ce_class_balance=True,reset_best_on_resume=True,
                      predicted_prompt_sample_peaks=True,predicted_prompt_fraction=.25,predicted_prompt_start_epoch=1)
            new=build_cytonucs_model(nc);new.basedir=tmp;new.load_weights(weights)
            before=old.keras_model(x);after=new.keras_model(x)
            for key in before:
                if not key.endswith('_quality'):np.testing.assert_allclose(before[key],after[key],atol=1e-6)
            nt=CytoNucsTrainer(new,nc,CytoNucsLoss(nc),0.);self.assertTrue(nt.restore_training_state(weights))
            self.assertEqual(int(nt.state_epoch.numpy()),70)
            self.assertEqual(int(nt.optimizer.iterations.numpy()),1)
            for a,b in zip(trainer.optimizer.variables,nt.optimizer.variables):np.testing.assert_array_equal(a.numpy(),b.numpy())
            for a,b in zip(old.keras_model.trainable_variables,new.keras_model.trainable_variables):np.testing.assert_array_equal(a.numpy(),b.numpy())
            self.assertEqual(float(nt.state_best_full.numpy()),float('-inf'))
            self.assertTrue(np.isfinite(float(nt.train_step(x,y)[0])))
            full=new.keras_model(x);features=list(new.refinement_features_model(x[...,:1]))
            runtime=new.refinement_with_quality_model(features+[x[...,1:2],x[...,2:3]])
            np.testing.assert_allclose(full['nucleus_quality'][:1],runtime[2],atol=1e-6)
            np.testing.assert_allclose(full['cytoplasm_quality'][:1],runtime[3],atol=1e-6)

    def test_2d_local_quality_runtime(self):
        cfg=config(ndim=2,train_patch_size=(32,32),nucleus_grid=(2,2),cytoplasm_grid=(2,2),
                   anisotropy=(1,1),refinement_depth=1,mask_quality_head=True,mask_quality_pooling='mask')
        model=build_cytonucs_model(cfg)
        x=tf.random.stateless_uniform((1,32,32,5),seed=(21,42))
        full=model.keras_model(x);features=list(model.refinement_features_model(x[...,:1]))
        out=model.refinement_with_quality_model(features+[x[...,1:2],x[...,2:3]])
        for key,actual in zip(('nucleus_mask_logit','cytoplasm_mask_logit','nucleus_quality','cytoplasm_quality'),out):
            np.testing.assert_allclose(full[key][:1],actual,atol=1e-6)

    def test_disabled_quality_cannot_veto_finite_mask(self):
        cfg=config(mask_quality_head=True,mask_quality_min=0.,mask_quality_use_scores=False)
        labels=np.zeros((8,32,32),np.int32);labels[4,16,16]=1
        z=np.zeros((1,8,16,16,3),np.float32);z[...,1]=10
        def outputs(inputs):return tf.constant(z),tf.constant(z),tf.constant([[np.nan]]),tf.constant([[np.nan]])
        model=SimpleNamespace()
        with patch.object(inference,'_refinement_runners',return_value=(lambda x:[x]*7,outputs)), \
             patch.object(inference,'_cytoplasm_refinement_runner',return_value=lambda x:None):
            n,c,_=inference.predict_instances_prompted(model,cfg,np.zeros(labels.shape+(1,),np.float32),
                np.array([[4,16,16]]),np.ones(1),nuc_labels=labels,verbose=False)
        self.assertEqual(n.max(),1);self.assertEqual(c.max(),1)
        self.assertEqual(model.last_refinement_diagnostics['cytoplasm']['quality_nonfinite'],1)


if __name__=='__main__':unittest.main()
