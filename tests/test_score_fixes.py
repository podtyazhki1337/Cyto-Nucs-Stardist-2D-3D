import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import ROOT, config
from test_architecture_v2 import batch
from test_mask_quality_target import gaussian
from cytonucs_config import CytoNucsConfig
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_losses import CytoNucsLoss, quality_supervision_weights
from cytonucs_model import build_cytonucs_model
from cytonucs_refinement_audit import RefinementStageAudit
import cytonucs_inference as inference


class ScoreFixTests(unittest.TestCase):
    def test_reserved_slot_preserves_z_pair_and_ownership(self):
        cfg=config(prompts_per_crop=3,detector_prompt_slots=1,predicted_prompt_fraction=0.,predicted_prompt_sample_peaks=True)
        x,y=batch(cfg); x=tf.ones_like(x)*.123
        prob=np.zeros((1,8,16,16,1),np.float32);prob[0,3,7,7,0]=.9
        pred={h+'_prob':tf.constant(prob) for h in ('nucleus','cytoplasm')}
        pred.update({h+'_dist':tf.ones((1,8,16,16,12))*4 for h in ('nucleus','cytoplasm')})
        stub=SimpleNamespace(config=cfg,prompt_detector=lambda x,training:pred,optimizer=SimpleNamespace(iterations=tf.Variable(0,dtype=tf.int64)))
        for h in ('nucleus','cytoplasm'):y[h+'_z_prompt_pair']=tf.constant([True])
        out=CytoNucsTrainer._predicted_prompts(stub,x,y).numpy()
        np.testing.assert_array_equal(out[...,:5],x.numpy()[...,:5])
        self.assertFalse(np.array_equal(out[...,5:],x.numpy()[...,5:]))
        # Explicit XY border is also enforced by the training peak sampler.
        cfg.proposal_border_nucleus=8
        out_border=tf.function(lambda a:CytoNucsTrainer._predicted_prompts(stub,a,y))(x)
        np.testing.assert_array_equal(out_border[...,5],x[...,5])
        cfg.proposal_border_nucleus=0
        # Strong foreign peak never replaces the selected positive target.
        prob[:]=0;prob[0,3,1,1,0]=.99
        for h in ('nucleus','cytoplasm'):pred[h+'_prob']=tf.constant(prob)
        np.testing.assert_array_equal(CytoNucsTrainer._predicted_prompts(stub,x,y),x)

    def test_real_generator_preserves_pairs_with_third_slot(self):
        from test_pipeline_review import make_dataset
        from cytonucs_data_generator import CytoNucsDataGenerator
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            root=Path(tmp);make_dataset(root)
            cfg=config(prompts_per_crop=3,detector_prompt_slots=1,predicted_prompt_sample_peaks=True,
                       cell_z_pair_prob=1.,nucleus_z_pair_prob=1.,negative_prompt_prob=0.)
            gen=CytoNucsDataGenerator(root,cfg,cfg.rays)
            x,y=gen[0]
            self.assertEqual(x.shape[-1],7)
            for h in ('nucleus','cytoplasm'):
                self.assertEqual(y[h+'_target'].shape[0],3)
                self.assertTrue(bool(y[h+'_z_prompt_pair'][0]))
                np.testing.assert_array_equal(y[h+'_target'][0],y[h+'_target'][1])

    def test_quality_boundary_gradient_and_no_negative_only_update(self):
        t=np.zeros((3,8,12,12),np.float32)
        t[0,2:6,3:9,3:9]=1;t[1,:,3:9,3:9]=1
        w,pos=quality_supervision_weights(tf.constant(t),3)
        np.testing.assert_array_equal(w,[1,0,1])
        q=tf.Variable([.2,.2,.2])
        with tf.GradientTape() as tape:loss=tf.reduce_sum(w*tf.square(q-[1.,1.,0.]))/tf.reduce_sum(w)
        grad=tape.gradient(loss,q).numpy()
        self.assertLess(grad[0],0);self.assertEqual(grad[1],0);self.assertGreater(grad[2],0)
        w,_=quality_supervision_weights(tf.constant(t[1:]),3)
        np.testing.assert_array_equal(w,[0,0])
        w,_=quality_supervision_weights(tf.zeros((2,5,5)),2)
        self.assertTrue(np.isfinite(w.numpy()).all())

    def test_cell_roi_expansion_recovers_full_object_keeps_nucleus_and_prompt(self):
        cfg=config(nucleus_grid=(1,1,1),cytoplasm_grid=(1,1,1),refinement_patch_size=[0,32,32],cell_roi_expansion_steps=2)
        shape=(8,32,96);gt=np.zeros(shape,np.int32);gt[:,6:26,8:88]=1
        image=(gt>0).astype(np.float32)[...,None];ctr=np.array([[4,16,48]])
        labels=np.zeros(shape,np.int32);labels[tuple(ctr[0])]=1
        seen=[]
        def logits(x):
            b=x[...,0];return tf.stack([10*(1-b),10*b,tf.zeros_like(b)],-1)
        def joint(x):return logits(x[0]),logits(x[0])
        def cell(x):
            from cytonucs_mask_quality import gaussian_prompt_geometry
            seen.append(gaussian_prompt_geometry(x[-1].numpy()[0,...,0]))
            return logits(x[0])
        results=[]
        for steps in (0,1,2,10):
            cfg.cell_roi_expansion_steps=steps
            m=SimpleNamespace();audit={'cytoplasm':RefinementStageAudit(gt)}
            with patch.object(inference,'_refinement_runners',return_value=(lambda x:[x]*7,joint)),patch.object(inference,'_cytoplasm_refinement_runner',return_value=cell):
                n,c,_=inference.predict_instances_prompted(m,cfg,image,ctr,np.ones(1),nuc_dists=np.full((1,12),2.),nuc_labels=labels,stage_audit=audit,verbose=False)
            results.append((n,c,audit['cytoplasm'].report()))
        self.assertEqual(len(results[3][2]['rois']),3)  # Full-image boundaries never trigger another expansion.
        np.testing.assert_array_equal(results[0][0],results[2][0])
        self.assertEqual(np.count_nonzero(results[2][1]),np.count_nonzero(gt))
        self.assertLess(np.count_nonzero(results[1][1]),np.count_nonzero(gt))
        self.assertEqual(results[2][2]['stages']['before_nms']['oracle_one_to_one_tp'],1)
        # Only the final expanded candidate enters threshold/component stages.
        self.assertEqual(results[2][2]['stages']['threshold']['oracle_one_to_one_tp'],1)
        for _,sigma in seen:np.testing.assert_allclose(sigma,[1,1,1],atol=1e-4)

    def test_explicit_proposals_rescue_border_and_close_peaks(self):
        cfg=config(proposal_mode='peaks',median_max_dist_nucleus=54.,rays_anisotropy_nucleus=(19,1,1))
        prob=np.zeros((8,32,32),np.float32);prob[3,1,7]=.9;prob[3,7,7]=.9;prob[3,7,9]=.8
        stub=SimpleNamespace(config=cfg,ndim=3,rays_nucleus=cfg.rays_nucleus,rays_cytoplasm=cfg.rays_cytoplasm,_current_type='nucleus')
        def run():
            inference.CytoNucsPredictor._stardist_postprocess(stub,prob,np.ones(prob.shape+(12,),np.float32),.4,.3)
            return np.asarray(stub._last_coords).astype(int).tolist()
        old=run();cfg.proposal_border_nucleus=0;cfg.proposal_peak_window_nucleus=[3,3,3];new=run()
        self.assertNotIn([3,2,14],old);self.assertIn([3,2,14],new)
        self.assertNotIn([3,14,18],old);self.assertIn([3,14,18],new)

    def test_all_json_contract_and_invalid_settings(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            for path in ROOT.glob('config_cytonucs*.json'):
                raw=json.loads(path.read_text());raw['training']['resume_from_checkpoint']=None;raw['inference']['checkpoint_path']=None;raw['output']['basedir']=tmp
                f=Path(tmp)/'c.json';f.write_text(json.dumps(raw));cfg=CytoNucsConfig.from_json(f)
                for section,settings in cfg.FIXES_SETTINGS.items():
                    for key in settings:self.assertEqual(getattr(cfg,key),raw[section][key],(path.name,key))
        for args in [dict(detector_prompt_slots=1),dict(cell_roi_expansion_steps=-1),dict(proposal_border_nucleus=True),dict(proposal_peak_window_nucleus=[2,3,3]),dict(mask_quality_ignore_boundary_targets=1)]:
            with self.assertRaises(ValueError):config(**args)

    def test_k2_checkpoint_adam_restore_k3_train_val(self):
        tf.keras.utils.set_random_seed(42)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            cfg=config(mask_quality_head=True,mask_quality_weight=.1,mask_quality_target='component',refinement_depth=1)
            model=build_cytonucs_model(cfg,basedir=tmp);trainer=CytoNucsTrainer(model,cfg,CytoNucsLoss(cfg),0)
            def data(c):
                x,y=batch(c);x=x.numpy()
                for i in range(1,x.shape[-1]):x[0,...,i]=gaussian(c.train_patch_size,(3,16,16),(2,4,4))
                return tf.constant(x),y
            x,y=data(cfg);trainer.train_step(x,y);trainer.state_epoch.assign(420)
            path=Path(tmp)/'weights_epoch_420.weights.h5';trainer.save_checkpoint(path)
            new=config(mask_quality_head=True,mask_quality_weight=.1,mask_quality_target='component',refinement_depth=1,prompts_per_crop=3,detector_prompt_slots=1,predicted_prompt_sample_peaks=True,predicted_prompt_start_epoch=1,mask_quality_ignore_boundary_targets=True,proposal_peak_window_nucleus=[3,3,3],proposal_peak_window_cytoplasm=[3,3,3],proposal_border_nucleus=0,proposal_border_cytoplasm=0,cell_roi_expansion_steps=2)
            other=build_cytonucs_model(new,basedir=tmp);other.load_weights(path)
            restored=CytoNucsTrainer(other,new,CytoNucsLoss(new),0)
            self.assertTrue(restored.restore_training_state(path));self.assertEqual(int(restored.state_epoch.numpy()),420)
            for a,b in zip(trainer.optimizer.variables,restored.optimizer.variables):np.testing.assert_array_equal(a.numpy(),b.numpy())
            self.assertEqual(model.keras_model.count_params(),other.keras_model.count_params())
            restored.current_epoch_var.assign(421);x,y=data(new)
            for result in (restored.train_step(x,y),restored.val_step(x,y)):self.assertTrue(np.isfinite(float(result[0])))
            self.assertIn('nucleus_quality_supervised_fraction',result[-1])

if __name__=='__main__':unittest.main()
