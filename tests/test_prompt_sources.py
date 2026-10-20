import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from test_fixes import ROOT, config
from test_architecture_v2 import batch
from test_mask_quality_target import gaussian
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_model import build_cytonucs_model
from cytonucs_losses import CytoNucsLoss
from cytonucs_mask_quality import gaussian_prompt_geometry


def fixture(fraction=1., batch_size=1, grids=False):
    cfg=config(prompts_per_crop=3,detector_prompt_slots=1,predicted_prompt_fraction=1.,predicted_prompt_sample_peaks=True,
               cell_predicted_nucleus_fraction=fraction,nucleus_grid=(1,2,2),cytoplasm_grid=(2,4,4) if grids else (1,2,2))
    shape=(8,32,32);x=tf.fill((batch_size,)+shape+(7,),.123);pred={};y={}
    for h,center,distance in [('nucleus',(2,8,8),8.),('cytoplasm',(4,16,16),20.)]:
        grid=getattr(cfg,h+'_grid');spatial=tuple(s//g for s,g in zip(shape,grid))
        prob=np.zeros((batch_size,)+spatial+(1,),np.float32)
        for b in range(batch_size):prob[(b,)+tuple(c//g for c,g in zip(center,grid))+(0,)]=.9
        pred[h+'_prob']=tf.constant(prob);pred[h+'_dist']=tf.ones((batch_size,)+spatial+(12,))*distance
        target=np.zeros((batch_size,)+shape,np.float32);target[:,1:7,4:24,4:24]=1
        y[h+'_target']=tf.constant(np.concatenate([target]*3))
        y[h+'_instances']=tf.constant(target,tf.int32)
    stub=SimpleNamespace(config=cfg,prompt_detector=lambda x,training:pred,optimizer=SimpleNamespace(iterations=tf.Variable(0,dtype=tf.int64)))
    return stub,x,y,pred


class PromptSourceTests(unittest.TestCase):
    def test_cell_uses_nuclear_grid_center_and_sigma_b2(self):
        stub,x,y,pred=fixture(batch_size=2,grids=True)
        out,fraction=tf.function(lambda a:CytoNucsTrainer._predicted_prompts(stub,a,y,True))(x)
        self.assertEqual(float(fraction),1.)
        vertices=8*np.asarray(stub.config.rays_nucleus.vertices)
        expected=np.maximum((vertices.max(0)-vertices.min(0))/4,1)
        for b in range(2):
            for ch in (2,4,6):
                center,sigma=gaussian_prompt_geometry(out[b,...,ch].numpy())
                np.testing.assert_allclose(center,(2,8,8),atol=1e-4)
                np.testing.assert_allclose(sigma,expected,rtol=1e-4)
        np.testing.assert_array_equal(out[...,:1],x[...,:1])

    def test_no_owned_nuclear_peak_falls_back_to_cell_and_z_protection(self):
        stub,x,y,pred=fixture()
        for h in ('nucleus','cytoplasm'):y[h+'_z_prompt_pair']=tf.constant([True])
        p=np.zeros_like(pred['nucleus_prob']);p[0,2,1,1,0]=.99;pred['nucleus_prob']=tf.constant(p)
        out,fraction=CytoNucsTrainer._predicted_prompts(stub,x,y,True)
        np.testing.assert_array_equal(out[...,:5],x[...,:5]);self.assertEqual(float(fraction),0.)
        ctr,_=gaussian_prompt_geometry(out[0,...,6].numpy());np.testing.assert_allclose(ctr,(4,16,16),atol=1e-4)

    def test_nucleus_negative_inside_cell_without_mislabeling_real_nucleus(self):
        stub,x,y,pred=fixture(fraction=0.)
        y['nucleus_instances']=tf.zeros_like(y['nucleus_instances'])
        y['nucleus_target']=tf.zeros_like(y['nucleus_target']);y['cytoplasm_target']=tf.zeros_like(y['cytoplasm_target'])
        out=CytoNucsTrainer._predicted_prompts(stub,x,y)
        np.testing.assert_array_equal(out[...,2::2],x[...,2::2])
        self.assertFalse(np.array_equal(out[...,1::2],x[...,1::2]))
        # Same peak on a real nucleus must not be used for an empty nuclear target.
        actual=np.zeros((1,8,32,32),np.int32);actual[0,2,8,8]=1;y['nucleus_instances']=tf.constant(actual)
        np.testing.assert_array_equal(CytoNucsTrainer._predicted_prompts(stub,x,y),x)

    def test_mixture_reproducible_and_measured_after_protection(self):
        stub,x,y,pred=fixture(fraction=.5,batch_size=16)
        y['cytoplasm_z_prompt_pair']=tf.ones([16],tf.bool)
        a,frac=CytoNucsTrainer._predicted_prompts(stub,x,y,True)
        b,frac2=CytoNucsTrainer._predicted_prompts(stub,x,y,True)
        np.testing.assert_array_equal(a,b);self.assertEqual(float(frac),float(frac2))
        self.assertGreater(float(frac),0);self.assertLess(float(frac),1/3)
        np.testing.assert_array_equal(a[...,2],x[...,2]);np.testing.assert_array_equal(a[...,4],x[...,4])

    def test_real_training_and_new_preset_contract(self):
        cfg=config(prompts_per_crop=3,detector_prompt_slots=1,predicted_prompt_fraction=.25,predicted_prompt_sample_peaks=True,
                   cell_predicted_nucleus_fraction=.5,mask_quality_head=True,mask_quality_weight=.1,mask_quality_target='component',
                   mask_quality_ignore_boundary_targets=True,predicted_prompt_start_epoch=1,refinement_depth=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            model=build_cytonucs_model(cfg,basedir=tmp);trainer=CytoNucsTrainer(model,cfg,CytoNucsLoss(cfg),0)
            x,y=batch(cfg);x=x.numpy()
            for i in range(1,7):x[0,...,i]=gaussian((8,32,32),(3,16,16),(2,4,4))
            x=tf.constant(x);trainer.current_epoch_var.assign(0)
            cold=trainer.train_step(x,y);self.assertEqual(float(cold[-1]['cytoplasm_nuclear_detector_prompt_fraction']),0.)
            trainer.current_epoch_var.assign(421)
            hot=trainer.train_step(x,y);self.assertTrue(np.isfinite(float(hot[0])))
            self.assertIn('cytoplasm_nuclear_detector_prompt_fraction',hot[-1])
            self.assertTrue(np.isfinite(float(trainer.val_step(x,y)[0])))
        with self.assertRaises(ValueError):config(cell_predicted_nucleus_fraction=1.1,predicted_prompt_sample_peaks=True)
        with self.assertRaises(ValueError):config(cell_predicted_nucleus_fraction=.5)
        raw=json.loads((ROOT/'config_cytonucs_v2_10.json').read_text())
        self.assertEqual(raw['training']['cell_predicted_nucleus_fraction'],.5)
        self.assertTrue(raw['training']['resume_from_checkpoint'].endswith('weights_epoch_420.weights.h5'))

if __name__=='__main__':unittest.main()
