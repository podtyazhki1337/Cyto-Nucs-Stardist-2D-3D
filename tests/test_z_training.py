import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import axial_prompt_centers, CytoNucsDataGenerator
from cytonucs_losses import CytoNucsLoss
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_inference import CytoNucsPredictor
from cytonucs_postprocess import mask_candidate, assemble_masks
from cytonucs_model import build_cytonucs_model


class ZTrainingTests(unittest.TestCase):
    def test_axial_points_in_concave_sparse_mask(self):
        m=np.zeros((12,9,9),bool);m[1:11,1:8,1:8]=True;m[:,3:6,3:6]=False
        centers=axial_prompt_centers(m)
        self.assertGreaterEqual(centers[1][0]-centers[0][0],4)
        for c in centers:self.assertTrue(m[tuple(c.astype(int))])
        self.assertIsNone(axial_prompt_centers(m[1:2]))
        self.assertIsNone(axial_prompt_centers(np.zeros_like(m)))

    def test_two_z_peaks_and_fragments_survive_actual_postprocess(self):
        cfg=config(proposal_mode='peaks')
        stub=SimpleNamespace(config=cfg,ndim=3,rays_nucleus=cfg.rays_nucleus,rays_cytoplasm=cfg.rays_cytoplasm)
        p=np.zeros((12,16,16),np.float32);p[3,8,8]=.9;p[8,8,8]=.8
        anchors=CytoNucsPredictor._stardist_postprocess(stub,p,np.ones(p.shape+(12,),np.float32)*3,.3,.5)
        self.assertEqual(int(anchors.max()),2)
        a=np.zeros(anchors.shape,bool);b=a.copy();a[1:6,12:20,12:20]=1;b[6:11,12:20,12:20]=1
        candidates=[mask_candidate(m.astype('float32'),m,(0,0,0)) for m in (a,b)]
        for mode in ('voxel','object_quality'):
            out,stats=assemble_masks(candidates,anchors.shape,assembly_mode=mode)
            self.assertEqual(stats['written'],2)
            self.assertNotEqual(out[3,16,16],out[8,16,16])

    def test_generator_full_target_and_preserved_nuclear_branch(self):
        cfg=config(cell_z_pair_prob=1.,negative_prompt_prob=0.,repeat_cell_prompt_prob=0.)
        c=np.zeros(cfg.train_patch_size,np.int32);c[1:7,5:15,5:15]=1;c[1:7,20:28,20:28]=2
        sample=dict(image=np.zeros(c.shape+(1,),np.float32),nucleus_mask=c,cytoplasm_mask=c,
                    assignments={1:1,2:2},fg_coords=np.argwhere(c>0))
        g=SimpleNamespace(config=cfg,subset='train',seed=42,epoch=0,data_pool=[sample],augmenter=None,
            _nuc_grid=cfg.nucleus_grid,_cyto_grid=cfg.cytoplasm_grid,_nuc_rays=cfg.rays_nucleus,
            _cyto_rays=cfg.rays_cytoplasm,_nuc_aniso=cfg.anisotropy,_cyto_aniso=cfg.anisotropy)
        x,y=CytoNucsDataGenerator.__getitem__(g,0)
        self.assertTrue(y['cytoplasm_z_prompt_pair'][0])
        np.testing.assert_array_equal(y['cytoplasm_target'][0],y['cytoplasm_target'][1])
        centers=[np.unravel_index(x[0,...,k].argmax(),c.shape) for k in (2,4)]
        self.assertGreater(centers[1][0],centers[0][0])
        for center in centers:self.assertEqual(y['cytoplasm_target'][0][center],1)
        cfg.cell_z_pair_prob=0
        x0,y0=CytoNucsDataGenerator.__getitem__(g,0)
        np.testing.assert_array_equal(x[...,1::2],x0[...,1::2])
        np.testing.assert_array_equal(y['nucleus_target'],y0['nucleus_target'])

    def test_detector_replacement_preserves_z_pair(self):
        cfg=config(predicted_prompt_fraction=1.)
        x,y=batch(cfg);y['cytoplasm_z_prompt_pair']=tf.constant([True])
        pred={h+'_prob':tf.ones((1,8,16,16,1)) for h in ('nucleus','cytoplasm')}
        pred.update({h+'_dist':tf.ones((1,8,16,16,12))*4 for h in ('nucleus','cytoplasm')})
        stub=SimpleNamespace(config=cfg,prompt_detector=lambda x,training:pred,
                             optimizer=SimpleNamespace(iterations=tf.Variable(0,dtype=tf.int64)))
        out=CytoNucsTrainer._predicted_prompts(stub,x,y)
        np.testing.assert_array_equal(out[...,2::2],x[...,2::2])
        self.assertFalse(np.array_equal(out[...,1],x[...,1]))

    def test_slice_dice_restores_thin_ends_without_filling_gaps(self):
        loss=CytoNucsLoss(config(cell_z_dice_weight=.1))
        t=np.zeros((1,5,10,10,1),np.float32);t[:,0,0,0]=1;t[:,1,:,:]=1;t[:,4,0,0]=1
        p=tf.Variable(t*.8)
        with tf.GradientTape() as tape:v=loss._cell_z_dice(p,t)
        grad=tape.gradient(v,p).numpy()
        self.assertLess(grad[0,4,0,0,0],0)
        np.testing.assert_array_equal(grad[:,2:4],0)
        self.assertEqual(float(loss._cell_z_dice(tf.constant(t),t)),0.)
        self.assertEqual(float(loss._cell_z_dice(p,np.zeros_like(t))),0.)
        # Fully missing a tiny end plane is costly despite its small volume.
        q=t.copy();q[:,4]=0
        self.assertGreater(float(loss._cell_z_dice(q,t)),float(loss._refinement_dice_loss(q,t))*10)

    def test_preset_loading(self):
        raw=json.loads((ROOT/'config_cytonucs_v2_4.json').read_text())
        self.assertTrue(raw['training']['resume_from_checkpoint'].endswith('weights_epoch_080.weights.h5'))
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            raw['training']['resume_from_checkpoint']=None;raw['output']['basedir']=tmp
            p=Path(tmp)/'c.json';p.write_text(json.dumps(raw))
            cfg=CytoNucsConfig.from_json(p)
            for section,settings in cfg.FIXES_SETTINGS.items():
                for key in settings:self.assertEqual(getattr(cfg,key),raw[section][key])
        for kw in (dict(cell_z_pair_prob=2),dict(cell_z_dice_weight=-1),dict(cell_z_pair_prob=1,prompts_per_crop=1)):
            with self.assertRaises(ValueError):config(**kw)

    def test_real_training_step_with_protected_pair_and_slice_loss(self):
        cfg=config(cell_z_pair_prob=.5,cell_z_dice_weight=.1,refinement_depth=1,
                   predicted_prompt_fraction=1.,predicted_prompt_start_epoch=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            model=build_cytonucs_model(cfg,'z_training',tmp)
            trainer=CytoNucsTrainer(model,cfg,CytoNucsLoss(cfg),0.)
            trainer.current_epoch_var.assign(1)
            x,y=batch(cfg);y['cytoplasm_z_prompt_pair']=tf.constant([True])
            step=trainer.train_step(x,y)
            self.assertTrue(np.isfinite(float(step[0])))
            self.assertGreater(float(step[-1]['cell_z_dice']),0)
            self.assertEqual(int(trainer.optimizer.iterations.numpy()),1)


if __name__=='__main__':unittest.main()
