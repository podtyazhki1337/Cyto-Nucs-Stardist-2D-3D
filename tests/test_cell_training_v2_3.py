import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from cytonucs_config import CytoNucsConfig
from cytonucs_losses import CytoNucsLoss
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_data_generator import nearest_cell_neighbors, CytoNucsDataGenerator


class CellTrainingTests(unittest.TestCase):
    def targets(self):
        t=np.zeros((2,2,2,6,1),np.float32)
        t[0,:,:,:2]=1;t[1,:,:,4:]=1
        return t

    def test_gradient_separates_identical_predictions(self):
        loss=CytoNucsLoss(config(cell_prompt_separation_weight=.1))
        t=self.targets();z=tf.Variable(tf.zeros(t.shape[:-1]+(3,)))
        with tf.GradientTape() as tape:v=loss._cell_prompt_separation(z,t)
        g=tape.gradient(v,z).numpy()
        self.assertAlmostEqual(float(v),np.log(2),places=6)
        self.assertLess(g[0,0,0,0,1],0);self.assertGreater(g[1,0,0,0,1],0)
        self.assertGreater(g[0,0,0,5,1],0);self.assertLess(g[1,0,0,5,1],0)
        z.assign_sub(10*g)
        self.assertLess(float(loss._cell_prompt_separation(z,t)),float(v))
        np.testing.assert_array_equal(g[:,:,:,2:4],0)

    def test_empty_repeated_and_batch_ownership(self):
        loss=CytoNucsLoss(config())
        t=self.targets();z=tf.zeros(t.shape[:-1]+(3,))
        for invalid in (np.zeros_like(t),np.repeat(t[:1],2,axis=0)):
            self.assertEqual(float(loss._cell_prompt_separation(z,invalid)),0)
        # K-major: image0 has two different cells, image1 repeats one cell.
        tb=np.stack([t[0],t[0],t[1],t[0]])
        zb=tf.zeros(tb.shape[:-1]+(3,))
        self.assertAlmostEqual(float(tf.function(loss._cell_prompt_separation)(zb,tb)),np.log(2),places=6)

    def test_neighbor_selection_uses_physical_distance(self):
        lab=np.zeros((5,9,9),np.int32)
        lab[1,1,1]=42;lab[2,1,1]=7;lab[1,1,4]=99
        self.assertEqual(nearest_cell_neighbors(lab,42,(5,1,1)),[99,7])
        self.assertEqual(nearest_cell_neighbors(lab,42,(1,1,1)),[7,99])
        self.assertEqual(nearest_cell_neighbors(lab,123,(1,1,1)),[])

    def test_json_settings_and_resume(self):
        raw=json.loads((ROOT/'config_cytonucs_v2_3.json').read_text())
        self.assertEqual(raw['mode'],'train')
        self.assertTrue(raw['training']['resume_from_checkpoint'].endswith('weights_epoch_080.weights.h5'))
        self.assertTrue(raw['training']['reset_best_on_resume'])
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            raw['training']['resume_from_checkpoint']=None;raw['output']['basedir']=tmp
            p=Path(tmp)/'config.json';p.write_text(json.dumps(raw))
            cfg=CytoNucsConfig.from_json(p)
            for section,settings in cfg.FIXES_SETTINGS.items():
                for key in settings:self.assertEqual(getattr(cfg,key),raw[section][key])
            self.assertEqual(cfg.cell_prompt_separation_weight,.1)
            self.assertEqual(cfg.neighbor_cell_prompt_prob,1.)
        for kw in (dict(cell_prompt_separation_weight=-1),dict(neighbor_cell_prompt_prob=1.1),
                   dict(cell_prompt_separation_weight=.1,prompts_per_crop=1)):
            with self.assertRaises(ValueError):config(**kw)

    def test_generator_outputs_neighbor_targets_and_repeat_control(self):
        from types import SimpleNamespace
        cfg=config(neighbor_cell_prompt_prob=1.,repeat_cell_prompt_prob=0.,
                   negative_prompt_prob=0.,prompt_center_jitter=0.,prompt_sigma_jitter=0.)
        cells=np.zeros(cfg.train_patch_size,np.int32)
        cells[1:7,2:8,2:8]=1;cells[1:7,10:16,2:8]=2;cells[1:7,24:30,24:30]=3
        sample=dict(image=np.zeros(cells.shape+(1,),np.float32),nucleus_mask=cells,
                    cytoplasm_mask=cells,assignments={1:1,2:2,3:3},fg_coords=np.argwhere(cells>0))
        g=SimpleNamespace(config=cfg,subset='train',seed=42,epoch=0,data_pool=[sample],
            augmenter=None,_nuc_grid=cfg.nucleus_grid,_cyto_grid=cfg.cytoplasm_grid,
            _nuc_rays=cfg.rays_nucleus,_cyto_rays=cfg.rays_cytoplasm,
            _nuc_aniso=cfg.anisotropy,_cyto_aniso=cfg.anisotropy)
        for idx in range(6):
            _,y=CytoNucsDataGenerator.__getitem__(g,idx)
            lab=y['cytoplasm_instances'][0]
            ids=[int(np.unique(lab[t>.5])[0]) for t in y['cytoplasm_target']]
            self.assertEqual(ids[1],nearest_cell_neighbors(lab,ids[0],cfg.anisotropy)[0])
        cfg.repeat_cell_prompt_prob=1.
        _,y=CytoNucsDataGenerator.__getitem__(g,0)
        np.testing.assert_array_equal(y['cytoplasm_target'][0],y['cytoplasm_target'][1])

    def test_real_model_total_loss_and_refiner_gradient(self):
        cfg=config(cell_prompt_separation_weight=.1,refinement_depth=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            model=build_cytonucs_model(cfg,'separation',tmp)
            x,y=batch(cfg)
            target=np.array(y['cytoplasm_target']);target[1]=np.roll(target[0],12,axis=-1)
            target[1][target[0]>0]=0
            y['cytoplasm_target']=tf.constant(target)
            xa=x.numpy();xa[...,4]=np.roll(xa[...,2],12,axis=-1)
            loss=CytoNucsLoss(cfg)
            with tf.GradientTape() as tape:
                pred=model.keras_model(xa,training=True)
                total=loss(y,pred)
                term=loss.last_components['cell_prompt_separation']
            gradients=tape.gradient(term,model.keras_model.trainable_variables)
            selected=[g for g,v in zip(gradients,model.keras_model.trainable_variables)
                      if g is not None and 'cyto_refine' in v.path]
            self.assertGreater(float(tf.linalg.global_norm(selected)),0)
            self.assertTrue(np.isfinite(float(total)))
            cfg.cell_prompt_separation_weight=0
            baseline=loss(y,pred)
            self.assertAlmostEqual(float(total-baseline),float(term),places=4)

    def test_resume_keeps_weights_and_adam(self):
        cfg=config(refinement_depth=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            old=build_cytonucs_model(cfg,'old',tmp);x,y=batch(cfg)
            trainer=CytoNucsTrainer(old,cfg,CytoNucsLoss(cfg),0.)
            trainer.train_step(x,y);trainer.state_epoch.assign(80)
            weights=Path(tmp)/'weights_epoch_080.weights.h5';trainer.save_checkpoint(weights)
            nc=config(refinement_depth=1,cell_prompt_separation_weight=.1,
                      neighbor_cell_prompt_prob=1.,reset_best_on_resume=True)
            new=build_cytonucs_model(nc,'new',tmp);new.load_weights(weights)
            nt=CytoNucsTrainer(new,nc,CytoNucsLoss(nc),0.)
            self.assertTrue(nt.restore_training_state(weights))
            self.assertEqual(int(nt.state_epoch.numpy()),80)
            for a,b in zip(old.keras_model.weights,new.keras_model.weights):np.testing.assert_array_equal(a.numpy(),b.numpy())
            for a,b in zip(trainer.optimizer.variables,nt.optimizer.variables):np.testing.assert_array_equal(a.numpy(),b.numpy())
            self.assertTrue(np.isfinite(float(nt.train_step(x,y)[0])))


if __name__=='__main__':unittest.main()
