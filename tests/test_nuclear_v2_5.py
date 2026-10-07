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
from cytonucs_data_generator import CytoNucsDataGenerator
from cytonucs_losses import CytoNucsLoss
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_refinement_audit import RefinementStageAudit


class NuclearV25Tests(unittest.TestCase):
    def test_diagnostics_distinguish_peaks_duplicates_and_axial_fragments(self):
        gt=np.zeros((8,8,8),np.int32);gt[1:7,2:6,2:6]=42
        audit=RefinementStageAudit(gt)
        for z in (2,5):audit.record_roi((0,0,0),gt.shape,gt.shape,(z,3,3),(1,1,1))
        candidate=dict(lo=(0,0,0),mask=gt>0)
        audit.observe('before_nms',candidates=[candidate,candidate])
        labels=np.where(gt>0,1,0);labels[4:][gt[4:]>0]=2
        original=labels.copy();audit.observe('final',labels=labels)
        m=audit.report()['monitoring']
        self.assertEqual(m['gt_multiple_prompt_centers'],1)
        self.assertEqual(m['gt_multiple_good_candidates'],1)
        self.assertEqual(m['final_gt_multiple_parts'],1)
        self.assertEqual(m['final_z_disjoint_xy_bbox_pairs'],1)
        np.testing.assert_array_equal(labels,original)

    def test_neighbors_are_not_counted_as_one_fragmented_gt(self):
        gt=np.zeros((8,8,8),np.int32);gt[1:3,2:6,2:6]=5;gt[5:7,2:6,2:6]=99
        a=RefinementStageAudit(gt);a.observe('final',labels=gt)
        self.assertEqual(a.report()['monitoring']['final_gt_multiple_parts'],0)
        self.assertEqual(a.report()['monitoring']['final_z_disjoint_xy_bbox_pairs'],0)
        # A lateral division has two parts but is not counted as Z disjoint.
        gt[gt>0]=5;p=np.where(gt>0,1,0);p[:,:,4:][gt[:,:,4:]>0]=2
        a=RefinementStageAudit(gt);a.observe('final',labels=p)
        self.assertEqual(a.report()['monitoring']['final_gt_multiple_parts'],1)
        self.assertEqual(a.report()['monitoring']['final_z_disjoint_xy_bbox_pairs'],0)

    def test_nuclear_pair_is_one_nucleus_not_union_and_cells_unchanged(self):
        cfg=config(nucleus_z_pair_prob=1.,negative_prompt_prob=0.,repeat_cell_prompt_prob=0.)
        n=np.zeros(cfg.train_patch_size,np.int32);n[1:7,5:12,5:12]=1;n[1:7,18:25,18:25]=2
        c=np.zeros_like(n);c[1:7,3:28,3:28]=1
        sample=dict(image=np.zeros(n.shape+(1,),np.float32),nucleus_mask=n,cytoplasm_mask=c,
                    assignments={1:1,2:1},fg_coords=np.argwhere(c>0))
        g=SimpleNamespace(config=cfg,subset='train',seed=42,epoch=0,data_pool=[sample],augmenter=None,
            _nuc_grid=cfg.nucleus_grid,_cyto_grid=cfg.cytoplasm_grid,_nuc_rays=cfg.rays_nucleus,
            _cyto_rays=cfg.rays_cytoplasm,_nuc_aniso=cfg.anisotropy,_cyto_aniso=cfg.anisotropy)
        x,y=CytoNucsDataGenerator.__getitem__(g,0)
        self.assertTrue(y['nucleus_z_prompt_pair'][0])
        np.testing.assert_array_equal(y['nucleus_target'][0],y['nucleus_target'][1])
        self.assertEqual(len(np.unique(n[y['nucleus_target'][0]>.5])),1)
        centers=[np.unravel_index(x[0,...,k].argmax(),n.shape) for k in (1,3)]
        self.assertGreater(centers[1][0],centers[0][0])
        cfg.nucleus_z_pair_prob=0
        x0,y0=CytoNucsDataGenerator.__getitem__(g,0)
        np.testing.assert_array_equal(x[...,2::2],x0[...,2::2])
        np.testing.assert_array_equal(y['cytoplasm_target'],y0['cytoplasm_target'])

    def test_nuclear_detector_override_protected(self):
        cfg=config(predicted_prompt_fraction=1.)
        x,y=batch(cfg);y['nucleus_z_prompt_pair']=tf.constant([True])
        pred={h+'_prob':tf.ones((1,8,16,16,1)) for h in ('nucleus','cytoplasm')}
        pred.update({h+'_dist':tf.ones((1,8,16,16,12))*4 for h in ('nucleus','cytoplasm')})
        stub=SimpleNamespace(config=cfg,prompt_detector=lambda x,training:pred,
                             optimizer=SimpleNamespace(iterations=tf.Variable(0,dtype=tf.int64)))
        out=CytoNucsTrainer._predicted_prompts(stub,x,y)
        np.testing.assert_array_equal(out[...,1::2],x[...,1::2])
        self.assertFalse(np.array_equal(out[...,2],x[...,2]))

    def test_new_preset_contract(self):
        raw=json.loads((ROOT/'config_cytonucs_v2_5.json').read_text())
        self.assertTrue(raw['training']['resume_from_checkpoint'].endswith('weights_epoch_300.weights.h5'))
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            raw['training']['resume_from_checkpoint']=None;raw['output']['basedir']=tmp
            p=Path(tmp)/'config.json';p.write_text(json.dumps(raw))
            cfg=CytoNucsConfig.from_json(p)
            for section,settings in cfg.FIXES_SETTINGS.items():
                for key in settings:self.assertEqual(getattr(cfg,key),raw[section][key])
        for kw in (dict(nucleus_z_pair_prob=2),dict(nucleus_z_dice_weight=-1),dict(nucleus_z_pair_prob=1,prompts_per_crop=1)):
            with self.assertRaises(ValueError):config(**kw)

    def test_compiled_training_and_loss_contribution(self):
        cfg=config(nucleus_z_pair_prob=.5,nucleus_z_dice_weight=.1,refinement_depth=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            model=build_cytonucs_model(cfg,'nuclear',tmp);x,y=batch(cfg)
            y['nucleus_z_prompt_pair']=tf.constant([True])
            loss=CytoNucsLoss(cfg)
            with tf.GradientTape() as tape:
                pred=model.keras_model(x);total=loss(y,pred);term=loss.last_components['nucleus_z_dice']
            grads=tape.gradient(term,model.keras_model.trainable_variables)
            selected=[g for g,v in zip(grads,model.keras_model.trainable_variables) if g is not None and 'nuc_refine' in v.path]
            self.assertGreater(float(tf.linalg.global_norm(selected)),0)
            self.assertEqual(float(loss.last_components['nucleus_z_pair_fraction']),1)
            cfg.nucleus_z_dice_weight=0;baseline=loss(y,pred)
            self.assertAlmostEqual(float(total-baseline),float(term),places=4)
            cfg.nucleus_z_dice_weight=.1
            trainer=CytoNucsTrainer(model,cfg,loss,0.)
            result=trainer.train_step(x,y)
            self.assertGreater(float(result[-1]['nucleus_z_dice']),0)
            self.assertTrue(np.isfinite(float(result[0])))


if __name__=='__main__':unittest.main()
