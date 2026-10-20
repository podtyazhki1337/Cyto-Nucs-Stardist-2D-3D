import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
import tensorflow as tf
from test_fixes import ROOT, config
from cytonucs_runtime import read_model_image, write_instance_mask, aggregate_joint_counts, restored_best_metric
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_losses import CytoNucsLoss


class FullPipelineRecheckTests(unittest.TestCase):
    def test_reset_selection_boundary_survives_checkpoint_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            cfg=config(reset_best_on_resume=True)
            model=build_cytonucs_model(cfg,name='selection',basedir=tmp)
            trainer=CytoNucsTrainer(model,cfg,CytoNucsLoss(cfg),0)
            trainer.state_epoch.assign(420);trainer.state_best_full.assign(.9)
            old=Path(tmp)/'old.weights.h5';trainer.save_checkpoint(old)
            trainer.restore_training_state(old)
            self.assertEqual(int(trainer.state_selection_start.numpy()),420)
            self.assertEqual(float(trainer.state_best_full.numpy()),float('-inf'))
            trainer.state_epoch.assign(425);trainer.state_best_full.assign(.6)
            new=Path(tmp)/'new.weights.h5';trainer.save_checkpoint(new)
            cfg.reset_best_on_resume=False
            trainer.state_selection_start.assign(0)
            trainer.restore_training_state(new)
            history={'fullval_epoch':[400,425], 'fullval_metric':[.9,.6]}
            best=restored_best_metric(history,float(trainer.state_best_full.numpy()),
                                      int(trainer.state_selection_start.numpy()))
            self.assertAlmostEqual(best,.6,places=6)
            self.assertEqual(restored_best_metric(history,.6),.9)

    def test_legacy_state_without_selection_boundary_restores(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            cfg=config();model=build_cytonucs_model(cfg,name='legacy',basedir=tmp)
            trainer=CytoNucsTrainer(model,cfg,CytoNucsLoss(cfg),0)
            path=Path(tmp)/'legacy.weights.h5'
            old=tf.train.Checkpoint(model=model.keras_model,optimizer=trainer.optimizer,
                                   epoch=tf.Variable(123,dtype=tf.int64),
                                   best_val=tf.Variable(.2),best_full=tf.Variable(.7))
            old.write(str(path)+'.state')
            self.assertTrue(trainer.restore_training_state(path))
            self.assertEqual(int(trainer.state_epoch.numpy()),123)
            self.assertEqual(int(trainer.state_selection_start.numpy()),0)

    def test_tiff_spatial_axes_and_channels(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            expected=np.arange(2*5*7,dtype=np.uint16).reshape(2,5,7)
            path=Path(tmp)/'yxz.tif'
            tifffile.imwrite(path,expected.transpose(1,2,0),photometric='minisblack',metadata={'axes':'YXZ'})
            np.testing.assert_array_equal(read_model_image(path,3,1),expected)
            path=Path(tmp)/'cyxz.tif'
            rgb=np.stack([expected,expected+100,expected+200],axis=-1)
            tifffile.imwrite(path,rgb.transpose(3,1,2,0),photometric='minisblack',metadata={'axes':'CYXZ'})
            np.testing.assert_array_equal(read_model_image(path,3,3),rgb)

    def test_instance_tiff_roundtrip_preserves_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            for shape in ((2,3),(1,2,3)):
                for maximum,dtype in ((65535,np.uint16),(70000,np.uint32)):
                    labels=np.array([0,1,2,65534,65535,maximum],np.int64).reshape(shape)
                    path=Path(tmp)/'mask.tif';write_instance_mask(path,labels)
                    actual=tifffile.imread(path)
                    self.assertEqual(actual.dtype,dtype);np.testing.assert_array_equal(actual,labels)
            for labels in (np.array([[-1]],np.int32),np.zeros((2,2),np.float32)):
                with self.assertRaises(ValueError):write_instance_mask(Path(tmp)/'bad.tif',labels)

    def test_global_jtpr_not_mean_of_volumes(self):
        summary=aggregate_joint_counts([{'joint_tp':1,'total_pairs':1}, {'joint_tp':0,'total_pairs':99}])
        self.assertEqual(summary,dict(jtpr_global=.01,joint_tp=1,total_pairs=100))
        self.assertEqual(aggregate_joint_counts([])['jtpr_global'],0.)


if __name__=='__main__':unittest.main()
