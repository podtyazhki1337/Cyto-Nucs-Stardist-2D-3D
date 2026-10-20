import unittest
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from test_mask_quality_target import gaussian
from cytonucs_losses import CytoNucsLoss, object_balanced_probability
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer

class ProbabilityBalanceTests(unittest.TestCase):
    def test_equal_object_gradient_mass_and_sparse_ids(self):
        labels=tf.constant([[[1,900000,900000,900000,900000,0]]],tf.int32)
        z=tf.Variable(tf.zeros((1,1,6)))
        target=tf.constant([[[.8,.8,.8,.8,.8,0.]]])
        with tf.GradientTape() as tape:
            bce=tf.nn.sigmoid_cross_entropy_with_logits(labels=target,logits=z)
            loss,monitor=object_balanced_probability(bce,tf.sigmoid(z),labels)
        g=tape.gradient(loss,z).numpy().reshape(-1)
        np.testing.assert_allclose(g[0],g[1:5].sum(),atol=1e-7)
        self.assertEqual(g[5],0);self.assertEqual(float(monitor['prob_sampled_object_count']),2)
        self.assertAlmostEqual(float(loss),float(np.log(2)),places=6)

    def test_ids_are_per_image_and_empty_images_do_not_dilute(self):
        labels=tf.constant([[[1,1,0]],[[1,0,0]],[[0,0,0]]])
        bce=tf.constant([[[1.,1.,99.]],[[3.,99.,99.]],[[99.,99.,99.]]])
        loss,mon=tf.function(object_balanced_probability)(bce,tf.ones_like(bce)*.45,labels)
        self.assertEqual(float(loss),2.);self.assertEqual(float(mon['prob_sampled_object_count']),2.)
        self.assertEqual(float(mon['prob_gt_max_above_040']),1.)
        self.assertEqual(float(mon['prob_gt_max_above_050']),0.)
        empty,_=object_balanced_probability(bce,tf.ones_like(bce),tf.zeros_like(labels))
        self.assertEqual(float(empty),0.)

    def test_soft_edt_optimum_preserved_and_background_keeps_gradient(self):
        labels=tf.constant([[[1,1,0]]]);target=tf.constant([[[.2,.8,0.]]])
        z=tf.Variable(tf.math.log(tf.constant([[[.2,.8,.5]]])/(1-tf.constant([[[.2,.8,.5]]]))))
        with tf.GradientTape() as tape:
            bce=tf.nn.sigmoid_cross_entropy_with_logits(labels=target,logits=z)
            obj,_=object_balanced_probability(bce,tf.sigmoid(z),labels)
            total=tf.reduce_mean(bce)+.25*obj
        g=tape.gradient(total,z).numpy().reshape(-1)
        np.testing.assert_allclose(g[:2],0,atol=1e-7);self.assertGreater(g[2],0)

    def test_total_loss_addition_distance_unchanged_and_weight_zero(self):
        cfg=config(nucleus_prob_object_weight=0.)
        model=build_cytonucs_model(cfg);x,y=batch(cfg);pred=model.keras_model(x)
        base=CytoNucsLoss(cfg);a=base(y,pred,current_epoch=421);ad=float(base.last_components['nucleus_distance'])
        cfg.nucleus_prob_object_weight=.25
        added=CytoNucsLoss(cfg);b=added(y,pred,current_epoch=421)
        obj=added.last_components['nucleus_bce_object_weighted']
        self.assertAlmostEqual(float(b-a),float(cfg.lambda_prob_nucleus*obj),places=5)
        self.assertEqual(float(added.last_components['nucleus_distance']),ad)
        for value in (-1,True,float('nan')):
            with self.assertRaises(ValueError):config(nucleus_prob_object_weight=value)

    def test_real_train_val_and_checkpoint_with_new_loss(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as tmp:
            cfg=config(nucleus_prob_object_weight=.25, prompts_per_crop=3, detector_prompt_slots=1,
                       predicted_prompt_fraction=.25, predicted_prompt_sample_peaks=True,
                       predicted_prompt_start_epoch=1, cell_predicted_nucleus_fraction=.5,
                       mask_quality_head=True, mask_quality_weight=.1, mask_quality_target='component',
                       mask_quality_ignore_boundary_targets=True, refinement_depth=1,
                       refinement_gradient_scale=.1, prompt_consistency_weight=.1,
                       cell_prompt_separation_weight=.1, cell_z_dice_weight=.1, nucleus_z_dice_weight=.1)
            model=build_cytonucs_model(cfg,basedir=tmp);trainer=CytoNucsTrainer(model,cfg,CytoNucsLoss(cfg),0)
            x,y=batch(cfg);x=x.numpy()
            for i in range(1,7):x[0,...,i]=gaussian((8,32,32),(3,16,16),(2,4,4))
            x=tf.constant(x);trainer.current_epoch_var.assign(421)
            for result in (trainer.train_step(x,y),trainer.val_step(x,y)):
                self.assertTrue(np.isfinite(float(result[0])))
                self.assertIn('nucleus_bce_object_weighted',result[-1])
            trainer.state_epoch.assign(420);path=Path(tmp)/'weights_epoch_420.weights.h5';trainer.save_checkpoint(path)
            cfg.nucleus_prob_object_weight=0
            m2=build_cytonucs_model(cfg,basedir=tmp);m2.load_weights(path)
            t2=CytoNucsTrainer(m2,cfg,CytoNucsLoss(cfg),0);self.assertTrue(t2.restore_training_state(path))
            for a,b in zip(trainer.optimizer.variables,t2.optimizer.variables):np.testing.assert_array_equal(a.numpy(),b.numpy())

if __name__=='__main__':unittest.main()
