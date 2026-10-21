"""Training prompts anywhere inside the object (augmentation.prompt_uniform_prob), the Polar IoU ray loss
(loss_params.dist_polar_iou_weight) and the IoU-aware probability target (loss_params.prob_iou_aware)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_levels import dataset, quiet, BOTH
from cytonucs_data_generator import CytoNucsDataGenerator
from cytonucs_losses import CytoNucsLoss


class PromptPlacementTests(unittest.TestCase):
    def offsets(self, uniform):
        """Distance of every nuclear training prompt's peak from its object's centroid, in voxels."""
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            dataset(Path(tmp), BOTH)
            cfg = quiet(config, train_patch_size=(8, 64, 64), prompt_uniform_prob=uniform, negative_prompt_prob=0.)
            generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')
            distances = []
            for index in range(8):
                x, y = generator[index]
                crops, channels = x.shape[0], cfg.n_channel_in
                for k in range(cfg.prompts_per_crop):
                    for b in range(crops):
                        target = y['nucleus_target'][k * crops + b] > 0
                        if not target.any():
                            continue
                        peak = np.unravel_index(np.argmax(x[b, ..., channels + 2 * k]), target.shape)
                        self.assertTrue(target[peak], 'the prompt lies inside its object')
                        distances.append(np.linalg.norm(np.array(peak) - np.argwhere(target).mean(0)))
            return np.array(distances)

    def test_uniform_prompts_lie_inside_and_spread_over_the_object(self):
        centred, uniform = self.offsets(0.), self.offsets(1.)
        self.assertGreater(len(uniform), 10)
        self.assertGreater(uniform.mean(), 1.5 * centred.mean())


class PolarIoUTests(unittest.TestCase):
    def loss(self, weight, pred_scale):
        cfg = quiet(config, dist_polar_iou_weight=weight)
        loss = CytoNucsLoss(cfg)
        prob = np.zeros((1, 4, 8, 8, 1), np.float32); prob[:, 1:3, 2:6, 2:6] = 1.
        rays = np.random.default_rng(0).uniform(2, 6, (1, 4, 8, 8, 12)).astype(np.float32)
        pred = tf.Variable(rays * pred_scale)
        with tf.GradientTape() as tape:
            out = loss._stardist_loss(tf.constant(prob), tf.constant(prob), tf.constant(rays), pred, name='object')
        return float(out['distance']), tape.gradient(out['distance'], pred)

    def test_zero_at_the_target_and_larger_for_wrong_rays(self):
        exact, _ = self.loss(1., 1.)
        self.assertAlmostEqual(exact, self.loss(0., 1.)[0], places=6)       # adds nothing at the target
        plain, _ = self.loss(0., .5)
        polar, gradient = self.loss(1., .5)
        self.assertGreater(polar, plain)
        self.assertTrue(np.all(np.isfinite(gradient.numpy())))
        self.assertLess(float(tf.reduce_sum(gradient)), 0.)        # too short rays: the loss pushes them out


class IoUAwareProbabilityTests(unittest.TestCase):
    def run_loss(self, power, pred_scale, ramp):
        cfg = quiet(config, prob_iou_aware=power)
        loss = CytoNucsLoss(cfg)
        prob = np.zeros((1, 4, 8, 8, 1), np.float32); prob[:, 1:3, 2:6, 2:6] = 1.
        rays = np.random.default_rng(0).uniform(2, 6, (1, 4, 8, 8, 12)).astype(np.float32)
        pred = tf.Variable(np.full_like(prob, .7))
        with tf.GradientTape() as tape:
            out = loss._stardist_loss(tf.constant(prob), pred, tf.constant(rays), tf.constant(rays * pred_scale),
                                      name='object', quality_ramp=tf.constant(ramp))
        quality = loss._ray_quality(tf.constant(rays), tf.constant(rays * pred_scale), tf.constant(prob[..., 0]),
                                    power, ramp)
        return float(out['bce']), tape.gradient(out['bce'], pred).numpy(), quality.numpy()

    def test_target_follows_the_polygon_fit(self):
        off, grad_off, _ = self.run_loss(0., .5, 1.)
        exact, _, quality = self.run_loss(1., 1., 1.)
        self.assertAlmostEqual(exact, self.run_loss(0., 1., 1.)[0], places=6)      # rays fit: target unchanged
        self.assertTrue(np.allclose(quality, 1.))
        poor, grad_poor, quality = self.run_loss(1., .5, 1.)
        self.assertTrue(np.all((quality > .45) & (quality < .6)))                # half-length rays: IoU about 0.5
        inside = np.zeros((1, 4, 8, 8, 1), bool); inside[:, 1:3, 2:6, 2:6] = True
        self.assertTrue(np.all(grad_poor[inside] > grad_off[inside]))            # a lower target pulls p=0.7 down
        self.assertAlmostEqual(self.run_loss(1., .5, 0.)[0], off, places=6)        # before the ramp: no change


if __name__ == '__main__':
    unittest.main()
