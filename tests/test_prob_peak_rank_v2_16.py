"""v2.16: cell object-balanced probability BCE, per-object peak BCE, quality rank loss."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from test_mask_quality_target import gaussian
from cytonucs_config import CytoNucsConfig
from cytonucs_losses import CytoNucsLoss, object_peak_probability, quality_rank_loss
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer, accumulate_components, component_means

NEW = ('cytoplasm_prob_object_weight', 'prob_object_peak_weight', 'mask_quality_rank_weight')


def bce(t, p):
    return -(t * np.log(p) + (1 - t) * np.log(1 - p))


class ObjectPeakTests(unittest.TestCase):
    def test_value_gradient_only_at_each_objects_argmax(self):
        labels = tf.constant([[[1, 1, 1, 0, 2, 2]]], tf.int32)
        target = tf.constant([[[.5, 1., .5, 0., .6, .3]]])
        prob = tf.Variable([[[.2, .3, .25, .9, .4, .1]]])
        with tf.GradientTape() as tape:
            loss = object_peak_probability(prob, target, labels)
        expected = (bce(1., .3) + bce(.6, .4)) / 2
        self.assertAlmostEqual(float(loss), expected, places=5)
        g = tape.gradient(loss, prob).numpy().reshape(-1)
        self.assertLess(g[1], 0); self.assertLess(g[4], 0)          # raise each object's peak
        np.testing.assert_array_equal(g[[0, 2, 3, 5]], 0)             # other voxels and background untouched

    def test_soft_target_optimum_and_empty_and_per_image_ids(self):
        labels = tf.constant([[[1, 1, 0]], [[1, 0, 0]], [[0, 0, 0]]], tf.int32)
        target = tf.constant([[[.4, .8, 0.]], [[.7, 0., 0.]], [[0., 0., 0.]]])
        prob = tf.Variable(target.numpy().clip(1e-3, 1 - 1e-3))
        with tf.GradientTape() as tape:
            loss = tf.function(object_peak_probability)(prob, target, labels)
        np.testing.assert_allclose(tape.gradient(loss, prob).numpy(), 0, atol=1e-5)   # p = t is optimal
        self.assertAlmostEqual(float(loss), (bce(.8, .8) + bce(.7, .7)) / 2, places=4)  # 2 objects, not 3
        self.assertEqual(float(object_peak_probability(prob, target, tf.zeros_like(labels))), 0.)


class QualityRankTests(unittest.TestCase):
    def test_hinge_with_measured_gap_and_eligibility(self):
        target = tf.constant([.9, .5, .2]); eligible = tf.constant([1., 1., 0.])
        good = tf.constant([.95, .4, .9])                             # gap .55 >= .4, third ignored
        loss, accuracy, pairs = quality_rank_loss(good, target, eligible)
        self.assertEqual(float(loss), 0.); self.assertEqual(float(accuracy), 1.); self.assertEqual(float(pairs), 1.)
        q = tf.Variable([.5, .7, .0])
        with tf.GradientTape() as tape:
            loss, accuracy, _ = quality_rank_loss(q, target, eligible)
        self.assertAlmostEqual(float(loss), .4 - (.5 - .7), places=6)  # one ordered pair
        self.assertEqual(float(accuracy), 0.)
        g = tape.gradient(loss, q).numpy()
        self.assertLess(g[0], 0); self.assertGreater(g[1], 0); self.assertEqual(g[2], 0)

    def test_no_pairs_gives_zero(self):
        loss, accuracy, pairs = quality_rank_loss(tf.constant([.3, .3]), tf.constant([.5, .5]), tf.constant([1., 1.]))
        self.assertEqual(float(loss), 0.); self.assertEqual(float(accuracy), 0.); self.assertEqual(float(pairs), 0.)


class TotalLossTests(unittest.TestCase):
    def test_terms_add_exactly_and_legacy_unchanged(self):
        cfg = config(mask_quality_head=True, mask_quality_weight=.1, prompts_per_crop=2)
        model = build_cytonucs_model(cfg); x, y = batch(cfg); pred = model.keras_model(x)
        base = CytoNucsLoss(cfg); a = base(y, pred, current_epoch=501)
        for key in NEW:
            self.assertFalse(any(key.split('_weight')[0] in k for k in base.last_components), key)
        cfg.cytoplasm_prob_object_weight = .25; cfg.prob_object_peak_weight = .3; cfg.mask_quality_rank_weight = .2
        added = CytoNucsLoss(cfg); b = added(y, pred, current_epoch=501)
        c = added.last_components
        expected = (cfg.lambda_prob_cytoplasm * (c['cytoplasm_bce_object_weighted'] + c['cytoplasm_bce_object_peak_weighted'])
                    + cfg.lambda_prob_nucleus * c['nucleus_bce_object_peak_weighted']
                    + .2 * (c['nucleus_quality_rank_loss'] + c['cytoplasm_quality_rank_loss']))
        self.assertAlmostEqual(float(b - a), float(expected), places=5)
        self.assertEqual(float(c['cytoplasm_distance']), float(base.last_components['cytoplasm_distance']))
        for key in ('cytoplasm_prob_sampled_object_count', 'cytoplasm_prob_gt_max_mean', 'nucleus_quality_pair_accuracy',
                    'cytoplasm_quality_rank_pairs'):
            self.assertIn(key, c)

    def test_rank_uses_every_prompt_while_regression_keeps_its_exclusions(self):
        from unittest.mock import patch
        import cytonucs_losses
        cfg = config(mask_quality_head=True, mask_quality_weight=.1, mask_quality_rank_weight=.1,
                     mask_quality_ignore_boundary_targets=True, prompts_per_crop=2)
        model = build_cytonucs_model(cfg); x, y = batch(cfg); pred = model.keras_model(x)
        seen = []
        original = cytonucs_losses.quality_rank_loss
        def spy(q, t, w):
            seen.append(np.asarray(w)); return original(q, t, w)
        with patch.object(cytonucs_losses, 'quality_rank_loss', spy):
            loss = CytoNucsLoss(cfg); loss(y, pred, current_epoch=501)
        self.assertEqual(len(seen), 2)
        for w in seen:
            np.testing.assert_array_equal(w, np.ones_like(w))
        self.assertIn('nucleus_quality_supervised_fraction', loss.last_components)

    def test_rank_pairs_appear_when_mask_qualities_differ(self):
        cfg = config(mask_quality_head=True, mask_quality_weight=.1, mask_quality_rank_weight=.1, prompts_per_crop=2)
        model = build_cytonucs_model(cfg); x, y = batch(cfg); pred = dict(model.keras_model(x))
        loss = CytoNucsLoss(cfg)
        target = loss._subsample(tf.cast(y['nucleus_target'][..., None], tf.float32), cfg.nucleus_grid)
        self.assertGreater(float(tf.reduce_sum(target[0])), 0)
        perfect = tf.concat([target[:1], tf.zeros_like(target[1:])], 0)      # prompt 0 exact, others empty
        pred['nucleus_mask'] = perfect
        loss(y, pred, current_epoch=501)
        self.assertGreater(float(loss.last_components['nucleus_quality_rank_pairs']), 0)

    def test_object_weighted_history_uses_each_heads_count(self):
        sums = {}
        accumulate_components(sums, {'cytoplasm_prob_sampled_object_count': 3., 'cytoplasm_prob_gt_max_mean': .9,
                                     'nucleus_prob_sampled_object_count': 0., 'nucleus_prob_gt_max_mean': 0.})
        accumulate_components(sums, {'cytoplasm_prob_sampled_object_count': 1., 'cytoplasm_prob_gt_max_mean': .1,
                                     'nucleus_prob_sampled_object_count': 2., 'nucleus_prob_gt_max_mean': .5})
        means = component_means(sums)
        self.assertAlmostEqual(means['cytoplasm_prob_gt_max_mean'], (3 * .9 + .1) / 4)
        self.assertAlmostEqual(means['nucleus_prob_gt_max_mean'], .5)

    def test_real_train_val_and_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = config(nucleus_prob_object_weight=.25, cytoplasm_prob_object_weight=.25, prob_object_peak_weight=.25,
                         mask_quality_rank_weight=.1, prompts_per_crop=3, detector_prompt_slots=1,
                         predicted_prompt_fraction=.25, predicted_prompt_sample_peaks=True,
                         predicted_prompt_start_epoch=1, cell_predicted_nucleus_fraction=.5,
                         mask_quality_head=True, mask_quality_weight=.1, mask_quality_target='component',
                         mask_quality_ignore_boundary_targets=True, refinement_depth=1, refinement_gradient_scale=.1)
            model = build_cytonucs_model(cfg, basedir=tmp); trainer = CytoNucsTrainer(model, cfg, CytoNucsLoss(cfg), 0)
            x, y = batch(cfg); x = x.numpy()
            for i in range(1, 7):
                x[0, ..., i] = gaussian((8, 32, 32), (3, 16, 16), (2, 4, 4))
            x = tf.constant(x); trainer.current_epoch_var.assign(501)
            before = [v.numpy().copy() for v in model.keras_model.trainable_variables]
            for result in (trainer.train_step(x, y), trainer.val_step(x, y)):
                self.assertTrue(np.isfinite(float(result[0])))
                for key in ('cytoplasm_bce_object_peak_weighted', 'nucleus_bce_object_peak_weighted',
                            'cytoplasm_bce_object_weighted', 'cytoplasm_quality_rank_loss'):
                    self.assertIn(key, result[-1])
            self.assertTrue(any(np.any(a != b.numpy()) for a, b in zip(before, model.keras_model.trainable_variables)))
            trainer.state_epoch.assign(500); path = Path(tmp) / 'weights_epoch_500.weights.h5'; trainer.save_checkpoint(path)
            legacy = config(prompts_per_crop=3, mask_quality_head=True, refinement_depth=1, refinement_gradient_scale=.1)
            m2 = build_cytonucs_model(legacy, basedir=tmp); m2.load_weights(path)
            t2 = CytoNucsTrainer(m2, legacy, CytoNucsLoss(legacy), 0); self.assertTrue(t2.restore_training_state(path))
            for a, b in zip(trainer.optimizer.variables, t2.optimizer.variables):
                np.testing.assert_array_equal(a.numpy(), b.numpy())


class ConfigTests(unittest.TestCase):
    def test_json_contract_and_validation(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            expected = dict(cytoplasm_prob_object_weight=.25, prob_object_peak_weight=.25, mask_quality_rank_weight=.1) \
                if path.name in ('config_cytonucs_v2_16.json', 'config_cytonucs_v2_17.json', 'config_cytonucs_v3.json', 'config_cytonucs_v3_1.json', 'config_cytonucs_v3_2.json', 'config_cytonucs_v3_3.json', 'config_cytonucs_v3_4.json', 'config_cytonucs_v4.json') else dict.fromkeys(NEW, 0.)
            self.assertEqual({k: raw['loss_params'][k] for k in NEW}, expected, path.name)
        raw = json.loads((ROOT / 'config_cytonucs_v2_16.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['training']['resume_from_checkpoint'] = None; raw['inference']['checkpoint_path'] = None
            raw['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(raw))
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                preset = CytoNucsConfig.from_json(path)
        self.assertEqual({k: getattr(preset, k) for k in NEW},
                         dict(cytoplasm_prob_object_weight=.25, prob_object_peak_weight=.25, mask_quality_rank_weight=.1))
        self.assertTrue(preset.residual_proposals)
        for key in NEW:
            self.assertEqual(CytoNucsConfig.FIXES_SETTINGS['loss_params'][key], 0.)
            for bad in (-1, True, float('nan')):
                with self.assertRaises(ValueError):
                    config(**{key: bad})
        with self.assertRaises(ValueError):
            config(mask_quality_rank_weight=.1)                       # needs quality targets


if __name__ == '__main__':
    unittest.main()
