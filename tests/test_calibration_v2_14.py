"""v2.14: metric-driven quality gates and mask thresholds, object-weighted monitoring."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import ROOT, config
from cytonucs_config import CytoNucsConfig
from cytonucs_model import build_cytonucs_model
from cytonucs_postprocess import mask_candidate, shrink_candidate
from cytonucs_runtime import (calibrate_quality_gates, select_mask_thresholds, read_mask_thresholds,
                              accuracy_profile)
from cytonucs_losses import CytoNucsLoss
from cytonucs_trainer import CytoNucsTrainer, accumulate_components, component_means
import cytonucs_inference as inference


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def auto_config(**kwargs):
    params = dict(mode='predict', mask_thresh_nucleus='auto', mask_thresh_cytoplasm='auto')
    params.update(kwargs)
    return quiet(config, **params)


class GateObjectiveTests(unittest.TestCase):
    def test_outlier_true_positive_does_not_block_useful_gate(self):
        # One matched mask with very low quality (outside any joint pair) and many low-quality FP.
        rows = [(.05, .7, 0)] + [(.1 + .01 * i, 0., 0) for i in range(20)] + [(.9, .8, 1)] * 10
        counts = {'nucleus': dict(tp=[11], fp=[20], fn=[5])}
        gates, stats = calibrate_quality_gates({'nucleus': rows}, counts, [.5], 16, .7)
        self.assertGreater(gates['nucleus'], .29)
        self.assertEqual((stats['nucleus']['fp_removed'], stats['nucleus']['tp_removed']), (20, 1))
        self.assertAlmostEqual(stats['nucleus']['ap'], accuracy_profile([10], [0], [6]).mean())
        # The same mask in a joint pair is kept when JTPR dominates the metric.
        rows[0] = (.05, .7, 1)
        gates, stats = calibrate_quality_gates({'nucleus': rows}, counts, [.5], 16, .999)
        self.assertEqual(stats['nucleus']['tp_removed'], 0)
        self.assertLess(gates['nucleus'], .06)

    def test_mask_threshold_selection_trades_ap_and_jtpr(self):
        base = {'nucleus': dict(threshold=.45, tp=[8], fp=[4], fn=[2], joint_tp=7)}
        better = {'nucleus': {.55: dict(tp=[9], fp=[2], fn=[1], joint_tp=7), .65: dict(tp=[9], fp=[1], fn=[1], joint_tp=4)}}
        chosen, stats = select_mask_thresholds(base, better, 10, .7)
        self.assertEqual(chosen, {'nucleus': .55})
        self.assertAlmostEqual(stats['nucleus']['jtpr'], .7)
        chosen, _ = select_mask_thresholds(base, {'nucleus': {.55: dict(tp=[8], fp=[4], fn=[2], joint_tp=6)}}, 10, .7)
        self.assertEqual(chosen, {'nucleus': .45})


class ShrinkTests(unittest.TestCase):
    def test_higher_threshold_keeps_prompt_component_inside_original(self):
        z, y, x = np.mgrid[:6, :20, :20]
        prob = np.exp(-(((y - 10) / 5.) ** 2 + ((x - 10) / 5.) ** 2 + ((z - 3) / 2.) ** 2)).astype(np.float32)
        prob[:, 1:3, 1:3] = .95                                          # separate bright blob
        base = mask_candidate(prob, prob > .3, (0, 5, 7))
        base.update(probability=base['score'], prompt_center=np.array([3., 15., 17.]), prompt_sigma=np.ones(3))
        small = shrink_candidate(base, .6)
        self.assertLess(small['area'], base['area'])
        big = np.zeros((6, 25, 27), bool); sl = tuple(slice(l, l + s) for l, s in zip(base['lo'], base['mask'].shape))
        big[sl] = base['mask']; part = np.zeros_like(big)
        part[tuple(slice(l, l + s) for l, s in zip(small['lo'], small['mask'].shape))] = small['mask']
        self.assertTrue((part <= big).all())
        self.assertFalse(part[:, 5 + 1:5 + 3, 7 + 1:7 + 3].any())       # blob is another component
        self.assertIsNone(shrink_candidate(base, 1.))                    # nothing exceeds the peak 1.0
        base.update(predicted_quality=.5, quality=.5, score=base['score'] * .5, score_scaled_by_quality=True)
        scaled = shrink_candidate(base, .6)
        np.testing.assert_allclose(scaled['score'], scaled['probability'] * .5)


class PromptedThresholdTests(unittest.TestCase):
    def run_prompted(self, cfg, **kwargs):
        shape = (8, 32, 32); gt = np.zeros(shape, np.int32); gt[2:6, 8:24, 8:24] = 1
        z, y, x = np.mgrid[:8, :32, :32]
        field = np.exp(-(((y - 16) / 7.) ** 2 + ((x - 16) / 7.) ** 2)).astype(np.float32)
        image = field[..., None]; ctr = np.array([[4, 16, 16]])
        labels = np.zeros(shape, np.int32); labels[tuple(ctr[0])] = 1
        def logits(t):
            b = t[0][..., 0]
            return tf.stack([tf.zeros_like(b), 8 * (b - .5), tf.fill(tf.shape(b), -20.)], -1)
        sink = {}
        with patch.object(inference, '_refinement_runners', return_value=(lambda t: [t] * 7, lambda t: (logits(t), logits(t)))), \
                patch.object(inference, '_cytoplasm_refinement_runner', return_value=None):
            n, c, pairs = inference.predict_instances_prompted(
                SimpleNamespace(), cfg, image, ctr, np.ones(1), nuc_dists=np.full((1, 12), 3.), nuc_labels=labels,
                verbose=False, mask_thresh=.3, candidate_sink=sink, **kwargs)
        return n, c, pairs, sink

    def test_auto_uses_calibrated_threshold_and_cached_rethreshold_matches(self):
        cfg = auto_config(nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1), refinement_patch_size=[0, 32, 32],
                          mask_assembly_nucleus='object', mask_assembly_cytoplasm='object')
        n_base, c_base, _, sink = self.run_prompted(cfg)
        n_cal, c_cal, _, _ = self.run_prompted(cfg, mask_thresholds={'nucleus': .7, 'cytoplasm': .3})
        self.assertLess((n_cal > 0).sum(), (n_base > 0).sum())
        n_re, c_re, pairs = inference.rethreshold_prediction(cfg, sink, 'nucleus', .7, n_base, c_base)
        np.testing.assert_array_equal(n_re > 0, n_cal > 0)             # cache reproduces a real forward
        np.testing.assert_array_equal(c_re, c_base)
        self.assertEqual(pairs, {1: 1})

    def test_config_and_predictor_resolution(self):
        cfg = auto_config()
        self.assertEqual(cfg.mask_threshold('nucleus'), cfg.mask_thresh)
        self.assertEqual(cfg.mask_threshold('nucleus', {'nucleus': .6}), .6)
        for bad in (dict(mask_thresh_nucleus='AUTO'), dict(mask_thresh_nucleus=1.2),
                    dict(refinement_mode='polyhedron', proposal_mode='stardist')):
            with self.assertRaises(ValueError):
                auto_config(**bad)
        model = SimpleNamespace(loaded_mask_thresholds={'nucleus': .6, 'cytoplasm': .5})
        predictor = inference.CytoNucsPredictor(model, cfg)
        self.assertEqual(predictor._active_mask_thresholds(), {'nucleus': .6, 'cytoplasm': .5})
        predictor.calibrating = True
        self.assertIsNone(predictor._active_mask_thresholds())


class CalibrationPipelineTests(unittest.TestCase):
    def test_full_validation_does_not_calibrate_mask_thresholds(self):
        from cytonucs_trainer import DiceCallback
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = quiet(config, refinement_stage_audit=True, mask_quality_head=True, proposal_mode='peaks',
                        mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                        mask_thresh_nucleus='auto', mask_thresh_cytoplasm='auto', mask_thresh_range=[.4, .6, .7],
                        prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            model = quiet(build_cytonucs_model, cfg, basedir=tmp)
            gt = np.zeros((8, 32, 32), np.int32); gt[2:6, 10:20, 10:20] = 1
            sample = dict(image=(gt > 0).astype(np.float32)[..., None], nucleus_mask=gt, cytoplasm_mask=gt,
                          assignments={1: 1}, filename='a.tiff')
            gen = SimpleNamespace(image_files=['a.tiff'], data_pool=[sample])
            history = {'fullval_epoch': [5]}
            callback = DiceCallback(model, gen, eval_every=1, save_best=False, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            with patch.object(callback, '_save_epoch_visualization_full'),                     patch.object(inference, 'rethreshold_prediction', side_effect=AssertionError('no calibration during training')):
                quiet(callback.on_epoch_end, 9, history)
            self.assertFalse(any('mask_thresh' in k for k in history))
            self.assertIsNone(getattr(model, 'pending_mask_thresholds', None))

    def test_after_training_calibration_is_stored_with_checkpoint(self):
        from cytonucs_trainer import calibrate_mask_thresholds
        import train_cytonucs
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = auto_config(mask_thresh_range=[.4, .6, .7])
            model = quiet(build_cytonucs_model, cfg, basedir=tmp)
            trainer = CytoNucsTrainer(model, cfg, CytoNucsLoss(cfg), 0)
            trainer.state_epoch.assign(460); path = Path(tmp) / 'weights_best.weights.h5'
            quiet(trainer.save_checkpoint, path)
            gt = np.zeros((8, 32, 32), np.int32); gt[2:6, 10:20, 10:20] = 1
            sample = dict(image=(gt > 0).astype(np.float32)[..., None], nucleus_mask=gt, cytoplasm_mask=gt,
                          assignments={1: 1}, filename='a.tiff')
            predictor = inference.CytoNucsPredictor(model, cfg, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            calls = []
            original = inference.rethreshold_prediction
            def spy(*args):
                calls.append(args[3]); return original(*args)
            with patch.object(inference, 'rethreshold_prediction', spy):
                thresholds, report = quiet(calibrate_mask_thresholds, predictor, [sample])
            self.assertEqual(sorted(set(calls)), sorted({.4, .6, .7} - {cfg.mask_thresh}))   # both directions
            self.assertEqual(thresholds, {'nucleus': cfg.mask_thresh, 'cytoplasm': cfg.mask_thresh})
            self.assertFalse(predictor.calibrating)
            quiet(train_cytonucs.save_mask_calibration, path, thresholds, report, Path(tmp) / 'mask_calibration.json')
            self.assertEqual(read_mask_thresholds(path), thresholds)
            self.assertEqual(json.loads((Path(tmp) / 'mask_calibration.json').read_text())['checkpoint_epoch'], 460)
            restored = quiet(build_cytonucs_model, cfg, basedir=tmp); quiet(restored.load_weights, path)
            self.assertEqual(restored.loaded_mask_thresholds, thresholds)


class MonitoringTests(unittest.TestCase):
    def test_object_metrics_weighted_by_object_count(self):
        sums = {}
        for count, ratio, loss in ((0., 0., 1.), (4., .9, 3.)):
            accumulate_components(sums, {'nucleus_prob_sampled_object_count': count,
                                         'nucleus_prob_gt_max_above_040': ratio, 'nucleus_bce': loss})
        means = component_means(sums)
        self.assertAlmostEqual(means['nucleus_prob_gt_max_above_040'], .9)
        self.assertAlmostEqual(means['nucleus_bce'], 2.)
        self.assertAlmostEqual(means['nucleus_prob_sampled_object_count'], 2.)
        self.assertIsNone(component_means({'nucleus_prob_gt_max_mean': (0., 0.)})['nucleus_prob_gt_max_mean'])


class PresetTests(unittest.TestCase):
    def test_v2_14_changes_only_calibration_and_resume(self):
        old = json.loads((ROOT / 'config_cytonucs_v2_13.json').read_text())
        new = json.loads((ROOT / 'config_cytonucs_v2_14.json').read_text())
        changed = {(s, k) for s in new if isinstance(new[s], dict) for k in new[s] if old[s].get(k) != new[s][k]}
        self.assertEqual(changed, {('experiment', 'name'), ('training', 'resume_from_checkpoint'),
                                   ('training', 'reset_best_on_resume'), ('inference', 'checkpoint_path'),
                                   ('inference', 'mask_thresh_nucleus'), ('inference', 'mask_thresh_cytoplasm'),
                                   ('evaluation', 'mask_thresh_range')})
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            new['training']['resume_from_checkpoint'] = None; new['inference']['checkpoint_path'] = None
            new['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(new))
            cfg = quiet(CytoNucsConfig.from_json, path)
        self.assertEqual((cfg.mask_thresh_nucleus, cfg.mask_thresh_cytoplasm), ('auto', 'auto'))


if __name__ == '__main__':
    unittest.main()
