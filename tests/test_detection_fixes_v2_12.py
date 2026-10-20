"""v2.12: shared peak rule, crop-truncation aware supervision, calibrated quality gates."""
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
from cytonucs_config import CytoNucsConfig
from cytonucs_geometry import proposal_peak_settings
from cytonucs_losses import CytoNucsLoss, quality_supervision_weights, without_truncated_objects
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_runtime import calibrate_quality_gates, read_quality_gates
import cytonucs_inference as inference


class PeakRuleTests(unittest.TestCase):
    def test_auto_window_reproduces_measured_geometry_and_explicit_wins(self):
        cfg = config(proposal_mode='peaks', median_max_dist_nucleus=54., median_max_dist_cytoplasm=99.,
                     rays_anisotropy_nucleus=(19, 1, 1), rays_anisotropy_cytoplasm=(28, 1, 1))
        # Windows logged by v2.8 full validation for the same geometry.
        self.assertEqual(proposal_peak_settings(cfg, 'nucleus', (16, 256, 256)), ([3, 5, 5], 2))
        self.assertEqual(proposal_peak_settings(cfg, 'cytoplasm', (16, 256, 256)), ([3, 9, 9], 2))
        cfg.proposal_peak_window_nucleus = [3, 3, 3]; cfg.proposal_border_nucleus = 0
        self.assertEqual(proposal_peak_settings(cfg, 'nucleus', (16, 256, 256)), ([3, 3, 3], 0))

    def test_trainer_peak_sampler_uses_inference_auto_window(self):
        cfg = config(prompts_per_crop=3, detector_prompt_slots=1, predicted_prompt_fraction=0.,
                     predicted_prompt_sample_peaks=True, proposal_mode='peaks', median_max_dist_nucleus=60.)
        window, border = proposal_peak_settings(cfg, 'nucleus', (8, 16, 16))
        self.assertEqual((window, border), ([3, 5, 5], 2))
        x, y = batch(cfg)
        target = np.zeros((3,) + tuple(cfg.train_patch_size), np.float32); target[:, 3, 14, 18] = 1
        y['nucleus_target'] = tf.constant(target)
        prob = np.zeros((1, 8, 16, 16, 1), np.float32); prob[0, 3, 7, 7] = .9; prob[0, 3, 7, 9] = .8
        pred = {h + '_prob': tf.constant(prob) for h in ('nucleus', 'cytoplasm')}
        pred.update({h + '_dist': tf.ones((1, 8, 16, 16, 12)) * 4 for h in ('nucleus', 'cytoplasm')})
        stub = SimpleNamespace(config=cfg, prompt_detector=lambda x, training: pred,
                               optimizer=SimpleNamespace(iterations=tf.Variable(0, dtype=tf.int64)))
        # Auto window 5 in X: the weaker owned peak is not a local maximum, prompt kept.
        np.testing.assert_array_equal(CytoNucsTrainer._predicted_prompts(stub, x, y)[..., 5], x[..., 5])
        cfg.proposal_peak_window_nucleus = [3, 3, 3]
        self.assertFalse(np.array_equal(CytoNucsTrainer._predicted_prompts(stub, x, y)[..., 5], x[..., 5]))

    def test_inference_uses_same_rule(self):
        cfg = config(proposal_mode='peaks', median_max_dist_nucleus=60.)
        prob = np.zeros((8, 16, 16), np.float32); prob[3, 7, 7] = .9; prob[3, 7, 9] = .8
        stub = SimpleNamespace(config=cfg, ndim=3, rays_nucleus=cfg.rays_nucleus,
                               rays_cytoplasm=cfg.rays_cytoplasm, _current_type='nucleus')
        inference.CytoNucsPredictor._stardist_postprocess(stub, prob, np.ones(prob.shape + (12,), np.float32), .5, .3)
        self.assertEqual(np.asarray(stub._last_coords).astype(int).tolist(), [[3, 14, 14]])


class TruncationTests(unittest.TestCase):
    def test_quality_supervision_ignores_only_artificial_faces(self):
        t = np.zeros((2, 8, 12, 12), np.float32)
        t[0, :, 3:9, 3:9] = 1          # full Z: touches both Z faces
        t[1, 2:6, 3:9, 0:4] = 1        # touches the low X face
        legacy, _ = quality_supervision_weights(tf.constant(t), 3)
        np.testing.assert_array_equal(legacy, [0, 0])
        flags = tf.constant([[0, 1, 1], [0, 1, 1]], tf.float32)
        w, _ = quality_supervision_weights(tf.constant(t), 3, flags)
        np.testing.assert_array_equal(w, [1, 0])
        w, _ = quality_supervision_weights(tf.constant(t), 3, tf.zeros((2, 3)))
        np.testing.assert_array_equal(w, [1, 1])

    def test_objects_cut_by_crop_leave_object_mean_only(self):
        full = np.zeros((1, 8, 16, 16), np.int32)
        full[0, 0:3, 4:8, 4:8] = 1      # touches Z face (volume face)
        full[0, 3:6, 4:8, 15] = 2       # only on last X row, skipped by the grid
        full[0, 3:6, 10:14, 6:10] = 3   # interior
        grid = full[:, :, ::2, ::2]
        out = without_truncated_objects(full, grid, tf.constant([[0., 1., 1.]]), 3).numpy()
        self.assertEqual(set(np.unique(out)) - {0}, {1, 3})
        out = without_truncated_objects(full, grid, tf.constant([[1., 1., 1.]]), 3).numpy()
        self.assertEqual(set(np.unique(out)) - {0}, {3})

    def test_loss_counts_only_complete_objects_when_flags_present(self):
        cfg = config(nucleus_prob_object_weight=.25)
        model = build_cytonucs_model(cfg); x, y = batch(cfg)
        labels = y['nucleus_instances'].numpy().copy(); labels[0, 2:6, 12:20, 0:3] = 2
        y['nucleus_instances'] = tf.constant(labels)
        pred = model.keras_model(x)
        counts = []
        for flags in (None, [[0., 0., 0.]], [[0., 1., 1.]]):
            yy = dict(y)
            if flags is not None:
                yy['crop_truncation_axes'] = tf.constant(flags)
            loss = CytoNucsLoss(cfg); loss(yy, pred, current_epoch=451)
            counts.append(float(loss.last_components['nucleus_prob_sampled_object_count']))
        self.assertEqual(counts, [2., 2., 1.])

    def test_generator_flags_follow_source_volume(self):
        from test_pipeline_review import make_dataset
        from cytonucs_data_generator import CytoNucsDataGenerator
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp); make_dataset(root)
            for patch_size, expected in (((8, 32, 32), [0, 0, 0]), ((8, 16, 16), [0, 1, 1])):
                cfg = config(train_patch_size=patch_size, negative_prompt_prob=0.)
                _, y = CytoNucsDataGenerator(root, cfg, cfg.rays)[0]
                np.testing.assert_array_equal(y['crop_truncation_axes'], [expected])


class QualityGateTests(unittest.TestCase):
    def test_calibration_maximizes_checkpoint_metric(self):
        # v2.14: gate chosen by the checkpoint metric (see tests/test_calibration_v2_14.py).
        counts = {'nucleus': dict(tp=[2], fp=[2], fn=[2]), 'cytoplasm': dict(tp=[0], fp=[1], fn=[1])}
        gates, stats = calibrate_quality_gates({'nucleus': [(.8, .9, 1), (.55, .6, 0), (.3, 0., 0), (.6, .2, 0)],
                                                'cytoplasm': [(.2, .1, 0)]}, counts, [.5], 4, .7)
        self.assertEqual(gates, {'nucleus': .55, 'cytoplasm': 0.})  # removing the FP leaves AP at 0
        self.assertEqual((stats['nucleus']['fp_removed'], stats['nucleus']['tp_removed']), (1, 0))

    def test_config_accepts_auto_only_where_calibratable(self):
        good = dict(mask_quality_head=True, mask_assembly_nucleus='object_quality_containment',
                    mask_assembly_cytoplasm='object_quality', mask_containment_threshold=.95,
                    refinement_stage_audit=True, mask_quality_min='auto')
        self.assertEqual(config(**good).mask_quality_min, 'auto')
        for change in (dict(mask_quality_head=False), dict(mask_assembly_cytoplasm='object'),
                       dict(refinement_stage_audit=False), dict(mask_quality_min='AUTO'),
                       dict(mask_quality_min=1.5), dict(mask_quality_min=True)):
            with self.assertRaises(ValueError):
                config(**{**good, **change})
        self.assertEqual(config(**{**good, 'refinement_stage_audit': False, 'mode': 'predict'}).mask_quality_min, 'auto')

    def test_checkpoint_roundtrip_only_for_calibrated_epoch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = config(); model = build_cytonucs_model(cfg, basedir=tmp)
            trainer = CytoNucsTrainer(model, cfg, CytoNucsLoss(cfg), 0)
            model.pending_quality_gates = dict(epoch=455, nucleus=.31, cytoplasm=.12)
            trainer.state_epoch.assign(450); old = Path(tmp) / 'weights_epoch_450.weights.h5'
            trainer.save_checkpoint(old)
            trainer.state_epoch.assign(455); new = Path(tmp) / 'weights_epoch_455.weights.h5'
            trainer.save_checkpoint(new)
            self.assertIsNone(read_quality_gates(old))
            self.assertEqual(read_quality_gates(new), {'nucleus': .31, 'cytoplasm': .12})
            auto = config(mask_quality_head=True, mask_assembly_nucleus='object_quality',
                          mask_assembly_cytoplasm='object_quality', mode='predict', mask_quality_min='auto')
            restored = build_cytonucs_model(cfg, basedir=tmp); restored.load_weights(new)
            self.assertEqual(inference.CytoNucsPredictor(restored, auto).quality_gates,
                             {'nucleus': .31, 'cytoplasm': .12})
            self.assertIsNone(inference.CytoNucsPredictor(restored, cfg).quality_gates)

    def test_prompted_inference_applies_per_head_gate(self):
        cfg = config(nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1), refinement_patch_size=[0, 32, 32],
                     mask_quality_head=True, mask_assembly_nucleus='object_quality',
                     mask_assembly_cytoplasm='object_quality', mode='predict', mask_quality_min='auto')
        shape = (8, 32, 32); gt = np.zeros(shape, np.int32); gt[2:6, 10:20, 10:20] = 1
        image = (gt > 0).astype(np.float32)[..., None]; ctr = np.array([[4, 15, 15]])
        labels = np.zeros(shape, np.int32); labels[tuple(ctr[0])] = 1
        def logits(x):
            b = tf.cast(x[0][..., 0] > .5, tf.float32)
            return tf.stack([10 * (1 - b), 10 * b, tf.zeros_like(b)], -1)
        def joint(x):
            return logits(x), logits(x), tf.constant([[.5]]), tf.constant([[.9]])
        results = {}
        for gate in (.4, .6):
            model = SimpleNamespace()
            with patch.object(inference, '_refinement_runners', return_value=(lambda x: [x] * 7, joint)), \
                    patch.object(inference, '_cytoplasm_refinement_runner', return_value=None):
                n, _, _ = inference.predict_instances_prompted(
                    model, cfg, image, ctr, np.ones(1), nuc_dists=np.full((1, 12), 2.), nuc_labels=labels,
                    verbose=False, quality_gates={'nucleus': gate, 'cytoplasm': 0.})
            results[gate] = (int(n.max()), model.last_refinement_diagnostics['nucleus']['quality_rejected'])
        self.assertEqual(results, {.4: (1, 0), .6: (0, 1)})


class FullValidationTests(unittest.TestCase):
    def test_records_gates_interior_and_aligned_history(self):
        from cytonucs_trainer import DiceCallback
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = config(refinement_stage_audit=True, mask_quality_head=True, proposal_mode='peaks',
                         mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                         mask_quality_min='auto', prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            model = build_cytonucs_model(cfg, basedir=tmp)
            model.loaded_quality_gates = {'nucleus': .99, 'cytoplasm': .99}
            gt = np.zeros((8, 32, 32), np.int32); gt[2:6, 10:20, 10:20] = 1
            gen = SimpleNamespace(image_files=['a.tiff'], data_pool=[dict(
                image=(gt > 0).astype(np.float32)[..., None], nucleus_mask=gt, cytoplasm_mask=gt,
                assignments={1: 1}, filename='a.tiff')])
            history = {'fullval_epoch': [450]}
            callback = DiceCallback(model, gen, eval_every=1, save_best=False,
                                    prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            seen = []
            original = inference.predict_instances_prompted
            def spy(*args, **kwargs):
                seen.append(kwargs.get('quality_gates'))
                return original(*args, **kwargs)
            with patch.object(callback, '_save_epoch_visualization_full'), \
                    patch.object(inference, 'predict_instances_prompted', spy):
                callback.on_epoch_end(454, history)
            self.assertEqual(seen, [None])  # calibration never sees loaded gates
            self.assertEqual(history['fullval_epoch'], [450, 455])
            self.assertEqual(history['fullval_interior_total_pairs'], [None, 1])
            self.assertEqual(history['fullval_nucleus_quality_gate'], [None, 0.])
            self.assertEqual(len(history['fullval_ap_cytoplasm_quality_gated']), 2)
            self.assertEqual(model.pending_quality_gates, dict(epoch=455, nucleus=0., cytoplasm=0.))


class InteriorJtprTests(unittest.TestCase):
    def test_pairs_touching_xy_faces_are_excluded_from_diagnostic(self):
        n = np.zeros((4, 16, 16), np.int32); c = n.copy()
        n[1:3, 4:7, 4:7] = 1; c[0:4, 2:9, 2:9] = 1       # interior (Z faces allowed)
        n[1:3, 4:7, 12:15] = 2; c[1:3, 2:9, 10:16] = 2    # cell touches X face
        jtpr, details = inference.compute_jtpr(n, c, n, c, assignments_pred={1: 1, 2: 2},
                                               assignments_gt={1: 1, 2: 2})
        self.assertEqual(jtpr, 1.)
        self.assertEqual((details['interior_total_pairs'], details['interior_joint_tp']), (1, 1))


class PresetTests(unittest.TestCase):
    def test_v2_12_preset_changes_only_documented_values(self):
        old = json.loads((ROOT / 'config_cytonucs_v2_11.json').read_text())
        new = json.loads((ROOT / 'config_cytonucs_v2_12.json').read_text())
        changed = {(s, k) for s in new for k in (new[s] if isinstance(new[s], dict) else {})
                   if old[s].get(k) != new[s][k]}
        self.assertEqual(changed, {('experiment', 'name'), ('training', 'resume_from_checkpoint'),
                                   ('inference', 'checkpoint_path'), ('inference', 'mask_quality_min'),
                                   ('inference', 'proposal_peak_window_nucleus'),
                                   ('inference', 'proposal_peak_window_cytoplasm')})
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            new['training']['resume_from_checkpoint'] = None; new['inference']['checkpoint_path'] = None
            new['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(new))
            cfg = CytoNucsConfig.from_json(path)
            self.assertEqual(cfg.mask_quality_min, 'auto')
            self.assertIsNone(cfg.proposal_peak_window_nucleus)


if __name__ == '__main__':
    unittest.main()
