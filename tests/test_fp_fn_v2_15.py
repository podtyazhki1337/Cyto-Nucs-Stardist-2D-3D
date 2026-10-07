"""v2.15: residual prompts for window-suppressed peaks, joint quality/size gate, FN geometry audit."""
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import ROOT, config
from cytonucs_config import CytoNucsConfig
from cytonucs_postprocess import assemble_masks, drop_small_instances, mask_candidate, shrink_candidate
from cytonucs_refinement_audit import RefinementStageAudit
from cytonucs_runtime import (calibrate_quality_gates, accuracy_profile, checkpoint_objective_gain,
                              write_quality_gates, read_quality_gates)
import cytonucs_inference as inference


def box(shape, lo, hi, quality, residual=False):
    mask = np.zeros(shape, bool); mask[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    c = mask_candidate(np.where(mask, .9, 0.).astype(np.float32), mask, (0, 0, 0))
    c['predicted_quality'] = quality
    if residual:
        c['residual'] = True
    return c


class ResidualAssemblyTests(unittest.TestCase):
    def test_residual_candidates_fill_only_free_voxels_after_first_pass(self):
        shape = (4, 16, 16)
        primary = [box(shape, (0, 0, 0), (4, 8, 8), .5), box(shape, (0, 8, 8), (4, 16, 16), .4)]
        first, _ = assemble_masks(primary, shape, assembly_mode='object_quality')
        # A residual mask with higher quality overlapping the first object must not take its voxels.
        residual = [box(shape, (0, 4, 0), (4, 16, 8), .99, residual=True)]
        both, stats = assemble_masks(primary + residual, shape, assembly_mode='object_quality',
                                     min_remaining_fraction=.5)
        np.testing.assert_array_equal(both[first > 0], first[first > 0])
        self.assertEqual(stats['written'], 3)
        self.assertEqual(stats['written_residual'], 1)
        self.assertTrue(np.all(both[:, 8:16, 0:8] == 3))
        # Without the flag the same mask outranks and splits the first object.
        unflagged = dict(residual[0]); unflagged.pop('residual')
        other, _ = assemble_masks(primary + [unflagged], shape, assembly_mode='object_quality')
        self.assertFalse(np.array_equal(other[first > 0], first[first > 0]))

    def test_flag_survives_rethreshold(self):
        c = box((2, 4, 4), (0, 0, 0), (2, 4, 4), .7, residual=True)
        c.update(probability=c['score'], prompt_center=np.array([1., 2., 2.]), prompt_sigma=np.ones(3))
        self.assertTrue(shrink_candidate(c, .5)['residual'])

    def test_drop_small_instances_relabels_in_order(self):
        labels = np.zeros((1, 6, 6), np.int32); labels[0, 0, :3] = 1; labels[0, 2, :1] = 2; labels[0, 4, :4] = 3
        out, removed = drop_small_instances(labels, 3)
        self.assertEqual(removed, 1)
        self.assertEqual(sorted(np.unique(out)), [0, 1, 2])
        self.assertTrue(np.all(out[0, 4, :4] == 2))
        same, none = drop_small_instances(labels, 0)
        self.assertIs(same, labels); self.assertEqual(none, 0)


class ResidualPeakTests(unittest.TestCase):
    def stub(self, **kwargs):
        cfg = config(proposal_mode='peaks', median_max_dist_nucleus=54., rays_anisotropy_nucleus=(19, 1, 1),
                     proposal_border_nucleus=0, **kwargs)
        return SimpleNamespace(config=cfg, ndim=3, rays_nucleus=cfg.rays_nucleus, rays_cytoplasm=cfg.rays_cytoplasm,
                               _current_type='nucleus')

    def test_window_suppressed_peak_becomes_residual(self):
        prob = np.zeros((8, 32, 32), np.float32); prob[3, 7, 7] = .9; prob[3, 7, 9] = .8; prob[3, 20, 20] = .7
        dist = np.ones(prob.shape + (12,), np.float32)
        for enabled in (False, True):
            stub = self.stub(residual_proposals=enabled, mask_assembly_nucleus='object_quality')
            inference.CytoNucsPredictor._stardist_postprocess(stub, prob, dist, .4, .3)
            primary = np.asarray(stub._last_coords).astype(int).tolist()
            self.assertIn([3, 14, 14], primary); self.assertNotIn([3, 14, 18], primary)
            if enabled:
                self.assertEqual(np.asarray(stub._last_residual['coords']).astype(int).tolist(), [[3, 14, 18]])
                np.testing.assert_allclose(stub._last_residual['probs'], [.8])
            else:
                self.assertIsNone(stub._last_residual)

    def test_prompted_second_pass_only_where_first_pass_left_voxels_free(self):
        cfg = config(nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1), mask_quality_head=True, proposal_mode='peaks',
                     mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                     residual_proposals=True, mask_quality_use_scores=False)
        shape = (8, 32, 32); image = np.zeros(shape + (1,), np.float32)
        def logits(x):
            b = tf.cast(x[-1][..., 0] > .3, tf.float32)   # mask = region of the Gaussian prompt
            return tf.stack([10 * (1 - b), 10 * b, tf.zeros_like(b)], -1)
        def joint(x):
            return logits(x), logits(x), tf.constant([[.8]]), tf.constant([[.8]])
        ctr = np.array([[4, 10, 10]]); labels = np.zeros(shape, np.int32); labels[4, 10, 10] = 1
        dists = np.full((1, 12), 3.)
        residual = dict(nucleus=dict(coords=np.array([[4., 10., 11.], [4., 10., 24.]]),
                                     probs=np.array([.8, .7]), dists=np.full((2, 12), 3.)))
        results = {}
        for name, extra in (('without', None), ('with', residual)):
            model = SimpleNamespace(); sink = {}
            with patch.object(inference, '_refinement_runners', return_value=(lambda x: [x] * 7, joint)), \
                    patch.object(inference, '_cytoplasm_refinement_runner', return_value=None):
                n, c, pairs = inference.predict_instances_prompted(
                    model, cfg, image, ctr, np.ones(1), nuc_dists=dists, nuc_labels=labels,
                    verbose=False, residual=extra, candidate_sink=sink)
            results[name] = (n, c, model.last_refinement_diagnostics, sink)
        n0, c0, _, _ = results['without']; n1, c1, diag, sink = results['with']
        self.assertEqual(int(n0.max()), 1); self.assertEqual(int(n1.max()), 2)
        np.testing.assert_array_equal(n1[n0 > 0], n0[n0 > 0])            # first pass unchanged
        self.assertEqual(diag['nucleus']['residual_peaks'], 2)
        self.assertEqual(diag['nucleus']['residual_prompts'], 1)         # the covered peak is skipped
        self.assertEqual(diag['nucleus']['written_residual'], 1)
        self.assertEqual(int(c1.max()), 2)                                # its cell from the same prompt
        self.assertEqual(sum(bool(x.get('residual')) for x in sink['nucleus']), 1)
        # Calibration re-assembly from the sink reproduces the two-pass result.
        n2, _, _ = inference.rethreshold_prediction(cfg, sink, 'nucleus', .5, n1, c1)
        np.testing.assert_array_equal(n2 > 0, n1 > 0)


class SizeGateTests(unittest.TestCase):
    def rows(self):
        rng = np.random.default_rng(0)
        tp = [(q, .8, 1, a) for q, a in zip(rng.uniform(.5, .95, 40), rng.integers(2000, 20000, 40))]
        tiny_fp = [(q, 0., 0, a) for q, a in zip(rng.uniform(.5, .9, 25), rng.integers(50, 500, 25))]
        weak_fp = [(q, .2, 0, a) for q, a in zip(rng.uniform(.1, .3, 20), rng.integers(2000, 9000, 20))]
        return tp + tiny_fp + weak_fp

    def test_joint_gate_matches_exhaustive_search(self):
        rows = self.rows(); taus = np.arange(.5, .96, .05)
        tp = np.array([sum(r[1] >= t for r in rows) for t in taus]); fp = len(rows) - tp
        counts = {'nucleus': dict(tp=tp, fp=fp, fn=np.full(len(taus), 10) + 0)}
        gates, stats = calibrate_quality_gates({'nucleus': rows}, counts, taus, 60, .7)
        base = accuracy_profile(tp, fp, counts['nucleus']['fn']).mean()
        def gain(qg, ag):
            removed = [r for r in rows if r[0] < qg or r[3] < ag]
            m = np.array([sum(r[1] >= t for r in removed) for t in taus]); u = len(removed) - m
            ap = accuracy_profile(tp - m, fp - u, counts['nucleus']['fn'] + m).mean()
            return checkpoint_objective_gain(-sum(r[2] for r in removed) / 60, ap - base, .7)
        best = max(gain(qg, ag) for qg, ag in product([0.] + sorted({r[0] for r in rows}) + [1.],
                                                     [0] + sorted({r[3] + 1 for r in rows if r[1] < .5})))
        self.assertAlmostEqual(stats['nucleus']['gain'], best, places=12)
        self.assertAlmostEqual(gain(gates['nucleus'], gates.get('nucleus_min_area', 0)), best, places=12)
        tiny = max(r[3] for r in rows if r[1] == 0.); smallest_tp = min(r[3] for r in rows if r[1] >= .5)
        self.assertGreater(gates['nucleus_min_area'], tiny)               # tiny confident FP need the size gate
        self.assertLessEqual(gates['nucleus_min_area'], smallest_tp)
        self.assertEqual(stats['nucleus']['tp_removed'], 0)

    def test_rows_without_area_keep_previous_behaviour(self):
        rows = [(.8, .9, 1), (.55, .6, 0), (.3, 0., 0), (.6, .2, 0)]
        counts = {'nucleus': dict(tp=np.array([2]), fp=np.array([2]), fn=np.array([3]))}
        gates, stats = calibrate_quality_gates({'nucleus': rows}, counts, [.5], 16, .7)
        self.assertNotIn('nucleus_min_area', gates)
        self.assertEqual(stats['nucleus']['min_area'], 0)

    def test_size_gate_is_stored_with_checkpoint(self):
        import h5py
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            path = Path(tmp) / 'w.weights.h5'
            with h5py.File(path, 'w'):
                pass
            write_quality_gates(path, dict(epoch=5, nucleus=.3, cytoplasm=.2, nucleus_min_area=430))
            self.assertEqual(read_quality_gates(path), {'nucleus': .3, 'cytoplasm': .2, 'nucleus_min_area': 430.})

    def test_size_gate_applied_after_assembly_only_in_auto_mode(self):
        cfg = config(nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1), mask_quality_head=True, proposal_mode='peaks',
                     mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                     mask_quality_use_scores=False, mask_quality_min='auto', refinement_stage_audit=True)
        shape = (8, 32, 32); image = np.zeros(shape + (1,), np.float32)
        def logits(x):
            b = tf.cast(x[-1][..., 0] > .3, tf.float32)   # mask = region of the Gaussian prompt
            return tf.stack([10 * (1 - b), 10 * b, tf.zeros_like(b)], -1)
        def joint(x):
            return logits(x), logits(x), tf.constant([[.8]]), tf.constant([[.8]])
        ctr = np.array([[4, 10, 10]]); labels = np.zeros(shape, np.int32); labels[4, 10, 10] = 1
        out = {}
        for gates in ({'nucleus': 0., 'cytoplasm': 0.}, {'nucleus': 0., 'cytoplasm': 0., 'nucleus_min_area': 10 ** 6}):
            model = SimpleNamespace()
            with patch.object(inference, '_refinement_runners', return_value=(lambda x: [x] * 7, joint)), \
                    patch.object(inference, '_cytoplasm_refinement_runner', return_value=None):
                n, c, _ = inference.predict_instances_prompted(
                    model, cfg, image, ctr, np.ones(1), nuc_dists=np.full((1, 12), 3.), nuc_labels=labels,
                    verbose=False, quality_gates=gates)
            out[len(gates)] = (int(n.max()), int(c.max()), model.last_refinement_diagnostics['nucleus']['area_gate_removed'])
        self.assertEqual(out, {2: (1, 1, 0), 3: (0, 1, 1)})


class AuditTests(unittest.TestCase):
    def test_residual_peak_and_geometry_are_reported(self):
        gt = np.zeros((4, 8, 8), np.int32); gt[:2, 1:3, 1:3] = 1; gt[1:3, 4:7, 4:7] = 2
        audit = RefinementStageAudit(gt)
        prob = np.zeros(gt.shape, np.float32); prob[1, 5, 5] = .9; prob[1, 5, 6] = .8; prob[0, 2, 2] = .1
        audit.record_proposals(prob, (1, 1, 1), 0, [3, 5, 5], np.array([[1, 5, 5]]), .5, (1, 1, 1),
                               residual=np.array([[1, 5, 6]]))
        report = audit.report()
        self.assertTrue(report['proposals']['per_gt']['2']['residual_peak'])
        self.assertFalse(report['proposals']['per_gt']['1']['residual_peak'])
        self.assertEqual(report['gt_geometry'], {'1': dict(volume=8, faces='z0'), '2': dict(volume=18, faces='')})
        self.assertEqual(report['monitoring']['gt_touching_volume_face'], 1)
        self.assertIn('proposal_local_max_suppressed_with_residual_peak', report['monitoring'])


class ConfigTests(unittest.TestCase):
    def test_json_contract_and_validation(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            self.assertEqual(raw['inference']['residual_proposals'], path.name in ('config_cytonucs_v2_15.json', 'config_cytonucs_v2_16.json', 'config_cytonucs_v2_17.json'), path.name)
        raw = json.loads((ROOT / 'config_cytonucs_v2_15.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['training']['resume_from_checkpoint'] = None; raw['inference']['checkpoint_path'] = None
            raw['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(raw))
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                preset = CytoNucsConfig.from_json(path)
        self.assertIs(preset.residual_proposals, True)
        self.assertEqual(CytoNucsConfig.FIXES_SETTINGS['inference']['residual_proposals'], False)
        for bad in (dict(residual_proposals=1), dict(residual_proposals=True, proposal_mode='stardist'),
                    dict(residual_proposals=True, proposal_mode='peaks')):     # voxel assembly in both heads
            with self.assertRaises(ValueError):
                config(**bad)


if __name__ == '__main__':
    unittest.main()
