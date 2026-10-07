"""Z-slice / XY-contour split of the boundary error of matched masks (reporting only)."""
import unittest
import numpy as np
from test_fixes import ROOT  # noqa: F401  (puts the project root on sys.path)
from cytonucs_postprocess import mask_candidate, assemble_masks
from cytonucs_refinement_audit import RefinementStageAudit
from cytonucs_runtime import boundary_error_ap, calibrate_quality_gates, accuracy_profile

TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)


def iou(a, b):
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


def disk(shape, center, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2


class BoundarySplitTests(unittest.TestCase):
    def volumes(self):
        # GT nucleus on slices 2..6 (5 slices), a neighbour on another label. Prediction: slices 3..7
        # (slice 2 missing, slice 7 extra) with contours off by one pixel on two shared slices.
        shape = (10, 40, 40)
        gt = np.zeros(shape, np.int32)
        pred = np.zeros(shape, bool)
        for z, r in zip(range(2, 7), (6, 9, 10, 9, 6)):
            gt[z][disk(shape[1:], (20, 20), r)] = 7
        gt[4][disk(shape[1:], (5, 5), 3)] = 3
        for z, r in zip(range(3, 8), (10, 10, 8, 6, 4)):
            pred[z] = disk(shape[1:], (20, 20), r)
        return gt, pred

    def parts(self, pred, g):
        """Independent full-volume partition of the error voxels and the corrected predictions."""
        z = np.logical_xor(pred.any(axis=(1, 2)), g.any(axis=(1, 2)))[:, None, None] & np.ones_like(g)
        xy = np.logical_xor(pred.any(axis=0), g.any(axis=0))[None] & ~z
        out = {}
        for part, where in (('z', z), ('xy', xy), ('mixed', ~z & ~xy)):
            corrected = (pred & ~(pred & ~g & where)) | (g & ~pred & where)
            out[part] = (iou(corrected, g), int(np.logical_xor(pred, g)[where].sum()))
        return out

    def test_split_equals_explicit_corrected_masks(self):
        gt, pred = self.volumes()
        g = gt == 7
        lo = np.array([3, 8, 8]); crop = pred[3:8, 8:33, 8:33]
        self.assertEqual(crop.sum(), pred.sum())       # the GT extends outside the crop in Z
        split = RefinementStageAudit(gt).boundary_split(crop, lo, 7)
        expected = self.parts(pred, g)
        for part, (value, voxels) in expected.items():
            self.assertAlmostEqual(split['iou_' + part + '_exact'], value, places=12, msg=part)
            self.assertEqual(split[part + '_error_voxels'], voxels, msg=part)
            self.assertGreaterEqual(split['iou_' + part + '_exact'], iou(pred, g))
        self.assertEqual(sum(v for _, v in expected.values()), split['error_voxels'])
        self.assertEqual(split['error_voxels'], np.logical_xor(pred, g).sum())
        self.assertEqual((split['extra_slices'], split['missing_slices']), (1, 1))
        self.assertEqual(split['z_error_voxels'], pred[7].sum() + g[2].sum())

    def test_pure_cases(self):
        shape = (10, 40, 40)
        gt = np.zeros(shape, np.int32)
        for z in range(2, 6):
            gt[z][disk(shape[1:], (20, 20), 8)] = 1
        g = gt == 1
        audit = RefinementStageAudit(gt)
        extra_slice = g.copy(); extra_slice[6] = g[5]
        wider = np.zeros_like(g); wider[2:6] = disk(shape[1:], (20, 20), 9)
        for pred, only in ((extra_slice, 'z'), (wider, 'xy')):
            split = audit.boundary_split(pred, np.zeros(3, int), 1)
            self.assertEqual(split[only + '_error_voxels'], split['error_voxels'], msg=only)
            self.assertEqual(split['iou_' + only + '_exact'], 1.0, msg=only)
        # A tapered object one slice too low: whole-slice and rim errors, same XY projection.
        taper = np.zeros(shape, np.int32)
        for z, r in zip(range(2, 7), (6, 9, 10, 9, 6)):
            taper[z][disk(shape[1:], (20, 20), r)] = 1
        shifted = np.zeros(shape, bool); shifted[3:8] = taper[2:7] == 1
        split = RefinementStageAudit(taper).boundary_split(shifted, np.zeros(3, int), 1)
        self.assertEqual(split['xy_error_voxels'], 0)
        self.assertGreater(split['z_error_voxels'], 0); self.assertGreater(split['mixed_error_voxels'], 0)
        self.assertEqual((split['extra_slices'], split['missing_slices']), (1, 1))

    def test_recorded_for_matched_written_masks_only(self):
        gt, pred = self.volumes()
        good = mask_candidate(np.full(gt.shape, .9), pred, np.zeros(3, int)); good['predicted_quality'] = .9
        speck = np.zeros(gt.shape, bool); speck[0, 30:34, 30:34] = True
        bad = mask_candidate(np.full(gt.shape, .9), speck, np.zeros(3, int)); bad['predicted_quality'] = .5
        audit = RefinementStageAudit(gt)
        expected, _ = assemble_masks([good, bad], gt.shape, assembly_mode='object_quality')
        out, _ = assemble_masks([good, bad], gt.shape, assembly_mode='object_quality', observer=audit)
        np.testing.assert_array_equal(out, expected)            # observer only
        written = {d['rank']: d for d in audit.assembly_decisions if d['reason'] == 'written'}
        self.assertEqual(set(written), {0, 1})
        self.assertEqual(written[0]['boundary']['gt'], '7')
        self.assertAlmostEqual(written[0]['written_gt_iou']['7'], iou(pred, gt == 7), places=12)
        self.assertNotIn('boundary', written[1])
        # 2D volumes have no Z split.
        audit2 = RefinementStageAudit(gt[4])
        c2 = mask_candidate(np.full(gt.shape[1:], .9), pred[4], np.zeros(2, int)); c2['predicted_quality'] = .9
        assemble_masks([c2], gt.shape[1:], assembly_mode='object_quality', observer=audit2)
        self.assertNotIn('boundary', audit2.assembly_decisions[0])


class BoundaryErrorApTests(unittest.TestCase):
    def test_without_split_equals_calibrated_gated_ap(self):
        rng = np.random.default_rng(1)
        for trial in range(20):
            n = int(rng.integers(5, 60)); n_gt = n + int(rng.integers(0, 20))
            ious = np.where(rng.random(n) < .6, rng.uniform(.5, 1., n), rng.uniform(0, .5, n))
            quality = rng.random(n); area = rng.integers(10, 2000, n)
            tp = np.array([(ious >= t).sum() for t in TAUS])
            counts = dict(tp=tp, fp=n - tp, fn=n_gt - tp)
            gates, stats = calibrate_quality_gates(
                {'h': [(q, i, 0, a) for q, i, a in zip(quality, ious, area)]}, {'h': counts}, TAUS, 100, .7)
            got = boundary_error_ap([(q, i, a, None) for q, i, a in zip(quality, ious, area)], counts,
                                    gates['h'], stats['h']['min_area'], TAUS)
            self.assertAlmostEqual(got['ap'], stats['h']['ap'], places=10, msg=trial)
            self.assertEqual((got['ap_z_exact'], got['ap_xy_exact'], got['ap_mixed_exact']), (got['ap'],) * 3)

    def test_gains_of_kept_matched_masks(self):
        split = lambda z, xy, mixed: dict(iou_z_exact=z, iou_xy_exact=xy, iou_mixed_exact=mixed, extra_slices=1,
                                          missing_slices=0, z_error_voxels=30, xy_error_voxels=6,
                                          mixed_error_voxels=4, error_voxels=40)
        # Kept TP 0.8 (Z-exact 0.97), TP removed by the size gate, a background FP, 3 GT.
        masks = [(.9, .8, 500, split(.97, .82, .81)), (.9, .9, 50, split(1., 1., 1.)), (.9, .1, 300, None)]
        ious = np.array([m[1] for m in masks])
        tp = np.array([(ious >= t).sum() for t in TAUS]); counts = dict(tp=tp, fp=3 - tp, fn=3 - tp)
        got = boundary_error_ap(masks, counts, gate=.5, min_area=100, taus=TAUS)
        profile = lambda v: float(accuracy_profile(*(lambda h: (h, 2 - h, 3 - h))(
            np.array([float(v >= t) for t in TAUS]))).mean())
        self.assertAlmostEqual(got['ap'], profile(.8), places=12)
        self.assertAlmostEqual(got['ap_z_exact'], profile(.97), places=12)
        self.assertAlmostEqual(got['ap_xy_exact'], profile(.82), places=12)
        self.assertEqual(got['matched_masks'], 1)
        self.assertEqual((got['z_extent_exact_fraction'], got['z_extent_one_slice_fraction']), (0., 1.))
        self.assertAlmostEqual(got['ap_mixed_exact'], profile(.81), places=12)
        self.assertAlmostEqual(got['z_error_voxel_share'], .75)
        self.assertAlmostEqual(got['xy_error_voxel_share'] + got['mixed_error_voxel_share'], .25)


class FullValidationTests(unittest.TestCase):
    def test_history_gets_split_of_matched_masks(self):
        # Real DiceCallback aggregation; only the network prediction is replaced by object assembly
        # of GT-derived candidates observed by the callback's own audits. One extra Z slice per mask.
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from test_fixes import ROOT, config
        from cytonucs_model import build_cytonucs_model
        from cytonucs_trainer import DiceCallback
        gt = np.zeros((12, 32, 32), np.int32)
        gt[2:7, 4:14, 4:14] = 1; gt[3:8, 18:28, 16:28] = 2
        cfg = config(refinement_stage_audit=True, mask_quality_head=True, mask_assembly_nucleus='object_quality',
                     mask_assembly_cytoplasm='object_quality', prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)

        def fake(predictor, img):
            out = []
            for head in ('nucleus', 'cytoplasm'):
                candidates = []
                for lab in (1, 2):
                    m = gt == lab
                    z = np.flatnonzero(m.any(axis=(1, 2)))[-1]
                    m[z + 1] = m[z]
                    c = mask_candidate(np.full(gt.shape, .9), m, np.zeros(3, int)); c['predicted_quality'] = .9
                    candidates.append(c)
                labels, _ = assemble_masks(candidates, gt.shape, assembly_mode='object_quality',
                                           observer=predictor._refinement_stage_audit[head])
                out.append(labels)
            return out[0], out[1], {}

        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            model = build_cytonucs_model(cfg, basedir=tmp)
            gen = SimpleNamespace(image_files=['s.tiff'], data_pool=[dict(
                image=(gt > 0).astype(np.float32)[..., None], nucleus_mask=gt, cytoplasm_mask=gt,
                assignments={1: 1, 2: 2}, filename='s.tiff')])
            callback = DiceCallback(model, gen, eval_every=1, save_best=False,
                                    prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            history = {}
            with patch.object(callback, '_save_epoch_visualization_full'),                     patch.object(callback, '_predict_full_volume', side_effect=fake):
                callback.on_epoch_end(4, history)
        one_extra = [float(min(5 / 6, 60 / 72) >= t) for t in TAUS]
        for head in ('nucleus', 'cytoplasm'):
            gated = history[f'fullval_ap_{head}_quality_gated'][-1]
            self.assertAlmostEqual(gated, float(np.mean(one_extra)), places=12)
            self.assertEqual(history[f'fullval_ap_{head}_quality_gated_z_exact'][-1], 1.0)
            self.assertEqual(history[f'fullval_ap_{head}_quality_gated_xy_exact'][-1], gated)
            self.assertEqual(history[f'fullval_ap_{head}_quality_gated_mixed_exact'][-1], gated)
            self.assertEqual(history[f'fullval_{head}_z_error_voxel_share'][-1], 1.0)
            self.assertEqual(history[f'fullval_{head}_z_extent_one_slice_fraction'][-1], 1.0)


if __name__ == '__main__':
    unittest.main()
