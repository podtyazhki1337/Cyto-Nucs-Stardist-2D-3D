"""v3.3: AP with patch-edge fragments as ignore regions (reporting)."""
import unittest
import numpy as np
from test_fixes import ROOT  # noqa: F401  (puts the project root on sys.path)
from cytonucs_runtime import edge_fragment_ignored_ap, calibrate_quality_gates, accuracy_profile

TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)


class EdgeFragmentMetricTests(unittest.TestCase):
    def test_without_fragments_equals_calibrated_gated_ap(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            n = int(rng.integers(5, 60)); n_gt = n + int(rng.integers(0, 20))
            iou = np.where(rng.random(n) < .6, rng.uniform(.5, 1., n), rng.uniform(0, .5, n))
            quality = rng.random(n); area = rng.integers(10, 2000, n)
            tp = np.array([(iou >= t).sum() for t in TAUS])
            counts = dict(tp=tp, fp=n - tp, fn=n_gt - tp)
            gates, stats = calibrate_quality_gates(
                {'nucleus': [(q, i, 0, a) for q, i, a in zip(quality, iou, area)]}, {'nucleus': counts}, TAUS, 100, .7)
            got = edge_fragment_ignored_ap([(q, i, a, False) for q, i, a in zip(quality, iou, area)], [False] * n_gt,
                                           gates['nucleus'], stats['nucleus']['min_area'], TAUS)
            self.assertAlmostEqual(got['ap'], stats['nucleus']['ap'], places=10, msg=trial)

    def test_fragments_are_ignored_on_both_sides(self):
        # GT: two whole objects and one fragment. Masks: TP on a whole object, a mask on the fragment,
        # a background FP, and a small mask removed by the size gate.
        masks = [(.9, .8, 500, False), (.9, .7, 400, True), (.9, 0., 300, False), (.9, .9, 50, False)]
        got = edge_fragment_ignored_ap(masks, [False, False, True], gate=.5, min_area=100, taus=TAUS)
        tp = np.array([1. if t <= .8 else 0. for t in TAUS])
        self.assertAlmostEqual(got['ap'], float(accuracy_profile(tp, 2 - tp, 2 - tp).mean()), places=12)
        self.assertEqual((got['tp'], got['fp'], got['fn'], got['ignored_gt'], got['ignored_masks']), (1, 1, 1, 1, 1))
        # A low predicted quality removes the TP like the deployed gate.
        low = edge_fragment_ignored_ap([(.1,) + m[1:] if i == 0 else m for i, m in enumerate(masks)],
                                       [False, False, True], gate=.5, min_area=100, taus=TAUS)
        self.assertEqual((low['tp'], low['fp'], low['fn']), (0, 1, 2))


if __name__ == '__main__':
    unittest.main()
