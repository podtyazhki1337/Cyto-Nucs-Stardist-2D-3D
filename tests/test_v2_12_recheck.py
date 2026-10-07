"""Full recheck after v2.12: gate usage outside calibration and aligned fullval history."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from test_fixes import config
from cytonucs_inference import CytoNucsPredictor
from cytonucs_trainer import align_fullval_history


def auto_config():
    return config(mask_quality_head=True, mask_assembly_nucleus='object_quality',
                  mask_assembly_cytoplasm='object_quality', mode='predict', mask_quality_min='auto')


class GateUsageTests(unittest.TestCase):
    def test_missing_gates_warn_once_and_calibration_is_ungated(self):
        predictor = CytoNucsPredictor(SimpleNamespace(loaded_quality_gates=None), auto_config())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIsNone(predictor._active_quality_gates())
            self.assertIsNone(predictor._active_quality_gates())
        self.assertEqual(out.getvalue().count('no calibrated quality gates'), 1)
        gated = CytoNucsPredictor(SimpleNamespace(loaded_quality_gates={'nucleus': .3, 'cytoplasm': .2}), auto_config())
        self.assertEqual(gated._active_quality_gates(), {'nucleus': .3, 'cytoplasm': .2})
        gated.calibrating = True
        self.assertIsNone(gated._active_quality_gates())
        numeric = CytoNucsPredictor(SimpleNamespace(loaded_quality_gates={'nucleus': .3}), config())
        self.assertIsNone(numeric._active_quality_gates())

    def test_threshold_search_runs_ungated_and_restores_state(self):
        predictor = CytoNucsPredictor(SimpleNamespace(loaded_quality_gates={'nucleus': .9, 'cytoplasm': .9}),
                                      auto_config(), prob_thresh_nucleus=.5, prob_thresh_cytoplasm=.5, nms_thresh=.3)
        seen = []
        gt = np.zeros((8, 32, 32), np.int32); gt[2:5, 4:9, 4:9] = 1
        def predict(img, normalize_img=False):
            seen.append(predictor._active_quality_gates())
            predictor._prompt_assignments = {1: 1}
            return gt, gt, {}
        with patch.object(predictor, 'predict_instances', predict), contextlib.redirect_stdout(io.StringIO()):
            best = predictor.optimize_thresholds([np.zeros((8, 32, 32, 1), np.float32)], [gt], [gt],
                                                 prob_thresh_range=[.4, .5], nms_thresh_range=[.3],
                                                 mask_thresh_range=[.5], val_assignments=[{1: 1}])
        self.assertTrue(seen and all(g is None for g in seen))
        self.assertFalse(predictor.calibrating)
        self.assertEqual(best['jtpr'], 1.)


class DetectionGraphReuseTests(unittest.TestCase):
    def test_new_predictor_reuses_compiled_detection_graph(self):
        from cytonucs_model import build_cytonucs_model
        cfg = config()
        model = build_cytonucs_model(cfg)
        x = np.zeros((8, 32, 32, 1), np.float32)
        first = CytoNucsPredictor(model, cfg, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
        with contextlib.redirect_stdout(io.StringIO()):
            a = first._predict_with_tiling(x, (1, 1, 1), False)
        functions = dict(model._detection_forward_cache)
        second = CytoNucsPredictor(model, cfg, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
        with contextlib.redirect_stdout(io.StringIO()):
            b = second._predict_with_tiling(x, (1, 1, 1), False)
        for key, fn in functions.items():
            self.assertIs(model._detection_forward_cache[key], fn)
        np.testing.assert_array_equal(a[2]['nucleus_prob'], b[2]['nucleus_prob'])


class HistoryAlignmentTests(unittest.TestCase):
    def test_new_and_missing_keys_stay_on_their_epochs(self):
        history = {'fullval_epoch': [5, 10], 'fullval_jtpr': [.1, .2], 'fullval_old_only': [1, 2]}
        before = {k: len(v) for k, v in history.items()}
        history['fullval_epoch'].append(15); history['fullval_jtpr'].append(.3)
        history.setdefault('fullval_new_key', []).append(7)
        align_fullval_history(history, before)
        self.assertEqual(history['fullval_new_key'], [None, None, 7])
        self.assertEqual(history['fullval_old_only'], [1, 2, None])
        self.assertEqual(history['fullval_jtpr'], [.1, .2, .3])


if __name__ == '__main__':
    unittest.main()
