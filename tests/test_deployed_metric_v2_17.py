"""Checkpoint selection by the metric after the calibrated quality/size gates."""
import json
import unittest
from test_fixes import ROOT
from cytonucs_runtime import deployed_metric, history_selection_metrics, restored_best_metric


def history(**cols):
    h = {'fullval_' + k: v for k, v in cols.items()}
    h['fullval_epoch'] = list(range(len(cols['metric'])))
    return h


class DeployedMetricTests(unittest.TestCase):
    def test_formula(self):
        value = deployed_metric(0.7, 1000, {'nucleus': .4, 'cytoplasm': .3}, {'nucleus': 5, 'cytoplasm': 5}, .7)
        self.assertAlmostEqual(value, .7 * (.7 - .01) + .3 * .5 * .7)

    def test_history_prefers_stored_then_recomputes_then_raw(self):
        h = history(metric=[.50, .55, .56], jtpr=[.6, .7, .7],
                    ap_nucleus=[.2, .2, .2], ap_cytoplasm=[.2, .2, .2],
                    ap_nucleus_quality_gated=[None, .4, .4], ap_cytoplasm_quality_gated=[None, None, .3],
                    nucleus_quality_gate_pairs_lost=[None, 10, 0], cytoplasm_quality_gate_pairs_lost=[None, None, 0],
                    joint_fail_nucleus=[1, 200, 200], joint_fail_cell=[1, 100, 100], joint_fail_link=[1, 0, 0],
                    metric_deployed=[None, None, .99])
        out = history_selection_metrics(h, .7)
        self.assertEqual(out[0], .50)                                  # no gate information
        self.assertAlmostEqual(out[1], .7 * (.7 - 10 / 1000) + .15 * (.4 + .2))   # 300 fails / 0.3 = 1000 pairs
        self.assertEqual(out[2], .99)                                  # stored value wins
        self.assertEqual(restored_best_metric(h, .5, 0, .7), .99)
        self.assertAlmostEqual(restored_best_metric(h, .5, 1, .7), .99)  # selection start keeps later epochs

    def test_real_v2_17_history_selects_best_deployed_epoch(self):
        path = ROOT / 'fixes_v2.17' / 'history.json'
        if not path.exists():
            self.skipTest('user package not present')
        h = json.loads(path.read_text(encoding='utf-8'))
        values = history_selection_metrics(h, .7)
        epochs = h['fullval_epoch']
        best = max((v, e) for v, e in zip(values, epochs) if v is not None and e > 520)
        self.assertEqual(best[1], 545)
        self.assertAlmostEqual(best[0], 0.5875, places=4)
        # Raw metric of the same run is lower and ranks v2.17 below v2.16/515.
        raw = dict(zip(epochs, h['fullval_metric']))
        self.assertLess(raw[545], raw[515])
        self.assertGreater(values[epochs.index(545)], values[epochs.index(515)])


class FullValidationTests(unittest.TestCase):
    def test_callback_logs_and_selects_deployed_metric(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from test_fixes import config
        from cytonucs_model import build_cytonucs_model
        from cytonucs_trainer import DiceCallback
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = config(refinement_stage_audit=True, mask_quality_head=True, proposal_mode='peaks',
                         mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                         mask_quality_min='auto', prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            model = build_cytonucs_model(cfg, basedir=tmp)
            gt = np.zeros((8, 32, 32), np.int32); gt[2:6, 10:20, 10:20] = 1
            gen = SimpleNamespace(image_files=['a.tiff'], data_pool=[dict(
                image=(gt > 0).astype(np.float32)[..., None], nucleus_mask=gt, cytoplasm_mask=gt,
                assignments={1: 1}, filename='a.tiff')])
            history = {'fullval_epoch': [450], 'fullval_metric': [.1]}
            callback = DiceCallback(model, gen, eval_every=1, save_best=True,
                                    prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            with patch.object(callback, '_save_epoch_visualization_full'):
                callback.on_epoch_end(454, history)
            n = len(history['fullval_epoch'])
            for key in ('fullval_metric_deployed', 'fullval_total_pairs'):
                self.assertEqual(len(history[key]), n)
                self.assertIsNone(history[key][0])
            self.assertEqual(history['fullval_total_pairs'][-1], 1)
            self.assertGreaterEqual(history['fullval_metric_deployed'][-1], history['fullval_metric'][-1])
            best = json.loads((Path(tmp) / model.name / 'best_fullval.json').read_text())
            self.assertEqual(best['metric'], history['fullval_metric_deployed'][-1])
            self.assertEqual(best['metric_without_gates'], history['fullval_metric'][-1])
            self.assertEqual(callback.best_metric, best['metric'])


if __name__ == '__main__':
    unittest.main()
