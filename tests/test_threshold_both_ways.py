"""Mask-threshold calibration below the default: candidates store a probability window."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config
from cytonucs_model import build_cytonucs_model
from cytonucs_postprocess import grid_probability_window, upsample_grid
import cytonucs_inference as inference

BASE = dict(nucleus_grid=(1, 2, 2), cytoplasm_grid=(1, 2, 2), prompts_per_crop=2, mask_quality_head=True,
            refinement_depth=1, refinement_gradient_scale=.1, proposal_mode='peaks',
            mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality', mask_quality_use_scores=False)


def smooth_logits(prompt):
    """Foreground log-odds rising with the prompt: lower thresholds give larger masks."""
    pr = prompt[..., ::2, ::2, 0]
    return tf.stack([tf.zeros_like(pr), 8. * (pr - .5), tf.fill(tf.shape(pr), -10.)], -1)


class ThresholdBothWaysTests(unittest.TestCase):
    shape = (8, 32, 32)
    centers = np.array([[4, 9, 9], [4, 22, 22]])

    def labels(self):
        labels = np.zeros(self.shape, np.int32)
        for k, c in enumerate(self.centers, 1):
            labels[tuple(c)] = k
        return labels

    def predict(self, cfg, runners, fine, threshold, sink=None):
        image = np.zeros(self.shape + (cfg.n_channel_in,), np.float32)
        target = '_fine_refinement_runners' if fine else '_refinement_runners'
        extra = {} if fine else dict(_cytoplasm_refinement_runner=None)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(inference, target, return_value=runners))
            for name, value in extra.items():
                stack.enter_context(patch.object(inference, name, return_value=value))
            with contextlib.redirect_stdout(io.StringIO()):
                return inference.predict_instances_prompted(
                    SimpleNamespace(), cfg, image, self.centers, np.ones(2), nuc_dists=np.full((2, 12), 2.5),
                    nuc_labels=self.labels(), verbose=False, candidate_sink=sink, mask_thresh=threshold)

    def check(self, cfg, runners, fine):
        sink = {}
        nuclei, cells, _ = self.predict(cfg, runners, fine, .45, sink)
        self.assertTrue(all('probability_window' in c for c in sink['nucleus'] if c is not None))
        for threshold in (.3, .6):
            direct_n, direct_c, _ = self.predict(cfg, runners, fine, threshold)
            re_n, re_c, _ = inference.rethreshold_prediction(cfg, sink, 'nucleus', threshold, nuclei, cells)
            np.testing.assert_array_equal(re_n > 0, direct_n > 0, err_msg=str(threshold))
            self.assertEqual(len(np.unique(re_n)), len(np.unique(direct_n)))
        grown = inference.rethreshold_prediction(cfg, sink, 'nucleus', .3, nuclei, cells)[0]
        self.assertGreater((grown > 0).sum(), (nuclei > 0).sum())     # lower threshold really grows the masks

    def test_coarse_path_lower_threshold_equals_direct_inference(self):
        cfg = config(**BASE)

        def joint(x):
            lg = smooth_logits(x[-1]); return lg, lg, tf.constant([[.8]]), tf.constant([[.7]])
        self.check(cfg, (lambda x: [x] * 7, joint), fine=False)

    def test_fine_path_lower_threshold_equals_direct_inference(self):
        cfg = config(refinement_full_resolution=True, **BASE)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_cytonucs_model(cfg)
        channels = model.fine_head_models['nucleus'].inputs[2].shape[-1]

        def refine(xs):
            lg = smooth_logits(xs[-1]); hidden = tf.zeros(lg.shape[:-1] + (8,))
            return [lg, lg, tf.constant([[.8]]), tf.constant([[.7]]), hidden, hidden]
        heads = {h: (lambda net: (lambda xs: net(xs)))(m) for h, m in model.fine_head_models.items()}
        runners = (lambda x: [x] * 7, lambda x: tf.zeros(x.shape[:-1] + (channels,)), refine, None, heads)
        self.check(cfg, runners, fine=True)

    def test_window_interpolation_equals_upsample_grid(self):
        rng = np.random.default_rng(0)
        p = rng.random((6, 9, 11)).astype(np.float32); shape = (6, 18, 22)
        full = upsample_grid(p, shape, (1, 2, 2))
        for lo, hi in (((0, 0, 0), shape), ((1, 3, 5), (5, 17, 22)), ((2, 7, 0), (3, 8, 9))):
            np.testing.assert_allclose(grid_probability_window(p, (1, 2, 2), lo, hi),
                                       full[tuple(slice(a, b) for a, b in zip(lo, hi))], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
