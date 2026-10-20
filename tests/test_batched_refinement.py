"""Batched prompt refinement (inference.refinement_batch): the candidates of one prompt per pass."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config
from cytonucs_model import build_cytonucs_model
import cytonucs_inference as inference

BASE = dict(nucleus_grid=(1, 2, 2), cytoplasm_grid=(1, 2, 2), prompts_per_crop=2, mask_quality_head=True,
            refinement_depth=1, refinement_gradient_scale=.1, proposal_mode='peaks',
            mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
            mask_quality_use_scores=False, refinement_patch_size=(8, 16, 16), refinement_adaptive_roi=False,
            cell_roi_expansion_steps=2, median_max_dist_cytoplasm=14.)
SHAPE = (8, 40, 40)


def logits(prompt, level):
    """Foreground where the prompt exceeds about `level`; at .05 a cell prompt (sigma 7) reaches 19 voxels out, past
    the faces of its 16-voxel ROI (one expansion) and, away from the image border, of the 32-voxel one (two)."""
    pr = prompt[..., ::2, ::2, 0]
    return tf.stack([tf.zeros_like(pr), 8. * (pr - level), tf.fill(tf.shape(pr), -10.)], -1)


def quality(prompt, base):
    """One value per prompt that differs between prompts (batch rows must not be mixed up)."""
    return base + .1 * tf.reduce_mean(prompt, axis=list(range(1, len(prompt.shape))))[:, None]


def point_labels(centres):
    labels = np.zeros(SHAPE, np.int32)
    for k, c in enumerate(centres, 1):
        labels[tuple(c)] = k
    return labels


class BatchedRefinementTests(unittest.TestCase):
    nuclei = np.array([[4, 8, 8], [4, 20, 22], [3, 30, 12], [5, 9, 31]])
    cells = np.array([[4, 9, 9], [4, 28, 28], [4, 20, 20]])

    def predict(self, cfg, runners, fine, batch, sink=None):
        cfg.refinement_batch = batch
        model = SimpleNamespace()
        image = np.zeros(SHAPE + (cfg.n_channel_in,), np.float32)
        with contextlib.ExitStack() as stack:
            if fine:
                stack.enter_context(patch.object(inference, '_fine_refinement_runners', return_value=runners))
            else:
                stack.enter_context(patch.object(inference, '_refinement_runners', return_value=runners[:2]))
                stack.enter_context(patch.object(inference, '_cytoplasm_refinement_runner', return_value=runners[3]))
            with contextlib.redirect_stdout(io.StringIO()):
                out = inference.predict_instances_prompted(
                    model, cfg, image, self.nuclei, np.ones(len(self.nuclei)), self.cells, np.ones(len(self.cells)),
                    nuc_dists=np.full((len(self.nuclei), 12), 2.5), nuc_labels=point_labels(self.nuclei),
                    cyto_labels=point_labels(self.cells), verbose=False, candidate_sink=sink, mask_thresh=.45)
        return out, model.last_refinement_diagnostics

    def coarse_runners(self):
        def refine(xs):
            p = xs[-1]
            return logits(p, .3), logits(p, .05), quality(p, .8), quality(p, .7)

        def cell(xs):
            return logits(xs[-1], .05), quality(xs[-1], .6)
        return lambda x: [x] * 7, refine, None, cell

    def fine_runners(self, cfg):
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_cytonucs_model(cfg)
        rng = np.random.default_rng(0)
        for head in model.fine_head_models.values():          # a non-zero correction, small near the mask
            delta = next(layer for layer in head.layers if layer.name.endswith('_delta'))
            delta.set_weights([.05 * rng.standard_normal(w.shape).astype(np.float32) for w in delta.get_weights()])
        channels = model.fine_head_models['nucleus'].inputs[2].shape[-1]

        def hidden(lg):
            return tf.concat([lg, tf.zeros(lg.shape[:-1] + (5,))], -1)

        def refine(xs):
            p = xs[-1]; n, c = logits(p, .3), logits(p, .05)
            return [n, c, quality(p, .8), quality(p, .7), hidden(n), hidden(c)]

        def cell(xs):
            c = logits(xs[-1], .05)
            return [c, quality(xs[-1], .6), hidden(c)]
        heads = {h: (lambda net: (lambda xs: net(xs)))(m) for h, m in model.fine_head_models.items()}
        return lambda x: [x] * 7, lambda x: tf.zeros(x.shape[:-1] + (channels,)), refine, cell, heads

    def test_coarse_batched_equals_one_prompt(self):
        cfg = config(**BASE)
        one_sink, batched_sink = {}, {}
        (n1, c1, p1), d1 = self.predict(cfg, self.coarse_runners(), False, 0, one_sink)
        for batch in (1, 3, 64):
            sink = {}
            (n2, c2, p2), d2 = self.predict(cfg, self.coarse_runners(), False, batch, sink)
            np.testing.assert_array_equal(n1, n2); np.testing.assert_array_equal(c1, c2)
            self.assertEqual(p1, p2); self.assertEqual(d1, d2)
            for head in ('nucleus', 'cytoplasm'):
                self.assertEqual([c is None for c in one_sink[head]], [c is None for c in sink[head]])
                for a, b in zip(one_sink[head], sink[head]):
                    if a is not None:
                        self.assertEqual(a['quality'] if 'quality' in a else a['predicted_quality'],
                                         b['quality'] if 'quality' in b else b['predicted_quality'])
                        np.testing.assert_array_equal(a['mask'], b['mask'])
                        np.testing.assert_array_equal(a['probability_window'][1], b['probability_window'][1])
        # The test covers cell ROI expansions of both rounds.
        self.assertGreater(d1['cytoplasm']['roi_expansions'], len(self.cells))
        self.assertGreater(len(np.unique(c1)), 2)

    def test_fine_batched_equals_one_prompt_away_from_window_faces(self):
        # A 32-voxel ROI: nuclear windows (mask + margin) end inside it, cells still expand once.
        cfg = config(refinement_full_resolution=True, **dict(BASE, refinement_patch_size=(8, 32, 32)))
        one_sink, sink = {}, {}
        runners = self.fine_runners(cfg)                        # one set of random weights for both paths
        (n1, c1, p1), d1 = self.predict(cfg, runners, True, 0, one_sink)
        (n2, c2, p2), d2 = self.predict(cfg, runners, True, 4, sink)
        np.testing.assert_array_equal(n1, n2); np.testing.assert_array_equal(c1, c2)
        self.assertEqual(p1, p2)
        windows = 0
        for head in ('nucleus', 'cytoplasm'):
            for key in ('attempted', 'accepted', 'roi_expansions', 'empty_threshold'):
                self.assertEqual(d1[head][key], d2[head][key])
            for a, b in zip(one_sink[head], sink[head]):
                self.assertEqual(a is None, b is None)
                if a is None:
                    continue
                np.testing.assert_array_equal(a['lo'], b['lo']); np.testing.assert_array_equal(a['mask'], b['mask'])
                wa, wb = a['probability_window'][1], b['probability_window'][1]
                self.assertEqual(wa.shape, wb.shape)
                # The window pass saw zero padding beyond its faces (2 voxels deep through two 3x3 convolutions).
                inner = tuple(slice(2, -2) if s > 4 else slice(None) for s in wa.shape)
                np.testing.assert_allclose(wa[inner], wb[inner], atol=1e-5)
                if head == 'nucleus':
                    windows += int(wa.shape != tuple(cfg.refinement_patch_size))
        self.assertEqual(windows, len(self.nuclei))                # nuclear windows smaller than the ROI

    def test_windowed_refinement_equals_the_whole_roi(self):
        # A 96-voxel ROI (48 grid nodes): windows of 24-32 nodes around each prompt; masks inside them.
        shape = (8, 96, 96)
        nuclei = np.array([[4, 20, 20], [4, 50, 60], [3, 75, 30], [5, 30, 80]])
        cells = np.array([[4, 22, 24], [4, 70, 70]])

        def labels(points):
            out = np.zeros(shape, np.int32)
            for k, c in enumerate(points, 1):
                out[tuple(c)] = k
            return out

        def run(cfg, runners, fine, windows, sink):
            cfg.refinement_batch, cfg.refinement_windows = 4, windows
            model = SimpleNamespace()
            with contextlib.ExitStack() as stack:
                if fine:
                    stack.enter_context(patch.object(inference, '_fine_refinement_runners', return_value=runners))
                else:
                    stack.enter_context(patch.object(inference, '_refinement_runners', return_value=runners[:2]))
                    stack.enter_context(patch.object(inference, '_cytoplasm_refinement_runner', return_value=runners[3]))
                with contextlib.redirect_stdout(io.StringIO()):
                    out = inference.predict_instances_prompted(
                        model, cfg, np.zeros(shape + (cfg.n_channel_in,), np.float32), nuclei, np.ones(len(nuclei)),
                        cells, np.ones(len(cells)), nuc_dists=np.full((len(nuclei), 12), 2.5), nuc_labels=labels(nuclei),
                        cyto_labels=labels(cells), verbose=False, candidate_sink=sink, mask_thresh=.45)
            return out, model.last_refinement_diagnostics

        grid_features = lambda x: [x[:, :, ::2, ::2]] * 7                    # features on the head grid
        for fine in (False, True):
            base = dict(BASE, refinement_patch_size=shape, median_max_dist_cytoplasm=6.)
            cfg = config(refinement_full_resolution=True, **base) if fine else config(**base)
            runners = self.fine_runners(cfg) if fine else self.coarse_runners()
            runners = (grid_features,) + tuple(runners[1:])
            whole, windowed = {}, {}
            (n1, c1, p1), d1 = run(cfg, runners, fine, False, whole)
            (n2, c2, p2), d2 = run(cfg, runners, fine, True, windowed)
            np.testing.assert_array_equal(n1, n2, err_msg=str(fine)); np.testing.assert_array_equal(c1, c2, err_msg=str(fine))
            self.assertEqual(p1, p2)
            self.assertEqual(len(np.unique(n1)), len(nuclei) + 1)
            if not fine:
                self.assertEqual(len(np.unique(c1)), len(nuclei) + len(cells) + 1)
            for head in ('nucleus', 'cytoplasm'):
                self.assertEqual(d1[head]['accepted'], d2[head]['accepted'])
                self.assertEqual(d1[head]['windowed'], 0)
                self.assertGreater(d2[head]['windowed'], 0)
                if not fine:                       # (the random fine head grows its windows past the refinement windows)
                    self.assertEqual(d2[head]['window_fallbacks'], 0)
                for a, b in zip(whole[head], windowed[head]):
                    self.assertEqual(a is None, b is None)
                    if a is not None:
                        np.testing.assert_array_equal(a['mask'], b['mask'])
                        if not fine:
                            np.testing.assert_array_equal(a['probability_window'][1], b['probability_window'][1])

    def test_setting_is_validated(self):
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                config(refinement_batch=bad, **BASE)


if __name__ == '__main__':
    unittest.main()
