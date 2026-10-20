"""Detector prompts must preserve ownership and use unmasked probability peaks."""
import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from test_fixes import config
from cytonucs_trainer import CytoNucsTrainer


class PromptOwnershipTests(unittest.TestCase):
    def run_prompt(self, target, peaks, sample=True, graph=False):
        cfg = config(predicted_prompt_fraction=1., predicted_prompt_sample_peaks=sample,
                     prob_thresh_nucleus=.5, prob_thresh_cytoplasm=.5)
        shape = (8, 32, 32)
        x = np.full((1,) + shape + (5,), .123, np.float32)
        p = np.zeros((1, 8, 16, 16, 1), np.float32)
        for coordinate, value in peaks:
            p[(0,) + coordinate + (0,)] = value
        pred = {h + '_prob': tf.constant(p) for h in ('nucleus', 'cytoplasm')}
        pred.update({h + '_dist': tf.ones((1, 8, 16, 16, 12)) * 4
                     for h in ('nucleus', 'cytoplasm')})
        y = {}
        for h in ('nucleus', 'cytoplasm'):
            y[h + '_target'] = tf.constant(np.repeat(target[None], 2, axis=0), tf.float32)
            y[h + '_instances'] = tf.constant(target[None], tf.int32)
        stub = SimpleNamespace(config=cfg, prompt_detector=lambda x, training: pred,
                               optimizer=SimpleNamespace(iterations=tf.Variable(0, dtype=tf.int64)))
        call = lambda a: CytoNucsTrainer._predicted_prompts(stub, a, y)
        if graph:
            call = tf.function(call)
        return x, call(tf.constant(x)).numpy()

    def test_outside_neighbour_cannot_create_owned_peak(self):
        target = np.zeros((8, 32, 32), np.int32)
        target[3, 12, 12] = 1
        x, out = self.run_prompt(target, [((3, 6, 6), .6), ((3, 6, 7), .9)], graph=True)
        np.testing.assert_array_equal(out, x)

    def test_subthreshold_and_equal_threshold_keep_original(self):
        target = np.zeros((8, 32, 32), np.int32)
        target[3, 12, 12] = 1
        for value in (.4, .5):
            with self.subTest(value=value):
                x, out = self.run_prompt(target, [((3, 6, 6), value)])
                np.testing.assert_array_equal(out, x)

    def test_positive_without_grid_support_never_uses_background(self):
        target = np.zeros((8, 32, 32), np.int32)
        target[3, 13, 13] = 1
        for sample in (False, True):
            with self.subTest(sample=sample):
                x, out = self.run_prompt(target, [((3, 2, 2), .9)], sample=sample)
                np.testing.assert_array_equal(out, x)

    def test_valid_positive_and_negative_peaks_still_replace(self):
        for positive in (False, True):
            target = np.zeros((8, 32, 32), np.int32)
            if positive:
                target[3, 12, 12] = 1
            x, out = self.run_prompt(target, [((3, 6, 6), .9)], graph=True)
            np.testing.assert_array_equal(out[..., 0], x[..., 0])
            for channel in range(1, 5):
                self.assertEqual(np.unravel_index(np.argmax(out[0, ..., channel]), target.shape), (3, 12, 12))
                self.assertEqual(out[0, 3, 12, 12, channel], 1.)


if __name__ == '__main__':
    unittest.main()
