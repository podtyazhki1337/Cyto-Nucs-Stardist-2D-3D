"""v3: exact local mask evaluation, ROI snapping, Lovász overlap loss, gate-derived min object size."""
import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from cytonucs_config import CytoNucsConfig
from cytonucs_postprocess import local_prompt_mask, upsample_grid, prompt_component
from cytonucs_losses import CytoNucsLoss, lovasz_jaccard_loss, without_small_objects
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import sync_object_min_area
import cytonucs_inference as inference

NEW = {'inference': dict(refinement_roi_snap=(False, True)),
       'loss_params': dict(mask_overlap_loss=('dice', 'lovasz'), object_losses_min_area_from_gate=(False, True))}


def legacy_mask(p_grid, shape, grid, center, sigma, threshold):
    prob = upsample_grid(p_grid, shape, grid)
    fg = prob > threshold
    mask = prompt_component(fg, center, sigma) if fg.any() else fg
    return np.zeros(len(shape), int), prob, fg, mask, bool(fg.any())


def place(lo, window, shape):
    full = np.zeros(shape, window.dtype)
    full[tuple(slice(int(a), int(a) + s) for a, s in zip(lo, window.shape))] = window
    return full


class LocalMaskTests(unittest.TestCase):
    def test_exactly_equals_full_roi_evaluation(self):
        rng = np.random.default_rng(3)
        for trial in range(300):
            grid = [(1, 2, 2), (1, 1, 1), (2, 2, 2)][trial % 3]
            shape = (int(rng.integers(4, 17)), int(rng.integers(10, 60)), int(rng.integers(10, 60)))
            gs = tuple(-(-s // g) + int(rng.integers(0, 3)) for s, g in zip(shape, grid))
            p = gaussian_filter(rng.random(gs), (0.8, 1.5, 1.5)); p = (p - p.min()) / (np.ptp(p) + 1e-9)
            threshold = float(rng.uniform(.3, .8))
            center = rng.uniform([-2, -3, -3], [shape[0] + 2, shape[1] + 3, shape[2] + 3])
            sigma = rng.uniform([.5, 1, 1], [3, 8, 8])
            _, prob, fg, mask, any_fg = legacy_mask(p, shape, grid, center, sigma, threshold)
            lo, prob_w, fg_w, mask_w, any_w = local_prompt_mask(p, shape, grid, center, sigma, threshold)
            self.assertEqual(any_w, any_fg, trial)
            full_mask = place(lo, mask_w, shape) if mask_w.any() else np.zeros(shape, bool)
            np.testing.assert_array_equal(full_mask, mask, err_msg=str(trial))
            if mask.any():
                np.testing.assert_array_equal(place(lo, prob_w, shape)[mask], prob[mask])
                np.testing.assert_array_equal(place(lo, fg_w, shape) & mask, fg & mask)

    def test_faster_on_a_realistic_roi(self):
        rng = np.random.default_rng(0); zz, yy, xx = np.mgrid[:16, :224, :224]
        slow = fast = 0.
        for _ in range(5):
            p = np.zeros((16, 224, 224), np.float32)
            p[((zz - 8) / 5) ** 2 + ((yy - 100) / 20) ** 2 + ((xx - 110) / 20) ** 2 <= 1] = .9
            for _ in range(30):
                oy, ox = rng.integers(0, 224, 2); o = rng.integers(2, 15)
                p[((zz - rng.integers(0, 16)) / 3) ** 2 + ((yy - oy) / o) ** 2 + ((xx - ox) / o) ** 2 <= 1] = .7
            p = gaussian_filter(p, (.5, 1, 1)); args = (p, (16, 448, 448), (1, 2, 2), np.array([8, 200, 220.]),
                                                          np.array([2, 10, 10.]), .45)
            t = time.perf_counter(); legacy_mask(*args); slow += time.perf_counter() - t
            t = time.perf_counter(); local_prompt_mask(*args); fast += time.perf_counter() - t
        self.assertLess(fast * 3, slow)

    def test_prompted_inference_is_unchanged(self):
        cfg = config(nucleus_grid=(1, 2, 2), cytoplasm_grid=(1, 2, 2), mask_quality_head=True, proposal_mode='peaks',
                     mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                     mask_quality_use_scores=False)
        shape = (8, 32, 64); image = np.zeros(shape + (1,), np.float32)
        def logits(x):
            b = tf.cast(x[-1][..., ::2, ::2, 0] > .3, tf.float32)   # prompt region on the head grid
            return tf.stack([10 * (1 - b), 10 * b, tf.zeros_like(b)], -1)
        def joint(x):
            return logits(x), logits(x), tf.constant([[.8]]), tf.constant([[.7]])
        ctr = np.array([[4, 10, 10], [4, 20, 40], [4, 12, 14]]); labels = np.zeros(shape, np.int32)
        for k, c in enumerate(ctr, 1):
            labels[tuple(c)] = k
        outputs = []
        for impl in (legacy_mask, local_prompt_mask):
            model = SimpleNamespace()
            with patch.object(inference, '_refinement_runners', return_value=(lambda x: [x] * 7, joint)), \
                    patch.object(inference, '_cytoplasm_refinement_runner', return_value=None), \
                    patch.object(inference, 'local_prompt_mask', impl):
                n, c, pairs = inference.predict_instances_prompted(
                    model, cfg, image, ctr, np.ones(3), nuc_dists=np.full((3, 12), 3.), nuc_labels=labels, verbose=False)
            outputs.append((n, c, pairs, model.last_refinement_diagnostics))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        self.assertEqual(outputs[0][2], outputs[1][2])
        self.assertGreater(int(outputs[1][0].max()), 0)
        for h in ('nucleus', 'cytoplasm'):
            for key in ('attempted', 'accepted', 'empty_threshold', 'missing_component', 'written'):
                self.assertEqual(outputs[0][3][h][key], outputs[1][3][h][key], (h, key))


class MeshgridNumpy:
    """numpy whose ix_ returns the full 'ij' meshgrid: the pre-change prompt construction."""
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def ix_(*axes):
        return np.meshgrid(*axes, indexing='ij')


class PromptBuildTests(unittest.TestCase):
    def test_inference_prompts_bit_identical_to_meshgrid(self):
        cfg = config(nucleus_grid=(1, 2, 2), cytoplasm_grid=(1, 2, 2), mask_quality_head=True, proposal_mode='peaks',
                     mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                     mask_quality_use_scores=False)
        shape = (8, 32, 64); image = np.zeros(shape + (1,), np.float32)
        ctr = np.array([[4, 10, 10], [3, 20, 40], [5, 12, 14]]); labels = np.zeros(shape, np.int32)
        for k, c in enumerate(ctr, 1):
            labels[tuple(c)] = k
        runs = []
        for numpy_module in (MeshgridNumpy(), np):
            prompts = []
            def joint(x):
                prompts.append(np.asarray(x[-1]))
                b = tf.cast(x[-1][..., ::2, ::2, 0] > .3, tf.float32)
                logits = tf.stack([10 * (1 - b), 10 * b, tf.zeros_like(b)], -1)
                return logits, logits, tf.constant([[.8]]), tf.constant([[.7]])
            with patch.object(inference, '_refinement_runners', return_value=(lambda x: [x] * 7, joint)), \
                    patch.object(inference, '_cytoplasm_refinement_runner', return_value=None), \
                    patch.object(inference, 'np', numpy_module):
                n, c, _ = inference.predict_instances_prompted(
                    SimpleNamespace(), cfg, image, ctr, np.ones(3), nuc_dists=np.random.default_rng(1).uniform(2, 6, (3, 12)),
                    nuc_labels=labels, verbose=False)
            runs.append((prompts, n, c))
        self.assertEqual(len(runs[0][0]), 3)
        for old, new in zip(runs[0][0], runs[1][0]):
            self.assertEqual(old.dtype, new.dtype); np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(runs[0][1], runs[1][1]); np.testing.assert_array_equal(runs[0][2], runs[1][2])

    def test_generator_prompts_bit_identical_to_meshgrid(self):
        import cytonucs_data_generator as generator
        from test_crop_cut_v2_17 import make_split
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp, contextlib.redirect_stdout(io.StringIO()):
            make_split(Path(tmp) / 'train')
            cfg = config(train_patch_size=(8, 32, 32), nucleus_grid=(1, 2, 2), cytoplasm_grid=(1, 2, 2),
                         prompts_per_crop=3, zoom_range=None)
            batches = []
            for numpy_module in (MeshgridNumpy(), np):
                with patch.object(generator, 'np', numpy_module):
                    gen = generator.CytoNucsDataGenerator(dataset_dir=Path(tmp) / 'train', config=cfg, rays=cfg.rays,
                                                          augmenter=None, subset='train')
                    batches.append([gen[i] for i in range(len(gen))])
        checked = 0
        for (x_old, _), (x_new, _) in zip(*batches):
            old, new = np.asarray(x_old), np.asarray(x_new)      # prompts are channels after the image
            self.assertEqual(old.dtype, new.dtype); np.testing.assert_array_equal(old, new)
            checked += int(old[..., cfg.n_channel_in:].any())
        self.assertGreater(checked, 0)

    def test_prompt_construction_faster(self):
        net, center, sigma = (16, 448, 448), np.array([7., 200., 211.]), np.array([1.3, 18.2, 17.9])
        timings = []
        for build in (lambda a: np.meshgrid(*a, indexing='ij'), lambda a: np.ix_(*a)):
            t = time.perf_counter()
            for _ in range(3):
                axes = build([np.arange(s, dtype=np.float32) for s in net])
                np.exp(-0.5 * sum(((g - center[i]) / sigma[i]) ** 2 for i, g in enumerate(axes)))
            timings.append(time.perf_counter() - t)
        self.assertLess(timings[1] * 2, timings[0])


class RoiSnapTests(unittest.TestCase):
    def test_snapped_windows_are_shared_and_keep_the_prompt_inside(self):
        cfg = config(nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1), mask_quality_head=True, proposal_mode='peaks',
                     mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                     refinement_patch_size=[0, 32, 32])
        shape = (8, 40, 40); image = np.zeros(shape + (1,), np.float32)   # ROI 32: origins 0..8
        seen = {}
        def logits(x):
            b = tf.cast(x[-1][..., 0] > .3, tf.float32)
            return tf.stack([10 * (1 - b), 10 * b, tf.zeros_like(b)], -1)
        def features(x):
            seen.setdefault('features', 0); seen['features'] += 1
            return [x] * 7
        rng = np.random.default_rng(0)
        ctr = np.concatenate([np.full((40, 1), 4), rng.integers(2, 38, (40, 2))], 1)
        labels = np.zeros(shape, np.int32)
        for k, c in enumerate(ctr, 1):
            labels[tuple(c)] = k
        counts = {}
        for snap in (False, True):
            cfg.refinement_roi_snap = snap; seen.clear()
            with patch.object(inference, '_refinement_runners',
                              return_value=(features, lambda x: (logits(x), logits(x), tf.constant([[.8]]), tf.constant([[.8]])))), \
                    patch.object(inference, '_cytoplasm_refinement_runner', return_value=None):
                inference.predict_instances_prompted(SimpleNamespace(), cfg, image, ctr, np.ones(len(ctr)),
                                                     nuc_dists=np.full((len(ctr), 12), 2.), nuc_labels=labels,
                                                     verbose=False)
            counts[snap] = seen['features']
        self.assertLess(counts[True], counts[False])
        multiple = inference.architecture_plan(cfg)['input_multiple']
        self.assertLessEqual(counts[True], np.prod([(n - s) // m + 2 for n, s, m in zip(shape[1:], (32, 32), multiple[1:])]))


class LovaszTests(unittest.TestCase):
    def test_binary_prediction_equals_one_minus_iou_and_empty_excluded(self):
        t = np.zeros((3, 4, 8, 8, 1), np.float32); t[0, :2, :4, :4] = 1; t[1, 1:3, 2:6, 2:6] = 1
        p = np.zeros_like(t); p[0, :2, :4, :2] = 1; p[0, 3, 7, 7] = 1; p[1, 1:3, 2:6, 2:6] = 1; p[2, 0, 0, 0] = 1
        iou0 = (p[0] * t[0]).sum() / ((p[0] + t[0]) > 0).sum()
        loss = float(lovasz_jaccard_loss(tf.constant(p), tf.constant(t)))
        self.assertAlmostEqual(loss, ((1 - iou0) + 0.) / 2, places=6)   # third prompt: empty target

    def test_gradient_moves_towards_target_and_config_switch(self):
        t = np.zeros((1, 2, 6, 6, 1), np.float32); t[0, :, 1:4, 1:4] = 1
        p = tf.Variable(np.full_like(t, .5))
        with tf.GradientTape() as tape:
            loss = lovasz_jaccard_loss(p, tf.constant(t))
        g = tape.gradient(loss, p).numpy()
        self.assertTrue(np.all(g[t > .5] <= 0) and np.all(g[t < .5] >= 0) and np.any(g != 0))
        cfg = config(mask_quality_head=False, prompts_per_crop=2)
        model = build_cytonucs_model(cfg); x, y = batch(cfg); pred = model.keras_model(x)
        dice = CytoNucsLoss(cfg); a = dice(y, pred, current_epoch=551)
        cfg.mask_overlap_loss = 'lovasz'
        lov = CytoNucsLoss(cfg); b = lov(y, pred, current_epoch=551)
        expected = 0.
        for head, lam in (('nucleus', cfg.lambda_nuc_mask), ('cytoplasm', cfg.lambda_cyto_mask)):
            target = dice._subsample(tf.cast(y[head + '_target'][..., None], tf.float32), getattr(cfg, head + '_grid'))
            expected += lam * .5 * (float(lovasz_jaccard_loss(pred[head + '_mask'], target))
                                    - float(dice._refinement_dice_loss(pred[head + '_mask'], target)))
        self.assertAlmostEqual(float(b - a), expected, places=5)


class MinAreaTests(unittest.TestCase):
    def test_small_objects_dropped_from_object_losses_only(self):
        full = np.zeros((1, 4, 16, 16), np.int32); full[0, :, :8, :8] = 1; full[0, 0, 12, 12] = 2
        out = without_small_objects(tf.constant(full), tf.constant(full), tf.constant(10.)).numpy()
        self.assertEqual(set(np.unique(out)) - {0}, {1})
        self.assertEqual(set(np.unique(without_small_objects(tf.constant(full), tf.constant(full),
                                                              tf.constant(0.)).numpy())) - {0}, {1, 2})
        cfg = config(nucleus_prob_object_weight=.25, prob_object_peak_weight=.25, mask_quality_head=True,
                     mask_quality_weight=.1, refinement_stage_audit=True, mask_quality_min='auto',
                     mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                     object_losses_min_area_from_gate=True)
        model = build_cytonucs_model(cfg); x, y = batch(cfg); pred = model.keras_model(x)
        labels = y['nucleus_instances'].numpy().copy(); labels[0, 0, 0, 0] = 9
        y = dict(y); y['nucleus_instances'] = tf.constant(labels)
        loss = CytoNucsLoss(cfg)
        loss(y, pred, current_epoch=551); before = float(loss.last_components['nucleus_prob_sampled_object_count'])
        self.assertEqual(sync_object_min_area(loss, {'nucleus': .2, 'nucleus_min_area': 5, 'cytoplasm': .1}), [5., 0.])
        voxel_before = float(loss.last_components['nucleus_bce_voxel'])
        loss(y, pred, current_epoch=551)
        self.assertEqual(float(loss.last_components['nucleus_prob_sampled_object_count']), before - 1)
        self.assertEqual(float(loss.last_components['nucleus_object_min_voxels']), 5.)
        self.assertAlmostEqual(float(loss.last_components['nucleus_bce_voxel']), voxel_before, places=6)
        cfg.object_losses_min_area_from_gate = False
        self.assertIsNone(sync_object_min_area(CytoNucsLoss(cfg), {'nucleus_min_area': 5}))


class ConfigTests(unittest.TestCase):
    def test_json_contract_and_validation(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            for section, keys in NEW.items():
                for key, (legacy, v3) in keys.items():
                    self.assertEqual(raw[section][key], v3 if path.name in ('config_cytonucs_v3.json', 'config_cytonucs_v3_1.json', 'config_cytonucs_v3_2.json') else legacy,
                                     (path.name, key))
        raw = json.loads((ROOT / 'config_cytonucs_v3.json').read_text())
        self.assertFalse(raw['inference']['residual_proposals'])
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['training']['resume_from_checkpoint'] = None; raw['inference']['checkpoint_path'] = None
            raw['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(raw))
            with contextlib.redirect_stdout(io.StringIO()):
                preset = CytoNucsConfig.from_json(path)
        for keys in NEW.values():
            for key, (legacy, v3) in keys.items():
                self.assertEqual(getattr(preset, key), v3, key)
                section = 'inference' if key == 'refinement_roi_snap' else 'loss_params'
                self.assertEqual(CytoNucsConfig.FIXES_SETTINGS[section][key], legacy)
        for bad in (dict(refinement_roi_snap=1), dict(mask_overlap_loss='iou'),
                    dict(object_losses_min_area_from_gate=True)):   # needs mask_quality_min='auto'
            with self.assertRaises(ValueError):
                config(**bad)


if __name__ == '__main__':
    unittest.main()
