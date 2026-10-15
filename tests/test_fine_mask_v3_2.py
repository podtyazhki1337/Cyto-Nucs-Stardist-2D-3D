"""v3.2: full-resolution prompted mask head (model.refinement_full_resolution)."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from cytonucs_config import CytoNucsConfig
from cytonucs_fine_mask import (upsample_linear, pad_last_node, prompt_window_begin, training_window,
                                inference_window, crop)
from cytonucs_losses import CytoNucsLoss, lovasz_jaccard_loss
from cytonucs_model import build_cytonucs_model
from cytonucs_postprocess import upsample_grid, prompt_component
from cytonucs_trainer import CytoNucsTrainer
import cytonucs_inference as inference

BASE = dict(nucleus_grid=(1, 2, 2), cytoplasm_grid=(1, 2, 2), prompts_per_crop=2, mask_quality_head=True,
            refinement_depth=1, refinement_gradient_scale=.1)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def slot_values(optimizer):
    return {v.path.split('/', 1)[1]: v.numpy() for v in optimizer.variables}


class UpsampleWindowTests(unittest.TestCase):
    def test_linear_upsampling_equals_upsample_grid(self):
        rng = np.random.default_rng(0)
        for grid in ((1, 2, 2), (2, 2, 2), (1, 3, 3), (1, 1, 1)):
            nodes = (5, 7, 6)
            a = rng.random(nodes).astype(np.float32)
            shape = tuple(n * g for n, g in zip(nodes, grid))
            expected = upsample_grid(a, shape, grid)
            got = upsample_linear(pad_last_node(tf.constant(a)[None, ..., None], grid), grid)[0, ..., 0].numpy()
            self.assertEqual(got.shape, shape)
            np.testing.assert_allclose(got, expected, atol=1e-6)
            # A grid-aligned window with one extra node reproduces the same values.
            start = [g * 1 for g in grid]
            stop = [s - g for s, g in zip(shape, grid)]
            node_start = [s // g for s, g in zip(start, grid)]
            node_stop = [e // g + (1 if g > 1 else 0) for e, g in zip(stop, grid)]
            sub = a[tuple(slice(s, e) for s, e in zip(node_start, node_stop))]
            window = upsample_linear(tf.constant(sub)[None, ..., None], grid)[0, ..., 0].numpy()
            np.testing.assert_allclose(window, expected[tuple(slice(s, e) for s, e in zip(start, stop))], atol=1e-6)

    def test_prompt_window_begin_and_crop(self):
        grid = (1, 2, 2); window = (4, 10, 10)
        prompt = np.zeros((3, 6, 16, 20, 1), np.float32)
        centres = [(0, 0, 0), (5, 15, 19), (3, 7, 9)]
        for i, c in enumerate(centres):
            prompt[(i,) + c] = 1.
        begin = prompt_window_begin(tf.constant(prompt), window, grid).numpy()
        for (c, b) in zip(centres, begin):
            for a in range(3):
                self.assertEqual(b[a] % grid[a], 0)
                self.assertTrue(0 <= b[a] <= prompt.shape[a + 1] - window[a])
                self.assertTrue(b[a] <= c[a] < b[a] + window[a])
        cut = crop(tf.constant(prompt), tf.constant(begin), tf.constant(window)).numpy()
        self.assertEqual(cut.shape, (3,) + window + (1,))
        self.assertEqual(cut.sum(), 3.)
        # Windows larger than the input are clipped to it.
        small = prompt_window_begin(tf.constant(prompt[:, :, :8, :8]), (4, 12, 12), grid).numpy()
        self.assertTrue((small[:, 1:] == 0).all())

    def test_windows_from_geometry(self):
        cfg = config(**BASE)
        for head in ('nucleus', 'cytoplasm'):
            w = training_window(cfg, head)
            self.assertTrue(all(0 < x <= s and x % g == 0 for x, s, g in zip(w, cfg.train_patch_size, cfg.nucleus_grid)))
        start, stop = inference_window((1, 5, 7), (3, 6, 6), (8, 32, 32), (8, 32, 32), (1, 2, 2), (2, 4, 4))
        self.assertTrue(all(s % g == 0 and e % g == 0 for s, e, g in zip(start, stop, (1, 2, 2))))
        self.assertTrue(all(s <= l and e >= l + n for s, e, l, n in zip(start, stop, (1, 5, 7), (3, 6, 6))))


class ModelLossTests(unittest.TestCase):
    def test_model_outputs_start_from_coarse_and_order(self):
        cfg = config(refinement_full_resolution=True, semantic_foreground_weight=.5, **BASE)
        model = quiet(build_cytonucs_model, cfg); x, _ = batch(cfg)
        out = model.keras_model(x)
        shape = tuple(cfg.train_patch_size)
        for head in ('nucleus', 'cytoplasm'):
            logits, begins = out[head + '_fine_logit'].numpy(), out[head + '_fine_begin'].numpy()
            self.assertEqual(logits.shape[1:4], training_window(cfg, head))
            coarse = out[head + '_mask_logit'].numpy()
            for k in range(len(logits)):
                full = np.stack([upsample_grid(coarse[k, ..., c], shape, cfg.nucleus_grid) for c in range(3)], -1)
                sl = tuple(slice(int(b), int(b) + n) for b, n in zip(begins[k], logits.shape[1:4]))
                np.testing.assert_allclose(logits[k], full[sl], atol=1e-5)
        paths = [v.path for v in model.optimizer_variables()]
        fine = [i for i, p in enumerate(paths) if '_fine_' in p]
        semantic = [i for i, p in enumerate(paths) if '_semantic_logit' in p]
        self.assertEqual(fine, list(range(len(paths) - len(fine), len(paths))))
        self.assertEqual(semantic, list(range(fine[0] - 4, fine[0])))
        self.assertTrue(model.architecture_signature()['refinement_full_resolution'])
        legacy = quiet(build_cytonucs_model, config(**BASE))
        self.assertNotIn('nucleus_fine_logit', legacy.keras_model(x))

    def test_loss_value_and_gradient_path(self):
        cfg = config(refinement_full_resolution=True, mask_ce_class_balance=False, mask_overlap_loss='lovasz', **BASE)
        tf.keras.utils.set_random_seed(0)      # the tiny test network can otherwise start with dead ReLUs
        model = quiet(build_cytonucs_model, cfg); x, y = batch(cfg)
        # The zero-initialised output passes gradient to the earlier fine layers only after its
        # first update; start from a trained-like output to test the whole path.
        rng = np.random.default_rng(0)
        for head_model in model.fine_head_models.values():
            delta = head_model.get_layer(head_model.name + '_delta')
            kernel, bias = delta.get_weights()
            delta.set_weights([rng.normal(0, .05, kernel.shape).astype(np.float32), bias])
        with tf.GradientTape() as tape:
            pred = model.keras_model(x, training=True)
            loss = CytoNucsLoss(cfg); total = loss(y, pred, current_epoch=600)
            fine = loss.last_components['nucleus_fine_mask'] + loss.last_components['cytoplasm_fine_mask']
        for head, weight in (('nucleus', cfg.lambda_nuc_mask), ('cytoplasm', cfg.lambda_cyto_mask)):
            logit = pred[head + '_fine_logit']; begin = pred[head + '_fine_begin']
            size = tf.shape(logit)[1:4]
            self.assertEqual(logit.shape[0], 1)                  # one prompt slot of the single crop
            slot = int(pred['fine_prompt_slot'][0])
            target = crop(tf.cast(y[head + '_target'], tf.float32)[slot:slot + 1, ..., None], begin, size)
            inst = crop(tf.cast(y[head + '_instances'] > 0, tf.float32)[..., None], begin, size)
            expected = weight * (0.5 * loss._refinement_ce_loss(logit, target, inst) +
                                 0.5 * lovasz_jaccard_loss(tf.nn.softmax(logit)[..., 1:2], target))
            self.assertAlmostEqual(float(loss.last_components[head + '_fine_mask']), float(expected), places=5)
        off = copy.copy(cfg); off.refinement_full_resolution = False
        plain = CytoNucsLoss(off); total_off = plain(y, pred, current_epoch=600)
        self.assertAlmostEqual(float(total - total_off), float(fine), places=4)
        grads = dict(zip([v.path for v in model.keras_model.trainable_variables],
                         tape.gradient(fine, model.keras_model.trainable_variables)))
        nonzero = lambda prefix: any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0
                                     for p, g in grads.items() if p.startswith(prefix))
        for prefix in ('nuc_fine_head_delta', 'cyto_fine_head_delta', 'nuc_fine_head_feature_proj', 'nuc_refine'):
            self.assertTrue(nonzero(prefix), prefix)
        for prefix in ('nucleus_prob_logit', 'nucleus_dist', 'cytoplasm_dist'):
            self.assertFalse(nonzero(prefix), prefix)

    def test_no_direct_gradient_into_full_resolution_features(self):
        # Memory (server OOM in Conv3DBackpropInput at 32x16x448x448): the fine head reads the
        # first backbone features without gradient. With the coarse refinement path detached
        # (gradient scale 0) nothing of the fine loss may reach the backbone.
        cfg = config(refinement_full_resolution=True, **dict(BASE, refinement_gradient_scale=0.))
        tf.keras.utils.set_random_seed(0)
        model = quiet(build_cytonucs_model, cfg); x, y = batch(cfg)
        self.assertIsNotNone(model.keras_model.get_layer('fine_features_stop_gradient'))
        rng = np.random.default_rng(1)
        for head_model in model.fine_head_models.values():
            delta = head_model.get_layer(head_model.name + '_delta')
            kernel, bias = delta.get_weights()
            delta.set_weights([rng.normal(0, .05, kernel.shape).astype(np.float32), bias])
        with tf.GradientTape() as tape:
            pred = model.keras_model(x, training=True)
            loss = CytoNucsLoss(cfg); loss(y, pred, current_epoch=600)
            fine = loss.last_components['nucleus_fine_mask'] + loss.last_components['cytoplasm_fine_mask']
        grads = dict(zip([v.path for v in model.keras_model.trainable_variables],
                         tape.gradient(fine, model.keras_model.trainable_variables)))
        for path, g in grads.items():
            if path.startswith(('pre_pool', 'enc_', 'nuc_dec', 'cyto_dec')):
                self.assertTrue(g is None or float(tf.reduce_max(tf.abs(g))) == 0, path)
        self.assertTrue(any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0
                            for p, g in grads.items() if p.startswith('nuc_fine_head_feature_proj')))


class PromptSlotTests(unittest.TestCase):
    def test_training_picks_every_slot_and_eval_slot_zero(self):
        # The fine head must also learn from detector-derived prompts (last slots), which is
        # where inference puts prompts of objects cut by the patch face.
        cfg = config(refinement_full_resolution=True, **dict(BASE, prompts_per_crop=3))
        model = quiet(build_cytonucs_model, cfg); x, _ = batch(cfg)
        x = x.numpy(); x[..., cfg.n_channel_in:] = np.random.default_rng(0).random(
            x[..., cfg.n_channel_in:].shape).astype(np.float32); x = tf.constant(x)
        shape = tuple(cfg.train_patch_size); seen = set()
        for _ in range(15):
            out = model.keras_model(x, training=True)
            k = int(out['fine_prompt_slot'][0]); seen.add(k)
            for head in ('nucleus', 'cytoplasm'):
                coarse = out[head + '_mask_logit'].numpy()[k]
                full = np.stack([upsample_grid(coarse[..., c], shape, cfg.nucleus_grid) for c in range(3)], -1)
                begin = out[head + '_fine_begin'].numpy()[0]; fine = out[head + '_fine_logit'].numpy()[0]
                sl = tuple(slice(int(b), int(b) + n) for b, n in zip(begin, fine.shape[:3]))
                np.testing.assert_allclose(fine, full[sl], atol=1e-5)
        self.assertEqual(seen, {0, 1, 2})
        self.assertEqual(int(model.keras_model(x, training=False)['fine_prompt_slot'][0]), 0)


class InferenceTests(unittest.TestCase):
    def test_fine_candidates_equal_full_roi_reference(self):
        cfg = config(refinement_full_resolution=True, proposal_mode='peaks', mask_assembly_nucleus='object_quality',
                     mask_assembly_cytoplasm='object_quality', mask_quality_use_scores=False, **BASE)
        model = quiet(build_cytonucs_model, cfg)
        shape = (8, 32, 32); image = np.zeros(shape + (cfg.n_channel_in,), np.float32)   # ROI = volume
        store = {}

        def logits(prompt):
            b = tf.cast(prompt[..., ::2, ::2, 0] > .3, tf.float32)
            out = tf.stack([6 * (1 - b), 6 * b - 1, tf.zeros_like(b)], -1)
            store.setdefault('coarse', []).append(out.numpy()); store.setdefault('prompt', []).append(prompt.numpy())
            return out

        def refine(xs):
            prompt = xs[-1]; lg = logits(prompt); h = tf.zeros(lg.shape[:-1] + (8,))
            return [lg, lg, tf.constant([[.8]]), tf.constant([[.7]]), h, h]

        channels = model.fine_head_models['nucleus'].inputs[2].shape[-1]

        def fine_features(x):
            return tf.zeros(x.shape[:-1] + (channels,))
        heads = {h: (lambda net: (lambda xs: net(xs)))(m) for h, m in model.fine_head_models.items()}
        runners = (lambda x: [x] * 7, fine_features, refine, None, heads)
        ctr = np.array([[4, 10, 10], [3, 20, 24], [5, 12, 26]]); labels = np.zeros(shape, np.int32)
        for k, c in enumerate(ctr, 1):
            labels[tuple(c)] = k
        sink = {}
        with patch.object(inference, '_fine_refinement_runners', return_value=runners):
            quiet(inference.predict_instances_prompted, SimpleNamespace(), cfg, image, ctr, np.ones(3),
                  nuc_dists=np.full((3, 12), 3.), nuc_labels=labels, verbose=False, candidate_sink=sink,
                  mask_thresh=.45)
        self.assertEqual(len(sink['nucleus']), 3)
        for candidate, coarse, prompt in zip(sink['nucleus'], store['coarse'], store['prompt']):
            self.assertIsNotNone(candidate)
            # Zero-initialised residual: softmax of the linearly upsampled coarse logits.
            up = np.stack([upsample_grid(coarse[0, ..., c], shape, cfg.nucleus_grid) for c in range(3)], -1)
            prob = tf.nn.softmax(up).numpy()[..., 1]
            centre = np.array(np.unravel_index(np.argmax(prompt[0, ..., 0]), shape))
            reference = prompt_component(prob > .45, centre, candidate['prompt_sigma'])
            placed = np.zeros(shape, bool)
            placed[tuple(slice(int(l), int(l) + n) for l, n in zip(candidate['lo'], candidate['mask'].shape))] = candidate['mask']
            np.testing.assert_array_equal(placed, reference)


class ResumeTests(unittest.TestCase):
    def test_v3_1_checkpoint_into_v3_2_weights_and_adam_exact(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            old_cfg = config(semantic_foreground_weight=1., **BASE)
            new_cfg = config(semantic_foreground_weight=1., refinement_full_resolution=True, mask_ce_class_balance=False, **BASE)
            old = quiet(build_cytonucs_model, old_cfg, basedir=tmp)
            trainer = CytoNucsTrainer(old, old_cfg, CytoNucsLoss(old_cfg), 0)
            x, y = batch(old_cfg); trainer.current_epoch_var.assign(575)
            quiet(trainer.train_step, x, y); quiet(trainer.train_step, x, y)
            trainer.state_epoch.assign(580); path = Path(tmp) / 'weights_epoch_580.weights.h5'
            quiet(trainer.save_checkpoint, path)
            new = quiet(build_cytonucs_model, new_cfg, basedir=tmp); quiet(new.load_weights, path)
            t2 = CytoNucsTrainer(new, new_cfg, CytoNucsLoss(new_cfg), 0)
            self.assertTrue(quiet(t2.restore_training_state, path))
            restored = {l.name: l.get_weights() for l in new.keras_model.layers if l.weights}
            for layer in old.keras_model.layers:
                for a, b in zip(layer.get_weights(), restored.get(layer.name, [])):
                    np.testing.assert_array_equal(a, b, err_msg=layer.name)
            old_slots, new_slots = slot_values(trainer.optimizer), slot_values(t2.optimizer)
            for key, value in old_slots.items():
                np.testing.assert_array_equal(value, new_slots[key], err_msg=key)
            fine_slots = [k for k in new_slots if '_fine_' in k]
            self.assertTrue(fine_slots and all(not new_slots[k].any() for k in fine_slots))
            t2.current_epoch_var.assign(581); result = quiet(t2.train_step, x, y)
            self.assertIn('nucleus_fine_mask', result[-1])
            back = quiet(build_cytonucs_model, old_cfg, basedir=tmp)
            t2.state_epoch.assign(585); path2 = Path(tmp) / 'weights_epoch_585.weights.h5'; quiet(t2.save_checkpoint, path2)
            with self.assertRaises(ValueError):
                back.load_weights(path2)


V3_2_HEAD_ORDER = [
    'cytoplasm_semantic_logit/kernel', 'cytoplasm_semantic_logit/bias', 'nucleus_semantic_logit/kernel',
    'nucleus_semantic_logit/bias', 'cyto_fine_hidden_proj/kernel', 'cyto_fine_hidden_proj/bias',
    'nuc_fine_hidden_proj/kernel', 'nuc_fine_hidden_proj/bias', 'cyto_fine_head_feature_proj/kernel',
    'cyto_fine_head_feature_proj/bias', 'cyto_fine_head_c1/kernel', 'cyto_fine_head_c1/bias', 'cyto_fine_head_c2/kernel',
    'cyto_fine_head_c2/bias', 'cyto_fine_head_delta/kernel', 'cyto_fine_head_delta/bias',
    'nuc_fine_head_feature_proj/kernel', 'nuc_fine_head_feature_proj/bias', 'nuc_fine_head_c1/kernel',
    'nuc_fine_head_c1/bias', 'nuc_fine_head_c2/kernel', 'nuc_fine_head_c2/bias', 'nuc_fine_head_delta/kernel',
    'nuc_fine_head_delta/bias']


class OptimizerOrderTests(unittest.TestCase):
    def test_canonical_order_matches_saved_checkpoints(self):
        # Contract with the checkpoints of v3.1/v3.2: base in the Keras order of the graph without
        # the heads, then the heads in the order printed by the v3.2 code (audit snapshot).
        cfg = config(semantic_foreground_weight=1., refinement_full_resolution=True, **dict(BASE, prompts_per_crop=3))
        model = quiet(build_cytonucs_model, cfg)
        legacy = quiet(build_cytonucs_model, config(**dict(BASE, prompts_per_crop=3)))
        paths = [v.path for v in model.optimizer_variables()]
        base = [v.path for v in legacy.keras_model.trainable_variables]
        self.assertEqual(paths[:len(base)], base)
        self.assertEqual(paths[len(base):], V3_2_HEAD_ORDER)
        self.assertNotEqual([v.path for v in model.keras_model.trainable_variables][:len(base)], base)   # Keras reorders

    def test_restore_refuses_a_different_order(self):
        from unittest.mock import patch as mock_patch
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = config(refinement_full_resolution=True, **BASE)
            model = quiet(build_cytonucs_model, cfg, basedir=tmp)
            trainer = CytoNucsTrainer(model, cfg, CytoNucsLoss(cfg), 0)
            x, y = batch(cfg); quiet(trainer.train_step, x, y)
            path = Path(tmp) / 'weights_epoch_001.weights.h5'; quiet(trainer.save_checkpoint, path)
            second = quiet(build_cytonucs_model, cfg, basedir=tmp); quiet(second.load_weights, path)
            order = second.optimizer_variables()
            swapped = [order[1], order[0]] + order[2:]            # two variables with different values
            with mock_patch.object(type(second), 'optimizer_variables', lambda self: swapped):
                t2 = CytoNucsTrainer(second, cfg, CytoNucsLoss(cfg), 0)
                with self.assertRaises(ValueError):
                    quiet(t2.restore_training_state, path)


class WeightFileTests(unittest.TestCase):
    """Keras .weights.h5 addresses layers by generic names in graph order, so a graph change that
    reorders layers (the fine prompt slot did) would load weights into the wrong layers."""

    def weighted(self, model):
        return {path: [w.copy() for w in layer.get_weights()] for path, layer in model._weighted_layers()}

    def test_new_checkpoints_load_by_layer_name(self):
        import h5py
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = config(semantic_foreground_weight=1., refinement_full_resolution=True, **BASE)
            tf.keras.utils.set_random_seed(3); first = quiet(build_cytonucs_model, cfg, basedir=tmp)
            path = Path(tmp) / 'w.weights.h5'; first.save_weights(path)
            with h5py.File(path, 'r') as store:
                self.assertIn(first.BY_NAME, store)
            tf.keras.utils.set_random_seed(4); second = quiet(build_cytonucs_model, cfg, basedir=tmp)
            quiet(second.load_weights, path)
            saved, loaded = self.weighted(first), self.weighted(second)
            self.assertEqual(saved.keys(), loaded.keys())
            for key in saved:
                for a, b in zip(saved[key], loaded[key]):
                    np.testing.assert_array_equal(a, b, err_msg=key)
            second.keras_model.load_weights(path)        # the plain Keras reader ignores the extra group

    def test_v3_2_checkpoint_without_names_loads_through_its_graph(self):
        import h5py
        from cytonucs_model import CytoNucsStarDistModel
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = config(semantic_foreground_weight=1., refinement_full_resolution=True, **BASE)
            tf.keras.utils.set_random_seed(5)
            old = quiet(CytoNucsStarDistModel, cfg, basedir=tmp, fine_wiring='slot0')
            path = Path(tmp) / 'w.weights.h5'; old.save_weights(path)
            with h5py.File(path, 'a') as store:           # a v3.2 file: no name keys, no wiring field
                del store[old.BY_NAME]
            sidecar = Path(str(path) + '.architecture.json')
            signature = json.loads(sidecar.read_text()); signature.pop('fine_wiring'); sidecar.write_text(json.dumps(signature))
            new = quiet(build_cytonucs_model, cfg, basedir=tmp)
            self.assertNotEqual([l.name for l in old.keras_model.layers], [l.name for l in new.keras_model.layers])
            quiet(new.load_weights, path)
            saved, loaded = self.weighted(old), self.weighted(new)
            for key in saved:
                for a, b in zip(saved[key], loaded[key]):
                    np.testing.assert_array_equal(a, b, err_msg=key)


class ConfigTests(unittest.TestCase):
    def test_json_contract_and_validation(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            self.assertIs(raw['model']['refinement_full_resolution'], path.name in ('config_cytonucs_v3_2.json', 'config_cytonucs_v3_3.json', 'config_cytonucs_v3_4.json', 'config_cytonucs_v4.json', 'config_cytonucs_v5.json'), path.name)
        self.assertIs(CytoNucsConfig.FIXES_SETTINGS['model']['refinement_full_resolution'], False)
        raw = json.loads((ROOT / 'config_cytonucs_v3_2.json').read_text())
        base = json.loads((ROOT / 'config_cytonucs_v3_1.json').read_text())
        allowed = dict(experiment={'name'}, training={'resume_from_checkpoint', 'train_epochs', 'reset_best_on_resume'},
                       inference={'checkpoint_path'}, model={'refinement_full_resolution'},
                       loss_params={'mask_ce_class_balance'})
        for section in raw:
            if isinstance(raw[section], dict):
                changed = {k for k in raw[section] if raw[section][k] != base[section].get(k)}
                self.assertLessEqual(changed, allowed.get(section, set()), section)
        self.assertIs(raw['loss_params']['mask_ce_class_balance'], False)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['training']['resume_from_checkpoint'] = None; raw['inference']['checkpoint_path'] = None
            raw['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(raw))
            preset = quiet(CytoNucsConfig.from_json, path)
        self.assertIs(preset.refinement_full_resolution, True)
        self.assertIs(preset.mask_ce_class_balance, False)
        with self.assertRaises(ValueError):
            config(refinement_full_resolution=1)
        with self.assertRaises(ValueError):
            config(refinement_full_resolution=True, nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1))


if __name__ == '__main__':
    unittest.main()
