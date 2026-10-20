"""v3.1: semantic foreground head on both decoders; loading pre-v3.1 checkpoints with Adam state."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from cytonucs_config import CytoNucsConfig
from cytonucs_losses import CytoNucsLoss, lovasz_jaccard_loss
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer

BASE = dict(prompts_per_crop=2, mask_quality_head=True, refinement_depth=1, refinement_gradient_scale=.1)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def slot_values(optimizer):
    return {v.path.split('/', 1)[1]: v.numpy() for v in optimizer.variables}


def weighted_layers(model):
    return {l.name: l.get_weights() for l in model.keras_model.layers if l.weights}


class ModelTests(unittest.TestCase):
    def test_outputs_and_stable_optimizer_order(self):
        legacy = quiet(build_cytonucs_model, config(**BASE))
        head = quiet(build_cytonucs_model, config(semantic_foreground_weight=.5, **BASE))
        x, _ = batch(head.config)
        out, out_legacy = head.keras_model(x), legacy.keras_model(x)
        self.assertNotIn('nucleus_semantic_logit', out_legacy)
        for h in ('nucleus', 'cytoplasm'):
            self.assertEqual(out[h + '_semantic_logit'].shape, out[h + '_prob'].shape)
        paths = [v.path for v in head.optimizer_variables()]
        self.assertEqual(paths[:-4], [v.path for v in legacy.keras_model.trainable_variables])
        self.assertTrue(all('_semantic_logit/' in p for p in paths[-4:]))
        self.assertTrue(head.architecture_signature()['semantic_foreground_head'])
        self.assertFalse(legacy.architecture_signature()['semantic_foreground_head'])


class ResumeTests(unittest.TestCase):
    def test_pre_v3_1_checkpoint_weights_and_adam_restore_exactly(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            old_cfg = config(**BASE); new_cfg = config(semantic_foreground_weight=1., **BASE)
            old = quiet(build_cytonucs_model, old_cfg, basedir=tmp)
            trainer = CytoNucsTrainer(old, old_cfg, CytoNucsLoss(old_cfg), 0)
            x, y = batch(old_cfg); trainer.current_epoch_var.assign(566)
            quiet(trainer.train_step, x, y); quiet(trainer.train_step, x, y)
            trainer.state_epoch.assign(570); path = Path(tmp) / 'weights_epoch_570.weights.h5'
            quiet(trainer.save_checkpoint, path)

            new = quiet(build_cytonucs_model, new_cfg, basedir=tmp)
            head_init = new.keras_model.get_layer('nucleus_semantic_logit').get_weights()
            quiet(new.load_weights, path)
            t2 = CytoNucsTrainer(new, new_cfg, CytoNucsLoss(new_cfg), 0)
            self.assertTrue(quiet(t2.restore_training_state, path))
            self.assertEqual(int(t2.state_epoch.numpy()), 570)
            # Every pre-existing layer equals the checkpoint after weights AND state restore.
            restored = weighted_layers(new)
            for name, weights in weighted_layers(old).items():
                for a, b in zip(weights, restored[name]):
                    np.testing.assert_array_equal(a, b, err_msg=name)
            for a, b in zip(head_init, new.keras_model.get_layer('nucleus_semantic_logit').get_weights()):
                np.testing.assert_array_equal(a, b)
            old_slots, new_slots = slot_values(trainer.optimizer), slot_values(t2.optimizer)
            for key, value in old_slots.items():
                np.testing.assert_array_equal(value, new_slots[key], err_msg=key)
            semantic = [k for k in new_slots if '_semantic_logit_' in k]
            self.assertEqual(len(semantic), 8)
            self.assertTrue(all(not new_slots[k].any() for k in semantic))
            self.assertTrue(any(old_slots[k].any() for k in old_slots if 'momentum' in k))

            # Training continues; a v3.1 checkpoint resumes into v3.1 exactly.
            t2.current_epoch_var.assign(571); result = quiet(t2.train_step, x, y)
            self.assertIn('semantic_foreground', result[-1])
            self.assertTrue(any(new_slots_value.any() for k, new_slots_value in slot_values(t2.optimizer).items() if k in semantic))
            t2.state_epoch.assign(575); path2 = Path(tmp) / 'weights_epoch_575.weights.h5'; quiet(t2.save_checkpoint, path2)
            third = quiet(build_cytonucs_model, new_cfg, basedir=tmp); quiet(third.load_weights, path2)
            t3 = CytoNucsTrainer(third, new_cfg, CytoNucsLoss(new_cfg), 0)
            self.assertTrue(quiet(t3.restore_training_state, path2))
            for (k, a), (_, b) in zip(slot_values(t2.optimizer).items(), slot_values(t3.optimizer).items()):
                np.testing.assert_array_equal(a, b, err_msg=k)
            for a, b in zip(new.keras_model.weights, third.keras_model.weights):
                np.testing.assert_array_equal(a.numpy(), b.numpy())

            # A checkpoint with the head cannot be loaded into a config without it.
            back = quiet(build_cytonucs_model, old_cfg, basedir=tmp)
            with self.assertRaises(ValueError):
                back.load_weights(path2)


class LossTests(unittest.TestCase):
    def test_value_total_and_gradient_path(self):
        for overlap in ('dice', 'lovasz'):
            cfg = config(semantic_foreground_weight=.5, mask_overlap_loss=overlap, **BASE)
            model = quiet(build_cytonucs_model, cfg); x, y = batch(cfg)
            with tf.GradientTape(persistent=True) as tape:
                pred = model.keras_model(x, training=True)
                loss = CytoNucsLoss(cfg); total = loss(y, pred, current_epoch=571)
                semantic = loss.last_components['semantic_foreground']
            expected = 0.
            for head in ('nucleus', 'cytoplasm'):
                g = getattr(cfg, head + '_grid')
                target = tf.cast(y[head + '_instances'] > 0, tf.float32)[:, ::g[0], ::g[1], ::g[2], None]
                logit = pred[head + '_semantic_logit']
                bce = float(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logit)))
                p = tf.sigmoid(logit)
                ov = float(lovasz_jaccard_loss(p, target)) if overlap == 'lovasz' else float(loss._refinement_dice_loss(p, target))
                self.assertAlmostEqual(float(loss.last_components[head + '_semantic_bce']), bce, places=5)
                self.assertIn(head + '_semantic_dice', loss.last_components)
                expected += .5 * .5 * (.5 * bce + .5 * ov)
            self.assertAlmostEqual(float(semantic), expected, places=5)
            off = copy.copy(cfg); off.semantic_foreground_weight = 0.
            plain = CytoNucsLoss(off); total_off = plain(y, pred, current_epoch=571)
            self.assertNotIn('semantic_foreground', plain.last_components)
            self.assertAlmostEqual(float(total - total_off), float(semantic), places=4)
            grads = dict(zip([v.path for v in model.keras_model.trainable_variables],
                             tape.gradient(semantic, model.keras_model.trainable_variables)))
            nonzero = lambda prefix: any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0
                                         for p, g in grads.items() if p.startswith(prefix))
            for prefix in ('nuc_dec_l0', 'cyto_dec_l0', 'enc_l0', 'nucleus_semantic_logit', 'cytoplasm_semantic_logit'):
                self.assertTrue(nonzero(prefix), (overlap, prefix))
            for prefix in ('nuc_refine', 'cyto_refine', 'nucleus_prob_logit', 'nucleus_dist', 'nucleus_mask_out'):
                self.assertFalse(nonzero(prefix), (overlap, prefix))


class ConfigTests(unittest.TestCase):
    def test_json_contract_and_validation(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            self.assertEqual(raw['loss_params']['semantic_foreground_weight'],
                             1. if path.name in ('config_cytonucs_v3_1.json', 'config_cytonucs_v3_2.json', 'config_cytonucs_v3_3.json', 'config_cytonucs_v3_4.json', 'config_cytonucs_v4.json', 'config_cytonucs_v5.json') else 0., path.name)
        self.assertEqual(CytoNucsConfig.FIXES_SETTINGS['loss_params']['semantic_foreground_weight'], 0.)
        raw = json.loads((ROOT / 'config_cytonucs_v3_1.json').read_text())
        v3 = json.loads((ROOT / 'config_cytonucs_v3.json').read_text())
        for section in raw:
            if isinstance(raw[section], dict):
                changed = {k for k in raw[section] if raw[section][k] != v3[section].get(k)}
                allowed = dict(experiment={'name'}, training={'resume_from_checkpoint', 'train_epochs'},
                               inference={'checkpoint_path'}, loss_params={'semantic_foreground_weight'})
                self.assertLessEqual(changed, allowed.get(section, set()), section)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['training']['resume_from_checkpoint'] = None; raw['inference']['checkpoint_path'] = None
            raw['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(raw))
            preset = quiet(CytoNucsConfig.from_json, path)
        self.assertEqual(preset.semantic_foreground_weight, 1.)
        for bad in (-1., True, float('nan')):
            with self.assertRaises(ValueError):
                config(semantic_foreground_weight=bad)


if __name__ == '__main__':
    unittest.main()
