import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from test_fixes import ROOT, config
from test_architecture_v2 import batch
from cytonucs_mask_quality import gaussian_prompt_geometry, component_quality_targets
from cytonucs_postprocess import mask_candidate, prompt_component, assemble_masks
from cytonucs_inference import match_instances
from cytonucs_config import CytoNucsConfig
from cytonucs_model import build_cytonucs_model
from cytonucs_losses import CytoNucsLoss
from cytonucs_trainer import CytoNucsTrainer


def gaussian(shape, center, sigma):
    axes = np.indices(shape, dtype=np.float32)
    return np.exp(-.5 * sum(((axes[a]-center[a])/sigma[a])**2 for a in range(len(shape)))).astype(np.float32)


class MaskQualityTargetTests(unittest.TestCase):
    def test_geometry_recovers_augmented_and_boundary_prompts(self):
        for center in [(3, 7, 8), (0, 0, 0), (3.5, 7.25, 8.8), (7, 15, 17)]:
            p = gaussian((8, 16, 18), center, (1.2, 3.4, 8.))
            c, s = gaussian_prompt_geometry(p)
            np.testing.assert_allclose(c, center, atol=2e-4)
            np.testing.assert_allclose(s, (1.2, 3.4, 8.), rtol=2e-4)
        self.assertIsNone(gaussian_prompt_geometry(np.zeros((3, 4, 4))))

    def test_raw_quality_can_rank_a_fragment_over_a_perfect_final_mask(self):
        shape = (8, 32, 32)
        target = np.zeros(shape, bool); target[2:6, 4:12, 4:12] = True
        raw = target.copy(); raw[:, 16:32, 16:32] = True
        fragment = target.copy(); fragment[:, :, :7] = False; fragment[:, :, 9:] = False
        prompt = gaussian(shape, (3, 8, 8), (2, 4, 4))
        old = np.array([np.count_nonzero(m & target)/np.count_nonzero(m | target)
                        for m in (raw, fragment)])
        new = component_quality_targets(np.array([raw, fragment], np.float32)[..., None],
                                        np.array([target, target]), np.array([prompt, prompt]), (1, 1, 1), .45)
        np.testing.assert_allclose(new, [1., .25])
        self.assertLess(old[0], old[1])
        candidates = [mask_candidate(m.astype(np.float32), prompt_component(m, (3, 8, 8), (2, 4, 4)), (0, 0, 0))
                      for m in (raw, fragment)]
        matches = []
        for quality in (old, new):
            for c, q in zip(candidates, quality): c['predicted_quality'] = float(q)
            result, _ = assemble_masks(candidates, shape, assembly_mode='object_quality')
            matches.append(len(match_instances(result, target.astype(np.int32))))
        self.assertEqual(matches, [0, 1])

    def test_full_resolution_interpolation_empty_and_wrong_owner(self):
        shape = (4, 8, 8)
        target = np.zeros(shape, bool); target[:, 2:6, 2:6] = True
        pred = target[:, ::2, ::2].astype(np.float32)[..., None]
        prompt = gaussian(shape, (1, 3, 3), (1, 2, 2))
        from cytonucs_postprocess import upsample_grid
        expected = prompt_component(upsample_grid(pred[..., 0], shape, (1, 2, 2)) > .45,
                                    (1, 3, 3), (1, 2, 2))
        iou = (expected & target).sum()/(expected | target).sum()
        result = component_quality_targets(np.array([pred, pred]), np.array([target, np.zeros_like(target)]),
                                            np.array([prompt, prompt]), (1, 2, 2), .45)
        np.testing.assert_allclose(result, [iou, 0])

    def test_preset_and_invalid_mode(self):
        raw = json.loads((ROOT / 'config_cytonucs_v2_7.json').read_text())
        self.assertEqual(raw['loss_params']['mask_quality_target'], 'component')
        self.assertTrue(raw['training']['resume_from_checkpoint'].endswith('weights_epoch_400.weights.h5'))
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['training']['resume_from_checkpoint'] = None; raw['output']['basedir'] = tmp
            path = Path(tmp)/'config.json'; path.write_text(json.dumps(raw))
            cfg = CytoNucsConfig.from_json(path)
            for section, settings in cfg.FIXES_SETTINGS.items():
                for key in settings: self.assertEqual(getattr(cfg, key), raw[section][key])
        with self.assertRaises(ValueError): config(mask_quality_target='unknown')

    def test_real_train_val_step_and_target_is_detached(self):
        cfg = config(mask_quality_head=True, mask_quality_pooling='mask', mask_quality_weight=.1,
                     mask_quality_target='component', refinement_depth=1,
                     predicted_prompt_fraction=.5, predicted_prompt_sample_peaks=True,
                     predicted_prompt_start_epoch=1)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            model = build_cytonucs_model(cfg, basedir=tmp)
            trainer = CytoNucsTrainer(model, cfg, CytoNucsLoss(cfg), 0)
            x, y = batch(cfg); x = x.numpy()
            for channel in range(1, 5): x[0, ..., channel] = gaussian((8, 32, 32), (3, 16, 16), (2, 4, 4))
            x = tf.constant(x)
            for result in (trainer.train_step(x, y), trainer.val_step(x, y)):
                self.assertTrue(np.isfinite(float(result[0])))
            path = Path(tmp) / 'weights_epoch_400.weights.h5'
            trainer.state_epoch.assign(400)
            trainer.save_checkpoint(path)
            cfg.mask_quality_target = 'grid'
            old_mode = build_cytonucs_model(cfg, basedir=tmp)
            old_mode.load_weights(path)
            restored = CytoNucsTrainer(old_mode, cfg, CytoNucsLoss(cfg), 0)
            self.assertTrue(restored.restore_training_state(path))
            for a, b in zip(trainer.optimizer.variables, restored.optimizer.variables):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            cfg.mask_quality_target = 'component'
            targets = trainer._mask_quality_inputs(x, y)
            output = model.keras_model(x)
            # A detached label must not differentiate through threshold/component.
            p = tf.Variable(output['nucleus_mask'])
            with tf.GradientTape() as tape:
                label = tf.numpy_function(lambda a: component_quality_targets(a, y['nucleus_target'].numpy(),
                    targets['nucleus_quality_prompt'].numpy(), cfg.nucleus_grid, .5), [tf.stop_gradient(p)], tf.float32)
                score = tf.reduce_sum(label)
            self.assertIsNone(tape.gradient(score, p))


if __name__ == '__main__': unittest.main()
