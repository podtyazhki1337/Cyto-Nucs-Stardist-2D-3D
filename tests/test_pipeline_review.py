"""Cross-stage checks for the combined fixes, including a real TIFF/train/resume path."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import tifffile

from test_fixes import ROOT, config
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import CytoNucsDataGenerator, cytonucs_augmenter_3d
from cytonucs_runtime import align_training_history, restored_best_metric
from cytonucs_losses import CytoNucsLoss
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer
from cytonucs_inference import CytoNucsPredictor, predict_instances_prompted, compute_jtpr
from cytonucs_refinement_audit import RefinementStageAudit


def make_dataset(root):
    for folder in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
        (root / folder).mkdir(parents=True)
    n = np.zeros((8, 32, 32), np.uint16)
    c = np.zeros_like(n)
    n[2:6, 6:10, 6:10] = 7
    n[1:7, 22:26, 22:26] = 29
    c[1:7, 2:14, 2:14] = 11
    c[:, 18:30, 18:30] = 37
    for i in range(3):
        image = np.random.default_rng(i).normal(size=n.shape).astype(np.float32) + (n > 0)
        for folder, arr in [('images', image), ('nucleus_masks', n), ('cytoplasm_masks', c)]:
            tifffile.imwrite(root / folder / f'{i}.tif', arr, photometric='minisblack')
        (root / 'assignments' / f'{i}.json').write_text(json.dumps({7: 11, 29: 37}))


class PipelineReviewTests(unittest.TestCase):
    def test_epoch_checkpoint_contains_callback_best_and_missing_history_slots(self):
        # Exercise epoch control flow independently of neural-network convergence.
        cfg = config(save_every_n_epochs=1)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            saved = {}
            state_best = tf.Variable(float('-inf'))
            metric = tf.keras.metrics.Mean()
            def save(path):
                saved[str(path) + '.state'] = float(state_best.numpy())
            def train_step(x, y):
                metric.update_state(1.)
                return (tf.constant(1.),) * 6 + ({},)
            stub = SimpleNamespace(config=cfg, history={'train_loss': [], 'val_loss': [], 'lr': []}, model=SimpleNamespace(name='test'),
                state_best_val=tf.Variable(float('inf')), state_best_full=state_best,
                current_epoch_var=tf.Variable(0), state_epoch=tf.Variable(0),
                train_loss_metric=metric, val_loss_metric=tf.keras.metrics.Mean(), current_step=0,
                train_step=train_step, optimizer=SimpleNamespace(learning_rate=tf.Variable(.001)),
                checkpoint_dir=Path(tmp), save_checkpoint=save,
                checkpoint=SimpleNamespace(write=lambda path: saved.update({path: float(state_best.numpy())})))
            class Gen:
                def __len__(self): return 1
                def __getitem__(self, index): return np.zeros((1, 1), np.float32), {}
                def set_epoch(self, epoch): pass
            cb = SimpleNamespace(on_epoch_end=lambda epoch, history: state_best.assign(.75))
            CytoNucsTrainer.train(stub, Gen(), epochs=6, start_epoch=5, steps_per_epoch=1, callbacks=[cb])
            self.assertEqual(saved[str(Path(tmp) / 'weights_epoch_006.weights.h5') + '.state'], .75)
            self.assertEqual(stub.history['train_loss'], [None] * 5 + [1.])

    def test_different_head_grids_real_training(self):
        from test_architecture_v2 import batch
        for ng, cg in [((2, 4, 4), (1, 2, 2)), ((1, 2, 2), (2, 4, 4))]:
            with self.subTest(nucleus_grid=ng, cytoplasm_grid=cg):
                cfg = config(nucleus_grid=ng, cytoplasm_grid=cg, refinement_depth=1,
                             refinement_gradient_scale=.1, mask_quality_head=True,
                             mask_quality_pooling='mask', mask_quality_weight=.1,
                             predicted_prompt_fraction=.25, predicted_prompt_start_epoch=1,
                             predicted_prompt_sample_peaks=True, nucleus_prob_grid_normalize=True)
                with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
                    model = build_cytonucs_model(cfg, basedir=tmp)
                    x, y = batch(cfg)
                    trainer = CytoNucsTrainer(model, cfg, CytoNucsLoss(cfg), 0)
                    result = trainer.train_step(x, y)
                    self.assertTrue(np.isfinite(float(result[0])))
                    for h, grid in [('nucleus', ng), ('cytoplasm', cg)]:
                        self.assertEqual(tuple(model.keras_model(x)[h + '_prob'].shape[1:4]),
                                         tuple(s // g for s, g in zip(cfg.train_patch_size, grid)))

    def test_partial_missing_and_future_history(self):
        h = align_training_history({'train_loss': [3.], 'fullval_epoch': [1, 3, 5],
                                    'fullval_metric': [.2, .4, .6], 'fullval_new': [7]}, 3)
        self.assertEqual(h['train_loss'], [3., None, None])
        self.assertEqual(h['val_loss'], [None] * 3)
        self.assertEqual(h['fullval_metric'], [.2, .4])
        self.assertEqual(h['fullval_new'], [7, None])
        self.assertEqual(align_training_history({}, 4)['lr'], [None] * 4)
        self.assertEqual(restored_best_metric(h, .7), .7)
        self.assertEqual(restored_best_metric({'fullval_metric': [None, .8]}, .7), .8)

    def test_pool_resume_and_documented_all_sentinel(self):
        cfg = config(repool_freq=10)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            make_dataset(root)
            a = CytoNucsDataGenerator(root, cfg, cfg.rays, pool_size=1, repool_freq=3)
            a.set_epoch(3)
            a.set_epoch(5)
            b = CytoNucsDataGenerator(root, cfg, cfg.rays, pool_size=1, repool_freq=3)
            b.set_epoch(5)
            self.assertEqual(a._pool_seed, 45)
            self.assertEqual(a._pool_seed, b._pool_seed)
            self.assertEqual(a.data_pool[0]['filename'], b.data_pool[0]['filename'])
            np.testing.assert_array_equal(a[0][0], b[0][0])
            with patch.object(b, 'repool', wraps=b.repool) as repool:
                b.set_epoch(5)
                repool.assert_not_called()
            all_data = CytoNucsDataGenerator(root, cfg, cfg.rays, pool_size=-1)
            self.assertEqual(len(all_data.data_pool), 3)

    def test_bce_monitoring_does_not_change_objective(self):
        loss = CytoNucsLoss(config())
        p = tf.constant([[[[[.2], [.8]]]]])
        t = tf.constant([[[[[0.], [1.]]]]])
        d = tf.ones((1, 1, 1, 2, 12))
        result = loss._stardist_loss(t, p, d, d)
        self.assertAlmostEqual(float(result['prob_foreground_fraction']), .5)
        self.assertAlmostEqual(float(result['bce_foreground']), -np.log(.8), places=6)
        self.assertAlmostEqual(float(result['bce_background']), -np.log(.8), places=6)
        self.assertAlmostEqual(float(result['bce']), -np.log(.8), places=6)

    def test_active_fixes_tiff_train_checkpoint_runtime_and_matching(self):
        tf.keras.utils.set_random_seed(41)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            make_dataset(root / 'data')
            raw = json.loads((ROOT / 'config_cytonucs_v2_6.json').read_text())
            raw['output']['basedir'] = str(root)
            raw['experiment']['name'] = 'integration'
            raw['model'].update(n_rays=12, n_channel_in=1, unet_n_depth=2,
                unet_n_filter_base=2, net_conv_after_unet=4, mask_refinement_filters=4,
                unet_pool_schedule=[[1, 2, 2], [2, 2, 2]],
                unet_kernel_schedule=[[1, 3, 3], [3, 3, 3], [3, 3, 3]])
            raw['training'].update(train_patch_size=[8, 32, 32], image_batch_size=2,
                resume_from_checkpoint=None, reset_best_on_resume=False, steps_per_epoch=1,
                predicted_prompt_start_epoch=1)
            raw['evaluation']['save_every_n_epochs'] = 1
            raw['inference']['refinement_patch_size'] = [0, 32, 32]
            cfg_path = root / 'config.json'
            cfg_path.write_text(json.dumps(raw))
            cfg = CytoNucsConfig.from_json(cfg_path)
            for section, fields in cfg.FIXES_SETTINGS.items():
                for name in fields:
                    self.assertEqual(getattr(cfg, name), raw[section][name])
            gen = CytoNucsDataGenerator(root / 'data', cfg, cfg.rays, augmenter=cytonucs_augmenter_3d)
            model = build_cytonucs_model(cfg, name='integration', basedir=str(root))
            trainer = CytoNucsTrainer(model, cfg, CytoNucsLoss(cfg), 0)
            # No history supplied at resume: epoch3 must still occupy index2.
            def end(epoch, history):
                trainer.state_best_full.assign(.731)
                history.setdefault('fullval_epoch', []).append(epoch + 1)
                history.setdefault('fullval_metric', []).append(.731)
            trainer.train(gen, epochs=3, start_epoch=2, steps_per_epoch=1,
                          callbacks=[SimpleNamespace(on_epoch_end=end)])
            self.assertEqual(trainer.history['train_loss'][:2], [None, None])
            self.assertEqual(len(trainer.history['train_loss']), 3)
            self.assertTrue(np.isfinite(trainer.history['train_nucleus_bce_foreground'][2]))
            path = trainer.checkpoint_dir / 'weights_epoch_003.weights.h5'
            restored_model = build_cytonucs_model(cfg, name='restored', basedir=str(root))
            restored_model.load_weights(path)
            restored = CytoNucsTrainer(restored_model, cfg, CytoNucsLoss(cfg), 0)
            self.assertTrue(restored.restore_training_state(path))
            self.assertAlmostEqual(float(restored.state_best_full.numpy()), .731, places=6)
            self.assertEqual(int(restored.state_epoch.numpy()), 3)
            # One more real optimizer update must agree after restoring all slots.
            gen.set_epoch(3)
            x, y = gen[0]
            xt = tf.constant(x)
            yt = {k: tf.convert_to_tensor(v) for k, v in y.items()}
            for t in (trainer, restored):
                t.current_epoch_var.assign(4)
                result = t.train_step(xt, yt)
                self.assertTrue(all(np.isfinite(float(v)) for v in result[-1].values()))
            for a, b in zip(model.keras_model.weights, restored_model.keras_model.weights):
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)
            # Runtime heads agree with every K-major training output for B=2.
            output = model.keras_model(xt, training=False)
            for b in range(2):
                features = list(model.refinement_features_model(xt[b:b+1, ..., :1]))
                for k in range(2):
                    o = 1 + 2*k
                    result = model.refinement_with_quality_model(features +
                        [xt[b:b+1, ..., o:o+1], xt[b:b+1, ..., o+1:o+2]])
                    for tensor, name in zip(result, ['nucleus_mask_logit', 'cytoplasm_mask_logit',
                                                     'nucleus_quality', 'cytoplasm_quality']):
                        np.testing.assert_allclose(tensor, output[name][k*2+b:k*2+b+1], rtol=1e-4, atol=1e-5)
            # Actual 3D refinement, quality ranking, observer and final pairing.
            sample = gen.data_pool[0]
            audits = {h: RefinementStageAudit(sample[h + '_mask']) for h in ('nucleus', 'cytoplasm')}
            n, c, pairs = predict_instances_prompted(model, cfg, sample['image'],
                np.array([[3, 8, 8]]), np.array([.9]), nuc_labels=sample['nucleus_mask'],
                cyto_labels=sample['cytoplasm_mask'], stage_audit=audits)
            value, details = compute_jtpr(n, c, sample['nucleus_mask'], sample['cytoplasm_mask'],
                                         assignments_pred=pairs, assignments_gt=sample['assignments'])
            self.assertEqual(details['total_pairs'], 2)
            self.assertTrue(0 <= value <= 1)
            self.assertEqual(n.shape, (8, 32, 32))
            for audit in audits.values():
                self.assertIn('final', audit.report()['stages'])
            # No-candidate inference must clear diagnostics from the preceding volume.
            predictor = CytoNucsPredictor(model, cfg, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            image = sample['image'][..., None]
            model.last_refinement_diagnostics = {'stale': 1}
            _, _, direct = predictor.predict_instances(image, normalize_img=False)
            self.assertEqual(model.last_refinement_diagnostics, {})
            model.last_refinement_diagnostics = {'stale': 1}
            _, _, tiled = predictor._predict_with_tiling(image, (1, 1, 1), False)
            self.assertEqual(model.last_refinement_diagnostics, {})
            for key in ('nucleus_prob', 'nucleus_dist', 'cytoplasm_prob', 'cytoplasm_dist'):
                np.testing.assert_allclose(direct[key], tiled[key], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
