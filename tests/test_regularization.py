"""Against overfitting on one annotated volume: elastic XY deformation (augmentation.elastic_*),
decoupled weight decay (training.weight_decay) and fine-tuning from another run (training.init_weights)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_levels import quiet, mixed_dataset, BOTH
from test_unknown_label import sparse_dataset, sparse_config, HIDDEN
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import CytoNucsDataGenerator, cytonucs_augmenter_3d, elastic_displacement, warp_yx
from cytonucs_losses import CytoNucsLoss
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer
from test_levels import dataset, tensors

NEW = {('training', 'init_weights'): None, ('training', 'weight_decay'): 0.0, ('augmentation', 'elastic_prob'): 0.0,
       ('augmentation', 'elastic_alpha'): 0.0, ('augmentation', 'elastic_sigma'): 8.0, ('training', 'ema_decay'): 0.0,
       ('training', 'extra_train_dirs'): [], ('training', 'extra_train_share'): 0.0, ('inference', 'tta_flips'): False, ('inference', 'refinement_batch'): 0, ('inference', 'refinement_windows'): False, ('inference', 'max_prompts_per_head'): 0,
       ('training', 'val_loss_every_n_epochs'): 1,
       ('augmentation', 'prompt_uniform_prob'): 0.0, ('loss_params', 'dist_polar_iou_weight'): 0.0, ('loss_params', 'prob_iou_aware'): 0.0,
       ('augmentation', 'zoom_z_range'): None, ('augmentation', 'flip_z_prob'): 0.0, ('augmentation', 'saturation_prob'): 0.0}


class SettingTests(unittest.TestCase):
    def test_contract(self):
        for (section, key), default in NEW.items():
            self.assertEqual(CytoNucsConfig.FIXES_SETTINGS[section][key], default, key)
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            for (section, key), default in NEW.items():
                self.assertEqual(raw[section][key], default, f'{path.name} {key}')
        cfg = quiet(config, init_weights='w.weights.h5', weight_decay=.01, elastic_prob=.5, elastic_alpha=6., elastic_sigma=10.)
        self.assertEqual((cfg.init_weights, cfg.weight_decay, cfg.elastic_prob), ('w.weights.h5', .01, .5))
        for key, bad in (('init_weights', ''), ('init_weights', 3), ('weight_decay', -1.), ('elastic_prob', 1.5),
                         ('elastic_alpha', -2.), ('elastic_sigma', 0.), ('elastic_sigma', True), ('ema_decay', 1.),
                         ('ema_decay', -.1), ('extra_train_dirs', 'data'), ('extra_train_dirs', [3]),
                         ('extra_train_share', 1.5), ('extra_train_share', .5), ('tta_flips', 1), ('refinement_batch', -1), ('refinement_batch', True), ('refinement_windows', 1), ('max_prompts_per_head', -1), ('max_prompts_per_head', True),
                         ('val_loss_every_n_epochs', 0), ('val_loss_every_n_epochs', True),
                         ('prompt_uniform_prob', 1.5), ('prompt_uniform_prob', -.1), ('dist_polar_iou_weight', -1.), ('prob_iou_aware', -1.),
                         ('zoom_z_range', [1.2, .8]), ('zoom_z_range', [0, 1]), ('zoom_z_range', 1.1),
                         ('flip_z_prob', 2.), ('saturation_prob', -.5)):
            with self.assertRaises(ValueError, msg=f'{key}={bad!r}'):
                quiet(config, **{key: bad})


class ElasticTests(unittest.TestCase):
    def test_displacement_is_smooth_and_vanishes_at_the_faces(self):
        dy, dx = elastic_displacement((64, 80), 6., 8., np.random.default_rng(0))
        for d in (dy, dx):
            self.assertEqual(d.shape, (64, 80))
            for edge in (d[0], d[-1], d[:, 0], d[:, -1]):
                self.assertFalse(edge.any())
            self.assertGreater(np.abs(d[24:40, 30:50]).mean(), 1.)
            self.assertLess(np.abs(np.diff(d, axis=0)).max(), 3.)
        none = elastic_displacement((32, 32), 0., 8., np.random.default_rng(0))
        self.assertFalse(np.any(none[0]) or np.any(none[1]))

    def test_warp_moves_image_and_labels_together(self):
        rng = np.random.default_rng(1)
        labels = np.zeros((4, 48, 48), np.int32); labels[:, 10:30, 12:34] = 3; labels[1:3, 16:24, 18:26] = -1
        image = np.stack([labels.astype(np.float32)] * 2, axis=-1)
        dy, dx = elastic_displacement((48, 48), 4., 6., rng)
        warped_labels, warped_image = warp_yx(labels, dy, dx, 0, 3), warp_yx(image, dy, dx, 0, 3)
        self.assertEqual((warped_labels.shape, warped_image.shape), (labels.shape, image.shape))
        self.assertEqual(set(np.unique(warped_labels)), {-1, 0, 3})                      # no new labels
        np.testing.assert_array_equal(warped_image[..., 0], warped_labels)                # same displacement everywhere
        np.testing.assert_array_equal(warped_image[..., 1], warped_labels)
        self.assertFalse(np.array_equal(warped_labels, labels))
        flat = warp_yx(labels[0], dy, dx, 0, 2)                                           # 2D
        np.testing.assert_array_equal(flat, warped_labels[0])

    def test_generator_keeps_unknown_voxels_on_their_cells(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            sparse_dataset(Path(tmp))
            cfg = sparse_config(elastic_prob=1., elastic_alpha=5., elastic_sigma=6.)
            generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays,
                              augmenter=cytonucs_augmenter_3d, subset='train')
            for index in range(6):
                x, y = generator[index]
                hidden = np.isin(y['cytoplasm_instances'], HIDDEN)
                np.testing.assert_array_equal(y['nucleus_known'] == 0, hidden)       # deformed together
                self.assertTrue(np.isfinite(x).all())


class DepthAndSaturationTests(unittest.TestCase):
    def plain(self, **settings):
        return SimpleNamespace(rotation_yx=False, flip_prob=0., intensity_range=(1., 1.), intensity_offset=(0., 0.),
                               gamma_range=(1., 1.), noise_sigma=0., blur_sigma_range=None, **settings)

    def test_z_flip_and_saturation(self):
        rng = np.random.default_rng(0)
        image = rng.random((6, 8, 8, 3)).astype(np.float32)
        labels = np.arange(6 * 8 * 8).reshape(6, 8, 8).astype(np.int32)
        same = cytonucs_augmenter_3d(image, labels, labels, rng=np.random.default_rng(1), config=self.plain())
        np.testing.assert_allclose(same[0], image, rtol=1e-6)
        flipped = cytonucs_augmenter_3d(image, labels, labels, rng=np.random.default_rng(1), config=self.plain(flip_z_prob=1.))
        np.testing.assert_allclose(flipped[0], image[::-1], rtol=1e-6)
        np.testing.assert_array_equal(flipped[1], labels[::-1])
        saturated = cytonucs_augmenter_3d(image, labels, labels, rng=np.random.default_rng(1), config=self.plain(saturation_prob=1.))[0]
        changed = [c for c in range(3) if not np.allclose(saturated[..., c], image[..., c])]
        self.assertEqual(len(changed), 1)                                   # one channel clipped
        c = changed[0]
        self.assertLess(saturated[..., c].max(), image[..., c].max())
        np.testing.assert_allclose(saturated[..., c], np.minimum(image[..., c], saturated[..., c].max()), rtol=1e-6)

    def test_z_stretch_keeps_the_patch_and_the_unknown_voxels(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            sparse_dataset(Path(tmp))
            cfg = sparse_config(zoom_z_range=[1.25, 1.25])
            generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays,
                              augmenter=cytonucs_augmenter_3d, subset='train')
            depths = []
            for index in range(8):
                x, y = generator[index]
                self.assertEqual(x.shape[1:4], tuple(cfg.train_patch_size))
                np.testing.assert_array_equal(y['nucleus_known'] == 0, np.isin(y['cytoplasm_instances'], HIDDEN))
                depths.append(int((y['cytoplasm_instances'][0] > 0).any(axis=(1, 2)).sum()))
            self.assertEqual(min(depths), 6)                  # cells span 6 of 8 slices without the stretch
            self.assertGreater(max(depths), 6)                # and more slices when stretched by 1.25


class VisualisationTests(unittest.TestCase):
    def test_pictures_of_a_hierarchy_draw_only_the_image_channels(self):
        from cytonucs_trainer import DiceCallback
        cfg = quiet(config, n_channel_in=3, levels=['nucleolus', 'nucleus', 'cytoplasm'])
        self.assertEqual((cfg.n_image_channels, cfg.n_channel_in), (3, 5))
        stub = SimpleNamespace(model=SimpleNamespace(config=cfg))
        labels = np.zeros((4, 32, 32), np.int32); labels[1:3, 8:20, 8:20] = 1
        image = np.random.default_rng(0).random((4, 32, 32, 5)).astype(np.float32)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            quiet(DiceCallback._save_epoch_visualization_full, stub, 1, image, labels, labels, labels, labels,
                  {}, Path(tmp), filename='t.png')
            self.assertTrue((Path(tmp) / 't.png').is_file())


class OptimizerTests(unittest.TestCase):
    def test_weight_decay_shrinks_kernels_only(self):
        from train_cytonucs import make_optimizer
        self.assertIsInstance(make_optimizer(1e-3), tf.keras.optimizers.Adam)
        self.assertNotIsInstance(make_optimizer(1e-3), tf.keras.optimizers.AdamW)
        optimizer = make_optimizer(.1, weight_decay=.5)
        self.assertIsInstance(optimizer, tf.keras.optimizers.AdamW)
        layer = tf.keras.layers.Dense(2, kernel_initializer='ones', bias_initializer='ones')
        layer.build((None, 2))
        optimizer.apply_gradients([(tf.zeros_like(v), v) for v in layer.trainable_variables])
        self.assertTrue(np.all(layer.kernel.numpy() < 1.))
        np.testing.assert_array_equal(layer.bias.numpy(), 1.)


class EmaTests(unittest.TestCase):
    def test_averaged_weights_follow_the_steps_and_swap_back(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            dataset(Path(tmp), BOTH)
            cfg = quiet(config, ema_decay=.9, train_patch_size=(8, 64, 64))
            generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')
            model = quiet(build_cytonucs_model, cfg, basedir=tmp, name='ema')
            trainer = quiet(CytoNucsTrainer, model, cfg, CytoNucsLoss(cfg), 0., optimizer=tf.keras.optimizers.Adam(1e-2))
            variables = model.keras_model.trainable_variables
            start = [v.numpy().copy() for v in variables]
            np.testing.assert_array_equal(trainer.ema_variables[0].numpy(), start[0])
            x, y = tensors(generator[0])
            quiet(trainer.train_step, x, y)
            raw = [v.numpy().copy() for v in variables]
            average = [e.numpy().copy() for e in trainer.ema_variables]
            moved = [i for i, (a, b) in enumerate(zip(start, raw)) if not np.allclose(a, b)]
            self.assertTrue(moved)
            decay = 2. / 11.                                # warm-up after one step: (1 + 1) / (10 + 1) < 0.9
            for i in moved:
                np.testing.assert_allclose(average[i], decay * start[i] + (1 - decay) * raw[i], rtol=1e-5, atol=1e-6)
            trainer.swap_ema()
            self.assertTrue(trainer.ema_swapped)
            for v, a in zip(variables, average):
                np.testing.assert_array_equal(v.numpy(), a)
            trainer.swap_ema()
            for v, r in zip(variables, raw):
                np.testing.assert_array_equal(v.numpy(), r)
            plain = quiet(CytoNucsTrainer, model, quiet(config, train_patch_size=(8, 64, 64)), CytoNucsLoss(cfg), 0.)
            self.assertEqual(plain.ema_variables, [])
            plain.swap_ema()                                # no-op


class MixedDataTests(unittest.TestCase):
    def test_batches_come_from_the_extra_data_at_the_given_share(self):
        from train_cytonucs import MixedTrainGenerator
        calls = []

        class Source:
            def __init__(self, tag):
                self.tag = tag

            def __getitem__(self, index):
                return self.tag

            def __len__(self):
                return 10

            def set_epoch(self, epoch):
                calls.append((self.tag, epoch))

            def on_epoch_end(self):
                calls.append((self.tag, 'end'))
        main, extra = Source('main'), Source('extra')
        mixed = MixedTrainGenerator(main, [extra], .3, seed=1)
        picks = [mixed[i] for i in range(400)]
        self.assertAlmostEqual(picks.count('extra') / 400, .3, delta=.07)
        self.assertEqual(picks, [mixed[i] for i in range(400)])          # deterministic per epoch and index
        mixed.set_epoch(2)
        mixed.on_epoch_end()
        self.assertEqual(calls, [('main', 2), ('extra', 2), ('main', 'end'), ('extra', 'end')])
        self.assertEqual(len(mixed), 10)
        self.assertEqual(mixed.tag, 'main')                              # everything else reads the main generator
        self.assertEqual(set(MixedTrainGenerator(main, [extra], 0., 1)[i] for i in range(50)), {'main'})


class TtaTests(unittest.TestCase):
    def test_flip_averaging_runs_in_tiled_prediction(self):
        from cytonucs_inference import CytoNucsPredictor
        from types import SimpleNamespace as NS
        from cytonucs_trainer import DiceCallback
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = quiet(config, train_patch_size=(8, 32, 32), refinement_patch_size=(0, 32, 32))
            model = quiet(build_cytonucs_model, cfg, basedir=tmp, name='tta')
            image = np.random.default_rng(0).normal(size=(8, 64, 64, 1)).astype(np.float32)
            results, flips = {}, {}
            from unittest import mock
            for flag in (False, True):
                cfg.tta_flips = flag
                predictor = CytoNucsPredictor(model=model, config=cfg, prob_thresh_nucleus=.5, prob_thresh_cytoplasm=.5, nms_thresh=.3)
                with mock.patch('numpy.flip', wraps=np.flip) as flip:
                    nuclei, cells, _ = quiet(DiceCallback._predict_full_volume, NS(model=predictor.model), predictor, image)
                results[flag], flips[flag] = (nuclei.shape, cells.shape), flip.call_count
            self.assertGreaterEqual(flips[True] - flips[False], 9)          # 3 flipped inputs, 2 x 3 flipped outputs per tile
            self.assertEqual(results[False], results[True])
            self.assertEqual(results[True][0], image.shape[:3])


class InitWeightsTests(unittest.TestCase):
    def test_fine_tuning_starts_at_epoch_one_with_the_saved_geometry(self):
        import train_cytonucs
        raw = json.loads((ROOT / 'config_cytonucs_v4.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            mixed_dataset(root / 'data', (('train', (BOTH,) * 4), ('val', (BOTH,)), ('test', (BOTH,))))
            raw['data']['dataset_dir'] = str(root / 'data') + '/'
            raw['model'].update(n_rays=12, n_channel_in=1, use_gpu=False, unet_n_depth=2, unet_n_filter_base=4,
                                net_conv_after_unet=8, mask_refinement_filters=8, unet_norm_groups=2,
                                unet_pool_schedule=[[1, 2, 2], [2, 2, 2]],
                                unet_kernel_schedule=[[1, 3, 3], [3, 3, 3], [3, 3, 3]], anisotropy=[3, 1, 1])
            raw['training'].update(train_patch_size=[8, 32, 32], train_epochs=1, steps_per_epoch=2, train_region_crops=False,
                                   lr_warmup_epochs=0, prefetch_workers=1, prefetch_depth=1)
            raw['inference'].update(refinement_patch_size=[0, 32, 32])
            raw['evaluation'].update(eval_dice_every=100, min_eval_epoch=100, save_every_n_epochs=1, save_predictions=False)
            raw['gpu_config']['visible_devices'] = '-1'
            raw['output']['basedir'] = str(root / 'models')

            def run(name, **training):
                cfg = json.loads(json.dumps(raw))
                cfg['experiment']['name'] = name
                cfg['training'].update(training)
                cfg['inference']['checkpoint_path'] = str(root / f'models/{name}/weights_best.weights.h5')
                (root / f'{name}.json').write_text(json.dumps(cfg))
                cwd = os.getcwd()
                try:
                    os.chdir(root)
                    quiet(train_cytonucs.main, str(root / f'{name}.json'))
                finally:
                    os.chdir(cwd)
                return json.loads((root / f'models/{name}/history.json').read_text())

            run('first')
            geometry = json.loads((root / 'models/first/geometry.json').read_text())
            geometry['median_max_dist_nucleus'] += 3.                 # a marker the second run must not re-measure
            (root / 'models/first/geometry.json').write_text(json.dumps(geometry))
            history = run('second', init_weights=str(root / 'models/first/weights_epoch_001.weights.h5'), weight_decay=.01)
            self.assertEqual(len(history['train_loss']), 1)          # a new run of one epoch, not a resume
            self.assertTrue(np.isfinite(history['train_loss'][0]))
            second = json.loads((root / 'models/second/geometry.json').read_text())
            self.assertEqual(second['median_max_dist_nucleus'], geometry['median_max_dist_nucleus'])
            # The configuration of the second run reads that geometry (rays of the weights), not the preset one.
            cfg = quiet(CytoNucsConfig.from_json, str(root / 'second.json'))
            self.assertEqual(float(cfg.median_max_dist_nucleus), geometry['median_max_dist_nucleus'])
            self.assertEqual(list(cfg.rays_anisotropy_nucleus), list(geometry['rays_anisotropy_nucleus']))
            self.assertNotEqual(list(cfg.rays_anisotropy_nucleus), list(cfg.anisotropy))
            # Averaged weights and batches mixed from a second dataset run through the whole training script.
            mixed_dataset(root / 'extra', (('train', (BOTH,) * 2),))
            history = run('third', ema_decay=.9, extra_train_dirs=[str(root / 'extra')], extra_train_share=.5)
            self.assertTrue(np.isfinite(history['train_loss'][0]) and np.isfinite(history['val_loss'][0]))


if __name__ == '__main__':
    unittest.main()
