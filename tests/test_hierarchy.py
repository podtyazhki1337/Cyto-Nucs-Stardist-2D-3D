"""Hierarchies of three or more levels (cytonucs_hierarchy): adjacent pairs through the one pair
network, a pair code in the input, one pair per batch, full validation and prediction per pair."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
import tifffile
from test_fixes import config, ROOT
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import cytonucs_augmenter_3d
from cytonucs_hierarchy import (HierarchyGenerator, ModelPairView, PoolView, build_trainers, link_children, pair_name,
                                predict_hierarchy, read_calibration, with_pair_code)
from cytonucs_model import build_cytonucs_model

LEVELS = ['nucleolus', 'nucleus', 'cytoplasm']
SIZE = dict(nucleolus=2 * 4 * 4, nucleus=4 * 12 * 12, cytoplasm=6 * 22 * 22)     # voxels of one object
FEATURES = dict(refinement_depth=1, mask_quality_head=True, mask_quality_pooling='mask', mask_quality_weight=.1,
                semantic_foreground_weight=1., refinement_full_resolution=True)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def tile(seed, shape=(8, 64, 64)):
    """Four cells, each with one nucleus that holds one nucleolus."""
    rng = np.random.default_rng(seed)
    masks = {level: np.zeros(shape, np.uint16) for level in LEVELS}
    for k, (cy, cx) in enumerate(((16, 16), (16, 46), (46, 16), (46, 46)), 1):
        y, x = cy + int(rng.integers(-3, 4)), cx + int(rng.integers(-3, 4))
        masks['cytoplasm'][1:7, y - 11:y + 11, x - 11:x + 11] = k
        masks['nucleus'][2:6, y - 6:y + 6, x - 6:x + 6] = k
        masks['nucleolus'][3:5, y - 2:y + 2, x - 2:x + 2] = k
    image = sum(0.3 * (masks[level] > 0) for level in LEVELS) + rng.normal(0, .05, shape)
    return image.astype(np.float32), masks


def dataset(root, splits=(('train', 4),), levels_of=lambda split, index: LEVELS):
    for split, count in splits:
        for folder in ['images'] + [level + '_masks' for level in LEVELS]:
            (root / split / folder).mkdir(parents=True)
        for i in range(count):
            image, masks = tile(100 * len(split) + i)
            tifffile.imwrite(root / split / 'images' / f't{i}.tif', image, photometric='minisblack')
            for level in levels_of(split, i):
                tifffile.imwrite(root / split / (level + '_masks') / f't{i}.tif', masks[level], photometric='minisblack')


def tensors(batch):
    x, y = batch
    return tf.constant(x, tf.float32), {k: tf.constant(v, tf.float32 if v.dtype.kind == 'f' else tf.int32 if v.dtype.kind in 'iu' else None)
                                        for k, v in y.items() if isinstance(v, np.ndarray)}


class ConfigTests(unittest.TestCase):
    def test_three_levels_are_adjacent_pairs_with_a_pair_code(self):
        cfg = quiet(config, levels=LEVELS)
        self.assertEqual(cfg.level_pairs, [('nucleolus', 'nucleus'), ('nucleus', 'cytoplasm')])
        self.assertEqual((cfg.n_image_channels, cfg.n_pair_channels, cfg.n_channel_in), (1, 2, 3))
        for index, levels in enumerate((['nucleolus', 'nucleus'], ['nucleus', 'cytoplasm'])):
            view = quiet(cfg.pair_view, index)
            self.assertEqual((view.pair_index, view.levels, view.level_roles), (index, levels, dict(zip(('nucleus', 'cytoplasm'), levels))))
            self.assertEqual((view.n_channel_in, view.n_pair_channels), (3, 2))
            self.assertEqual(pair_name(view), '_'.join(levels))
        self.assertEqual(cfg.levels, LEVELS)                    # views do not change the config
        four = quiet(config, levels=['granule', 'nucleolus', 'nucleus', 'cytoplasm'])
        self.assertEqual((len(four.level_pairs), four.n_channel_in), (3, 4))

    def test_one_and_two_levels_have_no_pair_code(self):
        for levels in (['nucleus', 'cytoplasm'], ['cytoplasm'], ['cells'], ['nucleolus', 'nucleus']):
            cfg = quiet(config, levels=levels)
            self.assertEqual((cfg.n_pair_channels, cfg.n_channel_in, cfg.n_image_channels), (0, 1, 1))
            self.assertIs(cfg.pair_view(0), cfg)

    def test_invalid_hierarchies(self):
        with self.assertRaisesRegex(ValueError, 'inner first'):
            quiet(config, levels=['cytoplasm', 'organelle', 'nucleus'])
        with self.assertRaisesRegex(ValueError, 'must be equal'):
            quiet(config, levels=LEVELS, nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 2, 2))

    def test_geometry_is_measured_per_level(self):
        labels = {}
        for level, half in (('nucleolus', 2), ('nucleus', 5), ('cytoplasm', 9)):
            lab = np.zeros((16, 200, 200), np.uint16)
            for k in range(49):
                y, x = 14 + 28 * (k // 7), 14 + 28 * (k % 7)
                lab[8 - min(half, 3):8 + min(half, 3), y - half:y + half, x - half:x + half] = k + 1
            labels[level] = [lab]
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = quiet(config, levels=LEVELS, median_max_dist_nucleus=None, median_max_dist_cytoplasm=None,
                        anisotropy=(1, 1, 1), nucleus_grid=(1, 2, 2), cytoplasm_grid=(1, 2, 2))
            cfg.geometry_path = str(Path(tmp) / 'geometry.json')
            geometry = quiet(cfg.set_hierarchy_geometry, labels)
            self.assertEqual(sorted(geometry), sorted(LEVELS))
            scales = [geometry[level]['median_max_dist'] for level in LEVELS]
            self.assertEqual(scales, sorted(scales))
            self.assertLess(scales[0], scales[2])
            # Each pair reads the scales of its own levels; the network takes the largest per role.
            first, second = quiet(cfg.pair_view, 0), quiet(cfg.pair_view, 1)
            self.assertEqual((first.median_max_dist_nucleus, first.median_max_dist_cytoplasm), (scales[0], scales[1]))
            self.assertEqual((second.median_max_dist_nucleus, second.median_max_dist_cytoplasm), (scales[1], scales[2]))
            self.assertEqual((cfg.median_max_dist_nucleus, cfg.median_max_dist_cytoplasm), (scales[1], scales[2]))
            stored = json.loads(Path(cfg.geometry_path).read_text())
            self.assertEqual(stored['levels'], geometry)
            with self.assertRaisesRegex(ValueError, 'No objects'):
                quiet(quiet(config, levels=LEVELS, median_max_dist_nucleus=None, median_max_dist_cytoplasm=None).set_hierarchy_geometry,
                      dict(labels, nucleolus=[np.zeros((16, 200, 200), np.uint16)]))


class HelperTests(unittest.TestCase):
    def test_pair_code_and_views(self):
        cfg = quiet(config, levels=LEVELS)
        view = quiet(cfg.pair_view, 1)
        image = np.random.default_rng(0).random((8, 16, 16)).astype(np.float32)
        coded = with_pair_code(image, view)
        self.assertEqual(coded.shape, (8, 16, 16, 3))
        np.testing.assert_array_equal(coded[..., 0], image)
        self.assertTrue((coded[..., 1] == 0).all() and (coded[..., 2] == 1).all())
        with self.assertRaises(ValueError):
            with_pair_code(coded, view)
        pool = PoolView(type('G', (), dict(config=view, subset='val', image_files=['a'],
                                           data_pool=[dict(image=image, nucleus_mask=1, filename='a')]))())
        sample = pool.data_pool[0]
        self.assertEqual(sample['image'].shape, (8, 16, 16, 3))
        self.assertEqual((sample['nucleus_mask'], sample.get('filename'), sample.get('missing', 5)), (1, 'a', 5))
        model = type('M', (), dict(name='net', shared=1))()
        a, b = ModelPairView(model, quiet(cfg.pair_view, 0)), ModelPairView(model, view)
        a.pending_quality_gates = dict(nucleus=.5)
        self.assertEqual((a.name, a.shared, b.pending_quality_gates, a.config.pair_index, b.config.pair_index), ('net', 1, None, 0, 1))
        self.assertFalse(hasattr(model, 'pending_quality_gates'))

    def test_children_are_linked_to_the_parent_they_overlap_most(self):
        parents = np.zeros((1, 6, 12), np.int32); parents[0, :, :6] = 1; parents[0, :, 6:] = 2
        children = np.zeros_like(parents)
        children[0, 1:3, 1:3] = 1                                  # inside parent 1
        children[0, 3:5, 4:9] = 2                                  # 2 columns in parent 1, 3 in parent 2
        children[0, 0, 11] = 7
        parents[0, 0, 11] = 0                                      # child 7 outside every parent
        self.assertEqual(link_children(children, parents), {1: 1, 2: 2})
        self.assertEqual(link_children(np.zeros_like(parents), parents), {})


class GeneratorTests(unittest.TestCase):
    def test_a_batch_holds_one_pair_in_the_two_roles(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            dataset(Path(tmp))
            cfg = quiet(config, levels=LEVELS, image_batch_size=2)
            generator = quiet(HierarchyGenerator, Path(tmp) / 'train', cfg, cytonucs_augmenter_3d, subset='train')
            self.assertEqual([g.config.levels for g in generator.pairs], [list(p) for p in cfg.level_pairs])
            self.assertEqual([len(s['assignments']) for g in generator.pairs for s in g.data_pool], [4] * 8)   # found by overlap
            seen = set()
            for index in range(24):
                x, y = generator[index]
                pair = y['pair_index']
                seen.add(pair)
                self.assertEqual(x.shape, (2, 8, 32, 32, 1 + 2 + 2 * 2))
                code = x[..., 1:3]
                expected = np.zeros(2, np.float32); expected[pair] = 1.
                self.assertTrue((code == expected).all(), index)        # constant, untouched by augmentation
                for role, level in zip(('nucleus', 'cytoplasm'), cfg.level_pairs[pair]):
                    largest = max(int(np.bincount(crop.ravel())[1:].max(initial=0)) for crop in y[role + '_instances'])
                    self.assertLessEqual(largest, SIZE[level], level)       # the objects of this level, whole or cut
                    self.assertGreater(largest, SIZE[LEVELS[LEVELS.index(level) - 1]] if level != 'nucleolus' else 0, level)
            self.assertEqual(seen, {0, 1})
            self.assertEqual(sorted(generator.level_labels()), sorted(LEVELS))
            validation = quiet(HierarchyGenerator, Path(tmp) / 'train', cfg, None, shuffle=False, subset='val')
            self.assertEqual(len(validation), 8)
            self.assertEqual([validation[i][1]['pair_index'] for i in range(8)], [0] * 4 + [1] * 4)

    def test_files_may_annotate_different_pairs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            dataset(Path(tmp), levels_of=lambda split, i: LEVELS[:2] if i < 2 else LEVELS[1:])
            cfg = quiet(config, levels=LEVELS, partial_annotation=True)
            generator = quiet(HierarchyGenerator, Path(tmp) / 'train', cfg, None, subset='train')
            # Every file has nuclei; nucleoli only in the first two files, cells only in the last two.
            # 'annotated' lists the roles of the pair: nuclei are the outer role of pair 0, the inner of pair 1.
            both = ('nucleus', 'cytoplasm')
            expected = ([both, both, ('cytoplasm',), ('cytoplasm',)], [('nucleus',), ('nucleus',), both, both])
            for pair, roles in zip(generator.pairs, expected):
                self.assertEqual([s['annotated'] for s in sorted(pair.data_pool, key=lambda s: s['filename'])], roles)
            flags = set()
            for index in range(40):
                _, y = generator[index]
                flags.add((y['pair_index'],) + tuple(y['levels_present'].tolist()))
            self.assertEqual(flags, {(0, 1., 1.), (0, 0., 1.), (1, 1., 1.), (1, 1., 0.)})
            with self.assertRaises(AssertionError):                    # without the setting a missing mask is an error
                quiet(HierarchyGenerator, Path(tmp) / 'train', quiet(config, levels=LEVELS), None, subset='train')


class TrainingTests(unittest.TestCase):
    def test_pair_trainers_share_the_network_and_the_optimizer(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            dataset(Path(tmp))
            cfg = quiet(config, levels=LEVELS, **FEATURES)
            generator = quiet(HierarchyGenerator, Path(tmp) / 'train', cfg, None, subset='train')
            quiet(cfg.set_hierarchy_geometry, generator.level_labels())
            generator.set_config(cfg)
            model = quiet(build_cytonucs_model, cfg, basedir=tmp, name='hierarchy')
            self.assertEqual(model.keras_model.input_shape[-1], 1 + 2 + 2 * 2)
            trainers, views = quiet(build_trainers, model, cfg, 0.)
            self.assertEqual(len(trainers), 2)
            self.assertIs(trainers[0].optimizer, trainers[1].optimizer)
            self.assertIs(trainers[0].train_loss_metric, trainers[1].train_loss_metric)
            self.assertIs(views[0].keras_model, views[1].keras_model)
            self.assertEqual([t.config.levels for t in trainers], [list(p) for p in cfg.level_pairs])
            batches = {}
            for index in range(20):
                x, y = generator[index]
                batches.setdefault(y['pair_index'], (x, y))
            self.assertEqual(sorted(batches), [0, 1])
            steps = int(trainers[0].optimizer.iterations.numpy())
            for pair, (x, y) in sorted(batches.items()):
                stepper = trainers[0]._pair(y)
                self.assertIs(stepper, trainers[pair])
                self.assertNotIn('pair_index', y)
                before = [v.numpy().copy() for v in model.keras_model.trainable_variables]
                result = stepper.train_step(*tensors((x, y)))
                self.assertTrue(np.isfinite(float(result[0])))
                changed = sum(not np.array_equal(a, v.numpy()) for a, v in zip(before, model.keras_model.trainable_variables))
                self.assertGreater(changed, len(before) // 2, pair)
            self.assertEqual(int(trainers[0].optimizer.iterations.numpy()), steps + 2)
            # A two-level run has no pair trainers: batches go to the trainer itself.
            self.assertIs(trainers[1]._pair({'pair_index': 0}), trainers[1])

    def test_training_run_validation_per_pair_and_prediction(self):
        import train_cytonucs
        raw = json.loads((ROOT / 'config_cytonucs_v4.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            dataset(root / 'data', (('train', 4), ('val', 2), ('test', 2)))
            raw['experiment']['name'] = 'three'
            raw['data']['dataset_dir'] = str(root / 'data') + '/'
            raw['model'].update(levels=LEVELS, n_rays=12, n_channel_in=1, use_gpu=False, unet_n_depth=2, unet_n_filter_base=4,
                                net_conv_after_unet=8, mask_refinement_filters=8, unet_norm_groups=2,
                                unet_pool_schedule=[[1, 2, 2], [2, 2, 2]],
                                unet_kernel_schedule=[[1, 3, 3], [3, 3, 3], [3, 3, 3]], anisotropy=[3, 1, 1])
            raw['training'].update(train_patch_size=[8, 32, 32], train_epochs=2, steps_per_epoch=4, train_region_crops=False,
                                   lr_warmup_epochs=0, predicted_prompt_start_epoch=1, prefetch_workers=1, prefetch_depth=1)
            raw['inference'].update(refinement_patch_size=[0, 32, 32],
                                    checkpoint_path=str(root / 'models/three/weights_best.weights.h5'))
            raw['evaluation'].update(eval_dice_every=1, min_eval_epoch=1, save_every_n_epochs=1000, fullval_train_volumes=1,
                                     save_predictions=False)
            raw['gpu_config']['visible_devices'] = '-1'
            raw['output']['basedir'] = str(root / 'models')
            (root / 'config.json').write_text(json.dumps(raw))
            cwd = os.getcwd()
            try:
                os.chdir(root)
                result = quiet(train_cytonucs.main, str(root / 'config.json'))
            finally:
                os.chdir(cwd)
            history = json.loads((root / 'models/three/history.json').read_text())
            self.assertEqual(history['fullval_epoch'], [1, 2])
            self.assertEqual(sorted(history['pairs']), ['nucleolus_nucleus', 'nucleus_cytoplasm'])
            for log in history['pairs'].values():
                self.assertEqual(log['fullval_epoch'], [1, 2])
                self.assertEqual(len(log['fullval_jtpr']), 2)
            self.assertTrue(all(np.isfinite(v) for v in history['train_loss'] + history['val_loss']))
            final = root / 'models/three/weights_final.weights.h5'
            self.assertTrue(final.exists())
            self.assertEqual(sorted(read_calibration(final)), ['nucleolus_nucleus', 'nucleus_cytoplasm'])
            geometry = json.loads((root / 'models/three/geometry.json').read_text())
            self.assertEqual(sorted(geometry['levels']), sorted(LEVELS))
            image, _ = tile(7)
            out = quiet(predict_hierarchy, result['model'], result['config'], image, calibration=result['calibration'])
            self.assertEqual(list(out['levels']), LEVELS)
            for labels in out['levels'].values():
                self.assertEqual(labels.shape, image.shape)
            self.assertEqual(sorted(out['parents']), ['nucleolus', 'nucleus'])
            # The saved geometry comes back with the config.
            reloaded = quiet(CytoNucsConfig.from_json, root / 'config.json')
            self.assertEqual(reloaded.level_geometry, geometry['levels'])


if __name__ == '__main__':
    unittest.main()
