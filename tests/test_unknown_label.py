"""training.unknown_label: a value of training masks that marks the voxels whose annotation of that level
is unknown (sparse annotation, e.g. the nuclei of only some cells). The loss of a level averages over
its known voxels, a prompt whose empty target lies in unknown annotation teaches nothing, negatives
come only from known background; crops can be centred on the objects without unknown annotation."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
import tifffile
from test_fixes import config, ROOT
from test_levels import quiet, tile, tensors, FEATURES
from test_architecture_v2 import batch
from test_v4 import region_dataset, held_out_labels, NAME
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import (CytoNucsDataGenerator, cytonucs_augmenter_3d, split_unknown, with_unknown,
                                     without_unknown)
from cytonucs_losses import CytoNucsLoss
from cytonucs_model import build_cytonucs_model
from cytonucs_trainer import CytoNucsTrainer, DiceCallback

U = 65535
HIDDEN = [3, 4]                                         # cells whose nuclei are not annotated


def sparse_dataset(root, split='train', count=3):
    """Tiles of test_levels.tile; the voxels of cells 3 and 4 carry unknown_label in the nucleus masks."""
    for folder in ('images', 'nucleus_masks', 'cytoplasm_masks'):
        (root / split / folder).mkdir(parents=True)
    for i in range(count):
        image, masks = tile(i)
        nuclei = masks['nucleus'].copy()
        nuclei[np.isin(masks['cytoplasm'], HIDDEN)] = U
        tifffile.imwrite(root / split / 'images' / f't{i}.tif', image, photometric='minisblack')
        tifffile.imwrite(root / split / 'nucleus_masks' / f't{i}.tif', nuclei, photometric='minisblack')
        tifffile.imwrite(root / split / 'cytoplasm_masks' / f't{i}.tif', masks['cytoplasm'], photometric='minisblack')


def sparse_config(**settings):
    return quiet(config, unknown_label=U, train_patch_size=(8, 64, 64), **settings)


class SettingTests(unittest.TestCase):
    def test_contract(self):
        self.assertIsNone(CytoNucsConfig.FIXES_SETTINGS['training']['unknown_label'])
        self.assertEqual(CytoNucsConfig.FIXES_SETTINGS['training']['annotated_anchor_share'], 0.)
        for path in ROOT.glob('config_cytonucs*.json'):
            training = json.loads(path.read_text())['training']
            self.assertEqual((training['unknown_label'], training['annotated_anchor_share']), (None, 0.), path.name)
        self.assertEqual(quiet(config, unknown_label=U, annotated_anchor_share=.5).unknown_label, U)
        for bad in (0, -1, True, 1.5, 'x'):
            with self.assertRaises(ValueError, msg=repr(bad)):
                quiet(config, unknown_label=bad)
        for bad in (-.1, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                quiet(config, annotated_anchor_share=bad)

    def test_unknown_voxels_survive_flips_rotations_and_zoom(self):
        labels = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
        self.assertEqual(split_unknown(labels, None)[1], None)
        self.assertIsNone(split_unknown(labels, 99)[1])
        marked = labels.copy(); marked[:, 1:3, 2] = 99
        clean, unknown = split_unknown(marked, 99)
        self.assertEqual((clean.dtype, int(unknown.sum())), (np.uint16, 4))
        self.assertFalse((clean[unknown]).any())
        packed = np.flip(np.rot90(with_unknown(clean, unknown), 1, axes=(1, 2)), axis=2)
        back, moved = without_unknown(packed, np.uint16)
        np.testing.assert_array_equal(moved, np.flip(np.rot90(unknown, 1, axes=(1, 2)), axis=2))
        np.testing.assert_array_equal(back, np.flip(np.rot90(clean, 1, axes=(1, 2)), axis=2))
        self.assertEqual(back.dtype, np.uint16)


class GeneratorTests(unittest.TestCase):
    def test_known_maps_and_prompt_flags_follow_the_augmented_labels(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            sparse_dataset(Path(tmp))
            cfg = sparse_config(zoom_range=(1., 1.3), negative_prompt_prob=.3)
            generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays,
                              augmenter=cytonucs_augmenter_3d, subset='train')
            sample = generator.data_pool[0]
            self.assertEqual(int(sample['nucleus_unknown'].sum()), int(np.isin(sample['cytoplasm_mask'], HIDDEN).sum()))
            self.assertIsNone(sample['cytoplasm_unknown'])
            self.assertNotIn(U, np.unique(sample['nucleus_mask']))
            flags = {0.: 0, 1.: 0}
            for index in range(30):
                x, y = generator[index]
                hidden = np.isin(y['cytoplasm_instances'], HIDDEN)
                np.testing.assert_array_equal(y['nucleus_known'] == 0, hidden)       # same voxels after zoom, flips, rotations
                self.assertTrue((y['cytoplasm_known'] == 1).all())
                self.assertFalse((y['nucleus_instances'][hidden] > 0).any())
                self.assertEqual(y['nucleus_target_known'].shape, (2,))
                for k in range(2):
                    target, flag = y['nucleus_target'][k], y['nucleus_target_known'][k]
                    flags[float(flag)] += 1
                    if target.any():
                        self.assertEqual(flag, 1.)
                    if flag == 0.:                                  # an empty target next to hidden nuclei
                        self.assertFalse(target.any())
                        self.assertTrue(hidden[0][x[0, ..., 1 + 2 * k] > .6].any())
                    self.assertFalse((target.astype(bool) & hidden[0]).any())
            self.assertGreater(min(flags.values()), 0, flags)
            # Without the setting the batches have no such keys.
            plain = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', quiet(config, train_patch_size=(8, 64, 64)),
                          cfg.rays, subset='train')
            self.assertNotIn('nucleus_known', plain[0][1])

    def test_validation_masks_must_be_complete(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            sparse_dataset(Path(tmp), split='val')
            cfg = sparse_config()
            with self.assertRaisesRegex(ValueError, 'must be complete'):
                quiet(CytoNucsDataGenerator, Path(tmp) / 'val', cfg, cfg.rays, subset='val', shuffle=False)

    def test_train_side_metric_skips_tiles_with_unknown_voxels(self):
        _, masks = tile(0)
        pool = [dict(filename=f't{i}_z000_y0000_x{64 * i:04d}_a00.tif', nucleus_mask=masks['nucleus'],
                     cytoplasm_mask=masks['cytoplasm'], nucleus_unknown=unknown, cytoplasm_unknown=None)
                for i, unknown in enumerate((None, masks['cytoplasm'] == 3, None))]
        callback = SimpleNamespace(model=SimpleNamespace(config=quiet(config, fullval_train_volumes=3)),
                                   train_generator=SimpleNamespace(data_pool=pool))
        self.assertEqual(sorted(s['filename'][:2] for s in DiceCallback._train_volumes(callback)), ['t0', 't2'])


class GeometryLabelTests(unittest.TestCase):
    def test_level_without_objects_in_the_tiles_is_measured_on_the_volumes(self):
        from train_cytonucs import geometry_labels
        empty, full = np.zeros((4, 8, 8), np.uint16), np.ones((4, 8, 8), np.uint16)
        whole = np.full((4, 16, 16), 2, np.uint16)
        tiles = [dict(nucleus_mask=empty, cytoplasm_mask=full)] * 2
        regions = [dict(nucleus_mask=whole, cytoplasm_mask=whole)]
        nuclei, cells = quiet(geometry_labels, SimpleNamespace(data_pool=tiles, region_pool=regions))
        self.assertEqual([m.shape for m in nuclei], [(4, 16, 16)])                     # from the volume
        self.assertEqual([m.shape for m in cells], [(4, 8, 8)] * 2)                     # still the tiles
        nuclei, _ = quiet(geometry_labels, SimpleNamespace(data_pool=tiles, region_pool=None))
        self.assertEqual(len(nuclei), 2)                                                # nothing to fall back on


class RegionAnchorTests(unittest.TestCase):
    def test_object_centred_crops_can_prefer_annotated_cells(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            _, nuclei, cells, _ = region_dataset(root)
            hidden = [1, 2, 3]                              # the first row of blocks: nuclei not annotated
            nuclei[np.isin(cells, hidden)] = U
            tifffile.imwrite(root / 'train_regions' / 'nucleus_masks' / NAME, nuclei, photometric='minisblack')
            anchors = {}
            for share in (0., 1.):
                cfg = quiet(config, train_region_crops=True, unknown_label=U, annotated_anchor_share=share, fg_crop_prob=1.)
                generator = quiet(CytoNucsDataGenerator, root / 'train', cfg, cfg.rays, subset='train')
                sample = generator.region_pool[0]
                labels = sample['anchor_labels']
                expected = set(range(1, 10)) - held_out_labels() - set(hidden)
                self.assertEqual(set(labels[sample['annotated_anchors']].tolist()), expected)
                rng = np.random.default_rng(0)
                anchors[share] = {generator._region_crop(sample, (8, 32, 32), rng)[1] for _ in range(60)}
                self.assertEqual(int(sample['nucleus_unknown'].sum()), int(np.isin(cells, hidden).sum()))
            self.assertTrue(anchors[0.] & set(hidden))
            self.assertTrue(anchors[1.] <= expected)


class SparseDatasetTests(unittest.TestCase):
    def test_inner_level_kept_in_k_train_cells_and_source_untouched(self):
        sys.path.insert(0, str(ROOT / 'dataset_prep'))
        import sparse_annotation as sparse
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp) / 'data'
            _, nuclei, cells, _ = region_dataset(root)
            # Train tiles cut from the volume (same label IDs), and a val folder.
            shutil.rmtree(root / 'train')
            rows = []
            for i, (y, x) in enumerate(((0, 0), (32, 0))):
                name = f'volume_z000_y{y:04d}_x{x:04d}_a00.tiff'
                rows.append(dict(filename=name, volume='volume'))
                for folder, array in (('images', np.zeros((8, 32, 32), np.float32)), ('nucleus_masks', nuclei[:, y:y + 32, x:x + 32]),
                                      ('cytoplasm_masks', cells[:, y:y + 32, x:x + 32])):
                    (root / 'train' / folder).mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(root / 'train' / folder / name, array, photometric='minisblack')
            shutil.copytree(root / 'train', root / 'val')
            (root / 'split_manifest.json').write_text(json.dumps(dict(files=rows)))
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            out = Path(tmp) / 'k2'
            summary = quiet(sparse.sparse_annotation, root, out, 2, seed=1)
            chosen = summary['chosen']['volume']
            self.assertEqual(len(chosen), 2)
            self.assertTrue(set(chosen) <= set(range(1, 10)) - held_out_labels())
            self.assertEqual(summary['inner_kept']['volume'], [c + 100 for c in chosen])
            self.assertEqual({p: p.read_bytes() for p in root.rglob('*') if p.is_file()}, before)   # hard links not written through
            volume = tifffile.imread(out / 'train_regions' / 'nucleus_masks' / NAME)
            self.assertEqual(set(np.unique(volume)) - {0, U}, {c + 100 for c in chosen})
            np.testing.assert_array_equal(volume == U, (cells > 0) & ~np.isin(cells, chosen))
            tile_ = tifffile.imread(out / 'train' / 'nucleus_masks' / rows[0]['filename'])
            np.testing.assert_array_equal(tile_, sparse.sparse_inner(nuclei[:, :32, :32], cells[:, :32, :32], chosen,
                                                                     [c + 100 for c in chosen], U))
            pairs = json.loads((out / 'train_regions' / 'assignments' / 'volume.json').read_text())
            self.assertEqual(sorted(int(k) for k in pairs), [c + 100 for c in chosen])
            for path in (out / 'val').rglob('*.tiff'):                                # val and test stay complete
                self.assertEqual(path.stat().st_ino, (root / path.relative_to(out)).stat().st_ino)
            # A generator with the setting reads the copy: unknown voxels where cells lost their nuclei.
            cfg = quiet(config, train_region_crops=True, unknown_label=U)
            generator = quiet(CytoNucsDataGenerator, out / 'train', cfg, cfg.rays, subset='train')
            np.testing.assert_array_equal(generator.region_pool[0]['nucleus_unknown'], volume == U)


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        sparse_dataset(Path(cls.tmp.name))
        cls.cfg = sparse_config(**FEATURES, negative_prompt_prob=0.)
        generator = quiet(CytoNucsDataGenerator, Path(cls.tmp.name) / 'train', cls.cfg, cls.cfg.rays, subset='train')
        cls.model = quiet(build_cytonucs_model, cls.cfg, basedir=cls.tmp.name, name='sparse')
        cls.loss = CytoNucsLoss(cls.cfg)
        cls.batches = [tensors(generator[index]) for index in range(6)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_known_equals_no_maps(self):
        x, y = self.batches[0]
        full = dict(y)
        for key in ('nucleus_known', 'cytoplasm_known'):
            full[key] = tf.ones_like(y[key])
        for key in ('nucleus_target_known', 'cytoplasm_target_known'):
            full[key] = tf.ones_like(y[key])
        bare = {k: v for k, v in y.items() if not k.endswith('_known')}
        prediction = self.model.keras_model(x, training=False)
        self.assertAlmostEqual(float(self.loss(full, prediction)), float(self.loss(bare, prediction)), places=5)

    def test_predictions_on_unknown_voxels_do_not_count(self):
        for x, y in self.batches:
            prediction = dict(self.model.keras_model(x, training=False))
            base = float(self.loss(y, prediction))
            grid = self.cfg.nucleus_grid
            unknown = tf.cast(y['nucleus_known'][:, ::grid[0], ::grid[1], ::grid[2], None] == 0, tf.float32)
            changed = dict(prediction)
            for key in ('nucleus_prob_logit', 'nucleus_semantic_logit'):
                changed[key] = prediction[key] + 4. * unknown
            changed['nucleus_prob'] = tf.sigmoid(changed['nucleus_prob_logit'])
            self.assertAlmostEqual(float(self.loss(y, changed)), base, places=5)
            # Counted again once the voxels are declared known.
            known = dict(y, nucleus_known=tf.ones_like(y['nucleus_known']))
            if float(tf.reduce_sum(unknown)):
                self.assertGreater(abs(float(self.loss(known, changed)) - float(self.loss(known, prediction))), 1e-4)

    def test_mask_ce_skips_unknown_voxels_and_unknown_prompts(self):
        rng = np.random.default_rng(1)
        logit = tf.constant(rng.normal(size=(2, 4, 8, 8, 3)), tf.float32)
        target = np.zeros((2, 4, 8, 8, 1), np.float32); target[0, 1:3, 2:5, 2:5] = 1
        others = np.zeros_like(target); others[:, 1:3, 5:8, 5:8] = 1
        known = np.ones_like(target); known[:, :, 5:, 5:] = 0
        ce = lambda z, prompts: float(self.loss._refinement_ce_loss(z, tf.constant(target), tf.constant(target + others),
                                                                     tf.constant(known), prompts))
        base = ce(logit, None)
        shifted = logit + tf.constant(4. * (1 - known) * rng.normal(size=(2, 4, 8, 8, 3)), tf.float32)
        self.assertAlmostEqual(ce(shifted, None), base, places=5)
        # The empty target of prompt 1 is unknown: changing its logits changes nothing.
        flags = tf.constant([1., 0.])
        moved = tf.concat([logit[:1], logit[1:] + tf.constant([3., 0., 0.])], axis=0)
        self.assertAlmostEqual(ce(moved, flags), ce(logit, flags), places=5)
        self.assertNotAlmostEqual(ce(moved, None), base, places=3)


class DetectorNegativeTests(unittest.TestCase):
    def test_detector_negatives_come_from_known_background(self):
        cfg = quiet(config, predicted_prompt_fraction=1.)
        x, y = batch(cfg)
        k = cfg.prompts_per_crop
        for head in ('nucleus', 'cytoplasm'):
            y[head + '_target'] = tf.zeros_like(y[head + '_target'])          # every prompt asks for a negative
        pred = {h + '_prob': tf.ones((1, 8, 16, 16, 1)) for h in ('nucleus', 'cytoplasm')}
        pred.update({h + '_dist': tf.ones((1, 8, 16, 16, 12)) * 2 for h in ('nucleus', 'cytoplasm')})
        stub = SimpleNamespace(config=cfg, prompt_detector=lambda x, training: pred,
                               optimizer=SimpleNamespace(iterations=tf.Variable(0, dtype=tf.int64)))
        centres = lambda out: [np.unravel_index(int(np.argmax(out[0, ..., cfg.n_channel_in + c])), out.shape[1:4])
                               for c in range(2 * k)]
        plain = centres(CytoNucsTrainer._predicted_prompts(stub, x, y).numpy())
        self.assertTrue(all(c[1] < 16 and c[2] < 16 for c in plain), plain)           # first background voxel
        known = np.zeros(cfg.train_patch_size, np.float32); known[:, 16:, 16:] = 1
        sparse = dict(y, nucleus_known=tf.constant(known[None]), cytoplasm_known=tf.ones((1,) + tuple(cfg.train_patch_size)))
        moved = centres(CytoNucsTrainer._predicted_prompts(stub, x, sparse).numpy())
        self.assertTrue(all(c[1] >= 16 and c[2] >= 16 for c in moved), moved)


if __name__ == '__main__':
    unittest.main()
