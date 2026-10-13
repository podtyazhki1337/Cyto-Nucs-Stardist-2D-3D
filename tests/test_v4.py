"""v4: training crops at any position inside the train blocks of the source volume, two
object-centred crops per step, GroupNorm in the backbone blocks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from csbdeep.utils import normalize
from test_fixes import config, ROOT
from test_v3_4 import make_tiles, quiet
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import (CytoNucsDataGenerator, cytonucs_augmenter_3d, allowed_crop_starts,
                                     window_inside, window_sum_table)
from cytonucs_model import build_cytonucs_model

sys.path.insert(0, str(ROOT / 'dataset_prep'))
import add_train_regions
import make_spatial_split_patches as prep

PRESET = 'config_cytonucs_v4.json'
BLOCK, SHAPE, HELD_OUT = 32, (8, 96, 96), ((1, 1), (2, 2))
NAME = 'volume.tiff'


def region_dataset(root):
    """Tile folder `train` plus `train_regions` with one volume: a 3x3 grid of 32 px blocks, one cell
    with a nucleus in the middle of every block. The blocks in HELD_OUT are not train: their image
    voxels are NaN, so a crop that reads them cannot go unnoticed."""
    make_tiles(root / 'train', (1, 2))
    rng = np.random.default_rng(0)
    image = rng.normal(size=SHAPE).astype(np.float32)
    nuclei = np.zeros(SHAPE, np.uint16); cells = np.zeros(SHAPE, np.uint16)
    blocks, pairs = [], {}
    for by in range(3):
        for bx in range(3):
            index = by * 3 + bx
            y, x = by * BLOCK, bx * BLOCK
            cells[1:7, y + 8:y + 24, x + 8:x + 24] = index + 1
            nuclei[2:6, y + 13:y + 19, x + 13:x + 19] = index + 101
            pairs[index + 101] = index + 1
            if (by, bx) in HELD_OUT:
                image[:, y:y + BLOCK, x:x + BLOCK] = np.nan
            else:
                blocks.append([y, y + BLOCK, x, x + BLOCK])
    image += 2 * (nuclei > 0)
    out = root / 'train_regions'
    for folder, array in (('images', image), ('nucleus_masks', nuclei), ('cytoplasm_masks', cells)):
        (out / folder).mkdir(parents=True)
        tifffile.imwrite(out / folder / NAME, array, photometric='minisblack')
    (out / 'assignments').mkdir()
    (out / 'assignments' / 'volume.json').write_text(json.dumps(pairs))
    (out / 'regions.json').write_text(json.dumps({NAME: dict(train_blocks=blocks, block_size=[BLOCK, BLOCK])}))
    return image, nuclei, cells, blocks


def held_out_labels():
    return {by * 3 + bx + 1 for by, bx in HELD_OUT}


class WindowTests(unittest.TestCase):
    def test_allowed_starts_and_window_check_match_brute_force(self):
        rng = np.random.default_rng(3)
        region = rng.random((14, 17)) > .15
        region[4:9, 2:12] = True
        for size in ((1, 1), (3, 4), (5, 9), (14, 17)):
            allowed = allowed_crop_starts(region, size)
            expected = np.array([[region[y:y + size[0], x:x + size[1]].all() for x in range(17 - size[1] + 1)]
                                 for y in range(14 - size[0] + 1)])
            np.testing.assert_array_equal(allowed, expected)
            table = window_sum_table(region)
            for y in range(-1, 15):
                for x in range(-1, 18):
                    inside = 0 <= y <= 14 - size[0] and 0 <= x <= 17 - size[1] and bool(expected[y, x])
                    self.assertEqual(window_inside(table, (y, x), size), inside, (size, y, x))
        self.assertEqual(allowed_crop_starts(region, (15, 3)).size, 0)


class RegionCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        cls.root = Path(cls.tmp.name)
        cls.image, cls.nuclei, cls.cells, cls.blocks = region_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def generator(self, augmenter=None, **settings):
        cfg = quiet(config, train_region_crops=True, image_batch_size=2, negative_prompt_prob=0., **settings)
        generator = quiet(CytoNucsDataGenerator, self.root / 'train', cfg, cfg.rays, augmenter=augmenter, subset='train')
        calls, crop = [], generator._region_crop

        def spy(*args):
            calls.append(crop(*args))
            return calls[-1]
        generator._region_crop = spy
        return generator, calls

    def test_crops_come_from_any_position_inside_the_train_blocks(self):
        generator, calls = self.generator()
        sample = generator.region_pool[0]
        self.assertEqual(sorted(sample['anchor_labels'].tolist()), sorted(set(range(1, 10)) - held_out_labels()))
        seen = set()
        for index in range(40):
            x, y = generator[index]
            self.assertEqual(x.shape[:4], (2, 8, 32, 32))
            self.assertTrue(np.isfinite(x).all())               # no voxel of a held-out block
            for head in ('nucleus_instances', 'cytoplasm_instances'):
                self.assertEqual(y[head].shape, (2, 8, 32, 32))
            seen |= set(np.unique(y['cytoplasm_instances']).tolist())
        self.assertEqual(len(calls), 80)
        for starts, _ in calls:
            self.assertTrue(window_inside(sample['table'], starts[-2:], (32, 32)), starts)
            self.assertEqual(starts[0], 0)
        positions = {tuple(starts) for starts, _ in calls}
        self.assertGreater(len(positions), 30)
        self.assertTrue(any(s[1] % BLOCK or s[2] % BLOCK for s in positions))   # not the tile grid
        # Objects of the held-out blocks may only appear as the part lying in a train block: none here.
        self.assertFalse(seen & held_out_labels())
        self.assertTrue(seen >= set(range(1, 10)) - held_out_labels())

    def test_crop_is_centred_on_its_anchor_cell_and_prompts_it_first(self):
        generator, calls = self.generator()
        centred = prompted = 0
        for index in range(30):
            _, y = generator[index]
            for b, (starts, anchor) in enumerate(calls[-2:]):
                if not anchor:
                    continue
                cy, cx = [(v // BLOCK) * BLOCK + 16 for v in np.argwhere(self.cells == anchor)[0][1:]]
                wanted = np.clip([cy - 16, cx - 16], 0, 64)
                if window_inside(generator.region_pool[0]['table'], tuple(wanted), (32, 32)):
                    self.assertLessEqual(np.abs(np.array(starts[-2:]) - wanted).max(), 2 + 2)   # jitter, then nearest allowed
                    centred += 1
                owner = y['cytoplasm_instances'][b] == anchor
                if owner.any():
                    np.testing.assert_array_equal(y['cytoplasm_target'][b], owner.astype(np.float32))
                    prompted += 1
        self.assertGreater(centred, 20)
        self.assertGreater(prompted, 20)
        self.assertGreater(sum(1 for _, anchor in calls if not anchor), 0)       # some crops are not object-centred

    def test_crop_is_normalised_with_the_statistics_of_its_tile(self):
        generator, _ = self.generator()
        sample = generator.region_pool[0]
        patch = (8, 32, 32)
        for y0, y1, x0, x1 in self.blocks:
            tile = normalize(self.image[:, y0:y1, x0:x1].astype(np.float32), 1, 99.8, axis=(0, 1, 2))
            # A crop equal to one block is that block normalised as a tile is.
            starts = [0, y0, x0]
            crop = sample['image'][:, y0:y1, x0:x1]
            np.testing.assert_allclose(generator._region_normalize(sample, crop, starts, patch), tile, rtol=1e-5, atol=1e-5)
        # A crop over two blocks uses the block that holds its centre: 20 of its 32 columns lie in block 0.
        starts = [0, 0, 12]
        crop = sample['image'][:, 0:32, 12:44]
        out = generator._region_normalize(sample, crop, starts, patch)
        tile = normalize(self.image[:, 0:32, 0:32].astype(np.float32), 1, 99.8, axis=(0, 1, 2))
        np.testing.assert_allclose(out[:, :, :20], tile[:, :, 12:], rtol=1e-5, atol=1e-5)
        self.assertEqual(out.dtype, np.float32)

    def test_zoom_and_augmentation_keep_crops_inside_the_train_blocks(self):
        generator, _ = self.generator(augmenter=cytonucs_augmenter_3d, zoom_range=(0.75, 1.35))
        for index in range(30):
            x, y = generator[index]
            self.assertEqual(x.shape[:4], (2, 8, 32, 32))
            self.assertTrue(np.isfinite(x).all())
            self.assertFalse(set(np.unique(y['cytoplasm_instances']).tolist()) & held_out_labels())

    def test_setting_is_off_by_default_and_needs_the_regions_folder(self):
        cfg = quiet(config)
        legacy = quiet(CytoNucsDataGenerator, self.root / 'train', cfg, cfg.rays, subset='train')
        self.assertIsNone(legacy.region_pool)
        validation = quiet(CytoNucsDataGenerator, self.root / 'train', quiet(config, train_region_crops=True), cfg.rays,
                           subset='val', shuffle=False)
        self.assertIsNone(validation.region_pool)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            make_tiles(Path(tmp) / 'train', (1, 2))
            with self.assertRaisesRegex(FileNotFoundError, 'add_train_regions'):
                quiet(CytoNucsDataGenerator, Path(tmp) / 'train', quiet(config, train_region_crops=True), cfg.rays, subset='train')
            (Path(tmp) / 'train_regions').mkdir()
            for folder, array in (('images', self.image[:, :16, :16]), ('nucleus_masks', self.nuclei[:, :16, :16]),
                                  ('cytoplasm_masks', self.cells[:, :16, :16])):
                (Path(tmp) / 'train_regions' / folder).mkdir()
                tifffile.imwrite(Path(tmp) / 'train_regions' / folder / NAME, array, photometric='minisblack')
            (Path(tmp) / 'train_regions' / 'assignments').mkdir()
            (Path(tmp) / 'train_regions' / 'assignments' / 'volume.json').write_text('{}')
            (Path(tmp) / 'train_regions' / 'regions.json').write_text(json.dumps({NAME: dict(train_blocks=[[0, 16, 0, 16]])}))
            with self.assertRaisesRegex(ValueError, 'no .* crop fits'):
                quiet(CytoNucsDataGenerator, Path(tmp) / 'train', quiet(config, train_region_crops=True), cfg.rays, subset='train')


class AddTrainRegionsTests(unittest.TestCase):
    def test_volume_and_train_blocks_are_written_from_the_split_manifest(self):
        from test_spatial_split import source
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            tmp = Path(tmp)
            image, nuclei, cells = source(0, (40, 150, 200))
            paths = {}
            for key, array in (('image', image), ('nucleus', nuclei), ('cytoplasm', cells)):
                paths[key] = tmp / f'{key}.tif'
                tifffile.imwrite(paths[key], array, photometric='minisblack')
            given = dict(name='first', image=image, nucleus=nuclei, cytoplasm=cells,
                         source={k: dict(path=str(p)) for k, p in paths.items()})
            manifest = quiet(prep.build, [given], tmp / 'dataset', (16, 48, 48), (8, 40, 40), augmentations=0, seed=3)
            regions = quiet(add_train_regions.build, tmp / 'dataset')
            blocks = [[b['y0'], b['y1'], b['x0'], b['x1']] for b in manifest['volumes'][0]['blocks'] if b['split'] == 'train']
            self.assertEqual(regions, json.loads((tmp / 'dataset' / 'train_regions' / 'regions.json').read_text()))
            self.assertEqual(regions['first.tiff']['train_blocks'], blocks)
            self.assertLess(len(blocks), len(manifest['volumes'][0]['blocks']))
            folder = tmp / 'dataset' / 'train_regions'
            np.testing.assert_array_equal(tifffile.imread(folder / 'images' / 'first.tiff'), image)
            np.testing.assert_array_equal(tifffile.imread(folder / 'nucleus_masks' / 'first.tiff'), nuclei)
            np.testing.assert_array_equal(tifffile.imread(folder / 'cytoplasm_masks' / 'first.tiff'), cells)
            pairs = json.loads((folder / 'assignments' / 'first.json').read_text())
            self.assertEqual(len(pairs), len(np.unique(nuclei)) - 1)
            with self.assertRaises(FileExistsError):
                quiet(add_train_regions.build, tmp / 'dataset')
            # The generator reads this layout: every crop lies in train blocks of the manifest.
            cfg = quiet(config, train_region_crops=True, n_channel_in=3)
            generator = quiet(CytoNucsDataGenerator, tmp / 'dataset' / 'train', cfg, cfg.rays, subset='train')
            region = np.zeros(image.shape[1:3], bool)
            for y0, y1, x0, x1 in blocks:
                region[y0:y1, x0:x1] = True
            calls, crop = [], generator._region_crop
            generator._region_crop = lambda *args: calls.append(crop(*args)) or calls[-1]
            for index in range(10):
                x, _ = generator[index]
                self.assertEqual(x.shape[:4], (1, 8, 32, 32))
            for starts, _ in calls:
                self.assertTrue(region[starts[1]:starts[1] + 32, starts[2]:starts[2] + 32].all())
            self.assertGreater(len({s[0] for s, _ in calls}), 1)                # any Z position of the 40 slices

    def test_missing_source_is_reported(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            tmp = Path(tmp)
            (tmp / 'split_manifest.json').write_text(json.dumps(dict(patch_size=[8, 32, 32], volumes=[dict(
                name='gone', shape=[8, 64, 64], source=dict(image=dict(path=str(tmp / 'no.tif'))),
                blocks=[dict(y0=0, y1=32, x0=0, x1=32, split='train')])])))
            with self.assertRaisesRegex(FileNotFoundError, 'gone'):
                add_train_regions.build(tmp)


class GroupNormTests(unittest.TestCase):
    def test_backbone_blocks_are_normalised_only_when_set(self):
        import tensorflow as tf
        legacy = quiet(build_cytonucs_model, quiet(config))
        normed = quiet(build_cytonucs_model, quiet(config, unet_norm_groups=2))
        is_norm = lambda layer: isinstance(layer, tf.keras.layers.GroupNormalization)
        self.assertFalse([l.name for l in legacy.keras_model.layers if is_norm(l)])
        norms = [l.name for l in normed.keras_model.layers if is_norm(l)]
        blocks = {n.rsplit('_c', 1)[0] for n in (l.name for l in legacy.keras_model.layers)
                  if n.endswith(('_c1', '_c2')) and n.startswith(('enc_l', 'nuc_dec_l', 'cyto_dec_l'))}
        self.assertEqual(sorted(norms), sorted(f'{b}_n{j}' for b in blocks for j in (1, 2)))
        self.assertTrue(blocks)
        # Same convolutions under the same names; the activation follows the normalisation.
        for block in blocks:
            for j in (1, 2):
                conv = normed.keras_model.get_layer(f'{block}_c{j}')
                self.assertEqual(conv.kernel.shape, legacy.keras_model.get_layer(f'{block}_c{j}').kernel.shape)
                self.assertEqual(conv.activation.__name__, 'linear')
        self.assertNotIn('unet_norm_groups', legacy.architecture_signature())
        self.assertEqual(normed.architecture_signature()['unet_norm_groups'], 2)
        x = np.random.default_rng(0).normal(size=(1, 8, 32, 32, 1 + 2 * 2)).astype(np.float32)
        out = normed.keras_model(x, training=True)
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in out.values()))

    def test_detection_in_fixed_windows_of_the_training_crop(self):
        """The v4 preset fixes all tiling axes: every detection window has the size of a training
        crop, so GroupNorm sees the extent it was trained on."""
        from cytonucs_inference import CytoNucsPredictor
        cfg = quiet(config, unet_norm_groups=2, tiling_fixed_axes=[0, 1, 2], refinement_patch_size=[0, 32, 32])
        model = quiet(build_cytonucs_model, cfg)
        predictor = CytoNucsPredictor(model, cfg, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
        for shape in ((8, 32, 32, 1), (8, 80, 72, 1), (8, 100, 37, 1), (16, 64, 64, 1)):
            x = np.random.default_rng(1).normal(size=shape).astype(np.float32)
            n_tiles = (1, int(np.ceil(shape[1] / 32)), int(np.ceil(shape[2] / 32)))
            nuclei, cells, maps = quiet(predictor._predict_with_tiling, x, n_tiles, False)
            self.assertEqual((nuclei.shape, cells.shape), (shape[:3], shape[:3]))
            for head in ('nucleus', 'cytoplasm'):
                self.assertEqual(maps[head + '_prob'].shape, (shape[0], -(-shape[1] // 2), -(-shape[2] // 2)))
                self.assertTrue(np.isfinite(maps[head + '_prob']).all() and np.isfinite(maps[head + '_dist']).all())
            if shape[:3] == (8, 32, 32):                        # one window: the same numbers as a single pass
                direct = quiet(predictor.predict_instances, x, normalize_img=False)[2]
                for key in ('nucleus_prob', 'nucleus_dist', 'cytoplasm_prob', 'cytoplasm_dist'):
                    np.testing.assert_allclose(maps[key], direct[key], rtol=1e-5, atol=1e-6)

    def test_groups_must_divide_the_filters(self):
        for value in (3, -1, True, 1.5, '2'):
            with self.assertRaises(ValueError, msg=repr(value)):
                quiet(config, unet_norm_groups=value)
        quiet(config, unet_norm_groups=1)


class ConfigTests(unittest.TestCase):
    KEYS = (('training', 'train_region_crops', False, True), ('model', 'unet_norm_groups', 0, 8))

    def load(self, raw):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw = json.loads(json.dumps(raw))
            raw['output']['basedir'] = str(Path(tmp) / 'models')
            raw['training']['resume_from_checkpoint'] = None
            path = Path(tmp) / 'config.json'; path.write_text(json.dumps(raw))
            return quiet(CytoNucsConfig.from_json, path)

    def test_json_contract(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            for section, key, legacy, new in self.KEYS:
                self.assertEqual(raw[section][key], new if path.name == PRESET else legacy, f'{path.name} {key}')
        raw = json.loads((ROOT / PRESET).read_text())
        cfg = self.load(raw)
        for section, key, legacy, new in self.KEYS:
            self.assertEqual(CytoNucsConfig.FIXES_SETTINGS[section][key], legacy)
            self.assertEqual(getattr(cfg, key), new)
        for value in (1, 'yes', None):
            bad = json.loads(json.dumps(raw)); bad['training']['train_region_crops'] = value
            with self.assertRaises(ValueError, msg=repr(value)):
                self.load(bad)
        bad = json.loads(json.dumps(raw)); bad['model']['unet_norm_groups'] = 5
        with self.assertRaises(ValueError):
            self.load(bad)

    def test_preset_is_v3_4_with_region_crops_two_crops_per_step_and_normalisation(self):
        base = json.loads((ROOT / 'config_cytonucs_v3_4.json').read_text())
        raw = json.loads((ROOT / PRESET).read_text())
        changed = {(section, key) for section in raw if isinstance(raw[section], dict)
                   for key in raw[section] if raw[section][key] != base[section].get(key)}
        self.assertEqual(changed, {
            ('experiment', 'name'), ('inference', 'checkpoint_path'), ('model', 'unet_norm_groups'),
            ('training', 'train_region_crops'), ('training', 'train_patch_size'), ('training', 'image_batch_size'),
            ('training', 'tile_sampling_by_objects'), ('training', 'train_epochs'),
            ('inference', 'refinement_patch_size'), ('inference', 'tiling_fixed_axes'), ('evaluation', 'min_eval_epoch')})
        self.assertEqual(set(raw), set(base))
        self.assertIsNone(raw['training']['resume_from_checkpoint'])
        self.assertIn(raw['experiment']['name'], raw['inference']['checkpoint_path'])
        cfg = self.load(raw)
        self.assertEqual((list(cfg.train_patch_size), cfg.image_batch_size, cfg.prompts_per_crop), ([16, 320, 320], 2, 3))
        # Inference sees what training saw: refinement ROI and detection windows of the training crop.
        self.assertEqual(list(cfg.refinement_patch_size)[1:], list(cfg.train_patch_size)[1:])
        self.assertEqual(list(cfg.tiling_fixed_axes), [0, 1, 2])
        self.assertEqual(cfg.tile_sampling_by_objects, 0.)                     # not read with region crops


if __name__ == '__main__':
    unittest.main()
