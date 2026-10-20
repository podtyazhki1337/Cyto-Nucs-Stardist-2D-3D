"""v5: v4 trained at half the XY resolution. dataset_prep/downscale_split.py writes the downscaled
copy of a dataset, score_native_resolution.py scores predictions on it against the native masks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from test_fixes import ROOT
from test_v3_4 import quiet
from cytonucs_config import CytoNucsConfig
from score_native_resolution import downscaling, enlarge_labels

sys.path.insert(0, str(ROOT / 'dataset_prep'))
import downscale_split as down

PRESET = 'config_cytonucs_v5.json'


def small_dataset(root):
    """train: a.tiff (both levels, a one-pixel nucleus among them), c.tiff (cells only); val: b.tiff;
    train_regions with one 64 px volume; the split manifest and split info of a 64 px volume."""
    rng = np.random.default_rng(0)
    shape = (4, 32, 32)
    nuclei = np.zeros(shape, np.uint16); cells = np.zeros(shape, np.uint16)
    cells[:, 4:20, 4:20] = 1; nuclei[1:3, 8:14, 8:14] = 11
    cells[:, 20:30, 18:30] = 2; nuclei[2, 25, 25] = 12                       # lost at half resolution
    files = {'train': ['a.tiff', 'c.tiff'], 'val': ['b.tiff']}
    for split, names in files.items():
        for name in names:
            for sub, array in (('images', rng.integers(0, 255, shape + (3,), dtype=np.uint8)),
                               ('nucleus_masks', None if name == 'c.tiff' else nuclei), ('cytoplasm_masks', cells)):
                (root / split / sub).mkdir(parents=True, exist_ok=True)
                if array is not None:
                    tifffile.imwrite(root / split / sub / name, array, photometric='minisblack')
            (root / split / 'assignments').mkdir(exist_ok=True)
            (root / split / 'assignments' / (Path(name).stem + '.json')).write_text(json.dumps({'11': 1, '12': 2}))
    (root / 'train' / 'meta.json').write_text(json.dumps([dict(file='a.tiff', cells_median_extent_p=[4, 16, 12])]))
    volume = np.zeros((4, 64, 64), np.uint16); volume[:, 10:30, 10:30] = 1
    for sub, array in (('images', rng.normal(size=(4, 64, 64)).astype(np.float32)), ('nucleus_masks', volume), ('cytoplasm_masks', volume)):
        (root / 'train_regions' / sub).mkdir(parents=True)
        tifffile.imwrite(root / 'train_regions' / sub / 'volume.tiff', array, photometric='minisblack')
    (root / 'train_regions' / 'assignments').mkdir()
    (root / 'train_regions' / 'assignments' / 'volume.json').write_text(json.dumps({'1': 1}))
    (root / 'train_regions' / 'regions.json').write_text(json.dumps({'volume.tiff': dict(
        train_blocks=[[0, 32, 0, 32], [32, 64, 0, 32]], block_size=[32, 32], shape=[4, 64, 64])}))
    (root / 'split_manifest.json').write_text(json.dumps(dict(
        patch_size=[4, 32, 32], train_stride=[4, 16, 16], seed=1,
        volumes=[dict(name='volume', shape=[4, 64, 64], blocks=[dict(index=0, y0=0, y1=32, x0=32, x1=64, split='val')])],
        files=[dict(filename='b.tiff', split='val', z0=0, y0=0, x0=32, shape=[4, 32, 32])])))
    (root / 'split_info.json').write_text(json.dumps(dict(files, rescale_factor=1.0)))
    return nuclei, cells


class DownscaleTests(unittest.TestCase):
    def test_image_is_the_mean_of_each_block(self):
        image = np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)
        small = down.downscale_image(image, 2)
        self.assertEqual((small.shape, small.dtype), ((2, 2, 3, 3), np.uint8))
        np.testing.assert_array_equal(small, np.round(image.astype(float).reshape(2, 2, 2, 3, 2, 3).mean((2, 4))))
        odd = np.arange(7 * 5, dtype=np.float32).reshape(1, 7, 5)             # the remainder rows and columns are dropped
        np.testing.assert_allclose(down.downscale_image(odd, 3), [[[odd[0, :3, :3].mean()], [odd[0, 3:6, :3].mean()]]])

    def test_label_of_most_pixels_wins_and_ties_go_to_the_top_left(self):
        labels = np.array([[[1, 2, 5, 5, 7, 8, 3, 4],
                            [2, 2, 6, 5, 9, 0, 4, 3]]], np.uint16)
        np.testing.assert_array_equal(down.downscale_labels(labels, 2), [[[2, 5, 7, 3]]])
        three = np.zeros((1, 3, 3), np.uint16); three[0, 1:, :] = 4; three[0, 0, 0] = 9
        self.assertEqual(down.downscale_labels(three, 3)[0, 0, 0], 4)

    def test_factor_two_gives_the_labels_of_the_measured_half_resolution_run(self):
        """The rule of the _exp script that wrote patches_split_hela_xy2 for the h1 run."""
        labels = np.random.default_rng(0).integers(0, 3, (3, 10, 12)).astype(np.uint16)
        a, b, c, d = labels[..., 0::2, 0::2], labels[..., 1::2, 0::2], labels[..., 0::2, 1::2], labels[..., 1::2, 1::2]
        rule = np.where((a == b) | (a == c) | (a == d), a, np.where((b == c) | (b == d), b, np.where(c == d, c, a)))
        np.testing.assert_array_equal(down.downscale_labels(labels, 2), rule)

    def test_dataset_copy(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root, out = Path(tmp) / 'native', Path(tmp) / 'half'
            nuclei, cells = small_dataset(root)
            lost = quiet(down.downscale_split, root, out, 2)
            self.assertEqual(lost, {'nucleus_masks': 2, 'cytoplasm_masks': 0})                # the one-pixel nucleus of a.tiff and b.tiff
            self.assertEqual(tifffile.imread(out / 'train' / 'images' / 'a.tiff').shape, (4, 16, 16, 3))
            np.testing.assert_array_equal(tifffile.imread(out / 'train' / 'nucleus_masks' / 'a.tiff'), down.downscale_labels(nuclei, 2))
            np.testing.assert_array_equal(tifffile.imread(out / 'val' / 'cytoplasm_masks' / 'b.tiff'), down.downscale_labels(cells, 2))
            self.assertFalse((out / 'train' / 'nucleus_masks' / 'c.tiff').exists())       # partial annotation stays partial
            self.assertTrue((out / 'train' / 'cytoplasm_masks' / 'c.tiff').exists())
            for split, name in (('train', 'a'), ('val', 'b'), ('train_regions', 'volume')):
                path = Path(split) / 'assignments' / (name + '.json')
                self.assertEqual((out / path).read_text(), (root / path).read_text())
            self.assertEqual(json.loads((out / 'train' / 'meta.json').read_text())[0]['cells_median_extent_p'], [4, 8, 6])
            regions = json.loads((out / 'train_regions' / 'regions.json').read_text())['volume.tiff']
            self.assertEqual(regions, dict(train_blocks=[[0, 16, 0, 16], [16, 32, 0, 16]], block_size=[16, 16], shape=[4, 32, 32]))
            self.assertEqual(tifffile.imread(out / 'train_regions' / 'images' / 'volume.tiff').shape, (4, 32, 32))
            manifest = json.loads((out / 'split_manifest.json').read_text())
            self.assertEqual((manifest['patch_size'], manifest['train_stride'], manifest['volumes'][0]['shape']),
                             ([4, 16, 16], [4, 8, 8], [4, 32, 32]))
            self.assertEqual(manifest['volumes'][0]['blocks'][0], dict(index=0, y0=0, y1=16, x0=16, x1=32, split='val'))
            self.assertEqual({k: manifest['files'][0][k] for k in ('z0', 'y0', 'x0', 'shape')}, dict(z0=0, y0=0, x0=16, shape=[4, 16, 16]))
            info = json.loads((out / 'split_info.json').read_text())
            self.assertEqual((info['train'], info['xy_downscale']), (['a.tiff', 'c.tiff'], 2))
            self.assertEqual(downscaling(out), (root.resolve(), 2))
            # A copy of the copy keeps pointing at the native dataset, with the total factor.
            quiet(down.downscale_split, out, Path(tmp) / 'quarter', 2)
            self.assertEqual(downscaling(Path(tmp) / 'quarter'), (root.resolve(), 4))
            with self.assertRaises(FileExistsError):
                quiet(down.downscale_split, root, out, 2)
            with self.assertRaises(ValueError):
                quiet(down.downscale_split, root, Path(tmp) / 'one', 1)
            with self.assertRaises(ValueError):
                downscaling(root)                                                     # not a downscaled copy

    def test_enlarged_labels_take_the_native_shape(self):
        labels = np.arange(1, 1 + 2 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3)
        big = enlarge_labels(labels, 2, (2, 7, 5))
        self.assertEqual(big.shape, (2, 7, 5))
        np.testing.assert_array_equal(big[:, :6, :], np.repeat(np.repeat(labels, 2, 1), 2, 2)[:, :, :5])
        self.assertFalse(big[:, 6].any())                                             # the odd native row is background
        self.assertEqual(enlarge_labels(labels, 2, (2, 4, 4)).shape, (2, 4, 4))


class PresetTests(unittest.TestCase):
    def load(self, raw):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw = json.loads(json.dumps(raw))
            raw['output']['basedir'] = str(Path(tmp) / 'models')
            path = Path(tmp) / 'config.json'; path.write_text(json.dumps(raw))
            return quiet(CytoNucsConfig.from_json, path)

    def test_preset_is_v4_at_half_the_xy_resolution(self):
        base = json.loads((ROOT / 'config_cytonucs_v4.json').read_text())
        raw = json.loads((ROOT / PRESET).read_text())
        changed = {(section, key) for section in raw if isinstance(raw[section], dict)
                   for key in raw[section] if raw[section][key] != base[section].get(key)}
        self.assertEqual(changed, {
            ('experiment', 'name'), ('inference', 'checkpoint_path'), ('data', 'dataset_dir'), ('training', 'train_epochs'),
            ('model', 'anisotropy'), ('training', 'train_patch_size'), ('inference', 'refinement_patch_size'),
            ('inference', 'nms_border_exclude_xy'), ('loss_params', 'containment_margin'),
            ('loss_params', 'max_assignment_distance'), ('loss_params', 'wbr_sigma')})
        self.assertEqual(set(raw), set(base))
        self.assertIn(raw['experiment']['name'], raw['inference']['checkpoint_path'])
        self.assertTrue(raw['data']['dataset_dir'].rstrip('/').endswith('_xy2'))
        # Every length in XY pixels is halved, Z is not.
        for section, key in (('training', 'train_patch_size'), ('inference', 'refinement_patch_size')):
            self.assertEqual(raw[section][key], base[section][key][:1] + [v // 2 for v in base[section][key][1:]], key)
        self.assertEqual(raw['model']['anisotropy'], [base['model']['anisotropy'][0] / 2, 1, 1])
        for key in ('containment_margin', 'max_assignment_distance', 'wbr_sigma'):
            self.assertEqual(raw['loss_params'][key], base['loss_params'][key] / 2, key)
        self.assertEqual(raw['inference']['nms_border_exclude_xy'], base['inference']['nms_border_exclude_xy'] // 2)
        cfg = self.load(raw)
        self.assertEqual((list(cfg.train_patch_size), cfg.unet_n_depth, cfg.unet_norm_groups), ([16, 160, 160], 4, 8))
        self.assertEqual(list(cfg.refinement_patch_size)[1:], list(cfg.train_patch_size)[1:])
        self.assertTrue(cfg.train_region_crops)


if __name__ == '__main__':
    unittest.main()
