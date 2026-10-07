"""dataset_prep/make_spatial_split_patches.py: no voxel of the source volume is in two splits."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from test_fixes import ROOT

sys.path.insert(0, str(ROOT / 'dataset_prep'))
import make_spatial_split_patches as prep

PATCH, STRIDE = (16, 48, 48), (8, 40, 40)


def source(seed=0, shape=(40, 150, 200)):
    """RGB volume with cells (one nucleus each) spread over the whole XY plane."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    cells = np.zeros(shape, np.uint16); nuclei = np.zeros(shape, np.uint16)
    label = 0
    for cy in range(12, shape[1], 25):
        for cx in range(12, shape[2], 25):
            for cz in (10, 28):
                label += 1
                c = (cz + rng.uniform(-2, 2), cy + rng.uniform(-3, 3), cx + rng.uniform(-3, 3))
                d = ((zz - c[0]) / 5.) ** 2 + ((yy - c[1]) / 10.) ** 2 + ((xx - c[2]) / 10.) ** 2
                cells[(d <= 1) & (cells == 0)] = label
                nuclei[(d <= .2) & (cells == label)] = 500 + label
    image = np.stack([40 + 90 * (cells > 0), 30 + 150 * (nuclei > 0), np.full(shape, 20)], -1)
    image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    return image, nuclei, cells


class SpatialSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        cls.out = Path(cls.tmp.name) / 'dataset'
        cls.image, cls.nuclei, cls.cells = source()
        cls.manifest = prep.build(cls.image, cls.nuclei, cls.cells, cls.out, PATCH, STRIDE, augmentations=2,
                                  val_fraction=0.15, test_fraction=0.15, seed=3)
        cls.blocks = cls.manifest['blocks']; cls.files = cls.manifest['files']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_blocks_tile_the_plane_and_every_split_has_one(self):
        cover = np.zeros(self.nuclei.shape[1:], int)
        for b in self.blocks:
            cover[b['y0']:b['y1'], b['x0']:b['x1']] += 1
            self.assertGreaterEqual(min(b['y1'] - b['y0'], b['x1'] - b['x0']), PATCH[1])
        self.assertTrue((cover == 1).all())
        self.assertEqual({b['split'] for b in self.blocks}, {'train', 'val', 'test'})

    def test_no_voxel_in_two_splits_and_files_stay_in_their_block(self):
        owner = {s: np.zeros(self.nuclei.shape[1:], bool) for s in prep.SPLITS}
        for f in self.files:
            b = self.blocks[f['block']]
            self.assertEqual(f['split'], b['split'])
            y1, x1 = f['y0'] + f['shape'][1], f['x0'] + f['shape'][2]
            self.assertTrue(b['y0'] <= f['y0'] and y1 <= b['y1'] and b['x0'] <= f['x0'] and x1 <= b['x1'], f['filename'])
            owner[f['split']][f['y0']:y1, f['x0']:x1] = True
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            self.assertFalse((owner[a] & owner[b]).any(), (a, b))

    def test_validation_and_test_are_originals_once_per_block(self):
        for split in ('val', 'test'):
            members = [f for f in self.files if f['split'] == split]
            self.assertTrue(members)
            self.assertTrue(all(f['augmentation_index'] == 0 and f['augmentation_params'] == {} for f in members))
            for f in members:
                b = self.blocks[f['block']]
                self.assertEqual((f['y0'], f['x0'], f['shape'][1], f['shape'][2]),
                                 (b['y0'], b['x0'], b['y1'] - b['y0'], b['x1'] - b['x0']))
            z = sorted({f['z0'] for f in members})
            self.assertEqual(z, prep.patch_starts(self.nuclei.shape[0], PATCH[0], PATCH[0]))

    def test_original_tiles_equal_the_source_crop(self):
        for f in [f for f in self.files if f['augmentation_index'] == 0][::7]:
            sl = (slice(f['z0'], f['z0'] + f['shape'][0]), slice(f['y0'], f['y0'] + f['shape'][1]),
                  slice(f['x0'], f['x0'] + f['shape'][2]))
            folder = self.out / f['split']
            np.testing.assert_array_equal(tifffile.imread(folder / 'images' / f['filename']), self.image[sl])
            for sub, labels in (('nucleus_masks', self.nuclei), ('cytoplasm_masks', self.cells)):
                written = tifffile.imread(folder / sub / f['filename']); crop = labels[sl]
                np.testing.assert_array_equal(written > 0, crop > 0)
                # sequential relabelling is a one-to-one map of the source labels
                pairs = set(zip(written[crop > 0].tolist(), crop[crop > 0].tolist()))
                self.assertEqual(len(pairs), len({p[0] for p in pairs})); self.assertEqual(len(pairs), len({p[1] for p in pairs}))
                self.assertEqual(sorted({p[0] for p in pairs}), list(range(1, len(pairs) + 1)))

    def test_train_has_copies_with_the_same_objects(self):
        train = [f for f in self.files if f['split'] == 'train']
        self.assertEqual({f['augmentation_index'] for f in train}, {0, 1, 2})
        by_origin = {}
        for f in train:
            by_origin.setdefault((f['z0'], f['y0'], f['x0']), []).append(f)
        for members in by_origin.values():
            self.assertEqual(len(members), 3)
            self.assertEqual(len({(f['n_nuclei'], f['n_cells'], tuple(f['shape'])) for f in members}), 1)
        copy = next(f for f in train if f['augmentation_index'] == 1)
        self.assertEqual(tifffile.imread(self.out / 'train' / 'images' / copy['filename']).dtype, self.image.dtype)

    def test_layout_read_by_training(self):
        info = json.loads((self.out / 'split_info.json').read_text())
        for split in prep.SPLITS:
            names = sorted(f['filename'] for f in self.files if f['split'] == split)
            self.assertEqual(sorted(info[split]), names)
            for sub in ('images', 'nucleus_masks', 'cytoplasm_masks'):
                self.assertEqual(sorted(p.name for p in (self.out / split / sub).iterdir()), names)
            self.assertEqual(sorted(p.stem for p in (self.out / split / 'assignments').iterdir()), [Path(n).stem for n in names])
            self.assertEqual(len(json.loads((self.out / split / 'meta.json').read_text())), len(names))
        pairs = json.loads((self.out / 'val' / 'assignments' / (Path(info['val'][0]).stem + '.json')).read_text())
        self.assertTrue(pairs)
        self.assertEqual(info['rescale_factor'], 1.0)

    def test_refuses_existing_output_and_too_small_volume(self):
        with self.assertRaises(FileExistsError):
            prep.build(self.image, self.nuclei, self.cells, self.out, PATCH, STRIDE)
        with self.assertRaisesRegex(ValueError, 'cannot hold'):
            prep.split_blocks(np.ones((60, 60)), (48, 48), .15, .15, 0)

    def test_mask_given_as_folder_is_found_like_in_the_notebook(self):
        folder = Path(self.tmp.name) / 'masks'
        (folder / 'patches').mkdir(parents=True)                 # old patch outputs live in a subfolder
        for name in ('other name.tif', 'patches/stack.tif'):
            (folder / name).write_bytes(b'')
        image = Path('images') / 'stack.tif'
        self.assertEqual(prep.resolve_mask(folder, image), folder / 'other name.tif')          # the only TIFF
        (folder / 'stack.tiff').write_bytes(b'')
        self.assertEqual(prep.resolve_mask(folder, image), folder / 'stack.tiff')              # same stem
        (folder / 'stack.tif').write_bytes(b'')
        self.assertEqual(prep.resolve_mask(folder, image), folder / 'stack.tif')               # same name
        self.assertEqual(prep.resolve_mask(folder / 'other name.tif', image), folder / 'other name.tif')
        with self.assertRaises(FileNotFoundError):
            prep.resolve_mask(folder, Path('images') / 'unrelated.tif')
        # Several volumes in one folder: the mask is the file the image name starts with.
        shared = Path(self.tmp.name) / 'shared masks'
        shared.mkdir()
        for name in ('2D_EA.hy926_DAPI_Membright560_Phalloidin(green)_251210_Series002_z00.tif',
                     '2D_HK_DAPI_Membright560_Phalloidin647_251209_Series001_z00.tif',
                     'HeLa_MRC_DLS Acquire 2_012_Target_z00.tif', 'HeLa_MRC.tif'):
            (shared / name).write_bytes(b'')
        image = Path('images') / 'HeLa_MRC_DLS Acquire 2_012_Target_z00_merged_rgb_plain_stack.tif'
        self.assertEqual(prep.resolve_mask(shared, image), shared / 'HeLa_MRC_DLS Acquire 2_012_Target_z00.tif')
        with self.assertRaises(FileNotFoundError):
            prep.resolve_mask(shared, Path('images') / 'Other_volume_stack.tif')

    def test_split_follows_labelled_voxels_not_block_count(self):
        # 12 blocks, labelled voxels only in the left half: val/test must not be empty blocks.
        foreground = np.zeros((96, 288)); foreground[:, :144] = 1.
        blocks, assignment = prep.split_blocks(foreground, (48, 48), .15, .15, 0)
        for split in ('val', 'test'):
            share = sum(foreground[b[0]:b[1], b[2]:b[3]].sum() for b, s in zip(blocks, assignment) if s == split) / foreground.sum()
            self.assertGreater(share, .1, split)

    def test_overfit_subset_has_the_same_originals_in_train_and_val(self):
        import make_overfit_subset as subset
        out = Path(self.tmp.name) / 'subset'
        info = subset.build(self.out, out, count=3)
        self.assertEqual(info['train'], info['val']); self.assertEqual(len(set(info['train'])), 3)
        self.assertTrue(all(Path(n).stem.endswith('_a00') and info['cells_per_volume'][n] > 0 for n in info['train']))
        for name in info['train']:
            for sub in ('images', 'nucleus_masks', 'cytoplasm_masks'):
                a, b = (tifffile.imread(out / split / sub / name) for split in ('train', 'val'))
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(a, tifffile.imread(self.out / 'train' / sub / name))
            self.assertTrue((out / 'val' / 'assignments' / (Path(name).stem + '.json')).exists())
        with self.assertRaises(FileExistsError):
            subset.build(self.out, out, count=3)
        # The run config: the user's config continued on the subset; only these keys change.
        from cytonucs_config import CytoNucsConfig
        base = json.loads((ROOT / 'config_cytonucs_v3_2.json').read_text(encoding='utf-8'))
        path = subset.write_config(ROOT / 'config_cytonucs_v3_2.json', out, 'models/run/weights_epoch_830.weights.h5', epochs=40)
        raw = json.loads(path.read_text(encoding='utf-8'))
        changed = {(s, k): raw[s][k] for s in base if isinstance(base[s], dict) for k in base[s] if base[s][k] != raw[s][k]}
        name = base['experiment']['name'] + '_overfit_check'
        self.assertEqual(changed, {
            ('experiment', 'name'): name, ('data', 'dataset_dir'): str(out), ('augmentation', 'enable'): False,
            ('training', 'resume_from_checkpoint'): 'models/run/weights_epoch_830.weights.h5',
            ('training', 'train_epochs'): 870, ('training', 'lr_decay_rate'): 1.0, ('training', 'reset_best_on_resume'): True,
            ('inference', 'checkpoint_path'): f"{base['output']['basedir']}/{name}/weights_best.weights.h5",
            ('inference', 'mask_thresh_nucleus'): base['inference']['mask_thresh'],
            ('inference', 'mask_thresh_cytoplasm'): base['inference']['mask_thresh']})
        self.assertEqual(set(raw), set(base))
        with self.assertRaisesRegex(ValueError, 'numbered checkpoint'):
            subset.write_config(ROOT / 'config_cytonucs_v3_2.json', out, 'models/run/weights_best.weights.h5')


if __name__ == '__main__':
    unittest.main()
