"""dataset_prep/make_spatial_split_patches.py: no voxel of a source volume is in two splits."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
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
        # image ZYXC, nuclei, cells; the second volume is one slice deeper than the patch
        cls.sources = dict(first=source(0, (40, 150, 200)), second=source(1, (17, 100, 150)))
        cls.read = []

        def volumes():
            for name, (image, nuclei, cells) in cls.sources.items():
                cls.read.append(name)
                # the second image is stored channels-first (ZCYX), its masks with a singleton axis
                given = dict(image=image, nucleus=nuclei, cytoplasm=cells) if name == 'first' else \
                    dict(image=np.moveaxis(image, -1, 1), nucleus=nuclei[:, None], cytoplasm=cells[:, None])
                yield dict(given, name=name)

        cls.manifest = prep.build(volumes(), cls.out, PATCH, STRIDE, augmentations=2, val_fraction=0.15, test_fraction=0.15, seed=3)
        cls.volumes = {v['name']: v for v in cls.manifest['volumes']}
        cls.files = cls.manifest['files']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def members(self, name, split=None):
        return [f for f in self.files if f['volume'] == name and split in (None, f['split'])]

    def test_blocks_tile_the_plane_and_every_split_has_one(self):
        self.assertEqual(list(self.volumes), ['first', 'second']); self.assertEqual(self.read, ['first', 'second'])
        for name, volume in self.volumes.items():
            plane = self.sources[name][1].shape[1:]
            self.assertEqual(volume['shape'], list(self.sources[name][1].shape))
            cover = np.zeros(plane, int)
            for b in volume['blocks']:
                cover[b['y0']:b['y1'], b['x0']:b['x1']] += 1
                self.assertGreaterEqual(min(b['y1'] - b['y0'], b['x1'] - b['x0']), PATCH[1])
            self.assertTrue((cover == 1).all())
            self.assertEqual({b['split'] for b in volume['blocks']}, {'train', 'val', 'test'})
            self.assertTrue(all(volume['summary'][s]['files'] == len(self.members(name, s)) > 0 for s in prep.SPLITS))
        self.assertEqual({s: v['files'] for s, v in self.manifest['summary'].items()},
                         {s: sum(f['split'] == s for f in self.files) for s in prep.SPLITS})

    def test_no_voxel_in_two_splits_and_files_stay_in_the_blocks_of_their_split(self):
        for name, volume in self.volumes.items():
            blocks = volume['blocks']
            owner = {s: np.zeros(self.sources[name][1].shape[1:], bool) for s in prep.SPLITS}
            block_of = np.zeros(self.sources[name][1].shape[1:], int)
            for b in blocks:
                block_of[b['y0']:b['y1'], b['x0']:b['x1']] = b['index']
            for f in self.members(name):
                window = (slice(f['y0'], f['y0'] + f['shape'][1]), slice(f['x0'], f['x0'] + f['shape'][2]))
                self.assertEqual(f['blocks'], np.unique(block_of[window]).tolist(), f['filename'])
                self.assertEqual({blocks[b]['split'] for b in f['blocks']}, {f['split']}, f['filename'])
                self.assertTrue(f['filename'].startswith(name + '_z'))
                owner[f['split']][window] = True
            for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                self.assertFalse((owner[a] & owner[b]).any(), (name, a, b))
        self.assertEqual(len({f['filename'] for f in self.files}), len(self.files))

    def test_training_windows_cross_neighbouring_train_blocks_and_cover_every_train_block(self):
        # 4 x 4 blocks of one patch, as a 2048 px volume with 512 px patches; four blocks are held out.
        patch, stride, n = (48, 48), (40, 40), 4
        blocks = prep.block_grid((n * 48, n * 48), patch)
        held_out = {1, 6, 8, 15}                                   # block 11 (row 2, column 3) lies above held-out block 15
        train_blocks = [b for i, b in enumerate(blocks) if i not in held_out]
        region = np.zeros((n * 48, n * 48), bool)
        for y0, y1, x0, x1 in train_blocks:
            region[y0:y1, x0:x1] = True
        starts = prep.train_windows(region, train_blocks, patch, stride)
        covered = np.zeros_like(region)
        for y, x in starts:
            self.assertTrue(region[y:y + 48, x:x + 48].all(), (y, x))        # never a held-out voxel
            covered[y:y + 48, x:x + 48] = True
        np.testing.assert_array_equal(covered, region)                       # every train voxel is in some patch
        self.assertEqual(len(starts), len(set(starts)))
        crossing = [(y, x) for y, x in starts if y % 48 or x % 48]
        self.assertTrue(crossing)                                            # objects on a block border are seen whole
        self.assertIn((96, 144), starts)                                     # block 11: covered only by its own window
        # Without held-out blocks these are exactly the notebook's windows.
        everything = np.ones_like(region)
        grid = prep.patch_starts(n * 48, 48, 40)
        self.assertEqual(prep.train_windows(everything, blocks, patch, stride), sorted((y, x) for y in grid for x in grid))
        # In the written dataset some train patch spans two blocks.
        self.assertTrue(any(len(f['blocks']) > 1 for f in self.members('first', 'train')))

    def test_validation_and_test_are_originals_once_per_block(self):
        for name, volume in self.volumes.items():
            for split in ('val', 'test'):
                members = self.members(name, split)
                self.assertTrue(members)
                self.assertTrue(all(f['augmentation_index'] == 0 and f['augmentation_params'] == {} for f in members))
                for f in members:
                    (index,) = f['blocks']; b = volume['blocks'][index]
                    self.assertEqual((f['y0'], f['x0'], f['shape'][1], f['shape'][2]),
                                     (b['y0'], b['x0'], b['y1'] - b['y0'], b['x1'] - b['x0']))
                self.assertEqual(sorted({f['z0'] for f in members}), [0, 16, 24] if name == 'first' else [0])

    def test_depth_windows_do_not_repeat_a_slab_for_one_extra_slice(self):
        cases = {(16, 16, 8): [0], (10, 16, 8): [0], (17, 16, 8): [0], (17, 16, 16): [0], (40, 16, 8): [0, 8, 16, 24],
                 (40, 16, 16): [0, 16, 24], (45, 16, 8): [0, 8, 16, 24, 29], (45, 16, 16): [0, 16, 29], (20, 16, 8): [0, 4]}
        for (size, patch, stride), expected in cases.items():
            self.assertEqual(prep.slab_starts(size, patch, stride), expected, (size, patch, stride))
        self.assertEqual(sorted({f['z0'] for f in self.members('second', 'train')}), [0])
        self.assertEqual(sorted({f['z0'] for f in self.members('first', 'train')}), [0, 8, 16, 24])

    def test_channel_axis_follows_the_mask_shape(self):
        mask = (17, 100, 150)
        self.assertIsNone(prep.channel_axis(mask, mask))
        self.assertEqual(prep.channel_axis((17, 100, 150, 3), mask), 3)
        self.assertEqual(prep.channel_axis((17, 3, 100, 150), mask), 1)
        self.assertEqual(prep.channel_axis((3, 17, 100, 150), mask), 0)
        for wrong in ((16, 100, 150, 3), (17, 100, 150, 3, 1), (100, 150)):
            with self.assertRaises(ValueError):
                prep.channel_axis(wrong, mask)
        for name, volume in self.volumes.items():
            self.assertEqual((volume['channels'], volume['dtype'], volume['nucleus_channel']), (3, 'uint8', 1))
            means = volume['channel_means']
            self.assertGreater(means['nuclei'][1], means['cell_body'][1] + 100)       # source(): nuclei are in channel 1
            self.assertGreater(means['cell_body'][0], means['background'][0] + 50)    # and the cell body in channel 0

    def test_original_tiles_equal_the_source_crop(self):
        for f in [f for f in self.files if f['augmentation_index'] == 0][::5]:
            image, nuclei, cells = self.sources[f['volume']]
            sl = (slice(f['z0'], f['z0'] + f['shape'][0]), slice(f['y0'], f['y0'] + f['shape'][1]),
                  slice(f['x0'], f['x0'] + f['shape'][2]))
            folder = self.out / f['split']
            np.testing.assert_array_equal(tifffile.imread(folder / 'images' / f['filename']), image[sl])
            for sub, labels in (('nucleus_masks', nuclei), ('cytoplasm_masks', cells)):
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
            by_origin.setdefault((f['volume'], f['z0'], f['y0'], f['x0']), []).append(f)
        for members in by_origin.values():
            self.assertEqual(len(members), 3)
            self.assertEqual(len({(f['n_nuclei'], f['n_cells'], tuple(f['shape'])) for f in members}), 1)
        copy = next(f for f in train if f['augmentation_index'] == 1)
        self.assertEqual(tifffile.imread(self.out / 'train' / 'images' / copy['filename']).dtype, np.uint8)

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
        image, nuclei, cells = self.sources['second']
        volume = dict(name='v', image=image, nucleus=nuclei, cytoplasm=cells)
        with self.assertRaises(FileExistsError):
            prep.build([volume], self.out, PATCH, STRIDE)
        with self.assertRaisesRegex(ValueError, 'cannot hold'):
            prep.split_blocks(np.ones((60, 60)), (48, 48), .15, .15, 0)
        for wrong, message in ((dict(volume, cytoplasm=cells[1:]), 'one shape'), (dict(volume, image=image[:, :-1]), 'does not match'),
                               (dict(volume, name='again'), None)):
            out = Path(self.tmp.name) / f'refused {message}'
            with self.assertRaisesRegex(ValueError, message or 'Two volumes are named again'):
                prep.build([dict(volume, name='again'), wrong] if message is None else [wrong], out, PATCH, STRIDE, augmentations=0)
        with self.assertRaisesRegex(ValueError, r'gray: 1 channel\(s\), v has 3'):
            prep.build([volume, dict(volume, name='gray', image=image[..., 0])], Path(self.tmp.name) / 'mixed', PATCH, STRIDE, augmentations=0)

    def test_command_line_takes_folders_with_the_names_found_on_the_server(self):
        root = Path(self.tmp.name) / 'server'
        hela, ea, hk = ('HeLa_MRC_DLS Acquire 2_012_Target_z00', '2D_EA.hy926_DAPI_Membright560_Phalloidin(green)_251210_Series002_z00',
                        '2D_HK_DAPI_Membright560_Phalloidin647_251209_Series001_z00')
        image, nuclei, cells = self.sources['second']
        for sub in ('images/cannot be used', 'nuc_masks/patches/nuc_masks', 'cyto_masks'):
            (root / sub).mkdir(parents=True)
        tifffile.imwrite(root / 'images' / f'{hela}_merged_rgb_plain_stack.tif', image)
        tifffile.imwrite(root / 'images' / f'{ea}.tif', image[:16])
        tifffile.imwrite(root / 'images' / 'cannot be used' / 'broken.tif', image[:2])
        (root / 'images' / 'Thumbs.db').write_bytes(b'x')
        for folder, labels in (('nuc_masks', nuclei), ('cyto_masks', cells)):
            tifffile.imwrite(root / folder / f'{hela}.tif', labels)
            tifffile.imwrite(root / folder / f'{ea}.tif', labels[:16])
            tifffile.imwrite(root / folder / f'{hk}.tif', labels[:3])               # a mask without an image is not used
        tifffile.imwrite(root / 'nuc_masks' / 'patches' / 'nuc_masks' / 'old patch.tif', nuclei[:2])
        command = ['make_spatial_split_patches.py', '--image', str(root / 'images'), '--nucleus-mask', str(root / 'nuc_masks'),
                   '--cytoplasm-mask', str(root / 'cyto_masks'), '--patch-size', '16', '48', '48', '--train-stride', '8', '40', '40',
                   '--augmentations', '1']
        text = io.StringIO()
        with mock.patch.object(sys, 'argv', command + ['--output', str(root / 'out')]), contextlib.redirect_stdout(text):
            prep.main()
        manifest = json.loads((root / 'out' / 'split_manifest.json').read_text())
        names = ['2D_EA_hy926_DAPI_Membright560_Phalloidin_green_251210_Series002_z00',
                 'HeLa_MRC_DLS_Acquire_2_012_Target_z00_merged_rgb_plain_stack']
        self.assertEqual([v['name'] for v in manifest['volumes']], names)
        self.assertEqual([v['shape'] for v in manifest['volumes']], [[16, 100, 150], [17, 100, 150]])
        self.assertEqual([Path(v['source']['nucleus']['path']).name for v in manifest['volumes']], [f'{ea}.tif', f'{hela}.tif'])
        self.assertEqual([Path(v['source']['cytoplasm']['path']).parent.name for v in manifest['volumes']], ['cyto_masks'] * 2)
        for split in prep.SPLITS:
            written = sorted(p.name for p in (root / 'out' / split / 'images').iterdir())
            self.assertEqual(written, sorted(f['filename'] for f in manifest['files'] if f['split'] == split))
            self.assertEqual({n for n in names if any(w.startswith(n + '_z') for w in written)}, set(names), split)
        self.assertIn('Total:', text.getvalue()); self.assertNotIn('WARNING', text.getvalue())
        # A volume whose masks do not fit stops the run before anything is written; every problem is listed.
        tifffile.imwrite(root / 'cyto_masks' / f'{ea}.tif', cells[:15])
        tifffile.imwrite(root / 'images' / f'{hk}.tif', image[:3, :40])
        with mock.patch.object(sys, 'argv', command + ['--output', str(root / 'out2')]), contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(SystemExit) as stopped:
            prep.main()
        self.assertIn('Nothing was written', str(stopped.exception))
        self.assertIn('Series002_z00: the masks must be ZYX volumes of one shape', str(stopped.exception))
        self.assertIn('Series001_z00: image (3, 40, 150, 3) does not match the masks (3, 100, 150)', str(stopped.exception))
        self.assertFalse((root / 'out2').exists())

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
