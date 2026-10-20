"""Support for the public datasets of the paper: 2D runs measure their object scale and write geometry.json (a
checkpoint is read back with it),
hard-link subset views of a dataset, KiTS23 instance levels (masses inside kidneys)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from test_fixes import config, ROOT
from cytonucs_config import CytoNucsConfig

sys.path.insert(0, str(ROOT / 'dataset_prep'))


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def config_2d(basedir):
    """The v4 preset as a 2D colour run (the settings of the BBBC020 run)."""
    d = json.loads((ROOT / 'config_cytonucs_v4.json').read_text())
    d['experiment']['name'] = 'flat'
    d['output']['basedir'] = str(basedir)
    d['model'].update(ndim=2, anisotropy=[1, 1], nucleus_grid=[2, 2], cytoplasm_grid=[2, 2], n_channel_in=3,
                      unet_pool_schedule=[[2, 2]] * 4, unet_kernel_schedule=[[3, 3]] * 5)
    d['training'].update(train_patch_size=[64, 64], dist_target_stride=[2, 2], train_region_crops=False)
    d['inference'].update(tiling_fixed_axes=[0, 1], refinement_patch_size=[64, 64])
    d['loss_params'].update(cell_z_dice_weight=0., nucleus_z_dice_weight=0.)
    d['augmentation'].update(cell_z_pair_prob=0., nucleus_z_pair_prob=0., zoom_z_range=None, flip_z_prob=0.)
    return d


class Geometry2DTests(unittest.TestCase):
    def test_2d_run_writes_geometry_and_its_checkpoint_loads(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            d = config_2d(tmp)
            path = Path(tmp) / 'train.json'
            path.write_text(json.dumps(d))
            cfg = quiet(CytoNucsConfig.from_json, str(path))
            labels = np.zeros((64, 64), np.uint16)
            labels[10:20, 10:20], labels[30:50, 30:50] = 1, 2
            quiet(cfg.set_anisotropy_from_labels, [labels], [labels])
            geometry = json.loads((Path(tmp) / 'flat' / 'geometry.json').read_text())
            self.assertEqual(geometry['n_rays'], d['model']['n_rays'])
            self.assertEqual(geometry['nucleus_grid'], [2, 2])
            # Scale = half the diagonal of the median box (10 and 20 px wide objects: 15 x 15), as in 3D;
            # before, 2D runs kept the 3D defaults (15 / 60 px) whatever the objects.
            self.assertEqual(geometry['median_max_dist_nucleus'], 11.0)
            self.assertEqual(geometry['rays_anisotropy_nucleus'], [1.0, 1.0])
            d['mode'] = 'inference'
            d['inference']['checkpoint_path'] = str(Path(tmp) / 'flat' / 'weights_best.weights.h5')
            path.write_text(json.dumps(d))
            loaded = quiet(CytoNucsConfig.from_json, str(path))        # raised before: geometry.json missing
            self.assertEqual(tuple(loaded.nucleus_grid), (2, 2))
            self.assertEqual(loaded.median_max_dist_nucleus, 11.0)

    def test_3d_geometry_file_is_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = quiet(config, anisotropy=(1, 1, 1), median_max_dist_nucleus=None, median_max_dist_cytoplasm=None)
            cfg.geometry_path = str(Path(tmp) / 'geometry.json')
            labels = np.zeros((8, 96, 96), np.uint16)
            for k in range(40):
                y, x = 4 + 14 * (k // 6), 4 + 14 * (k % 6)
                labels[2:6, y:y + 10, x:x + 10] = k + 1
            quiet(cfg.set_anisotropy_from_labels, [labels], [labels])
            geometry = json.loads(Path(cfg.geometry_path).read_text())
            self.assertEqual(sorted(geometry), sorted(['n_rays', 'rays_anisotropy_nucleus', 'rays_anisotropy_cytoplasm',
                                                       'nucleus_grid', 'cytoplasm_grid', 'median_max_dist_nucleus',
                                                       'median_max_dist_cytoplasm']))
            self.assertIsInstance(geometry['median_max_dist_nucleus'], float)
            self.assertEqual(geometry['median_max_dist_nucleus'], float(cfg.median_max_dist_nucleus))


class TwoLevelScoringTests(unittest.TestCase):
    def test_score_hierarchy_runs_a_two_level_model_with_named_levels_and_ignore_regions(self):
        import score_hierarchy
        from cytonucs_hierarchy import predict_hierarchy
        from cytonucs_model import build_cytonucs_model
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            d = config_2d(root / 'models')
            d['model'].update(levels=['spot', 'blob'], n_rays=8, use_gpu=False, unet_n_depth=2, unet_n_filter_base=4,
                              net_conv_after_unet=8, mask_refinement_filters=8, unet_norm_groups=2,
                              unet_pool_schedule=[[2, 2]] * 2, unet_kernel_schedule=[[3, 3]] * 3)
            d['data']['dataset_dir'] = str(root / 'data') + '/'
            d['inference'].update(prob_thresh_nucleus=0.99, prob_thresh_cytoplasm=0.99)
            d['gpu_config']['visible_devices'] = '-1'
            for sub in ('images', 'spot_masks', 'blob_masks', 'spot_unknown'):
                (root / 'data' / 'test' / sub).mkdir(parents=True)
            blob = np.zeros((64, 64), np.uint16); blob[8:40, 8:40] = 1
            spot = np.zeros((64, 64), np.uint16); spot[16:24, 16:24] = 1
            image = np.stack([(blob > 0) * .5 + (spot > 0) * .5] * 3, -1).astype(np.float32)
            tifffile.imwrite(root / 'data/test/images/a.tif', image)
            tifffile.imwrite(root / 'data/test/blob_masks/a.tif', blob, photometric='minisblack')
            tifffile.imwrite(root / 'data/test/spot_masks/a.tif', spot, photometric='minisblack')
            tifffile.imwrite(root / 'data/test/spot_unknown/a.tif', np.zeros((64, 64), np.uint8), photometric='minisblack')
            path = root / 'train.json'
            path.write_text(json.dumps(d))
            cfg = quiet(CytoNucsConfig.from_json, str(path))
            quiet(cfg.set_anisotropy_from_labels, [spot], [blob])
            model = quiet(build_cytonucs_model, cfg, name='flat', basedir=str(root / 'models'))
            weights = root / 'models' / 'flat' / 'weights_best.weights.h5'
            quiet(model.save_weights, str(weights))
            out = quiet(predict_hierarchy, model, cfg, image)
            self.assertEqual(sorted(out['levels']), ['blob', 'spot'])
            self.assertEqual(list(out['parents']), ['spot'])                  # raised before: no pair index
            result = quiet(score_hierarchy.score, str(path), weights, 'test')
            self.assertEqual(result['objects'], {'spot': 1, 'blob': 1})
            self.assertEqual(result['pairs_spot_blob'], 1)
            for key in ('ap_spot', 'ap_blob', 'jtpr_spot_blob'):
                self.assertIn(key, result)


class PaperMetricsTests(unittest.TestCase):
    def test_hydrastardist_paper_metrics_on_a_toy_image(self):
        from score_hierarchy import PAPER_TAUS, mean_paper_metrics, paper_metrics
        cells = np.zeros((40, 80), np.uint16); cells[2:38, 2:38], cells[2:38, 42:78] = 1, 2
        nuclei = np.zeros((40, 80), np.uint16); nuclei[10:20, 10:20], nuclei[10:20, 50:60] = 1, 2
        found = np.where(nuclei == 1, 1, 0).astype(np.uint16)                     # nucleus 2 is missed
        m = paper_metrics(nuclei, cells, found, cells, {1: 1, 2: 2})
        half = PAPER_TAUS.index(0.5)
        self.assertEqual((m['ap_inner'][half], m['iour_inner'][half]), (0.5, 0.5))   # 1 TP, 1 FN; IoU 1 / 2 objects
        self.assertEqual((m['ap_outer'][half], m['iour_outer'][half]), (1.0, 1.0))
        self.assertEqual((m['jtpr_inner'][half], m['jtpr_outer'][half]), (0.5, 0.5))  # pair 1 of 2 nuclei, of 2 cells
        # No link between the matched predictions is asked: swapping the predicted cell ids changes nothing.
        swapped = np.where(cells == 1, 2, np.where(cells == 2, 1, 0)).astype(np.uint16)
        self.assertEqual(paper_metrics(nuclei, cells, found, swapped, {1: 1, 2: 2})['jtpr_inner'][half], 0.5)
        # Two children in one parent: the parent counts once, so the rate over parents stays at most 1.
        two = np.zeros((40, 80), np.uint16); two[10:20, 5:15], two[22:32, 20:30] = 1, 3
        both = paper_metrics(two, cells, two, cells, {1: 1, 3: 1})
        self.assertEqual((both['jtpr_inner'][half], both['jtpr_outer'][half]), (1.0, 0.5))      # 2 of 2 children, 1 of 2 cells
        # Images without ground-truth nuclei do not enter the nucleus means; the best tau is the maximum.
        empty = paper_metrics(np.zeros_like(nuclei), cells, np.zeros_like(nuclei), cells, {})
        mean = mean_paper_metrics([m, empty])
        self.assertEqual(mean['images'], 2)
        self.assertEqual(mean['jtpr_inner'][half], 0.5)
        self.assertEqual(mean['ap_outer'][half], 1.0)
        self.assertEqual(mean['ap_inner_best'], max(v for v in mean['ap_inner'] if v is not None))


class PerImageRecordTests(unittest.TestCase):
    def test_records_summarise_like_pooled_counts_and_predictions_round_trip(self):
        from score_hierarchy import TAUS, save_prediction, score_image, summarise
        cells = np.zeros((40, 80), np.uint16); cells[2:38, 2:38], cells[2:38, 42:78] = 1, 2
        nuclei = np.zeros((40, 80), np.uint16); nuclei[10:20, 10:20], nuclei[10:20, 50:60] = 1, 2
        found = np.where(nuclei == 1, 1, 0).astype(np.uint16)
        ignore = np.zeros((40, 80), bool); ignore[30:40, 70:80] = True
        stray = found.copy(); stray[32:38, 72:78] = 5                             # a predicted nucleus in the ignore region
        pairs = [('nucleus', 'cytoplasm')]
        first, scored = score_image(dict(nucleus=nuclei, cytoplasm=cells), dict(nucleus=stray, cytoplasm=cells), pairs,
                                    {'nucleus_cytoplasm': {1: 1, 2: 2}}, ignore=dict(nucleus=ignore, cytoplasm=None))
        self.assertFalse((scored['nucleus'] == 5).any())                           # dropped before scoring
        self.assertEqual(first['levels']['nucleus']['tp'][0], 1)
        self.assertEqual((first['levels']['nucleus']['fp'][0], first['levels']['nucleus']['fn'][0]), (0, 1))
        self.assertEqual(first['pairs']['nucleus_cytoplasm']['joint_tp'], 1)
        self.assertEqual(first['pairs']['nucleus_cytoplasm']['pairs'], 2)
        second, _ = score_image(dict(nucleus=nuclei, cytoplasm=cells), dict(nucleus=nuclei, cytoplasm=cells), pairs,
                                {'nucleus_cytoplasm': {1: 1, 2: 2}})
        summary = summarise([first, second], ['nucleus', 'cytoplasm'], pairs)
        self.assertEqual(summary['objects'], dict(nucleus=4, cytoplasm=4))
        self.assertEqual(summary['ap50_nucleus'], 3 / 4)                            # pooled: TP 3, FN 1
        self.assertEqual(summary['ap_cytoplasm'], 1.0)
        self.assertEqual(summary['jtpr_nucleus_cytoplasm'], 3 / 4)
        self.assertEqual(summary['f1_50_nucleus'], 6 / 7)
        self.assertEqual(len(summary['paper_nucleus_cytoplasm']['jtpr_inner']), 9)
        self.assertEqual(len(first['levels']['nucleus']['tp']), len(TAUS))
        json.dumps(first)                                                           # records are plain JSON
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            save_prediction(tmp, 'a.tif', scored)
            np.testing.assert_array_equal(tifffile.imread(Path(tmp) / 'nucleus' / 'a.tif'), scored['nucleus'])


class SparseTilesTests(unittest.TestCase):
    def make(self, root, count=4, dtype=np.uint8):
        for split in ('train', 'test'):
            for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
                (root / split / sub).mkdir(parents=True)
            for i in range(count):
                cells = np.zeros((20, 40), dtype); cells[2:18, 2:18], cells[2:18, 22:38] = 1, 2
                nuclei = np.zeros((20, 40), dtype); nuclei[6:12, 6:12], nuclei[6:12, 26:32] = 1, 2
                tifffile.imwrite(root / split / 'images' / f'{i}.tif', np.zeros((20, 40, 3), np.uint8), photometric='rgb')
                tifffile.imwrite(root / split / 'nucleus_masks' / f'{i}.tif', nuclei, photometric='minisblack')
                tifffile.imwrite(root / split / 'cytoplasm_masks' / f'{i}.tif', cells, photometric='minisblack')
                (root / split / 'assignments' / f'{i}.json').write_text(json.dumps({'1': 1, '2': 2}))

    def test_unknown_label_survives_uint8_masks(self):
        from sparse_tiles import reduce_annotation
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            self.make(Path(tmp) / 'data')
            quiet(reduce_annotation, Path(tmp) / 'data', Path(tmp) / 'out', sparse=[('nucleus', 'cytoplasm', 0.5)])
            for i in range(4):
                nuclei = tifffile.imread(Path(tmp) / 'out/train/nucleus_masks' / f'{i}.tif')
                self.assertEqual(sorted(np.unique(nuclei).tolist())[-1], 65535)         # was wrapped to 255 in uint8
                self.assertEqual(len([v for v in np.unique(nuclei) if v not in (0, 65535)]), 1)
            self.assertEqual(tifffile.imread(Path(tmp) / 'out/test/nucleus_masks/0.tif').max(), 2)   # test untouched

    def test_complementary_halves_keep_one_level_each(self):
        from sparse_tiles import reduce_annotation
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            self.make(Path(tmp) / 'data', count=6)
            summary = quiet(reduce_annotation, Path(tmp) / 'data', Path(tmp) / 'out', complementary=('nucleus', 'cytoplasm'))
            out = Path(tmp) / 'out' / 'train'
            nuclei_only, cells_only = summary['complementary']['nucleus_only'], summary['complementary']['cytoplasm_only']
            self.assertEqual((len(nuclei_only), len(cells_only)), (3, 3))
            for name in nuclei_only:
                self.assertTrue((out / 'nucleus_masks' / name).exists())
                self.assertFalse((out / 'cytoplasm_masks' / name).exists())
            for name in cells_only:
                self.assertFalse((out / 'nucleus_masks' / name).exists())
                self.assertTrue((out / 'cytoplasm_masks' / name).exists())
            self.assertEqual(list((out / 'assignments').iterdir()), [])                 # no file has both levels
            self.assertEqual(len(list((Path(tmp) / 'out/test/cytoplasm_masks').iterdir())), 6)


class NormalisationTests(unittest.TestCase):
    def test_equal_to_csbdeep_except_for_nearly_empty_channels(self):
        from csbdeep.utils import normalize
        from cytonucs_runtime import normalize_percentiles
        rng = np.random.default_rng(0)
        image = rng.random((64, 80, 3)).astype(np.float32) * 200
        np.testing.assert_allclose(normalize_percentiles(image, 1, 99.8, axis=(0, 1)),
                                   normalize(image, 1, 99.8, axis=(0, 1)), rtol=1e-5, atol=1e-5)
        image[..., 2] = 0
        image[:1, :5, 2] = 255                                   # 5 bright pixels (0.1 %): the 99.8th percentile is 0
        before = normalize(image, 1, 99.8, axis=(0, 1))
        after = normalize_percentiles(image, 1, 99.8, axis=(0, 1))
        self.assertGreater(before[..., 2].max(), 1e15)            # csbdeep divides by ~0
        self.assertAlmostEqual(float(after[..., 2].max()), 1.0, places=5)
        np.testing.assert_allclose(after[..., :2], before[..., :2], rtol=1e-5, atol=1e-5)
        image[..., 2] = 0                                        # an empty channel stays 0
        self.assertEqual(float(normalize_percentiles(image, 1, 99.8, axis=(0, 1))[..., 2].max()), 0.0)


class SubsetViewTests(unittest.TestCase):
    def test_view_keeps_first_files_of_a_split_and_links_everything_else(self):
        from subset_view import subset_view
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            source = Path(tmp) / 'data'
            for split, count in (('train', 3), ('val', 5)):
                for sub in ('images', 'mass_masks', 'assignments_mass_kidney'):
                    (source / split / sub).mkdir(parents=True)
                for i in range(count):
                    for sub in ('images', 'mass_masks'):
                        tifffile.imwrite(source / split / sub / f'case_{i}.tif', np.full((2, 2), i, np.uint8))
                    (source / split / 'assignments_mass_kidney' / f'case_{i}.json').write_text('{}')
            (source / 'val' / 'meta.json').write_text('[]')
            (source / 'split_info.json').write_text('{}')
            counts = quiet(subset_view, source, Path(tmp) / 'view', {'val': 2})
            self.assertEqual(counts, {'train': 3, 'val': 2})
            view = Path(tmp) / 'view'
            for sub in ('images', 'mass_masks'):
                self.assertEqual(sorted(p.name for p in (view / 'val' / sub).iterdir()), ['case_0.tif', 'case_1.tif'])
                self.assertEqual(len(list((view / 'train' / sub).iterdir())), 3)
            self.assertEqual(sorted(p.name for p in (view / 'val' / 'assignments_mass_kidney').iterdir()),
                             ['case_0.json', 'case_1.json'])
            self.assertTrue((view / 'split_info.json').is_file() and (view / 'val' / 'meta.json').is_file())
            self.assertEqual((view / 'val' / 'images' / 'case_1.tif').stat().st_ino,
                             (source / 'val' / 'images' / 'case_1.tif').stat().st_ino)        # a hard link
            with self.assertRaises(FileExistsError):
                subset_view(source, view, {})


class TissueNetTests(unittest.TestCase):
    def test_pairs_need_half_the_nucleus_inside_cells(self):
        from tissuenet_to_cytonucs import nucleus_cells, relabel
        cells = np.zeros((20, 40), np.int32)
        cells[0:10, 0:10], cells[0:10, 10:20] = 7, 9
        nuclei = np.zeros((20, 40), np.int32)
        nuclei[2:6, 2:6] = 3                                       # inside cell 7
        nuclei[2:6, 8:16] = 5                                      # 2 columns in cell 7, 6 in cell 9
        nuclei[8:14, 2:6] = 8                                      # one third inside cell 7
        nuclei[15:18, 30:33] = 11                                  # outside every cell
        cells, n_cells = relabel(cells)
        nuclei, n_nuclei = relabel(nuclei)
        self.assertEqual((n_cells, n_nuclei), (2, 4))
        self.assertEqual(nucleus_cells(nuclei, cells), {1: 1, 2: 2})

    def test_images_are_plain_rgb_with_the_stains_in_green_and_blue(self):
        from tissuenet_to_cytonucs import rgb, write_image
        from cytonucs_runtime import read_model_image
        x = np.zeros((32, 48, 2), np.float32)
        x[4:10, 4:10, 0] = 50.                                     # a nucleus
        x[20:30, 20:40, 1] = 7.                                    # a membrane
        image = rgb(x)
        self.assertEqual((image.dtype, image.shape), (np.uint8, (32, 48, 3)))
        self.assertEqual(int(image[..., 0].max()), 0)                       # red is empty
        self.assertEqual((int(image[5, 5, 1]), int(image[5, 5, 2])), (255, 0))
        self.assertEqual((int(image[25, 25, 1]), int(image[25, 25, 2])), (0, 255))
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            write_image(Path(tmp) / 'a.tif', x)
            with tifffile.TiffFile(Path(tmp) / 'a.tif') as tif:
                self.assertEqual(tif.series[0].axes, 'YXS')                 # one RGB image, not a stack
            np.testing.assert_array_equal(read_model_image(Path(tmp) / 'a.tif', 2, 3), image)

    def test_metadata_unpickler_only_rebuilds_numpy_arrays(self):
        import pickle
        from tissuenet_to_cytonucs import NumpyOnly
        rows = np.array([['filename', 'specimen'], ['a.npz', 'Colon']], dtype=object)
        self.assertEqual(NumpyOnly(io.BytesIO(pickle.dumps(rows))).load().tolist(), rows.tolist())
        with self.assertRaises(pickle.UnpicklingError):
            NumpyOnly(io.BytesIO(pickle.dumps(Path('x')))).load()


class KiTSTests(unittest.TestCase):
    def test_masses_are_separate_instances_inside_their_kidney(self):
        from kits23_to_cytonucs import instances
        seg = np.zeros((6, 20, 40), np.uint8)
        seg[1:5, 2:18, 2:18] = 1                                  # left kidney
        seg[1:5, 2:18, 22:38] = 1                                 # right kidney
        seg[2:4, 4:8, 4:8] = 2                                    # tumour in the left kidney
        seg[2:4, 12:16, 4:8] = 3                                  # cyst in the left kidney
        seg[2:4, 4:8, 26:30] = 2                                  # tumour in the right kidney
        kidneys, masses, links = instances(seg)
        self.assertEqual(int(kidneys.max()), 2)
        self.assertEqual(int(masses.max()), 3)
        self.assertTrue(np.all(kidneys[masses > 0] > 0))           # every mass voxel belongs to a kidney
        left, right = int(kidneys[2, 5, 5]), int(kidneys[2, 5, 27])
        self.assertNotEqual(left, right)
        self.assertEqual(links, {int(masses[2, 5, 5]): left, int(masses[2, 13, 5]): left, int(masses[2, 5, 27]): right})


if __name__ == '__main__':
    unittest.main()
