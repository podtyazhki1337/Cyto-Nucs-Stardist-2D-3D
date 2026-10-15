"""v3.3: ray targets on a sub-lattice of the head grid, deployed metrics on training volumes."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from test_pipeline_review import make_dataset
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import CytoNucsDataGenerator
from cytonucs_losses import CytoNucsLoss, compute_stardist_gt
from cytonucs_model import build_cytonucs_model
import cytonucs_trainer
from cytonucs_trainer import (CytoNucsTrainer, DiceCallback, deployed_tile_metrics, source_volume,
                              without_xy_face_objects)

PRESET = 'config_cytonucs_v3_3.json'
NEW = {'training': dict(dist_target_stride=(None, [1, 2, 2])),
       'evaluation': dict(fullval_train_volumes=(0, 6))}
TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def labels():
    lab = np.zeros((8, 32, 32), np.int32)
    lab[2:6, 5:17, 4:14] = 1
    lab[1:7, 18:30, 15:29] = 2
    return lab


class RayTargetStrideTests(unittest.TestCase):
    def test_targets_are_the_head_grid_targets_at_the_same_nodes(self):
        full, coarse = quiet(config), quiet(config, dist_target_stride=[1, 2, 2])
        args = dict(rays=full.rays_nucleus, aniso=full.anisotropy)
        p0, d0 = compute_stardist_gt(labels(), full, full.nucleus_grid, **args)
        p1, d1 = compute_stardist_gt(labels(), coarse, coarse.nucleus_grid, **args)
        np.testing.assert_array_equal(p0, p1)                    # probability stays on the head grid
        self.assertEqual(d0.shape, (8, 16, 16, 12))
        self.assertEqual(d1.shape, (8, 8, 8, 12))
        np.testing.assert_array_equal(d1, d0[:, ::2, ::2])
        self.assertGreater(float(d1.max()), 1.)

    def test_loss_reads_the_prediction_only_at_target_nodes(self):
        full, coarse = quiet(config), quiet(config, dist_target_stride=[1, 2, 2])
        p, d = compute_stardist_gt(labels(), coarse, coarse.nucleus_grid, rays=coarse.rays_nucleus, aniso=coarse.anisotropy)
        prob = tf.constant(p[None]); dist = tf.constant(d[None], tf.float32)
        rng = np.random.default_rng(0)
        pred_prob = tf.constant(rng.uniform(.05, .95, prob.shape).astype(np.float32))
        pred = rng.uniform(0, 12, (1, 8, 16, 16, 12)).astype(np.float32)
        loss = CytoNucsLoss(coarse)._stardist_loss(prob, pred_prob, dist, tf.constant(pred), name='nucleus')
        reference = CytoNucsLoss(full)._stardist_loss(prob[:, :, ::2, ::2], pred_prob[:, :, ::2, ::2], dist,
                                                    tf.constant(pred[:, :, ::2, ::2]), name='nucleus')
        self.assertAlmostEqual(float(loss['distance']), float(reference['distance']), places=5)
        inside = np.argwhere(p[..., 0] > 0)
        even = next(tuple(v) for v in inside if v[1] % 2 == 0 and v[2] % 2 == 0)
        odd = next(tuple(v) for v in inside if v[1] % 2 == 1 and v[2] % 2 == 1)
        for node, changes in ((even, True), (odd, False)):
            moved = pred.copy(); moved[(0,) + node] += 5.
            other = CytoNucsLoss(coarse)._stardist_loss(prob, pred_prob, dist, tf.constant(moved), name='nucleus')
            self.assertEqual(abs(float(other['distance']) - float(loss['distance'])) > 1e-6, changes, node)

    def test_train_step_and_fallback_targets(self):
        cfg = quiet(config, dist_target_stride=[1, 2, 2], refinement_depth=1, mask_quality_head=True,
                    mask_quality_weight=.1)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            model = quiet(build_cytonucs_model, cfg, basedir=tmp, name='stride')
            loss = CytoNucsLoss(cfg)
            x, y = batch(cfg)
            self.assertEqual(tuple(y['nucleus_dist'].shape), (1, 8, 8, 8, 12))
            prediction = model.keras_model(x, training=False)
            with_targets = float(loss(y, prediction))
            # Without generator targets the loss computes them itself, on the same nodes.
            bare = {k: v for k, v in y.items() if not k.endswith(('_prob', '_dist'))}
            self.assertAlmostEqual(float(loss(bare, prediction)), with_targets, places=4)
            trainer = CytoNucsTrainer(model, cfg, loss, 0.)
            result = trainer.train_step(x, y)
            self.assertTrue(np.isfinite(float(result[0])))
            self.assertGreater(float(result[-1]['nucleus_distance']), 0.)

    def test_generator_emits_sub_lattice_targets(self):
        cfg = quiet(config, dist_target_stride=[1, 2, 2])
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            make_dataset(Path(tmp) / 'train')
            generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')
            _, y = generator[0]
        self.assertEqual(y['nucleus_prob'].shape, (1, 8, 16, 16, 1))
        for head in ('nucleus', 'cytoplasm'):
            self.assertEqual(y[head + '_dist'].shape, (1, 8, 8, 8, 12))


class WholeObjectTests(unittest.TestCase):
    def setUp(self):
        gt = np.zeros((4, 24, 24), np.int32)
        gt[1:3, 8:16, 8:16] = 3            # whole object
        gt[1:3, 2:10, 0:6] = 5             # cut by the x = 0 face
        gt[1:3, 18:23, 2:6] = 7            # whole, one row away from the y = 23 face
        self.gt = gt

    def test_removes_cut_ground_truth_and_the_masks_on_it(self):
        pred = np.zeros_like(self.gt)
        pred[self.gt == 3] = 1             # the whole object
        pred[1:3, 2:10, 0:3] = 2           # a part of the cut object, on the face
        pred[1:3, 3:9, 3:6] = 4            # another part of it, away from the face
        pred[1:3, 18:22, 18:22] = 6        # false positive without any GT
        pred[1:3, 18:24, 2:6] = 8          # reaches the face, but its GT is whole
        gt, kept = without_xy_face_objects(self.gt, pred)
        self.assertEqual(set(np.unique(gt)), {0, 3, 7})
        self.assertEqual(set(np.unique(kept)), {0, 1, 6, 8})
        for label in (1, 6, 8):
            np.testing.assert_array_equal(kept == label, pred == label)
        # A mask on two GT objects follows the one with the higher IoU.
        pred[:] = 0; pred[1:3, 2:16, 0:16] = 9
        self.assertEqual(set(np.unique(without_xy_face_objects(self.gt, pred)[1])), {0, 9})   # IoU with 3 is higher
        empty = np.zeros_like(pred)
        self.assertFalse(without_xy_face_objects(self.gt, empty)[1].any())

    def test_deployed_metrics_count_all_and_whole_objects(self):
        gt = np.where(self.gt == 7, 0, self.gt)
        nuclei = np.zeros_like(gt)
        nuclei[gt == 3] = 1
        nuclei[1:3, 2:10, 0:3] = 2         # half of the cut object: IoU 0.5 with it
        cells = gt.copy()
        predictor = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(ndim=3, train_patch_size=(4, 24, 24))),
            predict_instances=lambda img, normalize_img=False: (nuclei, cells, {}),
            _prompt_assignments={1: 3, 2: 5})
        sample = dict(image=np.zeros(gt.shape + (1,), np.float32), nucleus_mask=gt,
                      cytoplasm_mask=gt, assignments={3: 3, 5: 5})
        m = deployed_tile_metrics(predictor, [dict(sample, filename='HeLa_stack_z000_y0512_x0000_a00.tiff'),
                                              dict(sample, filename='EA_s2_z000_y0000_x0448_a00.tiff')], TAUS)
        self.assertEqual(m['volumes'], 2)
        self.assertEqual(list(m['by_volume']), ['EA_s2', 'HeLa_stack'])
        for row in m['by_volume'].values():
            self.assertEqual(row['volumes'], 1)
            self.assertEqual({k: v for k, v in row.items() if k != 'volumes'},
                             {k: v for k, v in m.items() if k not in ('volumes', 'by_volume')})
        self.assertEqual(m['ap_cytoplasm'], 1.0)
        self.assertEqual(m['ap_cytoplasm_whole'], 1.0)
        self.assertEqual(m['ap_nucleus_whole'], 1.0)
        # All objects: both matched at IoU 0.5, only the whole one above it.
        self.assertEqual(m['ap50_nucleus'], 1.0)
        self.assertAlmostEqual(m['ap_nucleus'], (1.0 + 9 * (1 / 3)) / 10)
        self.assertEqual(m['jtpr'], 1.0)


class TrainVolumeValidationTests(unittest.TestCase):
    def test_source_volume_names(self):
        self.assertEqual(source_volume('HeLa_x_merged_stack_z016_y1024_x0512_a03.tiff'), 'HeLa_x_merged_stack')
        self.assertEqual(source_volume(Path('val') / 'plain.tif'), 'plain')
        self.assertEqual(source_volume('z001_y1_x1_a00_kept_z2.tif'), 'z001_y1_x1_a00_kept_z2')

    def test_fixed_original_volumes_are_selected(self):
        def tile(volume, z, y, x, copy, nuclei):
            mask = np.zeros((4, 8, 8), np.int32)
            for i in range(nuclei):
                mask[0, i // 8, i % 8] = i + 1
            return dict(filename=f'{volume}_z{z:03d}_y{y:04d}_x{x:04d}_a{copy:02d}.tiff', nucleus_mask=mask, cytoplasm_mask=mask)
        pool = [tile('B', 0, 0, 0, 0, 9), tile('B', 2, 0, 0, 0, 8),      # overlaps the first in Z
                tile('B', 0, 0, 8, 0, 3), tile('B', 0, 0, 0, 1, 50),     # '_a01': an offline copy, never taken
                tile('A', 0, 0, 0, 0, 2), tile('A', 0, 8, 0, 0, 1)]

        def pick(count, pool=pool, generator=True):
            stub = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(fullval_train_volumes=count)),
                                   train_generator=SimpleNamespace(data_pool=list(reversed(pool))) if generator else None)
            return [s['filename'] for s in DiceCallback._train_volumes(stub)]
        self.assertEqual(pick(0), [])
        self.assertEqual(pick(3, generator=False), [])
        # Richest source volume first, then in turn; inside a volume the richest tile that does not overlap.
        self.assertEqual(pick(4), ['B_z000_y0000_x0000_a00.tiff', 'A_z000_y0000_x0000_a00.tiff',
                                   'B_z000_y0000_x0008_a00.tiff', 'A_z000_y0008_x0000_a00.tiff'])
        self.assertEqual(pick(4), pick(4))
        self.assertEqual(sorted(pick(50)), sorted(s['filename'] for s in pool if '_a00' in s['filename']))
        # Files without the patch suffix (other datasets) are all originals of their own source volume.
        plain = [dict(filename='b.tif', nucleus_mask=np.ones((2, 2, 2), np.int32)), dict(filename='a.tif', nucleus_mask=np.zeros((2, 2, 2), np.int32))]
        self.assertEqual(pick(2, plain), ['b.tif', 'a.tif'])
        # The selection is computed once for a pool.
        stub = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(fullval_train_volumes=2)),
                               train_generator=SimpleNamespace(data_pool=pool))
        first = DiceCallback._train_volumes(stub)
        self.assertIs(DiceCallback._train_volumes(stub), first)

    def test_full_validation_logs_train_metrics_with_calibrated_gates(self):
        gt = np.zeros((8, 32, 32), np.int32); gt[2:6, 10:20, 10:20] = 1
        sample = dict(image=(gt > 0).astype(np.float32)[..., None], nucleus_mask=gt, cytoplasm_mask=gt, assignments={1: 1})
        seen = {}

        def metrics(predictor, samples, taus):
            seen.update(gates=predictor.quality_gates, calibrating=predictor.calibrating, taus=list(taus))
            seen.setdefault('files', []).append([s['filename'] for s in samples])
            return dict(volumes=len(samples), jtpr=.5, ap_nucleus=.25, ap50_nucleus=.5, ap_nucleus_whole=.75,
                        ap50_nucleus_whole=1., ap_cytoplasm=.125, ap50_cytoplasm=.25, ap_cytoplasm_whole=.375,
                        ap50_cytoplasm_whole=.5, by_volume={})

        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = quiet(config, refinement_stage_audit=True, mask_quality_head=True, proposal_mode='peaks',
                        mask_assembly_nucleus='object_quality', mask_assembly_cytoplasm='object_quality',
                        mask_quality_min='auto', prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.,
                        fullval_train_volumes=1)
            model = quiet(build_cytonucs_model, cfg, basedir=tmp)
            val = SimpleNamespace(image_files=['a.tiff', 'b.tiff'], data_pool=[
                dict(sample, filename='HeLa_z000_y0000_x0000_a00.tiff'), dict(sample, filename='EA_z000_y0000_x0000_a00.tiff')])
            train = SimpleNamespace(data_pool=[dict(sample, filename='t_a01.tiff'), dict(sample, filename='t_a00.tiff')])
            history = {'fullval_epoch': [450], 'fullval_metric': [.1]}
            callback = DiceCallback(model, val, eval_every=1, save_best=False, prob_thresh_nucleus=1.,
                                    prob_thresh_cytoplasm=1., train_generator=train)
            with patch.object(callback, '_save_epoch_visualization_full'), \
                    patch.object(cytonucs_trainer, 'deployed_tile_metrics', side_effect=metrics):
                quiet(callback.on_epoch_end, 454, history)
        self.assertEqual(seen['files'], [['t_a00.tiff']])
        self.assertFalse(seen['calibrating'])
        self.assertEqual(seen['taus'], TAUS)
        n = len(history['fullval_epoch'])
        for key, value in (('train_ap_nucleus', .25), ('train_ap_cytoplasm_whole', .375), ('train_jtpr', .5),
                           ('train_volumes', 1)):
            self.assertEqual(history['fullval_' + key], [None, value], key)
            self.assertEqual(len(history['fullval_' + key]), n)
        # Validation totals are also reported per source volume, with the same gates.
        rows = history['fullval_by_volume']
        self.assertIsNone(rows[0])
        self.assertEqual(sorted(rows[1]), ['EA', 'HeLa'])
        for row in rows[1].values():
            self.assertEqual(row['volumes'], 1)
            for key in ('ap_nucleus', 'ap_cytoplasm', 'ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated', 'jtpr'):
                self.assertIn(key, row)
        # Whole-object AP after gates on validation comes from the same pass (no second prediction).
        for head in ('nucleus', 'cytoplasm'):
            self.assertEqual(len(history[f'fullval_ap_{head}_quality_gated_whole']), n)
            self.assertIsNone(history[f'fullval_ap_{head}_quality_gated_whole'][0])
        # Without the setting nothing is evaluated or logged.
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            cfg = quiet(config, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            model = quiet(build_cytonucs_model, cfg, basedir=tmp)
            callback = DiceCallback(model, val, eval_every=1, save_best=False, prob_thresh_nucleus=1.,
                                    prob_thresh_cytoplasm=1., train_generator=train)
            history = {}
            with patch.object(callback, '_save_epoch_visualization_full'), \
                    patch.object(cytonucs_trainer, 'deployed_tile_metrics', side_effect=AssertionError):
                quiet(callback.on_epoch_end, 4, history)
        self.assertFalse([k for k in history if k.startswith('fullval_train_')])


class ConfigTests(unittest.TestCase):
    def load(self, raw):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw = json.loads(json.dumps(raw))
            raw['output']['basedir'] = str(Path(tmp) / 'models')
            raw['training']['resume_from_checkpoint'] = None
            path = Path(tmp) / 'config.json'; path.write_text(json.dumps(raw))
            return quiet(CytoNucsConfig.from_json, path)

    def test_json_contract(self):
        later = {name: dict(dist_target_stride=[1, 2, 2], fullval_train_volumes=3)
                 for name in ('config_cytonucs_v3_4.json', 'config_cytonucs_v4.json', 'config_cytonucs_v5.json')}
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            for section, settings in NEW.items():
                for key, (legacy, new) in settings.items():
                    expected = later[path.name][key] if path.name in later else new if path.name == PRESET else legacy
                    self.assertEqual(raw[section][key], expected, f'{path.name} {key}')
        for section, settings in NEW.items():
            for key, (legacy, _) in settings.items():
                self.assertEqual(CytoNucsConfig.FIXES_SETTINGS[section][key], legacy)
        raw = json.loads((ROOT / 'config_cytonucs_v3_2.json').read_text())
        for section, settings in NEW.items():
            for key, (_, new) in settings.items():
                raw[section][key] = new
        cfg = self.load(raw)
        self.assertEqual(cfg.dist_target_stride, [1, 2, 2])
        self.assertEqual(cfg.fullval_train_volumes, 6)

    def test_preset_is_v3_2_from_scratch_on_the_split(self):
        base = json.loads((ROOT / 'config_cytonucs_v3_2.json').read_text())
        raw = json.loads((ROOT / PRESET).read_text())
        changed = {(section, key) for section in raw if isinstance(raw[section], dict)
                   for key in raw[section] if raw[section][key] != base[section].get(key)}
        self.assertEqual(changed, {
            ('experiment', 'name'), ('data', 'dataset_dir'), ('model', 'n_rays'), ('training', 'resume_from_checkpoint'),
            ('training', 'train_epochs'), ('training', 'dist_target_stride'),
            ('inference', 'checkpoint_path'), ('evaluation', 'eval_dice_every'), ('evaluation', 'min_eval_epoch'),
            ('evaluation', 'fullval_train_volumes')})
        self.assertEqual(set(raw), set(base))
        self.assertEqual(raw['mode'], 'train')
        self.assertIsNone(raw['training']['resume_from_checkpoint'])
        self.assertIs(raw['training']['reset_best_on_resume'], False)
        self.assertIs(raw['preprocessing']['run_preprocessing'], False)
        self.assertIn(raw['experiment']['name'], raw['inference']['checkpoint_path'])
        cfg = self.load(raw)
        self.assertEqual((cfg.n_rays, len(cfg.rays_nucleus), len(cfg.rays_cytoplasm)), (32, 32, 32))
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            model = quiet(build_cytonucs_model, cfg, basedir=tmp, name='preset')
        self.assertEqual(model.keras_model.output['nucleus_dist'].shape[-1], 32)
        self.assertEqual(model.keras_model.output['cytoplasm_dist'].shape[-1], 32)

    def test_invalid_values(self):
        base = json.loads((ROOT / 'config_cytonucs_v3_2.json').read_text())
        for section, key, value in (('training', 'dist_target_stride', [1, 2]), ('training', 'dist_target_stride', [0, 2, 2]),
                                    ('training', 'dist_target_stride', [1, 3, 3]), ('training', 'dist_target_stride', 2),
                                    ('training', 'dist_target_stride', [1, 2, True]),
                                    ('evaluation', 'fullval_train_volumes', -1), ('evaluation', 'fullval_train_volumes', 1.5),
                                    ('evaluation', 'fullval_train_volumes', True)):
            raw = json.loads(json.dumps(base)); raw[section][key] = value
            with self.assertRaises(ValueError, msg=f'{key}={value!r}'):
                self.load(raw)
        raw = json.loads(json.dumps(base)); raw['training']['dist_target_stride'] = [1, 2, 2]
        raw['loss_params']['w_center_separation'] = 0.2
        with self.assertRaisesRegex(ValueError, 'w_center_separation'):
            self.load(raw)


if __name__ == '__main__':
    unittest.main()
