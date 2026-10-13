"""model.levels: which object levels a run has. One level trains without the other level's
masks, decoder and mask branch; both levels (the default) are unchanged.
training.partial_annotation: training files may lack the masks of some levels of the run."""
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
from cytonucs_data_generator import CytoNucsDataGenerator
from cytonucs_inference import CytoNucsPredictor
from cytonucs_losses import CytoNucsLoss
from cytonucs_model import build_cytonucs_model
from cytonucs_runtime import checkpoint_jtpr_weight, deployed_metric
from cytonucs_trainer import CytoNucsTrainer

BOTH = ['nucleus', 'cytoplasm']
FEATURES = dict(refinement_depth=1, mask_quality_head=True, mask_quality_pooling='mask', mask_quality_weight=.1,
                semantic_foreground_weight=1., refinement_full_resolution=True)
OTHER = dict(nucleus='cytoplasm', cytoplasm='nucleus')
TAG = dict(nucleus='nuc', cytoplasm='cyto')


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def tile(seed, shape=(8, 64, 64)):
    """Four cells with one nucleus each; the image shows both."""
    rng = np.random.default_rng(seed)
    cells = np.zeros(shape, np.uint16); nuclei = np.zeros(shape, np.uint16)
    for k, (cy, cx) in enumerate(((16, 16), (16, 46), (46, 16), (46, 46)), 1):
        y, x = cy + int(rng.integers(-3, 4)), cx + int(rng.integers(-3, 4))
        cells[1:7, y - 11:y + 11, x - 11:x + 11] = k
        nuclei[2:6, y - 5:y + 5, x - 5:x + 5] = k
    image = (0.3 * (cells > 0) + 0.6 * (nuclei > 0) + rng.normal(0, .05, shape)).astype(np.float32)
    return image, dict(nucleus=nuclei, cytoplasm=cells)


def dataset(root, levels, splits=(('train', 4),)):
    """Tiles with mask folders only for the given levels."""
    for split, count in splits:
        for folder in ['images'] + [level + '_masks' for level in levels]:
            (root / split / folder).mkdir(parents=True)
        for i in range(count):
            image, masks = tile(100 * len(split) + i)
            tifffile.imwrite(root / split / 'images' / f't{i}.tif', image, photometric='minisblack')
            for level in levels:
                tifffile.imwrite(root / split / (level + '_masks') / f't{i}.tif', masks[level], photometric='minisblack')


def tensors(batch):
    x, y = batch
    return tf.constant(x, tf.float32), {k: tf.constant(v, tf.float32 if v.dtype.kind == 'f' else tf.int32 if v.dtype.kind in 'iu' else None)
                                        for k, v in y.items()}


class ConfigTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(quiet(config).levels, BOTH)
        self.assertEqual(CytoNucsConfig.FIXES_SETTINGS['model']['levels'], BOTH)
        for levels in (['nucleus'], ['cytoplasm'], BOTH):
            cfg = quiet(config, levels=levels)
            self.assertEqual(cfg.levels, levels)
            self.assertEqual([cfg.has_level(name) for name in BOTH], [name in levels for name in BOTH])
        for bad in ([], ['cytoplasm', 'nucleus'], ['Nucleus'], ['my level'], ['nucleus', 'nucleus'], 'nucleus', None, [1]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                quiet(config, levels=bad)
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            quiet(config, levels=['nucleolus', 'nucleus', 'cytoplasm'])

    def test_level_names_take_the_inner_and_outer_roles(self):
        self.assertEqual(quiet(config).level_roles, dict(nucleus='nucleus', cytoplasm='cytoplasm'))
        self.assertEqual(quiet(config, levels=['nucleolus', 'nucleus']).level_roles, dict(nucleus='nucleolus', cytoplasm='nucleus'))
        self.assertEqual(quiet(config, levels=['cytoplasm']).level_roles, dict(cytoplasm='cytoplasm'))
        self.assertEqual(quiet(config, levels=['cells']).level_roles, dict(nucleus='cells'))   # one plain level: inner role
        for levels, roles in ((['nucleolus', 'nucleus'], [True, True]), (['cells'], [True, False]), (['cytoplasm'], [False, True])):
            cfg = quiet(config, levels=levels)
            self.assertEqual([cfg.has_level(role) for role in BOTH], roles)

    def test_presets_keep_both_levels_and_json_reaches_the_config(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            self.assertEqual(json.loads(path.read_text())['model']['levels'], BOTH, path.name)
        raw = json.loads((ROOT / 'config_cytonucs_v4.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['output']['basedir'] = str(Path(tmp) / 'models'); raw['model']['levels'] = ['cytoplasm']
            path = Path(tmp) / 'config.json'; path.write_text(json.dumps(raw))
            self.assertEqual(quiet(CytoNucsConfig.from_json, path).levels, ['cytoplasm'])
            raw['model']['levels'] = ['Cells']; path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                quiet(CytoNucsConfig.from_json, path)

    def test_checkpoint_metric_of_one_level_is_its_ap(self):
        self.assertEqual(checkpoint_jtpr_weight(quiet(config, checkpoint_w_jtpr=.5)), .5)
        for levels in (['nucleus'], ['cytoplasm']):
            self.assertEqual(checkpoint_jtpr_weight(quiet(config, checkpoint_w_jtpr=.5, levels=levels)), 0.)
        ap = dict(nucleus=0., cytoplasm=.4)
        self.assertAlmostEqual(deployed_metric(0., 0, ap, dict(nucleus=0, cytoplasm=0), 0., levels=1), .4)
        # Both levels: the formula as before.
        self.assertAlmostEqual(deployed_metric(.7, 1000, dict(nucleus=.4, cytoplasm=.3), dict(nucleus=5, cytoplasm=5), .7),
                               .7 * (.7 - .01) + .3 * .5 * .7)

    def test_absent_level_takes_the_geometry_of_the_present_one(self):
        _, masks = tile(0, (8, 96, 96))
        labels = np.zeros((8, 96, 96), np.uint16)
        for k in range(40):                                       # enough objects for a measurement
            y, x = 4 + 14 * (k // 6), 4 + 14 * (k % 6)
            labels[2:6, y:y + 10, x:x + 10] = k + 1
        empty = np.zeros_like(labels)
        cfg = quiet(config, levels=['cytoplasm'], anisotropy=(1, 1, 1), nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1),
                    median_max_dist_nucleus=None, median_max_dist_cytoplasm=None)
        quiet(cfg.set_anisotropy_from_labels, [empty], [labels])
        self.assertEqual(cfg.rays_anisotropy_nucleus, cfg.rays_anisotropy_cytoplasm)
        self.assertEqual(tuple(cfg.nucleus_grid), tuple(cfg.cytoplasm_grid))
        self.assertEqual(cfg.median_max_dist_nucleus, cfg.median_max_dist_cytoplasm)
        self.assertGreater(cfg.median_max_dist_cytoplasm, 0)


class GeneratorTests(unittest.TestCase):
    def test_absent_level_needs_no_folder_and_is_empty(self):
        for level in BOTH:
            with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
                dataset(Path(tmp), [level])
                cfg = quiet(config, levels=[level])
                generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')
                self.assertTrue(all(s['assignments'] == {} for s in generator.data_pool))
                for index in range(4):
                    x, y = generator[index]
                    absent = OTHER[level]
                    self.assertEqual(int(y[absent + '_instances'].max()), 0)
                    self.assertEqual(float(y[absent + '_target'].max()), 0.)
                    self.assertEqual(float(y[absent + '_prob'].max()), 0.)
                    self.assertGreater(int(y[level + '_instances'].max()), 0)
                    self.assertGreater(float(y[level + '_target'].max()), 0.)       # prompts fall on the present level
                    self.assertGreater(float(x[..., 1:].max()), 0.)
                # The same folder cannot serve a run with both levels.
                with self.assertRaises(AssertionError):
                    quiet(CytoNucsDataGenerator, Path(tmp) / 'train', quiet(config), cfg.rays, subset='train')

    def test_masks_are_read_from_the_folders_of_the_level_names(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp) / 'train'
            for folder in ('images', 'nucleolus_masks', 'nucleus_masks', 'cells_masks'):
                (root / folder).mkdir(parents=True)
            image, masks = tile(3)
            tifffile.imwrite(root / 'images' / 't0.tif', image, photometric='minisblack')
            # Stored under other names: the small objects as 'nucleolus', the large ones as 'nucleus' and 'cells'.
            tifffile.imwrite(root / 'nucleolus_masks' / 't0.tif', masks['nucleus'], photometric='minisblack')
            for name in ('nucleus_masks', 'cells_masks'):
                tifffile.imwrite(root / name / 't0.tif', masks['cytoplasm'], photometric='minisblack')
            cfg = quiet(config, levels=['nucleolus', 'nucleus'])
            pair = quiet(CytoNucsDataGenerator, root, cfg, cfg.rays, subset='train')
            np.testing.assert_array_equal(pair.data_pool[0]['nucleus_mask'], masks['nucleus'])       # inner role
            np.testing.assert_array_equal(pair.data_pool[0]['cytoplasm_mask'], masks['cytoplasm'])   # outer role
            self.assertEqual(len(pair.data_pool[0]['assignments']), 4)
            cfg = quiet(config, levels=['cells'])
            single = quiet(CytoNucsDataGenerator, root, cfg, cfg.rays, subset='train')
            np.testing.assert_array_equal(single.data_pool[0]['nucleus_mask'], masks['cytoplasm'])
            self.assertEqual(int(single.data_pool[0]['cytoplasm_mask'].max()), 0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {tuple(levels): quiet(build_cytonucs_model, quiet(config, levels=levels, **FEATURES))
                      for levels in (BOTH, ['nucleus'], ['cytoplasm'])}

    def test_absent_level_has_no_layers_and_empty_outputs(self):
        both = self.models[tuple(BOTH)]
        x = tf.constant(np.random.default_rng(0).normal(size=(2, 8, 32, 32, 1 + 2 * 2)).astype(np.float32))
        reference = both.keras_model(x, training=False)
        for level in BOTH:
            model, absent = self.models[(level,)], OTHER[level]
            owners = {layer.name for layer in model.keras_model.layers if layer.weights}
            self.assertFalse([n for n in owners if n.startswith((TAG[absent] + '_', absent + '_'))], level)
            self.assertTrue([n for n in owners if n.startswith(TAG[level] + '_dec')])
            self.assertLess(model.keras_model.count_params(), .6 * both.keras_model.count_params())
            out = model.keras_model(x, training=True)
            self.assertEqual(set(out) | {absent + '_fine_logit', absent + '_fine_begin'}, set(reference))
            for key, value in out.items():
                if key != 'fine_prompt_slot':
                    self.assertEqual(tuple(value.shape), tuple(reference[key].shape), key)
            self.assertLess(float(tf.reduce_max(out[absent + '_prob'])), 1e-6)
            self.assertLess(float(tf.reduce_max(out[absent + '_mask'])), 1e-6)
            self.assertEqual(float(tf.reduce_max(tf.abs(out[absent + '_dist']))), 0.)
            self.assertEqual(float(tf.reduce_max(out[absent + '_quality'])), 0.)
            self.assertLess(float(tf.reduce_max(tf.sigmoid(out[absent + '_semantic_logit']))), 1e-6)
            self.assertGreater(float(tf.reduce_max(out[level + '_prob'])), .01)
            self.assertEqual(model.architecture_signature()['roles'], [level])
            self.assertEqual(sorted(model.fine_head_models), [level])
        self.assertNotIn('roles', both.architecture_signature())
        # Other names for the same two roles are the same network.
        named = quiet(build_cytonucs_model, quiet(config, levels=['nucleolus', 'nucleus'], **FEATURES))
        self.assertEqual(named.architecture_signature(), both.architecture_signature())
        self.assertEqual(named.keras_model.count_params(), both.keras_model.count_params())

    def test_runtime_models_keep_their_interface(self):
        x = tf.constant(np.random.default_rng(1).normal(size=(1, 8, 32, 32, 1)).astype(np.float32))
        prompt = tf.constant(np.random.default_rng(2).random((1, 8, 32, 32, 1)).astype(np.float32))
        for level in BOTH:
            model, absent = self.models[(level,)], OTHER[level]
            features = model.refinement_features_model(x, training=False)
            self.assertEqual(len(features), 7)
            outputs = model.refinement_fine_model(list(features) + [prompt, prompt], training=False)
            self.assertEqual(len(outputs), 6)
            logits = dict(nucleus=outputs[0], cytoplasm=outputs[1])
            self.assertLess(float(tf.reduce_max(tf.nn.softmax(logits[absent], -1)[..., 1])), 1e-6)
            self.assertEqual(logits[absent].shape, logits[level].shape)
            cell = model.cytoplasm_fine_model([features[i] for i in (1, 2, 5, 6)] + [prompt], training=False)
            self.assertEqual(cell[0].shape, logits['cytoplasm'].shape)
            # The detection pass fetches its four outputs by layer name.
            predictor = CytoNucsPredictor(model, model.config, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            nuclei, cells, maps = quiet(predictor.predict_instances, np.asarray(x[0]), normalize_img=False)
            self.assertEqual((int(nuclei.max()), int(cells.max())), (0, 0))
            self.assertLess(float(maps[absent + '_prob'].max()), 1e-6)


class TrainingTests(unittest.TestCase):
    def test_training_step_updates_only_the_present_level(self):
        for level in BOTH:
            with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
                dataset(Path(tmp), [level])
                cfg = quiet(config, levels=[level], **FEATURES)
                generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')
                model = quiet(build_cytonucs_model, cfg, basedir=tmp, name='one')
                trainer = CytoNucsTrainer(model, cfg, CytoNucsLoss(cfg), 0.)
                before = [v.numpy().copy() for v in model.keras_model.trainable_variables]
                result = trainer.train_step(*tensors(generator[0]))
                components = {k: float(v) for k, v in result[-1].items()}
                absent = OTHER[level]
                self.assertTrue(np.isfinite(float(result[0])))
                for key in ('bce', 'distance', 'mask'):
                    self.assertLess(abs(components[f'{absent}_{key}']), 1e-6, key)
                self.assertGreater(components[f'{level}_bce'], 0.)
                self.assertGreater(components[f'{level}_mask'], 0.)
                changed = sum(not np.array_equal(a, v.numpy()) for a, v in zip(before, model.keras_model.trainable_variables))
                self.assertGreater(changed, len(before) // 2)

    def test_training_run_with_full_validation_on_cells_only(self):
        """train_cytonucs.main on a dataset without nucleus masks: training, validation on patches,
        full validation with gate calibration, checkpoints and the final evaluation."""
        import train_cytonucs
        raw = json.loads((ROOT / 'config_cytonucs_v4.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            dataset(root / 'data', ['cytoplasm'], (('train', 4), ('val', 2), ('test', 2)))
            raw['experiment']['name'] = 'cells'
            raw['data']['dataset_dir'] = str(root / 'data') + '/'
            raw['model'].update(levels=['cytoplasm'], n_rays=12, n_channel_in=1, use_gpu=False, unet_n_depth=2,
                                unet_n_filter_base=4, net_conv_after_unet=8, mask_refinement_filters=8, unet_norm_groups=2,
                                unet_pool_schedule=[[1, 2, 2], [2, 2, 2]],
                                unet_kernel_schedule=[[1, 3, 3], [3, 3, 3], [3, 3, 3]], anisotropy=[3, 1, 1])
            raw['training'].update(train_patch_size=[8, 32, 32], train_epochs=2, steps_per_epoch=3, train_region_crops=False,
                                   lr_warmup_epochs=0, predicted_prompt_start_epoch=1, prefetch_workers=1, prefetch_depth=1)
            raw['inference'].update(refinement_patch_size=[0, 32, 32],
                                    checkpoint_path=str(root / 'models/cells/weights_best.weights.h5'))
            raw['evaluation'].update(eval_dice_every=1, min_eval_epoch=1, save_every_n_epochs=1, fullval_train_volumes=1,
                                     save_predictions=False)
            raw['gpu_config']['visible_devices'] = '-1'
            raw['output']['basedir'] = str(root / 'models')
            (root / 'config.json').write_text(json.dumps(raw))
            cwd = os.getcwd()
            try:
                os.chdir(root)
                quiet(train_cytonucs.main, str(root / 'config.json'))
            finally:
                os.chdir(cwd)
            history = json.loads((root / 'models/cells/history.json').read_text())
            self.assertEqual(history['fullval_epoch'], [1, 2])
            self.assertEqual(history['fullval_ap_nucleus'], [0., 0.])
            self.assertEqual(history['fullval_jtpr'], [0., 0.])
            self.assertEqual(history['fullval_metric'], history['fullval_ap_cytoplasm'])   # the level's AP, no pair term
            self.assertTrue(all(np.isfinite(v) for v in history['train_loss']))
            effective = json.loads((root / 'models/cells/effective_config.json').read_text())
            self.assertIn('cytoplasm', json.dumps(effective))
            self.assertTrue((root / 'models/cells/weights_epoch_002.weights.h5').exists())


def mixed_dataset(root, splits=(('train', (BOTH, BOTH, ['cytoplasm'], ['cytoplasm'], ['nucleus'], ['nucleus'])),)):
    """Tiles whose mask folders differ per file: both levels, cells only, nuclei only."""
    for split, files in splits:
        for folder in ('images', 'nucleus_masks', 'cytoplasm_masks'):
            (root / split / folder).mkdir(parents=True)
        for i, levels in enumerate(files):
            image, masks = tile(100 * len(split) + i)
            tifffile.imwrite(root / split / 'images' / f't{i}.tif', image, photometric='minisblack')
            for level in levels:
                tifffile.imwrite(root / split / (level + '_masks') / f't{i}.tif', masks[level], photometric='minisblack')


class PartialAnnotationTests(unittest.TestCase):
    def test_setting_contract(self):
        self.assertIs(CytoNucsConfig.FIXES_SETTINGS['training']['partial_annotation'], False)
        for path in ROOT.glob('config_cytonucs*.json'):
            self.assertIs(json.loads(path.read_text())['training']['partial_annotation'], False, path.name)
        self.assertIs(quiet(config, partial_annotation=True).partial_annotation, True)
        for bad in (1, 'yes', None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                quiet(config, partial_annotation=bad)

    def test_batches_stay_inside_one_set_of_annotated_levels(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            mixed_dataset(Path(tmp))
            cfg = quiet(config, partial_annotation=True, image_batch_size=2)
            generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')
            self.assertEqual({k: len(v) for k, v in generator._annotation_groups.items()},
                             {('nucleus', 'cytoplasm'): 2, ('cytoplasm',): 2, ('nucleus',): 2})
            seen = set()
            for index in range(40):
                _, y = generator[index]
                present = tuple(y['levels_present'].tolist())
                seen.add(present)
                for flag, level in zip(present, BOTH):
                    per_crop = [int(y[level + '_instances'][b].max()) > 0 for b in range(2)]
                    self.assertEqual(per_crop, [bool(flag)] * 2, (index, level))    # every tile here has 4 objects
            self.assertEqual(seen, {(1., 1.), (0., 1.), (1., 0.)})
            # Without the setting a missing mask is an error; validation is always strict.
            with self.assertRaises(AssertionError):
                quiet(CytoNucsDataGenerator, Path(tmp) / 'train', quiet(config), cfg.rays, subset='train')
            with self.assertRaises(AssertionError):
                quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='val', shuffle=False)
            # The default writes no flag: batches are what they were.
            dataset(Path(tmp) / 'full', BOTH)
            full = quiet(CytoNucsDataGenerator, Path(tmp) / 'full' / 'train', quiet(config), cfg.rays, subset='train')
            self.assertNotIn('levels_present', full[0][1])
            self.assertIsNone(full._annotation_groups)

    def test_files_must_have_an_image_and_some_mask(self):
        cfg = quiet(config, partial_annotation=True)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            mixed_dataset(Path(tmp))
            _, masks = tile(0)
            tifffile.imwrite(Path(tmp) / 'train' / 'nucleus_masks' / 'stray.tif', masks['nucleus'], photometric='minisblack')
            with self.assertRaisesRegex(ValueError, 'without an image'):
                quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            mixed_dataset(Path(tmp), (('train', (BOTH, [])),))
            with self.assertRaisesRegex(ValueError, 'without masks'):
                quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')

    def test_unannotated_level_does_not_train_and_default_is_exact(self):
        settings = dict(FEATURES, lambda_wbr=0., lambda_containment=0.)       # no terms tying nuclei to cells
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            mixed_dataset(Path(tmp))
            cfg = quiet(config, partial_annotation=True, **settings)
            generator = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')
            model = quiet(build_cytonucs_model, cfg, basedir=tmp, name='partial')
            loss = CytoNucsLoss(cfg)
            batches = {}
            for index in range(40):
                x, y = tensors(generator[index])
                batches.setdefault(tuple(y['levels_present'].numpy().tolist()), (x, y))
            self.assertEqual(len(batches), 3)

            def gradients(x, y):
                with tf.GradientTape() as tape:
                    value = loss(y, model.keras_model(x, training=True))
                grads = tape.gradient(value, model.keras_model.trainable_variables)
                size = {}
                for g, v in zip(grads, model.keras_model.trainable_variables):
                    layer = v.path.split('/')[0]
                    # The nuclear context layers feed the cell decoder from detached nuclear features:
                    # they belong to the cell path.
                    owner = 'shared' if 'ctx' in layer or 'context' in layer else \
                        next((h for h in BOTH if layer.startswith((TAG[h] + '_', h + '_'))), 'shared')
                    size[owner] = size.get(owner, 0.) + (0. if g is None else float(tf.reduce_sum(tf.abs(g))))
                return float(value), size

            for present, (x, y) in batches.items():
                value, size = gradients(x, y)
                self.assertTrue(np.isfinite(value))
                for flag, level in zip(present, BOTH):
                    if flag:
                        self.assertGreater(size[level], 0., (present, level))
                    else:
                        self.assertEqual(size[level], 0., (present, level))
            # Flags of a fully annotated batch change nothing: the same number as without them.
            x, y = batches[(1., 1.)]
            bare = {k: v for k, v in y.items() if k != 'levels_present'}
            prediction = model.keras_model(x, training=False)
            self.assertEqual(float(loss(y, prediction)), float(loss(bare, prediction)))
            # An unannotated level leaves the loss when its flag is 0, whatever the network says there.
            x, y = batches[(0., 1.)]
            prediction = dict(model.keras_model(x, training=False))
            base = float(loss(y, prediction))
            prediction['nucleus_prob_logit'] = prediction['nucleus_prob_logit'] + 3.
            prediction['nucleus_prob'] = tf.sigmoid(prediction['nucleus_prob_logit'])
            self.assertAlmostEqual(float(loss(y, prediction)), base, places=6)
            on = dict(y, levels_present=tf.constant([1., 1.]))
            self.assertGreater(abs(float(loss(on, prediction)) - float(loss(on, dict(model.keras_model(x, training=False))))), 1e-3)

    def test_train_side_metric_uses_fully_annotated_tiles(self):
        from types import SimpleNamespace
        from cytonucs_trainer import DiceCallback
        _, masks = tile(0)
        pool = [dict(filename=f't{i}_z000_y0000_x{64 * i:04d}_a00.tif', nucleus_mask=masks['nucleus'], cytoplasm_mask=masks['cytoplasm'],
                     annotated=levels) for i, levels in enumerate((('cytoplasm',), ('nucleus', 'cytoplasm'), ('nucleus',), ('nucleus', 'cytoplasm')))]
        callback = SimpleNamespace(model=SimpleNamespace(config=quiet(config, fullval_train_volumes=4)),
                                   train_generator=SimpleNamespace(data_pool=pool))
        picked = DiceCallback._train_volumes(callback)
        self.assertEqual(sorted(s['filename'][:2] for s in picked), ['t1', 't3'])

    def test_training_run_on_mixed_files(self):
        import train_cytonucs
        raw = json.loads((ROOT / 'config_cytonucs_v4.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            mixed_dataset(root / 'data', (('train', (BOTH, BOTH, ['cytoplasm'], ['cytoplasm'], ['nucleus'], ['nucleus'])),
                                          ('val', (BOTH, BOTH)), ('test', (BOTH, BOTH))))
            raw['experiment']['name'] = 'mixed'
            raw['data']['dataset_dir'] = str(root / 'data') + '/'
            raw['model'].update(n_rays=12, n_channel_in=1, use_gpu=False, unet_n_depth=2, unet_n_filter_base=4,
                                net_conv_after_unet=8, mask_refinement_filters=8, unet_norm_groups=2,
                                unet_pool_schedule=[[1, 2, 2], [2, 2, 2]],
                                unet_kernel_schedule=[[1, 3, 3], [3, 3, 3], [3, 3, 3]], anisotropy=[3, 1, 1])
            raw['training'].update(train_patch_size=[8, 32, 32], train_epochs=2, steps_per_epoch=4, train_region_crops=False,
                                   partial_annotation=True, lr_warmup_epochs=0, predicted_prompt_start_epoch=1,
                                   prefetch_workers=1, prefetch_depth=1)
            raw['inference'].update(refinement_patch_size=[0, 32, 32],
                                    checkpoint_path=str(root / 'models/mixed/weights_best.weights.h5'))
            raw['evaluation'].update(eval_dice_every=1, min_eval_epoch=1, save_every_n_epochs=1, fullval_train_volumes=2,
                                     save_predictions=False)
            raw['gpu_config']['visible_devices'] = '-1'
            raw['output']['basedir'] = str(root / 'models')
            (root / 'config.json').write_text(json.dumps(raw))
            cwd = os.getcwd()
            try:
                os.chdir(root)
                quiet(train_cytonucs.main, str(root / 'config.json'))
            finally:
                os.chdir(cwd)
            history = json.loads((root / 'models/mixed/history.json').read_text())
            self.assertEqual(history['fullval_epoch'], [1, 2])
            self.assertTrue(all(np.isfinite(v) for v in history['train_loss']))
            self.assertEqual(history['fullval_train_volumes'], [2, 2])           # the two fully annotated tiles


if __name__ == '__main__':
    unittest.main()
