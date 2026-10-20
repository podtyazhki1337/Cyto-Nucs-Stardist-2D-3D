"""v2.17: per-object crop-cut masks instead of per-axis flags; optional supervision of crop-cut objects."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
import tifffile
from test_fixes import config, ROOT
from test_architecture_v2 import batch
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import CytoNucsDataGenerator, label_counts, partial_labels
from cytonucs_losses import CytoNucsLoss, quality_supervision_weights
from cytonucs_model import build_cytonucs_model


def make_split(folder, count=3):
    """8x48x48 volumes: one object cut by the volume face x=0, one interior, one wide object
    the 32-voxel training crop must cut (x 8..40)."""
    for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
        (folder / sub).mkdir(parents=True)
    zz, yy, xx = np.mgrid[:8, :48, :48]
    for i in range(count):
        cells = np.zeros((8, 48, 48), np.uint16); nuclei = np.zeros_like(cells)
        cells[1:7, 2:14, 0:12] = 1
        nuclei[((zz - 3.5) / 2.2) ** 2 + ((yy - 8) / 4.) ** 2 + (xx / 4.) ** 2 <= 1] = 101
        cells[1:7, 18:30, 26:46] = 2
        nuclei[((zz - 3.5) / 2.2) ** 2 + ((yy - 24) / 4.) ** 2 + ((xx - 36) / 4.) ** 2 <= 1] = 102
        cells[1:7, 34:46, 8:40] = 3
        nuclei[((zz - 3.5) / 2.2) ** 2 + ((yy - 40) / 4.) ** 2 + ((xx - 24) / 14.) ** 2 <= 1] = 103
        image = .1 + .4 * (cells > 0) + .5 * (nuclei > 0)
        for sub, arr in (('images', image.astype(np.float32)), ('nucleus_masks', nuclei), ('cytoplasm_masks', cells)):
            tifffile.imwrite(folder / sub / f'vol{i}.tif', arr, photometric='minisblack', metadata={'axes': 'ZYX'})
        (folder / 'assignments' / f'vol{i}.json').write_text(json.dumps({101: 1, 102: 2, 103: 3}))
    return label_counts(nuclei), label_counts(cells)


class HelperTests(unittest.TestCase):
    def test_counts_and_partial_labels_sparse_ids(self):
        full = np.zeros((4, 4), np.int64); full[0, :] = 7; full[2, :2] = 900000000
        counts = label_counts(full)
        self.assertEqual(counts, {0: 10, 7: 4, 900000000: 2})
        crop = full[:, :2]
        self.assertEqual(partial_labels(crop, counts), {7})          # 7 continues outside, the other does not
        self.assertEqual(label_counts(np.zeros((0,), np.int32)), {})


class GeneratorTests(unittest.TestCase):
    def run_generator(self, augment, patch=(8, 32, 32), steps=24, fg_crop_prob=1.0):
        from cytonucs_data_generator import cytonucs_augmenter_3d
        rows = []
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp, contextlib.redirect_stdout(io.StringIO()):
            nuc_counts, cell_counts = make_split(Path(tmp) / 'train')
            cfg = config(train_patch_size=patch, nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1),
                         fg_crop_prob=fg_crop_prob, prompts_per_crop=2, zoom_range=None)
            gen = CytoNucsDataGenerator(dataset_dir=Path(tmp) / 'train', config=cfg, rays=cfg.rays,
                                        augmenter=cytonucs_augmenter_3d if augment else None, subset='train')
            for step in range(steps):
                gen.epoch = step
                _, y = gen[step % len(gen)]
                for head, full in (('nucleus', nuc_counts), ('cytoplasm', cell_counts)):
                    labels = np.asarray(y[head + '_instances']); cut = np.asarray(y[head + '_crop_cut'])
                    self.assertEqual(cut.shape, labels.shape)
                    for b in range(len(labels)):
                        xy_flag = bool(np.asarray(y['crop_truncation_axes'])[b][1:].any())
                        for k, n in label_counts(labels[b]).items():
                            if k:
                                marked = set(np.unique(cut[b][labels[b] == k])) == {1}
                                lab = labels[b] == k
                                touches_xy = bool(lab[:, 0].any() or lab[:, -1].any() or lab[..., 0].any() or lab[..., -1].any())
                                rows.append((head, k, n < full[k], marked, touches_xy, xy_flag))
        return rows

    def test_mask_marks_exactly_the_objects_the_crop_cut(self):
        for augment in (False, True):
            rows = self.run_generator(augment)
            self.assertTrue(rows)
            for head, k, truly_cut, marked, *_ in rows:
                self.assertEqual(marked, truly_cut, (augment, head, k))
            self.assertTrue(any(r[3] for r in rows))                  # crops do cut objects

    def test_whole_object_on_volume_face_is_kept_although_axis_flag_is_set(self):
        # Training mode needs the augmenter (flips/rot90 move the face, so any XY face counts).
        rows = self.run_generator(True, patch=(8, 48, 46), steps=60, fg_crop_prob=0.)   # uniform x start 0..2
        face = [r for r in rows if r[1] in (101, 1) and not r[2] and r[4]]
        self.assertTrue(face)                                         # whole, touching an XY crop face
        self.assertTrue(all(r[5] for r in face))                      # the old per-axis flag would drop it
        self.assertFalse(any(r[3] for r in face))                     # the mask keeps it


class LossTests(unittest.TestCase):
    def test_object_losses_use_masks_and_include_option(self):
        cfg = config(nucleus_prob_object_weight=.25, mask_quality_head=True, mask_quality_weight=.1,
                     mask_quality_ignore_boundary_targets=True)
        model = build_cytonucs_model(cfg); x, y = batch(cfg); pred = model.keras_model(x)
        labels = y['nucleus_instances'].numpy().copy(); labels[0, 2:6, 12:20, 0:3] = 2
        y = dict(y); y['nucleus_instances'] = tf.constant(labels)
        counts = {}
        for name, cut, flags, include in (('none', None, None, False),
                                          ('axis_flags_only', None, [[0., 1., 1.]], False),
                                          ('mask_marks_2', 2, [[0., 1., 1.]], False),
                                          ('mask_marks_nothing', 0, [[0., 1., 1.]], False),
                                          ('include', 2, [[0., 1., 1.]], True)):
            yy = dict(y)
            if flags is not None:
                yy['crop_truncation_axes'] = tf.constant(flags)
            if cut is not None:
                yy['nucleus_crop_cut'] = tf.constant((labels == cut) & (cut > 0), tf.int32)
                yy['cytoplasm_crop_cut'] = tf.zeros_like(y['cytoplasm_instances'], tf.int32)
            cfg.object_losses_include_crop_cut = include
            loss = CytoNucsLoss(cfg); loss(yy, pred, current_epoch=516)
            counts[name] = float(loss.last_components['nucleus_prob_sampled_object_count'])
        # Masks replace the axis rule: an object at a flagged face is kept unless the mask marks it.
        self.assertEqual(counts, {'none': 2., 'axis_flags_only': 1., 'mask_marks_2': 1., 'mask_marks_nothing': 2.,
                                  'include': 2.})

    def test_quality_eligibility_by_mask(self):
        t = np.zeros((3, 4, 8, 8), np.float32)
        t[0, :, 2:5, 2:5] = 1; t[1, 1:3, 0:3, 0:3] = 1; t[2, 1:3, 5:8, 5:8] = 1
        cut = np.zeros_like(t, np.int32); cut[1, 1:3, 0:3, 0:3] = 1
        w, pos = quality_supervision_weights(tf.constant(t), 3, crop_cut=tf.constant(cut))
        np.testing.assert_array_equal(w, [1, 0, 1]); np.testing.assert_array_equal(pos, [1, 1, 1])
        w, _ = quality_supervision_weights(tf.constant(t), 3, crop_cut=tf.zeros_like(tf.constant(cut)))
        np.testing.assert_array_equal(w, [1, 1, 1])                   # full Z / faces no longer exclude
        w, _ = quality_supervision_weights(tf.constant(t[1:2]), 3, crop_cut=tf.constant(cut[1:2]))
        np.testing.assert_array_equal(w, [0])                         # no complete positive: skip head


class ConfigTests(unittest.TestCase):
    def test_json_contract_and_validation(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            self.assertIs(raw['loss_params']['object_losses_include_crop_cut'], path.name in ('config_cytonucs_v2_17.json', 'config_cytonucs_v3.json', 'config_cytonucs_v3_1.json', 'config_cytonucs_v3_2.json', 'config_cytonucs_v3_3.json'),
                          path.name)
        raw = json.loads((ROOT / 'config_cytonucs_v2_17.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['training']['resume_from_checkpoint'] = None; raw['inference']['checkpoint_path'] = None
            raw['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(raw))
            with contextlib.redirect_stdout(io.StringIO()):
                preset = CytoNucsConfig.from_json(path)
        self.assertIs(preset.object_losses_include_crop_cut, True)
        self.assertIs(CytoNucsConfig.FIXES_SETTINGS['loss_params']['object_losses_include_crop_cut'], False)
        with self.assertRaises(ValueError):
            config(object_losses_include_crop_cut=1)


if __name__ == '__main__':
    unittest.main()
