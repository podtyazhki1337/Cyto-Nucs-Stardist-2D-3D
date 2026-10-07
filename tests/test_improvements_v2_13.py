"""v2.13: anisotropy-aware refiner pooling, realistic negative prompts, declared window defaults, pad A/B."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from test_fixes import ROOT, config
from test_architecture_v2 import batch
from cytonucs_config import CytoNucsConfig
from cytonucs_geometry import refinement_pool_schedule, refinement_factors, architecture_plan
from cytonucs_model import build_cytonucs_model
from cytonucs_mask_quality import gaussian_prompt_geometry
from cytonucs_inference import CytoNucsPredictor, _pad_input


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        return fn(*args, **kwargs), out.getvalue()


class RefinerPoolingTests(unittest.TestCase):
    def test_schedule_follows_physical_step_and_keeps_preset_multiple(self):
        cfg = config(anisotropy=(5, 1, 1), refinement_depth=2)
        self.assertEqual(refinement_pool_schedule(cfg, 'nucleus'), [(2, 2, 2), (2, 2, 2)])
        cfg.refinement_pool = 'auto'
        # Z step 5 vs XY step 2 at grid (1,2,2): pool XY first, then all axes.
        self.assertEqual(refinement_pool_schedule(cfg, 'nucleus'), [(1, 2, 2), (2, 2, 2)])
        self.assertEqual(refinement_factors(cfg, 'cytoplasm'), (2, 4, 4))
        raw = json.loads((ROOT / 'config_cytonucs_v2_13.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            raw['training']['resume_from_checkpoint'] = None; raw['inference']['checkpoint_path'] = None
            raw['output']['basedir'] = tmp; path = Path(tmp) / 'c.json'; path.write_text(json.dumps(raw))
            preset, _ = quiet(CytoNucsConfig.from_json, path)
        self.assertEqual(preset.refinement_pool, 'auto')
        self.assertEqual(architecture_plan(preset)['input_multiple'], (4, 32, 32))

    def test_same_weights_shapes_and_compatible_restore(self):
        iso = config(anisotropy=(5, 1, 1), refinement_depth=2, mask_quality_head=True)
        auto = config(anisotropy=(5, 1, 1), refinement_depth=2, mask_quality_head=True, refinement_pool='auto')
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            old, _ = quiet(build_cytonucs_model, iso, basedir=tmp)
            new, _ = quiet(build_cytonucs_model, auto, basedir=tmp)
            self.assertEqual([w.shape for w in old.keras_model.weights], [w.shape for w in new.keras_model.weights])
            path = Path(tmp) / 'w.weights.h5'; old.save_weights(path)
            _, printed = quiet(new.load_weights, path)
            self.assertIn('Refiner pooling: isotropic -> auto', printed)
            for a, b in zip(old.keras_model.weights, new.keras_model.weights):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            x, _ = batch(auto)
            out = new.keras_model(x, training=False)
            self.assertEqual(out['nucleus_mask'].shape, old.keras_model(x, training=False)['nucleus_mask'].shape)
            self.assertTrue(np.isfinite(out['nucleus_mask'].numpy()).all())
            # A genuinely different architecture is still refused.
            other, _ = quiet(build_cytonucs_model, config(anisotropy=(5, 1, 1), refinement_depth=1,
                                                         mask_quality_head=True, refinement_pool='auto'), basedir=tmp)
            with self.assertRaisesRegex(ValueError, 'architecture differs'):
                quiet(other.load_weights, path)

    def test_invalid_settings_fail(self):
        for args in (dict(refinement_pool='anisotropic'), dict(nms_candidate_spacing=0),
                     dict(nms_candidate_spacing=True), dict(nms_border_exclude_xy=-1),
                     dict(nms_border_exclude_xy=1.5),
                     dict(refinement_pool='auto', refinement_depth=2, anisotropy=(1, 1, 1), train_patch_size=(6, 32, 32))):
            with self.assertRaises(ValueError):
                quiet(config, **args)


class NegativePromptTests(unittest.TestCase):
    def test_negatives_are_nucleus_sized_off_nucleus_and_keep_cell_owner(self):
        from test_pipeline_review import make_dataset
        from cytonucs_data_generator import CytoNucsDataGenerator
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp); make_dataset(root)
            cfg, _ = quiet(config, negative_prompt_prob=1.)
            generator, _ = quiet(CytoNucsDataGenerator, root, cfg, cfg.rays)
            seen_cell = seen_background = 0
            for step in range(6):
                generator.set_epoch(step)
                x, y = generator[0]
                nuclei, cells = y['nucleus_instances'][0], y['cytoplasm_instances'][0]
                sizes = {tuple(np.ptp(np.argwhere(nuclei == lab), 0) + 1) for lab in np.unique(nuclei) if lab}
                for k in range(cfg.prompts_per_crop):
                    self.assertFalse(y['nucleus_target'][k].any())
                    center, sigma = gaussian_prompt_geometry(x[0, ..., 1 + 2 * k])
                    point = tuple(np.round(center).astype(int))
                    self.assertEqual(nuclei[point], 0)
                    self.assertIn(tuple(np.round(np.asarray(sigma) * 4).astype(int)), {tuple(s) for s in sizes})
                    owner = cells[point]
                    np.testing.assert_array_equal(y['cytoplasm_target'][k] > .5, (cells == owner) if owner else False)
                    seen_cell += bool(owner); seen_background += not owner
            self.assertGreater(seen_cell, 0); self.assertGreater(seen_background, 0)


class PadModeTests(unittest.TestCase):
    def test_pad_modes_and_predictor_shapes(self):
        a = np.arange(6, dtype=np.float32).reshape(1, 6)
        np.testing.assert_array_equal(_pad_input(a, [(0, 0), (0, 2)], 'constant')[0, -2:], [0, 0])
        np.testing.assert_array_equal(_pad_input(a, [(0, 0), (0, 2)], 'reflect')[0, -2:], [4, 3])
        np.testing.assert_array_equal(_pad_input(a[:, :1], [(0, 0), (0, 2)], 'reflect')[0], [0, 0, 0])
        cfg, _ = quiet(config)
        model, _ = quiet(build_cytonucs_model, cfg)
        x = np.zeros((7, 31, 29, 1), np.float32)
        for mode in ('reflect', 'constant'):
            predictor = CytoNucsPredictor(model, cfg, prob_thresh_nucleus=1., prob_thresh_cytoplasm=1.)
            predictor.input_pad_mode = mode
            (n, c, _), _ = quiet(predictor.predict_instances, x, normalize_img=False)
            self.assertEqual(n.shape, x.shape[:3])
        help_text = subprocess.run([sys.executable, str(ROOT / 'audit' / 'refinement_stage_probe.py'), '--help'],
                                   capture_output=True, text=True, encoding='utf-8', errors='replace').stdout
        self.assertIn('--pad-mode', help_text)


class DeclaredDefaultsTests(unittest.TestCase):
    def test_every_preset_declares_legacy_window_defaults(self):
        for path in ROOT.glob('config_cytonucs*.json'):
            raw = json.loads(path.read_text())
            self.assertEqual((raw['inference']['nms_candidate_spacing'], raw['inference']['nms_border_exclude_xy']),
                             (0.12, 2), path.name)
            self.assertEqual(raw['model']['refinement_pool'], 'auto' if path.name in ('config_cytonucs_v2_13.json', 'config_cytonucs_v2_14.json', 'config_cytonucs_v2_15.json', 'config_cytonucs_v2_16.json', 'config_cytonucs_v2_17.json', 'config_cytonucs_v3.json', 'config_cytonucs_v3_1.json', 'config_cytonucs_v3_2.json', 'config_cytonucs_v3_3.json') else 'isotropic')


if __name__ == '__main__':
    unittest.main()
