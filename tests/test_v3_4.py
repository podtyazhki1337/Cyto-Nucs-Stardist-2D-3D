"""v3.4: training tiles sampled by the number of objects they hold; HeLa-only preset."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from test_fixes import config, ROOT
from cytonucs_config import CytoNucsConfig
from cytonucs_data_generator import CytoNucsDataGenerator

PRESET = 'config_cytonucs_v3_4.json'
KEY = 'tile_sampling_by_objects'


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def make_tiles(root, nuclei_per_tile):
    """One tile per entry with that many nuclei, each inside its own cell (2 objects per nucleus)."""
    for folder in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
        (root / folder).mkdir(parents=True)
    for index, count in enumerate(nuclei_per_tile):
        nuclei = np.zeros((8, 32, 32), np.uint16); cells = np.zeros_like(nuclei)
        for k in range(count):
            y, x = 2 + 10 * (k // 3), 2 + 10 * (k % 3)
            cells[1:7, y:y + 8, x:x + 8] = k + 1
            nuclei[2:6, y + 2:y + 6, x + 2:x + 6] = k + 1
        image = np.random.default_rng(index).normal(size=nuclei.shape).astype(np.float32) + (nuclei > 0)
        for folder, array in (('images', image), ('nucleus_masks', nuclei), ('cytoplasm_masks', cells)):
            tifffile.imwrite(root / folder / f'tile{index}.tif', array, photometric='minisblack')
        (root / 'assignments' / f'tile{index}.json').write_text(json.dumps({k + 1: k + 1 for k in range(count)}))


class TileSamplingTests(unittest.TestCase):
    COUNTS = (0, 1, 4)                      # objects per tile: 0, 2, 8

    def generator(self, tmp, share):
        cfg = quiet(config, **{KEY: share})
        return quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='train')

    def frequencies(self, generator, draws=6000):
        rng = np.random.default_rng(1)
        names = [s['filename'] for s in generator.data_pool]
        picks = np.bincount([generator._pick_tile(rng) for _ in range(draws)], minlength=len(names)) / draws
        return {name: picks[i] for i, name in enumerate(names)}

    def test_tiles_are_drawn_by_object_count(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            make_tiles(Path(tmp) / 'train', self.COUNTS)
            by_objects = self.generator(tmp, 1.0)
            counts = {s['filename']: s['object_count'] for s in by_objects.data_pool}
            self.assertEqual(counts, {'tile0.tif': 0, 'tile1.tif': 2, 'tile2.tif': 8})
            f = self.frequencies(by_objects)
            self.assertEqual(f['tile0.tif'], 0.)
            self.assertAlmostEqual(f['tile1.tif'], .2, delta=.02)
            self.assertAlmostEqual(f['tile2.tif'], .8, delta=.02)
            # Half of the crops by objects, half uniform: empty tiles keep a share.
            f = self.frequencies(self.generator(tmp, .5))
            self.assertAlmostEqual(f['tile0.tif'], .5 / 3, delta=.02)
            self.assertAlmostEqual(f['tile2.tif'], .5 * .8 + .5 / 3, delta=.02)
            # The default keeps the uniform choice and its random sequence.
            uniform = self.generator(tmp, 0.)
            a, b = np.random.default_rng(7), np.random.default_rng(7)
            self.assertEqual([uniform._pick_tile(a) for _ in range(50)], [int(b.choice(3)) for _ in range(50)])
            f = self.frequencies(uniform)
            for name in f:
                self.assertAlmostEqual(f[name], 1 / 3, delta=.02)

    def test_batches_come_from_the_drawn_tiles(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            make_tiles(Path(tmp) / 'train', (0, 0, 3))
            generator = self.generator(tmp, 1.0)
            for index in range(6):                              # the only tile with objects, every time
                _, y = generator[index]
                self.assertEqual(len(np.unique(y['nucleus_instances'])) - 1, 3)
            # Validation generators keep their fixed order.
            cfg = quiet(config, **{KEY: 1.0})
            val = quiet(CytoNucsDataGenerator, Path(tmp) / 'train', cfg, cfg.rays, subset='val', shuffle=False)
            seen = [len(np.unique(val[i][1]['nucleus_instances'])) - 1 for i in range(3)]
            self.assertEqual(sorted(seen), [0, 0, 3])

    def test_tiles_without_objects_fall_back_to_uniform(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            make_tiles(Path(tmp) / 'train', (0, 0))
            generator = self.generator(tmp, 1.0)
            self.assertIsNone(generator._tile_weights)
            f = self.frequencies(generator, 2000)
            for name in f:
                self.assertAlmostEqual(f[name], .5, delta=.05)


class ConfigTests(unittest.TestCase):
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
            self.assertEqual(raw['training'][KEY], .5 if path.name == PRESET else 0., path.name)
        self.assertEqual(CytoNucsConfig.FIXES_SETTINGS['training'][KEY], 0.)
        raw = json.loads((ROOT / PRESET).read_text())
        self.assertEqual(getattr(self.load(raw), KEY), .5)
        for value in (-.1, 1.5, True, 'objects'):
            bad = json.loads(json.dumps(raw)); bad['training'][KEY] = value
            with self.assertRaises(ValueError, msg=repr(value)):
                self.load(bad)

    def test_preset_is_v3_3_on_hela_only(self):
        base = json.loads((ROOT / 'config_cytonucs_v3_3.json').read_text())
        raw = json.loads((ROOT / PRESET).read_text())
        changed = {(section, key) for section in raw if isinstance(raw[section], dict)
                   for key in raw[section] if raw[section][key] != base[section].get(key)}
        self.assertEqual(changed, {
            ('experiment', 'name'), ('data', 'dataset_dir'), ('training', KEY), ('loss_params', 'checkpoint_w_jtpr'),
            ('inference', 'checkpoint_path'), ('evaluation', 'fullval_train_volumes'),
            ('evaluation', 'save_every_n_epochs')})
        self.assertEqual(set(raw), set(base))
        self.assertIsNone(raw['training']['resume_from_checkpoint'])
        self.assertIn(raw['experiment']['name'], raw['inference']['checkpoint_path'])
        cfg = self.load(raw)
        self.assertEqual((cfg.n_rays, cfg.dist_target_stride, cfg.checkpoint_w_jtpr), (32, [1, 2, 2], .5))


if __name__ == '__main__':
    unittest.main()
