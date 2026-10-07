"""File counts of dataset_prep/make_spatial_split_patches.py for the server geometry, at 1/8 of the XY size.

Four synthetic volumes of 256 x 256 px with depths 45 / 17 / 16 / 16 and patches 16 x 64 x 64, stride 8 x 56 x 56
stand for volumes of 2048 x 2048 px with patches 16 x 512 x 512, stride 8 x 448 x 448: the same 4 x 4 blocks of
exactly one patch and the same window grid. The depths of the real volumes are inferred from the file sizes and
are not verified. Counts only; nothing here is a measurement on the user's data.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'dataset_prep')]
import make_spatial_split_patches as prep

PATCH, STRIDE = (16, 64, 64), (8, 56, 56)
VOLUMES = (('HeLa', 45), ('EA_002', 17), ('EA_004', 16), ('EA_006', 16))


def source(seed, shape):
    """Cells with one nucleus each over the whole volume, so that no patch is dropped as background."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    cells = np.zeros(shape, np.uint16); nuclei = np.zeros(shape, np.uint16); label = 0
    for cy in range(8, shape[1], 18):
        for cx in range(8, shape[2], 18):
            for cz in range(6, shape[0], 12):
                label += 1
                c = (cz + rng.uniform(-2, 2), cy + rng.uniform(-3, 3), cx + rng.uniform(-3, 3))
                d = ((zz - c[0]) / 5.) ** 2 + ((yy - c[1]) / 8.) ** 2 + ((xx - c[2]) / 8.) ** 2
                cells[(d <= 1) & (cells == 0)] = label
                nuclei[(d <= .2) & (cells == label)] = label
    image = np.stack([40 + 90 * (cells > 0), 30 + 150 * (nuclei > 0), np.full(shape, 20)], -1).astype(np.uint8)
    return image, nuclei, cells


def main():
    (ROOT / '.cache').mkdir(exist_ok=True)
    volumes = [dict(zip(('image', 'nucleus', 'cytoplasm'), source(i, (depth, 256, 256))), name=name)
               for i, (name, depth) in enumerate(VOLUMES)]
    report = dict(scope=__doc__.strip(), patch=PATCH, stride=STRIDE, volumes={})
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
        manifest = prep.build(volumes, Path(tmp) / 'out', PATCH, STRIDE, augmentations=4)
    for volume in manifest['volumes']:
        files = [f for f in manifest['files'] if f['volume'] == volume['name']]
        train = [f for f in files if f['split'] == 'train' and f['augmentation_index'] == 0]
        letters = np.array([dict(train='T', val='v', test='x')[b['split']] for b in volume['blocks']]).reshape(4, 4)
        report['volumes'][volume['name']] = dict(
            shape=volume['shape'], blocks=[''.join(row) for row in letters],
            files={s: v['files'] for s, v in volume['summary'].items()},
            originals={s: v['originals'] for s, v in volume['summary'].items()},
            train_xy_windows=len({(f['y0'], f['x0']) for f in train}),
            train_xy_windows_on_two_or_more_blocks=len({(f['y0'], f['x0']) for f in train if len(f['blocks']) > 1}),
            train_z=sorted({f['z0'] for f in train}),
            val_z=sorted({f['z0'] for f in files if f['split'] == 'val'}),
            val_tile_shapes=sorted({tuple(f['shape']) for f in files if f['split'] == 'val'}))
    report.update(total_files={s: v['files'] for s, v in manifest['summary'].items()},
                  total_originals={s: v['originals'] for s, v in manifest['summary'].items()}, skipped=manifest['skipped'])
    (ROOT / 'audit' / 'spatial_split_layout.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, volume in report['volumes'].items():
        print(name, {k: v for k, v in volume.items()})
    print('TOTAL', report['total_files'], report['total_originals'], report['skipped'])


if __name__ == '__main__':
    main()
