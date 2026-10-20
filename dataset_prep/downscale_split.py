"""Copy of a CytoNucs dataset at a lower XY resolution (Z unchanged).

Training at half the XY resolution gave the validation AP of the native resolution about four
times faster per epoch on the HeLa spheroid (v5): Z steps are five times coarser than XY there,
and objects stay 45-95 px wide. This writes, for an integer factor f,
  images            mean of f x f pixels
  masks             the label that most of the f x f pixels carry (ties: the top-left one)
  assignments, meta copied; meta object extents divided by f in Y and X
  train_regions/    the same for the whole volumes, train blocks divided by f (if present)
  split_manifest.json, split_info.json   coordinates divided by f, 'xy_downscale': f,
                    'downscaled_from': the native dataset
Labels smaller than f x f pixels in every slice can disappear; their number is printed.
score_native_resolution.py compares predictions at the lower resolution with the original masks,
after enlarging them f times in Y and X.

Usage:
  python dataset_prep/downscale_split.py --dataset DIR --output DIR_xy2 [--factor 2]
Run dataset_prep/add_train_regions.py on the native dataset first, if whole-volume crops are wanted.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

FOLDERS = ('train', 'val', 'test', 'train_regions')


def downscale_image(image, factor):
    """Mean of factor x factor pixels in Y and X; axes Z, Y, X[, C]. Integer images are rounded."""
    z, h, w = image.shape[:3]
    h, w = h // factor * factor, w // factor * factor
    blocks = image[:, :h, :w].reshape((z, h // factor, factor, w // factor, factor) + image.shape[3:]).astype(np.float32)
    out = blocks.mean((2, 4))
    return np.round(out).astype(image.dtype) if np.issubdtype(image.dtype, np.integer) else out.astype(image.dtype)


def downscale_labels(labels, factor):
    """The label carried by most of the factor x factor pixels in Y and X; ties go to the first
    pixel in row order (the top-left one)."""
    h, w = labels.shape[-2] // factor * factor, labels.shape[-1] // factor * factor
    views = np.stack([labels[..., i:h:factor, j:w:factor] for i in range(factor) for j in range(factor)])
    counts = sum((views == view[None]).astype(np.int16) for view in views)
    return np.take_along_axis(views, np.argmax(counts, axis=0)[None], axis=0)[0]


def mask_folders(folder):
    """Every '<level>_masks' folder of a dataset folder (the levels of model.levels)."""
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name.endswith('_masks'))


def scale_manifest(manifest, factor):
    manifest = json.loads(json.dumps(manifest))
    for key in ('patch_size', 'train_stride'):
        if key in manifest:
            manifest[key] = [manifest[key][0]] + [v // factor for v in manifest[key][1:]]
    for volume in manifest.get('volumes', []):
        volume['shape'] = [volume['shape'][0]] + [v // factor for v in volume['shape'][1:]]
        for block in volume.get('blocks', []):
            for key in ('y0', 'y1', 'x0', 'x1'):
                block[key] //= factor
    for row in manifest.get('files', []):
        for key in ('y0', 'x0'):
            if key in row:
                row[key] //= factor
        if 'shape' in row:
            row['shape'] = [row['shape'][0]] + [v // factor for v in row['shape'][1:]]
    manifest['xy_downscale'] = factor * int(manifest.get('xy_downscale', 1))
    return manifest


def downscale_split(dataset, output, factor=2):
    dataset, output = Path(dataset), Path(output)
    if factor < 2 or int(factor) != factor:
        raise ValueError('factor must be an integer of at least 2')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty')
    lost = {}
    for folder in FOLDERS:
        source = dataset / folder
        if not (source / 'images').is_dir():
            continue
        names = sorted(p.name for p in (source / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
        for sub in ['images'] + mask_folders(source):
            (output / folder / sub).mkdir(parents=True, exist_ok=True)
        for name in names:
            image = downscale_image(tifffile.imread(source / 'images' / name), factor)
            # Without a channel axis, a volume of 3 or 4 slices would be stored as one RGB(A) image.
            tifffile.imwrite(output / folder / 'images' / name, image, **({'photometric': 'minisblack'} if image.ndim == 3 else {}))
            for sub in mask_folders(source):
                if not (source / sub / name).is_file():           # partially annotated files
                    continue
                full = tifffile.imread(source / sub / name)
                small = downscale_labels(full, factor)
                lost[sub] = lost.get(sub, 0) + len(np.unique(full)) - len(np.unique(small))
                tifffile.imwrite(output / folder / sub / name, small, photometric='minisblack')
        for pairs in sorted(p for p in source.iterdir() if p.is_dir() and p.name.startswith('assignments')):
            shutil.copytree(pairs, output / folder / pairs.name, dirs_exist_ok=True)
        if (source / 'meta.json').is_file():
            meta = json.loads((source / 'meta.json').read_text())
            for entry in meta:
                extent = entry.get('cells_median_extent_p')
                if extent:
                    entry['cells_median_extent_p'] = [extent[0]] + [v / factor for v in extent[1:]]
            (output / folder / 'meta.json').write_text(json.dumps(meta, indent=2))
        print(f'{folder}: {len(names)} file(s)')
    regions = dataset / 'train_regions' / 'regions.json'
    if regions.is_file():
        entries = json.loads(regions.read_text())
        for entry in entries.values():
            entry['train_blocks'] = [[v // factor for v in block] for block in entry['train_blocks']]
            if 'block_size' in entry:
                entry['block_size'] = [v // factor for v in entry['block_size']]
            if 'shape' in entry:
                entry['shape'] = [entry['shape'][0]] + [v // factor for v in entry['shape'][1:]]
        (output / 'train_regions' / 'regions.json').write_text(json.dumps(entries, indent=2))
    for name in ('split_manifest.json', 'split_info.json'):
        if (dataset / name).is_file():
            data = json.loads((dataset / name).read_text())
            data = scale_manifest(data, factor) if name == 'split_manifest.json' else dict(data, xy_downscale=factor * int(data.get('xy_downscale', 1)))
            data['downscaled_from'] = data.get('downscaled_from') or str(dataset.resolve())     # the native dataset
            (output / name).write_text(json.dumps(data, indent=2))
    print('labels lost by downscaling:', lost, '->', output)
    return lost


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--factor', type=int, default=2)
    args = parser.parse_args()
    downscale_split(args.dataset, args.output, args.factor)


if __name__ == '__main__':
    main()
