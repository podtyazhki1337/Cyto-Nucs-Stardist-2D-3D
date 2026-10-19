"""BBBC020 (murine bone-marrow macrophages, 2D, DAPI + CD11b) as a two-level CytoNucs dataset: cells with
their nuclei.

The collection gives one filled mask per cell (*_c1_N.TIF) and per nucleus (*_c5_N.TIF) for 20 of its 25
images; the nucleus set is known to be incomplete. Here:
  - images: the merged colour image (c1 + c5), 3 channels, downscaled by --factor (mean of f x f pixels);
  - labels: one label image per level; a pixel claimed by two objects (overlapping cells) goes to the smaller
    one; labels are downscaled by the majority of f x f pixels;
  - a cell without any annotated nucleus has unknown nucleus annotation: in train its pixels carry the unknown
    label 65535 in the nucleus masks (training.unknown_label); in val and test they are written to
    nucleus_unknown/<name>.tif (score with the ignore region, e.g. score_hierarchy.py);
  - nucleus -> cell pairs by the largest overlap (assignments/<stem>.json);
  - images split at random into train / val / test (--counts, --seed).
Licence: CC BY-NC-SA 3.0 (Roland Lang); cite Ljosa et al., Nature Methods 2012.

Usage:
  python dataset_prep/bbbc020_to_cytonucs.py --raw /storage02/miroslavm/data/BBBC020/raw \\
      --output /storage02/miroslavm/data/BBBC020/cytonucs_xy2 [--factor 2] [--counts 12 4 4] [--seed 0]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downscale_split import downscale_image, downscale_labels                     # noqa: E402

UNKNOWN = 65535


def objects(folder, image, channel):
    """Filled masks of one image, largest first."""
    pattern = re.compile(re.escape(image) + rf'_{channel}_(\d+)\.tif$', re.IGNORECASE)
    # Empty (0-byte) mask files occur in the nucleus set of the collection: they are skipped.
    masks = [tifffile.imread(p) > 0 for p in sorted(folder.iterdir()) if pattern.match(p.name) and p.stat().st_size]
    return sorted(masks, key=lambda m: -int(m.sum()))


def label_image(masks, shape):
    """Labels 1..n; a pixel of several objects goes to the smallest (painted last)."""
    labels = np.zeros(shape, np.uint16)
    for i, mask in enumerate(masks, 1):
        labels[mask] = i
    return labels


def convert(raw, output, factor=2, counts=(12, 4, 4), seed=0):
    raw, output = Path(raw), Path(output)
    cells_dir, nuclei_dir = raw / 'BBC020_v1_outlines_cells', raw / 'BBC020_v1_outlines_nuclei'
    names = lambda folder, c: {re.sub(rf'_{c}_\d+\.tif$', '', p.name, flags=re.IGNORECASE)
                               for p in folder.iterdir() if p.suffix.lower() == '.tif'}
    images = sorted(names(cells_dir, 'c1') & names(nuclei_dir, 'c5'))
    order = np.random.default_rng(seed).permutation(len(images))
    split_of = {images[i]: ('train' if k < counts[0] else 'val' if k < counts[0] + counts[1] else 'test') for k, i in enumerate(order)}
    summary = dict(factor=factor, seed=seed, images={})
    for image in images:
        split = split_of[image]
        rgb = tifffile.imread(raw / 'BBBC020_v1_images' / image / f'{image}_(c1+c5).TIF')
        cells = label_image(objects(cells_dir, image, 'c1'), rgb.shape[:2])
        nuclei = label_image(objects(nuclei_dir, image, 'c5'), rgb.shape[:2])
        pairs = {}
        for n in range(1, int(nuclei.max()) + 1):
            under = cells[(nuclei == n) & (cells > 0)]
            if under.size:
                pairs[n] = int(np.bincount(under).argmax())
        without = sorted(set(range(1, int(cells.max()) + 1)) - set(pairs.values()))
        unknown = np.isin(cells, without) & (nuclei == 0)
        name = re.sub(r'[^A-Za-z0-9_-]+', '_', image) + '.tif'
        if factor > 1:
            small = downscale_image(rgb[None], factor)[0]
            cells_s, nuclei_s = downscale_labels(cells, factor), downscale_labels(nuclei, factor)
            unknown_s = downscale_labels(unknown.astype(np.uint8), factor) > 0
        else:                                                      # native resolution
            small, cells_s, nuclei_s, unknown_s = rgb, cells, nuclei, unknown
        for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments') + (('nucleus_unknown',) if split != 'train' else ()):
            (output / split / sub).mkdir(parents=True, exist_ok=True)
        if split == 'train':
            nuclei_s = np.where(unknown_s, UNKNOWN, nuclei_s).astype(np.uint16)
        else:
            tifffile.imwrite(output / split / 'nucleus_unknown' / name, unknown_s.astype(np.uint8), photometric='minisblack')
        tifffile.imwrite(output / split / 'images' / name, small)
        tifffile.imwrite(output / split / 'nucleus_masks' / name, nuclei_s, photometric='minisblack')
        tifffile.imwrite(output / split / 'cytoplasm_masks' / name, cells_s, photometric='minisblack')
        present = set(np.unique(nuclei_s).tolist())
        (output / split / 'assignments' / (Path(name).stem + '.json')).write_text(
            json.dumps({str(n): c for n, c in pairs.items() if n in present}))
        summary['images'][name] = dict(source=image, split=split, cells=int(cells.max()), nuclei=int(nuclei.max()),
                                       cells_without_nucleus=len(without), shape=list(small.shape))
        print(f'{name} {split}: {int(cells.max())} cells, {int(nuclei.max())} nuclei, {len(without)} cells without annotated nucleus')
    info = {s: sorted(n for n, v in summary['images'].items() if v['split'] == s) for s in ('train', 'val', 'test')}
    (output / 'split_info.json').write_text(json.dumps(dict(info, seed=seed, split='bbbc020_random_images', rescale_factor=1. / factor), indent=1))
    (output / 'bbbc020.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--counts', type=int, nargs=3, default=[12, 4, 4])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    convert(args.raw, args.output, args.factor, args.counts, args.seed)


if __name__ == '__main__':
    main()
