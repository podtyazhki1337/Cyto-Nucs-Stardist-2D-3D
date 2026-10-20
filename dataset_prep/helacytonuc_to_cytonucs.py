"""HeLaCytoNuc (De et al. 2024, doi 10.14278/rodare.3001, CC BY 4.0; the dataset of the HydraStarDist paper,
arXiv 2504.12078) as a two-level CytoNucs dataset: cells (whole-cell 'cytoplasm' masks) with their nuclei.

The published preprocessed archives are used as they are: RGB 520 x 696 uint8 images (cytoplasm red, nuclei
blue), their 70 / 20 / 10 split (1873 / 535 / 268 images). Train and validation masks come from CellProfiler,
the test masks were drawn by hand; the CellProfiler masks of the test images are kept as an extra split.
  - images and masks are hard links to the extracted archives (no copy);
  - nucleus -> cell pairs: the cell with the largest overlap when at least half of the nucleus lies in cells;
  - splits: train; val = --val-count of the validation images (random, --seed: checkpoint selection and the
    StarDist / HydraStarDist thresholds; all of them in val_all); test (manual masks); test_cellprofiler.

Usage:
  python dataset_prep/helacytonuc_to_cytonucs.py --extracted /storage02/miroslavm/data/HeLaCytoNuc/extracted \\
      --output /storage02/miroslavm/data/HeLaCytoNuc/cytonucs [--val-count 128] [--seed 0]
Cite De, Urbanski, Thangamani, Wyrzykowska, Yakimovich, HeLaCytoNuc, RODARE 2024.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile


def nucleus_cells(nuclei, cells):
    """nucleus -> cell with the largest overlap, for nuclei lying at least half inside cells."""
    nuclei, cells = np.asarray(nuclei, np.int64), np.asarray(cells, np.int64)
    sizes = np.bincount(nuclei.ravel())
    both = (nuclei > 0) & (cells > 0)
    width = int(cells.max()) + 1
    keys, counts = np.unique(nuclei[both] * width + cells[both], return_counts=True)
    best = {}
    for key, count in zip(keys.tolist(), counts.tolist()):
        nucleus, cell = divmod(key, width)
        if count > best.get(nucleus, (0, 0))[0]:
            best[nucleus] = (count, cell)
    inside = np.bincount(nuclei[both].ravel(), minlength=len(sizes))
    return {int(n): int(cell) for n, (_, cell) in best.items() if inside[n] >= 0.5 * sizes[n]}


def link_split(images, masks, output, names):
    """Hard links of images and masks of `names` into output/{images,nucleus_masks,cytoplasm_masks} and the pairs."""
    for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
        (output / sub).mkdir(parents=True, exist_ok=True)
    counts = dict(images=0, nuclei=0, cells=0, pairs=0)
    for name in names:
        os.link(images / name, output / 'images' / name)
        os.link(masks / 'nuclei_masks' / name, output / 'nucleus_masks' / name)
        os.link(masks / 'cytoplasm_masks' / name, output / 'cytoplasm_masks' / name)
        nuclei, cells = tifffile.imread(masks / 'nuclei_masks' / name), tifffile.imread(masks / 'cytoplasm_masks' / name)
        pairs = nucleus_cells(nuclei, cells)
        (output / 'assignments' / (Path(name).stem + '.json')).write_text(json.dumps(pairs))
        counts['images'] += 1
        counts['nuclei'] += len(np.unique(nuclei)) - 1
        counts['cells'] += len(np.unique(cells)) - 1
        counts['pairs'] += len(pairs)
    return counts


def convert(extracted, output, val_count=128, seed=0):
    extracted, output = Path(extracted), Path(output)
    if output.exists():
        raise FileExistsError(output)
    listing = lambda folder: sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    validation = listing(extracted / 'validation')
    chosen = sorted(validation[i] for i in np.random.default_rng(seed).choice(len(validation), val_count, replace=False))
    plan = dict(train=(extracted / 'train', extracted / 'train', listing(extracted / 'train')),
                val=(extracted / 'validation', extracted / 'validation', chosen),
                val_all=(extracted / 'validation', extracted / 'validation', validation),
                test=(extracted / 'test', extracted / 'test', listing(extracted / 'test')),
                test_cellprofiler=(extracted / 'test', extracted / 'test_cellprofiler_masks', listing(extracted / 'test')))
    summary = dict(source=str(extracted), val_count=val_count, seed=seed, splits={})
    for split, (images, masks, names) in plan.items():
        summary['splits'][split] = link_split(images / 'images', masks, output / split, names)
        print(split, summary['splits'][split], flush=True)
    (output / 'split_info.json').write_text(json.dumps(dict({s: plan[s][2] for s in plan}, seed=seed, split='helacytonuc_published',
                                                            rescale_factor=1.0, levels=['nucleus', 'cytoplasm']), indent=1))
    (output / 'helacytonuc.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--extracted', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--val-count', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    convert(args.extracted, args.output, args.val_count, args.seed)


if __name__ == '__main__':
    main()
