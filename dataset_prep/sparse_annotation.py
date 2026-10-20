"""Copy of a CytoNucs dataset whose inner level is annotated only inside k cells of the training data,
for training with training.unknown_label (sparse annotation of the inner level).

The objects of the inner level (nuclei) assigned to k randomly chosen train cells are kept. In the
inner masks, the voxels of every other cell carry the unknown label, and the inner objects of other
cells that stick out of every cell become background, as with an annotator who marks the cells he did
not annotate as unknown. The outer level (cells) stays fully annotated; val and test are unchanged.

Cells are chosen among those of train_regions/ (whole volumes; cells whose box centre lies in a train
block and that have at least one assigned inner object); the same label IDs are applied to the train
tiles. Unchanged files are hard links to the source dataset; changed ones are new files.

Usage:
  python dataset_prep/sparse_annotation.py --dataset DIR --output DIR_k5 --cells 5 [--seed 0]
      [--inner nucleus] [--outer cytoplasm] [--unknown-label 65535]
Writes <output>/sparse_annotation.json with the chosen cells.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import find_objects

UNKNOWN = 65535


def train_cells(outer, blocks, pairs):
    """Labels of the outer objects whose box centre lies in a train block and that own an inner object."""
    region = np.zeros(outer.shape[-2:], bool)
    for y0, y1, x0, x1 in blocks:
        region[y0:y1, x0:x1] = True
    owners = set(pairs.values())
    cells = []
    for label, box in enumerate(find_objects(outer), 1):
        if box is not None and label in owners and region[(box[-2].start + box[-2].stop) // 2, (box[-1].start + box[-1].stop) // 2]:
            cells.append(label)
    return cells


def sparse_inner(inner, outer, cells, kept, unknown_label):
    """Inner labels with the objects in `kept` only; voxels of other outer objects -> unknown_label."""
    keep = np.isin(inner, sorted(kept))
    out = np.where(keep, inner, 0).astype(inner.dtype)
    out[(outer > 0) & ~np.isin(outer, sorted(cells)) & ~keep] = unknown_label
    return out


def replace(path, write):
    """Write a new file where the copy holds a hard link: the source file stays untouched."""
    path.unlink()
    write(path)


def sparse_annotation(dataset, output, cells, seed=0, inner='nucleus', outer='cytoplasm', unknown_label=UNKNOWN):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    regions = json.loads((dataset / 'train_regions' / 'regions.json').read_text())
    pairs_dir = 'assignments' if (inner, outer) == ('nucleus', 'cytoplasm') else f'assignments_{inner}_{outer}'
    volumes, candidates = {}, []
    for name, entry in regions.items():
        stem = Path(name).stem
        inner_labels = tifffile.imread(dataset / 'train_regions' / f'{inner}_masks' / name)
        outer_labels = tifffile.imread(dataset / 'train_regions' / f'{outer}_masks' / name)
        if max(int(inner_labels.max()), int(outer_labels.max())) >= unknown_label:
            raise ValueError(f'{name}: labels reach the unknown label {unknown_label}')
        pairs = {int(k): int(v) for k, v in json.loads((dataset / 'train_regions' / pairs_dir / f'{stem}.json').read_text()).items()}
        volumes[stem] = dict(name=name, pairs=pairs)
        candidates += [(stem, label) for label in train_cells(outer_labels, entry['train_blocks'], pairs)]
    rng = np.random.default_rng(seed)
    picked = sorted(candidates[i] for i in rng.choice(len(candidates), min(cells, len(candidates)), replace=False))
    for stem, volume in volumes.items():
        volume['cells'] = {label for s, label in picked if s == stem}
        volume['kept'] = {n for n, c in volume['pairs'].items() if c in volume['cells']}

    shutil.copytree(dataset, output, copy_function=os.link)
    summary = dict(source=str(dataset.resolve()), inner=inner, outer=outer, cells=cells, seed=seed,
                   unknown_label=unknown_label, candidates=len(candidates),
                   chosen={stem: sorted(v['cells']) for stem, v in volumes.items()},
                   inner_kept={stem: sorted(v['kept']) for stem, v in volumes.items()}, files={})
    manifest = json.loads((dataset / 'split_manifest.json').read_text())
    tile_volume = {row['filename']: row['volume'] for row in manifest.get('files', [])}
    targets = [('train_regions', volumes[stem]['name'], stem) for stem in volumes]
    targets += [('train', path.name, tile_volume[path.name]) for path in sorted((dataset / 'train' / f'{inner}_masks').iterdir())]
    for folder, name, stem in targets:
        volume = volumes[stem]
        inner_labels = tifffile.imread(dataset / folder / f'{inner}_masks' / name)
        outer_labels = tifffile.imread(dataset / folder / f'{outer}_masks' / name)
        sparse = sparse_inner(inner_labels, outer_labels, volume['cells'], volume['kept'], unknown_label)
        replace(output / folder / f'{inner}_masks' / name, lambda p: tifffile.imwrite(p, sparse, photometric='minisblack'))
        pairs_file = output / folder / pairs_dir / (Path(name).stem + '.json')
        if pairs_file.is_file():
            pairs = {k: v for k, v in json.loads(pairs_file.read_text()).items() if int(k) in volume['kept']}
            replace(pairs_file, lambda p: p.write_text(json.dumps(pairs, indent=2)))
        summary['files'][f'{folder}/{name}'] = dict(
            inner_objects=int(np.count_nonzero(np.unique(sparse[sparse != unknown_label]))),
            unknown_fraction=float(np.mean(sparse == unknown_label)))
    (output / 'sparse_annotation.json').write_text(json.dumps(summary, indent=1))
    region_files = [v for k, v in summary['files'].items() if k.startswith('train_regions/')]
    print(f'{len(picked)} of {len(candidates)} train cells keep their {inner} annotation: '
          f'{sum(len(v["kept"]) for v in volumes.values())} {inner} object(s); unknown fraction of the volumes '
          f'{np.mean([v["unknown_fraction"] for v in region_files]):.3f} -> {output}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cells', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inner', default='nucleus')
    parser.add_argument('--outer', default='cytoplasm')
    parser.add_argument('--unknown-label', type=int, default=UNKNOWN)
    args = parser.parse_args()
    sparse_annotation(args.dataset, args.output, args.cells, args.seed, args.inner, args.outer, args.unknown_label)


if __name__ == '__main__':
    main()
