"""How the annotated children sit in their parents, per dataset. Ground truth only, CPU. Per annotated pair (child -> parent of the stored assignments, else the
largest overlap, as score_hierarchy.ground_truth):
  inside       share of the child's pixels / voxels inside its own parent;
  offset_r     |centroid child - centroid parent| / equivalent radius of the parent, in physical units (2D: pixels;
               KiTS: 3 x 0.78 x 0.78 mm); None where the voxel size is not known (spheroid, synthetic);
  offset_w     the same offset measured in the parent's own whitened coordinates, sqrt(d' S^-1 d) / k with S the
               covariance of the parent's voxel coordinates and k = 2 (2D) or sqrt(5) (3D): equal to offset_r for a
               round parent, and independent of the voxel anisotropy (so comparable across all datasets).
Per image: outside = share of all child pixels outside every parent (the region L_WBR penalises).
Children lying mostly in an unknown region of their level are left out. Spheroid: the training region (train_regions,
the blocks without their augmented copies) and the val / test blocks.
Usage: python paper/nesting_geometry.py OUTPUT_DIR [WORKERS]   (from the folder that holds data/)"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

D = Path('data')                                   # the datasets as dataset_prep writes them (README)
DATASETS = {   # name: (folder, splits, pairs (child, parent), voxel size or None (2D: 1 px))
    'TissueNet': (D / 'TissueNet/cytonucs', ('train', 'val', 'test'), [('nucleus', 'cytoplasm')], (1.0, 1.0)),
    'HeLaCytoNuc clean': (D / 'HeLaCytoNuc/cytonucs_clean', ('train', 'val', 'test'), [('nucleus', 'cytoplasm')], (1.0, 1.0)),
    'BBBC020': (D / 'BBBC020/cytonucs_native', ('train', 'val', 'test'), [('nucleus', 'cytoplasm')], (1.0, 1.0)),
    'Spheroid (half XY)': (D / 'Cytonucs_Emese_Hela/patches_split_hela_xy2', ('train_regions', 'val', 'test'),
                           [('nucleus', 'cytoplasm')], None),
    'Synthetic hela_v1': (D / 'Cytonucs_Synthetic/hela_v1', ('train', 'val', 'test'),
                          [('nucleolus', 'nucleus'), ('nucleus', 'cytoplasm')], None),
    'KiTS23 (masses in kidneys)': (D / 'KiTS23/cytonucs_3x078', ('train', 'val', 'test'), [('mass', 'kidney')], (3.0, 0.78, 0.78)),
}


def pairs_folder(inner, outer):                     # as score_hierarchy.pairs_folder
    return 'assignments' if (inner, outer) == ('nucleus', 'cytoplasm') else f'assignments_{inner}_{outer}'


def links_of(folder, name, child, parent, children, parents):
    stored = folder / pairs_folder(child, parent) / (Path(name).stem + '.json')
    if stored.is_file():
        return {int(k): int(v) for k, v in json.loads(stored.read_text()).items()}
    both = (children > 0) & (parents > 0)
    links = {}
    if both.any():
        width = int(parents.max()) + 1
        keys, counts = np.unique(children[both].astype(np.int64) * width + parents[both], return_counts=True)
        for key, count in sorted(zip(keys.tolist(), counts.tolist()), key=lambda kc: kc[1]):
            links[int(key // width)] = int(key % width)
    return links


def unknown(folder, level, name):
    path = folder / f'{level}_unknown' / name
    return tifffile.imread(path) > 0 if path.is_file() else None


def measure(task):
    dataset, split, name = task
    folder, _, pairs, spacing = DATASETS[dataset]
    folder = folder / split
    out = []
    labels = {}
    for child, parent in pairs:
        for level in (child, parent):
            if level not in labels:
                labels[level] = np.asarray(tifffile.imread(folder / f'{level}_masks' / name))
        children, parents = labels[child], labels[parent]
        if children.shape != parents.shape:
            raise ValueError(f'{dataset} {split} {name}: {child} {children.shape} vs {parent} {parents.shape}')
        ndim = children.ndim
        ignore = unknown(folder, child, name)
        links = links_of(folder, name, child, parent, children, parents)
        child_boxes, parent_boxes = ndi.find_objects(children), ndi.find_objects(parents)
        fg_child = children > 0
        if ignore is not None:
            fg_child &= ~ignore
        outside = float((fg_child & (parents == 0)).sum() / max(fg_child.sum(), 1)) if fg_child.any() else None
        records = []
        k = 2.0 if ndim == 2 else np.sqrt(5.0)
        for c, p in links.items():
            if c > len(child_boxes) or child_boxes[c - 1] is None or p > len(parent_boxes) or parent_boxes[p - 1] is None:
                continue
            cbox = child_boxes[c - 1]
            cmask = children[cbox] == c
            if ignore is not None and ignore[cbox][cmask].mean() > 0.5:
                continue
            inside = float((parents[cbox][cmask] == p).mean())
            pbox = parent_boxes[p - 1]
            pcoords = np.argwhere(parents[pbox] == p).astype(np.float64) + [s.start for s in pbox]
            ccoords = np.argwhere(cmask).astype(np.float64) + [s.start for s in cbox]
            delta = ccoords.mean(0) - pcoords.mean(0)
            cov = np.cov(pcoords, rowvar=False) if len(pcoords) > ndim else None
            offset_w = None
            if cov is not None and np.linalg.matrix_rank(cov) == ndim:
                offset_w = float(np.sqrt(delta @ np.linalg.solve(cov, delta)) / k)
            offset_r = None
            if spacing is not None:
                sp = np.asarray(spacing, np.float64)
                size = len(pcoords) * float(np.prod(sp))
                radius = np.sqrt(size / np.pi) if ndim == 2 else np.cbrt(3 * size / (4 * np.pi))
                offset_r = float(np.linalg.norm(delta * sp) / radius)
            records.append((inside, offset_r, offset_w))
        out.append(dict(pair=f'{child}-{parent}', records=records, outside=outside,
                        children=int(len(np.unique(children[fg_child])) ), paired=len(records)))
    return dataset, split, name, out


def summary(values):
    values = np.asarray([v for v in values if v is not None], np.float64)
    if not len(values):
        return None
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return dict(n=int(len(values)), median=float(med), q1=float(q1), q3=float(q3), mean=float(values.mean()))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 else 2)
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    tasks = []
    for dataset, (folder, splits, _, _) in DATASETS.items():
        for split in splits:
            names = sorted(p.name for p in (folder / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
            tasks += [(dataset, split, n) for n in names]
    print(len(tasks), 'images / volumes', flush=True)
    raw = {}
    with Pool(workers) as pool:
        for i, (dataset, split, name, res) in enumerate(pool.imap_unordered(measure, tasks, chunksize=4)):
            raw.setdefault(dataset, {}).setdefault(split, {})[name] = res
            if (i + 1) % 500 == 0:
                print(i + 1, 'done', flush=True)
    (out / 'nesting_geometry_raw.json').write_text(json.dumps(raw))
    table = {}
    for dataset, splits in raw.items():
        for pair in [f'{c}-{p}' for c, p in DATASETS[dataset][2]]:
            for split in list(splits) + ['all']:
                images = [r for s in (splits if split == 'all' else [split]) for res in splits[s].values() for r in res if r['pair'] == pair]
                recs = [rec for r in images for rec in r['records']]
                table[f'{dataset} | {pair} | {split}'] = dict(
                    images=len(images), pairs=len(recs),
                    inside=summary([r[0] for r in recs]),
                    fully_inside=float(np.mean([r[0] >= 0.99 for r in recs])) if recs else None,
                    offset_r=summary([r[1] for r in recs]), offset_w=summary([r[2] for r in recs]),
                    outside_per_image=summary([r['outside'] for r in images]))
    (out / 'nesting_geometry.json').write_text(json.dumps(table, indent=1))
    f = lambda s, d=3: '—' if s is None else f"{s['median']:.{d}f} [{s['q1']:.{d}f}, {s['q3']:.{d}f}]"
    lines = ['| dataset | pair | split | images | pairs | child inside parent | ≥ 99 % inside | offset / r_eq | offset, whitened | child outside every parent (per image) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for key, t in table.items():
        dataset, pair, split = key.split(' | ')
        fully = '—' if t['fully_inside'] is None else f"{t['fully_inside']:.3f}"
        lines.append(f"| {dataset} | {pair} | {split} | {t['images']} | {t['pairs']} | {f(t['inside'])} | {fully} | "
                     f"{f(t['offset_r'])} | {f(t['offset_w'])} | {f(t['outside_per_image'])} |")
    (out / 'nesting_geometry.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
