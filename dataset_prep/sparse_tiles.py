"""Copy of a tile dataset (e.g. synthetic data with several levels) whose training annotation is reduced, to
study training with levels that are missing or only partly annotated. Val and test stay complete (hard links).

Three kinds of reduction, applied to the train split, in this order:
  --drop LEVEL:FRACTION     the masks of LEVEL are removed from that fraction of the training files
                            (a file without them counts as not annotating LEVEL: training.partial_annotation);
  --complementary INNER:OUTER  'two datasets': a random half of the training files keeps only the INNER masks,
                            the other half only the OUTER masks (pairs files removed: no file has both levels);
  --sparse INNER:OUTER:KEEP INNER objects stay annotated only inside a random KEEP fraction of the OUTER
                            objects of each file; inside every other OUTER object the INNER masks carry the
                            unknown label (training.unknown_label), and so do INNER objects outside every
                            OUTER object. Applied in the order given; an object whose parent became unknown
                            in an earlier --sparse is unknown too.
Children of an outer object come from assignments (nucleus -> cytoplasm) or assignments_<inner>_<outer>,
else from the largest overlap.

Usage:
  python dataset_prep/sparse_tiles.py --dataset DIR --output DIR_v --sparse nucleus:cytoplasm:0.25 \\
      --sparse nucleolus:nucleus:0.25 [--drop nucleolus:0.5] [--complementary nucleus:cytoplasm] [--seed 0]
      [--unknown-label 65535]
Writes <output>/reduced_annotation.json.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile

UNKNOWN = 65535


def pairs_folder(inner, outer):
    return 'assignments' if (inner, outer) == ('nucleus', 'cytoplasm') else f'assignments_{inner}_{outer}'


def children_of(inner, outer, stored):
    """inner id -> outer id (stored pairs if given, else the largest overlap)."""
    if stored is not None:
        return {int(k): int(v) for k, v in stored.items()}
    links = {}
    for child in np.unique(inner):
        if child == 0 or child == UNKNOWN:
            continue
        under = outer[(inner == child) & (outer > 0) & (outer != UNKNOWN)]
        if under.size:
            values, counts = np.unique(under, return_counts=True)
            links[int(child)] = int(values[np.argmax(counts)])
    return links


def replace(path, write):
    if path.exists():
        path.unlink()                                   # a hard link to the source: write a new file
    write(path)


def reduce_annotation(dataset, output, drops=(), sparse=(), seed=0, unknown_label=UNKNOWN, complementary=None):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(dataset, output, copy_function=os.link)
    rng = np.random.default_rng(seed)
    train = output / 'train'
    names = sorted(p.name for p in (train / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    summary = dict(source=str(dataset.resolve()), seed=seed, drops={}, sparse=[], unknown_label=unknown_label)
    for level, fraction in drops:
        chosen = sorted(rng.choice(len(names), int(round(fraction * len(names))), replace=False).tolist())
        for i in chosen:
            path = train / f'{level}_masks' / names[i]
            if path.exists():
                path.unlink()
        summary['drops'][level] = [names[i] for i in chosen]
    if complementary:
        inner, outer = complementary
        order = rng.permutation(len(names))
        half = len(names) // 2
        for k, i in enumerate(order):
            gone = outer if k < half else inner                  # first half: only the inner level, then only the outer
            for path in (train / f'{gone}_masks' / names[i], train / pairs_folder(inner, outer) / (Path(names[i]).stem + '.json')):
                if path.exists():
                    path.unlink()
        summary['complementary'] = {f'{inner}_only': sorted(names[i] for i in order[:half]),
                                    f'{outer}_only': sorted(names[i] for i in order[half:])}
    for inner, outer, keep in sparse:
        kept_total = parents_total = 0
        for name in names:
            inner_path, outer_path = train / f'{inner}_masks' / name, train / f'{outer}_masks' / name
            if not inner_path.exists() or not outer_path.exists():
                continue
            inner_labels, outer_labels = tifffile.imread(inner_path), tifffile.imread(outer_path)
            stored = train / pairs_folder(inner, outer) / (Path(name).stem + '.json')
            links = children_of(inner_labels, outer_labels, json.loads(stored.read_text()) if stored.is_file() else None)
            parents = sorted(int(v) for v in np.unique(outer_labels) if v not in (0, unknown_label))
            chosen = set(rng.choice(parents, int(round(keep * len(parents))), replace=False).tolist()) if parents else set()
            kept_children = {c for c, p in links.items() if p in chosen}
            # A type that holds the unknown label (uint8 masks would wrap 65535 to 255, a valid object id).
            out = inner_labels.astype(np.uint32 if unknown_label > 65535 else np.promote_types(inner_labels.dtype, np.uint16))
            hidden = ((outer_labels > 0) & ~np.isin(outer_labels, sorted(chosen))) | (outer_labels == unknown_label)
            loose = (inner_labels > 0) & (inner_labels != unknown_label) & ~np.isin(inner_labels, sorted(kept_children))
            out[(hidden | loose) & ~np.isin(inner_labels, sorted(kept_children))] = unknown_label
            replace(inner_path, lambda p, a=out: tifffile.imwrite(p, a, photometric='minisblack'))
            if stored.is_file():
                kept_links = {str(c): p for c, p in links.items() if c in kept_children}
                replace(stored, lambda p, k=kept_links: p.write_text(json.dumps(k)))
            kept_total += len(kept_children)
            parents_total += len(parents)
        summary['sparse'].append(dict(inner=inner, outer=outer, keep=keep, kept_inner_objects=kept_total, parents=parents_total))
    (output / 'reduced_annotation.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: (v if k not in ('drops', 'complementary') else {l: len(f) for l, f in v.items()}) for k, v in summary.items()}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--drop', action='append', default=[], help='LEVEL:FRACTION of training files without LEVEL')
    parser.add_argument('--sparse', action='append', default=[], help='INNER:OUTER:KEEP fraction of OUTER objects with annotated INNER')
    parser.add_argument('--complementary', help='INNER:OUTER half of the training files with only INNER, half with only OUTER')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--unknown-label', type=int, default=UNKNOWN)
    args = parser.parse_args()
    drops = [(d.split(':')[0], float(d.split(':')[1])) for d in args.drop]
    sparse = [(s.split(':')[0], s.split(':')[1], float(s.split(':')[2])) for s in args.sparse]
    reduce_annotation(args.dataset, args.output, drops, sparse, args.seed, args.unknown_label,
                      tuple(args.complementary.split(':')) if args.complementary else None)


if __name__ == '__main__':
    main()
