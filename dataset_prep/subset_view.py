"""Hard-link view of a dataset with fewer files in some splits (no extra disk space), e.g. checkpoint selection on
part of a large validation split, or a small copy for a smoke test.

Every file of every split folder (images, masks, assignments, ...) is linked when its stem belongs to the kept
files of that split: the first N image names in name order for --keep SPLIT:N, all of them otherwise. Files at
the top of the dataset (split_info.json, ...) are linked as they are.

Usage:
  python dataset_prep/subset_view.py --dataset DIR --output DIR_view --keep val:16 [--keep train:3 ...]
"""
import argparse
import os
from pathlib import Path


def subset_view(dataset, output, keep):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    counts = {}
    for entry in sorted(dataset.iterdir()):
        if entry.is_file():
            os.link(entry, output / entry.name)
            continue
        names = sorted(p.stem for p in (entry / 'images').iterdir()) if (entry / 'images').is_dir() else None
        stems = set(names[:keep[entry.name]] if names is not None and entry.name in keep else names or [])
        for folder in sorted(p for p in entry.iterdir() if p.is_dir()):
            (output / entry.name / folder.name).mkdir(parents=True)
            for path in folder.iterdir():
                if names is None or path.stem in stems:
                    os.link(path, output / entry.name / folder.name / path.name)
        for path in (p for p in entry.iterdir() if p.is_file()):
            os.link(path, output / entry.name / path.name)
        counts[entry.name] = len(stems) if names is not None else None
    print({split: n for split, n in counts.items() if n is not None})
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--keep', action='append', default=[], help='SPLIT:N first files of SPLIT in name order')
    args = parser.parse_args()
    subset_view(args.dataset, args.output, {k.split(':')[0]: int(k.split(':')[1]) for k in args.keep})


if __name__ == '__main__':
    main()
