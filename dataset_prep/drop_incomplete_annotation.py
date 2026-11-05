"""Hard-link view of a dataset without the images whose inner-level annotation is incomplete: fewer than --min-share of
the nuclei visible in the nucleus channel are annotated (no extra disk space; the other splits are linked whole).

HeLaCytoNuc's train and validation masks come from CellProfiler and miss most cells on part of the images (val
2272.tif: 2 objects of about 60 cells); the masks are the published files (MD5-checked archives). Visible nuclei are
counted on the nucleus channel: per-image percentile normalisation, Gaussian sigma 1, local threshold against a 61 px
neighbourhood, components of 0.3-4 x the median annotated nucleus area of the split. On the hand-drawn HeLa test this
flags no image at a share of 0.5, so it does not overcount where the annotation is complete.

Writes excluded.json (per split: name, annotated nuclei, visible nuclei) at the top of the output.

Usage:
  python dataset_prep/drop_incomplete_annotation.py --dataset DIR --output DIR_clean --splits train val val_all \\
      [--channel 2] [--min-share 0.5]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile


def visible_nuclei(channel, median_area):
    """Nucleus-sized bright components of one image channel."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_local
    x = channel.astype(np.float32)
    x = ndi.gaussian_filter(np.clip(x / max(float(np.percentile(x, 99.8)), 1.0), 0, 1), 1)
    sizes = np.bincount(ndi.label((x > threshold_local(x, 61, offset=-0.05)) & (x > 0.1))[0].ravel())[1:]
    return int(np.sum((sizes >= 0.3 * median_area) & (sizes <= 4 * median_area)))


def incomplete(split_dir, channel, min_share):
    """[(name, annotated, visible)] of the images of a split with fewer than min_share of the visible nuclei annotated
    (and at least 5 visible; an empty annotation with 3 or more visible counts too)."""
    names = sorted(p.name for p in (split_dir / 'images').iterdir())
    masks = lambda name: tifffile.imread(split_dir / 'nucleus_masks' / name)
    areas = np.concatenate([np.bincount(masks(n).ravel())[1:] for n in names[::10]])
    median_area = float(np.median(areas[areas > 0]))
    out = []
    for name in names:
        annotated = len(np.unique(masks(name))) - 1
        visible = visible_nuclei(tifffile.imread(split_dir / 'images' / name)[..., channel], median_area)
        if (visible >= 5 and annotated < min_share * visible) or (annotated == 0 and visible >= 3):
            out.append((name, int(annotated), visible))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--splits', nargs='+', required=True)
    parser.add_argument('--channel', type=int, default=2)
    parser.add_argument('--min-share', type=float, default=0.5)
    args = parser.parse_args()
    dataset, output = Path(args.dataset), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    excluded = {split: incomplete(dataset / split, args.channel, args.min_share) for split in args.splits}
    output.mkdir(parents=True)
    for entry in sorted(dataset.iterdir()):
        if entry.is_file():
            os.link(entry, output / entry.name)
            continue
        drop = {Path(name).stem for name, _, _ in excluded.get(entry.name, [])}
        for folder in sorted(p for p in entry.iterdir() if p.is_dir()):
            (output / entry.name / folder.name).mkdir(parents=True)
            for path in folder.iterdir():
                if path.stem not in drop:
                    os.link(path, output / entry.name / folder.name / path.name)
        for path in (p for p in entry.iterdir() if p.is_file()):
            os.link(path, output / entry.name / path.name)
    (output / 'excluded.json').write_text(json.dumps(dict(
        min_share=args.min_share, channel=args.channel,
        splits={split: [dict(name=n, annotated=a, visible=v) for n, a, v in rows] for split, rows in excluded.items()}),
        indent=1))
    for split, rows in excluded.items():
        print(f'{split}: {len(rows)} of {len(list((dataset / split / "images").iterdir()))} images dropped')


if __name__ == '__main__':
    main()
