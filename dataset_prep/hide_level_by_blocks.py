"""Copy of a CytoNucs dataset in which some levels are not annotated in some train blocks, for training with
training.unknown_label: e.g. nuclei annotated in one part of the volume and cells in another, as if the
training data came from two datasets with one annotated level each and no annotated hierarchy.

In the masks of a hidden level, every voxel of the given blocks (full Z) carries the unknown label, in the
whole volume of train_regions and in the train tiles (by their position in the volume; an offline flipped or
rotated tile copy '_aNN', NN > 0, that touches a hidden block loses the level entirely); assignments that
lose their nucleus or cell are dropped. Val and test are unchanged (hard links), as is everything else.

Usage:
  python dataset_prep/hide_level_by_blocks.py --dataset DIR --output DIR_two --hide cytoplasm:0,1,2,3,5,7 \\
      --hide nucleus:8,9,10,11,12,15 [--unknown-label 65535]
Block indices are those of the native split_manifest.json (copies by downscale_split.py are handled).
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repository root
from score_native_resolution import native_dataset

MASKS = dict(nucleus='nucleus_masks', cytoplasm='cytoplasm_masks')


def hidden_area(shape_yx, blocks, origin=(0, 0)):
    """Boolean YX map of the given blocks (y0, y1, x0, x1, volume coordinates) on an array placed at origin."""
    area = np.zeros(shape_yx, bool)
    for y0, y1, x0, x1 in blocks:
        ya, yb = max(y0 - origin[0], 0), min(y1 - origin[0], shape_yx[0])
        xa, xb = max(x0 - origin[1], 0), min(x1 - origin[1], shape_yx[1])
        if ya < yb and xa < xb:
            area[ya:yb, xa:xb] = True
    return area


def replace(path, write):
    path.unlink()                                   # a hard link to the source: write a new file
    write(path)


def hide_levels(dataset, output, hide, unknown_label=65535):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((dataset / 'split_manifest.json').read_text())
    factor = int(manifest.get('xy_downscale', 1))
    native = json.loads((native_dataset(dataset, manifest) / 'split_manifest.json').read_text()) if factor > 1 else manifest
    by_index = {b['index']: b for v in native['volumes'] for b in v['blocks']}
    blocks = {}
    for level, indices in hide.items():
        for i in indices:
            if by_index[i]['split'] != 'train':
                raise ValueError(f'block {i} is {by_index[i]["split"]}, only train blocks can be hidden')
        blocks[level] = [tuple(by_index[i][k] // factor for k in ('y0', 'y1', 'x0', 'x1')) for i in indices]
    shutil.copytree(dataset, output, copy_function=os.link)
    summary = dict(source=str(dataset.resolve()), hide={k: list(v) for k, v in hide.items()}, unknown_label=unknown_label, files={})
    targets = [('train_regions', p.name, (0, 0), 0) for p in sorted((dataset / 'train_regions' / 'images').iterdir())]
    for p in sorted((dataset / 'train' / 'images').iterdir()):
        found = re.search(r'_y(\d+)_x(\d+)_a(\d+)', p.stem)            # tile position, native coordinates
        targets.append(('train', p.name, (int(found.group(1)) // factor, int(found.group(2)) // factor), int(found.group(3))))
    for folder, name, origin, copy in targets:
        removed = {}
        for level, level_blocks in blocks.items():
            path = dataset / folder / MASKS[level] / name
            if not path.is_file():
                continue
            labels = tifffile.imread(path)
            area = hidden_area(labels.shape[-2:], level_blocks, origin)
            if not area.any():
                continue
            if copy:
                area[:] = True          # flipped / rotated tile copy: its position in the volume is not exact
            removed[level] = set(np.unique(labels[:, area]).tolist()) - set(np.unique(labels[:, ~area]).tolist()) - {0}
            out = labels.astype(np.uint32 if unknown_label > 65535 else labels.dtype).copy()
            out[:, area] = unknown_label
            replace(output / folder / MASKS[level] / name, lambda p, a=out: tifffile.imwrite(p, a, photometric='minisblack'))
            summary['files'][f'{folder}/{name}:{level}'] = float(area.mean())
        if removed:
            pairs_file = output / folder / 'assignments' / (Path(name).stem + '.json')
            if pairs_file.is_file():
                pairs = json.loads(pairs_file.read_text())
                kept = {k: v for k, v in pairs.items() if int(k) not in removed.get('nucleus', set())
                        and int(v) not in removed.get('cytoplasm', set())}
                replace(pairs_file, lambda p, k=kept: p.write_text(json.dumps(k, indent=2)))
    (output / 'hidden_levels.json').write_text(json.dumps(summary, indent=1))
    print(f'hidden {summary["hide"]} in {len(summary["files"])} mask file(s) -> {output}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hide', action='append', required=True, help='LEVEL:i,j,k (block indices of the split manifest)')
    parser.add_argument('--unknown-label', type=int, default=65535)
    args = parser.parse_args()
    hide = {}
    for item in args.hide:
        level, indices = item.split(':')
        if level not in MASKS:
            parser.error(f'unknown level {level}')
        hide[level] = [int(i) for i in indices.split(',')]
    hide_levels(args.dataset, args.output, hide, args.unknown_label)


if __name__ == '__main__':
    main()
