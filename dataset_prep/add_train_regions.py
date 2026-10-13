"""Whole source volumes for training crops at any position inside the train blocks of a spatial split.

make_spatial_split_patches.py cuts the train blocks into fixed tiles (512 px by default). A training crop of
such a tile is almost the same picture every time, and most objects of a tile are cut by one of its faces.
This script adds, next to the tiles of an existing split,
  <dataset>/train_regions/{images,nucleus_masks,cytoplasm_masks,assignments}/<volume>.tiff
  <dataset>/train_regions/regions.json     {file: {"train_blocks": [[y0, y1, x0, x1], ...], "block_size": [h, w]}}
With training.train_region_crops the training generator takes its crops from these volumes at any position
and keeps every crop inside the train blocks, so no voxel of a val or test block is ever read for training.
The tiles of train, val and test are not touched: validation, geometry and the train-side metric still use them.

Usage:
  python dataset_prep/add_train_regions.py --dataset DATASET_DIR
The block layout and the source files are read from DATASET_DIR/split_manifest.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_spatial_split_patches import channel_axis, pad_to_minimum_size

FOLDER = 'train_regions'


def train_blocks(volume):
    """[y0, y1, x0, x1] of every train block of one manifest volume."""
    return [[b['y0'], b['y1'], b['x0'], b['x1']] for b in volume['blocks'] if b['split'] == 'train']


def build(dataset, read=None, max_assignment_distance=50.0):
    """Write <dataset>/train_regions from <dataset>/split_manifest.json; read(path) loads a TIFF."""
    from cytonucs_losses import compute_nucleus_cell_assignments
    dataset = Path(dataset)
    manifest = json.loads((dataset / 'split_manifest.json').read_text())
    output = dataset / FOLDER
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty')
    read = read or (lambda path: tifffile.imread(str(path)))
    for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
        (output / sub).mkdir(parents=True, exist_ok=True)
    regions = {}
    for volume in manifest['volumes']:
        blocks = train_blocks(volume)
        if not blocks:
            raise ValueError(f"{volume['name']}: no train block in the manifest")
        source = volume.get('source') or {}
        missing = [k for k in ('image', 'nucleus', 'cytoplasm') if not Path(source.get(k, {}).get('path', '')).is_file()]
        if missing:
            raise FileNotFoundError(f"{volume['name']}: source file(s) not found: {missing}")
        image = read(source['image']['path'])
        nucleus, cytoplasm = (np.squeeze(read(source[k]['path'])) for k in ('nucleus', 'cytoplasm'))
        axis = channel_axis(image.shape, nucleus.shape)
        image = image if axis is None else np.moveaxis(image, axis, -1)
        image, nucleus, cytoplasm = pad_to_minimum_size(image, nucleus, cytoplasm, manifest['patch_size'])
        if list(nucleus.shape) != list(volume['shape']):
            raise ValueError(f"{volume['name']}: the source is {list(nucleus.shape)}, the split was made on {volume['shape']}")
        name = volume['name'] + '.tiff'
        tifffile.imwrite(str(output / 'images' / name), image[..., :3] if image.ndim == 4 else image)
        for sub, mask in (('nucleus_masks', nucleus), ('cytoplasm_masks', cytoplasm)):
            tifffile.imwrite(str(output / sub / name), mask, photometric='minisblack')
        pairs = compute_nucleus_cell_assignments(nucleus, cytoplasm, max_distance=max_assignment_distance)[0]
        (output / 'assignments' / (volume['name'] + '.json')).write_text(json.dumps(pairs, indent=2))
        sizes = {(b[1] - b[0], b[3] - b[2]) for b in blocks}
        regions[name] = dict(train_blocks=blocks, block_size=list(max(sizes)), shape=list(nucleus.shape))
        print(f"{volume['name']}: volume {list(nucleus.shape)}, {len(blocks)} train block(s) of {len(volume['blocks'])}")
    (output / 'regions.json').write_text(json.dumps(regions, indent=2))
    return regions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help='folder written by make_spatial_split_patches.py')
    args = parser.parse_args()
    build(args.dataset)
    print('Written:', Path(args.dataset) / FOLDER)


if __name__ == '__main__':
    main()
