"""The HeLa-MRC-5 spheroid of the paper from its public figshare record (Piccinini and Diosdi, 2024).

Record 10.6084/m9.figshare.24982227 ("2024_Hollandi-4_Co_culture_images_and_masks", GPL 3.0+). The paper uses
acquisition 012:
  channels.zip  channels/HeLa_MRC_DLS Acquire 2_012_Target_z00_ch0{0,1,2,3}.tif   four RGB stacks (Z, Y, X, 3)
  masks2.zip    masks2/HeLa_MRC_DLS Acquire 2_012_Target_z00_ch00_mask.tif       nuclei (instance labels)
                masks2/HeLa_MRC_DLS Acquire 2_012_Target_z00_ch03_mask.tif       whole cells
The four stacks are merged into one RGB volume: each stack is scaled to 0..1 by its own minimum and maximum, the
four are summed (or their maximum taken, --mode max), clipped to 1 and stored as 8 bit, one RGB page per Z slice.
This writes the input of make_spatial_split_patches.py, under the names the paper's split was made from:
  <output>/images/HeLa_MRC_DLS Acquire 2_012_Target_z00_merged_rgb_plain_stack.tif
  <output>/nuc_masks/HeLa_MRC_DLS Acquire 2_012_Target_z00.tif
  <output>/cyto_masks/HeLa_MRC_DLS Acquire 2_012_Target_z00.tif

Usage:
  python dataset_prep/hela_spheroid_from_figshare.py --download data/Cytonucs_Emese_Hela/figshare \\
      --output data/Cytonucs_Emese_Hela
  (--channels-zip / --masks-zip instead of --download for archives downloaded by hand)
The split of the paper follows from it with make_spatial_split_patches.py --blocks hela_spheroid_blocks.json,
add_train_regions.py and downscale_split.py (README.md, Datasets of the paper).
"""
import argparse
import hashlib
import io
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# File id and MD5 of the two archives in version 2 of the record.
ARCHIVES = {
    'channels.zip': ('https://ndownloader.figshare.com/files/44001699', 'e20c108bf38e130220f64034f8964fce'),
    'masks2.zip': ('https://ndownloader.figshare.com/files/71817598', '90649c9aa814329aeba09d5b16def025'),
}
CHANNEL = re.compile(r'_ch0([0-3])\.tiff?$', re.IGNORECASE)


def md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, checksum) in ARCHIVES.items():
        path = folder / name
        if not (path.is_file() and md5(path) == checksum):
            print(f'downloading {name} ({url})', flush=True)
            partial = path.with_suffix('.part')
            with urllib.request.urlopen(url) as response, open(partial, 'wb') as out:
                shutil.copyfileobj(response, out, 1 << 24)
            if md5(partial) != checksum:
                raise SystemExit(f'{name}: MD5 differs from the record ({checksum}); download it again')
            partial.replace(path)
        paths[name] = path
    return paths['channels.zip'], paths['masks2.zip']


def read_tiff(archive, member):
    return tifffile.imread(io.BytesIO(archive.read(member)))


def rgb_stack(image, name):
    """(Z, Y, X, 3) of an RGB stack or of one RGB image."""
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[np.newaxis, ..., :3]
    if image.ndim == 4 and image.shape[-1] in (3, 4):
        return image[..., :3]
    raise SystemExit(f'{name}: expected an RGB stack, got shape {image.shape}')


def unit_range(stack):
    stack = stack.astype(np.float32)
    low, high = stack.min(), stack.max()
    if high <= low:
        return np.zeros_like(stack, dtype=np.float32)
    return (stack - low) / (high - low)


def merge_channels(archive, members, mode):
    merged = None
    for channel in sorted(members):
        stack = unit_range(rgb_stack(read_tiff(archive, members[channel]), members[channel]))
        print(f'  ch0{channel}: {stack.shape}', flush=True)
        if merged is None:
            merged = stack
        elif stack.shape != merged.shape:
            raise SystemExit(f'{members[channel]}: shape {stack.shape} differs from ch00 {merged.shape}')
        elif mode == 'sum':
            merged += stack
        else:
            np.maximum(merged, stack, out=merged)
    return (np.clip(merged, 0, 1) * 255).astype(np.uint8)


def save_rgb_pages(path, stack):
    """One plain RGB page per Z slice, uncompressed."""
    pages = [Image.fromarray(np.ascontiguousarray(stack[z]), mode='RGB') for z in range(stack.shape[0])]
    pages[0].save(path, save_all=True, append_images=pages[1:], compression='raw')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--download', help='folder to download channels.zip and masks2.zip into (kept if complete)')
    source.add_argument('--channels-zip', help='channels.zip of the record, downloaded by hand (with --masks-zip)')
    parser.add_argument('--masks-zip', help='masks2.zip of the record')
    parser.add_argument('--output', required=True, help='folder for images/, nuc_masks/ and cyto_masks/')
    parser.add_argument('--acquisition', default='012', help='acquisition number (the paper: 012)')
    parser.add_argument('--mode', choices=('sum', 'max'), default='sum', help='merge of the four stacks')
    args = parser.parse_args()
    if args.download:
        channels_zip, masks_zip = download(Path(args.download))
    elif not args.masks_zip:
        raise SystemExit('--channels-zip needs --masks-zip')
    else:
        channels_zip, masks_zip = Path(args.channels_zip), Path(args.masks_zip)

    tag = f'_{args.acquisition}_'
    output = Path(args.output)
    with zipfile.ZipFile(channels_zip) as archive:
        members = {}
        for member in archive.namelist():
            match = CHANNEL.search(member)
            if match and tag in Path(member).name:
                members[int(match.group(1))] = member
        if sorted(members) != [0, 1, 2, 3]:
            raise SystemExit(f'{channels_zip}: acquisition {args.acquisition} has channels {sorted(members)}, '
                             'expected ch00-ch03')
        stem = CHANNEL.sub('', Path(members[0]).name)            # e.g. 'HeLa_MRC_DLS Acquire 2_012_Target_z00'
        print(f'{stem}: merging the four RGB stacks ({args.mode})', flush=True)
        merged = merge_channels(archive, members, args.mode)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    image_path = output / 'images' / f'{stem}_merged_rgb_plain_stack.tif'
    save_rgb_pages(image_path, merged)
    print(f'  -> {image_path} {merged.shape} uint8')

    with zipfile.ZipFile(masks_zip) as archive:
        for channel, folder in (('ch00', 'nuc_masks'), ('ch03', 'cyto_masks')):
            found = [m for m in archive.namelist() if Path(m).name == f'{stem}_{channel}_mask.tif']
            if len(found) != 1:
                raise SystemExit(f'{masks_zip}: no {stem}_{channel}_mask.tif')
            (output / folder).mkdir(parents=True, exist_ok=True)
            path = output / folder / f'{stem}.tif'
            path.write_bytes(archive.read(found[0]))
            labels = tifffile.imread(path)
            if labels.shape != merged.shape[:3]:
                raise SystemExit(f'{found[0]}: shape {labels.shape} differs from the image {merged.shape[:3]}')
            print(f'  -> {path} {labels.shape} {labels.dtype}, {len(np.unique(labels)) - 1} objects')


if __name__ == '__main__':
    main()
