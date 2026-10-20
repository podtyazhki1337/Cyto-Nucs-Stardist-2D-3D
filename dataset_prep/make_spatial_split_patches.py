"""CytoNucs patch dataset with a spatially disjoint train / val / test split, from one or several volumes.

make_diverse_patches_from_single_image.ipynb cuts overlapping patches (Z stride 8 of 16, XY stride 448 of 512)
plus augmented copies, and the split was made per file afterwards, so validation objects were also in train.
Here the XY plane of every source volume is first divided into blocks and every block goes to exactly one
split (over the full Z); a patch never leaves the blocks of its split, so no voxel belongs to two splits.
  train      the notebook's overlapping windows wherever they lie entirely in train blocks (also across
             neighbouring train blocks), plus offline flip/appearance copies
  val, test  one tile per block in XY, Z slabs of one patch, originals only

Output is the layout the training reads directly (set preprocessing.run_preprocessing: false and
data.dataset_dir to <output>):
  <output>/{train,val,test}/{images,nucleus_masks,cytoplasm_masks,assignments}/  + meta.json
  <output>/split_info.json, <output>/split_manifest.json

Usage (all volumes of a folder; masks are found by name in their folders):
  python dataset_prep/make_spatial_split_patches.py --image IMAGES_DIR --nucleus-mask NUC_DIR \\
      --cytoplasm-mask CYTO_DIR --output OUT_DIR
--image also takes one or several files. A mask folder is searched without subfolders for the file named like
the image: same name, same stem, or the longest stem the image name starts with; else its only TIFF.
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.segmentation import relabel_sequential

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # compute_nucleus_cell_assignments of the project
SPLITS = ('train', 'val', 'test')
# Foreground filter of training patches, as in the notebook.
MIN_FOREGROUND_FRACTION = 0.001
KEEP_BACKGROUND_PROBABILITY = 0.05
MIN_FOREGROUND_VOXELS = 20


def patch_starts(size, patch, stride):
    """Sliding-window starts; the last window is forced to reach the end."""
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def slab_starts(size, patch, stride):
    """patch_starts along Z without a final window shifted by less than half a stride: such a window repeats
    the previous one almost entirely (17 slices, patch 16: one slab, not two)."""
    starts = patch_starts(size, patch, stride)
    if len(starts) > 1 and starts[-1] - starts[-2] < stride / 2:
        starts.pop()
    return starts


def pad_to_minimum_size(image, nucleus, cytoplasm, patch):
    pad = [((max(0, p - s)) // 2, max(0, p - s) - max(0, p - s) // 2) for s, p in zip(nucleus.shape, patch)]
    if not any(a or b for a, b in pad):
        return image, nucleus, cytoplasm
    print('Padding volume:', pad)
    image = np.pad(image, pad + [(0, 0)] * (image.ndim - 3), mode='reflect')
    return image, np.pad(nucleus, pad), np.pad(cytoplasm, pad)


def relabel_mask(mask):
    relabeled = relabel_sequential(np.asarray(mask))[0]
    return relabeled.astype(np.uint16 if int(relabeled.max()) <= np.iinfo(np.uint16).max else np.uint32)


def block_grid(shape_yx, patch_yx):
    """Blocks of the XY plane, each at least one patch wide."""
    edges = [np.linspace(0, s, max(1, s // p) + 1).astype(int) for s, p in zip(shape_yx, patch_yx)]
    return [(int(y0), int(y1), int(x0), int(x1)) for y0, y1 in zip(edges[0], edges[0][1:])
            for x0, x1 in zip(edges[1], edges[1][1:])]


def held_out_blocks(n, patch_yx, val_fraction, test_fraction):
    counts = {'val': max(1, round(n * val_fraction)) if val_fraction > 0 else 0,
              'test': max(1, round(n * test_fraction)) if test_fraction > 0 else 0}
    if counts['val'] + counts['test'] >= n:
        raise ValueError(f'{n} XY block(s) of at least {tuple(patch_yx)} px cannot hold train, '
                         f'{counts["val"]} val and {counts["test"]} test block(s): use a smaller XY patch or fractions')
    return counts


def split_blocks(foreground_yx, patch_yx, val_fraction, test_fraction, seed):
    """Blocks of the XY plane and the split of every block.

    Among the possible assignments the one whose labelled-voxel shares are closest to the requested
    fractions is taken: blocks at the rim of a spheroid can be almost empty.
    """
    blocks = block_grid(foreground_yx.shape, patch_yx)
    n = len(blocks)
    counts = held_out_blocks(n, patch_yx, val_fraction, test_fraction)
    weight = np.array([foreground_yx[y0:y1, x0:x1].sum() for y0, y1, x0, x1 in blocks], np.float64)
    weight /= max(weight.sum(), 1.0)
    rng = np.random.default_rng(seed)
    exhaustive = n <= 12
    orders = itertools.permutations(range(n), counts['val'] + counts['test']) if exhaustive else \
        (rng.permutation(n)[:counts['val'] + counts['test']] for _ in range(2000))
    best = None
    for order in orders:
        val, test = list(order[:counts['val']]), list(order[counts['val']:])
        if exhaustive and (val != sorted(val) or test != sorted(test)):
            continue                                              # the same sets in another order
        score = abs(weight[val].sum() - val_fraction) + abs(weight[test].sum() - test_fraction)
        if best is None or score < best[0] - 1e-12:
            best = (score, val, test)
    assignment = ['train'] * n
    for split, members in (('val', best[1]), ('test', best[2])):
        for b in members:
            assignment[b] = split
    return blocks, assignment


def train_windows(train_region, train_blocks, patch_yx, stride_yx):
    """XY starts of the training windows: the sliding grid of the whole plane wherever a window lies entirely in
    train blocks (it may span neighbouring train blocks, so objects on their border are also seen whole), and
    the windows of a single block where that grid leaves a part of a train block uncovered."""
    (py, px), (sy, sx) = patch_yx, stride_yx
    starts = [(y, x) for y in patch_starts(train_region.shape[0], py, sy) for x in patch_starts(train_region.shape[1], px, sx)
              if train_region[y:y + py, x:x + px].all()]
    covered = np.zeros(train_region.shape, bool)
    for y, x in starts:
        covered[y:y + py, x:x + px] = True
    for y0, y1, x0, x1 in train_blocks:
        if not covered[y0:y1, x0:x1].all():
            starts += [(y0 + y, x0 + x) for y in patch_starts(y1 - y0, py, sy) for x in patch_starts(x1 - x0, px, sx)]
    return sorted(set(starts))


def geometric_augmentation(image, nucleus, cytoplasm, rng):
    """Lossless only: rot90 in XY (square patches) and flips; masks are never interpolated."""
    params = {}
    k = int(rng.integers(0, 4)) if image.shape[1] == image.shape[2] else 0
    if k:
        image, nucleus, cytoplasm = (np.rot90(a, k=k, axes=(1, 2)) for a in (image, nucleus, cytoplasm))
    params['rotation_xy'] = 90 * k
    for name, axis in (('flip_y', 1), ('flip_x', 2)):
        params[name] = bool(rng.random() < 0.5)
        if params[name]:
            image, nucleus, cytoplasm = (np.flip(a, axis=axis) for a in (image, nucleus, cytoplasm))
    return (*(np.ascontiguousarray(a) for a in (image, nucleus, cytoplasm)), params)


def robust_normalize_01(image):
    """Percentile normalization to [0, 1], per channel."""
    image = image.astype(np.float32)
    channels = [image[..., c] for c in range(image.shape[-1])] if image.ndim == 4 else [image]
    out = []
    for channel in channels:
        lo, hi = np.percentile(channel, 0.5), np.percentile(channel, 99.5)
        out.append(np.clip((channel - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(channel))
    return np.stack(out, -1) if image.ndim == 4 else out[0]


def intensity_augmentation(image, rng):
    """Appearance only (gamma, contrast, brightness, illumination gradient, Z drift, blur, sharpening,
    Gaussian and Poisson noise); geometry is untouched. Ranges as in the notebook."""
    image = robust_normalize_01(image)
    params = {}
    spatial = (slice(None),) * 3 + (None,) * (image.ndim - 3)
    if rng.random() < 0.85:
        params['gamma'] = float(rng.uniform(0.70, 1.40))
        image = np.power(np.clip(image, 0, 1), params['gamma'])
    if rng.random() < 0.85:
        params['contrast'] = float(rng.uniform(0.70, 1.35))
        mean = float(image.mean())
        image = (image - mean) * params['contrast'] + mean
    if rng.random() < 0.75:
        params['brightness'] = float(rng.uniform(-0.12, 0.12))
        image = image + params['brightness']
    if rng.random() < 0.60:
        yy, xx = np.meshgrid(np.linspace(-1, 1, image.shape[1]), np.linspace(-1, 1, image.shape[2]), indexing='ij')
        ax, ay, axy = (float(rng.uniform(-r, r)) for r in (0.25, 0.25, 0.10))
        image = image * np.exp(ax * xx + ay * yy + axy * xx * yy).astype(np.float32)[None][spatial]
        params.update(illumination_ax=ax, illumination_ay=ay, illumination_axy=axy)
    if rng.random() < 0.45 and image.shape[0] > 2:
        factors = rng.normal(1.0, float(rng.uniform(0.02, 0.08)), image.shape[0]).astype(np.float32)
        factors = gaussian_filter1d(factors, sigma=1.0, mode='reflect')
        image = image * factors[:, None, None][spatial]
        params['z_intensity_std'] = float(np.std(factors))
    tail = (0,) * (image.ndim - 3)
    if rng.random() < 0.35:
        sigma_xy, sigma_z = float(rng.uniform(0.20, 0.80)), float(rng.uniform(0.0, 0.20))
        image = gaussian_filter(image, sigma=(sigma_z, sigma_xy, sigma_xy) + tail, mode='reflect')
        params.update(blur_sigma_z=sigma_z, blur_sigma_xy=sigma_xy)
    if rng.random() < 0.25:
        blurred = gaussian_filter(image, sigma=(0, 0.8, 0.8) + tail, mode='reflect')
        params['sharpen_strength'] = float(rng.uniform(0.15, 0.50))
        image = image + params['sharpen_strength'] * (image - blurred)
    if rng.random() < 0.65:
        params['gaussian_noise'] = float(rng.uniform(0.003, 0.025))
        image = image + rng.normal(0.0, params['gaussian_noise'], image.shape).astype(np.float32)
    if rng.random() < 0.20:
        params['poisson_levels'] = float(10 ** rng.uniform(2.7, 4.0))
        image = (rng.poisson(np.clip(image, 0, 1) * params['poisson_levels']) / params['poisson_levels']).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32), params


def restore_dtype(image, dtype):
    image = np.clip(image, 0, 1)
    return np.round(image * np.iinfo(dtype).max).astype(dtype) if np.issubdtype(dtype, np.integer) else image.astype(np.float32)


def write_tile(folder, name, image, nucleus, cytoplasm):
    """Same TIFF form as the training's own preprocessing (plain pages, RGB kept)."""
    from cytonucs_losses import compute_nucleus_cell_assignments
    tifffile.imwrite(str(folder / 'images' / name), image[..., :3] if image.ndim == 4 else image)
    for sub, mask in (('nucleus_masks', nucleus), ('cytoplasm_masks', cytoplasm)):
        tifffile.imwrite(str(folder / sub / name), mask, photometric='minisblack')
    assignments = compute_nucleus_cell_assignments(nucleus, cytoplasm, max_distance=50.0)[0]
    (folder / 'assignments' / (Path(name).stem + '.json')).write_text(json.dumps(assignments, indent=2))


def median_extent(nucleus, cytoplasm):
    from scipy.ndimage import find_objects
    extents = [[s.stop - s.start for s in sl] for mask in (nucleus, cytoplasm) for sl in find_objects(mask) if sl is not None]
    return np.median(extents, axis=0).tolist() if extents else None


def volume_name(path):
    """File stem reduced to letters, digits, '-' and '_' (names go into patch file names)."""
    return re.sub(r'_+', '_', re.sub(r'[^A-Za-z0-9-]+', '_', Path(path).stem)).strip('_')


def channel_axis(image_shape, mask_shape):
    """Axis of the image that holds the channels (None: grayscale), given the ZYX shape of its masks."""
    image_shape, mask_shape = tuple(image_shape), tuple(mask_shape)
    if image_shape == mask_shape:
        return None
    for axis in (3, 1, 0):                                       # ZYXC, ZCYX, CZYX
        if len(image_shape) == 4 and image_shape[:axis] + image_shape[axis + 1:] == mask_shape:
            return axis
    raise ValueError(f'image {image_shape} does not match the masks {mask_shape} (ZYX with an optional channel axis)')


def channel_signal(image, nucleus, cytoplasm):
    """Mean of every channel in nuclei, in the rest of the cells and in the background: which channel shows what."""
    regions = dict(nuclei=nucleus > 0, cell_body=(cytoplasm > 0) & (nucleus == 0), background=(cytoplasm == 0) & (nucleus == 0))
    image = image[..., None] if image.ndim == 3 else image
    return {k: [float(image[..., c][m].mean()) if m.any() else 0.0 for c in range(image.shape[-1])] for k, m in regions.items()}


def build(volumes, output, patch_size=(16, 512, 512), train_stride=(8, 448, 448), augmentations=4,
          val_fraction=0.15, test_fraction=0.15, seed=42):
    """volumes: iterable of dict(name, image, nucleus ZYX, cytoplasm ZYX[, source]); it is read one volume at a time."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty: choose a new output folder')
    for split in SPLITS:
        for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
            (output / split / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, meta, skipped, described = [], {s: [] for s in SPLITS}, dict(background=0, tiny=0), []

    for number, volume in enumerate(volumes):
        name, image = volume['name'], volume['image']
        nucleus, cytoplasm = np.squeeze(volume['nucleus']), np.squeeze(volume['cytoplasm'])
        if name in {d['name'] for d in described}:
            raise ValueError(f'Two volumes are named {name}')
        if nucleus.ndim != 3 or nucleus.shape != cytoplasm.shape:
            raise ValueError(f'{name}: the masks must be ZYX volumes of one shape, got {nucleus.shape} and {cytoplasm.shape}')
        try:
            axis = channel_axis(image.shape, nucleus.shape)
        except ValueError as error:
            raise ValueError(f'{name}: {error}') from None
        image = image if axis is None else np.moveaxis(image, axis, -1)
        channels = 1 if axis is None else image.shape[-1]
        if described and channels != described[0]['channels']:
            raise ValueError(f"{name}: {channels} channel(s), {described[0]['name']} has {described[0]['channels']}")
        signal = channel_signal(image, nucleus, cytoplasm)
        image, nucleus, cytoplasm = pad_to_minimum_size(image, nucleus, cytoplasm, patch_size)
        depth, plane = nucleus.shape[0], nucleus.shape[1:]
        labelled = (nucleus > 0) | (cytoplasm > 0)
        blocks, assignment = split_blocks(labelled.sum(0), patch_size[1:], val_fraction, test_fraction, seed + number)
        region, block_of = {s: np.zeros(plane, bool) for s in SPLITS}, np.zeros(plane, int)
        for index, ((y0, y1, x0, x1), split) in enumerate(zip(blocks, assignment)):
            region[split][y0:y1, x0:x1] = True
            block_of[y0:y1, x0:x1] = index

        def save(split, z0, y0, x0, img, nuc, cyto, aug_index, params, touched):
            file = f'{name}_z{z0:03d}_y{y0:04d}_x{x0:04d}_a{aug_index:02d}.tiff'
            nuc, cyto = relabel_mask(nuc), relabel_mask(cyto)
            write_tile(output / split, file, img, nuc, cyto)
            rows.append(dict(filename=file, split=split, volume=name, blocks=touched, z0=z0, y0=y0, x0=x0, shape=list(nuc.shape),
                             augmentation_index=aug_index, augmentation_params=params,
                             n_nuclei=int(nuc.max()), n_cells=int(cyto.max())))
            meta[split].append(dict(filename=file, cells_median_extent_p=median_extent(nuc, cyto), rescale_factor=1.0,
                                    n_nuclei=int(nuc.max()), n_cells=int(cyto.max())))

        train_blocks = [b for b, a in zip(blocks, assignment) if a == 'train']
        windows = [('train', z, y, x, patch_size[1], patch_size[2]) for z in slab_starts(depth, patch_size[0], train_stride[0])
                   for y, x in train_windows(region['train'], train_blocks, patch_size[1:], train_stride[1:])]
        windows += [(split, z, y0, x0, y1 - y0, x1 - x0)           # the whole block in XY, Z slabs of one patch
                    for (y0, y1, x0, x1), split in zip(blocks, assignment) if split != 'train'
                    for z in slab_starts(depth, patch_size[0], patch_size[0])]
        for split, z, y, x, height, width in windows:
            sl = (slice(z, z + patch_size[0]), slice(y, y + height), slice(x, x + width))
            img, nuc, cyto = image[sl].copy(), nucleus[sl].copy(), cytoplasm[sl].copy()
            voxels = int(labelled[sl].sum())
            if split != 'train':
                if voxels == 0:
                    skipped['background'] += 1
                    continue
            elif 0 < voxels < MIN_FOREGROUND_VOXELS:
                skipped['tiny'] += 1
                continue
            elif voxels / nuc.size < MIN_FOREGROUND_FRACTION and rng.random() > KEEP_BACKGROUND_PROBABILITY:
                skipped['background'] += 1                         # a few almost empty patches are kept
                continue
            touched = np.unique(block_of[sl[1:]]).tolist()
            save(split, z, y, x, img, nuc, cyto, 0, {}, touched)
            for index in range(1, augmentations + 1) if split == 'train' else ():
                a_img, a_nuc, a_cyto, params = geometric_augmentation(img, nuc, cyto, rng)
                a_img, appearance = intensity_augmentation(a_img, rng)
                save(split, z, y, x, restore_dtype(a_img, image.dtype), a_nuc, a_cyto, index, dict(params, **appearance), touched)

        summary = {}
        for split in SPLITS:
            objects = lambda labels: int(np.count_nonzero(np.unique(labels[:, region[split]])))
            mine = [r for r in rows if r['volume'] == name and r['split'] == split]
            summary[split] = dict(blocks=assignment.count(split), xy_fraction=float(region[split].mean()),
                                  labelled_voxel_fraction=float(labelled[:, region[split]].sum() / max(labelled.sum(), 1)),
                                  source_nuclei=objects(nucleus), source_cells=objects(cytoplasm),
                                  files=len(mine), originals=sum(r['augmentation_index'] == 0 for r in mine))
        contrast = np.subtract(signal['nuclei'], signal['cell_body'])
        described.append(dict(name=name, source=volume.get('source'), shape=list(nucleus.shape), channels=channels,
                              dtype=str(image.dtype), channel_means=signal, nucleus_channel=int(np.argmax(contrast)),
                              summary=summary,
                              blocks=[dict(index=i, y0=b[0], y1=b[1], x0=b[2], x1=b[3], split=a)
                                      for i, (b, a) in enumerate(zip(blocks, assignment))]))
    if not described:
        raise ValueError('No source volume')

    for split in SPLITS:
        (output / split / 'meta.json').write_text(json.dumps(meta[split], indent=2))
    files = {s: [r['filename'] for r in rows if r['split'] == s] for s in SPLITS}
    (output / 'split_info.json').write_text(json.dumps(dict(files, seed=seed, split='spatial_blocks', rescale_factor=1.0), indent=2))
    total = {s: {k: sum(d['summary'][s][k] for d in described) for k in ('blocks', 'files', 'originals', 'source_nuclei', 'source_cells')}
             for s in SPLITS}
    manifest = dict(patch_size=list(patch_size), train_stride=list(train_stride), augmentations_per_train_patch=augmentations,
                    val_fraction=val_fraction, test_fraction=test_fraction, seed=seed, volumes=described, summary=total,
                    skipped=skipped, files=rows)
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def source_images(given):
    """Files as given; a folder stands for the TIFF files directly inside it."""
    images = []
    for path in map(Path, given):
        images += sorted(c for c in path.iterdir() if c.is_file() and c.suffix.lower() in ('.tif', '.tiff')) if path.is_dir() else [path]
    return images


def resolve_mask(path, image):
    """A mask file, or its folder searched without subfolders: same name as the image, same stem, the longest
    stem the image name starts with (image 'X_merged_rgb_plain_stack', mask 'X'), else the only TIFF."""
    path, image = Path(path), Path(image)
    if not path.is_dir():
        return path
    candidates = sorted(c for c in path.iterdir() if c.is_file() and c.suffix.lower() in ('.tif', '.tiff'))
    for same in (lambda c: c.name == image.name, lambda c: c.stem == image.stem):
        found = [c for c in candidates if same(c)]
        if found:
            return found[0]
    prefixes = sorted((c for c in candidates if image.stem.startswith(c.stem)), key=lambda c: -len(c.stem))
    if prefixes and (len(prefixes) == 1 or len(prefixes[0].stem) > len(prefixes[1].stem)):
        return prefixes[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f'Cannot choose the mask for {image.name} in {path}: {[c.name for c in candidates]}')


def check_sources(plan, patch_size, val_fraction, test_fraction):
    """Every volume checked on its TIFF headers; all problems are returned before anything is read or written."""
    problems, channels, names = [], {}, set()
    for item in plan:
        name = volume_name(item['image'])
        shapes = {}
        for key in ('image', 'nucleus', 'cytoplasm'):
            with tifffile.TiffFile(str(item[key])) as tif:
                shapes[key] = tuple(tif.series[0].shape), str(tif.series[0].dtype)
        nucleus, cytoplasm = (tuple(n for n in shapes[k][0] if n != 1) for k in ('nucleus', 'cytoplasm'))
        print(f"  {name}: image {shapes['image'][0]} {shapes['image'][1]}, nuclei {nucleus}, cells {cytoplasm}")
        try:
            if name in names:
                raise ValueError('another volume has the same name')
            names.add(name)
            if len(nucleus) != 3 or nucleus != cytoplasm:
                raise ValueError('the masks must be ZYX volumes of one shape')
            axis = channel_axis(shapes['image'][0], nucleus)
            channels[name] = 1 if axis is None else shapes['image'][0][axis]
            plane = [max(n, p) for n, p in zip(nucleus[1:], patch_size[1:])]
            held_out_blocks(len(block_grid(plane, patch_size[1:])), patch_size[1:], val_fraction, test_fraction)
        except ValueError as error:
            problems.append(f'{name}: {error}')
    if len(set(channels.values())) > 1:
        problems.append(f'the volumes have different numbers of channels: {channels}')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True, nargs='+', help='source volume(s), or the folder that holds them')
    parser.add_argument('--nucleus-mask', required=True, help='folder with the masks (a file is allowed for one volume)')
    parser.add_argument('--cytoplasm-mask', required=True, help='folder with the masks (a file is allowed for one volume)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--patch-size', type=int, nargs=3, default=(16, 512, 512), metavar=('Z', 'Y', 'X'))
    parser.add_argument('--train-stride', type=int, nargs=3, default=(8, 448, 448), metavar=('Z', 'Y', 'X'))
    parser.add_argument('--augmentations', type=int, default=4, help='offline copies of every train patch')
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--test-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    images = source_images(args.image)
    if not images:
        parser.error('no TIFF found for --image')
    if len(images) > 1 and not (Path(args.nucleus_mask).is_dir() and Path(args.cytoplasm_mask).is_dir()):
        parser.error('several volumes need the masks as folders')
    plan = [dict(image=i, nucleus=resolve_mask(args.nucleus_mask, i), cytoplasm=resolve_mask(args.cytoplasm_mask, i)) for i in images]
    for item in plan:                                           # shown before anything is read or written
        print(f"{volume_name(item['image'])}\n  image     {item['image']}\n  nuclei    {item['nucleus']}\n  cytoplasm {item['cytoplasm']}")
    print('Shapes:')
    problems = check_sources(plan, args.patch_size, args.val_fraction, args.test_fraction)
    if problems:
        sys.exit('Nothing was written:\n  ' + '\n  '.join(problems))

    def volumes():
        for item in plan:
            print('Reading', item['image'].name, flush=True)
            yield dict(name=volume_name(item['image']), image=tifffile.imread(str(item['image'])),
                       nucleus=tifffile.imread(str(item['nucleus'])), cytoplasm=tifffile.imread(str(item['cytoplasm'])),
                       source={k: dict(path=str(v), bytes=v.stat().st_size) for k, v in item.items()})

    manifest = build(volumes(), args.output, tuple(args.patch_size), tuple(args.train_stride), args.augmentations,
                     args.val_fraction, args.test_fraction, args.seed)
    for volume in manifest['volumes']:
        print(f"\n{volume['name']}: volume {volume['shape']}, nuclei are brightest in channel {volume['nucleus_channel']}")
        for region, means in volume['channel_means'].items():
            print(f"  channel means in {region:10s}: " + ', '.join(f'{m:.1f}' for m in means))
        print('  blocks: ' + ', '.join(f"y{b['y0']}-{b['y1']} x{b['x0']}-{b['x1']} {b['split']}" for b in volume['blocks']))
        for split, v in volume['summary'].items():
            print(f"  {split:5s}: {v['files']} files ({v['originals']} originals), {v['blocks']} block(s), "
                  f"{v['labelled_voxel_fraction']:.1%} of labelled voxels, {v['source_nuclei']} nuclei, {v['source_cells']} cells")
    print('\nTotal:')
    for split, v in manifest['summary'].items():
        print(f"  {split:5s}: {v['files']} files ({v['originals']} originals), {v['source_nuclei']} nuclei, {v['source_cells']} cells")
    print('Skipped train patches:', manifest['skipped'], '| output:', args.output)
    if len({v['nucleus_channel'] for v in manifest['volumes']}) > 1:
        print('WARNING: the nuclei are brightest in different channels of different volumes: '
              + ', '.join(f"{v['name']} -> {v['nucleus_channel']}" for v in manifest['volumes'])
              + '. The network gets the channels as they are; check the channel order of the source images.')


if __name__ == '__main__':
    main()
