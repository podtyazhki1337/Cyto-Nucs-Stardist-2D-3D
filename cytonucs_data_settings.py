"""Settings of a run read from its training data when the configuration leaves them out.

- model.levels: the '<level>_masks' folders of the train split, inner first (the level with less foreground lies
  inside the other);
- model.n_channel_in: the channels of the first training image;
- training.partial_annotation: whether some training image lacks the masks of a level;
- training.unknown_label: UNKNOWN_LABEL when it occurs in the training masks (sparse annotation);
- data.pool_size: training images kept in memory, as many as fit into POOL_RAM_SHARE of the host memory (all of
  them when they fit).

derive_data_settings fills them into a config dict at the start of training and returns what it derived; the run
stores that in geometry.json, so a checkpoint is read back with the same settings.
"""
import os
from pathlib import Path

import numpy as np

UNKNOWN_LABEL = 65535          # dataset_prep/sparse_annotation.py: voxels whose annotation is unknown
POOL_RAM_SHARE = 0.15          # share of the host memory for the training images of one run (three runs share a host)
BYTES_PER_VOXEL_OVERHEAD = 1.5 # crop indices etc. on top of image and masks (TissueNet: ~6.3 MB per 5.2 MB image)
SAMPLE = 32                    # mask files read per level for the level order and the unknown label


def level_folders(split_dir):
    """Level names of the '<level>_masks' folders of a split."""
    return sorted(p.name[:-len('_masks')] for p in Path(split_dir).iterdir() if p.is_dir() and p.name.endswith('_masks'))


def _sample(files, n=SAMPLE):
    files = sorted(files)
    return files if len(files) <= n else [files[int(i)] for i in np.linspace(0, len(files) - 1, n)]


def inner_first(split_dir, levels):
    """Levels ordered inner first: by the mean foreground share of their masks (an inner object lies inside its outer
    one, so it covers less)."""
    import tifffile
    share = {}
    for level in levels:
        files = _sample(list((Path(split_dir) / f'{level}_masks').iterdir()))
        values = [float(np.mean((m > 0) & (m != UNKNOWN_LABEL))) for m in (tifffile.imread(f) for f in files)]
        share[level] = float(np.mean(values)) if values else 0.0
    return sorted(levels, key=lambda level: share[level])


def image_channels(path, ndim):
    """Channels of an image: 1 without a channel axis; else the TIFF 'C' / 'S' axis, or the last (or, in 3D, the
    second) axis of at most 4 entries."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        shape, axes = tif.series[0].shape, tif.series[0].axes
    if len(shape) == ndim:
        return 1
    if len(shape) != ndim + 1:
        raise ValueError(f'{path}: expected {ndim} spatial axes and an optional channel axis, got shape {shape}')
    for i, axis in enumerate(axes):
        if axis in ('C', 'S'):
            return int(shape[i])
    for i in ([-1, 1] if ndim == 3 else [-1]):
        if shape[i] <= 4:
            return int(shape[i])
    raise ValueError(f'{path}: cannot tell the channel axis of shape {shape} (axes {axes})')


def has_unknown_label(split_dir, levels):
    import tifffile
    for level in levels:
        for f in _sample(list((Path(split_dir) / f'{level}_masks').iterdir())):
            m = tifffile.imread(f)
            if np.issubdtype(m.dtype, np.integer) and np.iinfo(m.dtype).max >= UNKNOWN_LABEL and np.any(m == UNKNOWN_LABEL):
                return True
    return False


def pool_size(split_dir, ndim, channels, ram_bytes=None):
    """Training images that fit into POOL_RAM_SHARE of the host memory, None (all) when every one fits. The size of
    an image is estimated from a sample of the image files."""
    import tifffile
    images = sorted((Path(split_dir) / 'images').iterdir())
    if not images:
        return None
    if ram_bytes is None:
        ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    voxels = []
    for f in _sample(images):
        with tifffile.TiffFile(f) as tif:
            shape = tif.series[0].shape
        voxels.append(int(np.prod(shape)) // (1 if len(shape) == ndim else max(int(channels), 1)))
    per_image = float(np.mean(voxels)) * (4 * channels + 8) * BYTES_PER_VOXEL_OVERHEAD    # float32 image, two int32 masks
    fit = int(POOL_RAM_SHARE * ram_bytes // max(per_image, 1.0))
    return None if fit >= len(images) else max(1, fit)


def derive_data_settings(config_dict, ram_bytes=None):
    """Fill the settings this module derives into a config dict (in place) where it has none, from the train split of
    data.dataset_dir. Returns {key: value} of what it derived (for geometry.json)."""
    train = Path(config_dict['data']['dataset_dir']) / 'train'
    model = config_dict.setdefault('model', {})
    training = config_dict.setdefault('training', {})
    data = config_dict.setdefault('data', {})
    ndim = int(model['ndim'])
    derived = {}
    if 'levels' not in model:
        derived['levels'] = inner_first(train, level_folders(train))
    levels = model.get('levels', derived.get('levels'))
    if 'n_channel_in' not in model:
        derived['n_channel_in'] = image_channels(sorted((train / 'images').iterdir())[0], ndim)
    if 'partial_annotation' not in training:
        n_images = len(list((train / 'images').iterdir()))
        derived['partial_annotation'] = any(len(list((train / f'{level}_masks').iterdir())) < n_images for level in levels)
    if 'unknown_label' not in training:
        derived['unknown_label'] = UNKNOWN_LABEL if has_unknown_label(train, levels) else None
    if 'pool_size' not in data:
        derived['pool_size'] = pool_size(train, ndim, model.get('n_channel_in', derived.get('n_channel_in', 1)), ram_bytes)
    section = dict(levels=model, n_channel_in=model, partial_annotation=training, unknown_label=training, pool_size=data)
    for key, value in derived.items():
        section[key][key] = value
    if derived:
        print('data settings derived from the training data: ' + ', '.join(f'{k}={v!r}' for k, v in derived.items()))
    return derived
