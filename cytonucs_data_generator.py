"""
CytoNucs StarDist Data Generator

Loads paired nucleus + cytoplasm annotations.
Creates training batches with augmentation.
Supports repooling and automatic assignment creation.
"""
import numpy as np
from scipy import ndimage as ndi
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from pathlib import Path
import imageio
from cytonucs_runtime import normalize_percentiles, percentile_range
from skimage.transform import rescale
import json
import random
import shutil
from cytonucs_losses import compute_nucleus_cell_assignments, compute_stardist_gt


def axial_prompt_centers(owner):
    """Two interior points on separated occupied Z planes; None for a thin mask."""
    if owner.ndim != 3:
        return None
    zs = np.flatnonzero(np.any(owner, axis=(1, 2)))
    if len(zs) < 2:
        return None
    # Quantiles of occupied planes, not voxel count: thick middle must not
    # collapse both prompts onto the same plane. For two planes use both.
    planes = (zs[int(np.floor(.25 * (len(zs)-1)))], zs[int(np.ceil(.75 * (len(zs)-1)))])
    centers = []
    for z in planes:
        points = np.argwhere(owner[z])
        xy = points[np.argmin(np.sum((points - points.mean(0)) ** 2, axis=1))]
        centers.append(np.array([z, *xy], np.float32))
    return centers


def nearest_cell_neighbors(labels, anchor, anisotropy):
    """Distinct cells ordered by physical centroid distance, stable on label ties."""
    from scipy.ndimage import center_of_mass
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if anchor not in ids:
        return []
    centers = np.asarray(center_of_mass(labels > 0, labels, ids))
    origin = centers[np.flatnonzero(ids == anchor)[0]]
    distance = np.sum(((centers - origin) * np.asarray(anisotropy)) ** 2, axis=1)
    return [int(ids[i]) for i in np.lexsort((ids, distance)) if ids[i] != anchor]


def project_prompt_center(center, owner, extent):
    """Keep the positive prompt on its labelled instance, including concave GT."""
    center = np.asarray(center, np.float32)
    rounded = np.round(center).astype(int)
    if np.all(rounded >= 0) and np.all(rounded < owner.shape) and owner[tuple(rounded)]:
        return center
    points = np.argwhere(owner)
    if not len(points):
        raise ValueError('Positive prompt requires a nonempty owner')
    distance = np.sum(((points - center) / np.maximum(extent, 1)) ** 2, axis=1)
    return points[np.argmin(distance)].astype(np.float32)


def label_counts(labels):
    """Voxel count of every label ID (sparse IDs allowed)."""
    labels = np.asarray(labels)
    if labels.size and labels.min() >= 0 and labels.max() < 1 << 20:
        counts = np.bincount(labels.ravel())
        ids = np.flatnonzero(counts)
        return dict(zip(ids.tolist(), counts[ids].tolist()))
    ids, counts = np.unique(labels, return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


def partial_labels(crop, reference):
    """IDs present in `crop` with fewer voxels than in `reference` (a label_counts dict)."""
    return {k for k, n in label_counts(crop).items() if k and n < reference.get(k, n)}


def split_unknown(labels, unknown_label):
    """training.unknown_label: the labels with those voxels set to background, and the boolean mask of
    the voxels whose annotation is unknown (None when there are none)."""
    if unknown_label is None:
        return labels, None
    unknown = labels == unknown_label
    if not unknown.any():
        return labels, None
    return np.where(unknown, 0, labels).astype(labels.dtype), unknown


def with_unknown(labels, unknown):
    """Labels as int32 with the voxels of unknown annotation at -1: flips, rotations and order-0
    zoom carry the mark along with the labels."""
    out = labels.astype(np.int32)
    out[unknown] = -1
    return out


def elastic_displacement(shape, alpha, sigma, rng):
    """Smooth random XY displacement (dy, dx) of about `alpha` px, correlated over `sigma` px; it fades
    to zero over 2 * sigma px at the crop faces, so objects there are not pushed out of the crop."""
    fields = []
    for _ in range(2):
        field = ndi.gaussian_filter(rng.normal(size=shape), sigma)
        fields.append(field / (field.std() + 1e-8) * alpha)
    ramp = lambda n: np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / (2 * sigma), 0, 1)
    taper = np.minimum.outer(ramp(shape[0]), ramp(shape[1]))
    return fields[0] * taper, fields[1] * taper


def warp_yx(array, dy, dx, order, nd):
    """The same XY displacement applied to every slice and channel of (Z,) Y, X[, C]; order 0 for labels."""
    a = np.asarray(array)
    h, w = dy.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    coords = np.stack([yy + dy, xx + dx])
    moved = np.moveaxis(a, (nd - 2, nd - 1), (-2, -1)) if a.ndim > nd else a
    out = np.stack([ndi.map_coordinates(plane, coords, order=order, mode='nearest')
                    for plane in moved.reshape((-1, h, w))]).reshape(moved.shape)
    return np.ascontiguousarray(np.moveaxis(out, (-2, -1), (nd - 2, nd - 1)) if a.ndim > nd else out)


def without_unknown(labels, dtype):
    """Inverse of with_unknown: (labels in `dtype`, boolean mask of the unknown voxels)."""
    unknown = labels < 0
    return np.where(unknown, 0, labels).astype(dtype), unknown


def window_sum_table(region):
    """Summed-area table of a boolean YX region, one row and column of zeros in front."""
    return np.pad(np.cumsum(np.cumsum(region, 0, dtype=np.int64), 1), ((1, 0), (1, 0)))


def window_inside(table, start, size):
    """Whether the YX window lies entirely inside the region of window_sum_table."""
    (y, x), (h, w) = start, size
    if y < 0 or x < 0 or y + h >= table.shape[0] or x + w >= table.shape[1]:
        return False
    return int(table[y + h, x + w] - table[y, x + w] - table[y + h, x] + table[y, x]) == h * w


def allowed_crop_starts(region, size):
    """allowed[y, x]: the size window starting at (y, x) lies entirely inside the boolean YX region."""
    h, w = size
    if h > region.shape[0] or w > region.shape[1]:
        return np.zeros((0, 0), bool)
    t = window_sum_table(region)
    return t[h:, w:] - t[:-h, w:] - t[h:, :-w] + t[:-h, :-w] == h * w


def inside_outer_point(inner, outer, unknown, rng):
    """A voxel, uniformly drawn, of an outer object that no inner object covers and no unknown region holds, or
    None when the crop has none."""
    region = (outer > 0) & (inner == 0)
    for u in unknown.values():
        if u is not None:
            region &= ~u
    points = np.flatnonzero(region)
    if not len(points):
        return None
    return tuple(int(v) for v in np.unravel_index(points[int(rng.integers(len(points)))], region.shape))


class CytoNucsDataGenerator(Sequence):
    """
    Data generator for CytoNucs StarDist training.
    Similar to StarDistSequence from original code.

    Expected data structure:
    dataset_dir/
        images/
            sample001.tif
            sample002.tif
        nucleus_masks/
            sample001.tif
            sample002.tif
        cytoplasm_masks/
            sample001.tif
            sample002.tif
        assignments/
            sample001.json  # mapping: {nucleus_id: cytoplasm_id}
            sample002.json
        meta.json  # optional metadata
    """

    def __init__(
            self,
            dataset_dir,
            config,
            rays,
            augmenter=None,
            shuffle=True,
            subset='train',
            pool_size=None,
            repool_freq=None
    ):
        """
        Args:
            dataset_dir: path to dataset
            config: CytoNucsConfig object
            rays: Rays object for distance computation
            augmenter: optional augmentation function
            shuffle: shuffle data after each epoch
            subset: 'train' or 'val'
            pool_size: how many images to keep in memory (-1 for all)
            repool_freq: how often to reload data (in batches)
        """
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.rays = rays
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.subset = subset
        self.counter = 0
        self.epoch = 0
        self.seed = int(getattr(config, 'seed', 42))
        self.r = random.Random(42)

        def _rd(nm):
            g = tuple(getattr(config, nm, (1,) * config.ndim))
            if len(g) != config.ndim or any(int(v) < 1 for v in g):
                g = (1,) * config.ndim
            return tuple(int(v) for v in g)

        self._nuc_grid = _rd('nucleus_grid')
        self._cyto_grid = _rd('cytoplasm_grid')
        self._nuc_rays = getattr(config, 'rays_nucleus', rays)
        self._cyto_rays = getattr(config, 'rays_cytoplasm', rays)
        self._nuc_aniso = getattr(config, 'rays_anisotropy_nucleus', config.anisotropy)
        self._cyto_aniso = getattr(config, 'rays_anisotropy_cytoplasm', config.anisotropy)
        # Zoom is applied in __getitem__ to the patch, not to the whole volume.

        # Load file lists
        def find_all_images(path):
            """All image files of the supported formats (any case of the extension)."""
            files = []
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
                files.extend(list(path.glob(ext)))
            return sorted(set(files))

        self.image_files = find_all_images(self.dataset_dir / 'images')
        if not self.image_files:
            raise ValueError(f'No images found in {self.dataset_dir / "images"}')
        # A level the run does not have (model.levels) needs no mask folder: its masks are empty.
        # Role -> level name of the run; the masks of a role are read from '<level>_masks'.
        self.role_levels = dict(getattr(config, 'level_roles', None) or {'nucleus': 'nucleus', 'cytoplasm': 'cytoplasm'})
        self.levels = tuple(role for role in ('nucleus', 'cytoplasm') if role in self.role_levels)   # roles present
        # training.partial_annotation: a training file may lack the masks of some levels of the
        # run (not annotated there). Validation and test files must have them all.
        self.partial = bool(getattr(config, 'partial_annotation', False)) and subset == 'train'

        def mask_files(name):
            if name not in self.levels:
                return [None] * len(self.image_files)
            found = find_all_images(self.dataset_dir / (self.role_levels[name] + '_masks'))
            if not self.partial:
                return found
            by_name = {f.name: f for f in found}
            unknown = sorted(set(by_name) - {f.name for f in self.image_files})
            if unknown:
                raise ValueError(f'{self.role_levels[name]}_masks holds files without an image: {unknown[:5]}')
            return [by_name.get(f.name) for f in self.image_files]

        self.nucleus_files, self.cytoplasm_files = mask_files('nucleus'), mask_files('cytoplasm')
        if self.partial:
            bare = [f.name for f, n, c in zip(self.image_files, self.nucleus_files, self.cytoplasm_files) if n is None and c is None]
            if bare and getattr(config, 'pair_index', None) is not None:
                # One pair of a hierarchy: files that annotate neither of its levels belong to other pairs.
                keep = [i for i, (n, c) in enumerate(zip(self.nucleus_files, self.cytoplasm_files)) if n is not None or c is not None]
                self.image_files, self.nucleus_files, self.cytoplasm_files = (
                    [files[i] for i in keep] for files in (self.image_files, self.nucleus_files, self.cytoplasm_files))
                if not self.image_files:
                    raise ValueError(f'No training file annotates {list(self.role_levels.values())}')
            elif bare:
                raise ValueError(f'Images without masks of any level: {bare[:5]}')

        assert len(self.image_files) == len(self.nucleus_files) == len(self.cytoplasm_files), \
            "Mismatch in number of images, nucleus masks, and cytoplasm masks"

        # Check that filenames match
        for img_f, nuc_f, cyto_f in zip(self.image_files, self.nucleus_files, self.cytoplasm_files):
            assert all(f is None or f.name == img_f.name for f in (nuc_f, cyto_f)), \
                f"Filename mismatch: {img_f.name}, {nuc_f and nuc_f.name}, {cyto_f and cyto_f.name}"

        # Load assignments if available: 'assignments' holds the nucleus -> cytoplasm pairs; the
        # pairs of other levels are read from 'assignments_<inner>_<outer>' or found by overlap.
        self.assignments_name = ('assignments' if self.role_levels == {'nucleus': 'nucleus', 'cytoplasm': 'cytoplasm'} or
                                 len(self.role_levels) < 2 else
                                 'assignments_' + self.role_levels['nucleus'] + '_' + self.role_levels['cytoplasm'])
        self.assignments_dir = self.dataset_dir / self.assignments_name
        self.has_assignments = self.assignments_dir.exists()
        self.paired = len(self.levels) == 2                     # nucleus-cell pairs exist only with both levels

        # Load metadata if available
        self.meta = None
        meta_file = self.dataset_dir / 'meta.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                self.meta = json.load(f)

        # Pooling setup (like StarDistSequence)
        if pool_size is None or pool_size == -1:
            self.pool_size = len(self.image_files)
        else:
            if isinstance(pool_size, bool) or not isinstance(pool_size, (int, np.integer)) or pool_size < 1:
                raise ValueError('pool_size must be null, -1, or a positive integer')
            self.pool_size = min(int(pool_size), len(self.image_files))

        if repool_freq is None:
            repool_freq = getattr(config, 'repool_freq', 10)
        if isinstance(repool_freq, bool) or not isinstance(repool_freq, (int, np.integer)) or repool_freq < 1:
            raise ValueError('repool_freq must be a positive number of epochs')
        self.repool_freq = int(repool_freq)

        self.data_pool = None

        print(f'\n{subset.upper()} Database:')
        for idx, im_file in enumerate(self.image_files):
            print(f'{idx}\t{im_file.name}')

        # Initial pool
        self.repool(seed=0)

        self.region_pool = None
        if subset == 'train' and getattr(config, 'train_region_crops', False):
            self._load_regions()

        self.indices = np.arange(len(self.data_pool))
        self.on_epoch_end()

        print(f"Loaded {len(self.image_files)} samples from {subset} set")
        print(f"Pool size: {self.pool_size}, Repool frequency: {self.repool_freq}")

    def repool(self, seed=0):
        """Reload a random subset of data into memory.
        Normalisation and foreground coordinates are computed here, once per volume."""
        print(f'Repooling (seed={seed})...')

        r_ = random.Random(seed)
        sample_ids = list(range(len(self.image_files)))
        pool_ids = r_.sample(sample_ids, min(self.pool_size, len(sample_ids)))

        nd = int(self.config.ndim)
        self.data_pool = []
        for idx in pool_ids:
            from cytonucs_runtime import read_model_image
            # Channels of the image files; a hierarchy adds its pair code in __getitem__.
            channels = int(getattr(self.config, 'n_image_channels', self.config.n_channel_in))
            img = read_model_image(self.image_files[idx], nd, channels)
            # 3D multichannel stack stored as (Z, C, H, W) -> channels last (Z, H, W, C)
            if nd == 3 and img.ndim == 4 and img.shape[-1] != channels and img.shape[1] == channels:
                img = np.moveaxis(img, 1, -1)
            nucleus_mask, cytoplasm_mask = (
                np.zeros(img.shape[:nd], np.uint16) if f is None else np.asarray(imageio.volread(f))
                for f in (self.nucleus_files[idx], self.cytoplasm_files[idx]))
            # training.unknown_label: voxels whose annotation of a level is unknown (sparse annotation).
            nucleus_mask, nucleus_unknown = split_unknown(nucleus_mask, getattr(self.config, 'unknown_label', None))
            cytoplasm_mask, cytoplasm_unknown = split_unknown(cytoplasm_mask, getattr(self.config, 'unknown_label', None))
            if self.subset != 'train' and (nucleus_unknown is not None or cytoplasm_unknown is not None):
                raise ValueError(f'{self.image_files[idx].name}: unknown_label in {self.subset} masks; '
                                 'validation and test masks must be complete')

            # Percentile normalisation, once per volume: the statistics are global to the volume and do not depend
            # on the crop or its augmentation.
            if nd == 2 and img.ndim == 3 and img.shape[-1] in (3, 4):
                img = normalize_percentiles(img, 1, 99.8, axis=(0, 1))
            else:
                img = normalize_percentiles(img, 1, 99.8,
                                axis=tuple(range(nd)))
            img = np.ascontiguousarray(img, dtype=np.float32)

            # Foreground coordinates, also once per volume.
            seed_mask = nucleus_mask if nucleus_mask.max() > 0 else cytoplasm_mask

            from scipy.ndimage import maximum_filter
            _p = tuple(int(v) for v in self.config.train_patch_size)
            _cover = maximum_filter(seed_mask, size=_p, mode='constant') > 0
            _bs = tuple(slice(p // 2, s - p + p // 2 + 1)
                        for p, s in zip(_p, seed_mask.shape))
            _vi = np.where(_cover[_bs])
            fg_coords = np.stack([v + s.start for s, v in zip(_bs, _vi)],
                                 axis=1).astype(np.int32)
            if len(fg_coords) == 0:  # fallback, as StarDist
                fg_coords = np.argwhere(seed_mask > 0).astype(np.int32)
            # training.object_crop_shares: bounding-box centres of the objects of each level, for crops centred
            # on one object picked uniformly (a small cyst as often as a large tumour).
            object_centres = None
            if getattr(self.config, 'object_crop_shares', None):
                from scipy.ndimage import find_objects
                object_centres = {role: np.array([[(sl.start + sl.stop) // 2 for sl in box] for box in find_objects(mask)
                                                  if box is not None], np.int32).reshape(-1, nd)
                                  for role, mask in (('nucleus', nucleus_mask), ('cytoplasm', cytoplasm_mask))}
            # Levels of the run annotated in this file (all of them unless training.partial_annotation).
            annotated = tuple(name for name, f in (('nucleus', self.nucleus_files[idx]), ('cytoplasm', self.cytoplasm_files[idx]))
                              if f is not None)
            if not getattr(self, 'paired', True) or len(annotated) < 2:
                assignments = {}
            elif self.has_assignments:
                assignment_file = (self.assignments_dir /
                                   (Path(self.image_files[idx].name).stem + '.json'))
                if assignment_file.exists():
                    with open(assignment_file, 'r') as f:
                        assignments = {int(k): int(v) for k, v in json.load(f).items()}
                else:
                    max_dist = getattr(self.config, 'max_assignment_distance', 100.0)
                    assignments, _, _ = compute_nucleus_cell_assignments(
                        nucleus_mask, cytoplasm_mask, max_distance=max_dist)
            else:
                max_dist = getattr(self.config, 'max_assignment_distance', 100.0)
                assignments, _, _ = compute_nucleus_cell_assignments(
                    nucleus_mask, cytoplasm_mask, max_distance=max_dist)

            self.data_pool.append({
                'image': img,
                'nucleus_mask': nucleus_mask,
                'cytoplasm_mask': cytoplasm_mask,
                'nucleus_unknown': nucleus_unknown,
                'cytoplasm_unknown': cytoplasm_unknown,
                'fg_coords': fg_coords,
                'object_centres': object_centres,
                'assignments': assignments,
                'filename': self.image_files[idx].name,
                'annotated': annotated,
                'object_count': int(np.count_nonzero(np.unique(nucleus_mask))) +
                                int(np.count_nonzero(np.unique(cytoplasm_mask)))
            })

        print(f'  Loaded {len(self.data_pool)} samples into pool '
              f'(normalised and indexed once)')
        self._pool_seed = seed
        counts = np.array([s['object_count'] for s in self.data_pool], np.float64)
        self._tile_weights = counts / counts.sum() if counts.sum() > 0 else None
        # Partial annotation: tiles grouped by their annotated levels; a batch stays inside one group.
        self._annotation_groups = None
        if getattr(self, 'partial', False):
            self._annotation_groups = {}
            for index, sample in enumerate(self.data_pool):
                self._annotation_groups.setdefault(sample['annotated'], []).append(index)
            print('  Annotated levels of the training files: ' +
                  ', '.join(f"{'+'.join(k)}: {len(v)}" for k, v in sorted(self._annotation_groups.items())))

    def _pick_tile(self, rng):
        """Pool tile of one training crop: uniform, or with probability
        training.tile_sampling_by_objects in proportion to the objects it holds.

        A crop covers most of a tile, so the tile decides what a step sees. With uniform tiles
        the few dense ones, which hold most objects, get as many steps as nearly empty ones.
        """
        share = float(getattr(self.config, 'tile_sampling_by_objects', 0.0))
        weights = getattr(self, '_tile_weights', None)
        if share > 0 and weights is not None and rng.random() < share:
            return int(rng.choice(len(self.data_pool), p=weights))
        return int(rng.choice(len(self.data_pool)))

    def _load_regions(self):
        """Whole source volumes for training crops at any position inside the train blocks
        (dataset_prep/add_train_regions.py). The tile pool stays: geometry and the train-side
        metric read it. Images are kept as stored and normalised per crop."""
        from scipy.ndimage import distance_transform_edt, find_objects
        from cytonucs_runtime import read_model_image
        root = self.dataset_dir.parent / 'train_regions'
        index = root / 'regions.json'
        if not index.is_file():
            raise FileNotFoundError(f'training.train_region_crops needs {index}: '
                                    'run dataset_prep/add_train_regions.py on the dataset')
        nd = int(self.config.ndim)
        patch = tuple(int(v) for v in self.config.train_patch_size)
        self.region_pool = []
        for name, entry in json.loads(index.read_text()).items():
            channels = int(getattr(self.config, 'n_image_channels', self.config.n_channel_in))
            image = read_model_image(root / 'images' / name, nd, channels)
            if nd == 3 and image.ndim == 4 and image.shape[-1] != channels and image.shape[1] == channels:
                image = np.moveaxis(image, 1, -1)
            # Levels of the run annotated in this volume: all of them, or with
            # training.partial_annotation those whose mask file exists.
            folder = lambda level: root / (self.role_levels[level] + '_masks') / name
            annotated = tuple(level for level in ('nucleus', 'cytoplasm') if level in self.levels
                              and (not self.partial or folder(level).is_file()))
            if not annotated:
                raise ValueError(f'{name}: no masks of any level in {root}')
            # Label IDs of the source volume are kept (assignments refer to them), in the tiles' dtype.
            nucleus, cytoplasm = (
                m.astype(np.uint16) if 0 <= m.min() and m.max() <= np.iinfo(np.uint16).max else m
                for m in (np.asarray(imageio.volread(folder(level))) if level in annotated
                          else np.zeros(image.shape[:nd], np.uint16) for level in ('nucleus', 'cytoplasm')))
            nucleus, nucleus_unknown = split_unknown(nucleus, getattr(self.config, 'unknown_label', None))
            cytoplasm, cytoplasm_unknown = split_unknown(cytoplasm, getattr(self.config, 'unknown_label', None))
            pairs = {}
            if self.paired and len(annotated) == 2:
                stored = root / self.assignments_name / (Path(name).stem + '.json')
                pairs = (json.loads(stored.read_text()) if stored.is_file() else compute_nucleus_cell_assignments(
                    nucleus, cytoplasm, max_distance=getattr(self.config, 'max_assignment_distance', 100.0))[0])
            region = np.zeros(nucleus.shape[-2:], bool)
            for y0, y1, x0, x1 in entry['train_blocks']:
                region[y0:y1, x0:x1] = True
            allowed = allowed_crop_starts(region, patch[-2:])
            if not allowed.any():
                raise ValueError(f'{name}: no {patch[-2:]} crop fits into the train blocks')
            # Bounding-box centres of the cells (nuclei where a volume has no cell) inside the train blocks.
            labels = cytoplasm if cytoplasm.max() > 0 else nucleus
            boxes = find_objects(labels)
            anchors = [([(sl.start + sl.stop) // 2 for sl in box], label)
                       for label, box in enumerate(boxes, 1) if box is not None]
            anchors = [(c, label) for c, label in anchors if region[c[-2], c[-1]]]
            # Sparse annotation: anchors none of whose voxels has an unknown level
            # (training.annotated_anchor_share of the object-centred crops start from them).
            unknown = [u for u in (nucleus_unknown, cytoplasm_unknown) if u is not None]
            annotated_anchors = None
            if unknown:
                unknown = np.logical_or.reduce(unknown)
                annotated_anchors = np.array([i for i, (_, label) in enumerate(anchors) if not
                                              unknown[boxes[label - 1]][labels[boxes[label - 1]] == label].any()], np.int64)
            self.region_pool.append(dict(
                image=image, nucleus_mask=nucleus, cytoplasm_mask=cytoplasm, filename=name, annotated=annotated,
                nucleus_unknown=nucleus_unknown, cytoplasm_unknown=cytoplasm_unknown, annotated_anchors=annotated_anchors,
                assignments={int(k): int(v) for k, v in pairs.items()},
                blocks=[tuple(b) for b in entry['train_blocks']], table=window_sum_table(region),
                starts=np.flatnonzero(allowed), starts_shape=allowed.shape,
                nearest=distance_transform_edt(~allowed, return_distances=False, return_indices=True).astype(np.int32),
                anchors=np.array([c for c, _ in anchors], np.int64).reshape(-1, nd),
                anchor_labels=np.array([label for _, label in anchors], np.int64), percentiles={},
                # Counted here once: the first batches come from several prefetch threads.
                _label_counts=dict(nucleus_mask=label_counts(nucleus), cytoplasm_mask=label_counts(cytoplasm))))
            print(f'  Train region {name}: {len(entry["train_blocks"])} block(s), {len(anchors)} anchor object(s), '
                  f'{int(allowed.sum())} crop positions in XY' +
                  ('' if annotated_anchors is None else f', {len(annotated_anchors)} anchor(s) without unknown annotation'))
        counts = np.array([len(s['anchors']) for s in self.region_pool], np.float64)
        self._region_weights = counts / counts.sum() if counts.sum() > 0 else None

    def _region_crop(self, sample, patch, rng):
        """Start of one patch-sized training crop inside the train blocks, and its anchor cell.

        With probability fg_crop_prob the crop is centred on a random object, shifted by up to
        1/16 of the patch (the refinement ROI at inference is centred on its prompt up to the
        snapping step); else it is any allowed position. A start outside the train blocks moves
        to the nearest allowed one.
        """
        nd = len(patch)
        spatial = sample['nucleus_mask'].shape
        free = [max(spatial[a] - patch[a], 0) for a in range(nd)]
        if len(sample['anchors']) and rng.random() < getattr(self.config, 'fg_crop_prob', 0.9):
            annotated = sample.get('annotated_anchors')
            share = float(getattr(self.config, 'annotated_anchor_share', 0.0))
            if share > 0 and annotated is not None and len(annotated) and rng.random() < share:
                k = int(annotated[int(rng.integers(len(annotated)))])
            else:
                k = int(rng.integers(len(sample['anchors'])))
            center = sample['anchors'][k] + [int(rng.integers(-(p // 16), p // 16 + 1)) for p in patch]
            starts = [int(np.clip(center[a] - patch[a] // 2, 0, free[a])) for a in range(nd)]
            starts[-2:] = [int(v) for v in sample['nearest'][:, starts[-2], starts[-1]]]
            return starts, int(sample['anchor_labels'][k])
        starts = [int(rng.integers(0, free[a] + 1)) for a in range(nd - 2)]
        y, x = np.unravel_index(int(sample['starts'][int(rng.integers(len(sample['starts'])))]), sample['starts_shape'])
        return starts + [int(y), int(x)], 0

    def _region_normalize(self, sample, crop, starts, patch):
        """Percentile normalisation of a region crop with the statistics of the tile around it:
        the train block that holds the crop centre, over the Z range of the crop. Train and
        validation tiles are normalised per tile (one block, one Z slab) in the same way."""
        from csbdeep.utils import normalize_mi_ma
        nd = len(patch)
        cy, cx = starts[-2] + patch[-2] // 2, starts[-1] + patch[-1] // 2
        block = next(i for i, (y0, y1, x0, x1) in enumerate(sample['blocks']) if y0 <= cy < y1 and x0 <= cx < x1)
        key = (block,) + tuple(starts[:-2])
        if key not in sample['percentiles']:
            y0, y1, x0, x1 = sample['blocks'][block]
            context = sample['image'][tuple(slice(s, s + p) for s, p in zip(starts[:-2], patch[:-2])) +
                                      (slice(y0, y1), slice(x0, x1))]
            sample['percentiles'][key] = percentile_range(context, axis=tuple(range(nd)))
        mi, ma = sample['percentiles'][key]
        return np.ascontiguousarray(normalize_mi_ma(crop.astype(np.float32), mi, ma), dtype=np.float32)

    def get_cell_properties(self, prop='cells_median_extent_p'):
        """
        Returns cell properties from metadata.
        Used to compute anisotropy.
        """
        if self.meta is None:
            return None

        extents = []
        for s in self.meta:
            if prop in s:
                extents.append(s[prop])

        return np.array(extents) if extents else None

    def __len__(self):
        """
        Batches per epoch: steps_per_epoch for training (every call draws a new batch of random crops, also from a
        pool of one or two volumes); a deterministic pass for validation.
        """
        if self.subset == 'train':
            return max(1, int(getattr(self.config, 'train_steps_per_epoch', 1)))
        return max(1, len(self.data_pool))

    def __getitem__(self, idx):
        """Crop first, then augment the patch (the image comes from the pool already normalised).

        The K prompts of a crop go into the channels of that crop, not into copies of the batch: the backbone runs
        once and the refinement branch K times with shared weights.

        K = train_batch_size, as the model reads it. The channel order [img..., prompt_nuc_0, prompt_cyto_0,
        prompt_nuc_1, ...] matches the Lambda slices of _build_model.

        The *_target arrays are stacked into a batch of K*B in the order k*B + b, as the model concatenates the K
        refinement outputs.
        """
        bs = int(self.config.train_batch_size)
        nd = int(self.config.ndim)
        patch = tuple(int(p) for p in self.config.train_patch_size)
        zoom_axes = (1, 2) if nd == 3 else (0, 1)
        is_train = (self.subset == 'train')
        n_pr = max(1, int(getattr(self.config, 'prompts_per_crop', bs)))

        # a local generator: __getitem__ runs in several prefetch threads, which would share the global np.random
        pick = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch if is_train else 0, int(idx)]))
        regions = getattr(self, 'region_pool', None) if is_train else None
        if regions:
            crop_indices = [int(pick.choice(len(regions), p=self._region_weights))
                            for _ in range(int(getattr(self.config, 'image_batch_size', 1)))]
            if getattr(self, 'partial', False):                 # one batch, one set of annotated levels
                same = [i for i, r in enumerate(regions) if r['annotated'] == regions[crop_indices[0]]['annotated']]
                crop_indices = [i if i in same else int(pick.choice(same)) for i in crop_indices]
        elif is_train:
            crop_indices = [CytoNucsDataGenerator._pick_tile(self, pick)
                            for _ in range(int(getattr(self.config, 'image_batch_size', 1)))]
            groups = getattr(self, '_annotation_groups', None)
            if groups:
                # One batch, one set of annotated levels: the loss of a level is switched per batch.
                same = groups[self.data_pool[crop_indices[0]]['annotated']]
                crop_indices = [i if i in same else int(pick.choice(same)) for i in crop_indices]
        else:
            crop_indices = [int(self.indices[int(idx) % max(1, len(self.indices))])]

        X_batch = []
        y_batch = {'nucleus_instances': [], 'cytoplasm_instances': []}
        # k-major: tg_n_k[k][b]
        tg_n_k = [[] for _ in range(n_pr)]
        tg_c_k = [[] for _ in range(n_pr)]
        # training.unknown_label: '<role>_known' voxel maps and, per prompt, whether its target is known
        # (an empty target where the annotation is unknown does not say "no object here").
        sparse = is_train and getattr(self.config, 'unknown_label', None) is not None
        tk_n_k = [[] for _ in range(n_pr)]
        tk_c_k = [[] for _ in range(n_pr)]

        for _ci, i in enumerate(crop_indices):
            # Deterministic validation: the same crops and prompts every epoch.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch if is_train else 0, int(idx), int(_ci), 1]))

            sample = regions[i] if regions else self.data_pool[i]
            img_full = sample['image']
            nuc_full = sample['nucleus_mask']
            cyto_full = sample['cytoplasm_mask']
            spatial = img_full.shape[:nd]

            # The zoom is drawn first, so that the crop gives exactly one patch after scaling (the augmenter does not
            # zoom).
            zr = getattr(self.config, 'zoom_range', (0.75, 1.35))
            f = 1.0
            if (is_train and self.augmenter is not None
                    and zr is not None and rng.random() < 0.5):
                f = float(rng.uniform(*zr))
                # Never invent reflected foreground with background labels.
                f = max(f, max(patch[a] / spatial[a] for a in zoom_axes))
            # augmentation.zoom_z_range: a separate stretch along Z (object depth in slices).
            fz = 1.0
            zoom_z = getattr(self.config, 'zoom_z_range', None)
            if nd == 3 and is_train and self.augmenter is not None and zoom_z is not None and rng.random() < 0.5:
                fz = max(float(rng.uniform(*zoom_z)), patch[0] / spatial[0])

            need = list(patch)
            if f != 1.0:
                # Rounding margin only for a real zoom: without zoom the +2 margin was
                # trimmed back by the central crop, so crops at the volume face lost its
                # outer XY row and objects there were always cut by one voxel in training.
                for a in zoom_axes:
                    need[a] = int(np.ceil(patch[a] / f)) + 2
            if fz != 1.0:
                need[0] = int(np.ceil(patch[0] / fz)) + 1
            need = [min(need[a], spatial[a]) for a in range(nd)]

            anchor = 0
            if regions:
                starts, anchor = self._region_crop(sample, patch, rng)
                if f != 1.0 or fz != 1.0:
                    # The larger crop of a zoom-out keeps its centre; where it would leave the
                    # train blocks or the volume, this crop is taken without zoom.
                    grown = [starts[a] - (need[a] - patch[a]) // 2 for a in range(nd)]
                    if (all(0 <= g and g + n <= s for g, n, s in zip(grown, need, spatial))
                            and window_inside(sample['table'], grown[-2:], need[-2:])):
                        starts = grown
                    else:
                        f, fz, need = 1.0, 1.0, [min(patch[a], spatial[a]) for a in range(nd)]
            else:
                # crop centre: on a random object with probability p_fg
                p_fg = getattr(self.config, 'fg_crop_prob', 0.9)
                center = None
                fg = sample['fg_coords']
                shares = getattr(self.config, 'object_crop_shares', None)
                if shares:
                    # training.object_crop_shares (instead of fg_crop_prob): centred on one object of the inner
                    # or the outer level, shifted by up to a quarter of the crop; else at any position.
                    u = rng.random()
                    role = 'nucleus' if u < shares[0] else 'cytoplasm' if u < shares[0] + shares[1] else None
                    centres = sample['object_centres'][role] if role else ()
                    if len(centres):
                        center = centres[int(rng.integers(len(centres)))] + [
                            int(rng.integers(-(p // 4), p // 4 + 1)) for p in need]
                elif rng.random() < p_fg and len(fg) > 0:
                    center = fg[int(rng.integers(len(fg)))][:nd]

                starts = []
                for a in range(nd):
                    sh, p = spatial[a], need[a]
                    if sh <= p:
                        starts.append(0)
                    elif center is not None:
                        # center is a valid patch centre (as StarDist), not an object voxel
                        starts.append(max(0, min(int(center[a]) - p // 2, sh - p)))
                    else:
                        starts.append(int(rng.integers(0, sh - p + 1)))
            sl = tuple(slice(s, s + n) for s, n in zip(starts, need))

            if regions:
                img = self._region_normalize(sample, img_full[sl], starts, need)
            else:
                img = np.ascontiguousarray(img_full[sl])
            nucleus_mask = np.ascontiguousarray(nuc_full[sl])
            cytoplasm_mask = np.ascontiguousarray(cyto_full[sl])
            # Objects cut by this crop: fewer voxels than in the source volume. A crop
            # face lying on the volume face cuts nothing more than inference sees.
            # IDs survive zoom (order 0), flips and rotations, unlike face positions.
            counts = sample.setdefault('_label_counts', {})
            for key, full in (('nucleus_mask', nuc_full), ('cytoplasm_mask', cyto_full)):
                if key not in counts:
                    counts[key] = label_counts(full)
            crop_cut = {'nucleus': partial_labels(nucleus_mask, counts['nucleus_mask']),
                        'cytoplasm': partial_labels(cytoplasm_mask, counts['cytoplasm_mask'])}
            # Voxels of unknown annotation go through zoom and augmentation as label -1.
            unknown = {role: None if sample.get(role + '_unknown') is None else sample[role + '_unknown'][sl]
                       for role in ('nucleus', 'cytoplasm')}
            dtypes = (nucleus_mask.dtype, cytoplasm_mask.dtype)
            if unknown['nucleus'] is not None:
                nucleus_mask = with_unknown(nucleus_mask, unknown['nucleus'])
            if unknown['cytoplasm'] is not None:
                cytoplasm_mask = with_unknown(cytoplasm_mask, unknown['cytoplasm'])

            # zoom of the patch
            if f != 1.0 or fz != 1.0:
                from scipy.ndimage import zoom as ndzoom
                zf = [1.0] * nd
                for a in zoom_axes:
                    zf[a] = f
                if nd == 3:
                    zf[0] = fz
                zf_img = tuple(zf) + ((1.0,) if img.ndim > nd else ())
                img = ndzoom(img, zf_img, order=1)
                nucleus_mask = ndzoom(nucleus_mask, tuple(zf), order=0)
                cytoplasm_mask = ndzoom(cytoplasm_mask, tuple(zf), order=0)

            # Elastic deformation in XY (augmentation.elastic_prob): one smooth displacement for every
            # slice, fading out at the crop faces; labels (unknown voxels packed as -1) by nearest neighbour.
            elastic = float(getattr(self.config, 'elastic_prob', 0.0))
            if is_train and self.augmenter is not None and elastic > 0 and rng.random() < elastic:
                dy, dx = elastic_displacement(nucleus_mask.shape[-2:], float(self.config.elastic_alpha),
                                              float(self.config.elastic_sigma), rng)
                img = warp_yx(img, dy, dx, 1, nd)
                nucleus_mask = warp_yx(nucleus_mask, dy, dx, 0, nd)
                cytoplasm_mask = warp_yx(cytoplasm_mask, dy, dx, 0, nd)

            # augmentation of the patch (flips, transposition, intensity)
            if self.augmenter is not None:
                if self.augmenter in (cytonucs_augmenter_3d, cytonucs_augmenter_2d):
                    img, nucleus_mask, cytoplasm_mask = self.augmenter(
                        img, nucleus_mask, cytoplasm_mask, rng=rng, config=self.config)
                else:
                    img, nucleus_mask, cytoplasm_mask = self.augmenter(img, nucleus_mask, cytoplasm_mask)

            if unknown['nucleus'] is not None:
                nucleus_mask, unknown['nucleus'] = without_unknown(nucleus_mask, dtypes[0])
            if unknown['cytoplasm'] is not None:
                cytoplasm_mask, unknown['cytoplasm'] = without_unknown(cytoplasm_mask, dtypes[1])

            # centre crop / padding to the patch size
            cur = img.shape[:nd]
            # Axes whose patch faces cut objects that continue in the source volume.
            # A face equal to the volume face (e.g. full Z) truncates GT and image
            # identically at inference, so it is not artificial. rot90 may swap
            # the zoom axes, hence one shared flag for them.
            cut = [need[a] < spatial[a] or cur[a] > patch[a] for a in range(nd)]
            if any(cut[a] for a in zoom_axes):
                for a in zoom_axes:
                    cut[a] = True
            beg = [max(0, (cur[a] - patch[a]) // 2) for a in range(nd)]
            sl2 = tuple(slice(beg[a], beg[a] + min(cur[a], patch[a]))
                        for a in range(nd))
            img = img[sl2]
            central = any(cur[a] > patch[a] for a in range(nd))
            if central:
                before_central = (label_counts(nucleus_mask), label_counts(cytoplasm_mask))
            nucleus_mask = nucleus_mask[sl2]
            cytoplasm_mask = cytoplasm_mask[sl2]
            unknown = {role: None if u is None else u[sl2] for role, u in unknown.items()}
            if central:
                crop_cut['nucleus'] |= partial_labels(nucleus_mask, before_central[0])
                crop_cut['cytoplasm'] |= partial_labels(cytoplasm_mask, before_central[1])

            if any(img.shape[a] < patch[a] for a in range(nd)):
                pw = [(0, patch[a] - img.shape[a]) for a in range(nd)]
                pw_img = pw + [(0, 0)] if img.ndim > nd else pw
                img = np.pad(img, pw_img, mode='constant')
                nucleus_mask = np.pad(nucleus_mask, pw, mode='constant')
                cytoplasm_mask = np.pad(cytoplasm_mask, pw, mode='constant')
                unknown = {role: None if u is None else np.pad(u, pw, mode='constant') for role, u in unknown.items()}

            # The StarDist targets depend on the masks only: computed once per crop
            _pn, _dn = compute_stardist_gt(nucleus_mask, self.config,
                                           self._nuc_grid, rays=self._nuc_rays,
                                           aniso=self._nuc_aniso,
                                           normalize_prob=self.config.nucleus_prob_grid_normalize)
            _pc, _dc = compute_stardist_gt(cytoplasm_mask, self.config,
                                           self._cyto_grid, rays=self._cyto_rays,
                                           aniso=self._cyto_aniso)

            img_ch_base = img[..., np.newaxis] if img.ndim == nd else img
            pair_channels = int(getattr(self.config, 'n_pair_channels', 0) or 0)
            if pair_channels:
                # Hierarchy of three or more levels: one constant channel per adjacent pair says
                # which pair the two roles hold. Added after augmentation, right behind the image.
                code = np.zeros(img_ch_base.shape[:nd] + (pair_channels,), np.float32)
                code[..., int(self.config.pair_index)] = 1.
                img_ch_base = np.concatenate([img_ch_base, code], axis=-1)

            def _prompt_at(shape, c, extent, rng_=rng, augment=True, owner=None):
                # PROMPT AUGMENTATION. In training a prompt is built from the exact centroid and bounding box of a
                # ground-truth object, at inference from a probability peak with a sigma from the predicted rays; the
                # noise prepares the refinement for that shift. It is a fraction of the object size, so it does not
                # depend on the scale of the data.
                ext = np.asarray(extent, np.float32)
                shift = float(getattr(self.config, 'prompt_center_jitter', 0.1))
                sscale = float(getattr(self.config, 'prompt_sigma_jitter', 0.1))
                c = np.asarray(c, np.float32)
                uniform = float(getattr(self.config, 'prompt_uniform_prob', 0.0))
                if (is_train and augment and owner is not None and uniform > 0
                        and rng_.random() < uniform):
                    # Anywhere inside the object (as SAM samples its training points): detector peaks
                    # at inference may lie off-centre, and the mask must still be the whole object.
                    inside = np.argwhere(owner)
                    c = inside[int(rng_.integers(len(inside)))].astype(np.float32)
                else:
                    if is_train and augment and shift > 0:
                        c = c + rng_.normal(0.0, shift,
                                            size=c.shape).astype(np.float32) * ext
                    if self.config.prompt_inside_target and owner is not None:
                        c = project_prompt_center(c, owner, ext)
                sig = np.maximum(ext / 4.0, 1.0)
                if is_train and augment and sscale > 0:
                    sig = sig * float(rng_.uniform(1.0 - sscale, 1.0 + sscale))
                # Open axes: identical values to a full meshgrid, broadcast in the sum.
                grids = np.ix_(*[np.arange(s, dtype=np.float32) for s in shape])
                d2 = sum(((g - c[a]) / sig[a]) ** 2 for a, g in enumerate(grids))
                return np.exp(-0.5 * d2).astype(np.float32)

            # Repeated prompts on one cell (repeat_cell_prompt_prob) give pairs of prompts for the same-instance
            # consistency term.
            _asg = sample.get('assignments') or {}
            cyto_ids = np.unique(cytoplasm_mask)
            cyto_ids = list(cyto_ids[cyto_ids > 0])
            if len(cyto_ids) > 1:
                rng.shuffle(cyto_ids)
            if anchor in cyto_ids:
                # The crop is centred on this cell: it is the first prompt, its neighbours follow.
                cyto_ids.remove(anchor)
                cyto_ids.insert(0, anchor)
            nuc_present = {int(v) for v in np.unique(nucleus_mask) if v > 0}

            # Compute only when enabled; preserve the old RNG sequence at zero.
            neighbor_ids = []
            if self.config.neighbor_cell_prompt_prob and len(cyto_ids) > 1:
                neighbor_ids = nearest_cell_neighbors(cytoplasm_mask, cyto_ids[0],
                                                       self.config.anisotropy)
            prompts = []
            for _pi in range(n_pr):
                if cyto_ids:
                    repeat = _pi > 0 and rng.random() < self.config.repeat_cell_prompt_prob
                    chosen_cyto = int(cyto_ids[0 if repeat else _pi % len(cyto_ids)])
                    if (_pi > 0 and not repeat and neighbor_ids and
                            rng.random() < self.config.neighbor_cell_prompt_prob):
                        chosen_cyto = int(neighbor_ids[(_pi - 1) % len(neighbor_ids)])
                else:
                    chosen_cyto = 0

                if chosen_cyto:
                    # nuclei assigned to this cell and present in the patch
                    nuc_ids = [n for n, c in _asg.items()
                               if int(c) == chosen_cyto and int(n) in nuc_present]
                    if not nuc_ids:  # fallback: by overlap
                        nuc_ids = [int(v) for v in
                                   np.unique(nucleus_mask[cytoplasm_mask == chosen_cyto])
                                   if v > 0]
                else:
                    nuc_ids = sorted(nuc_present)
                chosen_nuc = int(rng.choice(nuc_ids)) if nuc_ids else 0

                tg_n = np.zeros(nucleus_mask.shape, np.float32)
                tg_c = np.zeros(cytoplasm_mask.shape, np.float32)
                pr = np.zeros(nucleus_mask.shape, np.float32)

                if chosen_nuc:
                    mn = nucleus_mask == chosen_nuc
                    tg_n[mn] = 1.0
                    pts = np.argwhere(mn)
                    pr = _prompt_at(nucleus_mask.shape, pts.mean(axis=0),
                                    pts.max(axis=0) - pts.min(axis=0) + 1, owner=mn)
                elif chosen_cyto:
                    pts = np.argwhere(cytoplasm_mask == chosen_cyto)
                    pr = _prompt_at(cytoplasm_mask.shape, pts.mean(axis=0),
                                    pts.max(axis=0) - pts.min(axis=0) + 1, owner=cytoplasm_mask == chosen_cyto)

                if chosen_cyto:
                    tg_c[cytoplasm_mask == chosen_cyto] = 1.0

                pr_c = pr
                if chosen_cyto and rng.random() < float(getattr(self.config, 'cyto_center_prompt_prob', 0.25)):
                    pts = np.argwhere(cytoplasm_mask == chosen_cyto)
                    pr_c = _prompt_at(cytoplasm_mask.shape, pts.mean(axis=0),
                                      pts.max(axis=0)-pts.min(axis=0)+1, owner=cytoplasm_mask == chosen_cyto)
                if is_train and rng.random() < self.config.negative_prompt_prob:
                    # Like false detector peaks at inference: any non-nuclear voxel
                    # (background or cytoplasm), nucleus-sized. The nuclear target is
                    # empty; the cell head still owns the cell containing the prompt.
                    # augmentation.negative_prompt_inside_outer: that share of them at a voxel of
                    # an outer object no inner object covers (hard negatives: on KiTS the
                    # detector fires on kidney tissue that looks like a mass).
                    inside = self.config.negative_prompt_inside_outer
                    hard = inside_outer_point(nucleus_mask, cytoplasm_mask, unknown, rng) if (
                        inside and rng.random() < inside) else None
                    for _attempt in range(64):
                        center_neg = hard if hard is not None else tuple(
                            int(rng.integers(s)) for s in nucleus_mask.shape)
                        if nucleus_mask[center_neg] == 0 and not any(u is not None and u[center_neg] for u in unknown.values()):
                            if nuc_present:
                                sized = np.argwhere(nucleus_mask == int(rng.choice(sorted(nuc_present))))
                                extent = sized.max(0) - sized.min(0) + 1
                            else:
                                extent = np.maximum(np.array(patch) / 8, 1)
                            pr = _prompt_at(nucleus_mask.shape, center_neg, extent, augment=False)
                            pr_c = pr
                            tg_n.fill(0)
                            cell = int(cytoplasm_mask[center_neg])
                            tg_c[:] = cytoplasm_mask == cell if cell else 0
                            break
                prompts.append((pr, pr_c))
                tg_n_k[_pi].append(tg_n)
                tg_c_k[_pi].append(tg_c)

            # Explicit axial pair: both cell prompts supervise the SAME entire
            # GT object. Keep nuclear prompts/targets independent and intact.
            z_pair = False
            if (self.config.cell_z_pair_prob and nd == 3 and n_pr >= 2 and cyto_ids
                    and rng.random() < self.config.cell_z_pair_prob):
                owner = cytoplasm_mask == cyto_ids[0]
                centers = axial_prompt_centers(owner)
                if centers is not None:
                    points = np.argwhere(owner)
                    extent = points.max(0) - points.min(0) + 1
                    for k, center in enumerate(centers):
                        prompts[k] = (prompts[k][0], _prompt_at(owner.shape, center, extent,
                                                               augment=False, owner=owner))
                        tg_c_k[k][-1] = owner.astype(np.float32)
                    z_pair = True
            # The trainer must not replace these paired cell prompts by peaks.
            y_batch.setdefault('cytoplasm_z_prompt_pair', []).append(z_pair)

            nuclear_z_pair = False
            if (self.config.nucleus_z_pair_prob and nd == 3 and n_pr >= 2 and nuc_present
                    and rng.random() < self.config.nucleus_z_pair_prob):
                # One GT nucleus, never the union of nuclei belonging to a cell.
                lab = int(rng.choice(sorted(nuc_present)))
                owner = nucleus_mask == lab
                centers = axial_prompt_centers(owner)
                if centers is not None:
                    points = np.argwhere(owner)
                    extent = points.max(0) - points.min(0) + 1
                    for k, center in enumerate(centers):
                        prompts[k] = (_prompt_at(owner.shape, center, extent,
                                                 augment=False, owner=owner), prompts[k][1])
                        tg_n_k[k][-1] = owner.astype(np.float32)
                    nuclear_z_pair = True
            y_batch.setdefault('nucleus_z_prompt_pair', []).append(nuclear_z_pair)

            if sparse:
                # An empty target is known only if no voxel of unknown annotation lies within about one
                # sigma of its prompt (a cell whose nuclei are not annotated says nothing about nuclei).
                for k, (pr_n, pr_c) in enumerate(prompts):
                    for role, prompt, targets, flags in (('nucleus', pr_n, tg_n_k, tk_n_k), ('cytoplasm', pr_c, tg_c_k, tk_c_k)):
                        u = unknown[role]
                        flags[k].append(float(u is None or targets[k][-1].any() or not u[prompt > .6].any()))
                for role, u in unknown.items():
                    y_batch.setdefault(role + '_known', []).append(
                        np.ones(nucleus_mask.shape, np.float32) if u is None else (~u).astype(np.float32))

            # [img..., pr_nuc_0, pr_cyto_0, pr_nuc_1, pr_cyto_1, ...]
            img_ch = np.concatenate(
                [img_ch_base] + [p[..., np.newaxis] for pair in prompts for p in pair],
                axis=-1)

            X_batch.append(img_ch)
            y_batch['nucleus_instances'].append(nucleus_mask)
            y_batch['cytoplasm_instances'].append(cytoplasm_mask)
            y_batch.setdefault('crop_truncation_axes', []).append(np.asarray(cut, np.float32))
            for head, labels in (('nucleus', nucleus_mask), ('cytoplasm', cytoplasm_mask)):
                y_batch.setdefault(head + '_crop_cut', []).append(
                    np.isin(labels, sorted(crop_cut[head])).astype(np.uint8))
            y_batch.setdefault('nucleus_prob', []).append(_pn)
            y_batch.setdefault('nucleus_dist', []).append(_dn)
            y_batch.setdefault('cytoplasm_prob', []).append(_pc)
            y_batch.setdefault('cytoplasm_dist', []).append(_dc)

        if is_train and getattr(self, 'partial', False):
            # 1 where the level's loss applies to this batch: annotated here, or not a level of the run.
            annotated = sample.get('annotated', self.levels)
            y_batch['levels_present'] = np.array([float(name not in self.levels or name in annotated)
                                                  for name in ('nucleus', 'cytoplasm')], np.float32)
        X_batch = np.stack(X_batch)
        for _k in ('nucleus_instances', 'cytoplasm_instances',
                   'nucleus_prob', 'nucleus_dist', 'cytoplasm_prob', 'cytoplasm_dist',
                   'cytoplasm_z_prompt_pair', 'nucleus_z_prompt_pair', 'crop_truncation_axes',
                   'nucleus_crop_cut', 'cytoplasm_crop_cut'):
            y_batch[_k] = np.stack(y_batch[_k])
        # k*B + b: the order in which the model concatenates the K refinement outputs
        y_batch['nucleus_target'] = np.concatenate(
            [np.stack(v) for v in tg_n_k], axis=0)
        y_batch['cytoplasm_target'] = np.concatenate(
            [np.stack(v) for v in tg_c_k], axis=0)
        if sparse:
            for role, flags in (('nucleus', tk_n_k), ('cytoplasm', tk_c_k)):
                y_batch[role + '_known'] = np.stack(y_batch[role + '_known'])
                y_batch[role + '_target_known'] = np.concatenate([np.asarray(v, np.float32) for v in flags])
        return X_batch, y_batch

    def _visualize_gt_comparison(self, img, nucleus_mask_original, cytoplasm_mask_original,
                                 nucleus_mask_final, cytoplasm_mask_final,
                                 nucleus_dist_raw, cytoplasm_dist_raw, epoch):
        """
        Visualisation: original vs final ground truth
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from skimage.segmentation import find_boundaries
        from scipy.ndimage import center_of_mass
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        from stardist import random_label_cmap

        cmap = random_label_cmap()
        # Normalize image for display
        if img.ndim == self.config.ndim:
            img_display = img
        else:
            img_display = img[..., 0] if img.shape[-1] == 1 else img

        img_norm = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

        # === ROW 1: NUCLEUS COMPARISON ===

        # Original nucleus mask
        axes[0, 0].imshow(img_norm, cmap='gray')
        axes[0, 0].imshow(nucleus_mask_original, alpha=0.5, cmap=cmap)
        axes[0, 0].set_title(f'Nucleus Original GT ({len(np.unique(nucleus_mask_original)) - 1} objects)')
        axes[0, 0].axis('off')

        # Final nucleus mask (should be same)
        axes[0, 1].imshow(img_norm, cmap='gray')
        axes[0, 1].imshow(nucleus_mask_final, alpha=0.5, cmap=cmap)
        axes[0, 1].set_title(f'Nucleus Final ({len(np.unique(nucleus_mask_final)) - 1} objects)')
        axes[0, 1].axis('off')

        # Nucleus distance map (raw pixels)
        nucleus_dist_vis = nucleus_dist_raw.mean(axis=-1)  # Average over rays
        im = axes[0, 2].imshow(nucleus_dist_vis, cmap='hot')
        axes[0, 2].set_title(f'Nucleus Dist Map\n[{nucleus_dist_raw.min():.1f}, {nucleus_dist_raw.max():.1f}] px')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

        # Nucleus star-convex rays (sample one object)
        axes[0, 3].imshow(img_norm, cmap='gray')
        if len(np.unique(nucleus_mask_final)) > 1:
            # Pick object in center
            h, w = nucleus_mask_final.shape
            center_obj = nucleus_mask_final[h // 2, w // 2]
            if center_obj > 0:
                obj_mask = (nucleus_mask_final == center_obj)
                # Find centroid
                from scipy.ndimage import center_of_mass
                cy, cx = center_of_mass(obj_mask)
                cy, cx = int(cy), int(cx)

                # Get rays at this point
                rays_at_center = nucleus_dist_raw[cy, cx]

                # Draw rays
                rays_verts = self.rays.vertices[:, :2]  # 2D
                for i, (ry, rx) in enumerate(rays_verts):
                    dist = rays_at_center[i]
                    end_y = cy + ry * dist
                    end_x = cx + rx * dist
                    axes[0, 3].plot([cx, end_x], [cy, end_y], 'r-', linewidth=0.5, alpha=0.6)

                axes[0, 3].plot(cx, cy, 'yo', markersize=4)
                axes[0, 3].set_title(f'Nucleus Star Rays (obj {center_obj})')
        axes[0, 3].axis('off')

        # === ROW 2: CYTOPLASM COMPARISON ===

        # Original cytoplasm mask
        axes[1, 0].imshow(img_norm, cmap='gray')
        axes[1, 0].imshow(cytoplasm_mask_original, alpha=0.5, cmap=cmap)
        axes[1, 0].set_title(f'Cytoplasm Original GT ({len(np.unique(cytoplasm_mask_original)) - 1} objects)')
        axes[1, 0].axis('off')

        # Final cytoplasm mask (after watershed)
        axes[1, 1].imshow(img_norm, cmap='gray')
        axes[1, 1].imshow(cytoplasm_mask_final, alpha=0.5, cmap=cmap)
        axes[1, 1].set_title(f'Cytoplasm Final ({len(np.unique(cytoplasm_mask_final)) - 1} objects)')
        axes[1, 1].axis('off')

        # Cytoplasm distance map
        cytoplasm_dist_vis = cytoplasm_dist_raw.mean(axis=-1)
        im = axes[1, 2].imshow(cytoplasm_dist_vis, cmap='hot')
        axes[1, 2].set_title(f'Cytoplasm Dist Map\n[{cytoplasm_dist_raw.min():.1f}, {cytoplasm_dist_raw.max():.1f}] px')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

        # Cytoplasm star-convex rays
        axes[1, 3].imshow(img_norm, cmap='gray')
        if len(np.unique(cytoplasm_mask_final)) > 1:
            h, w = cytoplasm_mask_final.shape
            center_obj = cytoplasm_mask_final[h // 2, w // 2]
            if center_obj > 0:
                obj_mask = (cytoplasm_mask_final == center_obj)
                cy, cx = center_of_mass(obj_mask)
                cy, cx = int(cy), int(cx)

                rays_at_center = cytoplasm_dist_raw[cy, cx]
                rays_verts = self.rays.vertices[:, :2]

                for i, (ry, rx) in enumerate(rays_verts):
                    dist = rays_at_center[i]
                    end_y = cy + ry * dist
                    end_x = cx + rx * dist
                    axes[1, 3].plot([cx, end_x], [cy, end_y], 'g-', linewidth=0.5, alpha=0.6)

                axes[1, 3].plot(cx, cy, 'yo', markersize=4)
                axes[1, 3].set_title(f'Cytoplasm Star Rays (obj {center_obj})')
        axes[1, 3].axis('off')

        # === ROW 3: DIFFERENCES ===

        # Nucleus boundary comparison
        nuc_orig_boundary = find_boundaries(nucleus_mask_original, mode='thick')
        nuc_final_boundary = find_boundaries(nucleus_mask_final, mode='thick')

        overlay = np.zeros((*nucleus_mask_original.shape, 3), dtype=np.uint8)
        overlay[nuc_orig_boundary] = [255, 0, 0]  # Red = original
        overlay[nuc_final_boundary] = [0, 255, 0]  # Green = final

        axes[2, 0].imshow(img_norm, cmap='gray')
        axes[2, 0].imshow(overlay, alpha=0.7)
        axes[2, 0].set_title('Nucleus Boundaries\n(Red=Original, Green=Final)')
        axes[2, 0].axis('off')

        # Cytoplasm boundary comparison
        cyto_orig_boundary = find_boundaries(cytoplasm_mask_original, mode='thick')
        cyto_final_boundary = find_boundaries(cytoplasm_mask_final, mode='thick')

        overlay = np.zeros((*cytoplasm_mask_original.shape, 3), dtype=np.uint8)
        overlay[cyto_orig_boundary] = [255, 0, 0]  # Red = original
        overlay[cyto_final_boundary] = [0, 255, 0]  # Green = final

        axes[2, 1].imshow(img_norm, cmap='gray')
        axes[2, 1].imshow(overlay, alpha=0.7)
        axes[2, 1].set_title('Cytoplasm Boundaries\n(Red=Original, Green=Final)')
        axes[2, 1].axis('off')

        # Statistics
        from skimage.measure import regionprops

        nuc_orig_props = regionprops(nucleus_mask_original)
        nuc_final_props = regionprops(nucleus_mask_final)
        cyto_orig_props = regionprops(cytoplasm_mask_original)
        cyto_final_props = regionprops(cytoplasm_mask_final)

        stats_text = f"NUCLEUS:\n"
        stats_text += f"  Original: {len(nuc_orig_props)} objects\n"
        stats_text += f"  Final: {len(nuc_final_props)} objects\n"
        if len(nuc_final_props) > 0:
            nuc_areas = [p.area for p in nuc_final_props]
            stats_text += f"  Area: {np.mean(nuc_areas):.1f} ± {np.std(nuc_areas):.1f} px\n"

        stats_text += f"\nCYTOPLASM:\n"
        stats_text += f"  Original: {len(cyto_orig_props)} objects\n"
        stats_text += f"  Final: {len(cyto_final_props)} objects\n"
        if len(cyto_final_props) > 0:
            cyto_areas = [p.area for p in cyto_final_props]
            stats_text += f"  Area: {np.mean(cyto_areas):.1f} ± {np.std(cyto_areas):.1f} px\n"

        axes[2, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2, 2].axis('off')

        # Distance distribution comparison
        valid_nuc_orig = nucleus_dist_raw[nucleus_mask_original > 0].flatten()
        valid_cyto_final = cytoplasm_dist_raw[cytoplasm_mask_final > 0].flatten()

        axes[2, 3].hist(valid_nuc_orig, bins=50, alpha=0.5, label='Nucleus', color='red')
        axes[2, 3].hist(valid_cyto_final, bins=50, alpha=0.5, label='Cytoplasm', color='green')
        axes[2, 3].set_xlabel('Distance (pixels)')
        axes[2, 3].set_ylabel('Frequency')
        axes[2, 3].set_title('Distance Distribution')
        axes[2, 3].legend()
        axes[2, 3].grid(True, alpha=0.3)

        plt.suptitle(f'GT Data Comparison - Training Epoch {epoch}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        save_path = f'gt_comparison_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  📊 GT comparison saved: {save_path}")


    def on_epoch_end(self):
        # Trainer sets the epoch before creating its prefetch workers.
        pass

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.subset == 'train' and self.pool_size < len(self.image_files):
            # Resume within a pool period must reconstruct the same subset as
            # uninterrupted training, even when no repool boundary is crossed.
            pool_epoch = (self.epoch // self.repool_freq) * self.repool_freq
            pool_seed = self.seed + pool_epoch
            if getattr(self, '_pool_seed', None) != pool_seed:
                self.repool(seed=pool_seed)
                self.indices = np.arange(len(self.data_pool))


def cytonucs_augmenter_3d(img, nucleus_mask, cytoplasm_mask, rng=None, config=None):
    """Shape-preserving XY and intensity augmentation. Zoom belongs to the cropper."""
    rng = np.random.default_rng() if rng is None else rng
    if getattr(config, 'rotation_yx', True) and img.shape[1] == img.shape[2]:
        k = int(rng.integers(4))
        img, nucleus_mask, cytoplasm_mask = [np.rot90(a, k, axes=(1, 2))
                                            for a in (img, nucleus_mask, cytoplasm_mask)]
    for ax in (1, 2):
        if rng.random() < float(getattr(config, 'flip_prob', 0.5)):
            img, nucleus_mask, cytoplasm_mask = [np.flip(a, axis=ax)
                                                for a in (img, nucleus_mask, cytoplasm_mask)]
    # augmentation.flip_z_prob: flip along Z (depth order of the slices).
    flip_z = float(getattr(config, 'flip_z_prob', 0.0) or 0.0)
    if flip_z and rng.random() < flip_z:
        img, nucleus_mask, cytoplasm_mask = [np.flip(a, axis=0) for a in (img, nucleus_mask, cytoplasm_mask)]
    img = img * rng.uniform(*getattr(config, 'intensity_range', (0.8, 1.2)))
    img = img + rng.uniform(*getattr(config, 'intensity_offset', (-0.1, 0.1)))
    gamma = rng.uniform(*getattr(config, 'gamma_range', (0.9, 1.1)))
    img = np.sign(img) * np.abs(img) ** gamma
    # augmentation.saturation_prob: one channel clipped at a random high percentile of the crop, as a
    # saturated detector channel in a bright, dense region.
    saturation = float(getattr(config, 'saturation_prob', 0.0) or 0.0)
    if saturation and rng.random() < saturation:
        img = np.array(img, np.float32)
        channel = int(rng.integers(img.shape[-1])) if img.ndim == 4 else None
        plane = img[..., channel] if channel is not None else img
        level = np.percentile(plane, rng.uniform(80., 99.))
        if channel is not None:
            img[..., channel] = np.minimum(plane, level)
        else:
            img = np.minimum(plane, level)
    sigma = float(getattr(config, 'noise_sigma', 0.01))
    if sigma > 0:
        img = img + rng.normal(0, sigma, img.shape)
    blur = getattr(config, 'blur_sigma_range', None)
    if blur is not None:
        from scipy.ndimage import gaussian_filter
        sigma = float(rng.uniform(*blur))
        img = gaussian_filter(img, (0, sigma, sigma) + ((0,) if img.ndim == 4 else ()))
    return (np.ascontiguousarray(img, dtype=np.float32),
            np.ascontiguousarray(nucleus_mask), np.ascontiguousarray(cytoplasm_mask))


def cytonucs_augmenter_2d(img, nucleus_mask, cytoplasm_mask, rng=None, config=None):
    """Use the same local RNG and synchronized augmentation contract as 3D.

    Zoom/crop/padding is performed once by the generator, not a second time here.
    """
    result = cytonucs_augmenter_3d(img[None], nucleus_mask[None], cytoplasm_mask[None],
                                 rng=rng, config=config)
    return tuple(a[0] for a in result)


def prepare_cytonucs_dataset(
        raw_images_dir,
        raw_nucleus_masks_dir,
        raw_cytoplasm_masks_dir,
        output_dir,
        validation_images=None,
        test_split=0.1,
        val_split=0.1,
        compute_assignments=True,
        seed=42,
        train_patch_size=None,
        ndim=2
):
    """
    Prepare dataset for CytoNucs training (2D/3D, grayscale/RGB).
    """


    import cv2

    raw_images_dir = Path(raw_images_dir)
    raw_nucleus_masks_dir = Path(raw_nucleus_masks_dir)
    raw_cytoplasm_masks_dir = Path(raw_cytoplasm_masks_dir)
    output_dir = Path(output_dir)

    # ========== Helpers ==========

    def read_image_any_format(filepath, keep_rgb=False):
        """
        Reads an image:
        - 2D grayscale: (H, W)
        - 2D RGB/RGBA: (H, W, 3/4)
        - 3D grayscale: (Z, H, W)
        - 3D RGB/RGBA: (Z, H, W, 3/4)

        With keep_rgb=False:
        - 2D RGB -> grayscale
        - 3D RGB -> 3D grayscale, slice by slice
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext in [".tif", ".tiff"]:
            img = imageio.imread(str(filepath))
        elif ext in [".png", ".jpg", ".jpeg"]:
            img = imageio.imread(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # RGB / RGBA -> grayscale only with keep_rgb=False
        if not keep_rgb:
            # 2D RGB/RGBA: (H, W, 3/4)
            if img.ndim == 3 and img.shape[-1] in [3, 4]:
                if img.shape[-1] == 4:
                    img = img[..., :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # 3D RGB/RGBA stack: (Z, H, W, 3/4)
            elif img.ndim == 4 and img.shape[-1] in [3, 4]:
                if img.shape[-1] == 4:
                    img = img[..., :3]

                img = np.stack(
                    [cv2.cvtColor(slice_rgb, cv2.COLOR_RGB2GRAY) for slice_rgb in img],
                    axis=0
                )

        return img

    def save_image_smart(img, filepath, is_mask=False):
        """Written in the format of the sources: a plain multi-page TIFF without an ImageJ header ((Z, Y, X, 3)
        uint8, photometric RGB), which Fiji reads. Neither metadata={'axes': ...} (it replaces the ImageDescription)
        nor imagej=True (it turns RGB into a 16-bit ZCYX hyperstack) is used.
        """
        import tifffile
        filepath = str(filepath)
        img = np.asarray(img)

        if is_mask:
            tifffile.imwrite(filepath, img, photometric='minisblack')
            return

        if img.ndim == 2:
            tifffile.imwrite(filepath, img)
        elif img.ndim == 3 and img.shape[-1] in (3, 4):
            tifffile.imwrite(filepath, img[..., :3])
        elif img.ndim == 3:
            tifffile.imwrite(filepath, img)
        elif img.ndim == 4 and img.shape[-1] in (3, 4):
            tifffile.imwrite(filepath, img[..., :3])
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")



    def validate_instance_mask(mask, name):
        """Checks an instance mask."""
        unique_vals = np.unique(mask)

        if len(unique_vals) == 2 and 255 in unique_vals:
            print(f"      ⚠️  WARNING: {name} is semantic (0/255), not instance!")
            return False

        if len(unique_vals) > 200:
            print(f"      ⚠️  WARNING: {name} has {len(unique_vals)} values")

        return True

    def find_all_images(directory, base_name=None):
        """Finds the files."""
        extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']

        if base_name:
            for ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg']:
                candidate = directory / f"{base_name}.{ext}"
                if candidate.exists():
                    return candidate
            return None
        else:
            files = []
            for ext in extensions:
                files.extend(list(directory.glob(ext)))
            return sorted(set(files))

    # ========== Search and validation ==========

    print("Searching for images and masks...")
    image_files = find_all_images(raw_images_dir)

    if len(image_files) == 0:
        raise ValueError(f"No images found in {raw_images_dir}")

    print(f"Found {len(image_files)} images")

    valid_triplets = []
    for img_file in image_files:
        base_name = img_file.stem
        nuc_mask = find_all_images(raw_nucleus_masks_dir, base_name)
        cyto_mask = find_all_images(raw_cytoplasm_masks_dir, base_name)

        if nuc_mask and cyto_mask:
            valid_triplets.append((img_file, nuc_mask, cyto_mask))
        else:
            print(f"  ⚠️  Skipping {img_file.name}: missing masks")

    if len(valid_triplets) == 0:
        raise ValueError("No valid triplets found!")

    print(f"Valid triplets: {len(valid_triplets)}")

    # ========== Split ==========

    np.random.seed(seed)

    if validation_images and len(validation_images) > 0:
        val_triplets = [t for t in valid_triplets if t[0].name in validation_images]
        remaining = [t for t in valid_triplets if t[0].name not in validation_images]
        n_test = max(1, int(len(remaining) * test_split))
        indices = np.random.permutation(len(remaining))
        test_triplets = [remaining[i] for i in indices[:n_test]]
        train_triplets = [remaining[i] for i in indices[n_test:]]
    else:
        indices = np.random.permutation(len(valid_triplets))
        n_test = max(1, int(len(valid_triplets) * test_split))
        n_val = max(1, int(len(valid_triplets) * val_split))
        test_triplets = [valid_triplets[i] for i in indices[:n_test]]
        val_triplets = [valid_triplets[i] for i in indices[n_test:n_test + n_val]]
        train_triplets = [valid_triplets[i] for i in indices[n_test + n_val:]]

    print(f"\nSplit: Train={len(train_triplets)}, Val={len(val_triplets)}, Test={len(test_triplets)}")

    # ========== Output folders ==========

    for subset in ['train', 'val', 'test']:
        for subdir in ['images', 'nucleus_masks', 'cytoplasm_masks', 'assignments']:
            (output_dir / subset / subdir).mkdir(parents=True, exist_ok=True)

    # ========== Processing ==========
    rescale_factors = []
    for subset_name, triplets in [('train', train_triplets),
                                  ('val', val_triplets),
                                  ('test', test_triplets)]:
        print(f"\nProcessing {subset_name}...")

        for img_file, nuc_file, cyto_file in triplets:
            base_name = img_file.stem
            print(f"  {base_name}")

            # Reading (images keep their channels)
            img = read_image_any_format(img_file, keep_rgb=True)
            nucleus_mask = read_image_any_format(nuc_file, keep_rgb=False)
            cytoplasm_mask = read_image_any_format(cyto_file, keep_rgb=False)

            # ndim (2D RGB is 2D, not 3D)
            if img.ndim == 3 and img.shape[-1] == 3:
                ndim = 2  # 2D RGB
            else:
                ndim = img.ndim - (1 if img.ndim == 4 and img.shape[-1] in (3,4) else 0)

            print(f"    Detected: {ndim}D, img shape {img.shape}")

            rescale_factors.append(1.0)

            # Validation (no rescaling)
            validate_instance_mask(nucleus_mask, "nucleus")
            validate_instance_mask(cytoplasm_mask, "cytoplasm")

            # Saving
            output_filename = base_name + '.tiff'

            save_image_smart(
                img,
                output_dir / subset_name / 'images' / output_filename
            )
            save_image_smart(
                nucleus_mask,
                output_dir / subset_name / 'nucleus_masks' / output_filename, is_mask=True
            )
            save_image_smart(
                cytoplasm_mask,
                output_dir / subset_name / 'cytoplasm_masks' / output_filename, is_mask=True
            )

            # Assignments
            if compute_assignments:
                assignments, unassigned, stats = compute_nucleus_cell_assignments(
                    nucleus_mask, cytoplasm_mask, max_distance=50.0
                )

                assignment_file = output_dir / subset_name / 'assignments' / (base_name + '.json')
                with open(assignment_file, 'w') as f:
                    json.dump(assignments, f, indent=2)

                n_nuclei = len(np.unique(nucleus_mask)) - 1
                n_cells = len(np.unique(cytoplasm_mask)) - 1
                print(f"    ✓ {len(assignments)}/{n_nuclei} nuclei → {n_cells} cells")

                if len(unassigned) > 0:
                    print(f"    ⚠️  {len(unassigned)} unassigned: {unassigned}")

    # ========== Save split info ==========

    split_info = {
        'train': [t[0].name for t in train_triplets],
        'val': [t[0].name for t in val_triplets],
        'test': [t[0].name for t in test_triplets],
        'seed': seed,
        'test_split': test_split,
        'val_split': val_split if not validation_images else 'manual',
        'rescale_factor': float(np.median(rescale_factors)) if len(rescale_factors) > 0 else 1.0
    }
    # ========== Generate metadata ==========
    print("\nGenerating metadata...")

    from skimage.measure import regionprops

    for subset_name in ['train', 'val', 'test']:
        subset_dir = output_dir / subset_name
        image_files = sorted(
            p for ext in ('*.tif', '*.tiff') for p in (subset_dir / 'images').glob(ext))

        metadata = []

        for img_file in image_files:
            base_name = img_file.stem

            nuc_mask = imageio.volread(subset_dir / 'nucleus_masks' / img_file.name)
            cyto_mask = imageio.volread(subset_dir / 'cytoplasm_masks' / img_file.name)

            # Compute median extent
            all_extents = []

            for prop in regionprops(nuc_mask):
                bbox = prop.bbox
                if len(bbox) == 4:  # 2D
                    extent = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
                else:  # 3D
                    extent = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
                all_extents.append(extent)

            for prop in regionprops(cyto_mask):
                bbox = prop.bbox
                if len(bbox) == 4:  # 2D
                    extent = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
                else:  # 3D
                    extent = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
                all_extents.append(extent)

            if len(all_extents) > 0:
                median_extent = np.median(all_extents, axis=0).tolist()
            else:
                median_extent = None

            metadata.append({
                'filename': img_file.name,
                'cells_median_extent_p': median_extent,
                'rescale_factor': float(rescale_factor) if 'rescale_factor' in dir() else 1.0,
                'n_nuclei': int(nuc_mask.max()),
                'n_cells': int(cyto_mask.max())
            })

        # Save metadata
        with open(subset_dir / 'meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  {subset_name}: {len(metadata)} samples")
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print("✓ Dataset preparation complete!")
    print(f"✓ Output: {output_dir}")
    print(f"✓ Train: {len(train_triplets)}, Val: {len(val_triplets)}, Test: {len(test_triplets)}")
    print(f"{'=' * 60}")
