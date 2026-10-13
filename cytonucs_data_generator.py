"""
CytoNucs StarDist Data Generator

Loads paired nucleus + cytoplasm annotations.
Creates training batches with augmentation.
Supports repooling and automatic assignment creation.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from pathlib import Path
import imageio
from csbdeep.utils import normalize
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
        # Зум переносим в __getitem__ (по патчу, а не по полному тому).
        # ndzoom на 17x1024x1024x3 стоил секунды и вызывался на каждый сэмпл.

        # Load file lists
        def find_all_images(path):
            """Находит все изображения в разных форматах"""
            files = []
            # КРИТИЧНО: добавить все варианты написания расширений
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
                files.extend(list(path.glob(ext)))
            return sorted(set(files))

        self.image_files = find_all_images(self.dataset_dir / 'images')
        if not self.image_files:
            raise ValueError(f'No images found in {self.dataset_dir / "images"}')
        # A level the run does not have (model.levels) needs no mask folder: its masks are empty.
        self.levels = tuple(getattr(config, 'levels', ('nucleus', 'cytoplasm')))
        self.nucleus_files, self.cytoplasm_files = (
            find_all_images(self.dataset_dir / (name + '_masks')) if name in self.levels
            else [None] * len(self.image_files) for name in ('nucleus', 'cytoplasm'))

        assert len(self.image_files) == len(self.nucleus_files) == len(self.cytoplasm_files), \
            "Mismatch in number of images, nucleus masks, and cytoplasm masks"

        # Check that filenames match
        for img_f, nuc_f, cyto_f in zip(self.image_files, self.nucleus_files, self.cytoplasm_files):
            assert all(f is None or f.name == img_f.name for f in (nuc_f, cyto_f)), \
                f"Filename mismatch: {img_f.name}, {nuc_f and nuc_f.name}, {cyto_f and cyto_f.name}"

        # Load assignments if available
        self.assignments_dir = self.dataset_dir / 'assignments'
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
        Нормализация и координаты foreground считаются ЗДЕСЬ, один раз на том."""
        print(f'Repooling (seed={seed})...')

        r_ = random.Random(seed)
        sample_ids = list(range(len(self.image_files)))
        pool_ids = r_.sample(sample_ids, min(self.pool_size, len(sample_ids)))

        nd = int(self.config.ndim)
        self.data_pool = []
        for idx in pool_ids:
            from cytonucs_runtime import read_model_image
            img = read_model_image(self.image_files[idx], nd, self.config.n_channel_in)
            # 3D RGB-стек сохранён как (Z, C, H, W) -> канал в конец (Z, H, W, C)
            if nd == 3 and img.ndim == 4 and img.shape[-1] != self.config.n_channel_in and img.shape[1] == self.config.n_channel_in:
                img = np.moveaxis(img, 1, -1)
            nucleus_mask, cytoplasm_mask = (
                np.zeros(img.shape[:nd], np.uint16) if f is None else np.asarray(imageio.volread(f))
                for f in (self.nucleus_files[idx], self.cytoplasm_files[idx]))

            # ПЕРЦЕНТИЛЬНАЯ НОРМАЛИЗАЦИЯ — ОДИН РАЗ НА ТОМ.
            # Раньше normalize(1, 99.8) звался в __getitem__ по ПОЛНОМУ тому на
            # каждый сэмпл каждого шага: np.percentile делает partition по всем
            # ~10^8 элементам. Это и есть основной вклад в 50 с/батч.
            # Статистика глобальная по тому и от кропа с аугментацией не
            # зависит, поэтому кэшируется без изменения результата.
            if nd == 2 and img.ndim == 3 and img.shape[-1] in (3, 4):
                img = normalize(img.astype(np.float32), 1, 99.8, axis=(0, 1))
            else:
                img = normalize(img.astype(np.float32), 1, 99.8,
                                axis=tuple(range(nd)))
            img = np.ascontiguousarray(img, dtype=np.float32)

            # Координаты foreground — тоже один раз: np.argwhere по полному тому
            # аллоцировал массив на каждый сэмпл.
            seed_mask = nucleus_mask if nucleus_mask.max() > 0 else cytoplasm_mask

            from scipy.ndimage import maximum_filter
            _p = tuple(int(v) for v in self.config.train_patch_size)
            _cover = maximum_filter(seed_mask, size=_p, mode='constant') > 0
            _bs = tuple(slice(p // 2, s - p + p // 2 + 1)
                        for p, s in zip(_p, seed_mask.shape))
            _vi = np.where(_cover[_bs])
            fg_coords = np.stack([v + s.start for s, v in zip(_bs, _vi)],
                                 axis=1).astype(np.int32)
            if len(fg_coords) == 0:  # фолбэк как в каноне
                fg_coords = np.argwhere(seed_mask > 0).astype(np.int32)
            if not getattr(self, 'paired', True):
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
                'fg_coords': fg_coords,
                'assignments': assignments,
                'filename': self.image_files[idx].name,
                'object_count': int(np.count_nonzero(np.unique(nucleus_mask))) +
                                int(np.count_nonzero(np.unique(cytoplasm_mask)))
            })

        print(f'  Loaded {len(self.data_pool)} samples into pool '
              f'(нормализовано и проиндексировано один раз)')
        self._pool_seed = seed
        counts = np.array([s['object_count'] for s in self.data_pool], np.float64)
        self._tile_weights = counts / counts.sum() if counts.sum() > 0 else None

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
            image = read_model_image(root / 'images' / name, nd, self.config.n_channel_in)
            if nd == 3 and image.ndim == 4 and image.shape[-1] != self.config.n_channel_in and image.shape[1] == self.config.n_channel_in:
                image = np.moveaxis(image, 1, -1)
            # Label IDs of the source volume are kept (assignments refer to them), in the tiles' dtype.
            nucleus, cytoplasm = (
                m.astype(np.uint16) if 0 <= m.min() and m.max() <= np.iinfo(np.uint16).max else m
                for m in (np.asarray(imageio.volread(root / (level + '_masks') / name)) if level in self.levels
                          else np.zeros(image.shape[:nd], np.uint16) for level in ('nucleus', 'cytoplasm')))
            pairs = json.loads((root / 'assignments' / (Path(name).stem + '.json')).read_text()) if self.paired else {}
            region = np.zeros(nucleus.shape[-2:], bool)
            for y0, y1, x0, x1 in entry['train_blocks']:
                region[y0:y1, x0:x1] = True
            allowed = allowed_crop_starts(region, patch[-2:])
            if not allowed.any():
                raise ValueError(f'{name}: no {patch[-2:]} crop fits into the train blocks')
            # Bounding-box centres of the cells (nuclei where a volume has no cell) inside the train blocks.
            labels = cytoplasm if cytoplasm.max() > 0 else nucleus
            anchors = [([(sl.start + sl.stop) // 2 for sl in box], label)
                       for label, box in enumerate(find_objects(labels), 1) if box is not None]
            anchors = [(c, label) for c, label in anchors if region[c[-2], c[-1]]]
            self.region_pool.append(dict(
                image=image, nucleus_mask=nucleus, cytoplasm_mask=cytoplasm, filename=name,
                assignments={int(k): int(v) for k, v in pairs.items()},
                blocks=[tuple(b) for b in entry['train_blocks']], table=window_sum_table(region),
                starts=np.flatnonzero(allowed), starts_shape=allowed.shape,
                nearest=distance_transform_edt(~allowed, return_distances=False, return_indices=True).astype(np.int32),
                anchors=np.array([c for c, _ in anchors], np.int64).reshape(-1, nd),
                anchor_labels=np.array([label for _, label in anchors], np.int64), percentiles={},
                # Counted here once: the first batches come from several prefetch threads.
                _label_counts=dict(nucleus_mask=label_counts(nucleus), cytoplasm_mask=label_counts(cytoplasm))))
            print(f'  Train region {name}: {len(entry["train_blocks"])} block(s), {len(anchors)} anchor object(s), '
                  f'{int(allowed.sum())} crop positions in XY')
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
            sample['percentiles'][key] = np.percentile(context.astype(np.float32), (1, 99.8),
                                                       axis=tuple(range(nd)), keepdims=True)
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
        Число батчей за эпоху.
        При маленьком пуле (1-2 тома) деление на batch_size давало 1 батч из
        1 элемента: срез indices[0:bs] из массива длиной 1. Поэтому
        train_batch_size вообще ни на что не влиял, а трейнер крутил
        steps_per_epoch шагов по одному и тому же батчу (batch_idx = step % 1).
        Для train опираемся на steps_per_epoch — каждый вызов собирает свежий
        батч случайными кропами; для val сохраняем детерминированный проход.
        """
        if self.subset == 'train':
            return max(1, int(getattr(self.config, 'train_steps_per_epoch', 1)))
        return max(1, len(self.data_pool))

    def __getitem__(self, idx):
        """Кроп СНАЧАЛА, аугментация ПОТОМ — по патчу, а не по полному тому.
        Изображение приходит из пула уже нормализованным.

        K ПРОМПТОВ УХОДЯТ В КАНАЛЫ ОДНОГО КРОПА, А НЕ В КОПИИ БАТЧА.
        Прежняя схема размножала кроп bs раз, то есть платила ПОЛНЫМ форвардом
        backbone за каждый промпт — именно это не влезало в память, из-за чего
        bs=1 и рефайн видел один промпт на шаг ("валидных сэмплов: 1 / 1" в
        логе). Рефайн стоит за stop_gradient по всем входам, поэтому backbone
        считается один раз, а ветка применяется K раз общими весами.

        K = train_batch_size, ровно как читает модель. Порядок каналов
        [img..., prompt_nuc_0, prompt_cyto_0, prompt_nuc_1, ...] обязан
        совпадать с Lambda-срезами в _build_model.

        *_target складываются в батч K*B в порядке k*B + b — так же, как
        модель склеивает K выходов рефайна.
        """
        bs = int(self.config.train_batch_size)
        nd = int(self.config.ndim)
        patch = tuple(int(p) for p in self.config.train_patch_size)
        zoom_axes = (1, 2) if nd == 3 else (0, 1)
        is_train = (self.subset == 'train')
        n_pr = max(1, int(getattr(self.config, 'prompts_per_crop', bs)))

        # локальный rng: __getitem__ вызывается из НЕСКОЛЬКИХ потоков
        # префетчера, глобальный np.random между ними шарится
        pick = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch if is_train else 0, int(idx)]))
        regions = getattr(self, 'region_pool', None) if is_train else None
        if regions:
            crop_indices = [int(pick.choice(len(regions), p=self._region_weights))
                            for _ in range(int(getattr(self.config, 'image_batch_size', 1)))]
        elif is_train:
            crop_indices = [CytoNucsDataGenerator._pick_tile(self, pick)
                            for _ in range(int(getattr(self.config, 'image_batch_size', 1)))]
        else:
            crop_indices = [int(self.indices[int(idx) % max(1, len(self.indices))])]

        X_batch = []
        y_batch = {'nucleus_instances': [], 'cytoplasm_instances': []}
        # k-major: tg_n_k[k][b]
        tg_n_k = [[] for _ in range(n_pr)]
        tg_c_k = [[] for _ in range(n_pr)]

        for _ci, i in enumerate(crop_indices):
            # ДЕТЕРМИНИРОВАННЫЙ val: иначе каждая эпоха валидируется на другом
            # кропе с другим промптом, и val_loss мерит дисперсию выборки.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch if is_train else 0, int(idx), int(_ci), 1]))

            sample = regions[i] if regions else self.data_pool[i]
            img_full = sample['image']
            nuc_full = sample['nucleus_mask']
            cyto_full = sample['cytoplasm_mask']
            spatial = img_full.shape[:nd]

            # ZOOM выбираем ЗАРАНЕЕ, чтобы кропнуть ровно столько, сколько после
            # масштабирования даст patch. Внутри аугментера зум отключён.
            zr = getattr(self.config, 'zoom_range', (0.75, 1.35))
            f = 1.0
            if (is_train and self.augmenter is not None
                    and zr is not None and rng.random() < 0.5):
                f = float(rng.uniform(*zr))
                # Never invent reflected foreground with background labels.
                f = max(f, max(patch[a] / spatial[a] for a in zoom_axes))

            need = list(patch)
            if f != 1.0:
                # Rounding margin only for a real zoom: without zoom the +2 margin was
                # trimmed back by the central crop, so crops at the volume face lost its
                # outer XY row and objects there were always cut by one voxel in training.
                for a in zoom_axes:
                    need[a] = int(np.ceil(patch[a] / f)) + 2
            need = [min(need[a], spatial[a]) for a in range(nd)]

            anchor = 0
            if regions:
                starts, anchor = self._region_crop(sample, patch, rng)
                if f != 1.0:
                    # The larger crop of a zoom-out keeps its centre; where it would leave the
                    # train blocks or the volume, this crop is taken without zoom.
                    grown = [starts[a] - (need[a] - patch[a]) // 2 for a in range(nd)]
                    if (all(0 <= g and g + n <= s for g, n, s in zip(grown, need, spatial))
                            and window_inside(sample['table'], grown[-2:], need[-2:])):
                        starts = grown
                    else:
                        f, need = 1.0, [min(patch[a], spatial[a]) for a in range(nd)]
            else:
                # центр кропа: с вероятностью p_fg на случайном объекте
                p_fg = getattr(self.config, 'fg_crop_prob', 0.9)
                center = None
                fg = sample['fg_coords']
                if rng.random() < p_fg and len(fg) > 0:
                    center = fg[int(rng.integers(len(fg)))][:nd]

                starts = []
                for a in range(nd):
                    sh, p = spatial[a], need[a]
                    if sh <= p:
                        starts.append(0)
                    elif center is not None:
                        # center — валидный ЦЕНТР патча (канон), не воксель объекта
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

            # ZOOM по патчу
            if f != 1.0:
                from scipy.ndimage import zoom as ndzoom
                zf = [1.0] * nd
                for a in zoom_axes:
                    zf[a] = f
                zf_img = tuple(zf) + ((1.0,) if img.ndim > nd else ())
                img = ndzoom(img, zf_img, order=1)
                nucleus_mask = ndzoom(nucleus_mask, tuple(zf), order=0)
                cytoplasm_mask = ndzoom(cytoplasm_mask, tuple(zf), order=0)

            # Аугментация (флипы, транспоз, интенсивность) — уже по патчу
            if self.augmenter is not None:
                if self.augmenter in (cytonucs_augmenter_3d, cytonucs_augmenter_2d):
                    img, nucleus_mask, cytoplasm_mask = self.augmenter(
                        img, nucleus_mask, cytoplasm_mask, rng=rng, config=self.config)
                else:
                    img, nucleus_mask, cytoplasm_mask = self.augmenter(img, nucleus_mask, cytoplasm_mask)

            # Центральный кроп / паддинг ровно до patch
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
            if central:
                crop_cut['nucleus'] |= partial_labels(nucleus_mask, before_central[0])
                crop_cut['cytoplasm'] |= partial_labels(cytoplasm_mask, before_central[1])

            if any(img.shape[a] < patch[a] for a in range(nd)):
                pw = [(0, patch[a] - img.shape[a]) for a in range(nd)]
                pw_img = pw + [(0, 0)] if img.ndim > nd else pw
                img = np.pad(img, pw_img, mode='constant')
                nucleus_mask = np.pad(nucleus_mask, pw, mode='constant')
                cytoplasm_mask = np.pad(cytoplasm_mask, pw, mode='constant')

            # StarDist GT зависит ТОЛЬКО от масок -> считается один раз на кроп
            _pn, _dn = compute_stardist_gt(nucleus_mask, self.config,
                                           self._nuc_grid, rays=self._nuc_rays,
                                           aniso=self._nuc_aniso,
                                           normalize_prob=self.config.nucleus_prob_grid_normalize)
            _pc, _dc = compute_stardist_gt(cytoplasm_mask, self.config,
                                           self._cyto_grid, rays=self._cyto_rays,
                                           aniso=self._cyto_aniso)

            img_ch_base = img[..., np.newaxis] if img.ndim == nd else img

            def _prompt_at(shape, c, extent, rng_=rng, augment=True, owner=None):
                # PROMPT AUGMENTATION. На обучении промпт строится по ТОЧНОМУ
                # центроиду и bbox GT-инстанса, а на инференсе — по пику NMS с
                # сигмой из ПРЕДСКАЗАННЫХ лучей. Рефайн, обученный на идеальном
                # промпте, к этому сдвигу не готов. Шум задаётся в долях размера
                # объекта, поэтому не зависит от масштаба данных.
                ext = np.asarray(extent, np.float32)
                shift = float(getattr(self.config, 'prompt_center_jitter', 0.1))
                sscale = float(getattr(self.config, 'prompt_sigma_jitter', 0.1))
                c = np.asarray(c, np.float32)
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

            # Повторы одной клетки управляются repeat_cell_prompt_prob и дают
            # пары разных промптов для same-instance consistency.
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
                    # ядра, НАЗНАЧЕННЫЕ этой клетке и присутствующие в патче
                    nuc_ids = [n for n, c in _asg.items()
                               if int(c) == chosen_cyto and int(n) in nuc_present]
                    if not nuc_ids:  # фолбэк: перекрытие, как раньше
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
                    for _attempt in range(64):
                        center_neg = tuple(int(rng.integers(s)) for s in nucleus_mask.shape)
                        if nucleus_mask[center_neg] == 0:
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

        X_batch = np.stack(X_batch)
        for _k in ('nucleus_instances', 'cytoplasm_instances',
                   'nucleus_prob', 'nucleus_dist', 'cytoplasm_prob', 'cytoplasm_dist',
                   'cytoplasm_z_prompt_pair', 'nucleus_z_prompt_pair', 'crop_truncation_axes',
                   'nucleus_crop_cut', 'cytoplasm_crop_cut'):
            y_batch[_k] = np.stack(y_batch[_k])
        # k*B + b — порядок, в котором модель склеивает K выходов рефайна
        y_batch['nucleus_target'] = np.concatenate(
            [np.stack(v) for v in tg_n_k], axis=0)
        y_batch['cytoplasm_target'] = np.concatenate(
            [np.stack(v) for v in tg_c_k], axis=0)
        return X_batch, y_batch

    def _visualize_gt_comparison(self, img, nucleus_mask_original, cytoplasm_mask_original,
                                 nucleus_mask_final, cytoplasm_mask_final,
                                 nucleus_dist_raw, cytoplasm_dist_raw, epoch):
        """
        Визуализация: Original GT vs Final GT (после watershed)
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
    img = img * rng.uniform(*getattr(config, 'intensity_range', (0.8, 1.2)))
    img = img + rng.uniform(*getattr(config, 'intensity_offset', (-0.1, 0.1)))
    gamma = rng.uniform(*getattr(config, 'gamma_range', (0.9, 1.1)))
    img = np.sign(img) * np.abs(img) ** gamma
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

    # ========== Вспомогательные функции ==========

    def read_image_any_format(filepath, keep_rgb=False):
        """
        Читает изображение:
        - 2D grayscale: (H, W)
        - 2D RGB/RGBA: (H, W, 3/4)
        - 3D grayscale: (Z, H, W)
        - 3D RGB/RGBA: (Z, H, W, 3/4)

        Если keep_rgb=False:
        - 2D RGB -> grayscale
        - 3D RGB -> 3D grayscale, послойно
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext in [".tif", ".tiff"]:
            img = imageio.imread(str(filepath))
        elif ext in [".png", ".jpg", ".jpeg"]:
            img = imageio.imread(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # RGB/RGBA -> grayscale только если keep_rgb=False
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
        """Пишем тем же форматом, что и исходники: обычный многостраничный
        TIFF без ImageJ-заголовка. Исходные стеки — (Z,Y,X,3) uint8,
        photometric=RGB, is_imagej=False; Fiji читает их без проблем.
        Ни metadata={'axes':...}, ни imagej=True тут не нужны: первое
        подменяет ImageDescription своим JSON, второе перекладывает RGB
        в ZCYX-гиперстек и меняет глубину на 16 бит.
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
        """Проверка instance mask"""
        unique_vals = np.unique(mask)

        if len(unique_vals) == 2 and 255 in unique_vals:
            print(f"      ⚠️  WARNING: {name} is semantic (0/255), not instance!")
            return False

        if len(unique_vals) > 200:
            print(f"      ⚠️  WARNING: {name} has {len(unique_vals)} values")

        return True

    def find_all_images(directory, base_name=None):
        """Поиск файлов"""
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

    # ========== Поиск и валидация ==========

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

    # ========== Создание директорий ==========

    for subset in ['train', 'val', 'test']:
        for subdir in ['images', 'nucleus_masks', 'cytoplasm_masks', 'assignments']:
            (output_dir / subset / subdir).mkdir(parents=True, exist_ok=True)

    # ========== Обработка ==========
    rescale_factors = []
    for subset_name, triplets in [('train', train_triplets),
                                  ('val', val_triplets),
                                  ('test', test_triplets)]:
        print(f"\nProcessing {subset_name}...")

        for img_file, nuc_file, cyto_file in triplets:
            base_name = img_file.stem
            print(f"  {base_name}")

            # Чтение (RGB сохраняется для изображений)
            img = read_image_any_format(img_file, keep_rgb=True)
            nucleus_mask = read_image_any_format(nuc_file, keep_rgb=False)
            cytoplasm_mask = read_image_any_format(cyto_file, keep_rgb=False)

            # Определяем ndim (2D RGB считается 2D, не 3D)
            if img.ndim == 3 and img.shape[-1] == 3:
                ndim = 2  # 2D RGB
            else:
                ndim = img.ndim - (1 if img.ndim == 4 and img.shape[-1] in (3,4) else 0)

            print(f"    Detected: {ndim}D, img shape {img.shape}")

            # === RESCALE (как в стандартном StarDist train.py) ===
            # # Constraint: median cytoplasm extent должен влезать в FOV сети
            # if train_patch_size is not None:
            #     from skimage.measure import regionprops
            #     from stardist import fill_label_holes
            #
            #     cyto_props = regionprops(cytoplasm_mask)
            #     if len(cyto_props) > 0:
            #         extents = []
            #         for prop in cyto_props:
            #             bbox = prop.bbox
            #             n = len(bbox) // 2
            #             ext = np.array([bbox[d + n] - bbox[d] for d in range(n)])
            #             extents.append(ext)
            #         median_extent = np.median(extents, axis=0)
            #
            #         # FOV ≈ effective receptive field of UNet
            #         # Эмпирически ~60% patch_size для 4-level UNet с 3×3 convs
            #         patch_arr = np.array(train_patch_size[-ndim:], dtype=np.float64)
            #         fov = patch_arr * 0.6
            #
            #         # Rescale только если median extent НЕ влезает в FOV
            #         max_extent = np.max(median_extent)
            #         max_fov = np.min(fov)
            #
            #         if max_extent > max_fov:
            #             rescale_factor = float(max_fov / max_extent)
            #             rescale_factor = np.clip(rescale_factor, 0.25, 1.0)
            #
            #             print(f"    Median extent {median_extent} > FOV {fov}")
            #             print(f"    Rescaling: factor={rescale_factor:.3f}")
            #
            #             if ndim == 2 and img.ndim == 3 and img.shape[-1] in [3, 4]:
            #                 img = rescale(img, (rescale_factor, rescale_factor, 1),
            #                               order=1, preserve_range=True, anti_aliasing=True).astype(img.dtype)
            #             else:
            #                 img = rescale(img, rescale_factor,
            #                               order=1, preserve_range=True, anti_aliasing=True).astype(img.dtype)
            #
            #             nucleus_mask = rescale(nucleus_mask, rescale_factor,
            #                                    order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)
            #             nucleus_mask = fill_label_holes(nucleus_mask)
            #
            #             cytoplasm_mask = rescale(cytoplasm_mask, rescale_factor,
            #                                      order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)
            #             cytoplasm_mask = fill_label_holes(cytoplasm_mask)
            #
            #             print(f"    New shape: {img.shape}")
            #             rescale_factors.append(rescale_factor)
            #         else:
            #             print(f"    No rescale needed: median extent {max_extent:.0f}px < FOV {max_fov:.0f}px")
            rescale_factors.append(1.0)

            # Валидация
            validate_instance_mask(nucleus_mask, "nucleus")
            validate_instance_mask(cytoplasm_mask, "cytoplasm")

            # Единственный rescale — FOV-based блок выше (он ndim-aware).
            # Прежний второй блок (хардкод target_extent=[128,128], 2D-индексы
            # bbox[2]/bbox[3]) удалён: ломал 3D и противоречил "no rescaling".

            # Сохранение
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
