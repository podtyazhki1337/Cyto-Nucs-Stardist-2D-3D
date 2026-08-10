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
            return sorted(files)

        self.image_files = find_all_images(self.dataset_dir / 'images')
        self.nucleus_files = find_all_images(self.dataset_dir / 'nucleus_masks')
        self.cytoplasm_files = find_all_images(self.dataset_dir / 'cytoplasm_masks')

        assert len(self.image_files) == len(self.nucleus_files) == len(self.cytoplasm_files), \
            "Mismatch in number of images, nucleus masks, and cytoplasm masks"

        # Check that filenames match
        for img_f, nuc_f, cyto_f in zip(self.image_files, self.nucleus_files, self.cytoplasm_files):
            assert img_f.name == nuc_f.name == cyto_f.name, \
                f"Filename mismatch: {img_f.name}, {nuc_f.name}, {cyto_f.name}"

        # Load assignments if available
        self.assignments_dir = self.dataset_dir / 'assignments'
        self.has_assignments = self.assignments_dir.exists()

        # Load metadata if available
        self.meta = None
        meta_file = self.dataset_dir / 'meta.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                self.meta = json.load(f)

        # Pooling setup (like StarDistSequence)
        assert pool_size != 0 and repool_freq != 0, "pool_size and repool_freq should not be 0!"

        if pool_size is None:
            self.pool_size = len(self.image_files)
        else:
            self.pool_size = pool_size

        if repool_freq is None:
            self.repool_freq = self.pool_size
        else:
            self.repool_freq = self.pool_size * repool_freq

        self.data_pool = None

        print(f'\n{subset.upper()} Database:')
        for idx, im_file in enumerate(self.image_files):
            print(f'{idx}\t{im_file.name}')

        # Initial pool
        self.repool(seed=0)

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
            img = imageio.volread(self.image_files[idx])
            # 3D RGB-стек сохранён как (Z, C, H, W) -> канал в конец (Z, H, W, C)
            if nd == 3 and img.ndim == 4 and img.shape[1] in (3, 4):
                img = np.moveaxis(img, 1, -1)
            nucleus_mask = np.asarray(imageio.volread(self.nucleus_files[idx]))
            cytoplasm_mask = np.asarray(imageio.volread(self.cytoplasm_files[idx]))

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
            if self.has_assignments:
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
                'filename': self.image_files[idx].name
            })

        print(f'  Loaded {len(self.data_pool)} samples into pool '
              f'(нормализовано и проиндексировано один раз)')

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
        return max(1, int(np.ceil(len(self.data_pool) / self.config.train_batch_size)))

    def __getitem__(self, idx):
        """Кроп СНАЧАЛА, аугментация ПОТОМ — по патчу, а не по полному тому.
        Изображение приходит из пула уже нормализованным."""
        bs = int(self.config.train_batch_size)
        nd = int(self.config.ndim)
        patch = tuple(int(p) for p in self.config.train_patch_size)
        zoom_axes = (1, 2) if nd == 3 else (0, 1)

        # локальный rng и для выбора элементов пула: __getitem__ теперь
        # вызывается из НЕСКОЛЬКИХ потоков префетчера, глобальный np.random
        # между ними шарится
        pick = np.random.default_rng(None if self.subset == 'train' else int(idx))
        if self.subset == 'train':
            batch_indices = pick.choice(
                len(self.data_pool), bs, replace=len(self.data_pool) < bs)
        else:
            batch_indices = self.indices[idx * bs:(idx + 1) * bs]

        X_batch = []
        y_batch = {'nucleus_instances': [], 'cytoplasm_instances': []}

        for _pos, i in enumerate(batch_indices):
            # ДЕТЕРМИНИРОВАННЫЙ val: иначе каждая эпоха валидируется на другом
            # кропе с другим промптом, и val_loss мерит дисперсию выборки.
            rng = (np.random.default_rng() if self.subset == 'train'
                   else np.random.default_rng(1000 * int(idx) + int(_pos)))

            sample = self.data_pool[i]
            img_full = sample['image']
            nuc_full = sample['nucleus_mask']
            cyto_full = sample['cytoplasm_mask']
            spatial = img_full.shape[:nd]

            # ZOOM выбираем ЗАРАНЕЕ, чтобы кропнуть ровно столько, сколько после
            # масштабирования даст patch. Внутри аугментера зум отключён.
            zr = getattr(self.config, 'zoom_range', (0.75, 1.35))
            f = 1.0
            if (self.subset == 'train' and self.augmenter is not None
                    and zr is not None and rng.random() < 0.5):
                f = float(rng.uniform(*zr))

            need = list(patch)
            for a in zoom_axes:
                need[a] = int(np.ceil(patch[a] / f)) + 2      # запас на округление
            need = [min(need[a], spatial[a]) for a in range(nd)]

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
                    starts.append(
                        int(
                            rng.integers(
                                0,
                                sh - p + 1
                            )
                        )
                    )
            sl = tuple(slice(s, s + n) for s, n in zip(starts, need))

            img = np.ascontiguousarray(img_full[sl])
            nucleus_mask = np.ascontiguousarray(nuc_full[sl])
            cytoplasm_mask = np.ascontiguousarray(cyto_full[sl])

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
                img, nucleus_mask, cytoplasm_mask = self.augmenter(
                    img, nucleus_mask, cytoplasm_mask)

            # Центральный кроп / паддинг ровно до patch
            cur = img.shape[:nd]
            beg = [max(0, (cur[a] - patch[a]) // 2) for a in range(nd)]
            sl2 = tuple(slice(beg[a], beg[a] + min(cur[a], patch[a])) for a in range(nd))
            img = img[sl2]
            nucleus_mask = nucleus_mask[sl2]
            cytoplasm_mask = cytoplasm_mask[sl2]

            if any(img.shape[a] < patch[a] for a in range(nd)):
                pw = [(0, patch[a] - img.shape[a]) for a in range(nd)]
                pw_img = pw + [(0, 0)] if img.ndim > nd else pw
                img = np.pad(img, pw_img, mode='reflect')
                nucleus_mask = np.pad(nucleus_mask, pw, mode='constant')
                cytoplasm_mask = np.pad(cytoplasm_mask, pw, mode='constant')

            def _prompt_at(shape, c, extent, rng=rng):
                # PROMPT AUGMENTATION. На обучении промпт строится по ТОЧНОМУ
                # центроиду и bbox GT-инстанса, а на инференсе — по пику NMS с
                # сигмой из ПРЕДСКАЗАННЫХ лучей. Рефайн, обученный на идеальном
                # промпте, к этому сдвигу не готов. Шум задаётся в долях размера
                # объекта, поэтому не зависит от масштаба данных.
                # Изображение и StarDist-таргеты НЕ трогаются.
                ext = np.asarray(extent, np.float32)
                shift = float(getattr(self.config, 'prompt_center_jitter', 0.1))
                sscale = float(getattr(self.config, 'prompt_sigma_jitter', 0.1))
                c = np.asarray(c, np.float32)
                if self.subset == 'train' and shift > 0:
                    c = c + rng.normal(0.0, shift, size=c.shape).astype(np.float32) * ext
                sig = np.maximum(ext / 4.0, 1.0)
                if self.subset == 'train' and sscale > 0:
                    sig = sig * float(rng.uniform(1.0 - sscale, 1.0 + sscale))
                grids = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape],
                                    indexing='ij')
                d2 = sum(((g - c[a]) / sig[a]) ** 2 for a, g in enumerate(grids))
                return np.exp(-0.5 * d2).astype(np.float32)

            # СОГЛАСОВАННАЯ ПАРА: клетка -> одно из ЕЁ ядер
            _asg = sample.get('assignments') or {}
            cyto_ids = np.unique(cytoplasm_mask)
            cyto_ids = cyto_ids[cyto_ids > 0]
            chosen_cyto = int(rng.choice(cyto_ids)) if len(cyto_ids) else 0

            nuc_present = {int(v) for v in np.unique(nucleus_mask) if v > 0}
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
                                pts.max(axis=0) - pts.min(axis=0) + 1)
            elif chosen_cyto:
                pts = np.argwhere(cytoplasm_mask == chosen_cyto)
                pr = _prompt_at(cytoplasm_mask.shape, pts.mean(axis=0),
                                pts.max(axis=0) - pts.min(axis=0) + 1)

            if chosen_cyto:
                tg_c[cytoplasm_mask == chosen_cyto] = 1.0

            img_ch = img[..., np.newaxis] if img.ndim == nd else img
            img_ch = np.concatenate(
                [img_ch, pr[..., np.newaxis], pr[..., np.newaxis]], axis=-1)

            X_batch.append(img_ch)
            y_batch['nucleus_instances'].append(nucleus_mask)
            y_batch['cytoplasm_instances'].append(cytoplasm_mask)
            y_batch.setdefault('nucleus_target', []).append(tg_n)
            y_batch.setdefault('cytoplasm_target', []).append(tg_c)

            _pn, _dn = compute_stardist_gt(nucleus_mask, self.config, self._nuc_grid,
                                           rays=self._nuc_rays, aniso=self._nuc_aniso)
            _pc, _dc = compute_stardist_gt(cytoplasm_mask, self.config, self._cyto_grid,
                                           rays=self._cyto_rays, aniso=self._cyto_aniso)
            y_batch.setdefault('nucleus_prob', []).append(_pn)
            y_batch.setdefault('nucleus_dist', []).append(_dn)
            y_batch.setdefault('cytoplasm_prob', []).append(_pc)
            y_batch.setdefault('cytoplasm_dist', []).append(_dc)

        X_batch = np.stack(X_batch)
        for _k in ('nucleus_instances', 'cytoplasm_instances',
                   'nucleus_target', 'cytoplasm_target',
                   'nucleus_prob', 'nucleus_dist', 'cytoplasm_prob', 'cytoplasm_dist'):
            y_batch[_k] = np.stack(y_batch[_k])
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
        valid_cyto_orig = cytoplasm_dist_raw[cytoplasm_mask_original > 0].flatten()
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
        """Shuffle data after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def cytonucs_augmenter_3d(img, nucleus_mask, cytoplasm_mask):
    """
    3D augmentation for CytoNucs training.
    Applies same spatial transformations to image and both masks.
    Supports image with optional channel dimension, e.g.:
        img: (Z, Y, X) or (Z, Y, X, C)
        masks: (Z, Y, X)
    """
    import numpy as np

    # Spatial axes: Z=0, Y=1, X=2
    axis = (1, 2)  # Y, X axes only

    def transpose_spatial(arr, perm):
        transpose_axis = list(range(arr.ndim))

        for a, p in zip(axis, perm):
            if a < arr.ndim and p < arr.ndim:
                transpose_axis[a] = p

        return arr.transpose(transpose_axis)

    # 1. Random permutation of Y/X axes
    if np.random.rand() > 0.5:
        perm = np.random.permutation(axis)

        img = transpose_spatial(img, perm)
        nucleus_mask = transpose_spatial(nucleus_mask, perm)
        cytoplasm_mask = transpose_spatial(cytoplasm_mask, perm)

    # 2. Random flips in Y/X
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            nucleus_mask = np.flip(nucleus_mask, axis=ax)
            cytoplasm_mask = np.flip(cytoplasm_mask, axis=ax)
    # 2b. Масштабная аугментация в плоскости XY.
    # Объекты между томами различаются в ~1.4 раза (median_max_ray 29.5 vs 42.2),
    # и модель, обученная на одном масштабе, на другом промахивается по dist.
    # Z НЕ масштабируем: при 8-17 срезах интерполяция по Z разрушает объект,
    # а анизотропия [5,1,1] делает Z-шаг физически крупным.
    zr = getattr(cytonucs_augmenter_3d, 'zoom_range', (0.75, 1.35))
    if zr is not None and np.random.rand() < 0.5:
        from scipy.ndimage import zoom as ndzoom
        f = float(np.random.uniform(*zr))
        zf_img = (1.0, f, f) + ((1.0,) if img.ndim == 4 else ())
        img = ndzoom(img, zf_img, order=1)
        nucleus_mask = ndzoom(nucleus_mask, (1.0, f, f), order=0)  # nearest: id не мешать
        cytoplasm_mask = ndzoom(cytoplasm_mask, (1.0, f, f), order=0)
    # 3. Intensity augmentation only for image
    img = img * np.random.uniform(0.6, 2.0) + np.random.uniform(-0.2, 0.2)

    # 4. Gamma correction
    gamma = np.random.uniform(0.9, 1.1)
    img = np.sign(img) * (np.abs(img) ** gamma)

    # 5. Gaussian noise
    img = img + np.random.normal(loc=0.0, scale=0.01, size=img.shape)

    # Important after np.flip / transpose: avoid negative strides for torch
    img = np.ascontiguousarray(img)
    nucleus_mask = np.ascontiguousarray(nucleus_mask)
    cytoplasm_mask = np.ascontiguousarray(cytoplasm_mask)

    return img, nucleus_mask, cytoplasm_mask


def cytonucs_augmenter_2d(img, nucleus_mask, cytoplasm_mask):
    """
    2D augmentation for CytoNucs training.
    """
    from skimage.transform import rotate, rescale
    import numpy as np

    # 1. Random flip
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
        nucleus_mask = np.fliplr(nucleus_mask)
        cytoplasm_mask = np.fliplr(cytoplasm_mask)
    if np.random.rand() > 0.5:
        img = np.flipud(img)
        nucleus_mask = np.flipud(nucleus_mask)
        cytoplasm_mask = np.flipud(cytoplasm_mask)

    # 2. Random rotation
    angle = np.random.uniform(-180, 180)
    # ИСПРАВЛЕНО: принудительно приводим маски к int после ротации
    img = rotate(img, angle, order=1, preserve_range=True, resize=False)
    nucleus_mask = rotate(nucleus_mask, angle, order=0, preserve_range=True, resize=False).astype(nucleus_mask.dtype)
    cytoplasm_mask = rotate(cytoplasm_mask, angle, order=0, preserve_range=True, resize=False).astype(
        cytoplasm_mask.dtype)

    # 3. Intensity augmentation
    img = img * np.random.uniform(0.6, 2.0) + np.random.uniform(-0.2, 0.2)

    # 4. Gamma
    gamma = np.random.uniform(0.9, 1.1)
    img = np.clip(img, 0, None) ** gamma

    # 5. Noise
    noise = np.random.normal(0, 0.01, img.shape)
    img = np.clip(img + noise, 0, None)

    # 6. Zoom
    if np.random.rand() > 0.5:
        scale_factor = np.random.uniform(0.9, 1.1)

        img_scaled = rescale(img, scale_factor, order=1, preserve_range=True, anti_aliasing=True,
                             channel_axis=-1 if img.ndim == 3 else None)
        # ИСПРАВЛЕНО: принудительно приводим маски к int после масштабирования
        nucleus_scaled = rescale(nucleus_mask, scale_factor, order=0, preserve_range=True, anti_aliasing=False).astype(
            nucleus_mask.dtype)
        cytoplasm_scaled = rescale(cytoplasm_mask, scale_factor, order=0, preserve_range=True,
                                   anti_aliasing=False).astype(cytoplasm_mask.dtype)

        # Crop or pad to original size
        h, w = img.shape[:2]
        h_s, w_s = img_scaled.shape[:2]

        if h_s >= h and w_s >= w:
            start_h, start_w = (h_s - h) // 2, (w_s - w) // 2
            img, nucleus_mask, cytoplasm_mask = img_scaled[start_h:start_h + h, start_w:start_w + w], nucleus_scaled[
                                                                                                      start_h:start_h + h,
                                                                                                      start_w:start_w + w], cytoplasm_scaled[
                                                                                                                            start_h:start_h + h,
                                                                                                                            start_w:start_w + w]
        else:
            pad_h, pad_w = (h - h_s) // 2, (w - w_s) // 2
            pad_h_rem, pad_w_rem = h - h_s - pad_h, w - w_s - pad_w
            padding_mask = ((pad_h, pad_h_rem), (pad_w, pad_w_rem))
            padding_img = padding_mask + ((0, 0),) if img.ndim == 3 else padding_mask

            img = np.pad(img_scaled, padding_img, mode='reflect')
            nucleus_mask = np.pad(nucleus_scaled, padding_mask, mode='constant')
            cytoplasm_mask = np.pad(cytoplasm_scaled, padding_mask, mode='constant')


    return img, nucleus_mask.astype(np.int32), cytoplasm_mask.astype(np.int32)


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

    def save_image_smart(img, filepath):
        """
        Умное сохранение: определяет тип и использует правильный метод.
        Типы:
        - 2D grayscale: (H, W)            → imwrite uint16
        - 2D RGB:       (H, W, 3)         → imwrite uint8
        - 3D grayscale: (Z, H, W)         → tifffile ZYX uint16
        - 3D RGB-стек:  (Z, H, W, 3)      → tifffile ZYXC, каналы сохраняются
        """
        import tifffile
        filepath = str(filepath)

        if img.ndim == 2:
            # 2D grayscale
            imageio.imwrite(filepath, img.astype(np.uint16))

        elif img.ndim == 3 and img.shape[-1] in (3, 4):
            # 2D RGB/RGBA
            imageio.imwrite(filepath, img[..., :3].astype(np.uint8))


        elif img.ndim == 3:

            # 3D grayscale (Z, H, W)

            tifffile.imwrite(filepath, img.astype(np.uint16),

                             metadata={'axes': 'ZYX'})


        elif img.ndim == 4 and img.shape[-1] in (3, 4):

            # 3D RGB/RGBA стек (Z, H, W, C). ImageJ требует канал перед YX (TZCYXS),

            # поэтому пишем без imagej-режима, оси ZCYX (канал отдельным измерением).

            c = min(img.shape[-1], 3)

            arr = np.moveaxis(img[..., :c], -1, 1).astype(np.uint16)  # (Z,H,W,C) -> (Z,C,H,W)

            tifffile.imwrite(filepath, arr, metadata={'axes': 'ZCYX'})

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
            return sorted(files)

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
                ndim = img.ndim

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
                output_dir / subset_name / 'nucleus_masks' / output_filename
            )
            save_image_smart(
                cytoplasm_mask,
                output_dir / subset_name / 'cytoplasm_masks' / output_filename
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