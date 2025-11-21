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
from cytonucs_losses import compute_nucleus_cell_assignments

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

        # Load file lists
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
        """Reload a random subset of data into memory"""
        print(f'Repooling (seed={seed})...')

        r_ = random.Random(seed)
        sample_ids = list(range(len(self.image_files)))
        pool_ids = r_.sample(sample_ids, min(self.pool_size, len(sample_ids)))

        self.data_pool = []
        for idx in pool_ids:
            # Load data
            img = imageio.volread(self.image_files[idx])
            nucleus_mask = imageio.volread(self.nucleus_files[idx])
            cytoplasm_mask = imageio.volread(self.cytoplasm_files[idx])

            # Load or compute assignments
            if self.has_assignments:
                assignment_file = self.assignments_dir / self.image_files[idx].name.replace('.tif', '.json')
                if assignment_file.exists():
                    with open(assignment_file, 'r') as f:
                        assignments = json.load(f)
                        assignments = {int(k): int(v) for k, v in assignments.items()}
                else:
                    # Auto-compute if missing

                    assignments, _ = compute_nucleus_cell_assignments(nucleus_mask,cytoplasm_mask, max_distance=5.0)
            else:
                # Auto-compute assignments
                assignments, _ = compute_nucleus_cell_assignments(nucleus_mask,cytoplasm_mask, max_distance=5.0)

            self.data_pool.append({
                'image': img,
                'nucleus_mask': nucleus_mask,
                'cytoplasm_mask': cytoplasm_mask,
                'assignments': assignments,
                'filename': self.image_files[idx].name
            })

        print(f'  Loaded {len(self.data_pool)} samples into pool')

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
        """Number of batches per epoch"""
        return int(np.ceil(len(self.data_pool) / self.config.train_batch_size))

    def __getitem__(self, idx):
        """
        Возвращаем ТОЛЬКО сырые маски (как в train.py).
        Никаких distance maps здесь!
        """
        batch_indices = self.indices[
                        idx * self.config.train_batch_size:
                        (idx + 1) * self.config.train_batch_size
                        ]

        X_batch = []
        y_batch = {
            'nucleus_instances': [],
            'cytoplasm_instances': []
        }

        for i in batch_indices:
            sample = self.data_pool[i]

            img = sample['image'].copy()
            nucleus_mask = sample['nucleus_mask'].copy()
            cytoplasm_mask = sample['cytoplasm_mask'].copy()

            # Аугментация
            if self.augmenter is not None:
                img, nucleus_mask, cytoplasm_mask = self.augmenter(img, nucleus_mask, cytoplasm_mask)

            # Нормализация
            img = normalize(img.astype(np.float32), 1, 99.8, axis=tuple(range(self.config.ndim)))

            # Crop/Pad до patch_size
            patch_size = self.config.train_patch_size
            spatial_shape = img.shape[:self.config.ndim]

            if any(spatial_shape[i] > patch_size[i] for i in range(self.config.ndim)):
                starts = [np.random.randint(0, sh - p) if sh > p else 0
                          for sh, p in zip(spatial_shape, patch_size)]
                slices = tuple(slice(s, s + p) for s, p in zip(starts, patch_size))
                img = img[slices]
                nucleus_mask = nucleus_mask[slices]
                cytoplasm_mask = cytoplasm_mask[slices]

            if any(img.shape[i] < patch_size[i] for i in range(self.config.ndim)):
                pad_width = [(max(0, p - s) // 2, max(0, p - s) - max(0, p - s) // 2)
                             for s, p in zip(img.shape, patch_size)]
                img = np.pad(img, pad_width[:self.config.ndim], mode='reflect')
                nucleus_mask = np.pad(nucleus_mask, pad_width[:self.config.ndim], mode='constant')
                cytoplasm_mask = np.pad(cytoplasm_mask, pad_width[:self.config.ndim], mode='constant')

            # Добавляем в батч (как в train.py!)
            X_batch.append(img[..., np.newaxis] if img.ndim == self.config.ndim else img)
            y_batch['nucleus_instances'].append(nucleus_mask)
            y_batch['cytoplasm_instances'].append(cytoplasm_mask)

        # Stack
        X_batch = np.stack(X_batch)
        y_batch['nucleus_instances'] = np.stack(y_batch['nucleus_instances'])
        y_batch['cytoplasm_instances'] = np.stack(y_batch['cytoplasm_instances'])

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
    Applies same transformations to image and both masks.
    Same as augmenter in original train.py
    """
    from scipy.ndimage import affine_transform, gaussian_filter

    # 1. Random flip and rotation (only in YX plane like original)
    axis = (1, 2)  # Y, X axes

    # Random permutation of axes
    if np.random.rand() > 0.5:
        perm = np.random.permutation(axis)
        transpose_axis = list(range(img.ndim))
        for a, p in zip(axis, perm):
            transpose_axis[a] = p
        img = img.transpose(transpose_axis)
        nucleus_mask = nucleus_mask.transpose(transpose_axis)
        cytoplasm_mask = cytoplasm_mask.transpose(transpose_axis)

    # Random flips
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            nucleus_mask = np.flip(nucleus_mask, axis=ax)
            cytoplasm_mask = np.flip(cytoplasm_mask, axis=ax)

    # 2. Intensity augmentation (only for image)
    img = img * np.random.uniform(0.6, 2.0) + np.random.uniform(-0.2, 0.2)

    # 3. Gamma correction
    gamma = np.random.uniform(0.9, 1.1)
    img = np.clip(img, 0, None)
    img = img ** gamma

    # 4. Gaussian noise
    noise = np.random.normal(loc=0.0, scale=0.01, size=img.shape)
    img = img + noise
    img = np.clip(img, 0, None)

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
    img = rotate(img, angle, order=1, preserve_range=True)
    nucleus_mask = rotate(nucleus_mask, angle, order=0, preserve_range=True).astype(nucleus_mask.dtype)
    cytoplasm_mask = rotate(cytoplasm_mask, angle, order=0, preserve_range=True).astype(cytoplasm_mask.dtype)

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

    # Возвращаем гарантированно целочисленные маски
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
        seed=42
):
    """
    Prepare dataset for CytoNucs training (2D/3D, grayscale/RGB).
    """
    from cytonucs_losses import compute_nucleus_cell_assignments
    from skimage.morphology import remove_small_holes
    import cv2

    raw_images_dir = Path(raw_images_dir)
    raw_nucleus_masks_dir = Path(raw_nucleus_masks_dir)
    raw_cytoplasm_masks_dir = Path(raw_cytoplasm_masks_dir)
    output_dir = Path(output_dir)

    # ========== Вспомогательные функции ==========

    def read_image_any_format(filepath, keep_rgb=False):
        """Читает изображение: 2D grayscale, 2D RGB, или 3D grayscale"""
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext in ['.tif', '.tiff']:
            img = imageio.volread(str(filepath))
        elif ext in ['.png', '.jpg', '.jpeg']:
            img = imageio.imread(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Конвертация RGB→grayscale только для масок
        if not keep_rgb and img.ndim == 3 and img.shape[-1] in [3, 4]:
            # Это 2D RGB - конвертируем в grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img

    def save_image_smart(img, filepath):
        """
        Умное сохранение: определяет тип и использует правильный метод.

        Типы:
        - 2D RGB: (H, W, 3) → imwrite uint8
        - 2D grayscale: (H, W) → imwrite uint16
        - 3D: (Z, H, W) → volwrite uint16
        """
        # Проверяем тип изображения
        if img.ndim == 3 and img.shape[-1] == 3:
            # 2D RGB - маленькая размерность по последней оси
            imageio.imwrite(filepath, img.astype(np.uint8))
        elif img.ndim == 2:
            # 2D grayscale
            imageio.imwrite(filepath, img.astype(np.uint16))
        elif img.ndim == 3:
            # 3D grayscale (Z, H, W)
            imageio.volwrite(filepath, img.astype(np.uint16))
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

    def fill_contour_mask(mask):
        """Заливка контуров БЕЗ перезаписи соседних объектов (FAST)"""
        from scipy.ndimage import binary_fill_holes, binary_dilation, distance_transform_edt
        from skimage.morphology import disk, ball

        mask = mask.astype(np.uint16)

        # Выбор структурного элемента
        ndim = mask.ndim
        structuring_element = ball(2) if ndim == 3 else disk(2)

        # ШАГ 1: Собираем все расширенные маски
        all_dilated = {}
        all_distances = {}

        for label_id in np.unique(mask)[1:]:
            binary = (mask == label_id).astype(bool)
            dilated = binary.copy()

            # Быстрая заливка с ограниченным числом итераций
            for i in range(15):
                dilated = binary_dilation(dilated, structuring_element)
                filled_attempt = binary_fill_holes(dilated)

                ratio = filled_attempt.sum() / max(dilated.sum(), 1)
                if ratio > 1.2:
                    all_dilated[label_id] = filled_attempt
                    break

                if i == 14:
                    all_dilated[label_id] = dilated

            # Pre-compute distance transform (ОДИН РАЗ для каждого объекта!)
            all_distances[label_id] = distance_transform_edt(~binary)

        # ШАГ 2: Создаём карты приоритетов (векторизованно)
        # Для каждого пикселя храним: (label_id, distance)
        label_ids = list(all_dilated.keys())

        # Инициализация результата
        filled_mask = np.zeros_like(mask, dtype=np.uint16)
        priority_map = np.full(mask.shape, np.inf, dtype=np.float32)

        # ВЕКТОРИЗОВАННОЕ присвоение: для каждого объекта
        for label_id in label_ids:
            dilated_mask = all_dilated[label_id]
            dist_map = all_distances[label_id]

            # Где этот объект хочет быть И где он ближе чем текущий winner
            better_mask = dilated_mask & (dist_map < priority_map)

            # Обновляем
            filled_mask[better_mask] = label_id
            priority_map[better_mask] = dist_map[better_mask]

        return filled_mask

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

            # Валидация
            validate_instance_mask(nucleus_mask, "nucleus")
            validate_instance_mask(cytoplasm_mask, "cytoplasm")

            # Заливка
            #nucleus_mask = fill_contour_mask(nucleus_mask)
            #cytoplasm_mask = fill_contour_mask(cytoplasm_mask)

            # Сохранение
            output_filename = base_name + '.tif'

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
        'val_split': val_split if not validation_images else 'manual'
    }
    # ========== Generate metadata ==========
    print("\nGenerating metadata...")

    from skimage.measure import regionprops

    for subset_name in ['train', 'val', 'test']:
        subset_dir = output_dir / subset_name
        image_files = sorted((subset_dir / 'images').glob('*.tif'))

        metadata = []

        for img_file in image_files:
            base_name = img_file.stem

            # Load masks
            nuc_mask = imageio.volread(subset_dir / 'nucleus_masks' / f'{base_name}.tif')
            cyto_mask = imageio.volread(subset_dir / 'cytoplasm_masks' / f'{base_name}.tif')

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