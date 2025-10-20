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
        self.image_files = sorted((self.dataset_dir / 'images').glob('*.tif'))
        self.nucleus_files = sorted((self.dataset_dir / 'nucleus_masks').glob('*.tif'))
        self.cytoplasm_files = sorted((self.dataset_dir / 'cytoplasm_masks').glob('*.tif'))

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
                    from cytonucs_losses import compute_nucleus_cell_assignments
                    assignments, _ = compute_nucleus_cell_assignments(nucleus_mask, cytoplasm_mask)
            else:
                # Auto-compute assignments
                from cytonucs_losses import compute_nucleus_cell_assignments
                assignments, _ = compute_nucleus_cell_assignments(nucleus_mask, cytoplasm_mask)

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
        Generate one batch of data.

        Returns:
            X: (batch_size, *patch_size, n_channels) images
            y: dict with nucleus and cytoplasm targets
        """
        # Check if we need to repool
        self.counter += 1
        if self.counter >= self.repool_freq:
            self.repool(seed=self.counter)
            self.counter = 0
            self.on_epoch_end()

        # Get indices for this batch
        batch_indices = self.indices[
                        idx * self.config.train_batch_size:
                        (idx + 1) * self.config.train_batch_size
                        ]

        X_batch = []
        y_batch = {
            'nucleus_prob': [],
            'nucleus_dist': [],
            'cytoplasm_prob': [],
            'cytoplasm_dist': [],
            'nucleus_instances': [],
            'cytoplasm_instances': [],
            'assignments': []
        }

        for i in batch_indices:
            sample = self.data_pool[i]

            img = sample['image'].copy()
            nucleus_mask = sample['nucleus_mask'].copy()
            cytoplasm_mask = sample['cytoplasm_mask'].copy()
            assignments = sample['assignments']

            # Apply augmentation
            if self.augmenter is not None:
                img, nucleus_mask, cytoplasm_mask = self.augmenter(
                    img, nucleus_mask, cytoplasm_mask
                )

            # Normalize image
            img = normalize(img.astype(np.float32), 1, 99.8,
                            axis=tuple(range(self.config.ndim)))

            # Create distance maps
            from cytonucs_losses import create_distance_maps

            nucleus_prob, nucleus_dist = create_distance_maps(nucleus_mask, self.rays)
            cytoplasm_prob, cytoplasm_dist = create_distance_maps(cytoplasm_mask, self.rays)

            # Add to batch
            X_batch.append(img[..., np.newaxis])  # Add channel dimension
            y_batch['nucleus_prob'].append(nucleus_prob)
            y_batch['nucleus_dist'].append(nucleus_dist)
            y_batch['cytoplasm_prob'].append(cytoplasm_prob)
            y_batch['cytoplasm_dist'].append(cytoplasm_dist)
            y_batch['nucleus_instances'].append(nucleus_mask)
            y_batch['cytoplasm_instances'].append(cytoplasm_mask)
            y_batch['assignments'].append(assignments)

        # Stack into batch
        X_batch = np.stack(X_batch, axis=0)
        for key in ['nucleus_prob', 'nucleus_dist', 'cytoplasm_prob', 'cytoplasm_dist']:
            y_batch[key] = np.stack(y_batch[key], axis=0)

        return X_batch, y_batch

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
    from skimage.transform import rotate

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
    img = rotate(img, angle, order=1, preserve_range=True)
    nucleus_mask = rotate(nucleus_mask, angle, order=0, preserve_range=True)
    cytoplasm_mask = rotate(cytoplasm_mask, angle, order=0, preserve_range=True)

    # 3. Intensity augmentation
    img = img * np.random.uniform(0.6, 2.0) + np.random.uniform(-0.2, 0.2)

    # 4. Gamma
    gamma = np.random.uniform(0.9, 1.1)
    img = np.clip(img, 0, None) ** gamma

    # 5. Noise
    noise = np.random.normal(0, 0.01, img.shape)
    img = np.clip(img + noise, 0, None)

    return img, nucleus_mask, cytoplasm_mask


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
    Prepare dataset for CytoNucs training with automatic train/val/test split.

    Creates train/val/test split and computes nucleus-cytoplasm assignments.

    Args:
        raw_images_dir: directory with raw images
        raw_nucleus_masks_dir: directory with nucleus instance masks
        raw_cytoplasm_masks_dir: directory with cytoplasm instance masks
        output_dir: where to save processed dataset
        validation_images: list of specific filenames for validation (optional)
        test_split: fraction for test set (if validation_images not provided)
        val_split: fraction for validation set (if validation_images not provided)
        compute_assignments: whether to compute nucleus-cell assignments
        seed: random seed for splitting
    """
    from cytonucs_losses import compute_nucleus_cell_assignments

    raw_images_dir = Path(raw_images_dir)
    raw_nucleus_masks_dir = Path(raw_nucleus_masks_dir)
    raw_cytoplasm_masks_dir = Path(raw_cytoplasm_masks_dir)
    output_dir = Path(output_dir)

    # Get file lists
    image_files = sorted(raw_images_dir.glob('*.tif'))

    print(f"Found {len(image_files)} samples")

    # Split train/val/test
    np.random.seed(seed)

    if validation_images is not None:
        # Use specified validation images
        val_files = [f for f in image_files if f.name in validation_images]
        remaining = [f for f in image_files if f.name not in validation_images]

        # Split remaining into train/test
        n_test = max(1, int(len(remaining) * test_split))
        indices = np.random.permutation(len(remaining))

        test_files = [remaining[i] for i in indices[:n_test]]
        train_files = [remaining[i] for i in indices[n_test:]]
    else:
        # Auto split
        indices = np.random.permutation(len(image_files))
        n_test = max(1, int(len(image_files) * test_split))
        n_val = max(1, int(len(image_files) * val_split))

        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test + n_val]
        train_indices = indices[n_test + n_val:]

        train_files = [image_files[i] for i in train_indices]
        val_files = [image_files[i] for i in val_indices]
        test_files = [image_files[i] for i in test_indices]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Create output directories
    for subset in ['train', 'val', 'test']:
        for subdir in ['images', 'nucleus_masks', 'cytoplasm_masks', 'assignments']:
            (output_dir / subset / subdir).mkdir(parents=True, exist_ok=True)

    # Process each subset
    for subset, subset_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print(f"\nProcessing {subset}...")

        for img_file in subset_files:
            filename = img_file.name

            # Copy image
            shutil.copy(img_file, output_dir / subset / 'images' / filename)

            # Copy nucleus mask
            shutil.copy(
                raw_nucleus_masks_dir / filename,
                output_dir / subset / 'nucleus_masks' / filename
            )

            # Copy cytoplasm mask
            shutil.copy(
                raw_cytoplasm_masks_dir / filename,
                output_dir / subset / 'cytoplasm_masks' / filename
            )

            # Compute assignments
            if compute_assignments:
                nucleus_mask = imageio.volread(raw_nucleus_masks_dir / filename)
                cytoplasm_mask = imageio.volread(raw_cytoplasm_masks_dir / filename)

                assignments, unassigned = compute_nucleus_cell_assignments(
                    nucleus_mask, cytoplasm_mask
                )

                # Save assignments
                assignment_file = output_dir / subset / 'assignments' / filename.replace('.tif', '.json')
                with open(assignment_file, 'w') as f:
                    json.dump(assignments, f, indent=2)

                if len(unassigned) > 0:
                    print(f"  {filename}: {len(unassigned)} unassigned nuclei")

    # Save split info
    split_info = {
        'train': [f.name for f in train_files],
        'val': [f.name for f in val_files],
        'test': [f.name for f in test_files],
        'seed': seed,
        'test_split': test_split,
        'val_split': val_split if validation_images is None else 'manual'
    }

    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\n✓ Dataset preparation complete!")
    print(f"✓ Split info saved to: {output_dir / 'split_info.json'}")
    print(f"✓ Train: {len(train_files)} samples")
    print(f"✓ Val: {len(val_files)} samples")
    print(f"✓ Test: {len(test_files)} samples")