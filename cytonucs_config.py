"""
CytoNucs StarDist Configuration
Unified config for 2D/3D cytoplasm + nucleus segmentation
"""
from stardist.models import Config2D, Config3D

import numpy as np
import json
from pathlib import Path


class CytoNucsConfig:
    """
    Configuration for CytoNucs StarDist model.

    Key features:
    - Dual decoders: one for nuclei, one for cytoplasm
    - Multi-nuclear support: cells can contain multiple nuclei
    - Additional loss components: containment, WBR, consistency (optional)
    """

    def __init__(
            self,
            ndim=3,  # 2 or 3
            n_rays=128,
            grid=(1, 1),  # subsampling grid
            anisotropy=None,
            n_channel_in=1,

            # Architecture
            backbone='unet',
            shared_encoder=True,

            # Training patches
            train_patch_size=(512, 512),  # (Z,Y,X) for 3D or (Y,X) for 2D
            train_batch_size=2,
            mode='train',

            # Loss weights
            lambda_prob_nucleus=1.0,
            lambda_dist_nucleus=1.0,
            lambda_prob_cytoplasm=1.0,
            lambda_dist_cytoplasm=1.0,
            lambda_containment=0.5,
            w_containment_geometric=1.0,  # nucleus inside cell
            w_containment_distance=0.5,
            w_center_separation=0.2,
            w_cyto_support=0.1,
            lambda_wbr=0.3,  # within-boundary regularization
            lambda_variance=0.5,  # NEW: variance regularization
            lambda_consistency=0.0,
            enable_consistency=False,  # explicitly enable/disable
            lambda_area_cytoplasm=0.1,
            lambda_fragmentation=0.0,
            lambda_background_penalty=0.0,

            # Containment loss params
            containment_margin=2.0,  # allow small margin (voxels)
            max_assignment_distance=100.0,  # max distance for nucleus-cell pairing (voxels)
            prob_thresh_cytoplasm=0.4,
            prob_thresh_nucleus=0.5,
            nms_thresh=0.3,
            # WBR params
            wbr_sigma=2.0,  # gaussian smoothing for boundary detection
            wbr_warmup_epochs=5,

            # Distance normalization params
            median_radius_train=None,  # Медианный радиус из training data
            scale_factor_nucleus=None,  # Или напрямую scale_factor
            scale_factor_cytoplasm=None,
            median_max_dist_nucleus=None,  # НОВЫЙ: для нормализации в loss
            median_max_dist_cytoplasm=None,  # НОВЫЙ: для нормализации в loss

            # Training
            use_gpu=False,
            train_epochs=400,
            train_steps_per_epoch=100,
            train_learning_rate=3e-4,
            lr_decay_steps=700,  # НОВЫЙ: для optimizer
            lr_decay_rate=0.5,  # НОВЫЙ: для optimizer
            lr_staircase=True,  # НОВЫЙ: для optimizer
            save_every_n_epochs=50,
            gradient_clip_norm=10,

            **kwargs
    ):
        self.ndim = ndim
        self.n_rays = n_rays
        self.grid = grid
        self.nucleus_grid = kwargs.get('nucleus_grid', grid)
        self.cytoplasm_grid = kwargs.get('cytoplasm_grid', grid)
        self.anisotropy = anisotropy if anisotropy is not None else (1,) * ndim
        self.n_channel_in = n_channel_in
        self.gpu_config = kwargs.get('gpu_config', {
            'allow_growth': True,
            'memory_limit': None,
            'visible_devices': None
        })
        # Проверка и исправление anisotropy
        if len(self.anisotropy) != self.ndim:
            print(f"⚠️  Fixing anisotropy: {self.anisotropy} -> {(1,) * self.ndim}")
            self.anisotropy = (1,) * self.ndim

        # Architecture
        self.backbone = backbone
        self.shared_encoder = shared_encoder

        # Patches
        self.train_patch_size = train_patch_size
        self.train_batch_size = train_batch_size
        self.gradient_clip_norm = gradient_clip_norm

        # Loss weights
        self.lambda_prob_nucleus = lambda_prob_nucleus
        self.lambda_dist_nucleus = lambda_dist_nucleus
        self.lambda_prob_cytoplasm = lambda_prob_cytoplasm
        self.lambda_dist_cytoplasm = lambda_dist_cytoplasm
        self.lambda_containment = lambda_containment
        self.lambda_wbr = lambda_wbr
        self.lambda_area_cytoplasm = lambda_area_cytoplasm
        self.lambda_variance = lambda_variance
        self.lambda_consistency = lambda_consistency if enable_consistency else 0.0
        self.enable_consistency = enable_consistency
        self.lambda_fragmentation = lambda_fragmentation
        self.lambda_background_penalty = lambda_background_penalty
        self.mode = mode

        # Containment
        self.containment_margin = containment_margin
        self.w_containment_geometric = w_containment_geometric
        self.w_containment_distance = w_containment_distance
        self.w_center_separation = w_center_separation
        self.w_cyto_support = w_cyto_support
        self.max_assignment_distance = max_assignment_distance

        # WBR
        self.wbr_sigma = wbr_sigma
        self.wbr_warmup_epochs = wbr_warmup_epochs

        # Distance normalization
        self.median_radius_train = median_radius_train
        self.scale_factor_nucleus = scale_factor_nucleus
        self.scale_factor_cytoplasm = scale_factor_cytoplasm
        self.median_max_dist_nucleus = median_max_dist_nucleus
        self.median_max_dist_cytoplasm = median_max_dist_cytoplasm

        # Training
        self.use_gpu = use_gpu
        self.train_epochs = train_epochs
        self.train_steps_per_epoch = train_steps_per_epoch
        self.train_learning_rate = train_learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_staircase = lr_staircase
        self.save_every_n_epochs = save_every_n_epochs

        # Inference thresholds
        self.prob_thresh_cytoplasm = prob_thresh_cytoplasm
        self.prob_thresh_nucleus = prob_thresh_nucleus
        self.nms_thresh = nms_thresh
        self.lambda_fragmentation=lambda_fragmentation
        self.lambda_background_penalty=lambda_background_penalty

        if self.ndim == 2:
            # ДЛЯ 2D: StarDist использует Config2D, который внутри создаёт лучи
            # через Rays_GoldenSpiral с anisotropy=(1,1)
            # Мы делаем то же самое:

            from stardist.rays3d import Rays_GoldenSpiral as RaysBase

            # ХИТРОСТЬ: Rays_GoldenSpiral для 2D создаётся с ndim=2
            # Но класс Rays_GoldenSpiral ожидает 3D вершины.
            # Поэтому используем ОБХОДНОЙ ПУТЬ через numpy:

            import numpy as np

            # Создаём углы равномерно на окружности (как в StarDist для 2D)
            phi = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

            # Координаты на единичной окружности (Y, X)
            # В StarDist 2D первый луч направлен ВВЕРХ: [0, -1]
            vertices_2d = np.stack([
                np.sin(phi),  # Y координата
                -np.cos(phi)  # X координата (инвертирована для "вверх")
            ], axis=-1).astype(np.float32)

            # Простой класс-обёртка (как внутри StarDist)
            class Rays2D:
                def __init__(self, vertices):
                    self.vertices = np.array(vertices, dtype=np.float32)
                    self.ndim = 2

                def __len__(self):
                    return len(self.vertices)

                def dist_loss_weights(self, anisotropy=None):
                    # Для совместимости с StarDist
                    return np.ones(len(self.vertices), dtype=np.float32)

            self.rays = Rays2D(vertices_2d)

            print(f"✅ Created 2D rays (like train.py): {n_rays} rays")
            print(f"   Vertices shape: {self.rays.vertices.shape}")
            print(f"   Ray 0 (should point UP): {self.rays.vertices[0]}")

        else:  # 3D

            from stardist import Rays_GoldenSpiral

            self.rays = Rays_GoldenSpiral(n_rays, anisotropy=self.anisotropy)

            print(f"✅ Created 3D rays (like train.py): {n_rays} rays")
            print(f"   Anisotropy: {self.anisotropy}")
            print(f"   Vertices shape: {self.rays.vertices.shape}")
            print(f"   Ray 0: {self.rays.vertices[0]}")
        if kwargs:
            print(f"⚠️ DEBUG: kwargs contains: {list(kwargs.keys())}")
            if 'prob_thresh_cytoplasm' in kwargs:
                print(f"   WARNING: prob_thresh_cytoplasm in kwargs = {kwargs['prob_thresh_cytoplasm']}")
            if 'nms_thresh' in kwargs:
                print(f"   WARNING: nms_thresh in kwargs = {kwargs['nms_thresh']}")
            if 'gradient_clip_norm' in kwargs:
                print(f"   gradient_clip_norm in kwargs = {kwargs['gradient_clip_norm']}")
        # Store additional kwargs
        for k, v in kwargs.items():
            if k not in ['prob_thresh_nucleus', 'prob_thresh_cytoplasm', 'nms_thresh']:
                setattr(self, k, v)
            else:
                print(f"⚠️ Skipping kwargs['{k}'] = {v} (already set)")

    @classmethod
    def from_json(cls, json_path):
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        # Extract relevant sections
        model_cfg = config_dict.get('model', {})
        training_cfg = config_dict.get('training', {})
        loss_weights = config_dict.get('loss_weights', {})
        loss_params = config_dict.get('loss_params', {})
        inference_cfg = config_dict.get('inference', {})
        gpu_cfg = config_dict.get('gpu_config', {})
        # Merge into single dict for constructor
        params = {
            **model_cfg,
            **training_cfg,
            **loss_weights,
            **loss_params
        }
        if gpu_cfg:
            params['gpu_config'] = gpu_cfg
        # Add inference thresholds if available
        if 'prob_thresh_nucleus' in inference_cfg:
            params['prob_thresh_nucleus'] = inference_cfg['prob_thresh_nucleus']
        if 'prob_thresh_cytoplasm' in inference_cfg:
            params['prob_thresh_cytoplasm'] = inference_cfg['prob_thresh_cytoplasm']
        if 'nms_thresh' in inference_cfg:  # ← ДОБАВЛЕНО!
            params['nms_thresh'] = inference_cfg['nms_thresh']
        if 'steps_per_epoch' in training_cfg:
            training_cfg['train_steps_per_epoch'] = training_cfg['steps_per_epoch']
        # Convert list to tuple for sizes
        if 'train_patch_size' in params:
            params['train_patch_size'] = tuple(params['train_patch_size'])
        if 'anisotropy' in params and params['anisotropy'] is not None:
            params['anisotropy'] = tuple(params['anisotropy'])
        if 'grid' in params and params['grid'] is not None:
            params['grid'] = tuple(params['grid'])

        # Convert nucleus_grid and cytoplasm_grid to tuples if present
        if 'nucleus_grid' in params and params['nucleus_grid'] is not None:
            params['nucleus_grid'] = tuple(params['nucleus_grid'])
        if 'cytoplasm_grid' in params and params['cytoplasm_grid'] is not None:
            params['cytoplasm_grid'] = tuple(params['cytoplasm_grid'])

        return cls(**params)

    def to_base_config(self, head='nucleus'):
        """
        Convert to base StarDist Config2D/Config3D for compatibility.
        Used internally for each decoder head.
        """
        ConfigClass = Config3D if self.ndim == 3 else Config2D

        return ConfigClass(
            rays=self.rays,
            grid=self.grid,
            anisotropy=self.anisotropy,
            use_gpu=self.use_gpu,
            n_channel_in=self.n_channel_in,
            train_patch_size=self.train_patch_size,
            train_batch_size=self.train_batch_size,
        )

    def __repr__(self):
        attrs = [
            f"ndim={self.ndim}",
            f"n_rays={self.n_rays}",
            f"grid={self.grid}",
            f"shared_encoder={self.shared_encoder}",
            f"patch_size={self.train_patch_size}",
            f"batch_size={self.train_batch_size}",
            f"consistency={'ON' if self.enable_consistency else 'OFF'}",
        ]
        return f"CytoNucsConfig({', '.join(attrs)})"


# Factory function for easy configuration
def create_cytonucs_config_3d(
        anisotropy=(1, 4, 4),
        n_rays=128,
        patch_size=(12, 96, 96),
        batch_size=2,
        **kwargs
):
    """Create a typical 3D CytoNucs config"""
    grid = tuple(1 if a > 1.5 else 4 for a in anisotropy)

    return CytoNucsConfig(
        ndim=3,
        n_rays=n_rays,
        grid=grid,
        anisotropy=anisotropy,
        train_patch_size=patch_size,
        train_batch_size=batch_size,
        **kwargs
    )


def create_cytonucs_config_2d(
        n_rays=128,
        patch_size=(256, 256),
        batch_size=4,
        **kwargs
):
    """Create a typical 2D CytoNucs config"""
    return CytoNucsConfig(
        ndim=2,
        n_rays=n_rays,
        grid=(1, 1),
        anisotropy=(1, 1),
        train_patch_size=patch_size,
        train_batch_size=batch_size,
        **kwargs
    )