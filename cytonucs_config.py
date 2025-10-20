"""
CytoNucs StarDist Configuration
Unified config for 2D/3D cytoplasm + nucleus segmentation
"""
from stardist.models import Config2D, Config3D
from stardist import Rays_GoldenSpiral
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
            grid=(1, 2, 2),  # subsampling grid
            anisotropy=None,
            n_channel_in=1,

            # Architecture
            backbone='unet',
            shared_encoder=True,

            # Training patches
            train_patch_size=(16, 128, 128),  # (Z,Y,X) for 3D or (Y,X) for 2D
            train_batch_size=2,

            # Loss weights
            lambda_prob_nucleus=1.0,
            lambda_dist_nucleus=1.0,
            lambda_prob_cytoplasm=1.0,
            lambda_dist_cytoplasm=1.0,
            lambda_containment=0.5,  # nucleus inside cell
            lambda_wbr=0.3,  # within-boundary regularization
            lambda_consistency=0.0,  # cell center near nuclei cluster
            enable_consistency=False,  # explicitly enable/disable

            # Containment loss params
            containment_margin=2.0,  # allow small margin (voxels)
            max_assignment_distance=5.0,  # max distance for nucleus-cell pairing (voxels)

            # WBR params
            wbr_sigma=2.0,  # gaussian smoothing for boundary detection

            # Training
            use_gpu=False,
            train_epochs=400,
            train_steps_per_epoch=100,
            train_learning_rate=3e-4,

            **kwargs
    ):
        self.ndim = ndim
        self.n_rays = n_rays
        self.grid = grid
        self.anisotropy = anisotropy if anisotropy is not None else (1,) * ndim
        self.n_channel_in = n_channel_in

        # Architecture
        self.backbone = backbone
        self.shared_encoder = shared_encoder

        # Patches
        self.train_patch_size = train_patch_size
        self.train_batch_size = train_batch_size

        # Loss weights
        self.lambda_prob_nucleus = lambda_prob_nucleus
        self.lambda_dist_nucleus = lambda_dist_nucleus
        self.lambda_prob_cytoplasm = lambda_prob_cytoplasm
        self.lambda_dist_cytoplasm = lambda_dist_cytoplasm
        self.lambda_containment = lambda_containment
        self.lambda_wbr = lambda_wbr
        self.lambda_consistency = lambda_consistency if enable_consistency else 0.0
        self.enable_consistency = enable_consistency

        # Containment
        self.containment_margin = containment_margin
        self.max_assignment_distance = max_assignment_distance

        # WBR
        self.wbr_sigma = wbr_sigma

        # Training
        self.use_gpu = use_gpu
        self.train_epochs = train_epochs
        self.train_steps_per_epoch = train_steps_per_epoch
        self.train_learning_rate = train_learning_rate

        # Generate rays based on anisotropy
        self.rays = Rays_GoldenSpiral(n_rays, anisotropy=self.anisotropy)

        # Store additional kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

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

        # Merge into single dict for constructor
        params = {
            **model_cfg,
            **training_cfg,
            **loss_weights,
            **loss_params
        }

        # Convert list to tuple for sizes
        if 'train_patch_size' in params:
            params['train_patch_size'] = tuple(params['train_patch_size'])
        if 'anisotropy' in params and params['anisotropy'] is not None:
            params['anisotropy'] = tuple(params['anisotropy'])
        if 'grid' in params and params['grid'] is not None:
            params['grid'] = tuple(params['grid'])

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