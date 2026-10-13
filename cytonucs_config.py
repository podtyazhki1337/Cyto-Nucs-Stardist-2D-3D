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
    - Per-head ray sets and grids, derived canonically from object extents
    """

    # Public fixes_v1 settings: canonical JSON section and effective default.
    # Keep config_cytonucs.json and the loader regression test in sync.
    # The network has two roles for nested objects: the inner one ('nucleus') and the outer one
    # ('cytoplasm'). model.levels names the object levels of a run, inner first; each takes a role
    # (level_roles) and its masks are read from '<level>_masks'.
    ROLES = ('nucleus', 'cytoplasm')

    FIXES_SETTINGS = {
        'evaluation': dict(refinement_stage_audit=False, fullval_train_volumes=0),
        'model': dict(unet_pool_schedule=None, unet_kernel_schedule=None,
                      refinement_depth=0, refinement_gradient_scale=0.0, mask_quality_head=False,
                      mask_quality_pooling='global', refinement_pool='isotropic',
                      refinement_full_resolution=False, unet_norm_groups=0,
                      levels=['nucleus', 'cytoplasm']),
        'training': dict(seed=42, image_batch_size=1, prompts_per_crop=None, nucleus_prob_grid_normalize=False,
                         log_every_steps=20, prefetch_workers=3, prefetch_depth=3,
                         predicted_prompt_fraction=0.0, predicted_prompt_start_epoch=10,
                         detector_prompt_slots=0, cell_predicted_nucleus_fraction=0.0,
                         predicted_prompt_sample_peaks=False, reset_best_on_resume=False,
                         dist_target_stride=None, tile_sampling_by_objects=0.0, train_region_crops=False,
                         partial_annotation=False),
        'inference': dict(refinement_mode='mask', refine_feature_cache_windows=1,
                          mask_nms_nucleus=0.8, mask_nms_cytoplasm=0.5,
                          pair_min_overlap=0.1, pair_min_mask_iou=0.25,
                          mask_thresh_nucleus=None, mask_thresh_cytoplasm=None,
                          tiling_fixed_axes=None, refinement_patch_size=None,
                          refinement_adaptive_roi=False, proposal_mode='stardist',
                          cell_roi_expansion_steps=0,
                          proposal_peak_window_nucleus=None, proposal_peak_window_cytoplasm=None,
                          proposal_border_nucleus=None, proposal_border_cytoplasm=None,
                          mask_containment_threshold=0.0, mask_quality_min=0.0,
                          mask_quality_use_scores=True, mask_overlap_nucleus=0.0,
                          mask_overlap_cytoplasm=0.0, mask_assembly_nucleus='voxel',
                          mask_assembly_cytoplasm='voxel', mask_min_remaining_fraction=0.5,
                          nms_candidate_spacing=0.12, nms_border_exclude_xy=2,
                          residual_proposals=False, refinement_roi_snap=False),
        'augmentation': dict(cyto_center_prompt_prob=0.25, negative_prompt_prob=0.1,
                             prompt_center_jitter=0.1, prompt_sigma_jitter=0.1,
                             repeat_cell_prompt_prob=0.0, prompt_inside_target=False,
                             neighbor_cell_prompt_prob=0.0, cell_z_pair_prob=0.0,
                             nucleus_z_pair_prob=0.0),
        'loss_params': dict(nucleus_prob_object_weight=0.0, cytoplasm_prob_object_weight=0.0,
                            prob_object_peak_weight=0.0, mask_quality_rank_weight=0.0, object_losses_include_crop_cut=False,
                            mask_overlap_loss='dice', object_losses_min_area_from_gate=False,
                            mask_ce_empty_weight=0.25, prompt_consistency_weight=0.0, mask_quality_weight=0.0,
                            mask_quality_target='grid', mask_quality_ignore_boundary_targets=False,
                            mask_ce_class_balance=False, cell_prompt_separation_weight=0.0,
                            cell_z_dice_weight=0.0, nucleus_z_dice_weight=0.0,
                            semantic_foreground_weight=0.0),
    }

    def __init__(
            self,
            ndim=3,  # 2 or 3
            n_rays=128,
            grid=(1, 1),  # subsampling grid
            anisotropy=None,
            n_channel_in=1,
            unet_n_depth=4,
            unet_n_filter_base=32,
            net_conv_after_unet=128,

            # Architecture

            shared_encoder=True,

            # Training patches
            train_patch_size=(512, 512),  # (Z,Y,X) for 3D or (Y,X) for 2D
            train_batch_size=2,
            mode='train',
            fg_crop_prob=0.5,
            # Loss weights
            lambda_prob_nucleus=1.0,
            lambda_dist_nucleus=1.0,
            lambda_prob_cytoplasm=1.0,
            lambda_dist_cytoplasm=1.0,
            lambda_containment=0.5,
            w_containment_geometric=1.0,  # nucleus inside cell
            w_containment_distance=0.0,
            w_center_separation=0.2,
            w_cyto_support=0.1,
            lambda_wbr=0.3,  # within-boundary regularization
            lambda_variance=0.5,
            lambda_consistency=0.0,
            enable_consistency=False,  # explicitly enable/disable
            lambda_area_cytoplasm=0.1,
            lambda_fragmentation=0.0,
            lambda_background_penalty=0.0,

            # Containment loss params
            containment_margin=2.0,  # allow small margin (voxels)
            max_assignment_distance=100.0,  # max distance for nucleus-cell pairing
            prob_thresh_cytoplasm=0.4,
            prob_thresh_nucleus=0.5,
            nms_thresh=0.3,
            # WBR params
            wbr_sigma=2.0,  # gaussian smoothing for boundary detection
            wbr_warmup_epochs=5,
            min_eval_epoch=1,  # start evaluating after this epoch
            # Distance normalization params
            median_radius_train=None,
            scale_factor_nucleus=None,
            scale_factor_cytoplasm=None,
            median_max_dist_nucleus=None,
            median_max_dist_cytoplasm=None,
            mask_thresh_range=(0.1, 0.9),
            threshold_opt_splits="val",
            eval_splits="test",
            dist_loss='mae',
            huber_delta=1.0,
            # Training
            use_gpu=False,
            train_epochs=400,
            train_steps_per_epoch=100,
            train_learning_rate=3e-4,
            lr_decay_steps=700,
            lr_decay_rate=0.5,
            lr_staircase=True,
            save_every_n_epochs=50,
            gradient_clip_norm=10,
            save_pred_tiff=True,
            lambda_cyto_mask=0.0,
            lambda_nuc_mask=0.0,
            lr_warmup_epochs=0,

            mask_refinement_filters=64,
            mask_thresh=0.5,

            bbox_padding_factor=1.5,

            **kwargs
    ):
        self.ndim = ndim
        self.n_rays = n_rays
        self.unet_n_depth = unet_n_depth
        self.unet_n_filter_base = unet_n_filter_base
        self.net_conv_after_unet = net_conv_after_unet

        def _coerce_grid(g):
            g = tuple(g) if g is not None else (1,) * ndim
            return g if len(g) == ndim else (1,) * ndim

        self.grid = _coerce_grid(grid)
        self.nucleus_grid = _coerce_grid(kwargs.get('nucleus_grid', grid))
        self.cytoplasm_grid = _coerce_grid(kwargs.get('cytoplasm_grid', grid))
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
        self.shared_encoder = shared_encoder
        # Patches
        self.train_patch_size = train_patch_size
        self.train_batch_size = train_batch_size
        self.gradient_clip_norm = gradient_clip_norm
        self.threshold_opt_splits = threshold_opt_splits
        self.mask_thresh_range = mask_thresh_range
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
        self.dist_loss = dist_loss
        self.huber_delta = huber_delta
        self.mode = mode
        self.lr_warmup_epochs = lr_warmup_epochs
        # Containment
        self.containment_margin = containment_margin
        self.w_containment_geometric = w_containment_geometric
        self.w_containment_distance = w_containment_distance
        self.w_center_separation = w_center_separation
        self.w_cyto_support = w_cyto_support
        self.max_assignment_distance = max_assignment_distance
        # Refinement params
        self.lambda_cyto_mask = lambda_cyto_mask
        self.lambda_nuc_mask = lambda_nuc_mask



        self.mask_refinement_filters = mask_refinement_filters
        self.mask_thresh = mask_thresh
        self.bbox_padding_factor = bbox_padding_factor

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
        self.min_eval_epoch = min_eval_epoch
        self.fg_crop_prob = fg_crop_prob
        # Inference thresholds
        self.prob_thresh_cytoplasm = prob_thresh_cytoplasm
        self.prob_thresh_nucleus = prob_thresh_nucleus
        self.nms_thresh = nms_thresh
        self.save_pred_tiff = save_pred_tiff
        self.eval_splits=eval_splits
        # ============================================================
        # ЛУЧИ
        # ============================================================
        # Канон StarDist 3D (examples/3D/2_training.ipynb):
        #     extents    = calculate_extents(Y)            # медиана bbox объектов
        #     anisotropy = tuple(np.max(extents)/extents)  # АСПЕКТ ОБЪЕКТА
        #     grid       = tuple(1 if a > 1.5 else 2 for a in anisotropy)
        #     rays       = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)
        # То есть anisotropy у StarDist — форма объектов в вокселях, а НЕ
        # физический шаг между срезами.
        #
        # Rays_GoldenSpiral делает verts = dir/anisotropy, поэтому она
        # распределяет направления: сколько лучей ляжет в плоскость объекта, а
        # сколько упрётся в его верх и низ. При физической [5,1,1] на плоской
        # 2D-культуре контур цитоплазмы (полутолщина ~2.5 вокселя, радиус
        # ~100 px) задают 12 лучей из 96 — остальные 84 меряют толщину слоя и
        # забирают 2/3 MAE. При анизотропии, равной аспекту (100/2.5 = 40),
        # контур задают 68 лучей и все длины лежат в одном масштабе.
        #
        # Ядро и цитоплазма — объекты разной формы, наборы лучей раздельные.
        # n_rays общий, поэтому формы выходов голов не меняются.
        self.rays_anisotropy_nucleus = self._coerce_aniso(
            kwargs.get('rays_anisotropy_nucleus'), self.anisotropy)
        self.rays_anisotropy_cytoplasm = self._coerce_aniso(
            kwargs.get('rays_anisotropy_cytoplasm'), self.anisotropy)

        self._build_rays()

        if kwargs:
            print(f"⚠️ DEBUG: kwargs contains: {list(kwargs.keys())}")
        # Store additional kwargs
        for k, v in kwargs.items():
            if k not in ['prob_thresh_nucleus', 'prob_thresh_cytoplasm',
                         'nms_thresh', 'nucleus_grid', 'cytoplasm_grid',
                         'rays_anisotropy_nucleus', 'rays_anisotropy_cytoplasm']:
                setattr(self, k, v)

        for defaults in self.FIXES_SETTINGS.values():
            for key, default in defaults.items():
                if not hasattr(self, key):
                    setattr(self, key, default)
        # Legacy train_batch_size meant prompts, not independent images.
        if self.prompts_per_crop is None:
            self.prompts_per_crop = self.train_batch_size
        self._validate_runtime_settings()

    def mask_threshold(self, head, calibrated=None):
        """Mask threshold of one head: explicit value, calibrated value for 'auto', else mask_thresh.

        'auto' without a calibration (training losses, calibration passes) uses mask_thresh.
        """
        value = getattr(self, 'mask_thresh_' + head)
        if value == 'auto':
            value = (calibrated or {}).get(head)
        return float(self.mask_thresh if value is None else value)

    def has_level(self, role):
        """Whether the run has an object level in this role ('nucleus' = inner, 'cytoplasm' = outer)."""
        return role in self.level_roles

    def _validate_runtime_settings(self):
        import re
        levels = self.levels
        if (not isinstance(levels, (list, tuple)) or not levels or len(set(levels)) != len(levels) or
                any(not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name) for name in levels)):
            raise ValueError('levels must be a non-empty list of distinct names (lower-case letters, digits, _), inner level first')
        if len(levels) > 2:
            raise ValueError(f'levels {list(levels)}: more than two nested levels are not implemented yet')
        if set(levels) == set(self.ROLES) and list(levels) != list(self.ROLES):
            raise ValueError('levels are listed inner first: ["nucleus", "cytoplasm"]')
        self.levels = list(levels)
        # Two levels: inner and outer role. One level: the outer role if it is called 'cytoplasm',
        # else the inner one (a plain object level).
        self.level_roles = (dict(zip(self.ROLES, levels)) if len(levels) == 2 else
                            {'cytoplasm' if levels[0] == 'cytoplasm' else 'nucleus': levels[0]})
        for key in ('detector_prompt_slots', 'cell_roi_expansion_steps', 'fullval_train_volumes', 'unet_norm_groups'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if self.detector_prompt_slots > max(0, self.prompts_per_crop - 2):
            raise ValueError('detector_prompt_slots requires two additional protected prompt slots')
        if self.detector_prompt_slots and not self.predicted_prompt_sample_peaks:
            raise ValueError('detector_prompt_slots requires predicted_prompt_sample_peaks')
        if self.cell_predicted_nucleus_fraction and not self.predicted_prompt_sample_peaks:
            raise ValueError('cell_predicted_nucleus_fraction requires predicted_prompt_sample_peaks')
        if not isinstance(self.mask_quality_ignore_boundary_targets, bool):
            raise ValueError('mask_quality_ignore_boundary_targets must be boolean')
        for head in ('nucleus', 'cytoplasm'):
            win = getattr(self, 'proposal_peak_window_' + head)
            if win is not None and (not isinstance(win, (list, tuple)) or len(win) != self.ndim or
                    any(isinstance(v, bool) or not isinstance(v, int) or v < 1 or v % 2 == 0 for v in win)):
                raise ValueError('proposal_peak_window must contain one positive odd integer per axis')
            border = getattr(self, 'proposal_border_' + head)
            if border is not None and (isinstance(border, bool) or not isinstance(border, int) or border < 0):
                raise ValueError('proposal_border must be a nonnegative integer or null')
        if self.mask_overlap_loss not in ('dice', 'lovasz'):
            raise ValueError('mask_overlap_loss must be dice or lovasz')
        if self.object_losses_min_area_from_gate and self.mask_quality_min != 'auto':
            raise ValueError('object_losses_min_area_from_gate requires mask_quality_min="auto" (calibrated gates)')
        if self.mask_quality_target not in ('grid', 'component'):
            raise ValueError('mask_quality_target must be grid or component')
        for key in ('mask_assembly_nucleus', 'mask_assembly_cytoplasm'):
            if getattr(self, key) not in ('voxel', 'object', 'object_quality', 'object_quality_containment'):
                raise ValueError(f'{key} must be voxel, object, object_quality or object_quality_containment')
            if getattr(self, key) == 'object_quality_containment' and not 0 < self.mask_containment_threshold <= 1:
                raise ValueError('object_quality_containment requires mask_containment_threshold in (0,1]')
        for key in ('predicted_prompt_sample_peaks', 'reset_best_on_resume',
                    'mask_quality_use_scores', 'prompt_inside_target', 'mask_ce_class_balance', 'refinement_stage_audit', 'nucleus_prob_grid_normalize',
                    'residual_proposals', 'object_losses_include_crop_cut', 'refinement_roi_snap',
                    'object_losses_min_area_from_gate', 'refinement_full_resolution', 'train_region_crops',
                    'partial_annotation'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f'{key} must be boolean')
        # Residual prompts are added after the first object assembly into free voxels only.
        if self.residual_proposals and (self.proposal_mode != 'peaks' or self.refinement_mode != 'mask' or all(
                getattr(self, 'mask_assembly_' + h) == 'voxel' for h in ('nucleus', 'cytoplasm'))):
            raise ValueError('residual_proposals requires proposal_mode=peaks, refinement_mode=mask '
                             'and an object mask_assembly_* mode')
        if self.refinement_full_resolution and all(
                all(int(g) == 1 for g in getattr(self, h + '_grid')) for h in ('nucleus', 'cytoplasm')):
            raise ValueError('refinement_full_resolution requires a head grid coarser than the input')
        if self.mask_quality_pooling not in ('global', 'mask'):
            raise ValueError('mask_quality_pooling must be global or mask')
        if self.refinement_pool not in ('isotropic', 'auto'):
            raise ValueError('refinement_pool must be isotropic or auto')
        spacing = self.nms_candidate_spacing
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('nms_candidate_spacing must be a finite positive number')
        if isinstance(self.nms_border_exclude_xy, bool) or not isinstance(self.nms_border_exclude_xy, int) or self.nms_border_exclude_xy < 0:
            raise ValueError('nms_border_exclude_xy must be a nonnegative integer')
        if self.refinement_stage_audit and self.refinement_mode != 'mask':
            raise ValueError('refinement_stage_audit requires refinement_mode=mask')
        if not isinstance(self.shared_encoder, bool):
            raise ValueError('shared_encoder must be boolean')
        if not isinstance(self.mask_quality_head, bool):
            raise ValueError('mask_quality_head must be boolean')
        if self.mask_quality_weight and not self.mask_quality_head:
            raise ValueError('mask_quality_weight requires mask_quality_head=true')
        if self.mask_quality_rank_weight and not self.mask_quality_weight:
            raise ValueError('mask_quality_rank_weight requires mask_quality_weight > 0 (shared quality targets)')
        if isinstance(self.refinement_depth, bool) or not isinstance(self.refinement_depth, int) or not 0 <= self.refinement_depth <= 3:
            raise ValueError('refinement_depth must be an integer in [0,3]')
        if not isinstance(self.refinement_adaptive_roi, bool):
            raise ValueError('refinement_adaptive_roi must be boolean')
        if self.proposal_mode not in ('stardist', 'peaks'):
            raise ValueError('proposal_mode must be stardist or peaks')
        if self.proposal_mode == 'peaks' and self.refinement_mode != 'mask':
            raise ValueError('peak proposals require refinement_mode=mask')
        if self.unet_norm_groups and int(getattr(self, 'unet_n_filter_base', 32)) % self.unet_norm_groups:
            raise ValueError('unet_n_filter_base must be divisible by unet_norm_groups')
        stride = self.dist_target_stride
        if stride is not None:
            if (not isinstance(stride, (list, tuple)) or len(stride) != self.ndim or
                    any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in stride)):
                raise ValueError('dist_target_stride must be null or one positive integer per axis')
            for head in ('nucleus', 'cytoplasm'):
                if any((p // g) % s for p, g, s in zip(self.train_patch_size, getattr(self, head + '_grid'), stride)):
                    raise ValueError('train patch on the head grid must be divisible by dist_target_stride')
            if self.w_center_separation:
                raise ValueError('dist_target_stride requires w_center_separation=0 (it reads ray targets on the head grid)')
        if self.refinement_patch_size is not None:
            if len(self.refinement_patch_size) != self.ndim or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.refinement_patch_size):
                raise ValueError('refinement_patch_size: one nonnegative integer per axis; 0 means full axis')
        if self.refinement_depth:
            from cytonucs_geometry import refinement_factors
            for head in ('nucleus', 'cytoplasm'):
                factors = refinement_factors(self, head)
                if any((p // g) % f for p, g, f in zip(self.train_patch_size, getattr(self, head + '_grid'), factors)):
                    raise ValueError(f'train patch/head grid must be divisible by the refiner pooling {factors}')
        from cytonucs_geometry import architecture_plan
        architecture_plan(self)
        for key in ('image_batch_size', 'prompts_per_crop', 'log_every_steps',
                    'prefetch_workers', 'prefetch_depth', 'refine_feature_cache_windows', 'predicted_prompt_start_epoch'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{key} must be a positive integer, got {value!r}')
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        for key in ('cell_predicted_nucleus_fraction', 'mask_nms_nucleus', 'mask_nms_cytoplasm', 'pair_min_overlap',
                    'pair_min_mask_iou', 'cyto_center_prompt_prob', 'negative_prompt_prob',
                    'mask_ce_empty_weight', 'mask_thresh_nucleus', 'mask_thresh_cytoplasm',
                    'refinement_gradient_scale', 'repeat_cell_prompt_prob', 'neighbor_cell_prompt_prob', 'cell_z_pair_prob', 'nucleus_z_pair_prob', 'mask_containment_threshold', 'predicted_prompt_fraction',
                    'mask_overlap_nucleus', 'mask_overlap_cytoplasm', 'mask_min_remaining_fraction',
                    'tile_sampling_by_objects'):
            value = getattr(self, key)
            if key.startswith('mask_thresh_') and (value is None or value == 'auto'):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{key} must be a number in [0, 1], got {value!r}')
        if 'auto' in (self.mask_thresh_nucleus, self.mask_thresh_cytoplasm) and self.refinement_mode != 'mask':
            raise ValueError('mask_thresh_*=auto calibrates prompted masks and requires refinement_mode=mask')
        if self.mask_quality_min == 'auto':
            # Per-head gates are calibrated by full validation and stored with checkpoints.
            if not self.mask_quality_head or any(
                    not getattr(self, 'mask_assembly_' + h).startswith('object_quality')
                    for h in ('nucleus', 'cytoplasm')):
                raise ValueError('mask_quality_min=auto requires mask_quality_head and object_quality assembly for both heads')
            if self.mode == 'train' and not self.refinement_stage_audit:
                raise ValueError('mask_quality_min=auto calibrates from refinement_stage_audit during training')
        elif (isinstance(self.mask_quality_min, bool) or not isinstance(self.mask_quality_min, (int, float))
              or not np.isfinite(self.mask_quality_min) or not 0 <= self.mask_quality_min <= 1):
            raise ValueError(f'mask_quality_min must be a number in [0, 1] or "auto", got {self.mask_quality_min!r}')
        for key in ('nucleus_prob_object_weight', 'cytoplasm_prob_object_weight', 'prob_object_peak_weight',
                    'mask_quality_rank_weight', 'gradient_clip_norm', 'prompt_center_jitter', 'prompt_sigma_jitter', 'prompt_consistency_weight', 'mask_quality_weight', 'cell_prompt_separation_weight', 'cell_z_dice_weight', 'nucleus_z_dice_weight',
                    'semantic_foreground_weight'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.cell_prompt_separation_weight > 0 and self.prompts_per_crop < 2:
            raise ValueError('cell_prompt_separation_weight requires prompts_per_crop >= 2')
        if (self.cell_z_pair_prob or self.cell_z_dice_weight) and self.ndim != 3:
            raise ValueError('Cell Z training requires ndim=3')
        if self.cell_z_pair_prob and self.prompts_per_crop < 2:
            raise ValueError('cell_z_pair_prob requires prompts_per_crop >= 2')
        if (self.nucleus_z_pair_prob or self.nucleus_z_dice_weight) and self.ndim != 3:
            raise ValueError('Nucleus Z training requires ndim=3')
        if self.nucleus_z_pair_prob and self.prompts_per_crop < 2:
            raise ValueError('nucleus_z_pair_prob requires prompts_per_crop >= 2')
        if self.refinement_mode not in ('mask', 'polyhedron', 'legacy_union'):
            raise ValueError(f'Unknown refinement_mode: {self.refinement_mode!r}')

    # ============================================================
    # === ЛУЧИ И АНИЗОТРОПИЯ ===
    # ============================================================

    def _coerce_aniso(self, value, fallback):
        """Приводит анизотропию к кортежу длины ndim. None и мусор -> fallback."""
        if value is None:
            return tuple(float(v) for v in fallback)
        try:
            a = tuple(float(v) for v in value)
        except TypeError:
            a = (float(value),) + (1.0,) * (self.ndim - 1)
        if len(a) != self.ndim or any(v <= 0 for v in a):
            return tuple(float(v) for v in fallback)
        return a

    def _poolable_axes(self):
        """Оси, по которым декодер реально делает pooling.

        Модель считает pool_size как `2 if a < 2*min(a) else 1` по
        self.anisotropy. Grid по НЕпулящейся оси обязан оставаться 1: иначе GT
        прорежен, а выход сети нет, и формы разъедутся. Дублируем правило
        здесь, чтобы канонический grid не сломал сеть."""
        a = tuple(float(v) for v in self.anisotropy)
        m = min(a)
        return tuple(v < 2.0 * m for v in a)

    def canonical_grid(self, aniso):
        """grid = tuple(1 if a > 1.5 else 2 for a in anisotropy) — канон
        StarDist, с поправкой на непулящиеся оси сети."""
        ok = self._poolable_axes()
        return tuple(2 if (a <= 1.5 and ok[i]) else 1
                     for i, a in enumerate(aniso))

    def _build_rays(self):
        """Строит наборы лучей из текущих rays_anisotropy_*.
        Вызывается из __init__ и повторно из set_anisotropy_from_labels."""
        n_rays = int(self.n_rays)

        if self.ndim == 2:
            # 2D: направления равномерно по окружности, анизотропия не нужна.
            # Первый луч вверх, как в StarDist 2D.
            phi = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
            vertices_2d = np.stack([np.sin(phi), -np.cos(phi)],
                                   axis=-1).astype(np.float32)

            class Rays2D:
                def __init__(self, vertices):
                    self.vertices = np.array(vertices, dtype=np.float32)
                    self.ndim = 2

                def __len__(self):
                    return len(self.vertices)

                def dist_loss_weights(self, anisotropy=None):
                    return np.ones(len(self.vertices), dtype=np.float32)

            self.rays_nucleus = Rays2D(vertices_2d)
            self.rays_cytoplasm = Rays2D(vertices_2d)
            self.rays = self.rays_nucleus
            print(f"✅ 2D rays: {n_rays}, ray 0 = {self.rays.vertices[0]}")
            return

        from stardist import Rays_GoldenSpiral

        self.rays_nucleus = Rays_GoldenSpiral(
            n_rays, anisotropy=self.rays_anisotropy_nucleus)
        self.rays_cytoplasm = Rays_GoldenSpiral(
            n_rays, anisotropy=self.rays_anisotropy_cytoplasm)
        # Обратная совместимость: старый код, читающий config.rays, получает
        # набор ЯДРА. Наборы больше не совпадают — все новые обращения должны
        # брать rays_nucleus / rays_cytoplasm явно.
        self.rays = self.rays_nucleus

        print(f"✅ 3D rays: {n_rays} на голову, наборы РАЗНЫЕ")
        print(f"   nucleus   anisotropy={tuple(round(v, 1) for v in self.rays_anisotropy_nucleus)}")
        print(f"   cytoplasm anisotropy={tuple(round(v, 1) for v in self.rays_anisotropy_cytoplasm)}")

    def set_anisotropy_from_labels(self, nucleus_labels, cytoplasm_labels,
                                   update_grid=True):
        """Замер геометрии по GT: анизотропия -> лучи -> гриды -> масштаб.

        Канон (examples/3D/2_training.ipynb):
            extents    = calculate_extents(Y)
            anisotropy = tuple(np.max(extents)/extents)
            grid       = tuple(1 if a > 1.5 else 2 for a in anisotropy)

        ОТЛИЧИЕ ОТ ПРЕЖНЕЙ ВЕРСИИ — фильтр обрезки ПО ОСЯМ, а не по объекту
        целиком. Прежде объект попадал в замер, только если не касался грани
        НИ ПО ОДНОЙ оси; на тайлах 16x448x448 это оставляло единицы объектов
        из тысяч, срабатывал фолбэк «по ВСЕМ», и обрезка входила в замер
        асимметрично: по Z тайл в 16 срезов режет каждую клетку, по XY 448 —
        почти нет. Z-размер занижался, аспект раздувался (замерено 25 при
        физических 5). Объект, срезанный по X, даёт корректный размер по Z —
        поэтому решение принимается для каждой оси отдельно.

        Анизотропия ОКРУГЛЯЕТСЯ ДО ЦЕЛОГО: это аспект объекта в вокселях,
        дробная точность в нём бессмысленна, а целое значение делает
        geometry.json воспроизводимым между прогонами.

        median_max_dist считается как половина наибольшего измеренного
        размера, без box-аппроксимации лучей: прежняя формула
        t_k = min_a(h_a/|v_ka|) меряла расстояние до границы ПАРАЛЛЕЛЕПИПЕДА
        и завышала масштаб (107 против замеренных по star_dist3D 87).
        dist_bias_init не выставляется вовсе — канонический StarDist обходится
        нулевым bias.
        """
        if self.ndim != 3:
            self._build_rays()
            return self.rays_anisotropy_nucleus, self.rays_anisotropy_cytoplasm

        from scipy.ndimage import find_objects

        base = tuple(float(v) for v in self.anisotropy)
        base_grid = tuple(int(v) for v in self.grid)
        nd = self.ndim
        measured_ext = {}
        # A level the run does not have (model.levels) has no objects: it takes the geometry of
        # the level the run has, so grids, rays and scales stay defined and equal.
        if not self.has_level('nucleus'):
            nucleus_labels = cytoplasm_labels
        if not self.has_level('cytoplasm'):
            cytoplasm_labels = nucleus_labels

        def _measure(lbls):
            """Медианный размер объекта по каждой оси НЕЗАВИСИМО."""
            if lbls is None or len(lbls) == 0:
                return None

            ext_in = [[] for _ in range(nd)]  # не срезан по этой оси
            ext_all = [[] for _ in range(nd)]  # все
            for lbl in lbls:
                lbl = np.asarray(lbl)
                if lbl.ndim != nd or lbl.max() == 0:
                    continue
                sh = lbl.shape
                for sl in find_objects(lbl):
                    if sl is None:
                        continue
                    for a in range(nd):
                        e = sl[a].stop - sl[a].start
                        ext_all[a].append(e)
                        if sl[a].start > 0 and sl[a].stop < sh[a]:
                            ext_in[a].append(e)

            if not any(len(v) for v in ext_all):
                return None

            ext, src = [], []
            for a in range(nd):
                if len(ext_in[a]) >= 30:
                    ext.append(float(np.median(ext_in[a])))
                    src.append(f"ось{a}: {len(ext_in[a])}/{len(ext_all[a])}")
                else:
                    ext.append(float(np.median(ext_all[a])))
                    src.append(f"ось{a}: ВСЕ {len(ext_all[a])} (целых мало)")
            ext = np.asarray(ext, np.float64)
            if np.any(ext <= 0):
                return None
            print(f"      замер {', '.join(src)}")
            # Аспект объекта, ЦЕЛЫЙ, не меньше 1.
            aniso = tuple(max(1, int(round(float(np.max(ext) / e)))) for e in ext)
            return aniso, ext

        print("\n📐 Замер геометрии по GT (фильтр обрезки по осям):")
        for name, lbls in (('nucleus', nucleus_labels),
                           ('cytoplasm', cytoplasm_labels)):
            key = f'rays_anisotropy_{name}'
            res = _measure(lbls)
            if getattr(self, key) != base:
                print(f"   {key}: задана в конфиге "
                      f"{tuple(getattr(self, key))}, замер не применяется")
                a = getattr(self, key)
                if res is not None:
                    measured_ext[name] = res[1]
            elif res is None:
                print(f"   {key}: объектов не найдено, оставляю {base}")
                a = getattr(self, key)
            else:
                a, ext = res
                measured_ext[name] = ext
                setattr(self, key, a)
                print(f"   {name}: медианный bbox "
                      f"{tuple(int(round(v)) for v in ext)} -> anisotropy {a}")

            if update_grid:
                cur = tuple(int(v) for v in getattr(self, f'{name}_grid'))
                if cur != base_grid:
                    print(f"   {name}_grid: {cur} (из конфига, замер не применён)")
                else:
                    g = self.canonical_grid(a)
                    setattr(self, f'{name}_grid', g)
                    print(f"   {name}_grid: {g} (канон, с поправкой на pooling)")

        self._build_rays()

        # --- МАСШТАБ ОБЪЕКТА ---
        for name in ('nucleus', 'cytoplasm'):
            ext = measured_ext.get(name)
            if ext is None:
                continue
            # Половина наибольшего измеренного размера — это и есть длина
            # самого длинного луча из центра. Идёт в r_hint промптов,
            # в нормировку _geo_prior и в окно preselection на инференсе.
            setattr(self, f'median_max_dist_{name}',
                    float(int(round(float(np.linalg.norm(ext)) / 2.0))))
            # Канон: нулевой bias у dist-головы.

            print(f"   {name}: median_max_dist -> "
                  f"{getattr(self, f'median_max_dist_{name}'):.0f} px")

        geo_path = getattr(self, 'geometry_path', None)
        if geo_path:
            p = Path(geo_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                json.dump({
                    'n_rays': int(self.n_rays),
                    'rays_anisotropy_nucleus': list(self.rays_anisotropy_nucleus),
                    'rays_anisotropy_cytoplasm': list(self.rays_anisotropy_cytoplasm),
                    'nucleus_grid': list(self.nucleus_grid),
                    'cytoplasm_grid': list(self.cytoplasm_grid),
                    'median_max_dist_nucleus': float(self.median_max_dist_nucleus),
                    'median_max_dist_cytoplasm': float(self.median_max_dist_cytoplasm),
                }, f, indent=2)
            print(f"   💾 geometry.json сохранён: {p}")

        return self.rays_anisotropy_nucleus, self.rays_anisotropy_cytoplasm

    # ============================================================
    # === ЗАГРУЗКА ===
    # ============================================================

    @classmethod
    def from_json(cls, json_path, checkpoint_path=None):
        """Load configuration from JSON file.

        Плюс подхватывает geometry.json рядом с чекпоинтом, если он есть.
        Без этого inference/optimize_thresholds строили лучи из ФИЗИЧЕСКОЙ
        anisotropy, тогда как веса обучены под ЗАМЕРЕННУЮ (аспект объекта):
        формы совпадают, n_rays тот же, ошибок нет — но dist интерпретируется
        другим набором направлений, и полиэдры выходят чужой геометрии.
        Файл пишет set_anisotropy_from_labels в конце обучающего замера.
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        for expected_section, settings in cls.FIXES_SETTINGS.items():
            for section, values in config_dict.items():
                if section != expected_section and isinstance(values, dict):
                    misplaced = set(settings).intersection(values)
                    if section == 'preprocessing':
                        misplaced.discard('seed')  # Independent dataset split seed.
                    if misplaced:
                        raise ValueError(f'{sorted(misplaced)} belong in JSON section {expected_section}, not {section}')

        model_cfg = config_dict.get('model', {})
        training_cfg = config_dict.get('training', {})
        loss_weights = config_dict.get('loss_weights', {})
        loss_params = config_dict.get('loss_params', {})
        inference_cfg = config_dict.get('inference', {})
        gpu_cfg = config_dict.get('gpu_config', {})
        evaluation_cfg = config_dict.get('evaluation', {})
        # augmentation и data читались через getattr с дефолтами, но в params
        # не попадали — zoom_range/intensity_range/pool_size молча игнорировались.
        augmentation_cfg = config_dict.get('augmentation', {})
        data_cfg = config_dict.get('data', {})

        params = {
            **model_cfg,
            **training_cfg,
            **loss_weights,
            **loss_params,
            **evaluation_cfg,
            **{k: v for k, v in augmentation_cfg.items() if k != 'enable'},
            **{k: v for k, v in data_cfg.items()
               if k in ('pool_size', 'repool_freq')},
        }

        params['mode'] = config_dict.get('mode', 'train')

        if gpu_cfg:
            params['gpu_config'] = gpu_cfg

        # JSON uses initial_learning_rate, trainer reads train_learning_rate.
        if 'initial_learning_rate' in training_cfg:
            params['train_learning_rate'] = float(
                training_cfg['initial_learning_rate'])

        if 'steps_per_epoch' in training_cfg:
            params['train_steps_per_epoch'] = int(training_cfg['steps_per_epoch'])

        # Все inference-настройки кроме физического пути к чекпоинту.
        for key, value in inference_cfg.items():
            if key == 'checkpoint_path':
                continue
            params[key] = value

        # --- ГЕОМЕТРИЯ ОБУЧЕНИЯ ---
        geo_path = (Path(config_dict.get('output', {}).get('basedir', 'models')) /
                    config_dict.get('experiment', {}).get('name', 'model') /
                    'geometry.json')
        params['geometry_path'] = str(geo_path)
        checkpoint = (training_cfg.get('resume_from_checkpoint') if params['mode'] == 'train'
                      else inference_cfg.get('checkpoint_path'))
        checkpoint = checkpoint_path or checkpoint
        geo_load_path = Path(checkpoint).parent / 'geometry.json' if checkpoint else geo_path
        if geo_load_path.exists():
            geo = json.loads(geo_load_path.read_text(encoding='utf-8'))
            if int(geo.get('n_rays', params.get('n_rays', 0))) != int(params.get('n_rays', 0)):
                raise ValueError(
                    f"n_rays в {geo_path} ({geo.get('n_rays')}) не совпадает с "
                    f"конфигом ({params.get('n_rays')}) — веса и лучи от разных прогонов")
            for k in ('rays_anisotropy_nucleus', 'rays_anisotropy_cytoplasm',
                      'nucleus_grid', 'cytoplasm_grid',
                      'median_max_dist_nucleus', 'median_max_dist_cytoplasm'):
                if geo.get(k) is not None:
                    params[k] = geo[k]
            print(f"✅ Геометрия обучения подхвачена из {geo_load_path}")
        elif checkpoint:
            raise FileNotFoundError(f'{geo_load_path} missing: cannot interpret checkpoint rays safely')

        for key in ('train_patch_size', 'anisotropy', 'grid',
                    'nucleus_grid', 'cytoplasm_grid',
                    'rays_anisotropy_nucleus', 'rays_anisotropy_cytoplasm'):
            if params.get(key) is not None:
                params[key] = tuple(params[key])

        return cls(**params)

    def to_base_config(self, head='nucleus'):
        """Convert to base StarDist Config2D/Config3D for compatibility."""
        ConfigClass = Config3D if self.ndim == 3 else Config2D
        rays = self.rays_cytoplasm if head == 'cytoplasm' else self.rays_nucleus
        grid = self.cytoplasm_grid if head == 'cytoplasm' else self.nucleus_grid

        return ConfigClass(
            rays=rays,
            grid=grid,
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
            f"nuc_grid={self.nucleus_grid}",
            f"cyto_grid={self.cytoplasm_grid}",
            f"rays_aniso_nuc={tuple(round(v, 1) for v in self.rays_anisotropy_nucleus)}",
            f"rays_aniso_cyto={tuple(round(v, 1) for v in self.rays_anisotropy_cytoplasm)}",
            f"patch_size={self.train_patch_size}",
            f"image_batch_size={self.image_batch_size}",
            f"prompts_per_crop={self.prompts_per_crop}",
            f"gradient_clip_norm={self.gradient_clip_norm}",
            f"refinement_mode={self.refinement_mode}",
            f"refinement_depth={self.refinement_depth}",
            f"refinement_gradient_scale={self.refinement_gradient_scale}",
            f"proposal_mode={self.proposal_mode}",
            f"seed={self.seed}",
            f"consistency={'ON' if self.enable_consistency else 'OFF'}",
        ]
        return f"CytoNucsConfig({', '.join(attrs)})"
