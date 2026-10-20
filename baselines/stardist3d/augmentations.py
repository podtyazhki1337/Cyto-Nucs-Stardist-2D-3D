"""
augmentations.py — Benchmark-standard augmentation pipeline
for 3D neuron instance segmentation in DODT / oblique microscopy.

Design principles
─────────────────
• Pure NumPy + SciPy — no TF / PyTorch / framework dependency.
  Works identically with StarDist, U-Net, nnU-Net, Cellpose.
• Every transform is physically motivated by the imaging physics
  of cleared-tissue transmitted-light (DODT) and oblique microscopy.
• Three presets:
    NO_AUG    — strict baseline, identity transform (ablation control)
    LIGHT_AUG — minimal physics-motivated set (default benchmark policy)
    FULL      — heavy augmentation for max regularisation
  NO_AUG vs LIGHT_AUG comparison isolates the effect of augmentation
  in the paper; LIGHT_AUG is the common policy across all models.
• Fully configurable via a single dict that gets logged to Comet as-is.
• Deterministic when the global np.random seed is set.

Usage
─────
    from augmentations import augmenter, NO_AUG_CONFIG, LIGHT_AUG_CONFIG, FULL_CONFIG

    # StarDist / U-Net / any model that takes an (x, y) → (x, y) callable:
    model.train(..., augmenter=augmenter)

    # Switch preset before training:
    from augmentations import set_config
    set_config(NO_AUG_CONFIG)      # strict baseline
    set_config(LIGHT_AUG_CONFIG)   # default benchmark
    set_config(FULL_CONFIG)        # heavy

    # Or tweak individual knobs:
    from augmentations import ACTIVE_CONFIG
    ACTIVE_CONFIG["elastic_enabled"] = True
"""

from __future__ import annotations

import copy
import json
from typing import Tuple

import numpy as np
from scipy.ndimage import (
    affine_transform,
    gaussian_filter,
    map_coordinates,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration presets
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared template (everything OFF) ─────────────────────────────────────────
_TEMPLATE: dict = {
    "flip_xy":              False,
    "flip_z":               False,
    "rot90_xy":             False,

    "intensity_scale_range": (1.0, 1.0),
    "intensity_shift_range": (0.0, 0.0),
    "intensity_probability": 0.0,       # gate for scale+shift together

    "gamma_range":          (1.0, 1.0),
    "gamma_probability":    0.0,        # gate for gamma

    "noise_enabled":        False,
    "noise_std_range":      (0.0, 0.0),
    "noise_probability":    0.0,

    "blur_enabled":         False,
    "blur_probability":     0.0,
    "blur_sigma_z_range":   (0.0, 0.0),
    "blur_sigma_xy_range":  (0.0, 0.0),

    "depth_atten_enabled":  False,
    "depth_atten_prob":     0.0,
    "depth_atten_range":    (1.0, 1.0),

    "zoom_enabled":         False,
    "zoom_range":           (1.0, 1.0),

    "elastic_enabled":      False,
    "elastic_alpha":        80,
    "elastic_sigma":        10,
    "elastic_probability":  0.0,

    "local_contrast_enabled":   False,
    "local_contrast_prob":      0.0,
    "local_contrast_sigma":     30,
    "local_contrast_range":     (1.0, 1.0),

    "cutout_enabled":       False,
    "cutout_n_max":         3,
    "cutout_size_range":    (4, 16),
    "cutout_probability":   0.0,
}

# ── 1. NO_AUG  —  strict baseline, identity transform ───────────────────────
NO_AUG_CONFIG: dict = copy.deepcopy(_TEMPLATE)


# ── 2. LIGHT_AUG  —  minimal physics-motivated set for DODT / oblique ───────
#    Identical policy for StarDist, U-Net, nnU-Net, Cellpose.
#    No blur / zoom / elastic — only transformations that are trivially
#    justified and introduce no interpolation artefacts on labels.
#
#    KEY: intensity transforms are STOCHASTIC (p=0.5), NOT applied to every
#    patch. This is critical for models with weak supervision signal (e.g.
#    binary dice loss) that struggle with constant intensity perturbation.
LIGHT_AUG_CONFIG: dict = copy.deepcopy(_TEMPLATE)
LIGHT_AUG_CONFIG.update({
    # Geometry: flips + 90° in XY only.
    # Neurons have no preferred in-plane orientation.
    # Z-flip OFF: DODT illumination is directional.
    "flip_xy":              True,
    "rot90_xy":             True,

    # Intensity scale+shift: lamp-power / clearing-quality variation.
    # Applied stochastically — 50% of patches get identity intensity.
    "intensity_scale_range": (0.85, 1.15),
    "intensity_shift_range": (-0.05, 0.05),
    "intensity_probability": 0.5,

    # Gamma: slight non-linearity in detector response.
    "gamma_range":          (0.9, 1.1),
    "gamma_probability":    0.5,

    # Gaussian noise: detector shot+read noise, applied stochastically.
    "noise_enabled":        True,
    "noise_std_range":      (0.005, 0.02),
    "noise_probability":    0.3,
})


# ── 3. FULL  —  heavy augmentation for max regularisation ────────────────────
FULL_CONFIG: dict = copy.deepcopy(LIGHT_AUG_CONFIG)
FULL_CONFIG.update({
    "flip_z":                   True,
    "intensity_scale_range":    (0.6, 2.0),
    "intensity_shift_range":    (-0.2, 0.2),
    "intensity_probability":    0.7,
    "gamma_range":              (0.7, 1.3),
    "gamma_probability":        0.7,
    "noise_std_range":          (0.0, 0.03),
    "noise_probability":        0.5,
    "blur_enabled":             True,
    "blur_probability":         0.5,
    "blur_sigma_z_range":       (0.0, 1.5),
    "blur_sigma_xy_range":      (0.0, 0.8),
    "depth_atten_enabled":      True,
    "depth_atten_prob":         0.5,
    "depth_atten_range":        (0.5, 1.0),
    "zoom_enabled":             True,
    "zoom_range":               (0.85, 1.15),
    "elastic_enabled":          True,
    "elastic_probability":      0.4,
    "local_contrast_enabled":   True,
    "local_contrast_prob":      0.4,
    "cutout_enabled":           True,
    "cutout_probability":       0.3,
})


# The module-level config that augmenter() reads at runtime.
# Default = LIGHT_AUG.  Swap with set_config() before training.
ACTIVE_CONFIG: dict = copy.deepcopy(LIGHT_AUG_CONFIG)


def set_config(cfg: dict) -> None:
    """Replace the active augmentation config (in-place update)."""
    global ACTIVE_CONFIG
    ACTIVE_CONFIG.clear()
    ACTIVE_CONFIG.update(copy.deepcopy(cfg))


def get_config_json() -> str:
    """Serialise current config for Comet logging."""
    return json.dumps(ACTIVE_CONFIG, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
#  Individual transforms  (all operate on numpy arrays)
# ══════════════════════════════════════════════════════════════════════════════

def _flip_and_rotate(img: np.ndarray, mask: np.ndarray,
                     cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Random flips (XY, optionally Z) + 90° XY rotations."""
    if cfg["flip_xy"]:
        for ax in (1, 2):
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=ax)
                mask = np.flip(mask, axis=ax)

    if cfg["rot90_xy"]:
        k = np.random.randint(0, 4)
        if k:
            img  = np.rot90(img,  k=k, axes=(1, 2))
            mask = np.rot90(mask, k=k, axes=(1, 2))

    if cfg["flip_z"] and np.random.rand() > 0.5:
        img  = np.flip(img,  axis=0)
        mask = np.flip(mask, axis=0)

    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def _intensity_scale_shift(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Random linear intensity transform:  img * scale + shift.
    Gated by intensity_probability — identity on skip."""
    p = cfg.get("intensity_probability", 1.0)
    if p < 1.0 and np.random.rand() > p:
        return img
    scale = np.random.uniform(*cfg["intensity_scale_range"])
    shift = np.random.uniform(*cfg["intensity_shift_range"])
    return img * scale + shift


def _gamma(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Random gamma correction (power-law).
    Gated by gamma_probability — identity on skip."""
    p = cfg.get("gamma_probability", 1.0)
    if p < 1.0 and np.random.rand() > p:
        return img
    g = np.random.uniform(*cfg["gamma_range"])
    return np.clip(img, 0, None) ** g


def _gaussian_noise(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Additive Gaussian noise with per-sample random σ, applied stochastically."""
    if not cfg["noise_enabled"]:
        return img
    if np.random.rand() > cfg.get("noise_probability", 1.0):
        return img
    sigma = np.random.uniform(*cfg["noise_std_range"])
    if sigma > 0:
        img = img + np.random.normal(0, sigma, size=img.shape).astype(img.dtype)
    return img


def _anisotropic_blur(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Gaussian blur with independent Z and XY sigmas."""
    if not cfg["blur_enabled"]:
        return img
    if np.random.rand() > cfg["blur_probability"]:
        return img
    sz = np.random.uniform(*cfg["blur_sigma_z_range"])
    sxy = np.random.uniform(*cfg["blur_sigma_xy_range"])
    if sz > 0 or sxy > 0:
        img = gaussian_filter(img, sigma=(sz, sxy, sxy))
    return img


def _depth_attenuation(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Simulate depth-dependent intensity loss along Z."""
    if not cfg["depth_atten_enabled"]:
        return img
    if np.random.rand() > cfg["depth_atten_prob"]:
        return img
    bottom_frac = np.random.uniform(*cfg["depth_atten_range"])
    nz = img.shape[0]
    ramp = np.linspace(1.0, bottom_frac, nz).astype(img.dtype)
    if np.random.rand() > 0.5:
        ramp = ramp[::-1]
    return img * ramp[:, None, None]


def _zoom_xy(img: np.ndarray, mask: np.ndarray,
             cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic zoom in XY only."""
    if not cfg["zoom_enabled"]:
        return img, mask
    z = np.random.uniform(*cfg["zoom_range"])
    if abs(z - 1.0) < 1e-4:
        return img, mask

    matrix = np.diag([1.0, 1.0 / z, 1.0 / z])
    center = (np.array(img.shape) - 1) / 2.0
    offset = center - matrix.dot(center)

    img_z  = affine_transform(img,  matrix, offset=offset, order=1,
                               mode="constant", cval=0)
    mask_z = affine_transform(mask, matrix, offset=offset, order=0,
                               mode="constant", cval=0)
    return img_z, mask_z


def _elastic_deformation(img: np.ndarray, mask: np.ndarray,
                         cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth random elastic deformation (3D)."""
    if not cfg["elastic_enabled"]:
        return img, mask
    if np.random.rand() > cfg["elastic_probability"]:
        return img, mask

    alpha = cfg["elastic_alpha"]
    sigma = cfg["elastic_sigma"]
    shape = img.shape

    dz = gaussian_filter(np.random.randn(*shape) * alpha, sigma)
    dy = gaussian_filter(np.random.randn(*shape) * alpha, sigma)
    dx = gaussian_filter(np.random.randn(*shape) * alpha, sigma)

    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij")

    coords = np.array([z + dz, y + dy, x + dx])

    img_e  = map_coordinates(img,  coords, order=1, mode="reflect")
    mask_e = map_coordinates(mask, coords, order=0, mode="reflect")
    return img_e.astype(img.dtype), mask_e.astype(mask.dtype)


def _local_contrast(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Smooth multiplicative intensity field."""
    if not cfg["local_contrast_enabled"]:
        return img
    if np.random.rand() > cfg["local_contrast_prob"]:
        return img

    lo, hi = cfg["local_contrast_range"]
    sigma  = cfg["local_contrast_sigma"]

    field = np.random.uniform(lo, hi, size=img.shape).astype(img.dtype)
    field = gaussian_filter(field, sigma=sigma)
    return img * field


def _cutout(img: np.ndarray, mask: np.ndarray,
            cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Random 3D rectangular cutout."""
    if not cfg["cutout_enabled"]:
        return img, mask
    if np.random.rand() > cfg["cutout_probability"]:
        return img, mask

    n = np.random.randint(1, cfg["cutout_n_max"] + 1)
    lo, hi = cfg["cutout_size_range"]
    shape = img.shape

    for _ in range(n):
        sz = np.random.randint(lo, hi + 1)
        sy = np.random.randint(lo, hi + 1)
        sx = np.random.randint(lo, hi + 1)
        z0 = np.random.randint(0, max(1, shape[0] - sz))
        y0 = np.random.randint(0, max(1, shape[1] - sy))
        x0 = np.random.randint(0, max(1, shape[2] - sx))
        img[z0:z0+sz, y0:y0+sy, x0:x0+sx]  = 0
        mask[z0:z0+sz, y0:y0+sy, x0:x0+sx] = 0

    return img, mask


# ══════════════════════════════════════════════════════════════════════════════
#  Main augmenter callable  (signature: (x, y) → (x, y))
# ══════════════════════════════════════════════════════════════════════════════

def augmenter(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benchmark-standard augmentation for DODT / oblique 3D neuron segmentation.

    The execution order is designed to be physically consistent:
      1. Geometric transforms (flips, rotation, zoom, elastic)
      2. Intensity transforms (scale/shift, gamma, depth atten, local contrast)
      3. Degradation (blur, noise, cutout)

    Reads from the module-level ACTIVE_CONFIG dict.
    Compatible with StarDist, U-Net, nnU-Net, Cellpose train APIs.
    """
    cfg = ACTIVE_CONFIG
    x = x.copy()
    y = y.copy()

    # ── Stage 1: Geometry ────────────────────────────────────────────────
    x, y = _flip_and_rotate(x, y, cfg)
    x, y = _zoom_xy(x, y, cfg)
    x, y = _elastic_deformation(x, y, cfg)

    # ── Stage 2: Intensity ───────────────────────────────────────────────
    x = _intensity_scale_shift(x, cfg)
    x = _gamma(x, cfg)
    x = _depth_attenuation(x, cfg)
    x = _local_contrast(x, cfg)

    # ── Stage 3: Degradation ─────────────────────────────────────────────
    x = _anisotropic_blur(x, cfg)
    x = _gaussian_noise(x, cfg)
    x, y = _cutout(x, y, cfg)

    # Final clip — keeps values non-negative (post-normalisation convention)
    x = np.clip(x, 0, None)

    return x, y