"""Detached training targets for the candidate actually used by mask inference."""
import numpy as np
from cytonucs_postprocess import local_prompt_mask


def gaussian_prompt_geometry(prompt):
    """Recover centre/sigma from a sampled Gaussian, including subvoxel centres.

    Three log samples determine each quadratic exactly. Prompts are generated
    after image augmentation by the generator/trainer, so they remain Gaussian.
    Zero prompts represent no selected object and return None.
    """
    prompt = np.asarray(prompt, np.float64)
    if not np.isfinite(prompt).all() or np.any(prompt < 0):
        raise ValueError('Quality target requires a finite nonnegative Gaussian prompt')
    if prompt.max() <= 0:
        return None
    peak = np.array(np.unravel_index(prompt.argmax(), prompt.shape))
    center = peak.astype(np.float64)
    sigma = np.ones(prompt.ndim, np.float64)
    for axis, length in enumerate(prompt.shape):
        if length < 3:
            # A singleton axis has no choice of component along this axis.
            continue
        start = int(np.clip(peak[axis] - 1, 0, length - 3))
        indices = np.repeat(peak[None], 3, axis=0)
        indices[:, axis] = np.arange(start, start + 3)
        samples = prompt[tuple(indices.T)]
        if np.any(samples <= 0):
            raise ValueError('Cannot recover Gaussian geometry from underflowed prompt')
        v = np.log(samples)
        a = .5 * (v[2] - 2*v[1] + v[0])
        b = .5 * (v[2] - v[0])
        if a >= 0:
            raise ValueError('Quality target prompt is not a resolved Gaussian')
        center[axis] = start + 1 - b / (2*a)
        sigma[axis] = np.sqrt(-.5 / a)
    # Float32 exp can put an exact half-integer slightly across a rounding tie.
    half = np.round(center * 2) / 2
    center = np.where(np.abs(center - half) < 1e-5, half, center)
    return center, sigma


def component_quality_targets(probabilities, targets, prompts, grid, threshold):
    """IoU after full-resolution interpolation, threshold and prompt component.

    This is a CPU, nondifferentiable label calculation for the quality head.
    Targets stay the selected training instances; another object's mask is not
    rewarded. Assembly/voxel competition is deliberately not part of this IoU.
    """
    probabilities = np.asarray(probabilities)
    targets, prompts = np.asarray(targets), np.asarray(prompts)
    if targets.shape != prompts.shape or len(probabilities) != len(targets):
        raise ValueError('Quality targets/prompts must use the same K-major batch order')
    values = np.zeros(len(targets), np.float32)
    for i, (prob, target, prompt) in enumerate(zip(probabilities, targets, prompts)):
        target = target > .5
        if not target.any():
            continue
        geometry = gaussian_prompt_geometry(prompt)
        if geometry is None:
            continue
        # The same mask as prompt_component(upsample_grid(prob) > threshold), evaluated only in the window that can
        # hold it (local_prompt_mask, as at inference): the whole crop upsampled for every prompt was most of the cost.
        lo, _, _, mask, _ = local_prompt_mask(prob[..., 0], target.shape, grid, *geometry, threshold)
        inside = np.count_nonzero(mask & target[tuple(slice(int(a), int(a) + n) for a, n in zip(lo, mask.shape))])
        union = np.count_nonzero(mask) + np.count_nonzero(target) - inside
        values[i] = inside / max(union, 1)
    return values
