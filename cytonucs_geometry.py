"""Shared network geometry. No TensorFlow dependency."""
import math
import numpy as np


def architecture_plan(config):
    nd = int(config.ndim)
    grids = [tuple(int(x) for x in getattr(config, name))
             for name in ('nucleus_grid', 'cytoplasm_grid')]
    for g in grids:
        if len(g) != nd or any(x < 1 or x & (x - 1) for x in g):
            raise ValueError(f'Head grids must contain {nd} positive powers of two: {g}')
    base = tuple(min(a, b) for a, b in zip(*grids))
    steps, cur = [], [1] * nd
    while tuple(cur) != base:
        p = tuple(2 if cur[a] < base[a] else 1 for a in range(nd))
        steps.append(p)
        cur = [v * q for v, q in zip(cur, p)]
    patch = tuple(int(x) for x in config.train_patch_size)
    if len(patch) != nd or any(s < g or s % g for s, g in zip(patch, base)):
        raise ValueError(f'Patch {patch} must be divisible by base grid {base}')
    size = [s // g for s, g in zip(patch, base)]
    pools = []
    levels = [base]
    schedule = getattr(config, 'unet_pool_schedule', None)
    kernels = getattr(config, 'unet_kernel_schedule', None)
    depth = int(config.unet_n_depth)
    if schedule is not None and len(schedule) != depth:
        raise ValueError('unet_pool_schedule length must equal unet_n_depth')
    if kernels is not None and (len(kernels) != depth + 1 or any(
            len(k) != nd or any(isinstance(v, bool) or v not in (1, 3) for v in k) for k in kernels)):
        raise ValueError('unet_kernel_schedule needs depth+1 kernels with entries 1 or 3')
    for level in range(depth):
        p = tuple(schedule[level]) if schedule is not None else tuple(2 if s % 2 == 0 and s // 2 >= 2 else 1 for s in size)
        if len(p) != nd or any(isinstance(q, bool) or q not in (1, 2) or s % q or (q > 1 and s // q < 2) for s, q in zip(size, p)):
            raise ValueError('pool schedule must divide every level and retain at least two voxels per axis')
        pools.append(p)
        size = [s // q for s, q in zip(size, p)]
        levels.append(tuple(v * q for v, q in zip(levels[-1], p)))
    if any(g not in levels for g in grids):
        raise ValueError(f'Head grids {grids} must match actual encoder levels {levels}')
    factors = [refinement_factors(config, head) for head in ('nucleus', 'cytoplasm')]
    multiple = tuple(math.lcm(levels[-1][a], *(g[a] * f[a] for g, f in zip(grids, factors)))
                     for a in range(nd))
    return dict(base_grid=base, base_steps=steps, pools=pools,
                nucleus_skip=levels.index(grids[0]), cytoplasm_skip=levels.index(grids[1]),
                input_multiple=multiple, bottleneck=tuple(size))


def refinement_pool_schedule(config, head):
    """Pool sizes of the prompted refiner per level for one head.

    'isotropic' (legacy) halves every axis. 'auto' follows the physical voxel
    step at the head grid, like the encoder: an axis is pooled only while its
    step is below twice the finest step, so a coarse Z is not collapsed first.
    """
    nd = int(config.ndim)
    depth = int(getattr(config, 'refinement_depth', 0))
    if getattr(config, 'refinement_pool', 'isotropic') != 'auto':
        return [(2,) * nd for _ in range(depth)]
    grid = tuple(int(g) for g in getattr(config, head + '_grid'))
    step = [float(a) * g for a, g in zip(config.anisotropy, grid)]
    schedule = []
    for _ in range(depth):
        finest = min(step)
        pool = tuple(2 if s < 2 * finest else 1 for s in step)
        schedule.append(pool)
        step = [s * q for s, q in zip(step, pool)]
    return schedule


def refinement_factors(config, head):
    """Total refiner downsampling per axis (input multiple of the head grid)."""
    nd = int(config.ndim)
    return tuple(int(np.prod([p[a] for p in refinement_pool_schedule(config, head)] or [1]))
                 for a in range(nd))


def detection_receptive_field(plan):
    """Conservative receptive field including the nucleus-to-cell context path."""
    result = []
    pools, ns, cs = plan['pools'], plan['nucleus_skip'], plan['cytoplasm_skip']
    for a in range(len(plan['base_grid'])):
        def conv(r, j, n=1):
            return r + 2 * n * j, j
        def pool(r, j, p):
            return r + (p - 1) * j, j * p
        r, j = 1, 1
        for st in plan['base_steps']:
            r, j = conv(r, j, 2)
            r, j = pool(r, j, st[a])
        enc = []
        for i in range(len(pools) + 1):
            r, j = conv(r, j, 2)
            enc.append((r, j))
            if i < len(pools):
                r, j = pool(r, j, pools[i][a])
        rn, jn = r, j
        for i in reversed(range(ns, len(pools))):
            jn //= pools[i][a]
            rn, jn = conv(max(rn, enc[i][0]), jn, 2)
        nr, _ = conv(rn, jn)
        cr, cj = conv(rn, jn)
        for i in range(ns, cs):
            cr, cj = pool(cr, cj, pools[i][a])
        for i in range(cs, ns):
            cj //= pools[i][a]
        for i in reversed(range(cs, len(pools))):
            j //= pools[i][a]
            rr, jj = cr, cj
            for k in range(cs, i):
                rr, jj = pool(rr, jj, pools[k][a])
            if jj != j:
                raise ValueError('Context and decoder strides do not match')
            r, j = conv(max(r, enc[i][0], rr), j, 2)
        rr, _ = conv(r, j)
        result.append(int(max(nr, rr)))
    return tuple(result)


def expected_object_extent(config, head, full_shape):
    """Typical object bbox (voxels) from measured geometry, as used by proposals."""
    nd = int(config.ndim)
    a = np.asarray(getattr(config, 'rays_anisotropy_' + head, config.anisotropy), np.float64)
    radius = float(getattr(config, 'median_max_dist_' + head, 0.0) or 0.0)
    if a.size == nd and radius > 0:
        relative = 1.0 / np.maximum(a, 1e-9)
        return 2.0 * relative * radius / np.linalg.norm(relative)
    return np.asarray(full_shape, np.float64) / 8.0


def proposal_peak_settings(config, head, grid_shape, peaks=None):
    """Peak window and XY border on the head grid; one rule for inference and training.

    Explicit JSON values win. Null derives the window from measured object geometry
    and uses the legacy border, so detector prompts see the same peaks as inference.
    peaks=True forces true 3D local maxima (as proposal_mode='peaks'); the trainer's
    peak sampler always needs them, NMS-based stardist proposals do not.
    """
    nd = int(config.ndim)
    grid = tuple(int(g) for g in getattr(config, head + '_grid'))
    grid_shape = tuple(int(s) for s in grid_shape)
    border = getattr(config, 'proposal_border_' + head)
    if border is None:
        border = int(config.nms_border_exclude_xy)
    window = getattr(config, 'proposal_peak_window_' + head)
    if window is None:
        ext = expected_object_extent(config, head, [s * g for s, g in zip(grid_shape, grid)])
        spacing = float(config.nms_candidate_spacing)
        window = [max(1, int(round(spacing * ext[a] / grid[a]))) for a in range(nd)]
        window = [min(w, max(1, grid_shape[a] // 4)) for a, w in enumerate(window)]
        window = [w + 1 - (w % 2) for w in window]
        if (config.proposal_mode == 'peaks') if peaks is None else peaks:
            window = [max(3, w) for w in window]
    return [int(w) for w in window], int(border)


def pad_to_multiple(array, multiple):
    spatial = array.shape[:len(multiple)]
    pads = [(0, (-int(s)) % int(m)) for s, m in zip(spatial, multiple)]
    pads += [(0, 0)] * (array.ndim - len(multiple))
    return np.pad(array, pads, mode='reflect') if any(p[1] for p in pads) else array


def fixed_window_plan(n, window, block, grid=1):
    """Overlapping fixed-size windows, disjoint centre ownership, grid alignment."""
    padded = math.ceil(n / block) * block
    window = min(int(window), padded)
    if window % block:
        raise ValueError('Fixed window must be divisible by network block')
    stride = max(block, (window // 2 // block) * block)
    starts = list(range(0, max(1, padded - window + 1), stride))
    if starts[-1] != padded - window:
        starts.append(padded - window)
    borders = [0] + [((l + window + r) // 2 // grid) * grid
                     for l, r in zip(starts, starts[1:])] + [n]
    return [(s, window, borders[i], borders[i + 1]) for i, s in enumerate(starts)]
