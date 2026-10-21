"""
CytoNucs StarDist Inference

Post-processing to convert probability and distance maps into instance segmentations.
"""
import numpy as np
from cytonucs_runtime import write_instance_mask
import tensorflow as tf
import tifffile
from cytonucs_geometry import architecture_plan, detection_receptive_field, pad_to_multiple, fixed_window_plan
from cytonucs_fine_mask import fine_window_arrays
from cytonucs_postprocess import (upsample_grid, prompt_component, mask_candidate, local_prompt_mask,
                                 assemble_masks, assign_final_masks, drop_small_instances)


def _refinement_runners(model):
    """Compile once per model; weights remain live across training epochs."""
    if not hasattr(model, '_compiled_refinement_runners'):
        features = model.refinement_features_model
        refinement = getattr(model, 'refinement_with_quality_model', model.refinement_model)
        feature_fn = tf.function(lambda x: features(x, training=False), reduce_retracing=True)
        refine_fn = tf.function(lambda x: refinement(x, training=False), reduce_retracing=True)
        model._compiled_refinement_runners = (feature_fn, refine_fn)
    return model._compiled_refinement_runners


def _fine_refinement_runners(model):
    """Runners of the full-resolution variant: backbone features, full-resolution features,
    refinement with projected features, cell refinement, and the fine head per head."""
    if not hasattr(model, '_compiled_fine_runners'):
        def compiled(net):
            return tf.function(lambda x: net(x, training=False), reduce_retracing=True)
        model._compiled_fine_runners = (
            compiled(model.refinement_features_model), compiled(model.fine_features_model),
            compiled(model.refinement_fine_model), compiled(model.cytoplasm_fine_model),
            {h: compiled(net) for h, net in model.fine_head_models.items()})
    return model._compiled_fine_runners


def _fine_prompt_mask(head_fn, coarse_logit, hidden, full, grid, lo, shape, run_shape, net_shape,
                      center, sigma, threshold):
    """Full-resolution prompt mask around a coarse candidate window (same rule as the coarse
    mask: threshold, then the prompt component). The window grows while the component
    reaches one of its faces inside the ROI. `full`: full-resolution [features, image, prompt]
    over the network input. Returns (start, probability, foreground, mask,
    any_foreground) with arrays over the window starting at `start` (ROI coordinates)."""
    from cytonucs_fine_mask import inference_window, fine_window_arrays, component_touches_open_face
    from cytonucs_postprocess import _component_near
    margin = [2 * g + 3 for g in grid]
    centre = np.clip(np.round(center).astype(int), 0, np.array(run_shape) - 1)
    for _ in range(4):
        start, stop = inference_window(lo, shape, run_shape, net_shape, grid, margin)
        logits = head_fn(fine_window_arrays(coarse_logit, hidden, full, start, stop, grid))
        end = np.minimum(stop, np.asarray(run_shape))
        probability = tf.nn.softmax(logits, axis=-1)[0, ..., 1].numpy()
        probability = probability[tuple(slice(0, int(e - b)) for b, e in zip(start, end))]
        foreground = probability > threshold
        mask = _component_near(foreground, start, centre, sigma) if foreground.any() else foreground
        if not component_touches_open_face(mask, start, end, run_shape):
            break
        margin = [2 * m for m in margin]
    return start, probability, foreground, mask, bool(foreground.any())


FINE_BATCH_VOXELS = 4_000_000        # full-resolution voxels per batched fine-head call (memory bound)
REFINE_BATCH_NODES = 1 << 20         # head-grid voxels per batched refinement call (memory bound)
_PENDING = object()                   # placeholder of a cell candidate prompted again over a larger ROI


def _fine_mask_windows(probability_at, lo, shape, run_shape, net_shape, grid, center, sigma, threshold):
    """The window rule of _fine_prompt_mask (grown while the component reaches an inner window face) with
    the full-resolution probability of a window [start, stop) from probability_at(start, stop), cropped
    to the ROI."""
    from cytonucs_fine_mask import inference_window, component_touches_open_face
    from cytonucs_postprocess import _component_near
    margin = [2 * g + 3 for g in grid]
    centre = np.clip(np.round(center).astype(int), 0, np.array(run_shape) - 1)
    for _ in range(4):
        start, stop = inference_window(lo, shape, run_shape, net_shape, grid, margin)
        end = np.minimum(stop, np.asarray(run_shape))
        probability = probability_at(start, stop)
        foreground = probability > threshold
        mask = _component_near(foreground, start, centre, sigma) if foreground.any() else foreground
        if not component_touches_open_face(mask, start, end, run_shape):
            break
        margin = [2 * m for m in margin]
    return start, probability, foreground, mask, bool(foreground.any())


def _place_window(start, stop, size, net_shape, grid):
    """Grid-aligned start of a window of `size` (grid multiples) inside the network input that contains
    [start, stop), centred on it where the input allows."""
    begin = []
    for s, e, w, n, g in zip(start, stop, size, net_shape, grid):
        b = int(s) - ((int(w) - (int(e) - int(s))) // (2 * g)) * g
        b = max(0, min(b, int(n) - int(w)))
        assert b <= s and e <= b + w and b % g == 0, (start, stop, size, net_shape, grid)
        begin.append(b)
    return np.array(begin)


def _window_size(extent, net_shape, grid):
    """Window size of a fine-head call: per axis the smallest of 16 g * (1, 1.5, 2, 3, 4, 6, ...) covering the
    extent, at most the network input. Few distinct static shapes keep the compiled calls (and cuDNN's
    choices) reusable; when that more than doubles the window's volume (large 3D windows: memory), each axis
    is only rounded up to a multiple of 8 g."""
    size = []
    for e, n, g in zip(extent, net_shape, grid):
        w, step = 16 * g, 0
        while w < e:
            w = 16 * g * (2 ** ((step + 1) // 2)) * (1.5 if step % 2 == 0 else 1)
            w = int(-(-w // g) * g)
            step += 1
        size.append(min(int(n), int(w)))
    if np.prod(size) > 2 * np.prod([max(int(e), 1) for e in extent]):
        size = [min(int(n), int(-(-int(e) // (8 * g)) * 8 * g)) for e, n, g in zip(extent, net_shape, grid)]
    return tuple(size)


REFINE_WINDOW_SIZES = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)   # head-grid nodes per axis


def _refiner_reach(config):
    """Receptive-field radius of the prompted refiner in head-grid nodes per axis (plus a safety node) and its
    total pooling factor, over both heads (they share the grid when windows are used)."""
    from cytonucs_geometry import refinement_pool_schedule
    nd = int(config.ndim)
    reach, factor = np.zeros(nd, int), np.ones(nd, int)
    for head in ('nucleus', 'cytoplasm'):
        schedule = [np.asarray(p, int) for p in refinement_pool_schedule(config, head)]
        r, stride = np.zeros(nd, int), np.ones(nd, int)
        for pool in schedule:
            r += stride + (pool - 1) * stride                    # 3x3 convolution, then pooling
            stride = stride * pool
        r += stride                                              # bottleneck (or the two convolutions without depth)
        for pool in reversed(schedule):
            stride = stride // pool
            r += stride                                          # convolution after upsampling
        reach = np.maximum(reach, r + 2)
        factor = np.lcm(factor, np.prod(schedule, axis=0) if schedule else np.ones(nd, int))
    return reach, factor


def _refine_window_runner(model, kind, fn, size):
    """Compiled refinement of per-prompt windows of static `size` (head-grid nodes) cut from the ROI features on
    the device: (features [1, *grid_roi, c] each, prompt windows [m, *size*grid, 1], window starts [m, nd] in
    head-grid nodes) -> the outputs of the nucleus (refine) or cell network for the windows."""
    cache = model.__dict__.setdefault('_compiled_refine_windows', {})
    key = (kind, id(fn), tuple(size))
    if key not in cache:
        nd = len(size)
        offsets = tf.stack(tf.meshgrid(*[tf.range(w, dtype=tf.int32) for w in size], indexing='ij'), -1)
        expand = (slice(None),) + (None,) * nd

        def run(features, prompt_windows, begins):
            index = begins[expand] + offsets[None]
            crops = [tf.gather_nd(f[0], index) for f in features]
            return fn(crops + ([prompt_windows] if kind == 'cytoplasm' else [prompt_windows, prompt_windows]))
        cache[key] = tf.function(run, reduce_retracing=True)
    return cache[key]


_EMBED_RUNNERS = {}


def _embed_windows(windows, starts, grid_roi):
    """Windows [m, *size, c] placed at starts [m, nd] into zeros [m, *grid_roi, c] on the device (one scatter)."""
    size, channels = tuple(int(v) for v in windows.shape[1:-1]), int(windows.shape[-1])
    key = (size, tuple(int(v) for v in grid_roi), channels)
    if key not in _EMBED_RUNNERS:
        nd = len(size)
        offsets = tf.stack(tf.meshgrid(*[tf.range(w, dtype=tf.int32) for w in size], indexing='ij'), -1)
        expand = (slice(None),) + (None,) * nd
        full = tf.constant(key[1], tf.int32)

        def embed(w, start):
            m = tf.shape(w)[0]
            index = start[expand] + offsets[None]
            batch = tf.broadcast_to(tf.reshape(tf.range(m), [-1] + [1] * (nd + 1)), tf.concat([tf.shape(index)[:-1], [1]], 0))
            return tf.scatter_nd(tf.concat([batch, index], -1), w, tf.concat([[m], full, [channels]], 0))
        _EMBED_RUNNERS[key] = tf.function(embed, reduce_retracing=True)
    return _EMBED_RUNNERS[key](windows, starts)


def _fine_window_runner(model, head, head_fn, grid, size):
    """Compiled fine head on per-prompt windows of static `size`, cut on the device: (coarse logits
    [n, *grid_shape, 3], hidden [n, *grid_shape, c], full-resolution features [1, *net, f], network input
    [1, *net, i], prompt windows [m, *size, 1], items [m], window starts [m, nd]) -> foreground probability
    [m, *size]. The windows are those of window_inputs (training) and fine_window_arrays (one-prompt
    inference)."""
    from cytonucs_fine_mask import pad_last_node
    cache = model.__dict__.setdefault('_compiled_fine_windows', {})
    key = (head, id(head_fn), tuple(grid), tuple(size))
    if key not in cache:
        nd = len(grid)
        grid_size = [w // g + (1 if g > 1 else 0) for w, g in zip(size, grid)]

        def offsets(extent):
            return tf.stack(tf.meshgrid(*[tf.range(w, dtype=tf.int32) for w in extent], indexing='ij'), -1)
        full_offsets, grid_offsets = offsets(size), offsets(grid_size)
        expand = (slice(None),) + (None,) * nd

        def cut(t, items, begins, window_offsets):
            index = begins[expand] + window_offsets[None]                   # [m, *extent, nd]
            if items is None:                                              # one tensor shared by all prompts
                return tf.gather_nd(t[0], index)
            batch = tf.broadcast_to(items[expand + (None,)], tf.concat([tf.shape(index)[:-1], [1]], 0))
            return tf.gather_nd(t, tf.concat([batch, index], -1))

        def run(logit, hidden, features, image, prompt_windows, items, begins):
            grid_begins = begins // tf.constant(grid, tf.int32)
            inputs = [cut(pad_last_node(logit, grid), items, grid_begins, grid_offsets),
                      cut(pad_last_node(hidden, grid), items, grid_begins, grid_offsets),
                      cut(features, None, begins, full_offsets), cut(image, None, begins, full_offsets),
                      prompt_windows]
            return tf.nn.softmax(head_fn(inputs), axis=-1)[..., 1]
        cache[key] = tf.function(run, reduce_retracing=True)
    return cache[key]


def _cytoplasm_refinement_runner(model):
    if not hasattr(model, '_compiled_cytoplasm_refinement'):
        cell_model = model.cytoplasm_refinement_model
        model._compiled_cytoplasm_refinement = tf.function(
            lambda x: cell_model(x, training=False), reduce_retracing=True)
    return model._compiled_cytoplasm_refinement


def assemble_head(config, head, candidates, shape, observer=None):
    """Assemble one head's candidates with its configured mode (shared with calibration)."""
    if head == 'nucleus':
        return assemble_masks(candidates, shape, getattr(config, 'mask_nms_nucleus', 0.8),
                              containment_threshold=(config.mask_containment_threshold
                                  if config.mask_assembly_nucleus == 'object_quality_containment' else 0.0),
                              overlap_threshold=config.mask_overlap_nucleus, observer=observer,
                              assembly_mode=config.mask_assembly_nucleus,
                              min_remaining_fraction=config.mask_min_remaining_fraction)
    return assemble_masks(candidates, shape, getattr(config, 'mask_nms_cytoplasm', 0.5),
                          config.mask_containment_threshold, config.mask_overlap_cytoplasm,
                          observer=observer, assembly_mode=config.mask_assembly_cytoplasm,
                          min_remaining_fraction=config.mask_min_remaining_fraction)


def rethreshold_prediction(config, sink, head, threshold, nuclei, cells):
    """Final labels and pairs when only `head` uses another mask threshold (higher or lower).

    Re-assembles that head from the cached candidates of the same forward passes;
    the other head keeps its labels. No network evaluation.
    """
    from cytonucs_postprocess import rethreshold_candidate
    # An empty sink means prompted refinement never ran on this volume (no proposals).
    candidates = [c for c in sink.get(head, []) if c is not None]
    shrunk = {id(c): rethreshold_candidate(c, threshold) for c in candidates}
    labels, _ = assemble_head(config, head, [shrunk[id(c)] for c in candidates], nuclei.shape)
    if head == 'nucleus':
        nuclei = labels
        evidence = [(shrunk.get(id(n)), c) for n, c in sink.get('evidence', [])]
    else:
        cells = labels
        evidence = [(n, shrunk.get(id(c)) if c is not None else None) for n, c in sink.get('evidence', [])]
    evidence = [(n, c) for n, c in evidence if n is not None]
    pairs = assign_final_masks(nuclei, cells, getattr(config, 'pair_min_overlap', 0.1),
                               evidence, getattr(config, 'pair_min_mask_iou', 0.25))
    return nuclei, cells, pairs


def _pad_input(array, pad_width, mode):
    """Pad detection input; reflect needs more voxels than padding, edge is its fallback."""
    if mode == 'constant':
        return np.pad(array, pad_width, mode='constant')
    try:
        return np.pad(array, pad_width, mode='reflect')
    except ValueError:
        return np.pad(array, pad_width, mode='edge')


def _group_prompt_jobs(jobs, origin_fn):
    """Group exact windows, retaining original indices for stable mask ordering."""
    from collections import OrderedDict
    groups = OrderedDict()
    for index, job in enumerate(jobs):
        groups.setdefault(origin_fn(job[1]), []).append((index, job))
    return groups.values()


def predict_instances_prompted(model, config, img_full, nuc_coords, nuc_probs,
                               cyto_coords=None, cyto_probs=None,
                               mask_thresh=0.5, iou_dedup=0.5, verbose=True,
                               nuc_dists=None, cyto_dists=None,
                               nuc_labels=None, cyto_labels=None, nms_thresh=None, stage_audit=None,
                               candidate_sink=None, prompt_sigmas=None, quality_gates=None,
                               mask_thresholds=None, residual=None):
    """Build instances from prompted masks; retain baselines for controlled comparison.

    quality_gates / mask_thresholds: per-head calibrated values for the 'auto' settings
    (quality_gates may also carry '<head>_min_area', applied after assembly).
    residual: {head: dict(coords, probs, dists)} peaks suppressed by the proposal window;
    prompted only where the first assembly left the voxel free, assembled after all
    first-pass candidates (object assembly modes only).
    """
    from collections import OrderedDict
    from scipy.ndimage import find_objects
    policy = getattr(config, 'refinement_mode', 'mask')
    if policy == 'legacy_union':
        return _predict_instances_prompted_legacy(
            model, config, img_full, nuc_coords, nuc_probs, cyto_coords, cyto_probs,
            mask_thresh, iou_dedup, verbose, nuc_dists, cyto_dists,
            nuc_labels, cyto_labels, nms_thresh)
    nd = int(config.ndim)
    shape = img_full.shape[:nd]
    if policy == 'polyhedron':
        n = np.zeros(shape, np.int32) if nuc_labels is None else np.asarray(nuc_labels, np.int32)
        c = np.zeros(shape, np.int32) if cyto_labels is None else np.asarray(cyto_labels, np.int32)
        return n, c, assign_final_masks(n, c, getattr(config, 'pair_min_overlap', 0.1))
    if policy != 'mask':
        raise ValueError(f'Unknown refinement_mode: {policy}')
    patch = tuple(int(v) for v in (config.refinement_patch_size or config.train_patch_size))
    base_shape = tuple(s if p == 0 else min(p, s) for p, s in zip(patch, shape))
    multiple = architecture_plan(config)['input_multiple']
    feature_cache = OrderedDict()
    cache_limit = max(1, int(getattr(config, 'refine_feature_cache_windows', 1)))
    runs = feature_runs = 0
    nuc_candidates, cell_candidates, evidence = [], [], []
    diagnostics = {h: dict(attempted=0, empty_threshold=0, missing_component=0,
                          nonfinite=0, quality_nonfinite=0, quality_rejected=0, accepted=0, quality=[],
                          roi_expansions=0, roi_boundary_remaining=0,
                          residual_peaks=0, residual_prompts=0, area_gate_removed=0,
                          windowed=0, window_fallbacks=0, prompts_capped=0)
                   for h in ('nucleus', 'cytoplasm')}
    if config.mask_quality_min == 'auto':
        quality_min = {h: float((quality_gates or {}).get(h, 0.0)) for h in ('nucleus', 'cytoplasm')}
    else:
        quality_min = {h: float(config.mask_quality_min) for h in ('nucleus', 'cytoplasm')}

    def anchors(labels, coords, probs, dists):
        if labels is None:
            return []
        by_label = {}
        if coords is not None:
            for i, ctr in enumerate(coords):
                ctr = np.clip(np.asarray(ctr, int), 0, np.array(shape)-1)
                lab = int(labels[tuple(ctr)])
                if not lab:
                    continue
                score = float(probs[i]) if probs is not None else 1.0
                if lab not in by_label or score > by_label[lab][2]:
                    by_label[lab] = (ctr, None if dists is None else dists[i], score)
        for lab, sl in enumerate(find_objects(np.asarray(labels, np.int32)), 1):
            if sl is not None and lab not in by_label:
                pts = np.argwhere(labels[sl] == lab)
                ctr = pts[np.argmin(np.sum((pts-pts.mean(0))**2, axis=1))]
                ctr += np.array([s.start for s in sl])
                by_label[lab] = (ctr, None, 0.0)
        return [by_label[k] for k in sorted(by_label)]

    fine_enabled = bool(getattr(config, 'refinement_full_resolution', False))
    fine_cache = {}
    if fine_enabled:
        feature_fn, fine_feature_fn, refine_fn, cell_fn, fine_head_fns = _fine_refinement_runners(model)
    else:
        feature_fn, refine_fn = _refinement_runners(model)
        cell_fn = _cytoplasm_refinement_runner(model)

    def roi_for(center, distances, head):
        run_shape = base_shape
        if config.refinement_adaptive_roi and distances is not None:
            rays = getattr(config, f'rays_{head}', config.rays)
            vertices = np.asarray(rays.vertices) * np.maximum(distances, 1e-3)[:, None]
            # Centre the ROI on the prompt and include both extreme sides with margin.
            extent = np.ceil(3 * np.max(np.abs(vertices), axis=0)).astype(int)
            run_shape = tuple(min(s, max(b, int(e))) for s, b, e in zip(shape, base_shape, extent))
        origin = tuple(max(0, min(int(c)-s//2, n-s)) for c, s, n in zip(center, run_shape, shape))
        if config.refinement_roi_snap:
            # Snap to the network input multiple: prompts of one volume then share a few
            # windows, so backbone features are reused; the prompt moves off the ROI centre
            # by at most half a step.
            origin = tuple(max(0, min(int(round(o / m)) * m, n - s))
                           for o, m, s, n in zip(origin, multiple, run_shape, shape))
        return origin, run_shape

    def prepare(center, distances, head, roi=None, fixed_sigma=None):
        """ROI, prompt centre in ROI coordinates and prompt sigma of one prompt."""
        origin, run_shape = roi_for(center, distances, head) if roi is None else roi
        center = np.asarray(center) - origin
        rays = getattr(config, f'rays_{head}', config.rays)
        if distances is not None:
            vertices = np.asarray(rays.vertices) * np.maximum(distances, 1e-3)[:, None]
            sigma = np.maximum((vertices.max(0)-vertices.min(0)) / 4, 1)
        else:
            a = np.asarray(getattr(config, f'rays_anisotropy_{head}', config.anisotropy))
            scale = getattr(config, f'median_max_dist_{head}', None)
            if scale is None:
                scale = 15.0 if head == 'nucleus' else 60.0
            sigma = np.maximum(float(scale) / a / 2, 1)
        if fixed_sigma is not None:
            sigma = fixed_sigma
        elif prompt_sigmas is not None:
            sigma = np.asarray(prompt_sigmas[(head, tuple((center + origin).astype(int)))], np.float32)
            if sigma.shape != (nd,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
                raise ValueError('Prompt sigma must have one finite positive value per spatial axis')
        return origin, run_shape, center, sigma

    def roi_input(origin, run_shape):
        """Features of a ROI (cached per ROI) and the shape of its network input (padded to the input multiple)."""
        nonlocal feature_runs
        cache_key = (origin, run_shape)
        sl = tuple(slice(o, o+s) for o, s in zip(origin, run_shape))
        crop = img_full[sl]
        if crop.ndim == nd:
            crop = crop[..., None]
        crop = pad_to_multiple(crop[..., :config.n_channel_in], multiple)
        if cache_key not in feature_cache:
            network_input = tf.convert_to_tensor(crop[None], tf.float32)
            feature_cache[cache_key] = feature_fn(network_input)
            if fine_enabled:
                fine_cache[cache_key] = (fine_feature_fn(network_input), network_input)
            feature_runs += 1
            if len(feature_cache) > cache_limit:
                fine_cache.pop(feature_cache.popitem(last=False)[0], None)
        feature_cache.move_to_end(cache_key)
        return cache_key, crop.shape[:nd]

    def gaussian_prompt(center, sigma, net_shape, out=None, start=None):
        # Open (broadcast) axes: the same per-voxel arithmetic as a full meshgrid, bit for bit,
        # without building nd full-ROI coordinate arrays per prompt. start: the window [start, start + net_shape)
        # of the ROI prompt (the same values).
        start = (0,) * len(net_shape) if start is None else tuple(int(v) for v in start)
        axes = np.ix_(*[np.arange(b, b + s, dtype=np.float32) for b, s in zip(start, net_shape)])
        return np.exp(-0.5 * sum(((g-center[a])/sigma[a])**2 for a, g in enumerate(axes)), out=out)

    def head_threshold(h):
        threshold = getattr(config, f'mask_thresh_{h}', None)
        if threshold == 'auto':
            threshold = (mask_thresholds or {}).get(h)
        return mask_thresh if threshold is None else float(threshold)

    def head_candidate(h, p_grid, quality, origin, run_shape, net_shape, center, sigma, fine_mask, expand,
                       expansion_step, coarse=None):
        """Candidate of one head for one prompt from its probability on the head grid (the rule of both the
        one-prompt and the batched path): threshold and prompt component (`coarse`: local_prompt_mask already
        evaluated); a cell cut by an inner ROI face is prompted again over a larger ROI (expand(...) returns
        that candidate); full-resolution mask (fine_mask(lo, mask_shape, threshold)); quality gate. None when
        it yields no candidate."""
        observer = stage_audit.get(h) if stage_audit is not None else None
        if observer is not None:
            observer.record_roi(origin, run_shape, net_shape, center + origin, getattr(config, h+'_grid'))
        diag = diagnostics[h]
        diag['attempted'] += 1
        bad_quality = h in quality and not np.isfinite(quality[h])
        if bad_quality:
            diag['quality_nonfinite'] += 1
        if not np.all(np.isfinite(p_grid)) or (bad_quality and (config.mask_quality_use_scores or quality_min[h] > 0)):
            diag['nonfinite'] += 1
            return None
        if h in quality and not bad_quality:
            diag['quality'].append(quality[h])
        threshold = head_threshold(h)
        # Same mask as thresholding the whole upsampled ROI, evaluated near the prompt.
        lo, p, foreground, mask, any_foreground = coarse if coarse is not None else local_prompt_mask(
            p_grid, run_shape, getattr(config, f'{h}_grid'), center, sigma, threshold)
        window_origin = np.asarray(origin) + lo
        if h == 'cytoplasm':
            # Expand only along clipped faces inside the image. Keep the
            # original prompt geometry, including when it came from a nucleus.
            axes_to_grow = [a for a in range(nd) if
                (origin[a] > 0 and lo[a] == 0 and np.any(np.take(mask, 0, axis=a))) or
                (origin[a] + run_shape[a] < shape[a] and lo[a] + mask.shape[a] == run_shape[a]
                 and np.any(np.take(mask, -1, axis=a)))]
            if axes_to_grow and expansion_step < config.cell_roi_expansion_steps:
                expanded = tuple(min(shape[a], 2*run_shape[a]) if a in axes_to_grow else run_shape[a]
                                 for a in range(nd))
                global_center = center + origin
                new_origin = tuple(max(0, min(int(c)-s//2, n-s))
                                   for c, s, n in zip(global_center, expanded, shape))
                diag['roi_expansions'] += 1
                return expand(global_center, (new_origin, expanded), sigma, expansion_step + 1)
            if axes_to_grow:
                diag['roi_boundary_remaining'] += 1
        fine_used = fine_enabled and any_foreground
        if fine_used:
            # The coarse window locates the object; the full-resolution head draws it.
            start, p, foreground, mask, any_foreground = fine_mask(lo, mask.shape, threshold)
            window_origin = np.asarray(origin) + start
        if observer is not None:
            # Foreground of the prompt neighbourhood (the evaluated window), not the whole ROI.
            observer.add('threshold', mask_candidate(p, foreground, window_origin))
        if not any_foreground:
            diag['empty_threshold'] += 1
            return None
        candidate = mask_candidate(p, mask, window_origin)
        if observer is not None:
            observer.add('component', candidate)
        if candidate is None:
            diag['missing_component'] += 1
            return None
        # Raw probabilities and prompt let calibration re-threshold without new forwards;
        # the window around the mask (only when candidates are collected) also allows lower ones.
        candidate.update(probability=candidate['score'], prompt_center=center + origin,
                         prompt_sigma=np.asarray(sigma, np.float64), threshold=threshold)
        if candidate_sink is not None:
            if fine_used:                              # the fine window already has a margin
                window = (np.asarray(window_origin), p)
            else:                                      # widen the tight coarse window by the same margin
                from cytonucs_postprocess import grid_probability_window
                grid_h = getattr(config, f'{h}_grid')
                margin = np.array([2 * g + 3 for g in grid_h])
                w_lo = np.maximum(0, np.asarray(lo) - margin)
                w_hi = np.minimum(np.asarray(run_shape), np.asarray(lo) + np.array(p.shape) + margin)
                window = (np.asarray(origin) + w_lo, grid_probability_window(p_grid, grid_h, w_lo, w_hi))
            candidate['probability_window'] = (window[0], np.asarray(window[1], np.float32))
        if h in quality and not bad_quality:
            candidate['predicted_quality'] = quality[h]
            if config.mask_quality_use_scores:
                candidate['quality'] = quality[h]
                candidate['score'] = candidate['score'] * quality[h]
                candidate['score_scaled_by_quality'] = True
            if quality[h] < quality_min[h]:
                diag['quality_rejected'] += 1
                return None
        diag['accepted'] += 1
        return candidate

    def run(center, distances, head, roi=None, fixed_sigma=None, expansion_step=0):
        """Candidates (nucleus, cytoplasm) of one prompt, one network pass."""
        nonlocal runs
        origin, run_shape, center, sigma = prepare(center, distances, head, roi, fixed_sigma)
        cache_key, net_shape = roi_input(origin, run_shape)
        prompt = tf.convert_to_tensor(gaussian_prompt(center, sigma, net_shape)[None, ..., None], tf.float32)
        hidden = {}
        if head == 'cytoplasm':
            cell_logits = cell_fn([feature_cache[cache_key][i] for i in (1, 2, 5, 6)] + [prompt])
            if fine_enabled:
                cell_logits = list(cell_logits); hidden['cytoplasm'] = cell_logits.pop()
                cell_logits = cell_logits if config.mask_quality_head else cell_logits[0]
            quality = {}
            if config.mask_quality_head:
                cell_logits, cell_quality = cell_logits
                quality['cytoplasm'] = float(np.asarray(cell_quality).ravel()[0])
            head_logits = [('cytoplasm', cell_logits)]
        else:
            logits = refine_fn(list(feature_cache[cache_key]) + [prompt, prompt])
            if fine_enabled:
                logits = list(logits); hidden['cytoplasm'] = logits.pop(); hidden['nucleus'] = logits.pop()
            quality = {h:float(np.asarray(q).ravel()[0]) for h,q in zip(
                ('nucleus','cytoplasm'), logits[2:])} if config.mask_quality_head else {}
            head_logits = list(zip(('nucleus', 'cytoplasm'), logits))
        runs += 1
        candidates = [None, None]
        for h, logit in head_logits:
            # Only the target channel leaves the device; the grid map is finite exactly
            # when its linear upsampling is.
            p_grid = tf.nn.softmax(logit, axis=-1)[0, ..., 1].numpy()
            fine_mask = (lambda lo, mask_shape, threshold, h=h, logit=logit: _fine_prompt_mask(
                fine_head_fns[h], logit, hidden[h], [*fine_cache[cache_key], prompt], getattr(config, f'{h}_grid'),
                lo, mask_shape, run_shape, net_shape, center, sigma, threshold))
            expand = lambda c, roi, s, step: run(c, None, 'cytoplasm', roi=roi, fixed_sigma=s, expansion_step=step)[1]
            candidates[0 if h == 'nucleus' else 1] = head_candidate(
                h, p_grid, quality, origin, run_shape, net_shape, center, sigma, fine_mask, expand, expansion_step)
        return candidates

    batch = int(getattr(config, 'refinement_batch', 0) or 0)
    batched = batch > 0

    def fine_windows(h, logit, hidden_h, cache_key, items, coarse, run_shape, net_shape):
        """Full-resolution probability of head h for a chunk of prompts over a window that contains each
        prompt's first fine window: {k: (begin, map)}. Prompts with the same window size (_window_size) go
        through the fine head together."""
        from cytonucs_fine_mask import inference_window
        grid = getattr(config, f'{h}_grid')
        windows, groups = {}, OrderedDict()
        for k, c in enumerate(coarse):
            if c is not None and c[4]:
                windows[k] = inference_window(c[0], c[3].shape, run_shape, net_shape, grid, [2 * g + 3 for g in grid])
                groups.setdefault(_window_size(windows[k][1] - windows[k][0], net_shape, grid), []).append(k)
        features, network_input = fine_cache[cache_key]
        maps = {}
        for size, members in groups.items():
            per_call = max(1, int(FINE_BATCH_VOXELS // int(np.prod(size))))
            runner = _fine_window_runner(model, h, fine_head_fns[h], grid, size)
            for m0 in range(0, len(members), per_call):
                part = members[m0:m0 + per_call]
                begins = np.stack([_place_window(*windows[k], size, net_shape, grid) for k in part])
                prompt_windows = np.empty((len(part),) + tuple(size), np.float32)
                for j, k in enumerate(part):
                    gaussian_prompt(*items[k], size, out=prompt_windows[j], start=begins[j])
                probability = runner(logit, hidden_h, features, network_input, tf.convert_to_tensor(prompt_windows[..., None]),
                                     tf.constant(part, tf.int32), tf.constant(begins, tf.int32)).numpy()
                for k, begin, prob in zip(part, begins, probability):
                    maps[k] = (begin, prob)
        return maps

    def run_batched(specs):
        """Candidates (nucleus, cytoplasm) of many prompts, spec = (head, centre, distances, roi, sigma, step).
        Prompts sharing a ROI go through the refinement network `batch` at a time (fewer for large ROIs),
        the next chunk on the device while the previous one is assembled on the host. The full-resolution
        head runs on windows of a few sizes that contain each prompt's first fine window, and the windows of
        _fine_prompt_mask are cut from them (a grown window outside it gets its own pass): values may
        differ within 2 voxels of a window face, where the per-window pass saw zero padding. Coarse
        candidates equal the one-prompt path. Cells cut by an inner ROI face are prompted over the larger
        ROI in a following round."""
        nonlocal runs
        out = [[None, None] for _ in specs]
        prepared = [prepare(spec[1], spec[2], spec[0], spec[3], spec[4]) for spec in specs]
        groups = OrderedDict()
        for i, prep in enumerate(prepared):
            groups.setdefault((prep[0], prep[1]), []).append(i)
        follow = []
        for (origin, run_shape), members in groups.items():
            cache_key, net_shape = roi_input(origin, run_shape)
            features = feature_cache[cache_key]
            repeated = {}
            grid = tuple(int(v) for v in config.nucleus_grid)
            grid_roi = tuple(int(v) // g for v, g in zip(net_shape, grid))
            # Refinement over a window around each prompt (inference.refinement_windows): the refiner is local
            # (convolutions and pooling at aligned window starts; the prompt enters as offsets from its own
            # peak), so inside the window minus its receptive field it gives the full-ROI values.
            windowed = (bool(getattr(config, 'refinement_windows', False))
                        and tuple(config.cytoplasm_grid) == grid
                        and all(tuple(int(v) for v in f.shape[1:1 + nd]) == grid_roi for f in features))
            reach, factor = _refiner_reach(config)
            a_cell = np.asarray(getattr(config, 'rays_anisotropy_cytoplasm', config.anisotropy), np.float64)
            cell_extent = 2 * float(getattr(config, 'median_max_dist_cytoplasm', None) or 60.0) / a_cell

            def window_of(kind, center, sigma):
                """Window (start, size) in head-grid nodes: 1.5x the prompt's object extent (4 sigma; for a
                nucleus prompt at least 1.5x a median cell, which the same pass draws) and the fine window's margin,
                plus the reach on both sides; aligned to the refiner's pooling; the whole axis when it does not fit."""
                extent = 6 * np.asarray(sigma, np.float64)
                if kind == 'nucleus':
                    extent = np.maximum(extent, 1.5 * cell_extent)
                extent = extent + 2 * np.array([3 * g + 3 for g in grid])
                start, size = [], []
                for ax in range(nd):
                    need = int(np.ceil(extent[ax] / grid[ax])) + 2 * int(reach[ax])
                    w = next((v for v in REFINE_WINDOW_SIZES if v >= need and v % factor[ax] == 0), None)
                    if w is None or w >= grid_roi[ax]:
                        start.append(0); size.append(grid_roi[ax]); continue
                    b0 = int(round(center[ax] / grid[ax] - w / 2))
                    start.append(max(0, min(b0, grid_roi[ax] - w)) // factor[ax] * factor[ax]); size.append(w)
                return np.array(start), tuple(size)

            def valid(start, size):
                """Full-resolution [lo, hi) where a window gives the full-ROI values (all of an axis at a ROI face)."""
                lo = [(b + int(r) if b > 0 else 0) * g for b, r, g in zip(start, reach, grid)]
                hi = [(b + w - int(r) if b + w < G else G) * g for b, w, r, g, G in zip(start, size, reach, grid, grid_roi)]
                return np.array(lo), np.array(hi)

            def parse(kind, outputs, n):
                """(logits, hidden, quality) per head from the outputs of the nucleus or cell network."""
                hidden = {}
                if kind == 'cytoplasm':
                    if fine_enabled:
                        outputs = list(outputs); hidden['cytoplasm'] = outputs.pop()
                        outputs = outputs if config.mask_quality_head else outputs[0]
                    quality = {}
                    if config.mask_quality_head:
                        outputs, cell_quality = outputs
                        quality['cytoplasm'] = np.asarray(cell_quality).reshape(n, -1)[:, 0]
                    return {'cytoplasm': outputs}, hidden, quality
                if fine_enabled:
                    outputs = list(outputs); hidden['cytoplasm'] = outputs.pop(); hidden['nucleus'] = outputs.pop()
                quality = ({h: np.asarray(q).reshape(n, -1)[:, 0] for h, q in zip(('nucleus', 'cytoplasm'), outputs[2:])}
                           if config.mask_quality_head else {})
                return dict(zip(('nucleus', 'cytoplasm'), outputs[:2])), hidden, quality

            def full_roi(kind, ids):
                """The network over the whole ROI for the prompts `ids` (indices into prepared)."""
                n = len(ids)
                if kind not in repeated:
                    # Backbone features repeated once per ROI for the largest chunk; smaller calls take a slice.
                    used = features if kind == 'nucleus' else [features[k] for k in (1, 2, 5, 6)]
                    largest = max(len(c) for kd, c in chunks if kd == kind)
                    repeated[kind] = [tf.repeat(f, max(largest, n), axis=0) for f in used]
                if int(repeated[kind][0].shape[0]) < n:
                    repeated[kind] = [tf.repeat(f[:1], n, axis=0) for f in repeated[kind]]
                inputs = repeated[kind] if n == int(repeated[kind][0].shape[0]) else [f[:n] for f in repeated[kind]]
                buffer = np.empty((n,) + tuple(net_shape), np.float32)
                for k, i in enumerate(ids):
                    gaussian_prompt(prepared[i][2], prepared[i][3], net_shape, out=buffer[k])
                prompts = tf.convert_to_tensor(buffer[..., None])
                if kind == 'cytoplasm':
                    return prompts, cell_fn(inputs + [prompts])
                return prompts, refine_fn(inputs + [prompts, prompts])

            def launch(kind, chunk):
                """Prompts and refinement network of one chunk; the device runs it while the previous
                chunk is assembled on the host."""
                nonlocal runs
                runs += len(chunk)
                if not windowed:
                    return ('full', chunk) + full_roi(kind, chunk)
                windows = [window_of(kind, prepared[i][2], prepared[i][3]) for i in chunk]
                groups = OrderedDict()
                for k, (_, size) in enumerate(windows):
                    groups.setdefault(size, []).append(k)
                used = features if kind == 'nucleus' else [features[k] for k in (1, 2, 5, 6)]
                fn = cell_fn if kind == 'cytoplasm' else refine_fn
                parts = []
                for size, ks in groups.items():
                    per_call = max(1, REFINE_BATCH_NODES // int(np.prod(size)))
                    runner = _refine_window_runner(model, kind, fn, size)
                    full_size = tuple(w * g for w, g in zip(size, grid))
                    for m0 in range(0, len(ks), per_call):
                        part = ks[m0:m0 + per_call]
                        starts = np.stack([windows[k][0] for k in part])
                        prompt_windows = np.empty((len(part),) + full_size, np.float32)
                        for j, k in enumerate(part):
                            gaussian_prompt(prepared[chunk[k]][2], prepared[chunk[k]][3], full_size,
                                            out=prompt_windows[j], start=starts[j] * np.array(grid))
                        parts.append((part, starts, size, runner(used, tf.convert_to_tensor(prompt_windows[..., None]),
                                                                 tf.constant(starts, tf.int32))))
                return ('windows', chunk, windows, parts)

            def process(mode, kind, chunk, *launched):
                n = len(chunk)
                items = [(prepared[i][2], prepared[i][3]) for i in chunk]
                heads = ('cytoplasm',) if kind == 'cytoplasm' else ('nucleus', 'cytoplasm')
                full_rows = {}                             # chunk index -> (prompt, logits, hidden) over the whole ROI
                if mode == 'full':
                    prompts, outputs = launched
                    logits, hidden, quality = parse(kind, outputs, n)
                    p_grids = {h: tf.nn.softmax(logits[h], axis=-1)[..., 1].numpy() for h in heads}
                    coarse = {}
                    for h in heads:
                        threshold = head_threshold(h)
                        coarse[h] = [local_prompt_mask(p_grids[h][k], run_shape, grid, center, sigma, threshold)
                                     if np.all(np.isfinite(p_grids[h][k])) else None for k, (center, sigma) in enumerate(items)]
                else:
                    windows, parts = launched
                    p_grids = {h: np.zeros((n,) + grid_roi, np.float32) for h in heads}
                    quality, coarse, embedded = {}, {h: [None] * n for h in heads}, {h: [] for h in heads}
                    for part, starts, size, outputs in parts:
                        logit_w, hidden_w, quality_w = parse(kind, outputs, len(part))
                        starts_t = tf.constant(starts, tf.int32)
                        for h in heads:
                            threshold = head_threshold(h)
                            prob = tf.nn.softmax(logit_w[h], axis=-1)[..., 1].numpy()
                            for j, k in enumerate(part):
                                p_grids[h][k][tuple(slice(int(b), int(b) + w) for b, w in zip(starts[j], size))] = prob[j]
                                if np.all(np.isfinite(prob[j])):
                                    # The prompt mask evaluated on the window (zero probability outside it).
                                    offset = starts[j] * np.array(grid)
                                    extent = np.minimum(np.array(size) * np.array(grid), np.asarray(run_shape) - offset)
                                    lo, pw, fg, mask, any_fg = local_prompt_mask(prob[j], tuple(int(v) for v in extent), grid,
                                                                                 items[k][0] - offset, items[k][1], threshold)
                                    coarse[h][k] = (np.asarray(lo) + offset, pw, fg, mask, any_fg)
                            if fine_enabled:
                                embedded[h].append((part, _embed_windows(logit_w[h], starts_t, grid_roi),
                                                    _embed_windows(hidden_w[h], starts_t, grid_roi)))
                        for hq, values in quality_w.items():
                            quality.setdefault(hq, np.zeros(n))[part] = values
                if mode == 'windows':
                    # A prompt whose mask (with the fine window's margin and one extra grid node) leaves the part of
                    # its window that equals the full ROI is refined over the whole ROI.
                    outside = []
                    for k in range(n):
                        lo_v, hi_v = valid(*windows[k])
                        for h in heads:
                            c = coarse[h][k]
                            if c is None or not c[4]:
                                continue
                            margin = np.array([2 * g + 3 for g in grid])
                            lo = np.asarray(c[0]) - margin
                            hi = np.asarray(c[0]) + np.array(c[3].shape) + margin + np.array(grid)
                            if np.any(np.maximum(lo, 0) < lo_v) or np.any(np.minimum(hi, np.asarray(run_shape)) > hi_v):
                                outside.append(k); break
                    diagnostics[kind]['windowed'] += n
                    diagnostics[kind]['window_fallbacks'] += len(outside)
                    if outside:
                        prompts_f, outputs_f = full_roi(kind, [chunk[k] for k in outside])
                        logit_f, hidden_f, quality_f = parse(kind, outputs_f, len(outside))
                        for j, k in enumerate(outside):
                            full_rows[k] = (prompts_f[j:j + 1], {h: logit_f[h][j] for h in heads},
                                            {h: hidden_f[h][j] for h in hidden_f})
                            for h in heads:
                                p_grids[h][k] = tf.nn.softmax(logit_f[h][j], axis=-1)[..., 1].numpy()
                                coarse[h][k] = (local_prompt_mask(p_grids[h][k], run_shape, grid, *items[k], head_threshold(h))
                                                if np.all(np.isfinite(p_grids[h][k])) else None)
                            for h, values in quality_f.items():
                                quality[h][k] = values[j]
                    # Logits and fine features over the ROI grid for the fine head: windows placed in zeros (the
                    # fine head reads them only inside the valid part), whole-ROI rows as computed.
                    logits, hidden = {h: None for h in heads}, {}
                    if fine_enabled:
                        inverse = np.argsort([k for part, _, _ in embedded[heads[0]] for k in part])
                        rows = sorted(full_rows)
                        for h in heads:
                            logits[h] = tf.gather(tf.concat([e[1] for e in embedded[h]], 0), inverse)
                            hidden[h] = tf.gather(tf.concat([e[2] for e in embedded[h]], 0), inverse)
                            if rows:
                                index = tf.constant([[k] for k in rows], tf.int32)
                                logits[h] = tf.tensor_scatter_nd_update(logits[h], index, tf.stack([full_rows[k][1][h] for k in rows]))
                                hidden[h] = tf.tensor_scatter_nd_update(hidden[h], index, tf.stack([full_rows[k][2][h] for k in rows]))
                    prompts = None

                def prompt_row(k):
                    if mode == 'full':
                        return prompts[k:k + 1]
                    if k in full_rows:
                        return full_rows[k][0]
                    return tf.convert_to_tensor(gaussian_prompt(*items[k], net_shape)[None, ..., None], tf.float32)

                for h in heads:
                    logit = logits[h]
                    maps = (fine_windows(h, logit, hidden[h], cache_key, items, coarse[h], run_shape, net_shape)
                            if fine_enabled else {})

                    def probability_at(start, stop, k, h=h, logit=logit, maps=maps):
                        end = np.minimum(stop, np.asarray(run_shape))
                        if k in maps:
                            begin, prob = maps[k]
                            if np.all(begin <= start) and np.all(end <= begin + np.array(prob.shape)):
                                return prob[tuple(slice(int(s0 - b), int(e - b)) for s0, e, b in zip(start, end, begin))]
                        # A grown window outside the computed one: the per-window pass of _fine_prompt_mask, with
                        # the whole-ROI refinement when the window leaves the valid part of a refinement window.
                        logit_k, hidden_k = logit[k:k + 1], hidden[h][k:k + 1]
                        if mode == 'windows' and k not in full_rows:
                            lo_v, hi_v = valid(*windows[k])
                            if np.any(np.asarray(start) < lo_v) or np.any(np.asarray(stop) + np.array(grid) > hi_v):
                                diagnostics[kind]['window_fallbacks'] += 1
                                prompts_f, outputs_f = full_roi(kind, [chunk[k]])
                                logit_f, hidden_f, _ = parse(kind, outputs_f, 1)
                                full_rows[k] = (prompts_f, {hh: logit_f[hh][0] for hh in heads},
                                                {hh: hidden_f[hh][0] for hh in hidden_f})
                        if k in full_rows:
                            logit_k, hidden_k = full_rows[k][1][h][None], full_rows[k][2][h][None]
                        arrays = fine_window_arrays(logit_k, hidden_k, [*fine_cache[cache_key], prompt_row(k)],
                                                    start, stop, grid)
                        prob = tf.nn.softmax(fine_head_fns[h](arrays), axis=-1)[0, ..., 1].numpy()
                        return prob[tuple(slice(0, int(e - s0)) for s0, e in zip(start, end))]

                    for k, i in enumerate(chunk):
                        center, sigma = items[k]
                        fine_mask = (lambda lo, mask_shape, threshold, k=k, center=center, sigma=sigma,
                                     probability_at=probability_at: _fine_mask_windows(
                                         lambda start, stop: probability_at(start, stop, k), lo, mask_shape,
                                         run_shape, net_shape, grid, center, sigma, threshold))
                        expand = (lambda c, roi, s, step, i=i:
                                  follow.append((i, ('cytoplasm', c, None, roi, s, step))) or _PENDING)
                        out[i][0 if h == 'nucleus' else 1] = head_candidate(
                            h, p_grids[h][k], {q: float(v[k]) for q, v in quality.items()}, origin, run_shape,
                            net_shape, center, sigma, fine_mask, expand, specs[i][5], coarse=coarse[h][k])

            # Prompts per refinement call: `batch`, fewer for large (3D) ROIs.
            nodes = int(np.prod([-(-s // g) for s, g in zip(net_shape, config.nucleus_grid)]))
            step = max(1, min(batch, REFINE_BATCH_NODES // nodes))
            chunks = []
            for kind in ('nucleus', 'cytoplasm'):
                ids = [i for i in members if specs[i][0] == kind]
                chunks += [(kind, ids[b0:b0 + step]) for b0 in range(0, len(ids), step)]
            pending = launch(*chunks[0]) if chunks else None
            for j, (kind, _) in enumerate(chunks):
                current = pending
                pending = launch(*chunks[j + 1]) if j + 1 < len(chunks) else None
                process(current[0], kind, *current[1:])
        if follow:
            for (i, _), (_, cell) in zip(follow, run_batched([spec for _, spec in follow])):
                out[i][1] = cell
        return out

    cap = int(getattr(config, 'max_prompts_per_head', 0) or 0)

    def capped(head, items):
        # At most `cap` prompts per head and image, the most probable ones (in their order): an untrained
        # detector can propose thousands per 3D volume, each refined over its ROI.
        if not cap or len(items) <= cap:
            return items
        keep = sorted(sorted(range(len(items)), key=lambda i: -float(items[i][2]))[:cap])
        diagnostics[head]['prompts_capped'] = len(items) - cap
        return [items[i] for i in keep]
    anchored = [('nucleus', ctr, d, score) for ctr, d, score in capped('nucleus', anchors(nuc_labels, nuc_coords, nuc_probs, nuc_dists))]
    anchored += [('cytoplasm', ctr, d, score) for ctr, d, score in capped('cytoplasm', anchors(cyto_labels, cyto_coords, cyto_probs, cyto_dists))]
    jobs = [job[:3] for job in anchored]
    results = [None] * len(jobs)
    if batched:
        for index, ((head, ctr, d), (n, c)) in enumerate(zip(jobs, run_batched([(head, ctr, d, None, None, 0)
                                                                                  for head, ctr, d in jobs]))):
            results[index] = (n if head == 'nucleus' else None, c)
    else:
        # Group by full ROI (origin AND shape), retaining original candidate indices.
        roi_jobs = [(head, roi_for(ctr, d, head), (ctr, d)) for head, ctr, d in jobs]
        for group in _group_prompt_jobs(roi_jobs, lambda roi: roi):
            for index, (head, _, (ctr, d)) in group:
                n, c = run(ctr, d, head)
                results[index] = (n if head == 'nucleus' else None, c)
    # Detection probability of the prompt's peak (0 for a label without a peak), for offline threshold tuning.
    for (_, _, _, score), pair in zip(anchored, results):
        for candidate in pair:
            if candidate is not None:
                candidate['detection_score'] = float(score)
    # Restore old candidate ordering: competition tie-breaks and evidence stay stable.
    for (head, _, _), (n, c) in zip(jobs, results):
        if head == 'nucleus':
            if n is not None:
                nuc_candidates.append(n)
                evidence.append((n, c))
            cell_candidates.append(c)
        else:
            cell_candidates.append(c)

    # Second pass: window-suppressed peaks that no first-pass object covers.
    residual_heads = [h for h in ('nucleus', 'cytoplasm')
                      if residual and residual.get(h) is not None and len(residual[h]['coords'])
                      and getattr(config, 'mask_assembly_' + h) != 'voxel']
    if residual_heads and prompt_sigmas is None:
        def flag(candidate):
            if candidate is not None:
                candidate['residual'] = True
            return candidate

        occupied = {h: assemble_head(config, h, nuc_candidates if h == 'nucleus' else cell_candidates,
                                     shape)[0] for h in residual_heads}
        for head in residual_heads:
            r = residual[head]
            diagnostics[head]['residual_peaks'] = int(len(r['coords']))
            coords = np.clip(np.round(np.asarray(r['coords'])).astype(int), 0, np.array(shape) - 1)
            free = [k for k, ctr in enumerate(coords) if occupied[head][tuple(ctr)] == 0]
            diagnostics[head]['residual_prompts'] = len(free)
            residual_jobs = [(head, roi_for(coords[k], r['dists'][k], head), (coords[k], r['dists'][k]))
                             for k in free]
            ordered = [(ctr, d) for group in _group_prompt_jobs(residual_jobs, lambda roi: roi)
                       for _, (_, _, (ctr, d)) in group]
            outs = (run_batched([(head, ctr, d, None, None, 0) for ctr, d in ordered]) if batched else
                    [run(ctr, d, head) for ctr, d in ordered])
            for n, c in outs:
                if head == 'nucleus' and config.mask_assembly_cytoplasm == 'voxel':
                    c = None    # voxel competition has no 'fill free voxels only' semantics
                if head == 'nucleus' and n is not None:
                    nuc_candidates.append(flag(n))
                    evidence.append((n, flag(c)))
                cell_candidates.append(flag(c))
    if candidate_sink is not None:
        candidate_sink.update(nucleus=nuc_candidates, cytoplasm=cell_candidates, evidence=evidence)
    nuclei, ns = assemble_head(config, 'nucleus', nuc_candidates, shape,
                               stage_audit.get('nucleus') if stage_audit is not None else None)
    cells, cs = assemble_head(config, 'cytoplasm', cell_candidates, shape,
                              stage_audit.get('cytoplasm') if stage_audit is not None else None)
    if config.mask_quality_min == 'auto' and quality_gates:
        # Calibrated size gate on written objects (the stage audit reports the ungated output).
        nuclei, diagnostics['nucleus']['area_gate_removed'] = drop_small_instances(
            nuclei, quality_gates.get('nucleus_min_area', 0))
        cells, diagnostics['cytoplasm']['area_gate_removed'] = drop_small_instances(
            cells, quality_gates.get('cytoplasm_min_area', 0))
    pairs = assign_final_masks(nuclei, cells, getattr(config, 'pair_min_overlap', 0.1),
                               evidence, getattr(config, 'pair_min_mask_iou', 0.25))
    for h, stats in (('nucleus', ns), ('cytoplasm', cs)):
        quality_values = diagnostics[h].pop('quality')
        diagnostics[h]['quality_quantiles_0_25_50_75_100'] = (
            np.quantile(quality_values, [0, .25, .5, .75, 1]).tolist() if quality_values else [])
        diagnostics[h].update(stats)
    model.last_refinement_diagnostics = diagnostics
    if verbose:
        import json
        print('[refinement diagnostics] ' + json.dumps(diagnostics, sort_keys=True))
        print(f'[refinement mask] nucleus={ns} cytoplasm={cs} pairs={len(pairs)} '
              f'feature forwards={feature_runs}, prompt forwards={runs}')
    return nuclei, cells, pairs

def _predict_instances_prompted_legacy(model, config, img_full, nuc_coords, nuc_probs,
                               cyto_coords=None, cyto_probs=None,
                               mask_thresh=0.5, iou_dedup=0.5, verbose=True,
                               nuc_dists=None, cyto_dists=None,
                               nuc_labels=None, cyto_labels=None,
                               nms_thresh=None):
    """
    Инстансы определяются в 3D как в каноническом StarDist: NMS -> полиэдры.
    Рефайн только ДОБИРАЕТ форму, по 2D-слайсам.

    Цикл идёт по МЕТКАМ, а не по пикам: polyhedron_to_label отдаёт
    непересекающиеся метки, одна метка = один инстанс.

    Форма инстанса = полиэдр (отрендерен по ВСЕМУ объёму, значит целостен по Z и
    не обрезается окном прогона) ОБЪЕДИНЁННЫЙ с выращенной маской внутри окна.
    Выращивание послойное: на слайсе центра берётся связная компонента,
    содержащая центр, дальше вверх и вниз по Z берутся компоненты, касающиеся
    уже принятой части соседнего слайса.

    ПОСЛЕ добора формы применяется ТОТ ЖЕ критерий, что в NMS StarDist
    (stardist3d_impl.cpp: A_inter/A_min > nms_thresh) — см. _place.

    Наборы лучей у голов РАЗНЫЕ (см. CytoNucsConfig.rays_nucleus /
    rays_cytoplasm), поэтому сигма промпта считается по набору своей головы.

    Один прогон на инстанс ядра даёт И маску ядра, И маску его клетки — ровно
    как учит генератор (промпт на ядре, cytoplasm_target = родительская клетка).
    Многоядерность выходит сама: два ядра приходят в одну и ту же клетку, второе
    просто записывается в assignments.

    ДИАГНОСТИКА ПОТЕРЬ. Полиэдров на входе может быть больше, чем инстансов на
    выходе, и до сих пор было не видно, где именно они исчезают. Теперь считаются
    все каналы утечки:
      occupied   — _place вернул 0: вся территория полиэдра уже занята ранее
                   записанными инстансами (w пусто). Раньше терялось молча.
                   Достижимо, когда полиэдр покрыт НЕСКОЛЬКИМИ соседями, каждый
                   из которых по отдельности не дотянул до nms_thresh: при
                   nms_thresh=0.3 хватает четырёх соседей по 25%. Для плотного
                   сфероида с 5-6 соседями на клетку это штатный сценарий, и он
                   даёт связные чёрные дыры в предсказании.
      seen       — метка клетки уже записана другим ядром. Законно для настоящей
                   многоядерной клетки, но так же выглядит случай, когда полиэдр
                   ядра залез в соседнюю клетку и «увёл» её.
      no_overlap — полиэдр ядра не пересёкся ни с одной меткой цитоплазмы.
    Печатается баланс: полиэдры - merged - occupied должно сойтись с записанными.

    iou_dedup оставлен в сигнатуре для обратной совместимости вызовов.
    Возвращает (nucleus_instances, cytoplasm_instances, assignments).
    """
    import numpy as np, tensorflow as tf
    from scipy.ndimage import zoom, find_objects
    from scipy.ndimage import label as _cc_label

    nd = int(config.ndim)
    patch = tuple(int(p) for p in config.train_patch_size)
    n_img = int(config.n_channel_in)
    # Модель строит вход на K промпт-пар (K = train_batch_size). На инференсе
    # промпт всегда один: слот 0 занят, остальные нулевые, читается строка 0.
    n_pr = max(1, int(getattr(config, 'prompts_per_crop', config.train_batch_size)))
    aniso = np.asarray(getattr(config, 'anisotropy', (1.,) * nd), np.float32)
    # nms_thresh приходит ОТ ПРЕДИКТОРА. Прежнее чтение из config давало
    # значение из json и не менялось при переборе в optimize_thresholds —
    # то есть отбор инстансов в _place шёл мимо подбора.
    nms_thresh = float(getattr(config, 'nms_thresh', 0.3)
                       if nms_thresh is None else nms_thresh)

    if nuc_labels is None and cyto_labels is None:
        raise ValueError("predict_instances_prompted: нужны nuc_labels/cyto_labels "
                         "(результат polyhedron_to_label) — они и ЕСТЬ инстансы")

    nuc_labels = None if nuc_labels is None else np.asarray(nuc_labels, np.int32)
    cyto_labels = None if cyto_labels is None else np.asarray(cyto_labels, np.int32)

    shape = tuple(int(s) for s in img_full.shape[:nd])
    nuc_out = np.zeros(shape, np.int32)
    cyto_out = np.zeros(shape, np.int32)
    area_n, area_c, assignments = {}, {}, {}

    # Окно прогона = train_patch_size. Универсально: от глубины стека не
    # зависит (пробовалось брать Z целиком — на 250 срезах не влезет в память).
    run_shape = tuple(min(patch[a], shape[a]) for a in range(nd))
    grids = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in run_shape],
                        indexing='ij')

    def _verts(name):
        r = getattr(config, f'rays_{name}', None) or getattr(config, 'rays', None)
        try:
            return np.asarray(r.vertices, np.float32)      # (n_rays, nd)
        except Exception:
            return None

    _verts_nuc, _verts_cyto = _verts('nucleus'), _verts('cytoplasm')

    def _anchors(labels, coords, probs, dists, objs):
        """Для каждой метки — точка промпта, её лучи и уверенность.
        objs — результат find_objects(labels), считается ОДИН раз снаружи:
        на томе 17x2048x2048 он не бесплатный, а нужен и здесь, и в _place."""
        res = {}
        if labels is None:
            return res
        if coords is not None and len(coords) > 0:
            hi = np.array(labels.shape) - 1
            for j in range(len(coords)):
                c = np.clip(np.asarray(coords[j], int), 0, hi)
                lab = int(labels[tuple(c)])
                if lab == 0:
                    continue
                p = float(probs[j]) if probs is not None else 1.0
                if lab in res and res[lab][2] >= p:
                    continue
                res[lab] = (c, None if dists is None else np.asarray(dists[j]), p)

        for i, sl_o in enumerate(objs, start=1):
            if sl_o is None or i in res:
                continue
            sub = (labels[sl_o] == i)
            loc = np.array([s.start for s in sl_o])
            ctr = np.array([(s.start + s.stop - 1) // 2 for s in sl_o], int)
            if not sub[tuple(ctr - loc)]:
                pts = np.argwhere(sub)
                ctr = pts[np.argmin(np.abs(pts + loc - ctr).sum(axis=1))] + loc
            res[i] = (ctr, None, 0.0)
        return res

    def _sigma(r_hint, rays_at_peak, verts):
        """Сигма промпта из ЛУЧЕЙ ЭТОГО пика, а не из глобальной медианы:
        генератор строит промпт как extent/4 по bbox конкретного объекта.
        Rays_GoldenSpiral хранит verts = dir/anisotropy, поэтому |verts*dist|
        уже в вокселях — единицы совпадают с bbox генератора. Набор берётся
        ПО ГОЛОВЕ: направления у ядра и цитоплазмы разные, а проверка по длине
        (96 == 96) их не различает."""
        if (rays_at_peak is not None and verts is not None
                and len(rays_at_peak) == len(verts)):
            ext = np.abs(verts * np.asarray(rays_at_peak, np.float32)[:, None]).max(axis=0)
            return np.maximum(2.0 * ext / 4.0, 1.0)          # extent = 2 * полуось
        return np.maximum(r_hint / np.maximum(aniso, 1e-6) / 2.0, 1.0)

    def _run(center, r_hint, rays_at_peak=None, head='nucleus'):
        """Возвращает (sl, cl, nucleus_mask, cytoplasm_mask):
        sl — окно в координатах тома, cl — центр промпта в координатах окна."""
        lo = [max(0, min(int(center[a]) - run_shape[a] // 2,
                         shape[a] - run_shape[a])) for a in range(nd)]
        sl = tuple(slice(lo[a], lo[a] + run_shape[a]) for a in range(nd))
        crop = img_full[sl]
        if crop.ndim == nd:
            crop = crop[..., None]
        cl = np.clip(np.asarray(center, int) - np.asarray(lo),
                     0, np.asarray(run_shape) - 1)
        sig = _sigma(r_hint, rays_at_peak,
                     _verts_cyto if head == 'cytoplasm' else _verts_nuc)
        d2 = sum(((g - cl[a]) / sig[a]) ** 2 for a, g in enumerate(grids))
        pr = np.exp(-0.5 * d2).astype(np.float32)[..., None]
        _pad = np.zeros(pr.shape[:-1] + (2 * (n_pr - 1),), np.float32)
        x = np.concatenate([crop[..., :n_img], pr, pr, _pad], axis=-1)
        x = pad_to_multiple(x, architecture_plan(config)['input_multiple'])[None]
        y = model.keras_model(tf.convert_to_tensor(x, tf.float32), training=False)

        def _get(k):
            # Выходы рефайна склеены по батчу как k*B + b; B=1, берём k=0.
            a = y[k].numpy()[0]
            a = a[..., 0] if a.shape[-1] == 1 else a
            if a.shape != run_shape:
                grid = config.nucleus_grid if k == 'nucleus_mask' else config.cytoplasm_grid
                a = upsample_grid(a, run_shape, grid)
            return a
        return sl, cl, _get('nucleus_mask'), _get('cytoplasm_mask')

    def _grow(m, cl):
        """Послойное выращивание маски от центра промпта.
        3D-связность здесь не годится: при anisotropy=[5,1,1] объект толщиной
        3-5 срезов легко распадается по Z, а по слайсам он всегда цел.

        СЕМЯ — максимум маски в окрестности cl, а не сам воксель cl. cl это пик
        prob из NMS; у плоской клетки prob почти плато и пик смещён (замерено
        12.5 px на объекте 153 px), а mask приходит из ОТДЕЛЬНОЙ рефайн-ветки
        со своим шумом. Прежняя версия при m[cl] <= порога возвращала пустоту,
        и рефайн для такого инстанса не давал НИЧЕГО — оставался сырой полиэдр,
        молча, без счётчика в статистике потерь.

        Радиус поиска — половина окна прогона по каждой оси, но не больше:
        дальше это уже соседний объект. Никаких констант: всё из run_shape.
        """
        out = np.zeros_like(m, bool)
        if not m.any():
            return out

        cl = np.asarray(cl, int)
        rad = np.maximum(np.asarray(m.shape) // 8, 1)

        def _seed(plane, c2d, rad2d):
            """(индекс компоненты, cc, n) по максимуму plane в окне вокруг c2d."""
            cc, n = _cc_label(plane)
            if n == 0:
                return 0, cc, n
            lo = np.maximum(c2d - rad2d, 0)
            hi = np.minimum(c2d + rad2d + 1, np.asarray(plane.shape))
            win = tuple(slice(int(lo[a]), int(hi[a])) for a in range(len(c2d)))
            sub = cc[win]
            if not sub.any():
                return 0, cc, n
            # компонента, ближайшая к c2d среди попавших в окно
            pts = np.argwhere(sub > 0)
            j = int(np.argmin(np.abs(pts + lo - c2d).sum(axis=1)))
            return int(sub[tuple(pts[j])]), cc, n

        if nd == 2:
            k, cc, n = _seed(m, cl, rad)
            return (cc == k) if k > 0 else out

        z0 = int(np.clip(cl[0], 0, m.shape[0] - 1))
        k, cc, n = _seed(m[z0], cl[1:], rad[1:])
        if k == 0:
            # центральный слайс пуст — ищем ближайший непустой по Z
            for dz in range(1, m.shape[0]):
                for z_try in (z0 + dz, z0 - dz):
                    if 0 <= z_try < m.shape[0] and m[z_try].any():
                        k, cc, n = _seed(m[z_try], cl[1:], rad[1:])
                        if k > 0:
                            z0 = z_try
                            break
                if k > 0:
                    break
        if k == 0:
            return out
        out[z0] = (cc == k)

        for step in (1, -1):
            z = z0 + step
            while 0 <= z < m.shape[0]:
                prev = out[z - step]
                if not prev.any():
                    break
                cc, n = _cc_label(m[z])
                if n == 0:
                    break
                ids = np.unique(cc[prev & (cc > 0)])
                ids = ids[ids > 0]
                if ids.size == 0:
                    break
                out[z] = np.isin(cc, ids)
                z += step
        return out

    def _place(labels_arr, lab, bb, sl, grown, out, area):
        """Полиэдр + добор формы, ОГРАНИЧЕННЫЙ ничейной территорией.

        Возвращает (метка, новый_ли, причина): 'ok' / 'merged' / 'occupied'.

        Почему рост ограничен. labels_arr приходит из polyhedron_to_label и
        по построению непересекающийся, а запись идёт только в out_reg == 0.
        Поэтому out_reg[poly] > 0 достижимо ЕДИНСТВЕННЫМ путём: ранее
        записанный инстанс разлился своей grown-маской на чужой полиэдр.
        В прежней версии такой разлив приводил к срабатыванию критерия
        A_inter/A_min > nms_thresh, и чужой инстанс объявлялся дубликатом —
        то есть счётчики merged/occupied считали не схлопывания, а ПОТЕРИ.
        Отсюда же оптимум подбора на верхнем краю nms_thresh_range: подбор
        загонял порог вверх, чтобы отключить этот merge.

        NMS уже решил, кому принадлежит территория. Рефайн имеет право
        занимать только то, что НЕ принадлежит другому полиэдру: свой полиэдр
        и фон. Merge оставлен как страховка и в норме не срабатывает.
        """
        reg = tuple(slice(min(bb[a].start, sl[a].start),
                          max(bb[a].stop, sl[a].stop)) for a in range(nd))
        lab_reg = labels_arr[reg]
        poly = (lab_reg == lab)
        out_reg = out[reg]  # view: запись идёт в out
        n_poly = int(poly.sum())
        if n_poly == 0:
            return 0, False, 'occupied'

        sub = out_reg[poly]
        sub = sub[sub > 0]
        if sub.size:
            vals, cnts = np.unique(sub, return_counts=True)
            a_min = np.minimum(n_poly, np.array([area.get(int(v), n_poly)
                                                 for v in vals], np.int64))
            iom = cnts / np.maximum(a_min, 1)
            k = int(np.argmax(iom))
            if iom[k] > nms_thresh:
                return int(vals[k]), False, 'merged'

        # Рост допускается на СВОЙ полиэдр и на фон полиэдральной карты.
        own = poly | (lab_reg == 0)
        g = np.zeros_like(poly)
        loc = tuple(slice(sl[a].start - reg[a].start, sl[a].stop - reg[a].start)
                    for a in range(nd))
        g[loc] = grown
        keep = poly | (g & own)

        w = keep & (out_reg == 0)
        if not w.any():
            w = poly & (out_reg == 0)
        if not w.any():
            return 0, False, 'occupied'
        out_reg[w] = lab
        area[lab] = int(w.sum())
        return lab, True, 'ok'

    def _bbox(objs, lab):
        bb = objs[lab - 1] if 0 < lab <= len(objs) else None
        if bb is None:
            raise RuntimeError(f"метка {lab} отсутствует в find_objects")
        return bb

    r_nuc = float(getattr(config, 'median_max_dist_nucleus', 25.0))
    r_cyto = float(getattr(config, 'median_max_dist_cytoplasm', 60.0))

    nuc_objs = [] if nuc_labels is None else find_objects(nuc_labels)
    cyto_objs = [] if cyto_labels is None else find_objects(cyto_labels)

    nuc_anchor = _anchors(nuc_labels, nuc_coords, nuc_probs, nuc_dists, nuc_objs)
    cyto_anchor = _anchors(cyto_labels, cyto_coords, cyto_probs, cyto_dists,
                           cyto_objs)

    n_runs, n_merge_n, n_merge_c, cyto_seen = 0, 0, 0, {}
    lost = dict(occ_n=0, occ_c=0, occ_c_free=0, seen=0, no_overlap=0)
    seen_labels = {}  # lab_c -> список ядер, которые её уже забрали
    # Отложенная запись цитоплазмы. Прежде клетки писались в порядке убывания
    # prob ЯДРА: слабая клетка, прицепленная к сильному ядру, занимала фон
    # раньше сильной соседки. Порядок обязан быть по СВОЕЙ уверенности.
    # Маска хранится плоскими индексами (int32), а не булевым томом: полный
    # bool на окно 16x448x448 это 3.2 МБ, на сотни клеток — гигабайты.
    cyto_pending = {}  # lab_c -> (prob_c, sl, flat_idx, shape)
    pairs = {}  # lab_c -> [записанные метки ядер]

    # ---------- ЯДРА: запись ядра + ОПРЕДЕЛЕНИЕ пары ----------
    for lab_n in sorted(nuc_anchor, key=lambda L: -nuc_anchor[L][2]):
        ctr, rays, _ = nuc_anchor[lab_n]

        sl, cl, mn, mc = _run(ctr, r_nuc, rays, head='nucleus')
        n_runs += 1

        this_n, new_n, why_n = _place(nuc_labels, lab_n, _bbox(nuc_objs, lab_n),
                                      sl, _grow(mn > mask_thresh, cl),
                                      nuc_out, area_n)
        if why_n == 'merged':
            n_merge_n += 1
        elif why_n == 'occupied':
            lost['occ_n'] += 1

        if cyto_labels is None:
            continue

        # Метка цито ПОД ЦЕНТРОМ ядра — как в спецификации задачи. Полиэдр
        # ядра это грубая звёздчатая оболочка, у стыка клеток он отдаёт
        # большинство вокселей соседу, и max-overlap уводит пару.
        lab_c = int(cyto_labels[tuple(int(v) for v in ctr)])
        if lab_c == 0:
            _m = (nuc_labels[sl] == lab_n)
            vals, cnts = np.unique(cyto_labels[sl][_m], return_counts=True)
            ok = vals > 0
            lab_c = int(vals[ok][np.argmax(cnts[ok])]) if ok.any() else 0
        if lab_c == 0:
            lost['no_overlap'] += 1
            continue

        if lab_c in cyto_pending:
            lost['seen'] += 1
            seen_labels.setdefault(lab_c, []).append(lab_n)
        else:
            g = _grow(mc > mask_thresh, cl)
            cyto_pending[lab_c] = (
                float(cyto_anchor.get(lab_c, (None, None, 0.0))[2]),
                sl, np.flatnonzero(g).astype(np.int32), tuple(g.shape))
        if this_n:
            pairs.setdefault(lab_c, []).append(this_n)

    # ---------- КЛЕТКИ: запись в порядке СВОЕЙ уверенности ----------
    # Клетки без ядра прогоняются здесь же, с промптом своей головы.
    _order = sorted(
        set(cyto_pending) | set(cyto_anchor),
        key=lambda L: -(cyto_pending[L][0] if L in cyto_pending
                        else float(cyto_anchor[L][2])))

    for lab_c in _order:
        if lab_c in cyto_pending:
            _p, sl, flat, shp = cyto_pending[lab_c]
            grown = np.zeros(shp, bool)
            grown.flat[flat] = True
            _free = False
        else:
            ctr, rays, _ = cyto_anchor[lab_c]
            sl, cl, _mn, mc = _run(ctr, r_cyto, rays, head='cytoplasm')
            n_runs += 1
            grown = _grow(mc > mask_thresh, cl)
            _free = True

        got, new_c, why_c = _place(cyto_labels, lab_c,
                                   _bbox(cyto_objs, lab_c), sl,
                                   grown, cyto_out, area_c)
        if why_c == 'merged':
            n_merge_c += 1
        elif why_c == 'occupied':
            lost['occ_c_free' if _free else 'occ_c'] += 1
        if got:
            cyto_seen[lab_c] = got
            for _tn in pairs.get(lab_c, ()):
                assignments[_tn] = got


    # ---------- КЛЕТКИ БЕЗ ЯДЕР ----------
    for lab_c in sorted(cyto_anchor, key=lambda L: -cyto_anchor[L][2]):
        if lab_c in cyto_seen:
            continue
        ctr, rays, _ = cyto_anchor[lab_c]
        sl, cl, _mn, mc = _run(ctr, r_cyto, rays, head='cytoplasm')
        n_runs += 1
        got, new_c, why_c = _place(cyto_labels, lab_c, _bbox(cyto_objs, lab_c),
                                   sl, _grow(mc > mask_thresh, cl),
                                   cyto_out, area_c)
        if why_c == 'merged':
            n_merge_c += 1
        elif why_c == 'occupied':
            lost['occ_c_free'] += 1
        if got:
            cyto_seen[lab_c] = got

    if verbose:
        _np_ = 0 if nuc_labels is None else int(len(np.unique(nuc_labels)) - 1)
        _cp_ = 0 if cyto_labels is None else int(len(np.unique(cyto_labels)) - 1)
        _wn = int(len(np.unique(nuc_out)) - 1)
        _wc = int(len(np.unique(cyto_out)) - 1)
        _occ_c = lost['occ_c'] + lost['occ_c_free']
        print(f"  [prompted] полиэдров nuc={_np_} cyto={_cp_} -> "
              f"ЗАПИСАНО nuc={_wn} cyto={_wc} | "
              f"схлопнуто как тот же объект {n_merge_n}/{n_merge_c} "
              f"(критерий NMS, nms_thresh={nms_thresh:.2f}) | "
              f"прогонов {n_runs}, пар ядро-клетка: {len(assignments)}")
        print(f"  [prompted] ПОТЕРИ nuc: территория занята {lost['occ_n']} | "
              f"баланс {_np_} - {n_merge_n} - {lost['occ_n']} = "
              f"{_np_ - n_merge_n - lost['occ_n']} (записано {_wn})")
        print(f"  [prompted] ПОТЕРИ cyto: территория занята {_occ_c} "
              f"({lost['occ_c']} через ядро + {lost['occ_c_free']} без ядра) | "
              f"метку уже забрало другое ядро {lost['seen']} | "
              f"ядро без клетки {lost['no_overlap']} | "
              f"баланс {_cp_} - {n_merge_c} - {_occ_c} = "
              f"{_cp_ - n_merge_c - _occ_c} (записано {_wc})")
        if seen_labels:
            _cnt = sorted((len(v) + 1 for v in seen_labels.values()), reverse=True)
            print(f"  [prompted] меток цито, забранных >1 ядром: "
                  f"{len(seen_labels)}, ядер на них: {_cnt[:8]}")
    return nuc_out, cyto_out, assignments

def create_overlay_with_iou(image, nucleus_pred, cytoplasm_pred,
                            nucleus_gt, cytoplasm_gt,
                            iou_thresh=0.5, z_slice=None):
    """
    Create overlay with IoU-based coloring:
    - Green: True Positive (IoU > threshold)
    - Red: False Positive (predicted but no match)
    - Blue: False Negative (GT but not predicted)

    NEW: Shows each instance with unique color from random colormap
    """

    from stardist import random_label_cmap
    import cv2
    import matplotlib.pyplot as plt


    # Handle 3D
    if nucleus_pred.ndim == 3:
        if z_slice is None:
            z_slice = nucleus_pred.shape[0] // 2
        nucleus_pred = nucleus_pred[z_slice]
        cytoplasm_pred = cytoplasm_pred[z_slice]
        nucleus_gt = nucleus_gt[z_slice]
        cytoplasm_gt = cytoplasm_gt[z_slice]
        if image.ndim >= 3:
            image = image[z_slice]  # (Z,Y,X)->(Y,X) или (Z,Y,X,C)->(Y,X,C)

    # Convert to grayscale
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        img_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image

    # Normalize
    img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8) * 255).astype(np.uint8)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Random colormap для instance масок
    cmap_instance = random_label_cmap()

    # === ROW 1: PREDICTIONS ===

    # 1. Nucleus prediction
    axes[0, 0].imshow(img_gray, cmap='gray')
    if nucleus_pred.max() > 0:
        axes[0, 0].imshow(nucleus_pred, alpha=0.6, cmap=cmap_instance,
                          interpolation='nearest', vmin=0, vmax=nucleus_pred.max())
    axes[0, 0].set_title(f'Nucleus Pred ({nucleus_pred.max()} objects on patch)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Cytoplasm prediction
    axes[0, 1].imshow(img_gray, cmap='gray')
    if cytoplasm_pred.max() > 0:
        axes[0, 1].imshow(cytoplasm_pred, alpha=0.6, cmap=cmap_instance,
                          interpolation='nearest', vmin=0, vmax=cytoplasm_pred.max())
    axes[0, 1].set_title(f'Cytoplasm Pred ({cytoplasm_pred.max()} objects on patch)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Combined prediction
    axes[0, 2].imshow(img_gray, cmap='gray')
    combined_pred = np.zeros_like(cytoplasm_pred)
    combined_pred[cytoplasm_pred > 0] = cytoplasm_pred[cytoplasm_pred > 0]
    # Offset nucleus IDs to avoid color collision
    combined_pred[nucleus_pred > 0] = nucleus_pred[nucleus_pred > 0] + cytoplasm_pred.max()
    if combined_pred.max() > 0:
        axes[0, 2].imshow(combined_pred, alpha=0.6, cmap=cmap_instance,
                          interpolation='nearest', vmin=0, vmax=combined_pred.max())
    axes[0, 2].set_title('Combined Pred', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # === ROW 2: GROUND TRUTH ===

    # 4. Nucleus GT
    axes[1, 0].imshow(img_gray, cmap='gray')
    if nucleus_gt.max() > 0:
        axes[1, 0].imshow(nucleus_gt, alpha=0.6, cmap=cmap_instance,
                          interpolation='nearest', vmin=0, vmax=nucleus_gt.max())
    axes[1, 0].set_title(f'Nucleus GT ({nucleus_gt.max()} objects)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # 5. Cytoplasm GT
    axes[1, 1].imshow(img_gray, cmap='gray')
    if cytoplasm_gt.max() > 0:
        axes[1, 1].imshow(cytoplasm_gt, alpha=0.6, cmap=cmap_instance,
                          interpolation='nearest', vmin=0, vmax=cytoplasm_gt.max())
    axes[1, 1].set_title(f'Cytoplasm GT ({cytoplasm_gt.max()} objects)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # 6. Combined GT
    axes[1, 2].imshow(img_gray, cmap='gray')
    combined_gt = np.zeros_like(cytoplasm_gt)
    combined_gt[cytoplasm_gt > 0] = cytoplasm_gt[cytoplasm_gt > 0]
    combined_gt[nucleus_gt > 0] = nucleus_gt[nucleus_gt > 0] + cytoplasm_gt.max()
    if combined_gt.max() > 0:
        axes[1, 2].imshow(combined_gt, alpha=0.6, cmap=cmap_instance,
                          interpolation='nearest', vmin=0, vmax=combined_gt.max())
    axes[1, 2].set_title('Combined GT', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # ИСПРАВЛЕНО: используем buffer_rgba() вместо tostring_rgb()
    fig.canvas.draw()

    # Получаем RGBA buffer
    buf = fig.canvas.buffer_rgba()
    width, height = fig.canvas.get_width_height()

    # Конвертируем в numpy array
    overlay = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))

    # Убираем alpha channel (оставляем RGB)
    overlay = overlay[:, :, :3]

    plt.close(fig)

    return overlay




def create_iou_comparison_overlay(image, nucleus_pred, cytoplasm_pred,
                                  nucleus_gt, cytoplasm_gt,
                                  iou_thresh=0.5, z_slice=None):
    """
    Create IoU-based comparison with color-coded matches.
    Each instance gets unique color, with boundary color indicating match quality.
    """
    from skimage.segmentation import find_boundaries
    from stardist import random_label_cmap
    import cv2
    from skimage.measure import regionprops

    # Handle 3D
    if nucleus_pred.ndim == 3:
        if z_slice is None:
            z_slice = image.shape[0] // 2
        image = image[z_slice]
        nucleus_pred = nucleus_pred[z_slice]
        cytoplasm_pred = cytoplasm_pred[z_slice]
        nucleus_gt = nucleus_gt[z_slice]
        cytoplasm_gt = cytoplasm_gt[z_slice]

    # Convert to grayscale
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        img_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image

    # Normalize
    img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8) * 255).astype(np.uint8)
    overlay = np.stack([img_gray, img_gray, img_gray], axis=-1)

    # Generate random colors for each instance
    cmap = random_label_cmap()

    # Helper to get color from colormap
    def get_instance_color(instance_id, max_id):
        """Get RGB color for instance from colormap"""
        if max_id == 0:
            return [128, 128, 128]
        normalized = instance_id / max(max_id, 1)
        rgb = cmap(normalized)[:3]
        return [int(c * 255) for c in rgb]

    # Compute IoU matches for nucleus
    nuc_pred_props = {p.label: p for p in regionprops(nucleus_pred)}
    nuc_gt_props = {p.label: p for p in regionprops(nucleus_gt)}

    nuc_matched_pred = {}  # pred_id -> (gt_id, iou)
    nuc_matched_gt = set()

    # Find matches
    for gt_id, gt_prop in nuc_gt_props.items():
        gt_mask = (nucleus_gt == gt_id)
        best_iou = 0
        best_pred_id = None

        for pred_id, pred_prop in nuc_pred_props.items():
            pred_mask = (nucleus_pred == pred_id)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_iou >= iou_thresh:
            nuc_matched_gt.add(gt_id)
            nuc_matched_pred[best_pred_id] = (gt_id, best_iou)

    # Draw nucleus instances with colored fills
    for pred_id in nuc_pred_props.keys():
        pred_mask = (nucleus_pred == pred_id)
        color = get_instance_color(pred_id, nucleus_pred.max())

        # Fill region with semi-transparent color
        for c in range(3):
            overlay[:, :, c] = np.where(pred_mask,
                                        overlay[:, :, c] * 0.5 + color[c] * 0.5,
                                        overlay[:, :, c])

        # Draw boundary with match-based color
        boundary = find_boundaries(pred_mask, mode='thick')
        if pred_id in nuc_matched_pred:
            iou = nuc_matched_pred[pred_id][1]
            # Green for good match, yellow for ok match
            if iou >= 0.8:
                overlay[boundary] = [0, 255, 0]  # Green = excellent match
            elif iou >= iou_thresh:
                overlay[boundary] = [255, 255, 0]  # Yellow = ok match
        else:
            overlay[boundary] = [255, 0, 0]  # Red = FP

    # Draw GT nucleus that were not matched (FN)
    for gt_id in nuc_gt_props.keys():
        if gt_id not in nuc_matched_gt:
            boundary = find_boundaries(nucleus_gt == gt_id, mode='inner')
            overlay[boundary] = [0, 0, 255]  # Blue = FN

    # Same for cytoplasm
    cyto_pred_props = {p.label: p for p in regionprops(cytoplasm_pred)}
    cyto_gt_props = {p.label: p for p in regionprops(cytoplasm_gt)}

    cyto_matched_pred = {}
    cyto_matched_gt = set()

    for gt_id, gt_prop in cyto_gt_props.items():
        gt_mask = (cytoplasm_gt == gt_id)
        best_iou = 0
        best_pred_id = None

        for pred_id, pred_prop in cyto_pred_props.items():
            pred_mask = (cytoplasm_pred == pred_id)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_iou >= iou_thresh:
            cyto_matched_gt.add(gt_id)
            cyto_matched_pred[best_pred_id] = (gt_id, best_iou)

    # Draw cytoplasm boundaries (thinner, dimmer)
    for pred_id in cyto_pred_props.keys():
        boundary = find_boundaries(cytoplasm_pred == pred_id, mode='inner')
        if pred_id in cyto_matched_pred:
            iou = cyto_matched_pred[pred_id][1]
            if iou >= 0.8:
                overlay[boundary] = [0, 180, 0]  # Dark green
            elif iou >= iou_thresh:
                overlay[boundary] = [180, 180, 0]  # Dark yellow
        else:
            overlay[boundary] = [180, 0, 0]  # Dark red = FP

    for gt_id in cyto_gt_props.keys():
        if gt_id not in cyto_matched_gt:
            boundary = find_boundaries(cytoplasm_gt == gt_id, mode='inner')
            overlay[boundary] = [0, 0, 180]  # Dark blue = FN

    return overlay
class CytoNucsPredictor:
    # Padding of the detection input up to the network multiple. 'reflect' is the
    # CSBDeep/StarDist default; 'constant' (zeros) is the A/B alternative exposed
    # by audit/refinement_stage_probe.py --pad-mode. Not a training setting.
    input_pad_mode = 'reflect'

    """
    Inference for CytoNucs StarDist model.

    Performs:
    1. Predict probability and distance maps for nucleus and cytoplasm
    2. Non-maximum suppression to find object centers
    3. Convert star-convex polygons/polyhedra to instance labels
    4. Post-process to ensure nuclei are inside cells
    """

    def __init__(self, model, config,
                 prob_thresh=None,
                 nms_thresh=None,
                 prob_thresh_nucleus=None,
                 prob_thresh_cytoplasm=None):
        """
        Args:
            model: trained CytoNucsStarDistModel
            config: CytoNucsConfig
            prob_thresh: default probability threshold (if specific not provided)
            nms_thresh: NMS threshold
            prob_thresh_nucleus: nucleus-specific threshold (optional)
            prob_thresh_cytoplasm: cytoplasm-specific threshold (optional)
        """
        self.model = model
        self.config = config

        self.ndim = config.ndim
        self.grid = tuple(getattr(config, 'grid', (1,) * self.ndim))[-self.ndim:]
        self.n_rays = config.n_rays
        self.rays_nucleus = getattr(config, 'rays_nucleus', getattr(config, 'rays', None))
        self.rays_cytoplasm = getattr(config, 'rays_cytoplasm', getattr(config, 'rays', None))
        self.rays = self.rays_nucleus  # обратная совместимость
        # ============================================================
        # === ПОРОГИ из аргументов, config, или defaults ===
        # ============================================================

        # Default thresholds
        default_prob = prob_thresh if prob_thresh is not None else 0.5
        default_nms = nms_thresh if nms_thresh is not None else 0.4

        # Nucleus threshold
        if prob_thresh_nucleus is not None:
            self.prob_thresh_nucleus = prob_thresh_nucleus
        elif hasattr(config, 'prob_thresh_nucleus'):
            self.prob_thresh_nucleus = config.prob_thresh_nucleus
        else:
            self.prob_thresh_nucleus = default_prob

        # Cytoplasm threshold
        if prob_thresh_cytoplasm is not None:
            self.prob_thresh_cytoplasm = prob_thresh_cytoplasm
        elif hasattr(config, 'prob_thresh_cytoplasm'):
            self.prob_thresh_cytoplasm = config.prob_thresh_cytoplasm
        else:
            self.prob_thresh_cytoplasm = default_prob

        # NMS threshold
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        elif hasattr(config, 'nms_thresh'):
            self.nms_thresh = config.nms_thresh
        else:
            self.nms_thresh = default_nms

        # mask_quality_min='auto': per-head gates calibrated by full validation
        # and stored with the checkpoint. Calibration passes (full validation,
        # threshold search) set calibrating=True and see ungated candidates.
        self.quality_gates = (getattr(model, 'loaded_quality_gates', None)
                              if config.mask_quality_min == 'auto' else None)
        self.mask_thresholds = (getattr(model, 'loaded_mask_thresholds', None)
                                if 'auto' in (config.mask_thresh_nucleus, config.mask_thresh_cytoplasm) else None)
        self.calibrating = False

    def _active_mask_thresholds(self):
        """Calibrated mask thresholds; calibration passes use the default mask_thresh."""
        if getattr(self, 'calibrating', False):
            return None
        thresholds = getattr(self, 'mask_thresholds', None)
        if (thresholds is None and 'auto' in (self.config.mask_thresh_nucleus, self.config.mask_thresh_cytoplasm)
                and not getattr(self, '_mask_threshold_warning', False)):
            self._mask_threshold_warning = True
            print('  ⚠️  mask_thresh_*=auto: checkpoint has no calibrated mask thresholds; '
                  f'using mask_thresh={self.config.mask_thresh}')
        return thresholds

    def _active_quality_gates(self):
        """Gates for this prediction; warn once when 'auto' has nothing to apply."""
        if getattr(self, 'calibrating', False):
            return None
        gates = getattr(self, 'quality_gates', None)
        if (gates is None and self.config.mask_quality_min == 'auto'
                and not getattr(self, '_quality_gate_warning', False)):
            self._quality_gate_warning = True
            print('  ⚠️  mask_quality_min=auto: checkpoint has no calibrated quality gates; '
                  'masks are not gated')
        return gates

    def _get_detection_model(self):
        """
        Detection-only submodel для первого StarDist прохода.

        На первом проходе нужны только:
          - nucleus_prob
          - nucleus_dist
          - cytoplasm_prob
          - cytoplasm_dist

        Refinement heads здесь не считаются, потому что с нулевым prompt
        их результат не используется для финальных instances.

        Настоящий refinement выполняется позже через
        predict_instances_prompted() с реальным Gaussian prompt.
        """
        if getattr(self.model, '_detection_submodel', None) is not None:
            return self.model._detection_submodel
        if not hasattr(self, '_detection_model') or self._detection_model is None:
            self._detection_model = tf.keras.Model(
                inputs=self.model.keras_model.inputs,
                outputs={
                    'nucleus_prob':
                        self.model.keras_model.get_layer('nucleus_prob').output,

                    'nucleus_dist':
                        self.model.keras_model.get_layer('nucleus_dist').output,

                    'cytoplasm_prob':
                        self.model.keras_model.get_layer('cytoplasm_prob').output,

                    'cytoplasm_dist':
                        self.model.keras_model.get_layer('cytoplasm_dist').output,
                },
                name='cytonucs_detection_inference'
            )
            # Shared by later predictors so the cached graphs stay valid.
            self.model._detection_submodel = self._detection_model

        return self._detection_model

    def _predict_with_tiling(self, img, n_tiles, normalize_img):
        """
        StarDist/CSBDeep tiled inference для CytoNucs. Работает для 2D и 3D.

        Оси делятся на два класса:

          * СВОБОДНАЯ ось: сшивается штатным csbdeep tile_iterator с overlap
            >= context, тайлы шире обучающего патча (сеть свёрточная).

          * ФИКСИРОВАННАЯ ось: окно всегда содержит ровно train_patch[a]
            отсчётов с padding до кратности сети. Окна перекрываются;
            выход каждого окна записывается только в его центральную область.

        Какие оси фиксированные — задаёт tiling_fixed_axes (см. ниже). Для 3D
        по умолчанию это Z: препарат 2D-культуры плоский, обучающий патч по Z
        покрывает его целиком, и сеть откалибрована именно на стек в train_Z
        срезов вместе с его краевым паддингом; подать другое число срезов —
        сменить режим целиком. Для 2D фиксированных осей нет, остаётся чистый
        канонический StarDist tiling.

        Отличия от прежней fixed-Z версии:
          * нет `raise ... "рассчитана на ndim=3"` -> 2D работает;
          * нет требования grid_Z=1: границы окон выравниваются по block_sizes,
            поэтому grid по фиксированной оси может быть любым (иначе смена
            анизотропии на новых данных ломала бы инференс уже после обучения);
          * произвольный набор фиксированных осей, а не жёстко ось 0.

        Первый проход считает только detection heads (prob/dist обеих голов);
        zero-prompt refinement не нужен, настоящий идёт в
        predict_instances_prompted() с реальным Gaussian prompt.
        """
        from cytonucs_runtime import normalize_percentiles as normalize
        from csbdeep.internals.predict import tile_iterator
        from math import gcd
        from itertools import product
        import numpy as np
        import tensorflow as tf
        import gc

        self.model.last_refinement_diagnostics = {}
        _n_img = int(self.config.n_channel_in)
        nd = int(self.ndim)

        # ============================================================
        # INPUT
        # ============================================================

        if img.ndim == nd:
            img = img[..., np.newaxis]
        elif img.ndim != nd + 1:
            raise ValueError(
                f"_predict_with_tiling: ожидалось ndim={nd} "
                f"или ndim+channels, получено shape={img.shape}")

        if normalize_img:
            img = normalize(img.astype(np.float32), 1, 99.8,
                            axis=tuple(range(nd)))
        else:
            img = img.astype(np.float32, copy=False)

        # Detection pass идёт с НУЛЕВЫМИ промптами. Их K пар, потому что модель
        # строит вход на K = train_batch_size промптов; на prob/dist они не
        # влияют — энкодер получает только img-каналы.
        _n_pr = max(1, int(getattr(self.config, 'prompts_per_crop', self.config.train_batch_size)))
        _n_pch = 2 * _n_pr
        if img.shape[-1] == _n_img:
            img = np.concatenate(
                [img, np.zeros(img.shape[:nd] + (_n_pch,), np.float32)], axis=-1)
        elif img.shape[-1] != _n_img + _n_pch:
            raise ValueError(
                f"_predict_with_tiling: channels={img.shape[-1]}, "
                f"ожидалось {_n_img} или {_n_img + _n_pch}")

        img_shape = tuple(int(v) for v in img.shape[:nd])
        self._last_img_shape = img_shape

        # ============================================================
        # HEAD GRIDS
        # ============================================================

        def _read_grid(name):
            g = getattr(self.config, name, None)
            if g is None:
                g = getattr(self.config, 'grid', (1,) * nd)
            g = tuple(int(v) for v in g)
            if len(g) != nd or any(v < 1 for v in g):
                raise ValueError(f"Некорректный {name}: {g}")
            return g

        nuc_grid = _read_grid('nucleus_grid')
        cyto_grid = _read_grid('cytoplasm_grid')

        # ============================================================
        # ARCHITECTURE
        # ============================================================

        plan = architecture_plan(self.config)
        block_sizes = plan['input_multiple']
        rf_detection = detection_receptive_field(plan)
        context = tuple(int(np.ceil((r - 1) / 2)) for r in rf_detection)

        _tp = tuple(int(v) for v in self.config.train_patch_size)[-nd:]

        # Какие оси фиксированные. По умолчанию для 3D это ось 0 (Z), для 2D —
        # ни одной. Переопределяется ключом tiling_fixed_axes в inference-секции
        # конфига: [] даёт канонический tiling по всем осям (изотропные 3D),
        # [0] — текущее поведение.
        _fx_cfg = getattr(self.config, 'tiling_fixed_axes', None)
        if _fx_cfg is None:
            _fx_cfg = [0] if nd == 3 else []
        _fx_cfg = sorted({int(a) for a in _fx_cfg})
        if any(a < 0 or a >= nd for a in _fx_cfg):
            raise ValueError(f"tiling_fixed_axes={_fx_cfg} вне диапазона [0,{nd})")
        fixed = tuple(a in _fx_cfg for a in range(nd))

        for a in _fx_cfg:
            if _tp[a] % block_sizes[a] != 0:
                raise ValueError(
                    f"Фиксированная ось {a}: train_patch={_tp[a]} не кратен "
                    f"block_size={block_sizes[a]}")

        # ============================================================
        # n_tiles
        # ============================================================

        if n_tiles is None:
            n_tiles = (1,) * nd
        else:
            n_tiles = tuple(int(v) for v in n_tiles)
            if len(n_tiles) == nd + 1:
                if n_tiles[-1] != 1:
                    raise ValueError("Канальная ось не может тайлиться")
                n_tiles = n_tiles[:nd]
            if len(n_tiles) != nd or any(v < 1 for v in n_tiles):
                raise ValueError(
                    f"n_tiles должен иметь {nd} положительных значений, "
                    f"получено {n_tiles}")

        # ============================================================
        # ФИКСИРОВАННЫЕ ОСИ: непересекающиеся окна по train_patch
        # ============================================================

        def _fdown(v, d):
            return (int(v) // int(d)) * int(d)

        def _fup(v, d):
            return -((-int(v)) // int(d)) * int(d)

        def _fixed_plan(a):
            grid = int(np.lcm(nuc_grid[a], cyto_grid[a]))
            return fixed_window_plan(img_shape[a], _tp[a], block_sizes[a], grid)

        fixed_axes = [a for a in range(nd) if fixed[a]]
        free_axes = [a for a in range(nd) if not fixed[a]]
        fixed_plans = {a: _fixed_plan(a) for a in fixed_axes}

        # ============================================================
        # СВОБОДНЫЕ ОСИ: паддинг до кратности + overlap >= context
        # ============================================================

        pad_free = {a: (-img_shape[a]) % block_sizes[a] for a in free_axes}
        padded = tuple(img_shape[a] + pad_free.get(a, 0) for a in range(nd))

        n_block_overlaps = tuple(
            0 if fixed[a] else int(np.ceil(context[a] / block_sizes[a]))
            for a in range(nd))

        n_tiles_eff = tuple(
            1 if fixed[a]
            else int(np.clip(n_tiles[a], 1, max(1, padded[a] // block_sizes[a])))
            for a in range(nd))

        print(f"  Detection RF={rf_detection}, required context={context}, "
              f"block_sizes={block_sizes}")
        print(f"  Оси: фиксированные {fixed_axes} -> окно "
              f"train_patch, перекрытие с центральной записью; свободные {free_axes} -> tile_iterator, "
              f"overlap="
              f"{tuple(n_block_overlaps[a] * block_sizes[a] for a in free_axes)} px")
        for a in fixed_axes:
            print(f"    ось {a}: n={img_shape[a]}, train={_tp[a]}, "
                  f"окон={len(fixed_plans[a])}")
            for i, (s0, sz, d0, d1) in enumerate(fixed_plans[a]):
                print(f"      #{i}: src={s0}:{s0 + sz} -> dst={d0}:{d1}")

        # ============================================================
        # ВЫХОДНЫЕ КАРТЫ
        # ============================================================
        #
        # По фиксированным осям адресация глобальная -> размер ceil(n/grid).
        # По свободным осям tile_iterator работает поверх ПАДДЕННОГО массива,
        # его s_dst покрывает [0, padded) -> держим padded/grid и режем в конце.
        # ============================================================

        def _map_shape(grid_):
            return tuple(
                int(np.ceil(img_shape[a] / grid_[a])) if fixed[a]
                else padded[a] // grid_[a]
                for a in range(nd))

        nuc_shape_tiled = _map_shape(nuc_grid)
        cyto_shape_tiled = _map_shape(cyto_grid)

        _dist_dtype = np.dtype(getattr(self.config, 'dist_map_dtype', 'float32'))

        nucleus_prob_full = np.empty(nuc_shape_tiled, np.float32)
        nucleus_dist_full = np.empty(nuc_shape_tiled + (self.n_rays,), _dist_dtype)
        cytoplasm_prob_full = np.empty(cyto_shape_tiled, np.float32)
        cytoplasm_dist_full = np.empty(cyto_shape_tiled + (self.n_rays,), _dist_dtype)

        cov_nuc = np.zeros(nuc_shape_tiled, np.uint8)
        cov_cyto = np.zeros(cyto_shape_tiled, np.uint8)

        # ============================================================
        # OUTPUT COPY
        # ============================================================

        def _put(dst, val, grid_, s_src, s_dst, fx, coverage=None):
            """fx: {axis: (src_lo, dst_lo, dst_hi)} для фиксированных осей.
            Для свободных осей берутся слайсы tile_iterator."""
            ss, sd = [], []
            for a in range(nd):
                g = int(grid_[a])
                if fixed[a]:
                    src_lo, dst_lo, dst_hi = fx[a]
                    if src_lo % g or dst_lo % g:
                        raise RuntimeError(
                            f"ось {a}: окно {src_lo}/{dst_lo} не кратно grid={g}")
                    g_lo = dst_lo // g
                    g_hi = -((-dst_hi) // g)          # ceil: последний блок
                    off = src_lo // g
                    sd.append(slice(g_lo, g_hi))
                    ss.append(slice(g_lo - off, g_hi - off))
                else:
                    a_src, a_dst = s_src[a], s_dst[a]
                    if (a_src.start % g or a_src.stop % g
                            or a_dst.start % g or a_dst.stop % g):
                        raise RuntimeError(
                            f"ось {a}: срезы тайла не кратны grid={g}: "
                            f"src={a_src}, dst={a_dst}")
                    ss.append(slice(a_src.start // g, a_src.stop // g))
                    sd.append(slice(a_dst.start // g, a_dst.stop // g))
            ss, sd = tuple(ss), tuple(sd)
            dst[sd] = val[ss]
            if coverage is not None:
                coverage[sd] += 1

        # ============================================================
        # DETECTION MODEL
        # ============================================================

        detection_model = self._get_detection_model()
        tiles_done = 0

        # ГРАФОВОЕ исполнение обязательно. Eager-вызов функциональной модели
        # держит выходы ВСЕХ слоёв живыми до конца прохода: пик = сумма по
        # графу, а не максимум. На тайле 16x896x896 один только
        # nuc_features_prob это [1,16,448,448,128] = 1.64 ГБ, и таких тензоров
        # в графе несколько -> OOM там, где graph mode проходит.
        # tf.function с КОНКРЕТНОЙ формой (все тайлы одинаковы по построению
        # tile_iterator) даёт планировщику переиспользовать буферы; кэш по
        # форме нужен только на случай отличающегося последнего окна.
        # Cached on the model: every full validation builds a new predictor, and
        # the graph reads live weights, so retracing per validation is wasted.
        if not hasattr(self.model, '_detection_forward_cache'):
            from collections import OrderedDict
            self.model._detection_forward_cache = OrderedDict()
        _fwd_cache = self._detection_forward_cache = self.model._detection_forward_cache

        def _forward(x):
            key = tuple(int(v) for v in x.shape)
            fn = _fwd_cache.get(key)
            if fn is None:
                fn = tf.function(
                    lambda t: detection_model(t, training=False),
                    input_signature=[tf.TensorSpec(key, tf.float32)],
                    reduce_retracing=True)
                _fwd_cache[key] = fn
                # Bound retained concrete graphs for heterogeneous volume sizes.
                if len(_fwd_cache) > 8:
                    _fwd_cache.popitem(last=False)
                print(f"    [graph] трассировка на форму {key}")
            _fwd_cache.move_to_end(key)
            return fn(tf.convert_to_tensor(x, tf.float32))

        def _flip_averaged(y_pred, tile):
            """inference.tta_flips: probability maps averaged over the tile flipped in Y, in X and in both
            (the distance maps stay those of the unflipped tile: flipping does not map the rays onto rays)."""
            maps = {k: np.array(y_pred[k], np.float32) for k in ('nucleus_prob', 'cytoplasm_prob')}
            for axes in ((nd - 2,), (nd - 1,), (nd - 2, nd - 1)):
                flipped = _forward(np.ascontiguousarray(np.flip(tile, axis=axes))[np.newaxis, ...])
                for k in maps:
                    maps[k] += np.flip(np.asarray(flipped[k], np.float32), axis=tuple(a + 1 for a in axes))
            out = dict(y_pred)
            out.update({k: v / 4. for k, v in maps.items()})
            return out

        _bs_full = tuple(block_sizes) + (1,)
        _nt_full = tuple(n_tiles_eff) + (1,)
        _ov_full = tuple(n_block_overlaps) + (0,)

        try:
            # Перебор окон по фиксированным осям (для 2D список пуст ->
            # ровно одна итерация со всем изображением).
            for combo in product(*[fixed_plans[a] for a in fixed_axes]):
                fx = {a: (c[0], c[2], c[3]) for a, c in zip(fixed_axes, combo)}
                win = {a: (c[0], c[1]) for a, c in zip(fixed_axes, combo)}

                sub = img[tuple(
                    slice(win[a][0], min(win[a][0] + win[a][1], img_shape[a]))
                    if fixed[a] else slice(None) for a in range(nd))]

                pad_width = []
                for a in range(nd):
                    if fixed[a]:
                        pad_width.append((0, win[a][1] - sub.shape[a]))
                    else:
                        pad_width.append((0, pad_free[a]))
                pad_width.append((0, 0))

                if any(p1 for _, p1 in pad_width):
                    sub = _pad_input(sub, pad_width, getattr(self, 'input_pad_mode', CytoNucsPredictor.input_pad_mode))

                for a in fixed_axes:
                    if sub.shape[a] != win[a][1]:
                        raise RuntimeError(
                            f"ось {a}: перед forward ожидалось {win[a][1]}, "
                            f"получено {sub.shape[a]}")

                for tile, s_src, s_dst in tile_iterator(
                        sub, n_tiles=_nt_full, block_sizes=_bs_full,
                        n_block_overlaps=_ov_full):

                    y_pred = _forward(tile[np.newaxis, ...])
                    if getattr(self.config, 'tta_flips', False):
                        y_pred = _flip_averaged(y_pred, tile)

                    def _arr(key):
                        a = np.asarray(y_pred[key])[0]
                        return a[..., 0] if (a.ndim == nd + 1
                                             and a.shape[-1] == 1) else a

                    _put(nucleus_prob_full, _arr('nucleus_prob'), nuc_grid,
                         s_src, s_dst, fx, coverage=cov_nuc)
                    _put(nucleus_dist_full, _arr('nucleus_dist'), nuc_grid,
                         s_src, s_dst, fx)
                    _put(cytoplasm_prob_full, _arr('cytoplasm_prob'), cyto_grid,
                         s_src, s_dst, fx, coverage=cov_cyto)
                    _put(cytoplasm_dist_full, _arr('cytoplasm_dist'), cyto_grid,
                         s_src, s_dst, fx)

                    tiles_done += 1
                    del y_pred

        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
            raise RuntimeError(
                f"OOM при StarDist tiling n_tiles={n_tiles}. "
                f"Auto-splitting отключён. Увеличьте n_tiles по свободным осям "
                f"{free_axes} явно. Ошибка: {type(e).__name__}: {e}") from e

        print(f"  ✅ StarDist tiling complete: {tiles_done} model forwards")

        # ============================================================
        # COVERAGE CHECK
        # ============================================================

        for _tag, _cov in (('nucleus', cov_nuc), ('cytoplasm', cov_cyto)):
            _mn_cov, _mx_cov = int(_cov.min()), int(_cov.max())
            print(f"  coverage[{_tag}]: min={_mn_cov}, max={_mx_cov}")
            if _mn_cov != 1 or _mx_cov != 1:
                raise RuntimeError(
                    f"Tiling coverage нарушен для {_tag}: min={_mn_cov}, "
                    f"max={_mx_cov}, missing={int((_cov == 0).sum())}, "
                    f"double={int((_cov > 1).sum())}")

        del cov_nuc, cov_cyto

        # ============================================================
        # CROP PADDING СВОБОДНЫХ ОСЕЙ
        # ============================================================

        def _crop(arr, grid_, rays=False):
            sl = tuple(slice(0, int(np.ceil(img_shape[a] / grid_[a])))
                       for a in range(nd))
            return arr[sl + (slice(None),)] if rays else arr[sl]

        nucleus_prob_full = _crop(nucleus_prob_full, nuc_grid)
        nucleus_dist_full = _crop(nucleus_dist_full, nuc_grid, rays=True)
        cytoplasm_prob_full = _crop(cytoplasm_prob_full, cyto_grid)
        cytoplasm_dist_full = _crop(cytoplasm_dist_full, cyto_grid, rays=True)

        print(f"  nucleus_dist: [{nucleus_dist_full.min():.1f}, "
              f"{nucleus_dist_full.max():.1f}] px")
        print(f"  cytoplasm_dist: [{cytoplasm_dist_full.min():.1f}, "
              f"{cytoplasm_dist_full.max():.1f}] px")

        _thr = float(self.prob_thresh_cytoplasm)
        if nd == 3:
            _z_frac = [float((cytoplasm_prob_full[z] > _thr).mean())
                       for z in range(cytoplasm_prob_full.shape[0])]
            print(f"  [cyto prob > {_thr:.2f} by Z] "
                  + ", ".join(f"{z}:{v:.3f}" for z, v in enumerate(_z_frac)))
        else:
            print(f"  [cyto prob > {_thr:.2f}] "
                  f"{float((cytoplasm_prob_full > _thr).mean()):.3f}")

        # ============================================================
        # CANONICAL NMS
        # ============================================================

        print(f"  [nucleus] Proposal mode: {self.config.proposal_mode}")
        self._current_type = 'nucleus'
        nucleus_instances = self._stardist_postprocess(
            nucleus_prob_full, nucleus_dist_full,
            prob_thresh=self.prob_thresh_nucleus,
            nms_thresh=self.nms_thresh)
        self._last_coords_nuc = self._last_coords
        self._last_probs_nuc = self._last_probs
        self._last_dists_nuc = self._last_dists
        residual = dict(nucleus=self._last_residual)

        print(f"  [cytoplasm] Proposal mode: {self.config.proposal_mode}")
        self._current_type = 'cytoplasm'
        cytoplasm_instances = self._stardist_postprocess(
            cytoplasm_prob_full, cytoplasm_dist_full,
            prob_thresh=self.prob_thresh_cytoplasm,
            nms_thresh=self.nms_thresh)
        self._last_coords_cyto = self._last_coords
        self._last_probs_cyto = self._last_probs
        self._last_dists_cyto = self._last_dists
        residual['cytoplasm'] = self._last_residual

        # ============================================================
        # PROMPTED REFINEMENT
        # ============================================================

        mask_thr = getattr(self.config, 'mask_thresh', 0.5)

        _nc = getattr(self, '_last_coords_nuc', None)
        _np_ = getattr(self, '_last_probs_nuc', None)
        _cc = getattr(self, '_last_coords_cyto', None)
        _cp = getattr(self, '_last_probs_cyto', None)

        self._prompt_coords_nuc = _nc
        self._prompt_coords_cyto = _cc

        _nl = np.asarray(nucleus_instances).astype(np.int32)
        _cl = np.asarray(cytoplasm_instances).astype(np.int32)

        if (getattr(self, '_refinement_stage_audit', None) is not None or (_nc is not None and len(_nc) > 0)
                or (_cc is not None and len(_cc) > 0)):
            (nucleus_instances,
             cytoplasm_instances,
             self._prompt_assignments) = predict_instances_prompted(
                self.model, self.config, img,
                _nc, _np_, _cc, _cp,
                mask_thresh=mask_thr,
                iou_dedup=getattr(self.config, 'mask_iou_dedup', 0.5),
                nuc_dists=getattr(self, '_last_dists_nuc', None),
                cyto_dists=getattr(self, '_last_dists_cyto', None),
                nuc_labels=_nl, cyto_labels=_cl,
                nms_thresh=self.nms_thresh, stage_audit=getattr(self, '_refinement_stage_audit', None),
                quality_gates=CytoNucsPredictor._active_quality_gates(self),
                mask_thresholds=CytoNucsPredictor._active_mask_thresholds(self),
                candidate_sink=getattr(self, '_refinement_candidate_sink', None),
                residual=residual)
        else:
            print("  ⚠️ Ни ядер, ни клеток -> prompted refinement не запускался")
            self._prompt_assignments = None

        # ============================================================
        # DETAILS
        # ============================================================

        details = {
            'nucleus_prob': nucleus_prob_full,
            'cytoplasm_prob': cytoplasm_prob_full,
            'nucleus_dist': nucleus_dist_full,
            'cytoplasm_dist': cytoplasm_dist_full,
            # Zero-prompt refinement намеренно не считается: с нулевым промптом
            # его результат в инстансы не идёт.
            'nucleus_mask': None,
            'cytoplasm_mask': None,
            'rays_nucleus': self.rays_nucleus,
            'rays_cytoplasm': self.rays_cytoplasm,
            'nucleus_grid': nuc_grid,
            'cytoplasm_grid': cyto_grid,
        }

        gc.collect()

        return nucleus_instances, cytoplasm_instances, details



    def predict_instances(self, img, normalize_img=True):
        """
        Одиночный forward для тома/кадра, целиком влезающего в сеть.

        Вход выравнивается по кратности ТАК ЖЕ, как это делает tiling. Без
        этого UNet глубины depth падает на Concatenate, когда сторона не делится
        на base_grid * pool**depth: прежняя версия молча полагалась на «удачный»
        размер (2048 кратно 32) и ломалась на любом другом препарате, а в 2D —
        почти всегда.

        Прежний `except ResourceExhaustedError` с обрезкой до train_patch убран:
        он возвращал сегментацию одного угла как результат по всему кадру. Если
        не влезает — это тайлинг, а не тихая обрезка.
        """
        from cytonucs_runtime import normalize_percentiles as normalize
        from math import gcd

        self.model.last_refinement_diagnostics = {}
        nd = int(self.ndim)
        _n_img = int(self.config.n_channel_in)

        if img.ndim == nd:
            img = img[..., np.newaxis]
        elif img.ndim != nd + 1:
            raise ValueError(f"predict_instances: ожидалось ndim={nd} "
                             f"или ndim+channels, получено {img.shape}")

        if normalize_img:
            img = normalize(img.astype(np.float32), 1, 99.8,
                            axis=tuple(range(nd)))
        else:
            img = img.astype(np.float32, copy=False)

        _n_pr = max(1, int(getattr(self.config, 'prompts_per_crop', self.config.train_batch_size)))
        _n_pch = 2 * _n_pr
        if img.shape[-1] == _n_img:
            img = np.concatenate(
                [img, np.zeros(img.shape[:nd] + (_n_pch,), np.float32)], axis=-1)
        elif img.shape[-1] != _n_img + _n_pch:
            raise ValueError(f"predict_instances: channels={img.shape[-1]}, "
                             f"ожидалось {_n_img} или {_n_img + _n_pch}")

        img_shape = tuple(int(s) for s in img.shape[:nd])
        self._last_img_shape = img_shape

        def _rd(nm):
            g = getattr(self.config, nm, None)
            if g is None:
                g = getattr(self.config, 'grid', (1,) * nd)
            g = tuple(int(v) for v in g)
            if len(g) != nd or any(v < 1 for v in g):
                raise ValueError(f"Некорректный {nm}: {g}")
            return g

        nuc_grid, cyto_grid = _rd('nucleus_grid'), _rd('cytoplasm_grid')

        block = architecture_plan(self.config)['input_multiple']

        pad = [(0, (-img_shape[a]) % block[a]) for a in range(nd)] + [(0, 0)]
        img_net = img
        if any(p1 for _, p1 in pad):
            img_net = _pad_input(img, pad, getattr(self, 'input_pad_mode', CytoNucsPredictor.input_pad_mode))

        y_pred = self._get_detection_model()(
            tf.convert_to_tensor(img_net[np.newaxis, ...], tf.float32),
            training=False)

        def _get(key, grid_):
            """Срез до РЕАЛЬНОГО размера входа в grid-разрешении головы."""
            if key not in y_pred:
                return None
            a = np.asarray(y_pred[key])[0]
            if a.ndim == nd + 1 and a.shape[-1] == 1:
                a = a[..., 0]
            sl = tuple(slice(0, int(np.ceil(img_shape[i] / grid_[i])))
                       for i in range(nd))
            return a[sl]

        nucleus_prob = _get('nucleus_prob', nuc_grid)
        nucleus_dist = _get('nucleus_dist', nuc_grid)
        cytoplasm_prob = _get('cytoplasm_prob', cyto_grid)
        cytoplasm_dist = _get('cytoplasm_dist', cyto_grid)
        nuc_mask_pred = _get('nucleus_mask', nuc_grid)
        cyto_mask_pred = _get('cytoplasm_mask', cyto_grid)
        del y_pred

        self._current_type = 'nucleus'
        nucleus_instances = self._stardist_postprocess(
            nucleus_prob, nucleus_dist,
            self.prob_thresh_nucleus, self.nms_thresh)
        nuc_coords, nuc_probs, nuc_dists = (
            self._last_coords, self._last_probs, self._last_dists)
        residual = dict(nucleus=self._last_residual)

        self._current_type = 'cytoplasm'
        cytoplasm_instances = self._stardist_postprocess(
            cytoplasm_prob, cytoplasm_dist,
            self.prob_thresh_cytoplasm, self.nms_thresh)
        cyto_coords, cyto_probs, cyto_dists = (
            self._last_coords, self._last_probs, self._last_dists)
        residual['cytoplasm'] = self._last_residual

        mask_thr = getattr(self.config, 'mask_thresh', 0.5)
        self._prompt_coords_nuc = nuc_coords
        self._prompt_coords_cyto = cyto_coords
        _nl = np.asarray(nucleus_instances).astype(np.int32)
        _cl = np.asarray(cytoplasm_instances).astype(np.int32)

        if (getattr(self, '_refinement_stage_audit', None) is not None or (nuc_coords is not None and len(nuc_coords) > 0)
                or (cyto_coords is not None and len(cyto_coords) > 0)):
            (nucleus_instances, cytoplasm_instances,
             self._prompt_assignments) = predict_instances_prompted(
                self.model, self.config, img,
                nuc_coords, nuc_probs, cyto_coords, cyto_probs,
                mask_thresh=mask_thr,
                iou_dedup=getattr(self.config, 'mask_iou_dedup', 0.5),
                nuc_dists=nuc_dists, cyto_dists=cyto_dists,
                nuc_labels=_nl, cyto_labels=_cl,
                nms_thresh=self.nms_thresh, stage_audit=getattr(self, '_refinement_stage_audit', None),
                quality_gates=CytoNucsPredictor._active_quality_gates(self),
                mask_thresholds=CytoNucsPredictor._active_mask_thresholds(self),
                candidate_sink=getattr(self, '_refinement_candidate_sink', None),
                residual=residual)
        else:
            print("  ⚠️  Ни ядер, ни клеток -> промпт-ветка НЕ ЗАПУСКАЛАСЬ. "
                  "Возвращены сырые полиэдры StarDist.")
            self._prompt_assignments = None

        details = {
            'nucleus_prob': nucleus_prob,
            'nucleus_dist': nucleus_dist,
            'cytoplasm_prob': cytoplasm_prob,
            'cytoplasm_dist': cytoplasm_dist,
            'nucleus_mask': nuc_mask_pred,
            'cytoplasm_mask': cyto_mask_pred,
            'rays_nucleus': self.rays_nucleus,
            'rays_cytoplasm': self.rays_cytoplasm,
            'nucleus_grid': nuc_grid,
            'cytoplasm_grid': cyto_grid,
        }
        return nucleus_instances, cytoplasm_instances, details

    def _stardist_postprocess(
            self,
            prob,
            dist,
            prob_thresh,
            nms_thresh
    ):
        """
        Fast StarDist post-processing для больших 3D cytoplasm объектов.

        Pipeline:
            prob/dist
              -> threshold
              -> spatial candidate preselection
              -> ОДИН StarDist sparse NMS
              -> polyhedron_to_label

        В отличие от прежней реализации:
            - нет iterative rounds;
            - нет covered mask;
            - нет повторных NMS;
            - нет повторного render внутри NMS loop.

        Candidate preselection используется только для вычислительной
        осуществимости: dense StarDist NMS для сотен тысяч крупных 3D
        polyhedra практически непригоден.

        GRID/RAYS и размер кандидирующего окна выбираются отдельно
        для nucleus/cytoplasm.
        """
        import numpy as np

        obj_type = getattr(
            self,
            '_current_type',
            'cytoplasm'
        )

        # ============================================================
        # GRID / RAYS
        # ============================================================

        grid_name = (
            'nucleus_grid'
            if obj_type == 'nucleus'
            else 'cytoplasm_grid'
        )

        grid = tuple(
            int(g)
            for g in getattr(
                self.config,
                grid_name,
                (1,) * self.ndim
            )
        )

        if (
                len(grid) != self.ndim
                or any(g < 1 for g in grid)
        ):
            grid = (1,) * self.ndim

        rays = (
            self.rays_nucleus
            if obj_type == 'nucleus'
            else self.rays_cytoplasm
        )

        if rays is None:
            raise RuntimeError(
                f"[{obj_type}] rays=None"
            )

        # ============================================================
        # INPUT
        # ============================================================

        prob = np.asarray(
            prob,
            dtype=np.float32
        )

        if (
                prob.ndim == self.ndim + 1
                and prob.shape[-1] == 1
        ):
            prob = prob[..., 0]

        dist = np.asarray(dist)

        if prob.ndim != self.ndim:
            raise ValueError(
                f"[{obj_type}] invalid prob shape={prob.shape}"
            )

        if (
                dist.ndim != self.ndim + 1
                or dist.shape[:self.ndim] != prob.shape
        ):
            raise ValueError(
                f"[{obj_type}] "
                f"prob={prob.shape}, dist={dist.shape}"
            )

        if dist.shape[-1] != len(rays):
            raise ValueError(
                f"[{obj_type}] "
                f"dist rays={dist.shape[-1]}, "
                f"rays={len(rays)}"
            )


        # ============================================================
        # FULL RESOLUTION SHAPE
        # ============================================================

        full_shape = tuple(
            int(s * g)
            for s, g in zip(
                prob.shape,
                grid
            )
        )

        _img_shape = getattr(
            self,
            '_last_img_shape',
            None
        )

        if (
                _img_shape is not None
                and len(_img_shape) == self.ndim
        ):
            full_shape = tuple(
                min(int(a), int(b))
                for a, b in zip(
                    full_shape,
                    _img_shape
                )
            )

        self._last_residual = None

        def _empty():
            self._last_coords = None
            self._last_probs = None
            self._last_dists = None

            return np.zeros(
                full_shape,
                dtype=np.uint16
            )

        # ============================================================
        # THRESHOLD + BORDER
        # ============================================================

        keep = (
                prob > float(prob_thresh)
        )

        from cytonucs_geometry import proposal_peak_settings, expected_object_extent
        # Same window/border rule as the trainer's detector-prompt sampler.
        win, b_xy = proposal_peak_settings(self.config, obj_type, prob.shape)

        if b_xy > 0:

            border_keep = np.zeros_like(
                keep,
                dtype=bool
            )

            if self.ndim == 3:
                border_keep[
                :,
                b_xy:prob.shape[1] - b_xy,
                b_xy:prob.shape[2] - b_xy
                ] = True

            else:
                border_keep[
                b_xy:prob.shape[0] - b_xy,
                b_xy:prob.shape[1] - b_xy
                ] = True

            keep &= border_keep

        n_raw = int(
            keep.sum()
        )

        proposal_audits = getattr(self, '_refinement_stage_audit', None)
        proposal_observer = proposal_audits.get(obj_type) if proposal_audits is not None else None
        if n_raw == 0 and proposal_observer is None:
            print(
                f"  [{obj_type}] No objects detected "
                f"(prob_thresh={prob_thresh:.2f})"
            )

            return _empty()

        # Typical object bbox from measured geometry; printed with the peak window.
        ext = expected_object_extent(self.config, obj_type, full_shape)

        # ============================================================
        # CANDIDATE PRESELECTION — ПЕРЕКРЫВАЮЩИЕСЯ ЛОКАЛЬНЫЕ МАКСИМУМЫ
        # ============================================================
        #
        # Прежняя версия резала карту на НЕПЕРЕСЕКАЮЩИЕСЯ блоки step и брала по
        # одному argmax на блок. Два дефекта:
        #
        #   1. Жёсткий лимит «один кандидат на блок». ext выводится из ОДНОЙ
        #      глобальной median_max_dist_*, а обучение идёт на двух доменах
        #      (плоские EA.hy926 bbox ~[5,153,154] и сфероиды HeLa). Медиана по
        #      объединению задаёт окно под крупные клетки, и в мелком домене два
        #      соседних центра штатно попадают в один блок — один не доживает
        #      до sparse NMS. Каждая такая потеря это потерянная пара JTPR.
        #   2. Блоки привязаны к координатной сетке: настоящий локальный
        #      максимум у границы блока проигрывает соседу, который локальным
        #      максимумом не является.
        #
        # maximum_filter даёт локальные максимумы в СКОЛЬЗЯЩЕМ окне: лимита на
        # блок нет, привязки к сетке нет, стоимость та же (фильтр сепарабельный).
        # Дедупликацию по-настоящему делает sparse NMS ниже — у него на каждый
        # кандидат свои лучи, в отличие от единого окна здесь. Поэтому окно
        # теперь ЧИСТО вычислительный тормоз, и его можно смело уменьшать
        # (nms_candidate_spacing 0.10-0.15) — качество от этого только растёт.
        # ============================================================

        from scipy.ndimage import maximum_filter

        pp = np.where(keep, prob, -np.inf)
        loc_max = keep & (pp >= maximum_filter(
            pp, size=tuple(win), mode='constant', cval=-np.inf))

        def plateau_peaks(maxima):
            # Flat maxima should produce one proposal per connected plateau.
            from scipy.ndimage import label, find_objects
            plateau, _ = label(maxima)
            unique_peaks = []
            for lab, sl in enumerate(find_objects(plateau), 1):
                if sl is None:
                    continue
                pts = np.argwhere(plateau[sl] == lab)
                pts += np.array([s.start for s in sl])
                values = prob[tuple(pts.T)]
                best = pts[values == values.max()]
                unique_peaks.append(best[np.argmin(np.sum((best-best.mean(0))**2,axis=1))])
            return np.asarray(unique_peaks, int).reshape(-1, self.ndim)

        candidates = np.argwhere(loc_max)
        if self.config.proposal_mode == 'peaks' and len(candidates):
            candidates = plateau_peaks(loc_max)

        # Residual peaks: local maxima of the smallest window (3 per axis) that the
        # proposal window suppressed. predict_instances_prompted prompts them only where
        # the first assembly left the voxel unlabelled, so a neighbour's stronger peak no
        # longer hides an object that nothing else covers.
        fine = [min(int(w), 3) for w in win]
        if (self.config.proposal_mode == 'peaks' and getattr(self.config, 'residual_proposals', False)
                and fine != [int(w) for w in win]):
            fine_max = keep & (pp >= maximum_filter(pp, size=tuple(fine), mode='constant', cval=-np.inf))
            residual = plateau_peaks(fine_max & ~loc_max) if np.any(fine_max & ~loc_max) else np.zeros((0, self.ndim), int)
            if len(residual):
                sel_r = tuple(residual[:, a] for a in range(self.ndim))
                r_prob = prob[sel_r].astype(np.float32)
                order_r = np.argsort(r_prob)[::-1][:int(getattr(self.config, 'nms_max_candidates', 300000))]
                self._last_residual = dict(
                    coords=(residual[order_r] * np.asarray(grid)[None, :]).astype(np.float32),
                    probs=r_prob[order_r],
                    dists=np.maximum(dist[sel_r].astype(np.float32), 1e-3)[order_r])
            print(f"  [{obj_type}] residual peaks (win={tuple(fine)}): {len(residual)}")

        def record_proposals(selected):
            if proposal_observer is not None:
                residual = self._last_residual
                proposal_observer.record_proposals(prob, grid, b_xy, win, selected, float(prob_thresh),
                    getattr(self.config, f'rays_anisotropy_{obj_type}', self.config.anisotropy),
                    residual=None if residual is None else residual['coords'] / np.asarray(grid)[None, :])

        if len(candidates) == 0:
            record_proposals(candidates)
            print(
                f"  [{obj_type}] "
                f"No candidates after preselection"
            )
            return _empty()

        # Потолок числа кандидатов — только ради стоимости sparse NMS.
        # Отбор по prob, а не по положению: пространственного смещения нет.
        _cap = int(getattr(self.config, 'nms_max_candidates', 300000))
        if len(candidates) > _cap:
            _v = prob[tuple(candidates[:, a] for a in range(self.ndim))]
            candidates = candidates[np.argsort(_v)[::-1][:_cap]]
            print(f"  [{obj_type}] кандидатов урезано до потолка {_cap}")

        # ============================================================
        # CANDIDATE DATA
        # ============================================================

        record_proposals(candidates)
        sel = tuple(
            candidates[:, a]
            for a in range(self.ndim)
        )

        candidate_prob = (
            prob[sel]
            .astype(np.float32)
        )

        candidate_dist = np.maximum(
            dist[sel].astype(np.float32),
            1e-3
        )

        candidate_points = (
                candidates
                * np.asarray(grid)[None, :]
        ).astype(np.float32)

        # Сортируем по prob до sparse NMS.
        order = np.argsort(
            candidate_prob
        )[::-1]

        candidate_prob = (
            candidate_prob[order]
        )

        candidate_dist = (
            candidate_dist[order]
        )

        candidate_points = (
            candidate_points[order]
        )

        print(
            f"  [{obj_type}] "
            f"{n_raw} threshold candidates "
            f"-> {len(candidate_points)} preselected "
            f"(win={tuple(win)}, "
            f"bbox≈{tuple(int(round(v)) for v in ext)} px)"
        )

        # ============================================================
        # ONE SPARSE STARDIST NMS
        # ============================================================

        if self.config.proposal_mode == 'peaks':
            # Defer geometric suppression until masks exist. Render only anchor
            # identities; these labels are never returned as final masks.
            out = np.zeros(full_shape, np.int32)
            for lab, ctr in enumerate(candidate_points.astype(int), 1):
                out[tuple(ctr)] = lab
            self._last_coords = candidate_points
            self._last_probs = candidate_prob
            self._last_dists = candidate_dist
            return out

        try:

            if self.ndim == 3:

                from stardist.nms import \
                    non_maximum_suppression_3d_sparse

                (
                    coord_full,
                    prob_nms,
                    dists,
                    _
                ) = non_maximum_suppression_3d_sparse(
                    candidate_dist,
                    candidate_prob,
                    candidate_points,
                    rays,
                    nms_thresh=float(nms_thresh),
                    verbose=False
                )

            else:

                from stardist.nms import \
                    non_maximum_suppression_sparse

                (
                    coord_full,
                    prob_nms,
                    dists,
                    _
                ) = non_maximum_suppression_sparse(
                    candidate_dist,
                    candidate_prob,
                    candidate_points,
                    nms_thresh=float(nms_thresh),
                    verbose=False
                )

        except Exception as e:

            print(
                f"  [{obj_type}] "
                f"⚠️ sparse NMS failed: {e}"
            )

            import traceback
            traceback.print_exc()

            return _empty()

        coord_full = np.asarray(
            coord_full
        )

        prob_nms = np.asarray(
            prob_nms,
            dtype=np.float32
        )

        dists = np.asarray(
            dists,
            dtype=np.float32
        )

        if len(coord_full) == 0:
            print(
                f"  [{obj_type}] "
                f"NMS retained 0"
            )

            return _empty()

        print(
            f"  [{obj_type}] "
            f"sparse NMS retained "
            f"{len(coord_full)}"
        )

        # ============================================================
        # DIST DIAGNOSTIC
        # ============================================================

        per_peak_max = (
            dists.max(axis=1)
        )

        print(
            f"  [{obj_type}] "
            f"dist at peaks: "
            f"median_max_ray="
            f"{np.median(per_peak_max):.1f} "
            f"min="
            f"{per_peak_max.min():.1f} "
            f"max="
            f"{per_peak_max.max():.1f} "
            f"median_mean_ray="
            f"{np.median(dists.mean(axis=1)):.1f}"
        )

        # ============================================================
        # RENDER ONCE
        # ============================================================

        try:

            if self.ndim == 3:

                from stardist.geometry import \
                    polyhedron_to_label

                instances = polyhedron_to_label(
                    dists,
                    coord_full,
                    rays,
                    shape=full_shape,
                    prob=prob_nms,
                    verbose=False
                )

            else:

                from stardist.geometry import \
                    polygons_to_label

                instances = polygons_to_label(
                    dists,
                    coord_full,
                    full_shape,
                    prob=prob_nms
                )

            from skimage.segmentation import \
                relabel_sequential

            instances, _, _ = (
                relabel_sequential(
                    instances
                )
            )

            print(
                f"    ✅ Rendered "
                f"{len(np.unique(instances)) - 1} "
                f"instances"
            )

            self._last_coords = (
                np.asarray(
                    coord_full
                ).astype(np.int32)
            )

            self._last_probs = (
                prob_nms
            )

            self._last_dists = (
                dists
            )

            return instances.astype(
                np.uint16
            )

        except Exception as e:

            print(
                f"  [{obj_type}] "
                f"⚠️ Rendering failed: {e}"
            )

            import traceback
            traceback.print_exc()

            return _empty()





    def _postprocess_containment(self, nucleus_instances, cytoplasm_instances):
        """
        Post-process to ensure nuclei are inside cells.

        Strategy:
        1. For each nucleus, check if it overlaps with a cell
        2. If no overlap, assign to nearest cell (if within distance threshold)
        3. If nucleus overlaps multiple cells, assign to cell with max overlap


        Args:
            nucleus_instances: nucleus instance labels
            cytoplasm_instances: cytoplasm instance labels

        Returns:
            nucleus_instances_refined: refined nucleus labels
            cytoplasm_instances_refined: refined cytoplasm labels
        """
        from cytonucs_losses import compute_nucleus_cell_assignments

        # Compute assignments
        assignments, unassigned, stats = compute_nucleus_cell_assignments(
            nucleus_instances,
            cytoplasm_instances,
            max_distance=self.config.max_assignment_distance
        )

        # For debugging/validation
        print(f"  Assigned {len(assignments)} nuclei to cells")
        if len(unassigned) > 0:
            print(f"  {len(unassigned)} nuclei без назначенной клетки (сохранены)")

        return nucleus_instances, cytoplasm_instances

    def optimize_thresholds(self, val_images, val_nucleus_masks, val_cytoplasm_masks,
                            prob_thresh_range=[0.3, 0.4, 0.5, 0.6, 0.7],
                            nms_thresh_range=[0.15, 0.2, 0.25, 0.3, 0.4],
                            mask_thresh_range=None, val_assignments=None):
        """
        Подбор порогов по JTPR — это целевая метрика задачи: она штрафует и за
        пропущенное ядро, и за пропущенную клетку, и за неверную пару. AP по
        одной голове ни того, ни другого не учитывает; compute_dice усредняет
        по совпавшим парам и за FP не штрафует вовсе.

        JTPR — ОДНО число на обе головы, поэтому пороги голов не разделяются на
        независимые задачи, как это было при подборе по AP. Вместо декартова
        произведения (которое стоило бы куба) идёт покоординатный спуск:

          1) общий prob x nms         — грубая настройка списка пиков;
          2) prob ядра при фикс. цито — уточнение одной координаты;
          3) prob цито при фикс. ядре — уточнение второй;
          4) mask_thresh              — размер масок при найденных порогах.

        Стоимость аддитивная: P*N + 2P + M прогонов вместо P^2*N*M. Каждый шаг
        не ухудшает JTPR по построению, потому что предыдущая точка входит в
        перебор и результаты кэшируются.

        nms_thresh здесь управляет ДВУМЯ шагами: sparse NMS и тестом
        тождественности в _place. Второй раньше молча брал config.nms_thresh и
        в подборе не участвовал.

        Тайлинг включается автоматически — тот же путь, что и на инференсе.
        """
        from itertools import product
        from stardist.matching import matching_dataset
        from cytonucs_inference import compute_jtpr
        import time

        _diag_saved = getattr(self.config, 'diag_max_prompts', 20)
        _mask_saved = float(getattr(self.config, 'mask_thresh', 0.5))
        _heads_saved = (self.config.mask_thresh_nucleus, self.config.mask_thresh_cytoplasm)
        # Gates are calibrated for other thresholds; they could only remove TP here.
        _calibrating_saved = getattr(self, 'calibrating', False)
        self.calibrating = True
        if self.config.proposal_mode == 'peaks':
            nms_thresh_range = [self.nms_thresh]  # geometric NMS is bypassed in this mode
        try:
            self.config.diag_max_prompts = 0
        except Exception:
            pass

        patch = tuple(int(p) for p in self.config.train_patch_size)
        taus = list(getattr(self.config, 'threshold_opt_taus',
                            [0.5, 0.6, 0.7, 0.8, 0.9]))

        def _predict(img):
            shp = img.shape[:self.ndim]
            if any(s > p for s, p in zip(shp, patch)):
                n_tiles = tuple(int(np.ceil(s / p)) for s, p in zip(shp, patch))
                return self._predict_with_tiling(img, n_tiles=n_tiles,
                                                 normalize_img=False)
            return self.predict_instances(img, normalize_img=False)

        # Кэш по ПОЛНОМУ набору порогов: покоординатный спуск переоценивает
        # уже посчитанные точки, а один прогон стоит минуты.
        _cache = {}

        def _score(tag=None):
            key = (round(float(self.prob_thresh_nucleus), 4),
                   round(float(self.prob_thresh_cytoplasm), 4),
                   round(float(self.nms_thresh), 4),
                   round(float(getattr(self.config, 'mask_thresh', 0.5)), 4),
                   self.config.mask_thresh_nucleus, self.config.mask_thresh_cytoplasm)
            if key in _cache:
                if tag:
                    print(f"  {tag} | JTPR={_cache[key][0]:.4f} (из кэша)")
                return _cache[key]

            t0 = time.time()
            tp_j, tot_j = 0, 0
            pn, pc = [], []
            _asg_gt = list(val_assignments) if val_assignments else [None] * len(val_images)
            for _k, (img, gn, gc) in enumerate(zip(val_images, val_nucleus_masks,
                                                   val_cytoplasm_masks)):
                nuc_pred, cyto_pred, _ = _predict(img)
                pn.append(nuc_pred)
                pc.append(cyto_pred)
                # None = промпт-ветка не запускалась (тогда законен фолбэк),
                # {} = запустилась и дала 0 пар (тогда JTPR обязан быть 0).
                # Прежнее `or None` эти два случая склеивало.
                _j, _d = compute_jtpr(nucleus_pred=nuc_pred,
                                      cytoplasm_pred=cyto_pred,
                                      nucleus_gt=gn, cytoplasm_gt=gc,
                                      iou_thresh=0.5,
                                      assignments_pred=getattr(
                                          self, '_prompt_assignments', None),
                                      assignments_gt=_asg_gt[_k])
                tp_j += int(_d['joint_tp'])
                tot_j += int(_d['total_pairs'])
            jtpr = tp_j / tot_j if tot_j else 0.0

            sn = matching_dataset(val_nucleus_masks, pn, thresh=taus,
                                  show_progress=False)
            sc = matching_dataset(val_cytoplasm_masks, pc, thresh=taus,
                                  show_progress=False)
            sn = list(sn) if isinstance(sn, (list, tuple)) else [sn]
            sc = list(sc) if isinstance(sc, (list, tuple)) else [sc]
            an = float(np.mean([s.accuracy for s in sn]))
            ac = float(np.mean([s.accuracy for s in sc]))

            if tag:
                print(f"  {tag} ({time.time() - t0:.0f} с) | "
                      f"JTPR={jtpr:.4f} ({tp_j}/{tot_j}) | "
                      f"nuc AP={an:.3f} @50={sn[0].accuracy:.3f} "
                      f"TP/FP/FN={sn[0].tp}/{sn[0].fp}/{sn[0].fn} | "
                      f"cyto AP={ac:.3f} @50={sc[0].accuracy:.3f} "
                      f"TP/FP/FN={sc[0].tp}/{sc[0].fp}/{sc[0].fn}")

            _cache[key] = (jtpr, an, ac)
            return _cache[key]

        def _set(pn_=None, pc_=None, nms_=None, mask_=None):
            if pn_ is not None:
                self.prob_thresh_nucleus = float(pn_)
            if pc_ is not None:
                self.prob_thresh_cytoplasm = float(pc_)
            if nms_ is not None:
                self.nms_thresh = float(nms_)
            if mask_ is not None:
                self.config.mask_thresh = float(mask_)
                for head in ('nucleus', 'cytoplasm'):
                    if getattr(self.config, 'mask_thresh_' + head) not in (None, 'auto'):
                        setattr(self.config, 'mask_thresh_' + head, float(mask_))

        # ---------- СТАДИЯ 1: общий prob x nms ----------
        print(f"[1/4] общий prob x nms — "
              f"{len(prob_thresh_range) * len(nms_thresh_range)} комбинаций "
              f"x {len(val_images)} томов (mask_thresh={_mask_saved})")
        best = (_score('initial thresholds')[0], float(self.prob_thresh_nucleus),
                float(self.prob_thresh_cytoplasm), float(self.nms_thresh))
        for p, n in product(prob_thresh_range, nms_thresh_range):
            _set(pn_=p, pc_=p, nms_=n)
            j = _score(f"prob={p:.2f} nms={n:.2f}")[0]
            if j > best[0]:
                best = (j, p, p, n)
        _set(pn_=best[1], pc_=best[2], nms_=best[3])
        print(f"  ▸ стадия 1: prob={best[1]} nms={best[3]} JTPR={best[0]:.4f}")

        # ---------- СТАДИЯ 2: prob ядра ----------
        print(f"[2/4] prob ядра при cyto={best[2]}, nms={best[3]}")
        for p in prob_thresh_range:
            _set(pn_=p)
            j = _score(f"prob_nuc={p:.2f}")[0]
            if j > best[0]:
                best = (j, p, best[2], best[3])
        _set(pn_=best[1])
        print(f"  ▸ стадия 2: prob_nuc={best[1]} JTPR={best[0]:.4f}")

        # ---------- СТАДИЯ 3: prob цитоплазмы ----------
        print(f"[3/4] prob цитоплазмы при nuc={best[1]}, nms={best[3]}")
        for p in prob_thresh_range:
            _set(pc_=p)
            j = _score(f"prob_cyto={p:.2f}")[0]
            if j > best[0]:
                best = (j, best[1], p, best[3])
        _set(pc_=best[2])
        print(f"  ▸ стадия 3: prob_cyto={best[2]} JTPR={best[0]:.4f}")

        # ---------- СТАДИЯ 4: mask_thresh ----------
        if mask_thresh_range is None:
            mask_thresh_range = list(getattr(self.config, 'mask_thresh_range',
                                             [0.25, 0.35, 0.45, 0.55]))
        mask_thresh_range = sorted(set([_mask_saved] + list(mask_thresh_range)))
        print(f"[4/4] mask_thresh — {len(mask_thresh_range)} значений "
              f"при найденных prob/nms")
        best_mask, best_mask_j = _mask_saved, _score()[0]
        best_heads = _heads_saved
        for mt in mask_thresh_range:
            _set(mask_=mt)
            j = _score(f"mask_thresh={mt:.2f}")[0]
            if j > best_mask_j:
                best_mask, best_mask_j = float(mt), j
                best_heads = (self.config.mask_thresh_nucleus, self.config.mask_thresh_cytoplasm)
        _set(mask_=best_mask)
        self.config.mask_thresh_nucleus, self.config.mask_thresh_cytoplasm = best_heads

        try:
            self.config.diag_max_prompts = _diag_saved
        except Exception:
            pass
        self.calibrating = _calibrating_saved

        _edge = []
        for nm, v, rng in (('prob_thresh_nucleus', best[1], prob_thresh_range),
                           ('prob_thresh_cytoplasm', best[2], prob_thresh_range),
                           ('nms_thresh', best[3], nms_thresh_range),
                           ('mask_thresh', best_mask, mask_thresh_range)):
            if len(rng) > 1 and (v == min(rng) or v == max(rng)):
                _edge.append(f"{nm}={v}")
        if _edge:
            print(f"  ⚠️  оптимум на КРАЮ диапазона: {', '.join(_edge)} — "
                  f"расширь список, настоящий оптимум может лежать дальше")

        best_params = {
            'prob_thresh_nucleus': float(best[1]),
            'prob_thresh_cytoplasm': float(best[2]),
            'nms_thresh': float(best[3]),
            'mask_thresh': float(best_mask),
            'mask_thresh_nucleus': best_heads[0],
            'mask_thresh_cytoplasm': best_heads[1],
            'jtpr': float(best_mask_j),
        }
        print(f"Best parameters (по JTPR): {best_params}")
        return best_params



def load_gt_assignments(mask_or_image_path, dirname='assignments'):
    """Истинный mapping {nucleus_id: cytoplasm_id} из <split>/assignments/<stem>.json.

    Тот же файл, который читает генератор (data_generator.repool). Метрика его
    до сих пор игнорировала и пересчитывала эвристикой compute_nucleus_cell_
    assignments с max_distance=100: она засчитывает принадлежность при 1%
    перекрытия и назначает ВСЕ оставшиеся ядра ближайшей клетке в изотропной
    метрике. Свободные ядра попадали в знаменатель JTPR, то есть метрика имела
    потолок ниже 1.0 при идеальной сегментации.
    """
    from pathlib import Path
    import json
    p = Path(mask_or_image_path)
    f = p.parent.parent / dirname / (p.stem + '.json')
    if not f.exists():
        return None
    with open(f, 'r') as fh:
        return {int(k): int(v) for k, v in json.load(fh).items()}
def compute_jtpr(nucleus_pred, cytoplasm_pred, nucleus_gt, cytoplasm_gt, iou_thresh=0.5,
                 assignments_pred=None, assignments_gt=None, with_pairs=False):
    """
    Compute Joint True Positive Rate (JTPR).

    A nucleus-cell pair is correct if:
    - Nucleus matches ground truth (IoU > threshold)
    - Its parent cell matches ground truth (IoU > threshold)

    Args:
        nucleus_pred: predicted nucleus instances
        cytoplasm_pred: predicted cytoplasm instances
        nucleus_gt: ground truth nucleus instances
        cytoplasm_gt: ground truth cytoplasm instances
        iou_thresh: IoU threshold for matching

    Returns:
        jtpr: joint true positive rate
        details: dict with matching statistics
    """
    from cytonucs_losses import compute_nucleus_cell_assignments


    # Get assignments
    if assignments_pred is None:
        assignments_pred, _, _ = compute_nucleus_cell_assignments(nucleus_pred, cytoplasm_pred)

    if assignments_gt is None:
        # Geometric fallback may differ from annotated biological pairs.
        assignments_gt, _, _ = compute_nucleus_cell_assignments(nucleus_gt, cytoplasm_gt)
    else:
        # Отфильтровать пары, которых нет в ЭТОМ томе: mapping считался по
        # полному изображению, а сюда мог прийти кроп или подмножество меток.
        _n_ids = set(int(v) for v in np.unique(nucleus_gt) if v > 0)
        _c_ids = set(int(v) for v in np.unique(cytoplasm_gt) if v > 0)
        assignments_gt = {int(n): int(c) for n, c in dict(assignments_gt).items()
                          if int(n) in _n_ids and int(c) in _c_ids}


    # Match nuclei
    nucleus_matches = match_instances(nucleus_pred, nucleus_gt, iou_thresh)

    # Match cells
    cell_matches = match_instances(cytoplasm_pred, cytoplasm_gt, iou_thresh)

    # Count joint true positives
    joint_ids = set()
    total_pairs_gt = len(assignments_gt)

    for nuc_gt_id, cell_gt_id in assignments_gt.items():
        # Check if nucleus is correctly detected
        nuc_pred_id = nucleus_matches.get(nuc_gt_id, None)
        if nuc_pred_id is None:
            continue

        # Check if this nucleus has an assigned cell in prediction
        if nuc_pred_id not in assignments_pred:
            continue
        cell_pred_id = assignments_pred[nuc_pred_id]

        # Check if the cell is correctly detected
        if cell_matches.get(cell_gt_id) == cell_pred_id:
            joint_ids.add(nuc_gt_id)
    joint_tp = len(joint_ids)

    jtpr = joint_tp / total_pairs_gt if total_pairs_gt > 0 else 0.0

    # Diagnostic subset: pairs not cut by the XY faces of this (patch) volume.
    def xy_face_ids(labels):
        labels = np.asarray(labels)
        ids = set()
        for axis in range(labels.ndim - 2, labels.ndim):
            for index in (0, -1):
                ids.update(int(v) for v in np.unique(np.take(labels, index, axis=axis)))
        return ids - {0}
    cut_nuclei, cut_cells = xy_face_ids(nucleus_gt), xy_face_ids(cytoplasm_gt)
    interior_pairs = [n for n, c in assignments_gt.items()
                      if n not in cut_nuclei and c not in cut_cells]

    details = {
        'joint_tp': joint_tp,
        'total_pairs': total_pairs_gt,
        'nucleus_matches': len(nucleus_matches),
        'cell_matches': len(cell_matches),
        'joint_fail_nucleus': sum(n not in nucleus_matches for n in assignments_gt),
        'joint_fail_cell': sum(n in nucleus_matches and c not in cell_matches
                               for n, c in assignments_gt.items()),
        'joint_both_matched': sum(n in nucleus_matches and c in cell_matches
                                 for n, c in assignments_gt.items()),
        'interior_total_pairs': len(interior_pairs),
        'interior_joint_tp': sum(n in joint_ids for n in interior_pairs),
    }

    details['joint_fail_link'] = details['joint_both_matched'] - joint_tp
    if with_pairs:
        # GT (nucleus, cell) ids of the joint true positives, for calibration.
        details['joint_pairs'] = sorted((int(n), int(assignments_gt[n])) for n in joint_ids)
    return jtpr, details


def match_instances(pred, gt, iou_thresh=0.5):
    """One-to-one IoU matching, identical to the instance AP evaluation."""
    from stardist.matching import matching
    stats = matching(np.asarray(gt, np.int32), np.asarray(pred, np.int32),
                     thresh=float(iou_thresh), criterion='iou', report_matches=True)
    return {int(stats.matched_pairs[i][0]): int(stats.matched_pairs[i][1])
            for i in stats.matched_tps}


def visualize_results(img, nucleus_pred, cytoplasm_pred, z_slice=None):
    """
    Visualize prediction results.

    Args:
        img: input image
        nucleus_pred: predicted nucleus instances
        cytoplasm_pred: predicted cytoplasm instances
        z_slice: which Z slice to show (for 3D)
    """
    import matplotlib.pyplot as plt
    from stardist import random_label_cmap

    ndim = img.ndim

    if ndim == 3:
        if z_slice is None:
            z_slice = img.shape[0] // 2
        img_show = img[z_slice]
        nucleus_show = nucleus_pred[z_slice]
        cytoplasm_show = cytoplasm_pred[z_slice]
    else:
        img_show = img
        nucleus_show = nucleus_pred
        cytoplasm_show = cytoplasm_pred

    cmap = random_label_cmap()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_show, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(nucleus_show, cmap=cmap)
    axes[1].set_title(f'Nucleus Instances ({nucleus_show.max()})')
    axes[1].axis('off')

    axes[2].imshow(cytoplasm_show, cmap=cmap)
    axes[2].set_title(f'Cytoplasm Instances ({cytoplasm_show.max()})')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


# ============================================================
# Extended Inference Functions (for standalone use)
# ============================================================

def create_overlay(image, nucleus_pred, cytoplasm_pred,
                   nucleus_gt=None, cytoplasm_gt=None, z_slice=None):
    """
    Create overlay visualization with boundaries.

    Args:
        image: raw image
        nucleus_pred: predicted nucleus instances
        cytoplasm_pred: predicted cytoplasm instances
        nucleus_gt: ground truth nucleus (optional)
        cytoplasm_gt: ground truth cytoplasm (optional)
        z_slice: which slice for 3D

    Returns:
        overlay: RGB image with colored boundaries
    """
    from skimage.segmentation import find_boundaries
    import cv2

    # Handle 3D
    if nucleus_pred.ndim == 3:
        if z_slice is None:
            z_slice = image.shape[0] // 2
        image = image[z_slice]
        nucleus_pred = nucleus_pred[z_slice]
        cytoplasm_pred = cytoplasm_pred[z_slice]
        if nucleus_gt is not None:
            nucleus_gt = nucleus_gt[z_slice]
        if cytoplasm_gt is not None:
            cytoplasm_gt = cytoplasm_gt[z_slice]

    # Convert to grayscale if RGB
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        img_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image

    # Normalize to 0-255
    img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8) * 255).astype(np.uint8)

    # Create RGB base
    overlay = np.stack([img_gray, img_gray, img_gray], axis=-1)

    # Draw cytoplasm boundaries (green)
    cyto_boundaries = find_boundaries(cytoplasm_pred, mode='thick')
    overlay[cyto_boundaries, 1] = 255
    overlay[cyto_boundaries, 0] = overlay[cyto_boundaries, 0] // 2
    overlay[cyto_boundaries, 2] = overlay[cyto_boundaries, 2] // 2

    # Draw nucleus boundaries (red)
    nuc_boundaries = find_boundaries(nucleus_pred, mode='thick')
    overlay[nuc_boundaries, 0] = 255
    overlay[nuc_boundaries, 1] = overlay[nuc_boundaries, 1] // 2
    overlay[nuc_boundaries, 2] = overlay[nuc_boundaries, 2] // 2

    # Draw GT if provided (different colors)
    if nucleus_gt is not None:
        gt_nuc_boundaries = find_boundaries(nucleus_gt, mode='inner')
        overlay[gt_nuc_boundaries, 2] = 255  # Blue
        overlay[gt_nuc_boundaries, 0] = overlay[gt_nuc_boundaries, 0] // 2
        overlay[gt_nuc_boundaries, 1] = overlay[gt_nuc_boundaries, 1] // 2

    if cytoplasm_gt is not None:
        gt_cyto_boundaries = find_boundaries(cytoplasm_gt, mode='inner')
        overlay[gt_cyto_boundaries, 1] = 200  # Cyan
        overlay[gt_cyto_boundaries, 2] = 200
        overlay[gt_cyto_boundaries, 0] = overlay[gt_cyto_boundaries, 0] // 2

    return overlay


def create_comprehensive_figure(image, nucleus_pred, cytoplasm_pred,
                                nucleus_gt=None, cytoplasm_gt=None,
                                metrics=None, z_slice=None):
    """
    Create comprehensive visualization with all panels and metrics.
    """
    import matplotlib.pyplot as plt
    from stardist import random_label_cmap

    # Determine layout
    has_gt = (nucleus_gt is not None and cytoplasm_gt is not None)
    n_cols = 5 if has_gt else 4  # Добавили колонку для GT

    fig = plt.figure(figsize=(6 * n_cols, 12))

    # Create gridspec for better layout control
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.3, wspace=0.2)

    # Handle 3D
    if nucleus_pred.ndim == 3:
        if z_slice is None:
            z_slice = image.shape[0] // 2
        img_show = image[z_slice]
        nuc_pred_show = nucleus_pred[z_slice]
        cyto_pred_show = cytoplasm_pred[z_slice]
        nuc_gt_show = nucleus_gt[z_slice] if nucleus_gt is not None else None
        cyto_gt_show = cytoplasm_gt[z_slice] if cytoplasm_gt is not None else None
    else:
        img_show = image
        nuc_pred_show = nucleus_pred
        cyto_pred_show = cytoplasm_pred
        nuc_gt_show = nucleus_gt
        cyto_gt_show = cytoplasm_gt

    # Use random colormap for instances
    cmap_instance = random_label_cmap()

    # === ROW 1: PREDICTIONS ===

    # Panel 1: Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_show, cmap='gray' if img_show.ndim == 2 else None)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Nucleus prediction
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_show, cmap='gray' if img_show.ndim == 2 else None)
    if nuc_pred_show.max() > 0:
        ax2.imshow(nuc_pred_show, alpha=0.6, cmap=cmap_instance,
                   interpolation='nearest', vmin=0, vmax=nuc_pred_show.max())
    ax2.set_title(f'Nucleus Pred\n({nuc_pred_show.max()} objects)',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Panel 3: Cytoplasm prediction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_show, cmap='gray' if img_show.ndim == 2 else None)
    if cyto_pred_show.max() > 0:
        ax3.imshow(cyto_pred_show, alpha=0.6, cmap=cmap_instance,
                   interpolation='nearest', vmin=0, vmax=cyto_pred_show.max())
    ax3.set_title(f'Cytoplasm Pred\n({cyto_pred_show.max()} objects)',
                  fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Panel 4: Combined prediction
    if has_gt:
        ax4 = fig.add_subplot(gs[0, 3])
    else:
        ax4 = fig.add_subplot(gs[0, 3])

    ax4.imshow(img_show, cmap='gray' if img_show.ndim == 2 else None)
    combined_pred = np.zeros_like(cyto_pred_show)
    combined_pred[cyto_pred_show > 0] = cyto_pred_show[cyto_pred_show > 0]
    combined_pred[nuc_pred_show > 0] = nuc_pred_show[nuc_pred_show > 0] + cyto_pred_show.max()

    if combined_pred.max() > 0:
        ax4.imshow(combined_pred, alpha=0.6, cmap=cmap_instance,
                   interpolation='nearest', vmin=0, vmax=combined_pred.max())
    ax4.set_title('Combined Pred', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # === ROW 2: GROUND TRUTH (if available) ===

    if has_gt:
        # Panel 5: Nucleus GT
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(img_show, cmap='gray' if img_show.ndim == 2 else None)
        if nuc_gt_show.max() > 0:
            ax5.imshow(nuc_gt_show, alpha=0.6, cmap=cmap_instance,
                       interpolation='nearest', vmin=0, vmax=nuc_gt_show.max())
        ax5.set_title(f'Nucleus GT\n({nuc_gt_show.max()} objects)',
                      fontsize=14, fontweight='bold')
        ax5.axis('off')

        # Panel 6: Cytoplasm GT
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(img_show, cmap='gray' if img_show.ndim == 2 else None)
        if cyto_gt_show.max() > 0:
            ax6.imshow(cyto_gt_show, alpha=0.6, cmap=cmap_instance,
                       interpolation='nearest', vmin=0, vmax=cyto_gt_show.max())
        ax6.set_title(f'Cytoplasm GT\n({cyto_gt_show.max()} objects)',
                      fontsize=14, fontweight='bold')
        ax6.axis('off')

        # Panel 7: Combined GT
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.imshow(img_show, cmap='gray' if img_show.ndim == 2 else None)
        combined_gt = np.zeros_like(cyto_gt_show)
        combined_gt[cyto_gt_show > 0] = cyto_gt_show[cyto_gt_show > 0]
        combined_gt[nuc_gt_show > 0] = nuc_gt_show[nuc_gt_show > 0] + cyto_gt_show.max()

        if combined_gt.max() > 0:
            ax7.imshow(combined_gt, alpha=0.6, cmap=cmap_instance,
                       interpolation='nearest', vmin=0, vmax=combined_gt.max())
        ax7.set_title('Combined GT', fontsize=14, fontweight='bold')
        ax7.axis('off')

        # Panel 8: IoU comparison overlay
        ax8 = fig.add_subplot(gs[1, 4])
        iou_overlay = create_iou_comparison_overlay(
            img_show, nuc_pred_show, cyto_pred_show,
            nuc_gt_show, cyto_gt_show
        )
        ax8.imshow(iou_overlay)
        ax8.set_title('IoU Comparison\nGreen=TP, Red=FP, Blue=FN',
                      fontsize=14, fontweight='bold')
        ax8.axis('off')

    # Add metrics text box (span bottom)
    if metrics:
        metrics_text = "Metrics:\n" + "=" * 40 + "\n"

        if 'dice_nucleus' in metrics:
            metrics_text += f"Dice Nucleus:    {metrics['dice_nucleus']:.4f}\n"
        if 'dice_cytoplasm' in metrics:
            metrics_text += f"Dice Cytoplasm:  {metrics['dice_cytoplasm']:.4f}\n"
        if 'jtpr' in metrics:
            metrics_text += f"JTPR:            {metrics['jtpr']:.4f}\n"
        if 'containment_penalty' in metrics:
            metrics_text += f"Containment:     {metrics['containment_penalty']:.4f}\n"
        if 'wbr_ratio' in metrics:
            metrics_text += f"WBR Ratio:       {metrics['wbr_ratio']:.4f}\n"
        if 'n_multi_nuclear' in metrics:
            metrics_text += f"Multi-nuclear:   {metrics['n_multi_nuclear']} cells\n"

        fig.text(0.02, 0.02, metrics_text, fontsize=11, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                 verticalalignment='bottom')

    return fig


def compute_all_metrics(img, nucleus_pred, cytoplasm_pred,
                        nucleus_gt=None, cytoplasm_gt=None, details=None,
                        assignments_pred=None, assignments_gt=None):
    """
    Compute comprehensive metrics including Dice, JTPR, containment, WBR.

    Returns:
        dict with all metrics
    """
    from collections import Counter
    from cytonucs_losses import compute_nucleus_cell_assignments
    from cytonucs_containment_helper import compute_containment_penalty, monitor_wbr_metric
    from cytonucs_trainer import compute_dice

    metrics = {
        'n_nuclei': int(np.count_nonzero(np.unique(nucleus_pred))),
        'n_cells': int(np.count_nonzero(np.unique(cytoplasm_pred)))
    }

    # Compute assignments
    assignments_heur, unassigned, stats = compute_nucleus_cell_assignments(
        nucleus_pred, cytoplasm_pred
    )
    if assignments_pred is None:
        assignments_pred = assignments_heur

    metrics['n_assigned'] = len(assignments_pred)
    metrics['n_unassigned'] = len(unassigned)

    # Multi-nuclear cells
    cell_counts = Counter(assignments_pred.values())
    multi_nuclear = {k: v for k, v in cell_counts.items() if v > 1}
    metrics['n_multi_nuclear'] = len(multi_nuclear)
    if len(multi_nuclear) > 0:
        metrics['max_nuclei_per_cell'] = max(multi_nuclear.values())

    # WBR metric
    if details and 'nucleus_prob' in details and 'cytoplasm_prob' in details:
        # Контейнер — бинарные инстансы клеток, а не cytoplasm_prob: последний
        # это edt_prob, и порог по нему относит бо́льшую часть клетки к «снаружи».
        wbr_ratio = monitor_wbr_metric(
            details['nucleus_prob'],
            details['cytoplasm_prob'],
            cytoplasm_instances=cytoplasm_pred
        )
        metrics['wbr_ratio'] = float(wbr_ratio)

    # Containment penalty
    if details and 'cytoplasm_dist' in details:
        containment_penalty, _ = compute_containment_penalty(
            nucleus_pred,
            cytoplasm_pred,
            details['cytoplasm_dist'],
            assignments_pred,
            containment_margin=2.0,
            rays=details.get('rays_cytoplasm'),
            grid=details.get('cytoplasm_grid')
        )
        metrics['containment_penalty'] = float(containment_penalty)

    # Ground truth metrics
    if nucleus_gt is not None and cytoplasm_gt is not None:
        metrics['dice_nucleus'] = float(compute_dice(nucleus_gt, nucleus_pred))
        metrics['dice_cytoplasm'] = float(compute_dice(cytoplasm_gt, cytoplasm_pred))
        metrics['dice_mean'] = (metrics['dice_nucleus'] + metrics['dice_cytoplasm']) / 2

        # JTPR
        jtpr, jtpr_details = compute_jtpr(
            nucleus_pred, cytoplasm_pred,
            nucleus_gt, cytoplasm_gt,
            assignments_pred=assignments_pred, assignments_gt=assignments_gt
        )
        metrics['jtpr'] = float(jtpr)
        metrics['jtpr_details'] = jtpr_details

        # GT multi-nuclear stats
        if assignments_gt is None:
            assignments_gt, _, _ = compute_nucleus_cell_assignments(nucleus_gt, cytoplasm_gt)
        gt_cell_counts = Counter(assignments_gt.values())
        gt_multi_nuclear = {k: v for k, v in gt_cell_counts.items() if v > 1}
        metrics['gt_n_multi_nuclear'] = len(gt_multi_nuclear)

    return metrics


def load_model_from_config(config_path):
    """
    Load trained model from config file for standalone inference.

    Args:
        config_path: path to config JSON

    Returns:
        model: loaded model
        config: CytoNucsConfig
        prob_thresh: probability threshold
        nms_thresh: NMS threshold
    """
    import json
    from pathlib import Path
    from cytonucs_config import CytoNucsConfig
    from cytonucs_model import build_cytonucs_model

    config_dict = json.load(open(config_path, 'r'))
    model_config = CytoNucsConfig.from_json(
        config_path, checkpoint_path=config_dict.get('inference', {}).get('checkpoint_path'))

    # Get checkpoint path
    if 'inference' in config_dict and 'checkpoint_path' in config_dict['inference']:
        checkpoint_path = Path(config_dict['inference']['checkpoint_path'])
    else:
        raise ValueError("Config must have 'inference.checkpoint_path' field")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Build model
    model = build_cytonucs_model(
        config=model_config,
        name=checkpoint_path.parent.name,
        basedir=str(checkpoint_path.parent.parent)
    )

    # Load weights
    model.load_weights(checkpoint_path)
    print("✓ Model loaded")

    # Get thresholds
    inf_cfg = config_dict.get('inference', {})
    prob_thresh_nucleus = inf_cfg.get('prob_thresh_nucleus', inf_cfg.get('prob_thresh', 0.5))
    prob_thresh_cytoplasm = inf_cfg.get('prob_thresh_cytoplasm', inf_cfg.get('prob_thresh', 0.5))
    nms_thresh = inf_cfg.get('nms_thresh', 0.4)

    return model, model_config, prob_thresh_nucleus, prob_thresh_cytoplasm, nms_thresh


def process_single_image(predictor, image_path, output_dir,
                         gt_nucleus_dir=None, gt_cytoplasm_dir=None,
                         save_all=True):
    """
    Process single image with full pipeline: predict, compute metrics, visualize, save.

    Args:
        predictor: CytoNucsPredictor instance
        image_path: path to input image
        output_dir: directory to save results
        gt_nucleus_dir: GT nucleus directory (optional)
        gt_cytoplasm_dir: GT cytoplasm directory (optional)
        save_all: whether to save all outputs

    Returns:
        metrics: dict with results
    """
    import imageio
    from pathlib import Path
    import json

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {image_path.name}")

    # Load image
    from cytonucs_runtime import read_model_image
    img = read_model_image(image_path, predictor.ndim, predictor.config.n_channel_in)
    print(f"  Shape: {img.shape}")

    # Load GT if available
    nucleus_gt = None
    cytoplasm_gt = None

    if gt_nucleus_dir:
        gt_nuc_path = Path(gt_nucleus_dir) / image_path.name
        if gt_nuc_path.exists():
            nucleus_gt = imageio.volread(gt_nuc_path)
            print(f"  Loaded GT nucleus")

    if gt_cytoplasm_dir:
        gt_cyto_path = Path(gt_cytoplasm_dir) / image_path.name
        if gt_cyto_path.exists():
            cytoplasm_gt = imageio.volread(gt_cyto_path)
            print(f"  Loaded GT cytoplasm")

    # Run inference. Тайлинг — тот же путь, что в run_inference_only: без него
    # том больше патча либо не влезает в память, либо молча обрезается до
    # патча внутри except ResourceExhaustedError.
    print("  Predicting...")
    _patch = tuple(int(p) for p in predictor.config.train_patch_size)
    _shp = img.shape[:predictor.ndim]
    if any(s > p for s, p in zip(_shp, _patch)):
        _nt = tuple(int(np.ceil(s / p)) for s, p in zip(_shp, _patch))
        print(f"  🔲 Using tiling: {_nt}")
        nucleus_pred, cytoplasm_pred, details = predictor._predict_with_tiling(
            img, n_tiles=_nt, normalize_img=True)
    else:
        nucleus_pred, cytoplasm_pred, details = predictor.predict_instances(
            img, normalize_img=True)

    print(f"  Detected: {nucleus_pred.max()} nuclei, {cytoplasm_pred.max()} cells")

    # Compute metrics
    print("  Computing metrics...")
    metrics = compute_all_metrics(
        img, nucleus_pred, cytoplasm_pred,
        nucleus_gt, cytoplasm_gt, details,
        assignments_pred=getattr(predictor, '_prompt_assignments', None),
        assignments_gt=load_gt_assignments(gt_nuc_path) if nucleus_gt is not None else None
    )

    if save_all:
        base_name = image_path.stem

        # Save predictions
        if nucleus_pred.ndim == 2:
            write_instance_mask(output_dir / f'{base_name}_nucleus.tif', nucleus_pred)
            write_instance_mask(output_dir / f'{base_name}_cytoplasm.tif', cytoplasm_pred)
            imageio.imwrite(output_dir / f'{base_name}_nucleus_prob.tif',
                            (details['nucleus_prob'] * 65535).astype(np.uint16))
            imageio.imwrite(output_dir / f'{base_name}_cytoplasm_prob.tif',
                            (details['cytoplasm_prob'] * 65535).astype(np.uint16))
        else:
            write_instance_mask(output_dir / f'{base_name}_nucleus.tif', nucleus_pred)
            write_instance_mask(output_dir / f'{base_name}_cytoplasm.tif', cytoplasm_pred)
            tifffile.imwrite(output_dir / f'{base_name}_nucleus_prob.tif',
                             (details['nucleus_prob'] * 65535).astype(np.uint16))
            tifffile.imwrite(output_dir / f'{base_name}_cytoplasm_prob.tif',
                             (details['cytoplasm_prob'] * 65535).astype(np.uint16))

        # Create visualizations
        print("  Creating visualizations...")

        fig = create_comprehensive_figure(
            img, nucleus_pred, cytoplasm_pred,
            nucleus_gt, cytoplasm_gt, metrics
        )
        fig.savefig(output_dir / f'{base_name}_complete.png',
                   dpi=150, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)

        overlay = create_overlay(img, nucleus_pred, cytoplasm_pred,
                                nucleus_gt, cytoplasm_gt)
        imageio.imwrite(output_dir / f'{base_name}_overlay.png', overlay)

        with open(output_dir / f'{base_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Saved to {output_dir}")

    # Print key metrics
    if 'dice_nucleus' in metrics:
        print(f"    Dice: Nuc={metrics['dice_nucleus']:.3f}, "
              f"Cyto={metrics['dice_cytoplasm']:.3f}, JTPR={metrics['jtpr']:.3f}")
    if 'n_multi_nuclear' in metrics and metrics['n_multi_nuclear'] > 0:
        print(f"    Multi-nuclear: {metrics['n_multi_nuclear']} cells")

    return metrics


# ============================================================
# Command-Line Interface
# ============================================================

def main():
    """Main function for standalone inference."""
    import argparse
    import sys
    from pathlib import Path
    from tqdm import tqdm
    import json

    parser = argparse.ArgumentParser(
        description='CytoNucs Inference - Predict nuclei and cytoplasm instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python cytonucs_inference.py -c config.json -i image.tif -o results/
  
  # Batch processing
  python cytonucs_inference.py -c config.json -i data/test/images/ -o results/
  
  # With ground truth for evaluation
  python cytonucs_inference.py -c config.json -i data/test/images/ -o results/ \\
      --gt-nucleus data/test/nucleus_masks/ \\
      --gt-cytoplasm data/test/cytoplasm_masks/
        """
    )

    parser.add_argument('-c', '--config', required=True,
                       help='Config JSON file with inference.checkpoint_path')
    parser.add_argument('-i', '--input', required=True,
                       help='Input image file or directory')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory')
    parser.add_argument('--gt-nucleus',
                       help='Ground truth nucleus masks directory (optional)')
    parser.add_argument('--gt-cytoplasm',
                       help='Ground truth cytoplasm masks directory (optional)')
    parser.add_argument('--pattern', default='*.tif',
                       help='File pattern for batch processing (default: *.tif)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save outputs (only print metrics)')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)

    # Load model
    print("=" * 60)
    print("CytoNucs Inference")
    print("=" * 60)

    try:
        model, config, prob_thresh_nucleus, prob_thresh_cytoplasm, nms_thresh = \
            load_model_from_config(args.config)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create predictor
    predictor = CytoNucsPredictor(
        model, config,
        prob_thresh_nucleus=prob_thresh_nucleus,
        prob_thresh_cytoplasm=prob_thresh_cytoplasm,
        nms_thresh=nms_thresh,
    )
    print(f"Thresholds: nuc={prob_thresh_nucleus}, cyto={prob_thresh_cytoplasm}, nms={nms_thresh}\n")

    # Process
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image
        metrics = process_single_image(
            predictor, input_path, args.output,
            args.gt_nucleus, args.gt_cytoplasm,
            save_all=not args.no_save
        )

    else:
        # Batch processing
        image_files = sorted(input_path.glob(args.pattern))

        if len(image_files) == 0:
            print(f"No images found matching pattern: {args.pattern}")
            sys.exit(1)

        print(f"Found {len(image_files)} images\n")

        all_metrics = []
        for img_path in tqdm(image_files, desc="Processing"):
            try:
                metrics = process_single_image(
                    predictor, img_path, args.output,
                    args.gt_nucleus, args.gt_cytoplasm,
                    save_all=not args.no_save
                )
                all_metrics.append(metrics)
            except Exception as e:
                print(f"\n❌ Error on {img_path.name}: {e}")
                continue

        # Summary
        if all_metrics:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)

            total_nuclei = sum(m['n_nuclei'] for m in all_metrics)
            total_cells = sum(m['n_cells'] for m in all_metrics)
            print(f"Processed: {len(all_metrics)}/{len(image_files)} images")
            print(f"Total: {total_nuclei} nuclei, {total_cells} cells")

            if any('dice_nucleus' in m for m in all_metrics):
                metrics_with_gt = [m for m in all_metrics if 'dice_nucleus' in m]
                avg_dice_nuc = np.mean([m['dice_nucleus'] for m in metrics_with_gt])
                avg_dice_cyto = np.mean([m['dice_cytoplasm'] for m in metrics_with_gt])
                avg_jtpr = np.mean([m['jtpr'] for m in metrics_with_gt])

                print(f"\nAverage Metrics:")
                print(f"  Dice Nucleus:   {avg_dice_nuc:.4f}")
                print(f"  Dice Cytoplasm: {avg_dice_cyto:.4f}")
                print(f"  JTPR:           {avg_jtpr:.4f}")

                if not args.no_save:
                    # Save summary
                    summary = {
                        'n_images': len(all_metrics),
                        'total_nuclei': total_nuclei,
                        'total_cells': total_cells,
                        'average': {
                            'dice_nucleus': float(avg_dice_nuc),
                            'dice_cytoplasm': float(avg_dice_cyto),
                            'jtpr': float(avg_jtpr)
                        },
                        'per_image': all_metrics
                    }

                    with open(Path(args.output) / 'summary.json', 'w') as f:
                        json.dump(summary, f, indent=2)

                    print(f"\n✓ Summary saved to: {Path(args.output) / 'summary.json'}")

    print("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
