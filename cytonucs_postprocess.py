"""Mask-authoritative instance assembly and final-mask parent assignment."""
import numpy as np
from scipy.ndimage import label, generate_binary_structure, find_objects, map_coordinates


def upsample_grid(array, shape, grid):
    axes = [np.arange(s, dtype=np.float32) / g for s, g in zip(shape, grid)]
    return map_coordinates(array, np.asarray(np.meshgrid(*axes, indexing='ij')),
                           order=1, mode='nearest', prefilter=False)


def prompt_component(mask, center, sigma):
    cc, count = label(mask, structure=generate_binary_structure(mask.ndim, mask.ndim))
    if not count:
        return np.zeros_like(mask, dtype=bool)
    center = np.clip(np.round(center).astype(int), 0, np.array(mask.shape) - 1)
    own = cc[tuple(center)]
    if own == 0:
        pts = np.argwhere(mask)
        d2 = np.sum(((pts - center) / np.maximum(sigma, 1)) ** 2, axis=1)
        k = int(np.argmin(d2))
        if d2[k] > 4.0:
            return np.zeros_like(mask, dtype=bool)
        own = cc[tuple(pts[k])]
    return cc == own


def _component_near(mask, lo, center, sigma):
    """prompt_component on a sub-window starting at `lo` of a larger array.

    `center` is in full-array coordinates (already clipped to the full array, as in
    prompt_component); it may lie outside the window, then no own component exists.
    """
    cc, count = label(mask, structure=generate_binary_structure(mask.ndim, mask.ndim))
    if not count:
        return np.zeros_like(mask, dtype=bool)
    local = center - np.asarray(lo)
    inside = np.all(local >= 0) and np.all(local < np.array(mask.shape))
    own = cc[tuple(local)] if inside else 0
    if own == 0:
        pts = np.argwhere(mask) + np.asarray(lo)
        d2 = np.sum(((pts - center) / np.maximum(sigma, 1)) ** 2, axis=1)
        k = int(np.argmin(d2))
        if d2[k] > 4.0:
            return np.zeros_like(mask, dtype=bool)
        own = cc[tuple(pts[k] - np.asarray(lo))]
    return cc == own


def local_prompt_mask(p_grid, shape, grid, center, sigma, threshold):
    """prompt_component(upsample_grid(p_grid, shape, grid) > threshold, center, sigma), near the prompt.

    Upsampling the whole ROI on the CPU for every prompt dominated validation time,
    while the object is a small part of it. A full-resolution voxel can exceed the
    threshold only if a grid node of its interpolation cell does, so every foreground
    voxel within the prompt_component search ellipsoid (2 sigma) belongs to a grid
    component with a node near it. Those components (+1 cell) bound the evaluated window.
    The result is exact; when the window cannot be proven sufficient (the mask touches
    an interior window face, or foreground may exist only past the ROI) the full ROI is
    evaluated. Returns (lo, probability, foreground, mask, any_foreground): the arrays
    cover the evaluated window starting at `lo` (the whole ROI after a fallback); the
    mask equals the full-ROI mask cropped to that window and is empty outside it.
    """
    shape = tuple(int(v) for v in shape)
    nd = len(shape)
    g = np.asarray(grid, int)
    sigma = np.asarray(sigma, np.float64)
    center_full = np.clip(np.round(np.asarray(center, np.float64)).astype(int), 0, np.array(shape) - 1)

    origin = np.zeros(nd, int)

    def full():
        prob = upsample_grid(p_grid, shape, grid)
        fg = prob > threshold
        mask = prompt_component(fg, center, sigma) if fg.any() else fg
        return origin, prob, fg, mask, bool(fg.any())

    empty = (origin, np.zeros((1,) * nd, np.float32), np.zeros((1,) * nd, bool), np.zeros((1,) * nd, bool))

    # Grid nodes that can influence full-resolution voxels of this ROI.
    reach = tuple(min(int(ps), (int(s) - 1) // int(gg) + 2) for ps, s, gg in zip(p_grid.shape, shape, g))
    nodes = p_grid[tuple(slice(0, r) for r in reach)] > threshold
    if not nodes.any():
        return (*empty, False)
    exact_nodes = nodes[tuple(slice(0, (int(s) - 1) // int(gg) + 1) for s, gg in zip(shape, g))]
    if not exact_nodes.any():
        return full()                      # foreground could appear only between the last nodes
    # Only nodes in a box around the prompt can pass the distance test below.
    pad = np.ceil((2 * np.maximum(sigma, 1) + g) / g).astype(int) + 1
    box_lo = np.maximum(0, center_full // g - pad)
    box_hi = np.minimum(np.array(nodes.shape), center_full // g + pad + 1)
    pts = np.argwhere(nodes[tuple(slice(int(a), int(b)) for a, b in zip(box_lo, box_hi))]) + box_lo
    # Distance from the prompt to the nearest point of each node's interpolation cell.
    gap = np.maximum(np.abs(pts * g - center_full) - g, 0) / np.maximum(sigma, 1)
    near = pts[np.sum(gap ** 2, axis=1) <= 4.0 + 1e-9]
    if len(near) == 0:
        return (*empty, True)                   # foreground exists, but none within reach of the prompt
    # Node components of the near nodes, labelled in a box grown until none of them reaches one of
    # its faces inside the grid (then they are whole).
    structure = generate_binary_structure(nd, nd)
    while True:
        cc, _ = label(nodes[tuple(slice(int(a), int(b)) for a, b in zip(box_lo, box_hi))], structure=structure)
        boxes = find_objects(cc)
        own = [boxes[k - 1] for k in np.unique(cc[tuple((near - box_lo).T)])]
        lo_node = box_lo + np.min([[sl.start for sl in b] for b in own], axis=0)
        hi_node = box_lo + np.max([[sl.stop - 1 for sl in b] for b in own], axis=0)
        open_lo = (lo_node == box_lo) & (box_lo > 0)
        open_hi = (hi_node == box_hi - 1) & (box_hi < np.array(nodes.shape))
        if not (open_lo.any() or open_hi.any()):
            break
        width = box_hi - box_lo
        box_lo = np.maximum(0, box_lo - width)
        box_hi = np.minimum(np.array(nodes.shape), box_hi + width)
    lo = np.maximum(0, (lo_node - 1) * g)
    # Foreground voxels lie strictly inside (n-1)g .. (n+1)g of a foreground node n, so the
    # first and last window rows stay background and a touch means the object continues.
    hi = np.minimum(np.array(shape), (hi_node + 1) * g + 1)
    axes = [np.arange(a, b, dtype=np.float32) / gg for a, b, gg in zip(lo, hi, g)]
    node_lo = lo // g
    node_hi = np.minimum(np.array(p_grid.shape), (hi - 1) // g + 2)
    sub = p_grid[tuple(slice(int(a), int(b)) for a, b in zip(node_lo, node_hi))]
    coords = np.asarray(np.meshgrid(*[ax - np.float32(n) for ax, n in zip(axes, node_lo)], indexing='ij'))
    prob_r = map_coordinates(sub, coords, order=1, mode='nearest', prefilter=False)
    fg_r = prob_r > threshold
    mask_r = _component_near(fg_r, lo, center_full, sigma)
    for axis in range(nd):
        if (lo[axis] > 0 and np.take(mask_r, 0, axis=axis).any()) or \
                (hi[axis] < shape[axis] and np.take(mask_r, -1, axis=axis).any()):
            return full()
    return lo, prob_r, fg_r, mask_r, True


def mask_candidate(prob, mask, origin):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    lo, hi = pts.min(0), pts.max(0) + 1
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    mask = mask[sl].copy()
    return dict(lo=np.asarray(origin) + lo, mask=mask,
                score=np.where(mask, prob[sl], 0).astype(np.float32),
                quality=float(prob[sl][mask].mean()), mask_probability=float(prob[sl][mask].mean()),
                area=int(mask.sum()))


def grid_probability_window(p_grid, grid, lo, hi):
    """upsample_grid of p_grid restricted to the full-resolution window [lo, hi) (same interpolation)."""
    g = np.asarray(grid); lo = np.asarray(lo, int); hi = np.asarray(hi, int)
    axes = [np.arange(a, b, dtype=np.float32) / gg for a, b, gg in zip(lo, hi, g)]
    node_lo = lo // g
    node_hi = np.minimum(np.array(p_grid.shape), (hi - 1) // g + 2)
    sub = p_grid[tuple(slice(int(a), int(b)) for a, b in zip(node_lo, node_hi))]
    coords = np.asarray(np.meshgrid(*[ax - np.float32(n) for ax, n in zip(axes, node_lo)], indexing='ij'))
    return map_coordinates(sub, coords, order=1, mode='nearest', prefilter=False)


def rethreshold_candidate(candidate, threshold):
    """The candidate at another mask threshold, without a new network pass.

    Higher than the threshold it was made with: its own probabilities inside the mask
    (shrink_candidate, exact). Lower: the probability window stored during calibration
    around the mask; the same rule as inference (threshold, then the prompt component) is
    applied inside that window, so an object growing past the window is cut at its faces.
    """
    if threshold >= candidate.get('threshold', -np.inf) or 'probability_window' not in candidate:
        return shrink_candidate(candidate, threshold)
    lo, probability = candidate['probability_window']
    probability = probability.astype(np.float32)
    keep = probability > threshold
    if not keep.any():
        return None
    center = np.asarray(candidate['prompt_center'], np.float64) - np.asarray(lo)
    mask = prompt_component(keep, center, candidate['prompt_sigma'])
    if not mask.any():
        return None
    grown = mask_candidate(probability, mask, lo)
    grown['probability'] = grown['score']
    grown['threshold'] = threshold
    for key in ('prompt_center', 'prompt_sigma', 'predicted_quality', 'residual', 'probability_window'):
        if key in candidate:
            grown[key] = candidate[key]
    if candidate.get('score_scaled_by_quality'):
        grown['quality'] = candidate['quality']
        grown['score'] = grown['score'] * candidate['predicted_quality']
        grown['score_scaled_by_quality'] = True
    return grown


def shrink_candidate(candidate, threshold):
    """The candidate at a higher mask threshold, from its own stored probabilities.

    p > t' is a subset of p > t, so the stored probabilities inside the original mask
    suffice; the same prompt component rule as inference selects the object.
    """
    probability = candidate['probability']
    keep = probability > threshold
    if not keep.any():
        return None
    center = np.asarray(candidate['prompt_center'], np.float64) - np.asarray(candidate['lo'])
    mask = prompt_component(keep, center, candidate['prompt_sigma'])
    if not mask.any():
        return None
    shrunk = mask_candidate(probability, mask, candidate['lo'])
    shrunk['probability'] = shrunk['score']
    for key in ('prompt_center', 'prompt_sigma', 'predicted_quality', 'residual'):
        if key in candidate:
            shrunk[key] = candidate[key]
    if candidate.get('score_scaled_by_quality'):
        shrunk['quality'] = candidate['quality']
        shrunk['score'] = shrunk['score'] * candidate['predicted_quality']
        shrunk['score_scaled_by_quality'] = True
    return shrunk


def candidate_iou(a, b):
    lo = np.maximum(a['lo'], b['lo'])
    hi = np.minimum(a['lo'] + a['mask'].shape, b['lo'] + b['mask'].shape)
    if np.any(hi <= lo):
        return 0.0
    sa = tuple(slice(int(l), int(h)) for l, h in zip(lo-a['lo'], hi-a['lo']))
    sb = tuple(slice(int(l), int(h)) for l, h in zip(lo-b['lo'], hi-b['lo']))
    inter = int(np.logical_and(a['mask'][sa], b['mask'][sb]).sum())
    return inter / max(1, a['area'] + b['area'] - inter)


def assemble_masks(candidates, shape, dedup_iou=0.5, containment_threshold=0.0,
                   overlap_threshold=0.0, observer=None, assembly_mode='voxel',
                   min_remaining_fraction=0.5):
    """Mask NMS then voxelwise probability competition; no polygon ownership."""
    candidates = [c for c in candidates if c is not None]
    original_count = len(candidates)
    if observer is not None:
        observer.observe('before_nms', candidates=candidates)
    if assembly_mode not in ('voxel', 'object', 'object_quality', 'object_quality_containment'):
        raise ValueError('Unknown mask assembly mode')
    if assembly_mode != 'voxel':
        return _assemble_objects(candidates, shape, assembly_mode,
                                 min_remaining_fraction, observer, containment_threshold)
    if containment_threshold > 0:
        # Prefer a complete mask over a much smaller nested proposal. Opt-in:
        # imperfect large masks can also contain a genuinely different object.
        ordered = sorted(candidates, key=lambda c: (-c['area'], -c['quality'], tuple(c['lo'])))
        candidates = []
        for c in ordered:
            duplicate = False
            for big in candidates:
                if big['area'] < 2 * c['area'] or big['quality'] + .2 < c['quality']:
                    continue
                iou = candidate_iou(c, big)
                intersection = iou * (c['area'] + big['area']) / (1 + iou)
                if intersection / max(c['area'], 1) >= containment_threshold:
                    duplicate = True
                    break
            if not duplicate:
                candidates.append(c)
    # Spatial tie-break makes equal-quality candidates independent of input order.
    candidates.sort(key=lambda c: (-c['quality'], tuple(c['lo']), -c['area']))
    kept = []
    for c in candidates:
        def duplicate(k):
            iou = candidate_iou(c, k)
            intersection = iou * (c['area'] + k['area']) / (1 + iou)
            # IoU misses a fragment almost fully covered by an accepted object.
            # Use candidate coverage, NOT min-area coverage: a small retained
            # mask must not suppress a larger, more complete overlapping object.
            covered = intersection / max(c['area'], 1)
            return iou >= dedup_iou or (overlap_threshold > 0 and covered >= overlap_threshold)
        if any(duplicate(k) for k in kept):
            continue
        kept.append(c)
    if observer is not None:
        observer.observe('after_nms', candidates=kept)
    out = np.zeros(shape, np.int32)
    best = np.zeros(shape, np.float32)
    for lab, c in enumerate(kept, 1):
        sl = tuple(slice(int(l), int(l+s)) for l, s in zip(c['lo'], c['mask'].shape))
        win = c['mask'] & (c['score'] > best[sl])
        out[sl][win] = lab
        best[sl][win] = c['score'][win]
    if observer is not None:
        observer.observe('competition', labels=out)
    # Competition can split a mask; retain its largest connected remainder.
    for lab, sl in enumerate(find_objects(out), 1):
        if sl is None:
            continue
        cc, n = label(out[sl] == lab, structure=generate_binary_structure(len(shape), len(shape)))
        if n > 1:
            counts = np.bincount(cc.ravel()); counts[0] = 0
            out[sl][(cc > 0) & (cc != int(counts.argmax()))] = 0
    ids = np.unique(out); ids = ids[ids > 0]
    lut = np.zeros(len(kept) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    if observer is not None:
        observer.observe('final', labels=lut[out])
    return lut[out], dict(proposals=original_count, after_mask_nms=len(kept), written=len(ids))


def _assemble_objects(candidates, shape, mode, min_remaining_fraction, observer,
                      containment_threshold=0.0):
    """Commit one connected object at a time; later masks cannot steal voxels.

    No geometric pre-NMS: reject duplicates only against actually committed voxels.
    This avoids discarding a usable mask because of a candidate never written.
    Quality ranks objects only, and never acts as an absolute rejection threshold.
    Candidates flagged 'residual' (second-pass prompts) rank after all others, so they
    only fill voxels left free and never change the first-pass objects.
    """
    if not 0 <= min_remaining_fraction <= 1:
        raise ValueError('min_remaining_fraction must be in [0,1]')
    def priority(c):
        probability = c.get('mask_probability', c['quality'])
        q = c.get('predicted_quality', probability) if mode.startswith('object_quality') else probability
        if not np.isfinite(q):
            q = probability
        return (bool(c.get('residual')), -float(q), -c['area'], tuple(c['lo']))
    ordered = sorted(candidates, key=priority)
    if observer is not None:
        observer.observe('after_nms', candidates=ordered)  # pre-NMS intentionally bypassed
    out = np.zeros(shape, np.int32)
    written = rejected = written_residual = 0
    nested = mode == 'object_quality_containment'
    if nested and not 0 < containment_threshold <= 1:
        raise ValueError('object_quality_containment requires containment_threshold in (0,1]')
    committed = []
    committed_lo, committed_hi = [], []      # boxes of the committed masks: only overlapping ones can conflict
    conflicts = []
    label_ranks = {}
    trace = getattr(observer, 'record_assembly_decision', None)
    for rank, c in enumerate(ordered):
        if nested:
            conflict = None
            overlapping = []
            if committed:
                c_lo = np.asarray(c['lo']); c_hi = c_lo + np.array(c['mask'].shape)
                overlapping = np.flatnonzero(np.all(np.minimum(np.asarray(committed_hi), c_hi) >
                                                    np.maximum(np.asarray(committed_lo), c_lo), axis=1))
            for accepted_rank, accepted, original in (committed[j] for j in overlapping):
                iou = candidate_iou(c, accepted)
                intersection = iou * (c['area'] + accepted['area']) / (1 + iou)
                coverage = intersection / max(1, min(c['area'], accepted['area']))
                if coverage < containment_threshold:
                    continue
                original_iou = candidate_iou(c, original)
                original_intersection = original_iou * (c['area'] + original['area']) / (1 + original_iou)
                original_coverage = original_intersection / max(1, min(c['area'], original['area']))
                # Clipping a neighbouring mask can create a small nested sliver.
                # Such a sliver is not evidence that the original objects duplicate.
                if original_coverage >= containment_threshold:
                    conflict = dict(rejected_rank=rank, accepted_rank=accepted_rank,
                                    rejected_area=c['area'], accepted_area=accepted['area'],
                                    rejected_quality=-priority(c)[1],
                                    accepted_quality=-priority(accepted)[1],
                                    smaller_coverage=float(coverage), iou=float(iou),
                                    original_smaller_coverage=float(original_coverage),
                                    rejected_contains_accepted=bool(c['area'] > accepted['area']))
                    break
            if conflict is not None:
                conflicts.append(conflict)
                if trace is not None:
                    trace(rank, 'nested', c, blockers=[conflict['accepted_rank']],
                          quality=-priority(c)[1])
                continue
        sl = tuple(slice(int(l), int(l+s)) for l,s in zip(c['lo'],c['mask'].shape))
        free = c['mask'] & (out[sl] == 0)
        free_count = int(np.count_nonzero(free))
        blockers = ([label_ranks[int(lab)] for lab in np.unique(out[sl][c['mask']]) if lab > 0]
                    if trace is not None else [])
        cc,n = label(free, structure=generate_binary_structure(len(shape),len(shape)))
        if n:
            sizes = np.bincount(cc.ravel());sizes[0]=0
            free = cc == int(np.argmax(sizes))
        # Evaluate the actual connected remainder, not a disconnected sum.
        if not n or np.count_nonzero(free) < min_remaining_fraction * c['area']:
            rejected += 1
            if trace is not None:
                trace(rank, 'occupied' if not n else 'insufficient_connected_remainder', c,
                      blockers=blockers, quality=-priority(c)[1], free_voxels=free_count,
                      connected_voxels=int(np.count_nonzero(free)))
            continue
        written += 1
        written_residual += bool(c.get('residual'))
        out[sl][free] = written
        label_ranks[written] = rank
        if trace is not None:
            trace(rank, 'written', c, blockers=blockers, quality=-priority(c)[1],
                  free_voxels=free_count, connected_voxels=int(np.count_nonzero(free)),
                  output_label=written,
                  remainder=dict(lo=c['lo'], mask=free))
        if nested:
            # Compare only with masks actually written, never rejected proposals
            # or parts removed by overlap/connected-component selection.
            accepted = mask_candidate(c['score'], free, c['lo'])
            accepted['predicted_quality'] = -priority(c)[1]
            committed.append((rank, accepted, c))
            committed_lo.append(np.asarray(accepted['lo'])); committed_hi.append(accepted['lo'] + np.array(accepted['mask'].shape))
    if observer is not None:
        observer.observe('competition', labels=out)
        observer.observe('final', labels=out)
    stats = dict(proposals=len(candidates), after_mask_nms=len(candidates), written=written,
                 rejected_remainder=rejected, assembly_mode=mode, written_residual=written_residual)
    if nested:
        stats.update(rejected_nested=len(conflicts), nested_conflicts=conflicts)
    return out, stats


def drop_small_instances(labels, min_area):
    """Remove instances smaller than min_area voxels and relabel 1..N in the original order."""
    labels = np.asarray(labels)
    if not min_area or not labels.any():
        return labels, 0
    counts = np.bincount(labels.ravel())
    small = counts < min_area
    small[0] = False
    lut = np.zeros(len(counts), labels.dtype)
    keep = np.flatnonzero((counts > 0) & ~small)
    keep = keep[keep > 0]
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[labels], int(np.count_nonzero(small & (counts > 0)))


def assign_final_masks(nuclei, cells, min_overlap=0.1, evidence=None, min_mask_iou=0.25):
    """Many nuclei per cell allowed; no forced nearest-cell assignment."""
    if nuclei.shape != cells.shape:
        raise ValueError('Nucleus/cell shapes differ')
    assignments = {}
    for lab, sl in enumerate(find_objects(nuclei), 1):
        if sl is None:
            continue
        mask = nuclei[sl] == lab
        values, counts = np.unique(cells[sl][mask], return_counts=True)
        ok = values > 0
        if ok.any():
            values, counts = values[ok], counts[ok]
            k = int(np.argmax(counts))
            if counts[k] / max(1, int(mask.sum())) >= min_overlap:
                assignments[int(lab)] = int(values[k])
    # Some annotations exclude the nucleus from the cytoplasm. Use the cell
    # predicted by this nuclear prompt as evidence, never nearest-cell distance.
    if evidence:
        cell_masks = []
        for lab, sl in enumerate(find_objects(cells), 1):
            if sl is not None:
                mask = cells[sl] == lab
                cell_masks.append((lab, mask_candidate(mask.astype(np.float32), mask,
                                                       [s.start for s in sl])))
        for lab, sl in enumerate(find_objects(nuclei), 1):
            if sl is None or lab in assignments:
                continue
            mask = nuclei[sl] == lab
            nuc = mask_candidate(mask.astype(np.float32), mask, [s.start for s in sl])
            candidates = [(candidate_iou(nuc, n), c) for n, c in evidence if c is not None]
            if not candidates:
                continue
            overlap, hint = max(candidates, key=lambda x: x[0])
            if overlap < 0.5:
                continue
            scores = [(candidate_iou(hint, c), cid) for cid, c in cell_masks]
            if scores:
                score, cid = max(scores)
                if score >= min_mask_iou:
                    assignments[int(lab)] = int(cid)
    return assignments
