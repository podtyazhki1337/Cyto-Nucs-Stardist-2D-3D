"""
Helper functions for containment loss computation in numpy.
Can be used in custom training loop or post-processing.
"""
import numpy as np
from skimage.measure import regionprops


def _to_shape(arr, shape):
    """A map resized to the given spatial shape by nearest neighbour (nucleus_grid and cytoplasm_grid may differ,
    so the maps of the two heads may have different sizes)."""
    arr = np.asarray(arr)
    if tuple(arr.shape) == tuple(shape):
        return arr
    idx = np.ix_(*[np.clip((np.arange(s) * a // max(s, 1)), 0, a - 1)
                   for s, a in zip(shape, arr.shape)])
    return arr[idx]


def compute_containment_penalty(
        nucleus_instances,
        cytoplasm_instances,
        cytoplasm_dist_pred,
        assignments,
        containment_margin=2.0,
        rays=None,
        grid=None
):
    """
    Compute containment penalty in numpy (for monitoring or post-processing).

    For a cell with several nuclei the penalty is computed for every nucleus and averaged over the cell: the
    cytoplasm must hold all of its nuclei.

    The cell radius is taken along the direction cell centre -> nucleus centre, not as the mean over the rays:
    with an anisotropic ray set most rays hit the top / bottom of a flat cell, so the mean is several times too
    short and the penalty would fire on normal cells. Without rays the mean is used.

    grid: the subsampling of cytoplasm_dist_pred. Centroids from regionprops are in full-resolution coordinates
    and are mapped to the grid of the distance map.

    Args:
        nucleus_instances: (Z, Y, X) nucleus instance labels
        cytoplasm_instances: (Z, Y, X) cytoplasm instance labels
        cytoplasm_dist_pred: (z, y, x, K) predicted cytoplasm distances (grid-res)
        assignments: dict {nucleus_id: cytoplasm_id}
        containment_margin: tolerance in voxels
        rays: ray set of the cytoplasm (config.rays_cytoplasm), optional
        grid: subsampling of the distance map, optional

    Returns:
        penalty: scalar, average containment violation
        details: dict with per-cell penalties
    """
    if len(assignments) == 0:
        return 0.0, {}

    cytoplasm_dist_pred = np.asarray(cytoplasm_dist_pred)
    nd = int(np.asarray(cytoplasm_instances).ndim)

    if grid is None:
        # The grid follows from the ratio of the shapes: the distance map is subsampled exactly so.
        grid = tuple(max(1, int(round(s / max(d, 1))))
                     for s, d in zip(np.asarray(cytoplasm_instances).shape,
                                     cytoplasm_dist_pred.shape[:nd]))
    grid = tuple(int(g) for g in grid)

    verts = None
    if rays is not None:
        try:
            verts = np.asarray(rays.vertices, np.float32)          # (K, nd)
            vnorm = np.linalg.norm(verts, axis=1)
            vunit = verts / np.maximum(vnorm[:, None], 1e-9)
        except Exception:
            verts = None

    nucleus_props = {prop.label: prop for prop in regionprops(nucleus_instances)}
    cytoplasm_props = {prop.label: prop for prop in regionprops(cytoplasm_instances)}

    # Group nuclei by cell
    cell_to_nuclei = {}
    for nuc_id, cell_id in assignments.items():
        cell_to_nuclei.setdefault(cell_id, []).append(nuc_id)

    dist_shape = cytoplasm_dist_pred.shape[:nd]
    cell_penalties = {}

    for cell_id, nucleus_ids in cell_to_nuclei.items():
        if cell_id not in cytoplasm_props:
            continue
        cell_center = np.array(cytoplasm_props[cell_id].centroid, np.float64)

        # The centroid of a concave cell may lie outside the cell, where the rays are background: move it to the
        # nearest voxel of its own cell.
        cc_int = np.clip(np.round(cell_center).astype(int),
                         0, np.array(cytoplasm_instances.shape) - 1)
        if cytoplasm_instances[tuple(cc_int)] != cell_id:
            pts = np.argwhere(cytoplasm_instances == cell_id)
            if len(pts) == 0:
                continue
            cc_int = pts[np.argmin(np.abs(pts - cell_center).sum(axis=1))]
            cell_center = cc_int.astype(np.float64)

        # full resolution -> grid coordinates of the distance map
        cc_grid = tuple(int(np.clip(cc_int[a] // grid[a], 0, dist_shape[a] - 1))
                        for a in range(nd))
        predicted_radii = np.asarray(cytoplasm_dist_pred[cc_grid], np.float64)

        penalties = []
        for nuc_id in nucleus_ids:
            if nuc_id not in nucleus_props:
                continue
            nucleus_center = np.array(nucleus_props[nuc_id].centroid, np.float64)
            vec = nucleus_center - cell_center
            distance = float(np.linalg.norm(vec))
            if distance < 1e-6:
                continue

            if verts is not None and len(predicted_radii) == len(verts):
                # Radius towards the nucleus: the ray with the largest cosine, its length in voxels (|d * v|).
                u = vec / distance
                k = int(np.argmax(vunit @ u))
                radius = float(predicted_radii[k] * vnorm[k])
            else:
                radius = float(np.mean(predicted_radii))

            if distance > radius + containment_margin:
                penalties.append((distance - radius) / (radius + 1e-6))

        if len(penalties) > 0:
            cell_penalties[cell_id] = {
                'penalty': float(np.mean(penalties)),
                'num_nuclei': len(nucleus_ids),
                'violations': len(penalties)
            }

    if len(cell_penalties) > 0:
        total_penalty = float(np.mean([v['penalty'] for v in cell_penalties.values()]))
    else:
        total_penalty = 0.0

    return total_penalty, cell_penalties


def monitor_wbr_metric(nucleus_prob, cytoplasm_prob, cytoplasm_instances=None,
                       prob_thresh=0.5):
    """
    Monitor WBR metric in numpy (for logging/visualization):
    the share of the nucleus mass that lies outside the cells.

    - Normalised by the nucleus area (not by the background area, which made the metric ~1e-5 whatever the model).
    - The container is preferably binary (cytoplasm_instances): cytoplasm_prob is an edt_prob, above 0.5 on only
      14-25 % of a cell's area, so thresholding it would count most of the cell as outside.
    - The maps of the two heads are brought to one shape (nucleus_grid and cytoplasm_grid may differ).

    Args:
        nucleus_prob: (Z, Y, X) nucleus probability
        cytoplasm_prob: (Z, Y, X) cytoplasm probability
        cytoplasm_instances: (Z, Y, X) binary or instance container, optional
        prob_thresh: threshold that binarises the probabilities

    Returns:
        ratio: fraction of nucleus outside cytoplasm, [0, 1]
    """
    nucleus_prob = np.asarray(nucleus_prob)
    shape = nucleus_prob.shape

    if cytoplasm_instances is not None:
        inside = _to_shape(np.asarray(cytoplasm_instances) > 0, shape)
    else:
        inside = _to_shape(np.asarray(cytoplasm_prob), shape) > prob_thresh

    nucleus_mask = nucleus_prob > prob_thresh
    n_nuc = int(nucleus_mask.sum())
    if n_nuc == 0:
        return 0.0
    return float(np.logical_and(nucleus_mask, ~inside).sum()) / n_nuc