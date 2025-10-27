"""
Helper functions for containment loss computation in numpy.
Can be used in custom training loop or post-processing.
"""
import numpy as np
from skimage.measure import regionprops


def compute_containment_penalty(
        nucleus_instances,
        cytoplasm_instances,
        cytoplasm_dist_pred,
        assignments,
        containment_margin=2.0
):
    """
    Compute containment penalty in numpy (for monitoring or custom training).

    For multi-nuclear cells:
    - Checks each nucleus in the cell
    - Averages penalty across all nuclei

    Args:
        nucleus_instances: (Z, Y, X) nucleus instance labels
        cytoplasm_instances: (Z, Y, X) cytoplasm instance labels
        cytoplasm_dist_pred: (Z, Y, X, K) predicted cytoplasm distances
        assignments: dict {nucleus_id: cytoplasm_id}
        containment_margin: tolerance in voxels

    Returns:
        penalty: scalar, average containment violation
        details: dict with per-cell penalties
    """
    if len(assignments) == 0:
        return 0.0, {}

    nucleus_props = {prop.label: prop for prop in regionprops(nucleus_instances)}
    cytoplasm_props = {prop.label: prop for prop in regionprops(cytoplasm_instances)}

    # Group nuclei by cell
    cell_to_nuclei = {}
    for nuc_id, cell_id in assignments.items():
        if cell_id not in cell_to_nuclei:
            cell_to_nuclei[cell_id] = []
        cell_to_nuclei[cell_id].append(nuc_id)

    cell_penalties = {}

    for cell_id, nucleus_ids in cell_to_nuclei.items():
        if cell_id not in cytoplasm_props:
            continue

        cell_center = np.array(cytoplasm_props[cell_id].centroid)
        penalties = []

        # Check each nucleus (handles multi-nuclear case)
        for nuc_id in nucleus_ids:
            if nuc_id not in nucleus_props:
                continue

            nucleus_center = np.array(nucleus_props[nuc_id].centroid)
            vec = nucleus_center - cell_center
            distance = np.linalg.norm(vec)

            if distance < 1e-6:
                continue

            # Get predicted cell radius
            cell_center_int = tuple(cell_center.astype(int))
            if all(0 <= cell_center_int[i] < cytoplasm_dist_pred.shape[i]
                   for i in range(len(cell_center_int))):
                predicted_radii = cytoplasm_dist_pred[cell_center_int]
                mean_radius = np.mean(predicted_radii)

                # Check violation
                if distance > mean_radius + containment_margin:
                    penalty = (distance - mean_radius) / (mean_radius + 1e-6)
                    penalties.append(penalty)

        # Average over nuclei in this cell
        if len(penalties) > 0:
            cell_penalties[cell_id] = {
                'penalty': np.mean(penalties),
                'num_nuclei': len(nucleus_ids),
                'violations': len(penalties)
            }

    # Overall penalty
    if len(cell_penalties) > 0:
        total_penalty = np.mean([v['penalty'] for v in cell_penalties.values()])
    else:
        total_penalty = 0.0

    return total_penalty, cell_penalties


def monitor_wbr_metric(nucleus_prob, cytoplasm_prob):
    """
    Monitor WBR metric in numpy (for logging/visualization).

    Args:
        nucleus_prob: (Z, Y, X) nucleus probability
        cytoplasm_prob: (Z, Y, X) cytoplasm probability

    Returns:
        ratio: fraction of nucleus outside cytoplasm
    """
    nucleus_mask = nucleus_prob > 0.5
    cytoplasm_inverted = cytoplasm_prob < 0.5

    nucleus_outside = np.logical_and(nucleus_mask, cytoplasm_inverted).sum()
    total_outside = cytoplasm_inverted.sum()

    ratio = nucleus_outside / max(total_outside, 1)
    return ratio