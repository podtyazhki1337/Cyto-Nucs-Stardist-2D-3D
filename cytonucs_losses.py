"""
CytoNucs StarDist Loss Functions

Implements:
1. Probability loss (BCE for object detection)
2. Distance loss (MSE for radial distances)
3. Containment loss (nuclei inside cells)
4. Within-Boundary Regularization (WBR)
5. Consistency loss (cell center near nucleus cluster)
"""
import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


class CytoNucsLoss:
    """
    Combined loss for CytoNucs StarDist.
    """

    def __init__(self, config):
        self.config = config
        self.ndim = config.ndim

    def __call__(self, y_true, y_pred, assignments=None):
        """
        Compute total loss.

        Args:
            y_true: dict with keys:
                - 'nucleus_prob': (B, Z, Y, X, 1) ground truth nucleus probability
                - 'nucleus_dist': (B, Z, Y, X, n_rays) ground truth nucleus distances
                - 'cytoplasm_prob': (B, Z, Y, X, 1) ground truth cytoplasm probability
                - 'cytoplasm_dist': (B, Z, Y, X, n_rays) ground truth cytoplasm distances
                - 'nucleus_instances': (B, Z, Y, X) instance labels for nuclei
                - 'cytoplasm_instances': (B, Z, Y, X) instance labels for cytoplasm

            y_pred: dict with keys matching y_true (model outputs)

            assignments: dict mapping nucleus IDs to cytoplasm IDs (per sample in batch)

        Returns:
            total_loss: scalar tensor
        """
        cfg = self.config

        # 1. Standard StarDist losses for nucleus
        loss_nucleus_prob = self.probability_loss(
            y_true['nucleus_prob'], y_pred['nucleus_prob']
        )
        loss_nucleus_dist = self.distance_loss(
            y_true['nucleus_dist'], y_pred['nucleus_dist'],
            y_true['nucleus_prob']  # mask by foreground
        )

        # 2. Standard StarDist losses for cytoplasm
        loss_cytoplasm_prob = self.probability_loss(
            y_true['cytoplasm_prob'], y_pred['cytoplasm_prob']
        )
        loss_cytoplasm_dist = self.distance_loss(
            y_true['cytoplasm_dist'], y_pred['cytoplasm_dist'],
            y_true['cytoplasm_prob']
        )

        # 3. Containment loss (nuclei inside cells)
        loss_containment = self.containment_loss(
            y_true, y_pred, assignments
        )

        # 4. Within-Boundary Regularization
        loss_wbr = self.wbr_loss(
            y_pred['nucleus_prob'],
            y_pred['cytoplasm_prob']
        )

        # 5. Consistency loss (optional)
        loss_consistency = self.consistency_loss(
            y_true, y_pred, assignments
        )

        # Weighted sum
        total_loss = (
                cfg.lambda_prob_nucleus * loss_nucleus_prob +
                cfg.lambda_dist_nucleus * loss_nucleus_dist +
                cfg.lambda_prob_cytoplasm * loss_cytoplasm_prob +
                cfg.lambda_dist_cytoplasm * loss_cytoplasm_dist +
                cfg.lambda_containment * loss_containment +
                cfg.lambda_wbr * loss_wbr +
                cfg.lambda_consistency * loss_consistency
        )

        return total_loss

    def probability_loss(self, y_true, y_pred):
        """
        Binary cross-entropy for object center detection.

        Args:
            y_true: (B, Z, Y, X, 1) ground truth probability
            y_pred: (B, Z, Y, X, 1) predicted probability

        Returns:
            loss: scalar
        """
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(y_true, y_pred)

    def distance_loss(self, y_true, y_pred, mask):
        """
        MSE for radial distance prediction, masked by foreground.

        Args:
            y_true: (B, Z, Y, X, n_rays) ground truth distances
            y_pred: (B, Z, Y, X, n_rays) predicted distances
            mask: (B, Z, Y, X, 1) foreground mask

        Returns:
            loss: scalar
        """
        # Expand mask to match distance shape
        mask_expanded = tf.tile(mask, [1, 1, 1, 1, tf.shape(y_pred)[-1]])

        # Compute squared error
        squared_error = tf.square(y_true - y_pred)

        # Apply mask
        masked_error = squared_error * tf.cast(mask_expanded, tf.float32)

        # Average over foreground pixels only
        loss = tf.reduce_sum(masked_error) / (tf.reduce_sum(mask) + 1e-8)

        return loss

    def containment_loss(self, y_true, y_pred, assignments):
        """
        Ensures each nucleus is inside its predicted cell.

        Works for both single-nucleus and multi-nucleus cells:
        - Single-nucleus cell: containment_loss = check_single_nucleus()
        - Multi-nucleus cell: containment_loss = mean([check_nuc1(), check_nuc2(), ...])

        For each nucleus:
        - Compute vector from cell center to nucleus center
        - Check if nucleus center is within predicted cell boundary along that direction
        - Penalize if outside

        Args:
            y_true: ground truth dict
            y_pred: predictions dict
            assignments: dict mapping nucleus_id -> cytoplasm_id for each batch sample

        Returns:
            loss: scalar
        """
        if assignments is None:
            return tf.constant(0.0)

        # This is a simplified placeholder - full implementation requires:
        # 1. Extract predicted rays for each cytoplasm instance
        # 2. For each assigned nucleus, compute distance from cell center
        # 3. Compare with predicted radius in that direction
        # 4. Penalize if nucleus is outside
        # 5. For multi-nucleus cells: average loss across all nuclei of that cell

        # For now, return zero (implement in training loop with numpy/scipy)
        return tf.constant(0.0)

    def wbr_loss(self, nucleus_prob, cytoplasm_prob):
        """
        Within-Boundary Regularization.
        Penalizes regions where nucleus probability is high but outside cytoplasm.

        Args:
            nucleus_prob: (B, Z, Y, X, 1) predicted nucleus probability
            cytoplasm_prob: (B, Z, Y, X, 1) predicted cytoplasm probability

        Returns:
            loss: scalar
        """
        # Nucleus probability outside cytoplasm
        # WBR = nucleus_prob * (1 - cytoplasm_prob)
        outside_penalty = nucleus_prob * (1 - cytoplasm_prob)

        # Average over all pixels
        loss = tf.reduce_mean(outside_penalty)

        return loss

    def consistency_loss(self, y_true, y_pred, assignments):
        """
        Encourages cell center to be near the centroid of its nuclei.

        Args:
            y_true: ground truth dict
            y_pred: predictions dict
            assignments: nucleus-cytoplasm assignments

        Returns:
            loss: scalar
        """
        # Placeholder - implement in training loop
        # For each cell:
        # 1. Find all assigned nuclei centers
        # 2. Compute their centroid
        # 3. Compute distance from predicted cell center to this centroid
        # 4. Penalize large distances

        return tf.constant(0.0)


def compute_nucleus_cell_assignments(nucleus_instances, cytoplasm_instances,
                                     max_distance=5.0):
    """
    Assign nuclei to cells based on containment and proximity.

    Strategy:
    1. If nucleus centroid is inside a cell mask → assign to that cell
    2. Else, use maximum overlap between nucleus and cell masks
    3. If no overlap, assign to nearest cell center (if within max_distance)
    4. Otherwise, mark as unassigned

    Args:
        nucleus_instances: (Z, Y, X) nucleus instance labels
        cytoplasm_instances: (Z, Y, X) cytoplasm instance labels
        max_distance: maximum distance (voxels) for nearest neighbor assignment

    Returns:
        assignments: dict {nucleus_id: cytoplasm_id}
        unassigned: list of nucleus_ids with no valid assignment
    """
    from skimage.measure import regionprops
    from scipy.spatial.distance import cdist

    nucleus_props = regionprops(nucleus_instances)
    cytoplasm_props = regionprops(cytoplasm_instances)

    assignments = {}
    unassigned = []

    # Build cell masks for overlap computation
    cell_mask_dict = {prop.label: (cytoplasm_instances == prop.label)
                      for prop in cytoplasm_props}

    # Build cell centers
    cell_centers = {prop.label: np.array(prop.centroid)
                    for prop in cytoplasm_props}

    for nuc_prop in nucleus_props:
        nuc_id = nuc_prop.label
        nuc_center = np.array(nuc_prop.centroid).astype(int)

        # Strategy 1: Check if nucleus center is inside any cell
        assigned = False
        if tuple(nuc_center) in zip(*np.where(cytoplasm_instances > 0)):
            cell_id = cytoplasm_instances[tuple(nuc_center)]
            if cell_id > 0:
                assignments[nuc_id] = int(cell_id)
                assigned = True

        if not assigned:
            # Strategy 2: Find cell with maximum overlap
            nuc_mask = (nucleus_instances == nuc_id)
            max_overlap = 0
            best_cell = None

            for cell_id, cell_mask in cell_mask_dict.items():
                overlap = np.logical_and(nuc_mask, cell_mask).sum()
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_cell = cell_id

            if max_overlap > 0:
                assignments[nuc_id] = best_cell
                assigned = True

        if not assigned:
            # Strategy 3: Nearest cell center (if within threshold)
            if len(cell_centers) > 0:
                centers_array = np.array(list(cell_centers.values()))
                cell_ids = list(cell_centers.keys())

                distances = cdist([nuc_center], centers_array)[0]
                nearest_idx = np.argmin(distances)

                if distances[nearest_idx] <= max_distance:
                    assignments[nuc_id] = cell_ids[nearest_idx]
                    assigned = True

        if not assigned:
            unassigned.append(nuc_id)

    return assignments, unassigned


def create_distance_maps(instances, rays):
    """
    Create radial distance maps for StarDist from instance labels.

    Args:
        instances: (Z, Y, X) or (Y, X) instance segmentation
        rays: Rays object with vertices

    Returns:
        prob_map: (Z, Y, X, 1) or (Y, X, 1) probability map
        dist_map: (Z, Y, X, n_rays) or (Y, X, n_rays) distance map
    """
    from stardist.geometry import star_dist

    # Use StarDist's built-in function
    dist_map = star_dist(instances, rays)

    # Create probability map (1 at object centers, 0 elsewhere)
    prob_map = (instances > 0).astype(np.float32)

    # Refine: probability should be higher at centers
    # Use distance transform to find centers
    for label in np.unique(instances):
        if label == 0:
            continue
        mask = (instances == label)
        dist_to_boundary = distance_transform_edt(mask)
        max_dist = dist_to_boundary.max()
        if max_dist > 0:
            prob_map[mask] = dist_to_boundary[mask] / max_dist

    prob_map = prob_map[..., np.newaxis]  # Add channel dimension

    return prob_map, dist_map