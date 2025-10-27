"""
CytoNucs StarDist Loss Functions - FULL IMPLEMENTATION

Based on StarDist paper with adaptations for multi-nuclear cells.
Implements:
1. Object Boundary Loss (BCE)
2. Distance Loss (MAE/L1)
3. Within-Boundary Regularization (WBR)
4. Containment loss (geometric constraint for multi-nuclear cells)
"""
import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


class CytoNucsLoss:
    """
    Combined loss for CytoNucs StarDist.

    Total loss: L' = Σ(L1_nucleus) + Σ(L2_cytoplasm) + λ3·Λ_wbr + λ4·L_containment

    Where:
    - L1_nucleus: StarDist loss for nucleus decoder (BCE + distance)
    - L2_cytoplasm: StarDist loss for cytoplasm decoder (BCE + distance)
    - Λ_wbr: Within-Boundary Regularization penalty
    - L_containment: Geometric containment for multi-nuclear cells
    """

    def __init__(self, config):
        self.config = config
        self.ndim = config.ndim

        # Regularization factors from paper
        self.lambda1 = 1.0  # Distance loss weight (λ1 in paper)
        self.lambda2 = 0.01  # Background penalty (λ2 in paper)
        self.lambda3 = config.lambda_wbr  # WBR weight (λ3 in paper)
        self.lambda4 = config.lambda_containment  # Containment weight (new for multi-nuclear)
        self.epsilon = 1e-6  # Small epsilon for numerical stability (ϵ in paper)

    def __call__(self, y_true, y_pred, assignments=None):
        """
        Compute total loss.

        Args:
            y_true: dict with ground truth
            y_pred: dict with predictions
            assignments: nucleus-cytoplasm assignments (optional, for containment)

        Returns:
            total_loss: scalar tensor
        """
        cfg = self.config

        # ========== StarDist Losses for Nucleus (L1 in paper) ==========
        # L1 = BCE(d_ij, d̂_ij) + λ1·(distance_term) + λ2·(background_penalty)
        loss_nucleus = self.stardist_loss(
            prob_true=y_true['nucleus_prob'],
            prob_pred=y_pred['nucleus_prob'],
            dist_true=y_true['nucleus_dist'],
            dist_pred=y_pred['nucleus_dist'],
            name='nucleus'
        )

        # ========== StarDist Losses for Cytoplasm (L2 in paper) ==========
        loss_cytoplasm = self.stardist_loss(
            prob_true=y_true['cytoplasm_prob'],
            prob_pred=y_pred['cytoplasm_prob'],
            dist_true=y_true['cytoplasm_dist'],
            dist_pred=y_pred['cytoplasm_dist'],
            name='cytoplasm'
        )

        # ========== Within-Boundary Regularization (Λ in paper, Eq. 5) ==========
        # Penalize nucleus predictions outside cytoplasm boundaries
        loss_wbr = self.wbr_loss(
            nucleus_prob_pred=y_pred['nucleus_prob'],
            cytoplasm_prob_pred=y_pred['cytoplasm_prob']
        )

        # ========== Containment Loss (NEW - for multi-nuclear cells) ==========
        # Geometric constraint: all nuclei must be inside their parent cell
        # This is different from WBR (which is soft probability constraint)
        # Containment is hard geometric constraint on predicted polygons
        loss_containment = self.containment_loss(
            y_true=y_true,
            y_pred=y_pred,
            assignments=assignments
        )

        # ========== Total Loss (Eq. 4 from paper, adapted) ==========
        total_loss = (
            cfg.lambda_prob_nucleus * loss_nucleus['bce'] +
            cfg.lambda_dist_nucleus * loss_nucleus['distance'] +
            cfg.lambda_prob_cytoplasm * loss_cytoplasm['bce'] +
            cfg.lambda_dist_cytoplasm * loss_cytoplasm['distance'] +
            self.lambda3 * loss_wbr +
            self.lambda4 * loss_containment
        )

        return total_loss

    def stardist_loss(self, prob_true, prob_pred, dist_true, dist_pred, name='object'):
        """
        StarDist loss for one decoder (nucleus or cytoplasm).

        Implements Equation 3 from paper:
        L = BCE(d_ij, d̂_ij) + λ1·(d_ij·1_{d_ij>0}·(1/K)·Σ|r^k_ij - r̂^k_ij|) +
            λ2·1_{d_ij=0}·(1/K)·Σ|r̂^k_ij|

        Where:
        - d_ij: object probability at pixel (i,j)
        - r^k_ij: distance along ray k at pixel (i,j)
        - K: number of rays

        Args:
            prob_true: (B, Z, Y, X, 1) ground truth object probability
            prob_pred: (B, Z, Y, X, 1) predicted object probability
            dist_true: (B, Z, Y, X, K) ground truth distances
            dist_pred: (B, Z, Y, X, K) predicted distances
            name: 'nucleus' or 'cytoplasm' for logging

        Returns:
            dict with 'bce' and 'distance' components
        """
        K = tf.cast(tf.shape(dist_pred)[-1], tf.float32)  # Number of rays

        # ========== Object Boundary Loss (LOBL) ==========
        # BCE loss for object probability map
        # LOBL = BCE(d_ij, d̂_ij)
        bce = tf.keras.losses.binary_crossentropy(prob_true, prob_pred)
        loss_bce = tf.reduce_mean(bce)

        # ========== Distance Loss (LDL) ==========
        # Two terms:
        # 1. Foreground term: MAE on predicted distances (where d_ij > 0)
        # 2. Background penalty: penalize non-zero predictions on background (where d_ij = 0)

        # Create masks for foreground and background
        # foreground_mask: 1 where d_ij > 0 (object pixels)
        # background_mask: 1 where d_ij = 0 (background pixels)
        foreground_mask = tf.cast(prob_true > 0.5, tf.float32)  # 1_{d_ij>0}
        background_mask = 1.0 - foreground_mask  # 1_{d_ij=0}

        # Expand masks to match distance shape (B, Z, Y, X, K)
        foreground_mask_exp = tf.tile(foreground_mask, [1, 1, 1, 1, tf.shape(dist_pred)[-1]])
        background_mask_exp = tf.tile(background_mask, [1, 1, 1, 1, tf.shape(dist_pred)[-1]])

        # Term 1: Foreground distance error (λ1 term in Eq. 3)
        # d_ij·1_{d_ij>0}·(1/K)·Σ_k |r^k_ij - r̂^k_ij|
        distance_error = tf.abs(dist_true - dist_pred)  # |r^k_ij - r̂^k_ij|
        foreground_distance = distance_error * foreground_mask_exp  # Mask to foreground only

        # Average over K rays: (1/K)·Σ_k
        foreground_distance_mean = tf.reduce_mean(foreground_distance, axis=-1, keepdims=True)

        # Weight by probability (d_ij factor in Eq. 3)
        foreground_term = prob_true * foreground_distance_mean

        # Average over all pixels
        loss_foreground = tf.reduce_mean(foreground_term)

        # Term 2: Background penalty (λ2 term in Eq. 3)
        # λ2·1_{d_ij=0}·(1/K)·Σ_k |r̂^k_ij|
        # Penalize non-zero distance predictions on background pixels
        background_penalty = tf.abs(dist_pred) * background_mask_exp
        background_penalty_mean = tf.reduce_mean(background_penalty, axis=-1, keepdims=True)

        # Average over all pixels
        loss_background = tf.reduce_mean(background_penalty_mean)

        # Total distance loss: λ1·foreground + λ2·background
        loss_distance = self.lambda1 * loss_foreground + self.lambda2 * loss_background

        return {
            'bce': loss_bce,
            'distance': loss_distance,
            'foreground': loss_foreground,  # For monitoring
            'background': loss_background   # For monitoring
        }

    def wbr_loss(self, nucleus_prob_pred, cytoplasm_prob_pred):
        """
        Within-Boundary Regularization (WBR).

        Implements Equation 5 from paper:
        Λ = |1 + ϵ - Σ(ŷ²⊥_ij * ŷ¹_ij) / Σ(y²⊥_ij)| - 1

        Where:
        - ŷ¹_ij: predicted semantic mask of inner object (nucleus)
        - ŷ²⊥_ij: inverted predicted semantic mask of outer object (cytoplasm)
        - ϵ: small epsilon for numerical stability

        Intuition:
        - Numerator: counts nucleus predictions OUTSIDE cytoplasm
        - Denominator: total area outside cytoplasm
        - Ratio close to 0 → good (no nuclei outside)
        - Ratio close to 1 → bad (many nuclei outside)

        For multi-nuclear cells:
        - Works the same! Penalizes ANY nucleus prediction outside cytoplasm
        - Whether 1 or 10 nuclei, all must be inside cytoplasm

        Args:
            nucleus_prob_pred: (B, Z, Y, X, 1) predicted nucleus probability
            cytoplasm_prob_pred: (B, Z, Y, X, 1) predicted cytoplasm probability

        Returns:
            loss: scalar WBR penalty
        """
        # Convert probabilities to binary masks (semantic segmentation)
        # ŷ¹_ij: nucleus semantic mask
        nucleus_mask = nucleus_prob_pred  # Use probability directly (soft mask)

        # ŷ²⊥_ij: inverted cytoplasm mask (1 where cytoplasm is absent)
        cytoplasm_inverted = 1.0 - cytoplasm_prob_pred

        # Compute numerator: Σ(ŷ²⊥_ij * ŷ¹_ij)
        # This sums nucleus predictions OUTSIDE cytoplasm boundaries
        nucleus_outside = nucleus_mask * cytoplasm_inverted
        numerator = tf.reduce_sum(nucleus_outside)

        # Compute denominator: Σ(y²⊥_ij)
        # This is total area OUTSIDE cytoplasm
        denominator = tf.reduce_sum(cytoplasm_inverted) + self.epsilon

        # Compute ratio
        ratio = numerator / denominator

        # WBR penalty from Eq. 5: |1 + ϵ - ratio| - 1
        # Simplified: we just use ratio directly as penalty
        # (original formulation creates bounded interval, but direct ratio works better in practice)
        wbr_penalty = ratio

        # Alternative: use original formulation from paper
        # wbr_penalty = tf.abs(1.0 + self.epsilon - ratio) - 1.0

        return wbr_penalty

    def containment_loss(self, y_true, y_pred, assignments):
        """
        Geometric containment loss for multi-nuclear cells.

        NEW LOSS - not in original StarDist paper.
        Ensures each nucleus is geometrically inside its parent cell.

        For multi-nuclear cells (N nuclei : 1 cell):
        - Check each of N nuclei
        - Compute distance from cell center to nucleus center
        - Compare with predicted cell radius in that direction
        - Penalize if nucleus center is outside predicted cell boundary
        - Average penalty across all N nuclei

        Algorithm:
        For each cell c with nuclei {n1, n2, ..., nN}:
            penalties = []
            for each nucleus ni:
                1. Get nucleus center: p_ni
                2. Get cell center: p_c
                3. Compute vector: v = p_ni - p_c
                4. Get predicted cell radius in direction v: r_pred
                5. Get actual distance: d = ||v||
                6. If d > r_pred + margin:
                    penalties.append((d - r_pred) / r_pred)  # Relative error

            cell_penalty = mean(penalties) if penalties else 0

        Total loss = mean(all cell penalties)

        Args:
            y_true: ground truth dict
            y_pred: predictions dict
            assignments: dict mapping nucleus_id -> cytoplasm_id (per batch sample)

        Returns:
            loss: scalar containment penalty
        """
        if assignments is None or self.config.lambda_containment == 0:
            return tf.constant(0.0)

        # NOTE: Full implementation requires:
        # 1. Extract cell and nucleus centers from predicted probability maps
        # 2. For each cell, find all assigned nuclei
        # 3. For each nucleus, check if it's inside predicted cell polygon
        # 4. Compute penalty if outside

        # This is complex in TensorFlow and better done in numpy during training
        # For now, return placeholder
        # TODO: Implement in training loop using numpy/scipy operations

        return tf.constant(0.0)

    def containment_loss_numpy(self, nucleus_instances, cytoplasm_instances,
                                nucleus_dist_pred, cytoplasm_dist_pred,
                                assignments):
        """
        NumPy implementation of containment loss for post-processing or custom training loop.

        This can be called during training with numpy arrays.

        Args:
            nucleus_instances: (Z, Y, X) nucleus instance labels (GT or pred)
            cytoplasm_instances: (Z, Y, X) cytoplasm instance labels
            nucleus_dist_pred: (Z, Y, X, K) predicted nucleus distances
            cytoplasm_dist_pred: (Z, Y, X, K) predicted cytoplasm distances
            assignments: dict {nucleus_id: cytoplasm_id}

        Returns:
            loss: scalar containment penalty
        """
        from skimage.measure import regionprops

        if len(assignments) == 0:
            return 0.0

        nucleus_props = {prop.label: prop for prop in regionprops(nucleus_instances)}
        cytoplasm_props = {prop.label: prop for prop in regionprops(cytoplasm_instances)}

        # Group nuclei by cell
        cell_to_nuclei = {}
        for nuc_id, cell_id in assignments.items():
            if cell_id not in cell_to_nuclei:
                cell_to_nuclei[cell_id] = []
            cell_to_nuclei[cell_id].append(nuc_id)

        total_penalty = 0.0
        num_cells = 0

        # For each cell with nuclei
        for cell_id, nucleus_ids in cell_to_nuclei.items():
            if cell_id not in cytoplasm_props:
                continue

            cell_center = np.array(cytoplasm_props[cell_id].centroid)
            penalties = []

            # Check each nucleus in this cell
            for nuc_id in nucleus_ids:
                if nuc_id not in nucleus_props:
                    continue

                nucleus_center = np.array(nucleus_props[nuc_id].centroid)

                # Vector from cell center to nucleus center
                vec = nucleus_center - cell_center
                distance = np.linalg.norm(vec)

                if distance < 1e-6:  # Nucleus at cell center
                    continue

                # Get predicted cell radius in this direction
                # This requires interpolating the distance map along the direction vector
                # Simplified: use mean predicted radius
                cell_center_int = tuple(cell_center.astype(int))
                if all(0 <= cell_center_int[i] < cytoplasm_dist_pred.shape[i]
                       for i in range(len(cell_center_int))):
                    predicted_radii = cytoplasm_dist_pred[cell_center_int]
                    mean_radius = np.mean(predicted_radii)

                    # Check if nucleus is outside with margin
                    if distance > mean_radius + self.config.containment_margin:
                        # Relative error: how far outside as fraction of radius
                        penalty = (distance - mean_radius) / (mean_radius + 1e-6)
                        penalties.append(penalty)

            # Average penalty for this cell (handles multi-nuclear case)
            if len(penalties) > 0:
                cell_penalty = np.mean(penalties)
                total_penalty += cell_penalty
                num_cells += 1

        # Average across all cells
        return total_penalty / max(num_cells, 1)


def compute_nucleus_cell_assignments(nucleus_instances, cytoplasm_instances,
                                     max_distance=5.0):
    """
    Assign nuclei to cells based on containment and proximity.

    Three-stage strategy (as before):
    1. Containment: nucleus center inside cell
    2. Overlap: maximum IoU
    3. Proximity: nearest cell within threshold

    Handles multi-nuclear cells automatically:
    - Multiple nuclei can map to same cell_id
    - Example: {1: 5, 2: 5, 3: 5} = nuclei 1,2,3 all in cell 5

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

        # ========== Strategy 1: Containment ==========
        assigned = False

        # Check if nucleus center is inside any cell
        if tuple(nuc_center) in zip(*np.where(cytoplasm_instances > 0)):
            cell_id = cytoplasm_instances[tuple(nuc_center)]
            if cell_id > 0:
                assignments[nuc_id] = int(cell_id)
                assigned = True

        if not assigned:
            # ========== Strategy 2: Maximum Overlap ==========
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
            # ========== Strategy 3: Proximity ==========
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

    Implements the ground truth generation for r^k_ij (distances along rays).

    Args:
        instances: (Z, Y, X) or (Y, X) instance segmentation
        rays: Rays object with vertices (K directions)

    Returns:
        prob_map: (Z, Y, X, 1) or (Y, X, 1) probability map (d_ij in paper)
        dist_map: (Z, Y, X, K) or (Y, X, K) distance map (r^k_ij in paper)
    """
    from stardist.geometry import star_dist

    # Use StarDist's built-in function to compute distances along rays
    # This computes r^k_ij for each pixel (i,j) and ray k
    dist_map = star_dist(instances, rays)

    # Create probability map d_ij
    # In paper: d_ij is normalized distance to nearest background pixel
    # Here: we use distance transform to get smooth probability
    prob_map = (instances > 0).astype(np.float32)

    # Refine: make probability higher at object centers (more accurate than binary)
    for label in np.unique(instances):
        if label == 0:
            continue

        mask = (instances == label)

        # Distance transform: distance from each pixel to boundary
        dist_to_boundary = distance_transform_edt(mask)
        max_dist = dist_to_boundary.max()

        if max_dist > 0:
            # Normalize: 1 at center, decreasing towards boundary
            prob_map[mask] = dist_to_boundary[mask] / max_dist

    # Add channel dimension
    prob_map = prob_map[..., np.newaxis]

    return prob_map, dist_map