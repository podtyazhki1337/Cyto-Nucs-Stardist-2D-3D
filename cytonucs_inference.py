"""
CytoNucs StarDist Inference

Post-processing to convert probability and distance maps into instance segmentations.
"""
import numpy as np
from scipy.ndimage import label, distance_transform_edt
from skimage.measure import regionprops
from stardist.geometry import polygons_to_label, polyhedron_to_label
from stardist.nms import non_maximum_suppression
import tensorflow as tf


class CytoNucsPredictor:
    """
    Inference for CytoNucs StarDist model.

    Performs:
    1. Predict probability and distance maps for nucleus and cytoplasm
    2. Non-maximum suppression to find object centers
    3. Convert star-convex polygons/polyhedra to instance labels
    4. Post-process to ensure nuclei are inside cells
    """

    def __init__(self, model, config, prob_thresh=0.5, nms_thresh=0.4):
        """
        Args:
            model: trained CytoNucsStarDistModel
            config: CytoNucsConfig
            prob_thresh: probability threshold for object detection
            nms_thresh: NMS threshold
        """
        self.model = model
        self.config = config
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.ndim = config.ndim

    def predict_instances(
            self,
            img,
            normalize_img=True,
            n_tiles=None,
            show_tile_progress=True
    ):
        """
        Predict nucleus and cytoplasm instances from image.

        Args:
            img: input image (Z, Y, X) or (Y, X)
            normalize_img: whether to normalize input
            n_tiles: tiling for large images
            show_tile_progress: show progress bar

        Returns:
            nucleus_instances: (Z, Y, X) or (Y, X) nucleus instance labels
            cytoplasm_instances: (Z, Y, X) or (Y, X) cytoplasm instance labels
            details: dict with intermediate results
        """
        from csbdeep.utils import normalize

        # Normalize
        if normalize_img:
            img = normalize(img.astype(np.float32), 1, 99.8,
                            axis=tuple(range(self.ndim)))

        # Add batch and channel dimensions
        img_input = img[np.newaxis, ..., np.newaxis]

        # Predict
        img_input_tf = tf.convert_to_tensor(img_input, dtype=tf.float32)
        y_pred = self.model.keras_model(img_input_tf, training=False)

        # Extract predictions (remove batch dimension)
        nucleus_prob = y_pred['nucleus_prob'].numpy()[0, ..., 0]
        nucleus_dist = y_pred['nucleus_dist'].numpy()[0]
        cytoplasm_prob = y_pred['cytoplasm_prob'].numpy()[0, ..., 0]
        cytoplasm_dist = y_pred['cytoplasm_dist'].numpy()[0]

        # Convert to instances
        nucleus_instances = self._prob_dist_to_instances(
            nucleus_prob, nucleus_dist, 'nucleus'
        )

        cytoplasm_instances = self._prob_dist_to_instances(
            cytoplasm_prob, cytoplasm_dist, 'cytoplasm'
        )

        # Post-process: ensure nuclei are inside cells
        nucleus_instances, cytoplasm_instances = self._postprocess_containment(
            nucleus_instances, cytoplasm_instances
        )

        details = {
            'nucleus_prob': nucleus_prob,
            'nucleus_dist': nucleus_dist,
            'cytoplasm_prob': cytoplasm_prob,
            'cytoplasm_dist': cytoplasm_dist,
        }

        return nucleus_instances, cytoplasm_instances, details

    def _prob_dist_to_instances(self, prob, dist, object_type='nucleus'):
        """
        Convert probability and distance maps to instance segmentation.

        Steps:
        1. Find local maxima in probability map (object centers)
        2. Apply NMS to remove duplicates
        3. Convert star-convex shapes to instance labels
        """
        from stardist.geometry import dist_to_coord

        # 1. Find peaks (local maxima)
        points = self._find_peaks(prob, self.prob_thresh)

        if len(points) == 0:
            return np.zeros(prob.shape, dtype=np.uint16)

        # 2. Apply NMS
        points, probes = self._non_maximum_suppression(
            points, prob, dist
        )

        if len(points) == 0:
            return np.zeros(prob.shape, dtype=np.uint16)

        # 3. Convert distances to coordinates
        coords = self._dist_to_coords(points, dist)

        # 4. Render instances
        if self.ndim == 3:
            instances = polyhedron_to_label(
                coords, prob.shape, verbose=False
            )
        else:
            instances = polygons_to_label(
                coords, prob.shape, shape=None
            )

        return instances.astype(np.uint16)

    def _find_peaks(self, prob, threshold):
        """
        Find local maxima in probability map.

        Args:
            prob: probability map
            threshold: minimum probability

        Returns:
            points: array of peak coordinates (N, ndim)
        """
        from scipy.ndimage import maximum_filter

        # Apply threshold
        prob_thresh = prob > threshold

        # Find local maxima
        footprint_size = 3
        if self.ndim == 3:
            footprint = np.ones((footprint_size, footprint_size, footprint_size))
        else:
            footprint = np.ones((footprint_size, footprint_size))

        local_max = (prob == maximum_filter(prob, footprint=footprint))
        peaks = np.logical_and(prob_thresh, local_max)

        # Get coordinates
        points = np.array(np.where(peaks)).T

        return points

    def _non_maximum_suppression(self, points, prob, dist):
        """
        Apply non-maximum suppression to remove overlapping detections.

        Args:
            points: (N, ndim) candidate points
            prob: probability map
            dist: distance map

        Returns:
            filtered_points: remaining points after NMS
            filtered_probes: probability values at those points
        """
        if len(points) == 0:
            return points, np.array([])

        # Get probability at each point
        probes = prob[tuple(points.T)]

        # Sort by probability (descending)
        sort_idx = np.argsort(probes)[::-1]
        points = points[sort_idx]
        probes = probes[sort_idx]

        # Simple NMS based on distance
        keep = []
        for i, pt in enumerate(points):
            # Check if too close to any already kept point
            if len(keep) == 0:
                keep.append(i)
                continue

            # Compute distances to kept points
            kept_points = points[keep]
            distances = np.linalg.norm(kept_points - pt, axis=1)

            # If far enough from all kept points, keep this one
            min_dist = self.nms_thresh * np.mean(dist[tuple(pt)])
            if np.all(distances > min_dist):
                keep.append(i)

        return points[keep], probes[keep]

    def _dist_to_coords(self, points, dist):
        """
        Convert distance maps to polygon/polyhedron coordinates.

        Args:
            points: (N, ndim) object centers
            dist: (H, W, n_rays) or (D, H, W, n_rays) distance map

        Returns:
            coords: list of coordinate arrays for each object
        """
        from stardist.geometry import dist_to_coord

        coords = []
        rays_vertices = self.config.rays.vertices

        for pt in points:
            # Extract distances at this point
            if self.ndim == 3:
                d = dist[pt[0], pt[1], pt[2]]
            else:
                d = dist[pt[0], pt[1]]

            # Convert to coordinates
            coord = dist_to_coord(d, rays_vertices)

            # Add center point
            coord = coord + pt[np.newaxis, :]

            coords.append(coord)

        return coords

    def _postprocess_containment(self, nucleus_instances, cytoplasm_instances):
        """
        Post-process to ensure nuclei are inside cells.

        Strategy:
        1. For each nucleus, check if it overlaps with a cell
        2. If no overlap, assign to nearest cell (if within distance threshold)
        3. If nucleus overlaps multiple cells, assign to cell with max overlap
        4. Optionally: remove nuclei that can't be assigned

        Args:
            nucleus_instances: nucleus instance labels
            cytoplasm_instances: cytoplasm instance labels

        Returns:
            nucleus_instances_refined: refined nucleus labels
            cytoplasm_instances_refined: refined cytoplasm labels
        """
        from cytonucs_losses import compute_nucleus_cell_assignments

        # Compute assignments
        assignments, unassigned = compute_nucleus_cell_assignments(
            nucleus_instances,
            cytoplasm_instances,
            max_distance=self.config.max_assignment_distance
        )

        # For debugging/validation
        print(f"  Assigned {len(assignments)} nuclei to cells")
        if len(unassigned) > 0:
            print(f"  Warning: {len(unassigned)} nuclei could not be assigned")

        # Optional: remove unassigned nuclei
        nucleus_instances_refined = nucleus_instances.copy()
        for nuc_id in unassigned:
            nucleus_instances_refined[nucleus_instances == nuc_id] = 0

        return nucleus_instances_refined, cytoplasm_instances

    def optimize_thresholds(self, val_images, val_nucleus_masks, val_cytoplasm_masks,
                            prob_thresh_range=[0.3, 0.4, 0.5, 0.6, 0.7],
                            nms_thresh_range=[0.3, 0.4, 0.5, 0.6]):
        """
        Optimize probability and NMS thresholds on validation set.

        Args:
            val_images: list of validation images
            val_nucleus_masks: list of ground truth nucleus masks
            val_cytoplasm_masks: list of ground truth cytoplasm masks
            prob_thresh_range: list of prob thresholds to try
            nms_thresh_range: list of NMS thresholds to try

        Returns:
            best_params: dict with optimized thresholds
        """
        from itertools import product

        best_score = -np.inf
        best_params = None

        print("Optimizing thresholds...")

        for prob_thresh, nms_thresh in product(prob_thresh_range, nms_thresh_range):
            # Update thresholds
            self.prob_thresh = prob_thresh
            self.nms_thresh = nms_thresh

            # Evaluate on validation set
            scores = []
            for img, nuc_gt, cyto_gt in zip(val_images, val_nucleus_masks, val_cytoplasm_masks):
                nuc_pred, cyto_pred, _ = self.predict_instances(img, normalize_img=False)

                # Compute metrics
                from cytonucs_trainer import compute_dice
                dice_nuc = compute_dice(nuc_gt, nuc_pred)
                dice_cyto = compute_dice(cyto_gt, cyto_pred)
                score = (dice_nuc + dice_cyto) / 2

                scores.append(score)

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = {'prob_thresh': prob_thresh, 'nms_thresh': nms_thresh}

        # Set best thresholds
        self.prob_thresh = best_params['prob_thresh']
        self.nms_thresh = best_params['nms_thresh']

        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")

        return best_params


def compute_jtpr(nucleus_pred, cytoplasm_pred, nucleus_gt, cytoplasm_gt, iou_thresh=0.5):
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
    assignments_pred, _ = compute_nucleus_cell_assignments(nucleus_pred, cytoplasm_pred)
    assignments_gt, _ = compute_nucleus_cell_assignments(nucleus_gt, cytoplasm_gt)

    # Match nuclei
    nucleus_matches = match_instances(nucleus_pred, nucleus_gt, iou_thresh)

    # Match cells
    cell_matches = match_instances(cytoplasm_pred, cytoplasm_gt, iou_thresh)

    # Count joint true positives
    joint_tp = 0
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
            joint_tp += 1

    jtpr = joint_tp / total_pairs_gt if total_pairs_gt > 0 else 0.0

    details = {
        'joint_tp': joint_tp,
        'total_pairs': total_pairs_gt,
        'nucleus_matches': len(nucleus_matches),
        'cell_matches': len(cell_matches),
    }

    return jtpr, details


def match_instances(pred, gt, iou_thresh=0.5):
    """
    Match predicted instances to ground truth based on IoU.

    Args:
        pred: predicted instance labels
        gt: ground truth instance labels
        iou_thresh: minimum IoU for matching

    Returns:
        matches: dict mapping gt_id -> pred_id
    """
    from skimage.measure import regionprops

    pred_props = regionprops(pred)
    gt_props = regionprops(gt)

    matches = {}

    for gt_prop in gt_props:
        gt_id = gt_prop.label
        gt_mask = (gt == gt_id)

        best_iou = 0
        best_pred_id = None

        for pred_prop in pred_props:
            pred_id = pred_prop.label
            pred_mask = (pred == pred_id)

            # Compute IoU
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_iou >= iou_thresh:
            matches[gt_id] = best_pred_id

    return matches


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