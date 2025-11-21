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
    from skimage.segmentation import find_boundaries
    from stardist import random_label_cmap
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import io

    # Handle 3D
    if image.ndim == 3 and image.shape[-1] not in [3, 4]:
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


def remove_overlapping_instances(instances, iou_threshold=0.3, containment_threshold=0.7):
    """
    Remove overlapping instances with improved logic:
    1. Remove small objects fully contained in larger ones
    2. Merge objects with high IoU
    3. Keep larger object when overlap is significant

    Args:
        instances: (H, W) instance mask
        iou_threshold: minimum IoU to consider overlap (default: 0.3)
        containment_threshold: if object A is 70%+ inside B, remove A (default: 0.7)

    Returns:
        cleaned_instances: instance mask without overlaps
    """
    from skimage.measure import regionprops

    if instances.max() == 0:
        return instances

    props = {p.label: p for p in regionprops(instances)}
    n_objects = len(props)

    if n_objects <= 1:
        return instances

    # ============================================================
    # STEP 1: Remove contained objects (small inside large)
    # ============================================================

    to_remove = set()

    for label_i, prop_i in props.items():
        if label_i in to_remove:
            continue

        mask_i = (instances == label_i)
        area_i = prop_i.area

        for label_j, prop_j in props.items():
            if label_i == label_j or label_j in to_remove:
                continue

            mask_j = (instances == label_j)
            area_j = prop_j.area

            # Check if i is contained in j
            intersection = np.logical_and(mask_i, mask_j).sum()

            # If small object is mostly (>70%) inside large object
            if area_i < area_j:
                containment_ratio = intersection / area_i
                if containment_ratio > containment_threshold:
                    to_remove.add(label_i)
                    break

    print(f"    Removed {len(to_remove)} contained objects")

    # ============================================================
    # STEP 2: Remove objects with high IoU overlap
    # ============================================================

    remaining_labels = [l for l in props.keys() if l not in to_remove]

    if len(remaining_labels) > 1:
        # Recompute props for remaining objects
        cleaned_temp = instances.copy()
        for label in to_remove:
            cleaned_temp[cleaned_temp == label] = 0

        props_remaining = {p.label: p for p in regionprops(cleaned_temp) if p.label > 0}

        # Compute IoU matrix
        labels_list = list(props_remaining.keys())
        n_remaining = len(labels_list)

        iou_matrix = np.zeros((n_remaining, n_remaining))

        for i, label_i in enumerate(labels_list):
            mask_i = (cleaned_temp == label_i)
            for j, label_j in enumerate(labels_list):
                if i >= j:
                    continue
                mask_j = (cleaned_temp == label_j)

                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()

                iou = intersection / union if union > 0 else 0
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou

        # Find overlapping pairs and remove smaller
        overlapping_pairs = np.argwhere(iou_matrix > iou_threshold)

        for i, j in overlapping_pairs:
            if i >= j:
                continue

            label_i = labels_list[i]
            label_j = labels_list[j]

            if label_i in to_remove or label_j in to_remove:
                continue

            # Keep larger object
            area_i = props_remaining[label_i].area
            area_j = props_remaining[label_j].area

            if area_i > area_j:
                to_remove.add(label_j)
            else:
                to_remove.add(label_i)

    # ============================================================
    # STEP 3: Create cleaned mask
    # ============================================================

    cleaned = instances.copy()
    for label in to_remove:
        cleaned[cleaned == label] = 0

    # Relabel consecutively
    unique_labels = np.unique(cleaned)
    unique_labels = unique_labels[unique_labels > 0]

    result = np.zeros_like(cleaned)
    for new_id, old_id in enumerate(unique_labels, start=1):
        result[cleaned == old_id] = new_id

    n_removed = len(to_remove)
    n_final = result.max()

    if n_removed > 0:
        print(f"    Overlap removal: {n_objects} → {n_final} (removed {n_removed})")

    return result


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
    if image.ndim == 3 and image.shape[-1] not in [3, 4]:
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
        self.rays = getattr(config, 'rays', None)
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

        print(f"Predictor initialized:")
        print(f"  prob_thresh_nucleus: {self.prob_thresh_nucleus}")
        print(f"  prob_thresh_cytoplasm: {self.prob_thresh_cytoplasm}")
        print(f"  nms_thresh: {self.nms_thresh}")

    def _predict_with_tiling(self, img, n_tiles, normalize_img):
        """
        Tiling с weighted blending на probability maps,
        затем ОДИН NMS на полном изображении - убирает все артефакты.
        """
        from csbdeep.utils import normalize
        import numpy as np

        if normalize_img:
            if img.ndim == 3 and img.shape[-1] in [3, 4]:
                img = normalize(img.astype(np.float32), 1, 99.8, axis=(0, 1))
            else:
                img = normalize(img.astype(np.float32), 1, 99.8,
                                axis=tuple(range(self.ndim)))

        patch_size = self.config.train_patch_size
        img_shape = img.shape[:self.ndim]

        # БОЛЬШОЙ overlap для гладкого blending
        overlap = tuple(p // 2 for p in patch_size)  # 50%
        print(f"  Tiling: {n_tiles} tiles, patch_size={patch_size}, overlap={overlap}")

        # Накапливаем PROBABILITY MAPS, не instances!
        nucleus_prob_full = np.zeros(img_shape, dtype=np.float32)
        nucleus_dist_full = np.zeros(img_shape + (self.n_rays,), dtype=np.float32)
        cytoplasm_prob_full = np.zeros(img_shape, dtype=np.float32)
        cytoplasm_dist_full = np.zeros(img_shape + (self.n_rays,), dtype=np.float32)
        weight_map = np.zeros(img_shape, dtype=np.float32)

        # Весовая функция (gaussian)
        weight_patch = self._create_gaussian_weight(patch_size)

        # ============================================================
        # ГЕНЕРАЦИЯ ПОЗИЦИЙ ТАЙЛОВ
        # ============================================================
        step = tuple(p - o for p, o in zip(patch_size, overlap))

        if self.ndim == 2:
            y_positions = list(range(0, img_shape[0] - patch_size[0] + 1, step[0]))
            if len(y_positions) == 0 or y_positions[-1] + patch_size[0] < img_shape[0]:
                y_positions.append(max(0, img_shape[0] - patch_size[0]))

            x_positions = list(range(0, img_shape[1] - patch_size[1] + 1, step[1]))
            if len(x_positions) == 0 or x_positions[-1] + patch_size[1] < img_shape[1]:
                x_positions.append(max(0, img_shape[1] - patch_size[1]))

            tile_positions = [(y, x) for y in y_positions for x in x_positions]

            print(f"    Y positions: {y_positions}")
            print(f"    X positions: {x_positions}")

        else:  # 3D
            z_positions = list(range(0, img_shape[0] - patch_size[0] + 1, step[0]))
            if len(z_positions) == 0 or z_positions[-1] + patch_size[0] < img_shape[0]:
                z_positions.append(max(0, img_shape[0] - patch_size[0]))

            y_positions = list(range(0, img_shape[1] - patch_size[1] + 1, step[1]))
            if len(y_positions) == 0 or y_positions[-1] + patch_size[1] < img_shape[1]:
                y_positions.append(max(0, img_shape[1] - patch_size[1]))

            x_positions = list(range(0, img_shape[2] - patch_size[2] + 1, step[2]))
            if len(x_positions) == 0 or x_positions[-1] + patch_size[2] < img_shape[2]:
                x_positions.append(max(0, img_shape[2] - patch_size[2]))

            tile_positions = [(z, y, x) for z in z_positions for y in y_positions for x in x_positions]

        print(f"    Total tiles: {len(tile_positions)}")

        # ============================================================
        # ОБРАБОТКА ТАЙЛОВ - НАКАПЛИВАЕМ PROBABILITY MAPS
        # ============================================================
        total_tiles = len(tile_positions)

        for tile_idx, tile_pos in enumerate(tile_positions, start=1):
            if self.ndim == 2:
                y_start, x_start = tile_pos
                y_end = min(y_start + patch_size[0], img_shape[0])
                x_end = min(x_start + patch_size[1], img_shape[1])

                if img.ndim == 3:
                    patch = img[y_start:y_end, x_start:x_end, :]
                else:
                    patch = img[y_start:y_end, x_start:x_end]

                actual_h, actual_w = patch.shape[:2]
                if actual_h < patch_size[0] or actual_w < patch_size[1]:
                    if patch.ndim == 3:
                        patch_padded = np.zeros((patch_size[0], patch_size[1], patch.shape[2]), dtype=patch.dtype)
                        patch_padded[:actual_h, :actual_w, :] = patch
                    else:
                        patch_padded = np.zeros(patch_size, dtype=patch.dtype)
                        patch_padded[:actual_h, :actual_w] = patch
                    patch = patch_padded

                slices_out = (slice(y_start, y_end), slice(x_start, x_end))
                slices_patch = (slice(0, y_end - y_start), slice(0, x_end - x_start))

            else:  # 3D
                z_start, y_start, x_start = tile_pos
                z_end = min(z_start + patch_size[0], img_shape[0])
                y_end = min(y_start + patch_size[1], img_shape[1])
                x_end = min(x_start + patch_size[2], img_shape[2])

                if img.ndim == 4:
                    patch = img[z_start:z_end, y_start:y_end, x_start:x_end, :]
                else:
                    patch = img[z_start:z_end, y_start:y_end, x_start:x_end]

                actual_d, actual_h, actual_w = patch.shape[:3]
                if actual_d < patch_size[0] or actual_h < patch_size[1] or actual_w < patch_size[2]:
                    if patch.ndim == 4:
                        patch_padded = np.zeros((patch_size[0], patch_size[1], patch_size[2], patch.shape[3]),
                                                dtype=patch.dtype)
                        patch_padded[:actual_d, :actual_h, :actual_w, :] = patch
                    else:
                        patch_padded = np.zeros(patch_size, dtype=patch.dtype)
                        patch_padded[:actual_d, :actual_h, :actual_w] = patch
                    patch = patch_padded

                slices_out = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
                slices_patch = (slice(0, z_end - z_start), slice(0, y_end - y_start), slice(0, x_end - x_start))

            # Forward pass
            if patch.ndim == self.ndim:
                patch_input = patch[np.newaxis, ..., np.newaxis]
            elif patch.ndim == self.ndim + 1:
                patch_input = patch[np.newaxis, ...]
            else:
                raise ValueError(f"Unexpected patch shape: {patch.shape}")

            import tensorflow as tf
            patch_input_tf = tf.convert_to_tensor(patch_input, dtype=tf.float32)

            try:
                y_pred = self.model.keras_model(patch_input_tf, training=False)
            except Exception as e:
                print(f"    ❌ Tile {tile_idx}/{total_tiles} failed: {e}")
                continue

            # Извлекаем RAW outputs (НЕ делаем NMS!)
            nuc_prob = y_pred['nucleus_prob'].numpy()[0]
            nuc_dist = y_pred['nucleus_dist'].numpy()[0]
            cyto_prob = y_pred['cytoplasm_prob'].numpy()[0]
            cyto_dist = y_pred['cytoplasm_dist'].numpy()[0]

            # Убираем channel для prob
            nuc_prob_2d = nuc_prob[..., 0] if nuc_prob.shape[-1] == 1 else nuc_prob
            cyto_prob_2d = cyto_prob[..., 0] if cyto_prob.shape[-1] == 1 else cyto_prob

            # Обрезаем до реального размера
            nuc_prob_2d = nuc_prob_2d[slices_patch]
            nuc_dist = nuc_dist[slices_patch]
            cyto_prob_2d = cyto_prob_2d[slices_patch]
            cyto_dist = cyto_dist[slices_patch]

            weight_cropped = weight_patch[slices_patch]

            # НАКАПЛИВАЕМ с весами
            nucleus_prob_full[slices_out] += nuc_prob_2d * weight_cropped
            cytoplasm_prob_full[slices_out] += cyto_prob_2d * weight_cropped

            # Distance с весами
            w_dist = weight_cropped[..., np.newaxis]
            nucleus_dist_full[slices_out] += nuc_dist * w_dist
            cytoplasm_dist_full[slices_out] += cyto_dist * w_dist

            weight_map[slices_out] += weight_cropped

            if tile_idx % 10 == 0 or tile_idx == total_tiles:
                print(f"    Processed {tile_idx}/{total_tiles} tiles")

        # ============================================================
        # НОРМАЛИЗАЦИЯ ПО ВЕСАМ
        # ============================================================
        weight_map_safe = np.maximum(weight_map, 1e-8)

        nucleus_prob_full = nucleus_prob_full / weight_map_safe
        cytoplasm_prob_full = cytoplasm_prob_full / weight_map_safe

        weight_map_dist = weight_map_safe[..., np.newaxis]
        nucleus_dist_full = nucleus_dist_full / weight_map_dist
        cytoplasm_dist_full = cytoplasm_dist_full / weight_map_dist

        print(f"  ✅ Tiling complete, shape: {nucleus_prob_full.shape}")
        print(f"  nucleus_dist: [{nucleus_dist_full.min():.1f}, {nucleus_dist_full.max():.1f}] px")
        print(f"  cytoplasm_dist: [{cytoplasm_dist_full.min():.1f}, {cytoplasm_dist_full.max():.1f}] px")

        # ============================================================
        # ТЕПЕРЬ ОДИН РАЗ NMS НА ВСЁМ ИЗОБРАЖЕНИИ
        # ============================================================
        print(f"  [nucleus] Running NMS on full image...")
        self._current_type = 'nucleus'
        nucleus_instances = self._stardist_postprocess(
            nucleus_prob_full,
            nucleus_dist_full,
            prob_thresh=self.prob_thresh_nucleus,
            nms_thresh=self.nms_thresh
        )

        print(f"  [cytoplasm] Running NMS on full image...")
        self._current_type = 'cytoplasm'
        cytoplasm_instances = self._stardist_postprocess(
            cytoplasm_prob_full,
            cytoplasm_dist_full,
            prob_thresh=self.prob_thresh_cytoplasm,
            nms_thresh=self.nms_thresh
        )

        # Enhanced NMS (удаление дубликатов)
        nucleus_instances = self._enhanced_nms_postprocess(nucleus_instances)
        cytoplasm_instances = self._enhanced_nms_postprocess(cytoplasm_instances)

        # Containment
        nucleus_instances, cytoplasm_instances = self._postprocess_containment(
            nucleus_instances, cytoplasm_instances
        )

        details = {
            'nucleus_prob': nucleus_prob_full,
            'nucleus_dist': nucleus_dist_full,
            'cytoplasm_prob': cytoplasm_prob_full,
            'cytoplasm_dist': cytoplasm_dist_full,
        }

        return nucleus_instances, cytoplasm_instances, details

    def _create_gaussian_weight(self, patch_size):
        """Создаёт гауссовскую весовую маску."""
        import numpy as np

        if self.ndim == 2:
            h, w = patch_size
            y = np.arange(h) - (h - 1) / 2.0
            x = np.arange(w) - (w - 1) / 2.0
            yy, xx = np.meshgrid(y, x, indexing='ij')

            sigma_y = h / 4.0
            sigma_x = w / 4.0

            weight = np.exp(-(yy ** 2 / (2 * sigma_y ** 2) + xx ** 2 / (2 * sigma_x ** 2)))
            weight = weight / weight.max()

        else:  # 3D
            d, h, w = patch_size
            z = np.arange(d) - (d - 1) / 2.0
            y = np.arange(h) - (h - 1) / 2.0
            x = np.arange(w) - (w - 1) / 2.0
            zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

            sigma_z = d / 4.0
            sigma_y = h / 4.0
            sigma_x = w / 4.0

            weight = np.exp(-(zz ** 2 / (2 * sigma_z ** 2) +
                              yy ** 2 / (2 * sigma_y ** 2) +
                              xx ** 2 / (2 * sigma_x ** 2)))
            weight = weight / weight.max()

        return weight.astype(np.float32)


    def _enhanced_nms_postprocess(self, labels):
        """
        Дополнительный NMS для удаления дубликатов клеток на границах тайлов.
        """
        from skimage.measure import regionprops
        from scipy.spatial.distance import cdist
        import numpy as np

        if labels.max() == 0:
            return labels

        props = regionprops(labels)
        centroids = np.array([p.centroid for p in props])
        areas = np.array([p.area for p in props])
        label_ids = np.array([p.label for p in props])

        # Вычисляем расстояния между центроидами
        distances = cdist(centroids, centroids)

        # Порог: если центроиды ближе чем 30% от среднего радиуса
        mean_radius = np.sqrt(areas / np.pi).mean()
        threshold = mean_radius * 0.3

        # Находим пары близких объектов
        close_pairs = np.where((distances < threshold) & (distances > 0))

        to_remove = set()
        for i, j in zip(*close_pairs):
            if i >= j or i in to_remove or j in to_remove:
                continue

            # Удаляем объект с меньшей площадью
            if areas[i] < areas[j]:
                to_remove.add(label_ids[i])
            else:
                to_remove.add(label_ids[j])

        # Удаляем дубликаты
        result = labels.copy()
        for label_id in to_remove:
            result[result == label_id] = 0

        if len(to_remove) > 0:
            print(f"    🔍 Enhanced NMS: removed {len(to_remove)} duplicate detections")

        return result



    def predict_instances(self, img, normalize_img=True):
        """
        Predict instances for both nuclei and cytoplasm.
        """
        from csbdeep.utils import normalize

        if normalize_img:
            img = normalize(img.astype(np.float32), 1, 99.8,
                            axis=tuple(range(self.ndim)))

        # Подготовка входа
        if img.ndim == self.ndim:
            img_input = img[np.newaxis, ..., np.newaxis]  # (1, H, W, 1) or (1, D, H, W, 1)
        else:
            img_input = img[np.newaxis, ...]  # уже есть channel dim

        img_input_tf = tf.convert_to_tensor(img_input, dtype=tf.float32)

        try:
            y_pred = self.model.keras_model(img_input_tf, training=False)
        except tf.errors.ResourceExhaustedError:
            print("  Warning: OOM - cropping to patch size")
            tile_size = self.config.train_patch_size
            slices = tuple(slice(0, min(s, p)) for s, p in zip(img.shape, tile_size))
            img_cropped = img[slices]
            img_input = img_cropped[np.newaxis, ..., np.newaxis]
            img_input_tf = tf.convert_to_tensor(img_input, dtype=tf.float32)
            y_pred = self.model.keras_model(img_input_tf, training=False)

        # === КРИТИЧНО: Правильная распаковка ===
        # Модель возвращает dict с tensor'ами
        nucleus_prob = y_pred['nucleus_prob'].numpy()[0]  # (H, W, 1) или (D, H, W, 1)
        nucleus_dist = y_pred['nucleus_dist'].numpy()[0]  # (H, W, n_rays) или (D, H, W, n_rays)
        cytoplasm_prob = y_pred['cytoplasm_prob'].numpy()[0]  # (H, W, 1) или (D, H, W, 1)
        cytoplasm_dist = y_pred['cytoplasm_dist'].numpy()[0]  # (H, W, n_rays) или (D, H, W, n_rays)

        print(f"  Model output shapes:")
        print(f"    nucleus_prob: {nucleus_prob.shape}")
        print(f"    nucleus_dist: {nucleus_dist.shape}")
        print(f"    cytoplasm_prob: {cytoplasm_prob.shape}")
        print(f"    cytoplasm_dist: {cytoplasm_dist.shape}")

        # NUCLEUS
        self._current_type = 'nucleus'
        nucleus_instances = self._stardist_postprocess(
            nucleus_prob,
            nucleus_dist,
            self.prob_thresh_nucleus,
            self.nms_thresh
        )

        # CYTOPLASM
        self._current_type = 'cytoplasm'
        cytoplasm_instances = self._stardist_postprocess(
            cytoplasm_prob,
            cytoplasm_dist,
            self.prob_thresh_cytoplasm,
            self.nms_thresh
        )

        nucleus_instances, cytoplasm_instances = self._postprocess_containment(
            nucleus_instances, cytoplasm_instances
        )

        details = {
            'nucleus_prob': nucleus_prob[..., 0] if nucleus_prob.shape[-1] == 1 else nucleus_prob,
            'nucleus_dist': nucleus_dist,
            'cytoplasm_prob': cytoplasm_prob[..., 0] if cytoplasm_prob.shape[-1] == 1 else cytoplasm_prob,
            'cytoplasm_dist': cytoplasm_dist,
        }

        return nucleus_instances, cytoplasm_instances, details

    def _stardist_postprocess(self, prob, dist, prob_thresh, nms_thresh):
        """
        Стандартный StarDist post-processing: NMS + rendering.
        ИСПРАВЛЕНО: правильная распаковка non_maximum_suppression (3 значения!)
        """
        from stardist.nms import non_maximum_suppression
        from stardist.geometry import polygons_to_label, polyhedron_to_label

        obj_type = 'nucleus' if hasattr(self, '_current_type') and self._current_type == 'nucleus' else 'cytoplasm'

        # === КРИТИЧНО: Убираем лишние размерности ===
        if prob.ndim == 3 and prob.shape[-1] == 1:
            prob = prob[..., 0]  # (H, W, 1) -> (H, W)

        # Проверяем размерности
        if self.ndim == 2:
            assert prob.ndim == 2, f"prob должен быть 2D, получено {prob.ndim}D"
            assert dist.ndim == 3, f"dist должен быть 3D, получено {dist.ndim}D"
            assert prob.shape == dist.shape[:2], f"prob.shape={prob.shape} != dist.shape[:2]={dist.shape[:2]}"

        print(f"  [{obj_type}] prob.shape={prob.shape}, dist.shape={dist.shape}")

        # === 1. NON-MAXIMUM SUPPRESSION ===
        try:
            grid = (1, 1) if self.ndim == 2 else self.grid

            # ИСПРАВЛЕНО: распаковываем 3 значения
            result = non_maximum_suppression(
                dist,  # ← ПЕРВЫЙ аргумент: distance map
                prob,  # ← ВТОРОЙ аргумент: probability map
                grid=grid,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh
            )

            if len(result) == 3:
                coord, prob_nms, points_grid = result
            elif len(result) == 2:
                # Fallback для старых версий StarDist
                coord, prob_nms = result
            else:
                raise ValueError(f"Unexpected non_maximum_suppression output: {len(result)} values")

        except Exception as e:
            print(f"  [{obj_type}] ⚠️ NMS failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(prob.shape, dtype=np.uint16)

        if len(coord) == 0:
            print(f"  [{obj_type}] No objects detected (prob_thresh={prob_thresh:.2f})")
            return np.zeros(prob.shape, dtype=np.uint16)

        print(f"  [{obj_type}] NMS: {len(coord)} peaks detected")

        # === 2. Extract distances at detected points ===
        dists = np.array([dist[tuple(c)] for c in coord])

        # === 3. RENDERING ===
        try:
            if self.ndim == 2:
                instances = polygons_to_label(
                    dists,  # (N, n_rays)
                    coord,  # (N, 2)
                    prob.shape,  # (H, W)
                    prob=prob_nms  # (N,)
                )
            else:  # 3D
                instances = polyhedron_to_label(
                    dists,
                    coord,
                    prob.shape,
                    rays=self.rays if hasattr(self, 'rays') else None,
                    prob=prob_nms
                )

            n_instances = len(np.unique(instances)) - 1
            print(f"    ✅ Rendered {n_instances} instances")

            # ============================================================
            # Post-processing NMS для удаления наложений
            # ============================================================

            if n_instances > 1:
                instances = remove_overlapping_instances(
                    instances,
                    iou_threshold=0.2,  # ← Было 0.3, теперь 0.2 (более агрессивно)
                    containment_threshold=0.6  # ← Объект 60%+ внутри другого = удалить
                )

            return instances.astype(np.uint16)

        except Exception as e:
            print(f"  [{obj_type}] ⚠️ Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(prob.shape, dtype=np.uint16)





    def _dist_to_coords(self, points, dist):
        coords = []
        rays = self.config.rays

        for pt in points:
            if self.ndim == 3:
                d = dist[pt[0], pt[1], pt[2]]
            else:
                d = dist[pt[0], pt[1]]

            # ИСПРАВЛЕНО: берём ПРАВИЛЬНЫЕ координаты
            if self.ndim == 2:
                # 2D: rays.vertices должен быть (n_rays, 2) с (Y, X)
                if rays.vertices.shape[1] == 2:
                    rays_verts = rays.vertices  # (n_rays, 2)
                else:
                    # Если случайно 3D лучи для 2D - берём последние 2 координаты (Y, X)
                    rays_verts = rays.vertices[:, -2:]
            else:  # 3D
                rays_verts = rays.vertices  # (n_rays, 3)

            coord = d[:, np.newaxis] * rays_verts
            pt_array = np.array(pt[:self.ndim], dtype=float)
            coord = coord + pt_array
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
        assignments, unassigned, stats = compute_nucleus_cell_assignments(
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
    assignments_pred, _, _ = compute_nucleus_cell_assignments(nucleus_pred, cytoplasm_pred)
    assignments_gt, _, _ = compute_nucleus_cell_assignments(nucleus_gt, cytoplasm_gt)

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
    if image.ndim == 3 and image.shape[-1] not in [3, 4]:
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
    n_cols = 5 if has_gt else 3  # Добавили колонку для GT

    fig = plt.figure(figsize=(6 * n_cols, 12))

    # Create gridspec for better layout control
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.3, wspace=0.2)

    # Handle 3D
    if image.ndim == 3 and image.shape[-1] not in [3, 4]:
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


def compute_all_metrics(image, nucleus_pred, cytoplasm_pred,
                       nucleus_gt=None, cytoplasm_gt=None, details=None):
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
        'n_nuclei': int(nucleus_pred.max()),
        'n_cells': int(cytoplasm_pred.max())
    }

    # Compute assignments
    assignments_pred, unassigned, stats = compute_nucleus_cell_assignments(
        nucleus_pred, cytoplasm_pred
    )

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
        wbr_ratio = monitor_wbr_metric(
            details['nucleus_prob'],
            details['cytoplasm_prob']
        )
        metrics['wbr_ratio'] = float(wbr_ratio)

    # Containment penalty
    if details and 'cytoplasm_dist' in details:
        containment_penalty, _ = compute_containment_penalty(
            nucleus_pred,
            cytoplasm_pred,
            details['cytoplasm_dist'],
            assignments_pred,
            containment_margin=2.0
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
            nucleus_gt, cytoplasm_gt
        )
        metrics['jtpr'] = float(jtpr)
        metrics['jtpr_details'] = jtpr_details

        # GT multi-nuclear stats
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
    model_config = CytoNucsConfig.from_json(config_path)

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
    prob_thresh = config_dict.get('inference', {}).get('prob_thresh', 0.5)
    nms_thresh = config_dict.get('inference', {}).get('nms_thresh', 0.4)

    return model, model_config, prob_thresh, nms_thresh


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
    img = imageio.volread(image_path)
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

    # Run inference
    print("  Predicting...")
    nucleus_pred, cytoplasm_pred, details = predictor.predict_instances(
        img, normalize_img=True
    )

    print(f"  Detected: {nucleus_pred.max()} nuclei, {cytoplasm_pred.max()} cells")

    # Compute metrics
    print("  Computing metrics...")
    metrics = compute_all_metrics(
        img, nucleus_pred, cytoplasm_pred,
        nucleus_gt, cytoplasm_gt, details
    )

    if save_all:
        base_name = image_path.stem

        # Save predictions
        if nucleus_pred.ndim == 2:
            imageio.imwrite(output_dir / f'{base_name}_nucleus.tif',
                            nucleus_pred.astype(np.uint16))
            imageio.imwrite(output_dir / f'{base_name}_cytoplasm.tif',
                            cytoplasm_pred.astype(np.uint16))

            # Save probability maps
            imageio.imwrite(output_dir / f'{base_name}_nucleus_prob.tif',
                            (details['nucleus_prob'] * 65535).astype(np.uint16))
            imageio.imwrite(output_dir / f'{base_name}_cytoplasm_prob.tif',
                            (details['cytoplasm_prob'] * 65535).astype(np.uint16))
        else:
            imageio.volwrite(output_dir / f'{base_name}_nucleus.tif',
                             nucleus_pred.astype(np.uint16))
            imageio.volwrite(output_dir / f'{base_name}_cytoplasm.tif',
                             cytoplasm_pred.astype(np.uint16))

            # Save probability maps
            imageio.volwrite(output_dir / f'{base_name}_nucleus_prob.tif',
                             (details['nucleus_prob'] * 65535).astype(np.uint16))
            imageio.volwrite(output_dir / f'{base_name}_cytoplasm_prob.tif',
                             (details['cytoplasm_prob'] * 65535).astype(np.uint16))

        # Create visualizations
        print("  Creating visualizations...")

        # Comprehensive figure
        fig = create_comprehensive_figure(
            img, nucleus_pred, cytoplasm_pred,
            nucleus_gt, cytoplasm_gt, metrics
        )
        fig.savefig(output_dir / f'{base_name}_complete.png',
                   dpi=150, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Simple overlay
        overlay = create_overlay(img, nucleus_pred, cytoplasm_pred,
                                nucleus_gt, cytoplasm_gt)
        imageio.imwrite(output_dir / f'{base_name}_overlay.png', overlay)

        # Save metrics
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
        model, config, prob_thresh, nms_thresh = load_model_from_config(args.config)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create predictor
    predictor = CytoNucsPredictor(model, config, prob_thresh, nms_thresh)
    print(f"Thresholds: prob={prob_thresh}, nms={nms_thresh}\n")

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