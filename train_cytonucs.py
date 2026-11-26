"""
Main Training Script for CytoNucs StarDist

Reads configuration from config_cytonucs.json
Similar logic to original train.py
"""
import numpy as np
from cytonucs_runtime import write_instance_mask
import tensorflow as tf
from pathlib import Path
import json
import sys
from tqdm import tqdm
from cytonucs_gpu_setup import setup_gpu, print_gpu_info
# Import CytoNucs modules
from cytonucs_config import CytoNucsConfig, read_config, auto_inference_keys, resolve_auto_inference
from cytonucs_model import build_cytonucs_model
from cytonucs_losses import CytoNucsLoss
from cytonucs_data_generator import (
    CytoNucsDataGenerator,
    cytonucs_augmenter_3d,
    cytonucs_augmenter_2d
)
from cytonucs_trainer import CytoNucsTrainer, DiceCallback
from cytonucs_inference import CytoNucsPredictor, compute_jtpr, load_gt_assignments
import tifffile

def load_config(config_path='config_cytonucs.json'):
    """Configuration of a run from its JSON file, normalised (cytonucs_config.read_config)."""
    return read_config(config_path)


def run_inference_only(config_dict, config_path):
    """
    Pure inference mode — loads full volumes, normalizes axes, runs tiled inference,
    dumps full-resolution prediction/GT masks as TIFF, computes Dice/AP/JTPR.
    2D and 3D, grayscale and multichannel stacks (Z, Y, X, C).
    """
    import imageio.v2 as imageio
    from pathlib import Path
    from cytonucs_runtime import normalize_percentiles as normalize
    from cytonucs_inference import CytoNucsPredictor, compute_jtpr
    from cytonucs_trainer import compute_dice, DiceCallback
    from stardist.matching import matching_dataset
    import tifffile

    # ========== GPU SETUP ==========
    use_gpu = config_dict['model'].get('use_gpu', False)
    if use_gpu:
        print("\n🔧 Configuring GPU for inference...")
        gpu_config = config_dict.get('gpu_config', {})
        gpus = setup_gpu(
            memory_limit=gpu_config.get('memory_limit', None),
            allow_growth=gpu_config.get('allow_growth', True),
            visible_devices=gpu_config.get('visible_devices', None)
        )
        print("✅ Using GPU for inference" if gpus else "⚠️  No GPU detected, using CPU")
    else:
        print("\n💻 Running inference on CPU")

    # ========== Load Model ==========
    model_config = CytoNucsConfig.from_json(config_path)
    ndim = model_config.ndim
    patch_size = model_config.train_patch_size
    checkpoint_path = Path(config_dict['inference']['checkpoint_path'])
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\n📥 Loading checkpoint: {checkpoint_path}")
    model = build_cytonucs_model(
        config=model_config,
        name=config_dict['experiment']['name'],
        basedir=config_dict['output']['basedir']
    )
    model.load_weights(checkpoint_path)
    if model_config.mask_quality_min == 'auto' and model.loaded_quality_gates is None:
        raise ValueError(f'{checkpoint_path} has no calibrated quality gates: use a checkpoint saved '
                         'at a full validation with refinement_stage_audit, or a numeric mask_quality_min')
    if ('auto' in (model_config.mask_thresh_nucleus, model_config.mask_thresh_cytoplasm)
            and model.loaded_mask_thresholds is None):
        raise ValueError(f'{checkpoint_path} has no calibrated mask thresholds: run this config with '
                         '"mode": "calibrate" (or finish training), or set numeric mask_thresh_nucleus/cytoplasm')
    print("✅ Model loaded")

    # ========== Loaders / helpers ==========
    def _load_volume(path):
        """Reads a volume; a multichannel stack is brought to the axis order (Z, Y, X, C)."""
        from cytonucs_runtime import read_model_image
        return read_model_image(path, ndim, model_config.n_channel_in)

    def _to_model_input(img):
        """(spatial...) or (spatial..., C) -> normalised (spatial..., C)."""
        if img.ndim == ndim:
            img = img[..., np.newaxis]
        img = normalize(img.astype(np.float32), 1, 99.8, axis=tuple(range(ndim)))
        return img

    def _semantic_dice(gt, pred):
        """Binary Dice (any object vs background), without instance matching."""
        g = np.asarray(gt) > 0
        p = np.asarray(pred) > 0
        if not g.any() and not p.any():
            return 1.0
        denom = g.sum() + p.sum()
        return float(2 * np.logical_and(g, p).sum() / denom) if denom else 0.0

    # ========== Load rescale factor (info only) ==========
    split_info_path = Path(config_dict['data']['dataset_dir']) / 'split_info.json'
    rescale_factor = 1.0
    if split_info_path.exists():
        with open(split_info_path, 'r') as f:
            rescale_factor = json.load(f).get('rescale_factor', 1.0)
    print(f"📐 Rescale factor from preprocessing: {rescale_factor:.3f}")

    # ========== Find images ==========
    def find_all_images(path):
        files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.PNG',
                    '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            files.extend(list(path.glob(ext)))
        return sorted(set(files))

    # Splits to run. A single string ("test") is accepted and turned into a list.
    eval_splits = config_dict.get('inference', {}).get('eval_splits', ['test', 'val'])
    if isinstance(eval_splits, str):
        eval_splits = [eval_splits]
    eval_splits = [str(s) for s in eval_splits]

    image_files, nucleus_mask_files, cytoplasm_mask_files = [], [], []
    split_of_file = []  # split of every file, for its name and the printout
    for split in eval_splits:
        d = Path(config_dict['data']['dataset_dir']) / split
        if not d.exists():
            print(f"⚠️  Split '{split}' not found: {d}")
            continue
        _im = find_all_images(d / 'images')
        _nu = find_all_images(d / 'nucleus_masks')
        _cy = find_all_images(d / 'cytoplasm_masks')
        assert len(_im) == len(_nu) == len(_cy), (
            f"{split}: {len(_im)} img, {len(_nu)} nuc, {len(_cy)} cyto")
        image_files.extend(_im)
        nucleus_mask_files.extend(_nu)
        cytoplasm_mask_files.extend(_cy)
        split_of_file.extend([split] * len(_im))
        print(f"📂 Loaded {len(_im)} images from {split}")

    if len(image_files) == 0:
        print(f"❌ No images in the splits {eval_splits} "
              f"({config_dict['data']['dataset_dir']})")
        sys.exit(1)

    assert len(image_files) == len(nucleus_mask_files) == len(cytoplasm_mask_files), \
        f"Mismatch: {len(image_files)} img, {len(nucleus_mask_files)} nuc, {len(cytoplasm_mask_files)} cyto"

    print(f"\n📊 Total images for inference: {len(image_files)}")
    print(f"\nINFERENCE Database ({', '.join(eval_splits)}):")
    for idx, im_file in enumerate(image_files):
        print(f"{idx}\t{im_file.name}\t[{split_of_file[idx]}]")

    # ========== Create Predictor ==========
    predictor = CytoNucsPredictor(
        model=model,
        config=model_config,
        prob_thresh_nucleus=config_dict['inference'].get('prob_thresh_nucleus', 0.3),
        prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),
        nms_thresh=config_dict['inference'].get('nms_thresh', 0.3)
    )
    print(f"Predictor initialized: prob_nuc={predictor.prob_thresh_nucleus}, "
          f"prob_cyto={predictor.prob_thresh_cytoplasm}, nms={predictor.nms_thresh}")


    dataset_name = "_".join(eval_splits) if eval_splits else "inference"

    viz_dir = Path(model.basedir) / model.name / f'inference_{dataset_name}'
    viz_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = viz_dir / 'pred_masks'
    masks_dir.mkdir(parents=True, exist_ok=True)

    save_tiff = config_dict.get('inference', {}).get('save_pred_tiff', True)

    def _save_mask(arr, path):
        write_instance_mask(path, arr)

    # once, not for every image
    dice_callback = DiceCallback(model, None, eval_every=1, save_best=False)

    # ========== Inference Loop ==========
    y_true_nucleus, y_pred_nucleus = [], []
    y_true_cytoplasm, y_pred_cytoplasm = [], []
    pred_assignments = []
    gt_assignments = [load_gt_assignments(f) for f in nucleus_mask_files]
    _n_gt = sum(1 for a in gt_assignments if a is not None)
    print(f"📎 Annotated nucleus -> cell pairs found for {_n_gt}/{len(gt_assignments)} volumes"
          + ("" if _n_gt == len(gt_assignments)
             else "; the others are derived by overlap"))
    for i in tqdm(range(len(image_files)), desc=f"Inference on {dataset_name}"):
        img_path = image_files[i]
        nuc_mask_path = nucleus_mask_files[i]
        cyto_mask_path = cytoplasm_mask_files[i]
        assert img_path.name == nuc_mask_path.name == cyto_mask_path.name, \
            f"Filename mismatch: {img_path.name}, {nuc_mask_path.name}, {cyto_mask_path.name}"

        img = _load_volume(img_path)
        nucleus_gt = np.asarray(imageio.volread(str(nuc_mask_path)))
        cytoplasm_gt = np.asarray(imageio.volread(str(cyto_mask_path)))

        if i == 0:
            print(f"\n  📊 First image: {img_path.name}")
            print(f"    img {img.shape} {img.dtype} range[{img.min()},{img.max()}]")
            print(f"    nuc GT {nucleus_gt.shape} "
                  f"({len(np.unique(nucleus_gt)) - 1} obj), "
                  f"cyto GT {cytoplasm_gt.shape} "
                  f"({len(np.unique(cytoplasm_gt)) - 1} obj)")
            print(f"    patch {patch_size}")

        img_normalized = _to_model_input(img)

        img_shape = (
            img_normalized.shape[:ndim]
        )

        needs_tiling = any(
            s > p
            for s, p
            in zip(
                img_shape,
                patch_size
            )
        )

        if needs_tiling:

            n_tiles = tuple(
                max(
                    1,
                    int(
                        np.ceil(
                            s / p
                        )
                    )
                )
                for s, p
                in zip(
                    img_shape,
                    patch_size
                )
            )

            # Z is handled inside _predict_with_tiling (fixed-Z windows); n_tiles here is for XY only.
            if ndim == 3:
                n_tiles = (
                    1,
                    n_tiles[1],
                    n_tiles[2]
                )

            if i == 0:
                print(
                    f"  🔲 Using tiling: "
                    f"{n_tiles}"
                )

            (
                nucleus_pred,
                cytoplasm_pred,
                details
            ) = predictor._predict_with_tiling(
                img_normalized,
                n_tiles=n_tiles,
                normalize_img=False
            )

        else:

            (
                nucleus_pred,
                cytoplasm_pred,
                details
            ) = predictor.predict_instances(
                img_normalized,
                normalize_img=False
            )

        # Prediction and ground truth brought to one shape
        if nucleus_pred.shape != nucleus_gt.shape:
            print(f"  ⚠️  Shape mismatch: pred={nucleus_pred.shape}, gt={nucleus_gt.shape}")
            min_shape = tuple(min(p, g) for p, g in zip(nucleus_pred.shape, nucleus_gt.shape))
            sl = tuple(slice(0, s) for s in min_shape)
            nucleus_pred, cytoplasm_pred = nucleus_pred[sl], cytoplasm_pred[sl]
            nucleus_gt, cytoplasm_gt = nucleus_gt[sl], cytoplasm_gt[sl]

        y_true_nucleus.append(nucleus_gt);   y_pred_nucleus.append(nucleus_pred)
        y_true_cytoplasm.append(cytoplasm_gt); y_pred_cytoplasm.append(cytoplasm_pred)
        pred_assignments.append(getattr(predictor, '_prompt_assignments', None))
        _pc = getattr(predictor, '_prompt_coords_nuc', None)
        if _pc is not None and len(_pc) > 0:
            _hi = np.array(nucleus_gt.shape) - 1
            _lab = [int(nucleus_gt[tuple(np.clip(np.asarray(p, int), 0, _hi))])
                    for p in _pc]
            from collections import Counter
            _cnt = Counter(l for l in _lab if l > 0)
            _dup = {k: v for k, v in _cnt.items() if v > 1}
            print(f"  [gt-diag] prompts {len(_lab)}: "
                  f"on background {sum(1 for l in _lab if l == 0)}, "
                  f"on {len(_cnt)} distinct GT nuclei, "
                  f"duplicates (>1 prompt per nucleus) {len(_dup)}"
                  + (f", counts {sorted(_dup.values(), reverse=True)[:8]}" if _dup else ""))
        sd_n = _semantic_dice(nucleus_gt, nucleus_pred)
        sd_c = _semantic_dice(cytoplasm_gt, cytoplasm_pred)
        print(f"  [{i}] semantic Dice  nuc={sd_n:.3f} cyto={sd_c:.3f} | "
              f"ids pred nuc={len(np.unique(nucleus_pred))-1} cyto={len(np.unique(cytoplasm_pred))-1} | "
              f"GT nuc={len(np.unique(nucleus_gt))-1} cyto={len(np.unique(cytoplasm_gt))-1}")

        if save_tiff:
            stem = img_path.stem
            _save_mask(nucleus_pred,   masks_dir / f'{stem}_nucleus_pred.tiff')
            _save_mask(cytoplasm_pred, masks_dir / f'{stem}_cytoplasm_pred.tiff')
            _save_mask(nucleus_gt,     masks_dir / f'{stem}_nucleus_gt.tiff')
            _save_mask(cytoplasm_gt,   masks_dir / f'{stem}_cytoplasm_gt.tiff')

        dice_callback._save_epoch_visualization_full(
            epoch=i, img=img,
            nuc_gt=nucleus_gt, cyto_gt=cytoplasm_gt,
            nuc_pred=nucleus_pred, cyto_pred=cytoplasm_pred,
            details=details, viz_dir=viz_dir,
            filename=f"prediction_{i:04d}_{img_path.stem}.png")

    # ========== Metrics ==========
    print("\n📊 Computing metrics...")
    mean_dice_nucleus = np.mean([compute_dice(g, p) for g, p in zip(y_true_nucleus, y_pred_nucleus)])
    mean_dice_cytoplasm = np.mean([compute_dice(g, p) for g, p in zip(y_true_cytoplasm, y_pred_cytoplasm)])

    taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    stats_nuc = matching_dataset(y_true_nucleus, y_pred_nucleus, thresh=taus, show_progress=False)
    stats_cyto = matching_dataset(y_true_cytoplasm, y_pred_cytoplasm, thresh=taus, show_progress=False)

    # AP in the StarDist / DSB-2018 convention: accuracy = TP / (TP + FP + FN), mean over the taus (as DiceCallback).
    acc_nuc = np.array([s.accuracy for s in stats_nuc])
    acc_cyto = np.array([s.accuracy for s in stats_cyto])
    precision_nucleus = np.array([s.precision for s in stats_nuc])
    recall_nucleus = np.array([s.recall for s in stats_nuc])
    precision_cytoplasm = np.array([s.precision for s in stats_cyto])
    recall_cytoplasm = np.array([s.recall for s in stats_cyto])
    ap_nucleus = float(np.mean(acc_nuc))
    ap_cytoplasm = float(np.mean(acc_cyto))
    ap50_nucleus = float(acc_nuc[0])
    ap50_cytoplasm = float(acc_cyto[0])

    # Breakdown: tells missed objects from found ones with too low an IoU. A drop of the accuracy between 0.5 and
    # 0.6 means systematically wrong mask sizes; a low accuracy already at 0.5 means missed objects.
    s_n, s_c = stats_nuc[0], stats_cyto[0]
    print(f"  @IoU=0.5 nucleus:   TP={s_n.tp} FP={s_n.fp} FN={s_n.fn} "
          f"P={s_n.precision:.3f} R={s_n.recall:.3f}")
    print(f"  @IoU=0.5 cytoplasm: TP={s_c.tp} FP={s_c.fp} FN={s_c.fn} "
          f"P={s_c.precision:.3f} R={s_c.recall:.3f}")
    print(f"  accuracy(tau) nucleus:   "
          f"{' '.join(f'{t:.2f}:{a:.3f}' for t, a in zip(taus, acc_nuc))}")
    print(f"  accuracy(tau) cytoplasm: "
          f"{' '.join(f'{t:.2f}:{a:.3f}' for t, a in zip(taus, acc_cyto))}")
    # compute_dice averages over the matched pairs (IoU > 0.5) only and does not penalise misses.
    print(f"  instance Dice over: nucleus {s_n.tp}/{s_n.tp + s_n.fn} pairs, "
          f"cytoplasm {s_c.tp}/{s_c.tp + s_c.fn} pairs")

    print("\n  Computing JTPR...")
    jtpr_values, joint_tp_total, total_pairs_total = [], 0, 0
    interior_tp, interior_pairs = 0, 0
    for idx_img, (nuc_pred, cyto_pred, nuc_gt, cyto_gt) in enumerate(zip(
            y_pred_nucleus, y_pred_cytoplasm, y_true_nucleus, y_true_cytoplasm)):
        _ap = pred_assignments[idx_img] if idx_img < len(pred_assignments) else None
        jtpr, d = compute_jtpr(nucleus_pred=nuc_pred, cytoplasm_pred=cyto_pred,
                               nucleus_gt=nuc_gt, cytoplasm_gt=cyto_gt, iou_thresh=0.5,
                               assignments_pred=_ap,
                               assignments_gt=gt_assignments[idx_img])
        jtpr_values.append(jtpr)
        joint_tp_total += d['joint_tp']
        total_pairs_total += d['total_pairs']
        interior_tp += d['interior_joint_tp']
        interior_pairs += d['interior_total_pairs']
    mean_jtpr = np.mean(jtpr_values) if jtpr_values else 0.0
    global_jtpr = joint_tp_total / total_pairs_total if total_pairs_total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"INFERENCE RESULTS ({dataset_name})")
    print("=" * 60)
    print(f"  Rescale factor: {rescale_factor:.3f}")
    print(f"  Nucleus   | Dice: {mean_dice_nucleus:.4f} | AP: {ap_nucleus:.4f} (AP@50: {ap50_nucleus:.4f})")
    print(f"  Cytoplasm | Dice: {mean_dice_cytoplasm:.4f} | AP: {ap_cytoplasm:.4f} (AP@50: {ap50_cytoplasm:.4f})")
    print(f"  JTPR      | Mean: {mean_jtpr:.4f} | Global: {global_jtpr:.4f} ({joint_tp_total}/{total_pairs_total})")
    print(f"  Matched nuclei={s_n.tp}, matched cells={s_c.tp}; one cell can support multiple pairs")
    print(f"  JTPR interior (no XY-face GT): {interior_tp}/{interior_pairs}")
    print("=" * 60)

    results = {
        'dataset': dataset_name, 'n_images': len(y_true_nucleus),
        'rescale_factor': float(rescale_factor),
        'dice_nucleus_mean': float(mean_dice_nucleus),
        'dice_cytoplasm_mean': float(mean_dice_cytoplasm),
        'ap_nucleus': float(ap_nucleus), 'ap_cytoplasm': float(ap_cytoplasm),
        'ap50_nucleus': float(ap50_nucleus), 'ap50_cytoplasm': float(ap50_cytoplasm),
        'accuracy_taus': [float(t) for t in taus],
        'accuracy_nucleus': [float(a) for a in acc_nuc],
        'accuracy_cytoplasm': [float(a) for a in acc_cyto],
        'tp_fp_fn_nucleus': [int(s_n.tp), int(s_n.fp), int(s_n.fn)],
        'tp_fp_fn_cytoplasm': [int(s_c.tp), int(s_c.fp), int(s_c.fn)],
        'precision50_nucleus': float(precision_nucleus[0]),
        'recall50_nucleus': float(recall_nucleus[0]),
        'precision50_cytoplasm': float(precision_cytoplasm[0]),
        'recall50_cytoplasm': float(recall_cytoplasm[0]),
        'jtpr_mean': float(mean_jtpr), 'jtpr_global': float(global_jtpr),
        'joint_tp': int(joint_tp_total), 'total_pairs': int(total_pairs_total),
        'checkpoint': str(checkpoint_path),
        'prob_thresh_nucleus': predictor.prob_thresh_nucleus,
        'prob_thresh_cytoplasm': predictor.prob_thresh_cytoplasm,
        'nms_thresh': predictor.nms_thresh,
        'mask_quality_min': model_config.mask_quality_min,
        'quality_gates': predictor.quality_gates,
        'mask_thresholds': predictor.mask_thresholds,
        'interior_joint_tp': int(interior_tp), 'interior_total_pairs': int(interior_pairs),
    }
    with open(viz_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {viz_dir}")
    return model, predictor, results

def save_mask_calibration(weights_path, thresholds, calibration, report_path):
    """Store calibrated mask thresholds with the checkpoint and a readable report."""
    from cytonucs_runtime import write_mask_thresholds, resolve_resume_epoch
    report = dict(thresholds=thresholds, calibration=calibration,
                  checkpoint=str(weights_path) if weights_path is not None else None)
    if weights_path is not None:
        try:
            report['checkpoint_epoch'] = resolve_resume_epoch(weights_path)
        except ValueError:
            report['checkpoint_epoch'] = None
        write_mask_thresholds(weights_path, dict(epoch=report['checkpoint_epoch'], **thresholds))
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"✅ Mask thresholds {thresholds} saved to {weights_path} and {report_path}")


def run_mask_calibration(config_dict, config_path):
    """Calibrate mask_thresh_*='auto' for inference.checkpoint_path on the validation split.

    For a run stopped before its end-of-training calibration. Writes the thresholds into
    the checkpoint (HDF5 attribute) and mask_calibration.json next to it.
    """
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_trainer import calibrate_mask_thresholds
    model_config = CytoNucsConfig.from_json(config_path)
    if 'auto' not in (model_config.mask_thresh_nucleus, model_config.mask_thresh_cytoplasm):
        raise ValueError('mode calibrate requires mask_thresh_nucleus or mask_thresh_cytoplasm = "auto"')
    checkpoint_path = Path(config_dict['inference']['checkpoint_path'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    if config_dict['model'].get('use_gpu', False):
        gpu_config = config_dict.get('gpu_config', {})
        setup_gpu(memory_limit=gpu_config.get('memory_limit', None),
                  allow_growth=gpu_config.get('allow_growth', True),
                  visible_devices=gpu_config.get('visible_devices', None))
    model = build_cytonucs_model(config=model_config, name=config_dict['experiment']['name'],
                                 basedir=config_dict['output']['basedir'])
    model.load_weights(checkpoint_path)
    val_generator = CytoNucsDataGenerator(dataset_dir=Path(config_dict['data']['dataset_dir']) / 'val',
                                          config=model_config, rays=model_config.rays, augmenter=None,
                                          shuffle=False, subset='val')
    predictor = CytoNucsPredictor(model=model, config=model_config,
                                  prob_thresh_nucleus=config_dict['inference'].get('prob_thresh_nucleus', 0.3),
                                  prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),
                                  nms_thresh=config_dict['inference'].get('nms_thresh', 0.3))
    thresholds, calibration = calibrate_mask_thresholds(predictor, val_generator.data_pool)
    save_mask_calibration(checkpoint_path, thresholds, calibration, checkpoint_path.parent / 'mask_calibration.json')
    return thresholds


def run_threshold_optimization(config_dict, config_path):
    """
    Threshold search on whole volumes through the inference path (tiling and prompt refinement). The thresholds
    are not written into the configuration; set the values found by hand.
    """
    import imageio.v2 as imageio
    from cytonucs_runtime import normalize_percentiles as normalize

    model_config = CytoNucsConfig.from_json(config_path)
    ndim = model_config.ndim
    checkpoint_path = Path(config_dict['inference']['checkpoint_path'])
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    use_gpu = config_dict['model'].get('use_gpu', False)
    if use_gpu:
        gpu_config = config_dict.get('gpu_config', {})
        setup_gpu(memory_limit=gpu_config.get('memory_limit', None),
                  allow_growth=gpu_config.get('allow_growth', True),
                  visible_devices=gpu_config.get('visible_devices', None))

    print(f"\n📥 Loading checkpoint: {checkpoint_path}")
    model = build_cytonucs_model(
        config=model_config,
        name=config_dict['experiment']['name'],
        basedir=config_dict['output']['basedir'],
    )
    model.load_weights(checkpoint_path)

    def _load_volume(path):
        from cytonucs_runtime import read_model_image
        return read_model_image(path, ndim, model_config.n_channel_in)

    def _to_model_input(img):
        if img.ndim == ndim:
            img = img[..., np.newaxis]
        return normalize(img.astype(np.float32), 1, 99.8, axis=tuple(range(ndim)))

    def find_all_images(path):
        files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.PNG',
                    '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            files.extend(list(path.glob(ext)))
        return sorted(set(files))

    # Splits to search on; val by default (a search on test would fit the test set).
    splits = config_dict.get('evaluation', {}).get('threshold_opt_splits', ['val'])
    image_files, nuc_files, cyto_files = [], [], []
    for split in splits:
        d = Path(config_dict['data']['dataset_dir']) / split
        if d.exists():
            image_files.extend(find_all_images(d / 'images'))
            nuc_files.extend(find_all_images(d / 'nucleus_masks'))
            cyto_files.extend(find_all_images(d / 'cytoplasm_masks'))
    if len(image_files) == 0:
        print(f"❌ No images in the splits {splits}")
        return None
    print(f"📂 Search on {len(image_files)} volumes of {splits}")

    val_images, val_nuc, val_cyto, val_asg = [], [], [], []
    for f_img, f_n, f_c in zip(image_files, nuc_files, cyto_files):
        assert f_img.name == f_n.name == f_c.name, f"Filename mismatch: {f_img.name}"
        val_images.append(_to_model_input(_load_volume(f_img)))
        val_nuc.append(np.asarray(imageio.volread(str(f_n))))
        val_cyto.append(np.asarray(imageio.volread(str(f_c))))
        val_asg.append(load_gt_assignments(f_n))
        print(f"   {f_img.name}: {val_images[-1].shape}, "
              f"GT nuc={len(np.unique(val_nuc[-1])) - 1} cyto={len(np.unique(val_cyto[-1])) - 1}"
              f"{'' if val_asg[-1] is None else f', GT pairs={len(val_asg[-1])}'}")

    predictor = CytoNucsPredictor(
        model=model, config=model_config,
        prob_thresh_nucleus=config_dict['inference'].get('prob_thresh_nucleus', 0.3),
        prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),
        nms_thresh=config_dict['inference'].get('nms_thresh', 0.3),
    )

    eval_cfg = config_dict.get('evaluation', {})
    best = predictor.optimize_thresholds(
        val_images, val_nuc, val_cyto,
        prob_thresh_range=eval_cfg.get('prob_thresh_range', [0.3, 0.4, 0.5, 0.6, 0.7]),
        nms_thresh_range=eval_cfg.get('nms_thresh_range', [0.15, 0.2, 0.25, 0.3, 0.4]),
        mask_thresh_range=eval_cfg.get('mask_thresh_range', [0.25, 0.35, 0.45, 0.55]),
        val_assignments=val_asg,
    )

    thresh_path = (Path(config_dict['output']['basedir']) /
                   config_dict['experiment']['name'] / 'optimized_thresholds.json')
    thresh_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thresh_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"  ✅ Optimized thresholds saved to: {thresh_path}")
    print(f"  ▶ Set in config.inference: {best}")
    return best


class MixedTrainGenerator:
    """Training batches from the main generator and, with probability training.extra_train_share, from
    generators over extra datasets (training.extra_train_dirs, e.g. synthetic data). Every batch comes
    from one source. Geometry, the train-side metric and the rest read the main generator."""

    def __init__(self, main, extras, share, seed=0):
        self.main, self.extras, self.share, self.seed = main, list(extras), float(share), int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.main)

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, int(idx), 7])
        if self.extras and rng.random() < self.share:
            return self.extras[int(rng.integers(len(self.extras)))][idx]
        return self.main[idx]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        for generator in [self.main] + self.extras:
            generator.set_epoch(epoch)

    def on_epoch_end(self):
        for generator in [self.main] + self.extras:
            generator.on_epoch_end()

    def __getattr__(self, name):
        if name == 'main':
            raise AttributeError(name)
        return getattr(self.main, name)


def make_optimizer(schedule, weight_decay=0.0):
    """Adam, or AdamW with decoupled weight decay (training.weight_decay) on kernels only."""
    if weight_decay:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=weight_decay)
        optimizer.exclude_from_weight_decay(var_names=['bias', 'gamma', 'beta'])
        return optimizer
    return tf.keras.optimizers.Adam(learning_rate=schedule)


def geometry_labels(train_generator):
    """Label images the object geometry is measured on: the training tiles; a level without any object
    there (sparse annotation can leave none) is measured on the whole training volumes of train_regions."""
    labels = {key: [s[key] for s in train_generator.data_pool] for key in ('nucleus_mask', 'cytoplasm_mask')}
    regions = getattr(train_generator, 'region_pool', None) or []
    for key, found in labels.items():
        if regions and not any(np.any(m) for m in found):
            labels[key] = [r[key] for r in regions]
            print(f'No {key} object in the training tiles: geometry measured on the train_regions volumes')
    return labels['nucleus_mask'], labels['cytoplasm_mask']


class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate,
                 min_lr=1e-6, step_offset=0, staircase=False):
        super().__init__()
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
        # The optimizer is created after load_weights and starts at iterations = 0; without this offset a resume
        # would warm up again and return the learning rate to initial_lr.
        self.step_offset = tf.cast(step_offset, tf.float32)
        self.staircase = bool(staircase)

    def __call__(self, step):
        step = tf.cast(step, tf.float32) + self.step_offset
        warmup_lr = self.min_lr + (self.initial_lr - self.min_lr) * (step / self.warmup_steps)
        decay_step = step - self.warmup_steps
        power = decay_step / self.decay_steps
        if self.staircase:
            power = tf.floor(power)
        decay_lr = self.initial_lr * tf.pow(self.decay_rate, power)
        return tf.where(step < self.warmup_steps, warmup_lr, tf.maximum(decay_lr, self.min_lr))

    def get_config(self):
        return {"initial_lr": float(self.initial_lr), "warmup_steps": float(self.warmup_steps),
                "decay_steps": float(self.decay_steps), "decay_rate": float(self.decay_rate),
                "min_lr": float(self.min_lr), "step_offset": float(self.step_offset), "staircase": self.staircase}


def main(config_path='config_cytonucs.json'):
    """
    Main training function.
    Follows the same logic as original train.py
    """
    import numpy as np
    # ========== Load Configuration ==========
    print("=" * 60)
    print("CytoNucs StarDist Training")
    print("=" * 60)

    config_dict = load_config(config_path)
    print(f"\nLoaded config from: {config_path}")
    # ========== GPU SETUP ==========
    # Configure GPU before creating any TensorFlow operations
    use_gpu = config_dict['model'].get('use_gpu', False)

    if use_gpu:
        print("\n🔧 Configuring GPU...")
        gpu_config = config_dict.get('gpu_config', {})

        gpus = setup_gpu(
            memory_limit=gpu_config.get('memory_limit', None),
            allow_growth=gpu_config.get('allow_growth', True),
            visible_devices=gpu_config.get('visible_devices', None)
        )

        if gpus:
            print_gpu_info()
        else:
            print("\n⚠️  No GPU detected, but use_gpu=True in config")
            print("   Training will continue on CPU")
    else:
        print("\n💻 Running on CPU (use_gpu=False in config)")
    # Set random seeds
    seed = int(config_dict['training'].get('seed', 42))
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # ========== MODE SELECTION ==========
    mode = config_dict.get('mode', 'train')
    # 'auto' inference settings: the values calibrated for the checkpoint (inference modes), the defaults while
    # training; training calibrates them on validation at its end.
    auto_keys = auto_inference_keys(config_dict.get('inference', {}))
    config_dict['inference'] = resolve_auto_inference(
        config_dict.get('inference', {}), config_dict['inference'].get('checkpoint_path') if mode != 'train' else None)
    if mode != 'train' and len(config_dict.get('model', {}).get('levels', [])) > 2:
        raise NotImplementedError("Three or more levels: only mode 'train' is implemented; "
                                  'predict with cytonucs_hierarchy.predict_hierarchy')

    if mode == 'inference':
        print("\n🔍 MODE: INFERENCE ONLY")
        result = run_inference_only(config_dict, config_path)
        return result
    elif mode == 'optimize_thresholds':
        print("\n🎯 MODE: THRESHOLD OPTIMIZATION ONLY")
        result = run_threshold_optimization(config_dict, config_path)
        return result
    elif mode == 'calibrate':
        print("\n🎯 MODE: MASK THRESHOLD CALIBRATION")
        return run_mask_calibration(config_dict, config_path)
    else:
        print("\n🎓 MODE: TRAINING")
    # No experiment tracking service: history.json, the logs and the run folder record a run.
    experiment = None

    # ========== Create Model Configuration ==========
    # Settings the config leaves out that the training data tell (levels, channels, sparse annotation, pool size).
    from cytonucs_data_settings import derive_data_settings
    data_settings = derive_data_settings(config_dict)
    model_config = CytoNucsConfig.from_dict(config_dict)
    model_config.data_settings = {**(model_config.data_settings or {}), **data_settings} or None
    if model_config.n_pair_channels:
        # Three or more nested levels: adjacent pairs through the one pair network.
        from cytonucs_hierarchy import train_hierarchy
        return train_hierarchy(config_dict, model_config, experiment)

    print("\nModel Configuration:")
    print(model_config)

    if experiment:
        experiment.log_parameters({
            'n_rays': model_config.n_rays,
            'train_patch_size': model_config.train_patch_size,
            'train_batch_size': model_config.train_batch_size,
            'anisotropy': model_config.anisotropy,
            'grid': model_config.grid,

            # loss weights
            'lambda_prob_nucleus': model_config.lambda_prob_nucleus,
            'lambda_dist_nucleus': model_config.lambda_dist_nucleus,
            'lambda_prob_cytoplasm': model_config.lambda_prob_cytoplasm,
            'lambda_dist_cytoplasm': model_config.lambda_dist_cytoplasm,
            'lambda_wbr': model_config.lambda_wbr,
            'lambda_containment': model_config.lambda_containment,
            'lambda_consistency': model_config.lambda_consistency,
            'enable_consistency': model_config.enable_consistency,

            # Containment sub-weights
            'w_containment_geometric': getattr(model_config, 'w_containment_geometric', 1.0),
            'w_containment_distance': getattr(model_config, 'w_containment_distance', 0.5),
            'w_center_separation': getattr(model_config, 'w_center_separation', 0.2),
        })

    # ========== Data Generators ==========
    print("\nSetting up data generators...")

    # Augmenter
    augmenter = cytonucs_augmenter_3d if model_config.ndim == 3 else cytonucs_augmenter_2d
    if not config_dict['augmentation']['enable']:
        augmenter = None

    # Training generator
    train_generator = CytoNucsDataGenerator(
        dataset_dir=Path(config_dict['data']['dataset_dir']) / 'train',
        config=model_config,
        rays=model_config.rays,
        augmenter=augmenter,
        shuffle=True,
        subset='train',
        pool_size=config_dict['data'].get('pool_size'),
        repool_freq=config_dict['data'].get('repool_freq', 10)
    )

    # Validation generator
    val_generator = CytoNucsDataGenerator(
        dataset_dir=Path(config_dict['data']['dataset_dir']) / 'val',
        config=model_config,
        rays=model_config.rays,
        augmenter=None,
        shuffle=False,
        subset='val'
    )

    print(f"Training samples: {len(train_generator.image_files)}")
    print(f"Validation samples: {len(val_generator.image_files)}")
    # training.extra_train_dirs: more training data in tile layout (e.g. synthetic), mixed into the batches.
    extra_generators = []
    if model_config.extra_train_share and model_config.extra_train_dirs:
        region_crops = model_config.train_region_crops
        model_config.train_region_crops = False                    # extra datasets are read as tiles
        try:
            extra_generators = [CytoNucsDataGenerator(dataset_dir=Path(folder) / 'train', config=model_config,
                                                      rays=model_config.rays, augmenter=augmenter, shuffle=True,
                                                      subset='train', repool_freq=config_dict['data'].get('repool_freq', 10))
                                for folder in model_config.extra_train_dirs]
        finally:
            model_config.train_region_crops = region_crops
        print(f"Extra training data, share {model_config.extra_train_share}: "
              + ', '.join(f'{folder} ({len(g.image_files)} files)' for folder, g in zip(model_config.extra_train_dirs, extra_generators)))

    # ========== Anisotropy, as StarDist measures it ==========
    # StarDist 3D derives the anisotropy from the median bounding boxes of the objects, not from the slice spacing
    # (examples/3D/2_training.ipynb); the rays and the grids depend on it. Measured after the generators are made
    # (the masks must be loaded) and before build_cytonucs_model (the model reads the grids).
    # training.init_weights (fine-tuning from another run) keeps that run's geometry like a resume does.
    resume_geometry = config_dict['training'].get('resume_from_checkpoint') or config_dict['training'].get('init_weights')
    if resume_geometry:
        # Ray channel identities must stay identical to the saved weights.
        source_geometry = Path(resume_geometry).parent / 'geometry.json'
        destination_geometry = Path(model_config.geometry_path)
        if source_geometry.resolve() != destination_geometry.resolve():
            destination_geometry.parent.mkdir(parents=True, exist_ok=True)
            destination_geometry.write_bytes(source_geometry.read_bytes())
        print('Resume: preserving checkpoint ray geometry and grids')
    else:
        model_config.set_anisotropy_from_labels(*geometry_labels(train_generator), update_grid=True)
    for _gen in [train_generator, val_generator] + extra_generators:
        _gen._nuc_grid = tuple(int(g) for g in model_config.nucleus_grid)
        _gen._cyto_grid = tuple(int(g) for g in model_config.cytoplasm_grid)
        _gen._nuc_rays = model_config.rays_nucleus
        _gen._cyto_rays = model_config.rays_cytoplasm
        _gen._nuc_aniso = model_config.rays_anisotropy_nucleus
        _gen._cyto_aniso = model_config.rays_anisotropy_cytoplasm
    print(f"  ✓ Grids of the generators: nucleus={train_generator._nuc_grid}, "
          f"cytoplasm={train_generator._cyto_grid}")
    print(f"Steps per epoch: {model_config.train_steps_per_epoch} (from config)")
    if extra_generators:
        train_generator = MixedTrainGenerator(train_generator, extra_generators, model_config.extra_train_share,
                                              seed=int(model_config.seed))

    # ========== Build Model ==========
    print("\nBuilding model...")

    model = build_cytonucs_model(
        config=model_config,
        name=config_dict['experiment']['name'],
        basedir=config_dict['output']['basedir']
    )

    print("\n=== MODEL INITIALIZATION DEBUG ===")
    for var in model.keras_model.trainable_variables:
        if 'dist_linear' in var.name or 'dist_relu' in var.name:
            w = var.numpy()
            print(f"{var.name}: shape={var.shape}")
            print(f"  mean={w.mean():.6f}, std={w.std():.6f}, range=[{w.min():.6f}, {w.max():.6f}]")
            if 'bias' in var.name:
                print(f"  zeros: {(w == 0).sum()}/{w.size}")
    # Load checkpoint if specified
    start_epoch = 0
    if config_dict['training'].get('resume_from_checkpoint'):
        checkpoint_path = Path(config_dict['training']['resume_from_checkpoint'])
        if checkpoint_path.exists():
            print(f"\n📥 Resuming from checkpoint: {checkpoint_path}")
            from cytonucs_runtime import resolve_resume_epoch
            if not Path(str(checkpoint_path) + '.state.index').exists():
                # Fail before accepting a weights-only checkpoint with unknown epoch.
                resolve_resume_epoch(checkpoint_path)
            model.load_weights(checkpoint_path)

            # Extract epoch from filename
            import re
            match = re.search(r'epoch_(\d+)', checkpoint_path.name)
            if match:
                start_epoch = int(match.group(1))
                print(f"   Starting from epoch {start_epoch + 1}")

            # Load history
            history_file = checkpoint_path.parent / 'history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    loaded_history = json.load(f)
                print(f"   Loaded history: {len(loaded_history.get('train_loss', []))} epochs")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    init_weights = config_dict['training'].get('init_weights')
    if init_weights and not config_dict['training'].get('resume_from_checkpoint'):
        if not Path(init_weights).exists():
            raise FileNotFoundError(f'init_weights not found: {init_weights}')
        if config_dict['training'].get('init_weights_partial'):
            model.load_init_partial(init_weights)
        else:
            model.load_weights(init_weights)
        # Gates and mask thresholds belong to the other run's data: full validation calibrates new ones.
        model.loaded_quality_gates = None
        model.loaded_mask_thresholds = None
        print(f'Initial weights (fine-tuning from epoch 0 with a new optimizer): {init_weights}')
    print("\nModel Summary:")
    model.summary()

    # Check FOV vs median object size
    # fov = np.array(model._axes_tile_overlap('ZYX') if model_config.ndim == 3 else model._axes_tile_overlap('YX'))
    # median_size = extents if 'extents' in locals() else np.array([96, 96])
    # print(f"\nMedian object size:     {median_size}")
    # print(f"Network field of view:  {fov}")
    # if any(np.array(median_size) > fov):
    #     print("⚠️  WARNING: median object size larger than field of view of the neural network.")

    # ========== Training Setup ==========
    print("\nSetting up training...")

    # Loss function
    loss_fn = CytoNucsLoss(model_config)

    # Trainer (with learning rate schedule)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import ExponentialDecay

    initial_lr = config_dict['training']['initial_learning_rate']
    warmup_epochs = 10
    steps_per_epoch = model_config.train_steps_per_epoch

    initial_lr = config_dict['training']['initial_learning_rate']
    steps_per_epoch = model_config.train_steps_per_epoch
    lr_warmup_epochs = config_dict['training'].get('lr_warmup_epochs', 3)
    decay_steps = config_dict['training'].get('lr_decay_steps', 10000)
    decay_rate = config_dict['training'].get('lr_decay_rate', 0.9)
    min_lr = config_dict['training'].get('min_learning_rate', 1e-7)
    schedule = WarmupExponentialDecay(
        initial_lr=initial_lr,
        warmup_steps=max(1, int(lr_warmup_epochs * steps_per_epoch)),
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        min_lr=min_lr,
        step_offset=int(start_epoch) * int(steps_per_epoch),
        staircase=config_dict['training'].get('lr_staircase', False),
    )
    print(f"  ✓ LR: warmup {lr_warmup_epochs} epochs -> {initial_lr}, "
          f"decay {decay_rate} / {decay_steps} steps, floor {min_lr}, "
          f"from step {int(start_epoch) * int(steps_per_epoch)}")
    gradient_clip_norm = config_dict['training']['gradient_clip_norm']
    trainer = CytoNucsTrainer(
        model=model,
        config=model_config,
        loss_fn=loss_fn,
        gradient_clip_norm=gradient_clip_norm,
        optimizer=make_optimizer(schedule, float(config_dict['training'].get('weight_decay', 0.0) or 0.0))
    )
    # Restore history if resuming
    resume_path = config_dict['training'].get('resume_from_checkpoint')
    if resume_path:
        from cytonucs_runtime import resolve_resume_epoch
        state_restored = trainer.restore_training_state(resume_path)
        start_epoch = resolve_resume_epoch(
            resume_path, int(trainer.state_epoch.numpy()) if state_restored else None)
        if model_config.reset_best_on_resume:
            trainer.state_selection_start.assign(start_epoch)
        # Exactly one time source: restored iterations OR weights-only epoch offset.
        schedule.step_offset = tf.constant(
            0 if state_restored else start_epoch * steps_per_epoch, tf.float32)
        trainer.current_step = (int(trainer.optimizer.iterations.numpy()) if state_restored
                                else start_epoch * steps_per_epoch)
        print(f'Resume effective: next_epoch={start_epoch + 1}, '
              f'optimizer_iterations={int(trainer.optimizer.iterations.numpy())}, '
              f'lr_step_offset={int(schedule.step_offset.numpy())}, '
              f'learning_rate={float(trainer.optimizer.learning_rate.numpy()):.9g}')
    if resume_path:
        from cytonucs_trainer import sync_object_min_area
        sync_object_min_area(loss_fn, getattr(model, 'loaded_quality_gates', None))
    if 'loaded_history' in locals():
        from cytonucs_runtime import align_training_history
        trainer.history = align_training_history(loaded_history, start_epoch)
        print(f"✓ Restored training history")

    # Callbacks
    callbacks = []

    if (
            config_dict[
                'evaluation'
            ][
                'eval_dice_every'
            ] > 0
    ):
        dice_callback = DiceCallback(
            model=model,

            val_generator=
            val_generator,

            eval_every=
            config_dict[
                'evaluation'
            ][
                'eval_dice_every'
            ],

            save_best=
            config_dict[
                'output'
            ][
                'save_best_dice'
            ],

            experiment=
            experiment,

            prob_thresh_nucleus=
            config_dict[
                'inference'
            ].get(
                'prob_thresh_nucleus',
                0.5
            ),

            prob_thresh_cytoplasm=
            config_dict[
                'inference'
            ].get(
                'prob_thresh_cytoplasm',
                0.75
            ),

            nms_thresh=
            config_dict[
                'inference'
            ].get(
                'nms_thresh',
                0.3
            ),

            train_generator=
            train_generator
        )

        dice_callback.trainer = trainer
        from cytonucs_runtime import restored_best_metric, checkpoint_jtpr_weight
        dice_callback.best_metric = (float('-inf') if resume_path and model_config.reset_best_on_resume
                                     else restored_best_metric(trainer.history, float(trainer.state_best_full.numpy()),
                                         int(trainer.state_selection_start.numpy()),
                                         checkpoint_jtpr_weight(model_config)))
        trainer.state_best_full.assign(dice_callback.best_metric)
        callbacks.append(dice_callback)

    # ========== Training Loop ==========
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    from cytonucs_runtime import save_effective_config
    save_effective_config(model, train_generator, val_generator)
    trainer.train(
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=config_dict['training']['train_epochs'],
        steps_per_epoch=model_config.train_steps_per_epoch,
        callbacks=callbacks,
        experiment=experiment,
        start_epoch=start_epoch
    )
    # ============================================================
    # RESTORE BEST FULL-VOLUME MODEL
    # ============================================================

    best_fullval_path = (
            Path(
                config_dict[
                    'output'
                ][
                    'basedir'
                ]
            )
            / config_dict[
                'experiment'
            ][
                'name'
            ]
            / 'weights_best.weights.h5'
    )

    if best_fullval_path.exists():

        print(
            f"\n📥 Loading best "
            f"FULL-VOLUME checkpoint:"
        )

        print(
            f"   {best_fullval_path}"
        )

        model.load_weights(
            best_fullval_path
        )

    else:

        print(
            "\n⚠️ No full-volume best checkpoint found. "
            "Using final epoch weights."
        )
        # Gates loaded at resume belong to another epoch; keep only a calibration
        # of these final weights.
        pending = getattr(model, 'pending_quality_gates', None)
        model.loaded_quality_gates = (
            {k: v for k, v in pending.items() if k != 'epoch'}
            if pending and pending['epoch'] == int(trainer.state_epoch.numpy()) else None)
        model.loaded_mask_thresholds = None

    # ========== MASK THRESHOLDS: calibrated once, on the final model ==========
    mask_thresholds = None
    if 'auto' in (model_config.mask_thresh_nucleus, model_config.mask_thresh_cytoplasm):
        print("\n" + "=" * 60)
        print("Calibrating mask thresholds on validation...")
        print("=" * 60 + "\n")
        from cytonucs_trainer import calibrate_mask_thresholds
        calibration_predictor = CytoNucsPredictor(
            model=model, config=model_config,
            prob_thresh_nucleus=config_dict['inference'].get('prob_thresh_nucleus', 0.3),
            prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),
            nms_thresh=config_dict['inference'].get('nms_thresh', 0.3))
        mask_thresholds, calibration = calibrate_mask_thresholds(calibration_predictor, val_generator.data_pool)
        model.loaded_mask_thresholds = mask_thresholds
        save_mask_calibration(best_fullval_path if best_fullval_path.exists() else None, mask_thresholds,
                              calibration, best_fullval_path.parent / 'mask_calibration.json')

    # ========== 'auto' INFERENCE SETTINGS: calibrated on a part of validation (time budget) ==========
    if auto_keys and val_generator.data_pool:
        print("\n" + "=" * 60)
        print(f"Calibrating {auto_keys} on validation (at most 5 minutes)...")
        print("=" * 60 + "\n")
        from cytonucs_calibration import calibrate_after_training
        calibration_predictor = CytoNucsPredictor(
            model=model, config=model_config,
            prob_thresh_nucleus=config_dict['inference']['prob_thresh_nucleus'],
            prob_thresh_cytoplasm=config_dict['inference']['prob_thresh_cytoplasm'],
            nms_thresh=config_dict['inference']['nms_thresh'])
        calibrated, _ = calibrate_after_training(
            calibration_predictor, val_generator.data_pool, auto_keys,
            weights_path=best_fullval_path if best_fullval_path.exists() else None)
        config_dict['inference'].update(calibrated)
        for key, value in calibrated.items():
            setattr(model_config, key, value)
    # ========== Threshold Optimization ==========

    print("\n" + "=" * 60)
    print("Optimizing thresholds...")
    print("=" * 60 + "\n")

    # Create predictor
    predictor = CytoNucsPredictor(
        model=model,
        config=model_config,
        prob_thresh_nucleus=config_dict['inference'].get('prob_thresh_nucleus', 0.3),
        prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),
        nms_thresh=config_dict['inference'].get('nms_thresh', 0.3)
    )

    # Collect validation data
    # ============================================================
    # COLLECT FULL VALIDATION VOLUMES
    # ============================================================
    #
    # Not val_generator[i] (crops of train_patch_size): val_generator.data_pool holds the whole volumes, already
    # normalised.
    # ============================================================

    if (
            val_generator.data_pool is None
            or len(
        val_generator.data_pool
    )
            != len(
        val_generator.image_files
    )
    ):
        raise RuntimeError(
            "Full validation requires "
            "the complete val data_pool."
        )

    val_images = []
    val_nucleus_masks = []
    val_cytoplasm_masks = []
    val_filenames = []
    # Annotated nucleus -> cell pairs, read by the generator from assignments/*.json in repool(), so that the JTPR
    # counts the annotated pairs.
    val_assignments_gt = []

    print(
        "\nLoading FULL validation volumes..."
    )

    for sample in val_generator.data_pool:
        img = np.asarray(
            sample['image'],
            dtype=np.float32
        )

        nucleus_gt = np.asarray(
            sample['nucleus_mask']
        )

        cytoplasm_gt = np.asarray(
            sample['cytoplasm_mask']
        )

        val_images.append(
            img
        )

        val_nucleus_masks.append(
            nucleus_gt
        )

        val_cytoplasm_masks.append(
            cytoplasm_gt
        )

        val_filenames.append(
            sample.get(
                'filename',
                f'val_{len(val_images):03d}'
            )
        )
        val_assignments_gt.append(
            sample.get('assignments')
        )
        print(
            f"   {val_filenames[-1]}: "
            f"{img.shape}, "
            f"GT nuc="
            f"{len(np.unique(nucleus_gt)) - 1}, "
            f"cyto="
            f"{len(np.unique(cytoplasm_gt)) - 1}"
        )

    def _predict_full_validation(
            predictor_,
            img_
    ):
        ndim_ = int(
            model_config.ndim
        )

        patch_ = tuple(
            int(p)
            for p in model_config.train_patch_size
        )

        spatial_ = tuple(
            int(s)
            for s in img_.shape[:ndim_]
        )

        needs_tiling_ = any(
            s > p
            for s, p in zip(
                spatial_,
                patch_
            )
        )

        if needs_tiling_:

            n_tiles_ = tuple(
                max(
                    1,
                    int(
                        np.ceil(
                            s / p
                        )
                    )
                )
                for s, p in zip(
                    spatial_,
                    patch_
                )
            )

            if ndim_ == 3:
                n_tiles_ = (
                    1,
                    n_tiles_[1],
                    n_tiles_[2]
                )

            return predictor_._predict_with_tiling(
                img_,
                n_tiles=n_tiles_,
                normalize_img=False
            )

        return predictor_.predict_instances(
            img_,
            normalize_img=False
        )

    # ========== Threshold Optimization (optional) ==========
    if config_dict.get('evaluation', {}).get('optimize_thresholds', False):
        print("\n" + "=" * 60)
        print("Optimizing thresholds...")
        print("=" * 60 + "\n")

        eval_cfg = config_dict['evaluation']
        best_params = predictor.optimize_thresholds(
            val_images, val_nucleus_masks, val_cytoplasm_masks,
            prob_thresh_range=eval_cfg.get('prob_thresh_range', [0.2, 0.3, 0.4, 0.5]),
            nms_thresh_range=eval_cfg.get('nms_thresh_range', [0.2, 0.3, 0.4, 0.5]),
            mask_thresh_range=eval_cfg.get('mask_thresh_range'),
            val_assignments=val_assignments_gt
        )

        if experiment:
            experiment.log_parameters(best_params)

        thresh_path = Path(config_dict['output']['basedir']) / config_dict['experiment'][
            'name'] / 'optimized_thresholds.json'
        with open(thresh_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Optimized thresholds saved to: {thresh_path}")
    else:
        # Use default predictor
        predictor = CytoNucsPredictor(
            model=model,
            config=model_config,
            prob_thresh_nucleus=config_dict['inference'].get('prob_thresh_nucleus', 0.3),
            prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),
            nms_thresh=config_dict['inference'].get('nms_thresh', 0.3)
        )

    # ========== Final Evaluation ==========
    print("\n" + "=" * 60)
    print("Final evaluation on validation set...")
    print("=" * 60 + "\n")

    dice_scores_nucleus = []
    dice_scores_cytoplasm = []
    jtpr_scores = []
    jtpr_counts = []

    # Predict on validation set
    print("Computing predictions...")
    predictions_nucleus = []
    predictions_cytoplasm = []

    for i, img in enumerate(tqdm(val_images)):
        nuc_pred, cyto_pred, details = _predict_full_validation(predictor,img)
        predictions_nucleus.append(nuc_pred)
        predictions_cytoplasm.append(cyto_pred)

        # Compute Dice
        from cytonucs_trainer import compute_dice
        dice_nuc = compute_dice(val_nucleus_masks[i], nuc_pred)
        dice_cyto = compute_dice(val_cytoplasm_masks[i], cyto_pred)

        dice_scores_nucleus.append(dice_nuc)
        dice_scores_cytoplasm.append(dice_cyto)

        # Compute JTPR if requested
        if config_dict['evaluation']['compute_jtpr']:
            jtpr, jtpr_details = compute_jtpr(
                nuc_pred, cyto_pred,
                val_nucleus_masks[i], val_cytoplasm_masks[i],
                assignments_pred=getattr(predictor, '_prompt_assignments', None),
                assignments_gt=(val_assignments_gt[i]
                                if i < len(val_assignments_gt) else None))
            jtpr_scores.append(jtpr)
            jtpr_counts.append(jtpr_details)
            print(f"    JTPR details: {jtpr_details}")
        # Save predictions if requested
        if config_dict['evaluation']['save_predictions']:
            import imageio.v2 as imageio
            pred_dir = Path(config_dict['output']['basedir']) / config_dict['experiment']['name'] / 'predictions'
            pred_dir.mkdir(parents=True, exist_ok=True)
            from cytonucs_inference import create_overlay_with_iou
            overlay = create_overlay_with_iou(
                val_images[i],
                nuc_pred, cyto_pred,
                val_nucleus_masks[i], val_cytoplasm_masks[i],
                iou_thresh=0.5
            )
            if nuc_pred.ndim == 2:
                write_instance_mask(pred_dir / f'val_{i:03d}_nucleus.tif', nuc_pred)
                write_instance_mask(pred_dir / f'val_{i:03d}_cytoplasm.tif', cyto_pred)
                imageio.imwrite(
                    pred_dir / f'val_{i:03d}_overlay_iou.png',
                    overlay.astype(np.uint8)
                )
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Image
                axes[0, 0].imshow(val_images[i], cmap='gray')
                axes[0, 0].set_title('Original')

                # GT
                axes[0, 1].imshow(val_nucleus_masks[i], cmap='tab20')
                axes[0, 1].set_title(f'GT Nucleus ({val_nucleus_masks[i].max()})')

                # Pred
                axes[0, 2].imshow(nuc_pred, cmap='tab20')
                axes[0, 2].set_title(f'Pred Nucleus ({nuc_pred.max()}) D={dice_nuc:.3f}')

                # Cytoplasm
                axes[1, 0].imshow(val_cytoplasm_masks[i], cmap='tab20')
                axes[1, 0].set_title(f'GT Cytoplasm ({val_cytoplasm_masks[i].max()})')

                axes[1, 1].imshow(cyto_pred, cmap='tab20')
                axes[1, 1].set_title(f'Pred Cytoplasm ({cyto_pred.max()}) D={dice_cyto:.3f}')

                axes[1, 2].imshow(overlay)
                axes[1, 2].set_title(f'IoU Overlay (JTPR={jtpr:.3f})')

                for ax in axes.flat:
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(pred_dir / f'val_{i:03d}_comparison.png', dpi=150)
                plt.close()
            else:
                write_instance_mask(pred_dir / f'val_{i:03d}_nucleus.tif', nuc_pred)
                write_instance_mask(pred_dir / f'val_{i:03d}_cytoplasm.tif', cyto_pred)

    from cytonucs_runtime import aggregate_joint_counts
    joint_summary = aggregate_joint_counts(jtpr_counts)
    # Print results
    print("\n" + "=" * 60)
    print("Final Validation Results:")
    print("=" * 60)
    print(f"Nucleus Dice:    {np.mean(dice_scores_nucleus):.4f} ± {np.std(dice_scores_nucleus):.4f}")
    print(f"Cytoplasm Dice:  {np.mean(dice_scores_cytoplasm):.4f} ± {np.std(dice_scores_cytoplasm):.4f}")
    if config_dict['evaluation']['compute_jtpr']:
        print(f"JTPR global:     {joint_summary['jtpr_global']:.4f} "
              f"({joint_summary['joint_tp']}/{joint_summary['total_pairs']})")
        print(f"JTPR mean/volume:{np.mean(jtpr_scores):.4f} ± {np.std(jtpr_scores):.4f}")
    print("=" * 60)

    # Log to Comet
    if experiment:
        experiment.log_metrics({
            'final_nucleus_dice_mean': np.mean(dice_scores_nucleus),
            'final_nucleus_dice_std': np.std(dice_scores_nucleus),
            'final_cytoplasm_dice_mean': np.mean(dice_scores_cytoplasm),
            'final_cytoplasm_dice_std': np.std(dice_scores_cytoplasm),
        })
        if config_dict['evaluation']['compute_jtpr']:
            experiment.log_metrics({
                'final_jtpr_mean': np.mean(jtpr_scores),
                'final_jtpr_std': np.std(jtpr_scores),
                'final_jtpr_global': joint_summary['jtpr_global'],
            })
        experiment.end()

    # Save final results
    results = {
        'dice_nucleus_mean': float(np.mean(dice_scores_nucleus)),
        'dice_nucleus_std': float(np.std(dice_scores_nucleus)),
        'dice_nucleus_all': [float(x) for x in dice_scores_nucleus],
        'dice_cytoplasm_mean': float(np.mean(dice_scores_cytoplasm)),
        'dice_cytoplasm_std': float(np.std(dice_scores_cytoplasm)),
        'dice_cytoplasm_all': [float(x) for x in dice_scores_cytoplasm],
    }
    viz_dir = Path(config_dict['output']['basedir']) / config_dict['experiment']['name'] / 'visualizations'
    if viz_dir.exists():
        import imageio.v2 as imageio
        from PIL import Image
        images = []
        target_size = None

        # Load all images and find the most common size
        img_paths = sorted(viz_dir.glob('epoch_*.png'))
        loaded_images = []
        for img_path in img_paths:
            img = imageio.imread(img_path)
            loaded_images.append((img_path, img))
            if target_size is None:
                target_size = img.shape[:2]  # (height, width)

        # Resize all images to the target size
        if len(loaded_images) > 0:
            for img_path, img in loaded_images:
                # Convert to PIL Image for resizing
                pil_img = Image.fromarray(img)
                # Resize if dimensions don't match
                if img.shape[:2] != target_size:
                    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
                    img = np.array(pil_img)

                # Ensure all images have the same number of channels
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img]*3, axis=-1)  # Convert to RGB
                elif img.shape[2] == 4:  # RGBA
                    img = img[:, :, :3]  # Remove alpha channel

                images.append(img)

            imageio.mimsave(
                viz_dir / 'training_progress.gif',
                images,
                duration=0.5  # seconds per frame
            )
            print(f"✅ Created training animation: training_progress.gif")
    if config_dict['evaluation']['compute_jtpr']:
        results['jtpr_mean'] = float(np.mean(jtpr_scores))
        results['jtpr_std'] = float(np.std(jtpr_scores))
        results['jtpr_all'] = [float(x) for x in jtpr_scores]
        results.update(joint_summary)

    if 'best_params' in locals():
        results['optimized_thresholds'] = best_params
    results['mask_thresholds'] = mask_thresholds

    results_path = Path(config_dict['output']['basedir']) / config_dict['experiment']['name'] / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Model saved to: {Path(config_dict['output']['basedir']) / config_dict['experiment']['name']}/")

    # Final FOV check
    # print(f"\nMedian object size:     {median_size}")
    # print(f"Network field of view:  {fov}")

    print("\n🎉 Training complete!")

    return model, predictor, results


if __name__ == '__main__':
    """
    Usage:
        python train_cytonucs.py [config_path]

    Example:
        python train_cytonucs.py config_cytonucs.json
    """

    import argparse

    parser = argparse.ArgumentParser(description='Train CytoNucs StarDist model')
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default='config_cytonucs.json',
        help='Path to configuration JSON file (default: config_cytonucs.json)'
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        print("\nPlease create a configuration file. Example:")
        print("  cp config_cytonucs.json my_config.json")
        print("  # Edit my_config.json with your settings")
        print("  python train_cytonucs.py my_config.json")
        sys.exit(1)

    try:
        from cytonucs_runtime import run_logged
        _out = run_logged(args.config, main)
        if isinstance(_out, tuple) and len(_out) == 3:
            model, predictor, results = _out
        else:
            model, predictor, results = None, None, _out
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
