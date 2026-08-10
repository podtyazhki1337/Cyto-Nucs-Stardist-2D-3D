"""
Main Training Script for CytoNucs StarDist

Reads configuration from config_cytonucs.json
Similar logic to original train.py
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import sys
from comet_ml import Experiment
from tqdm import tqdm
from cytonucs_gpu_setup import setup_gpu, print_gpu_info
# Import CytoNucs modules
from cytonucs_config import CytoNucsConfig
from cytonucs_model import build_cytonucs_model
from cytonucs_losses import CytoNucsLoss
from cytonucs_data_generator import (
    CytoNucsDataGenerator,
    cytonucs_augmenter_3d,
    cytonucs_augmenter_2d,
    prepare_cytonucs_dataset
)
from cytonucs_trainer import CytoNucsTrainer, DiceCallback
from cytonucs_inference import CytoNucsPredictor, compute_jtpr


def load_config(config_path='config_cytonucs.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_inference_only(config_dict, config_path):
    """
    Pure inference mode — loads full volumes, normalizes axes, runs tiled inference,
    dumps full-resolution prediction/GT masks as TIFF, computes Dice/AP/JTPR.
    Универсально для 2D/3D, grayscale и RGB-стеков (Z,Y,X,C).
    """
    import imageio.v2 as imageio
    from pathlib import Path
    from csbdeep.utils import normalize
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
    print("✅ Model loaded")

    # ========== Loaders / helpers ==========
    def _load_volume(path):
        """Читает том и нормализует порядок осей RGB-стека к (Z, Y, X, C)."""
        arr = np.asarray(imageio.volread(str(path)))
        if ndim == 3 and arr.ndim == 4 and arr.shape[-1] not in (3, 4) and arr.shape[1] in (3, 4):
            arr = np.moveaxis(arr, 1, -1)  # (Z,C,Y,X) -> (Z,Y,X,C)
        return arr

    def _to_model_input(img):
        """(spatial...) или (spatial..., C) -> нормализованный (spatial..., C)."""
        if img.ndim == ndim:
            img = img[..., np.newaxis]
        img = normalize(img.astype(np.float32), 1, 99.8, axis=tuple(range(ndim)))
        return img

    def _semantic_dice(gt, pred):
        """Бинарный Dice (любой объект vs фон) — без инстансного матчинга."""
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
        return sorted(files)

    image_files, nucleus_mask_files, cytoplasm_mask_files = [], [], []
    for split in ('test', 'val'):
        d = Path(config_dict['data']['dataset_dir']) / split
        if d.exists():
            start = len(image_files)
            image_files.extend(find_all_images(d / 'images'))
            nucleus_mask_files.extend(find_all_images(d / 'nucleus_masks'))
            cytoplasm_mask_files.extend(find_all_images(d / 'cytoplasm_masks'))
            print(f"📂 Loaded {len(image_files) - start} images from {split}")

    if len(image_files) == 0:
        print("❌ No images found in test or validation directories")
        sys.exit(1)

    assert len(image_files) == len(nucleus_mask_files) == len(cytoplasm_mask_files), \
        f"Mismatch: {len(image_files)} img, {len(nucleus_mask_files)} nuc, {len(cytoplasm_mask_files)} cyto"

    print(f"\n📊 Total images for inference: {len(image_files)}")
    print("\nINFERENCE Database (test + validation):")
    for idx, im_file in enumerate(image_files):
        source = "val" if "val" in str(im_file.parent.parent) else "test"
        print(f"{idx}\t{im_file.name}\t[{source}]")

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

    dataset_sources = []
    if any("test" in str(f.parent.parent) for f in image_files):
        dataset_sources.append("test")
    if any("val" in str(f.parent.parent) for f in image_files):
        dataset_sources.append("val")
    dataset_name = "_".join(dataset_sources) if dataset_sources else "inference"

    viz_dir = Path(model.basedir) / model.name / f'inference_{dataset_name}'
    viz_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = viz_dir / 'pred_masks'
    masks_dir.mkdir(parents=True, exist_ok=True)

    save_tiff = config_dict.get('inference', {}).get('save_pred_tiff', True)
    mask_axes = 'ZYX' if ndim == 3 else 'YX'

    def _save_mask(arr, path):
        tifffile.imwrite(str(path), np.asarray(arr).astype(np.uint16),
                         metadata={'axes': mask_axes})

    # один раз, а не на каждом изображении
    dice_callback = DiceCallback(model, None, eval_every=1, save_best=False)

    # ========== Inference Loop ==========
    y_true_nucleus, y_pred_nucleus = [], []
    y_true_cytoplasm, y_pred_cytoplasm = [], []
    pred_assignments = []
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

            # Z полностью контролируется внутри
            # _predict_with_tiling через fixed-Z windows.
            #
            # Здесь n_tiles относится только к XY.
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

        # Выравнивание форм pred/GT
        if nucleus_pred.shape != nucleus_gt.shape:
            print(f"  ⚠️  Shape mismatch: pred={nucleus_pred.shape}, gt={nucleus_gt.shape}")
            min_shape = tuple(min(p, g) for p, g in zip(nucleus_pred.shape, nucleus_gt.shape))
            sl = tuple(slice(0, s) for s in min_shape)
            nucleus_pred, cytoplasm_pred = nucleus_pred[sl], cytoplasm_pred[sl]
            nucleus_gt, cytoplasm_gt = nucleus_gt[sl], cytoplasm_gt[sl]

        y_true_nucleus.append(nucleus_gt);   y_pred_nucleus.append(nucleus_pred)
        y_true_cytoplasm.append(cytoplasm_gt); y_pred_cytoplasm.append(cytoplasm_pred)
        pred_assignments.append(dict(getattr(predictor, '_prompt_assignments', {}) or {}))
        _pc = getattr(predictor, '_prompt_coords_nuc', None)
        if _pc is not None and len(_pc) > 0:
            _hi = np.array(nucleus_gt.shape) - 1
            _lab = [int(nucleus_gt[tuple(np.clip(np.asarray(p, int), 0, _hi))])
                    for p in _pc]
            from collections import Counter
            _cnt = Counter(l for l in _lab if l > 0)
            _dup = {k: v for k, v in _cnt.items() if v > 1}
            print(f"  [gt-diag] промптов {len(_lab)}: "
                  f"в фоне {sum(1 for l in _lab if l == 0)}, "
                  f"на GT-ядрах {len(_cnt)} уникальных, "
                  f"дубликатов (>1 промпта на ядро) {len(_dup)}"
                  + (f", распределение {sorted(_dup.values(), reverse=True)[:8]}" if _dup else ""))
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

    # AP в конвенции StarDist / DSB-2018 = accuracy = TP/(TP+FP+FN), среднее по
    # taus. Раньше здесь печаталась precision = TP/(TP+FP) под именем «AP@50»
    # и результат _calculate_ap (интегрирование PR-кривой по 10 точкам) под
    # именем «AP» — обе величины несравнимы с тем, что печатает DiceCallback,
    # так что val и test были в разных единицах.
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

    # Разбивка: без неё «AP@50 0.045» не отличает «не нашли» от «нашли, но IoU
    # не дотянул» — это разные проблемы с разными лечениями. Обвал accuracy
    # между 0.5 и 0.6 = систематически неверный размер масок; низкая уже на
    # 0.5 = недодетекция.
    s_n, s_c = stats_nuc[0], stats_cyto[0]
    print(f"  @IoU=0.5 nucleus:   TP={s_n.tp} FP={s_n.fp} FN={s_n.fn} "
          f"P={s_n.precision:.3f} R={s_n.recall:.3f}")
    print(f"  @IoU=0.5 cytoplasm: TP={s_c.tp} FP={s_c.fp} FN={s_c.fn} "
          f"P={s_c.precision:.3f} R={s_c.recall:.3f}")
    print(f"  accuracy(tau) nucleus:   "
          f"{' '.join(f'{t:.2f}:{a:.3f}' for t, a in zip(taus, acc_nuc))}")
    print(f"  accuracy(tau) cytoplasm: "
          f"{' '.join(f'{t:.2f}:{a:.3f}' for t, a in zip(taus, acc_cyto))}")
    # compute_dice усредняет ТОЛЬКО по совпавшим парам (IoU>0.5) и за пропуски
    # не штрафует: 6 совпадений из 61 дают ровную «Dice 0.69».
    print(f"  instance Dice посчитан по: nucleus {s_n.tp}/{s_n.tp + s_n.fn} пар, "
          f"cytoplasm {s_c.tp}/{s_c.tp + s_c.fn} пар")

    print("\n  Computing JTPR...")
    jtpr_values, joint_tp_total, total_pairs_total = [], 0, 0
    for idx_img, (nuc_pred, cyto_pred, nuc_gt, cyto_gt) in enumerate(zip(
            y_pred_nucleus, y_pred_cytoplasm, y_true_nucleus, y_true_cytoplasm)):
        _ap = pred_assignments[idx_img] if idx_img < len(pred_assignments) else None
        jtpr, d = compute_jtpr(nucleus_pred=nuc_pred, cytoplasm_pred=cyto_pred,
                               nucleus_gt=nuc_gt, cytoplasm_gt=cyto_gt, iou_thresh=0.5,
                               assignments_pred=_ap if _ap else None)
        jtpr_values.append(jtpr)
        joint_tp_total += d['joint_tp']
        total_pairs_total += d['total_pairs']
    mean_jtpr = np.mean(jtpr_values) if jtpr_values else 0.0
    global_jtpr = joint_tp_total / total_pairs_total if total_pairs_total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"INFERENCE RESULTS ({dataset_name})")
    print("=" * 60)
    print(f"  Rescale factor: {rescale_factor:.3f}")
    print(f"  Nucleus   | Dice: {mean_dice_nucleus:.4f} | AP: {ap_nucleus:.4f} (AP@50: {ap50_nucleus:.4f})")
    print(f"  Cytoplasm | Dice: {mean_dice_cytoplasm:.4f} | AP: {ap_cytoplasm:.4f} (AP@50: {ap50_cytoplasm:.4f})")
    print(f"  JTPR      | Mean: {mean_jtpr:.4f} | Global: {global_jtpr:.4f} ({joint_tp_total}/{total_pairs_total})")
    print(f"  ▸ JTPR ограничен сверху min(nuc TP, cyto TP) = "
          f"min({s_n.tp}, {s_c.tp}) из {total_pairs_total} пар")
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
    }
    with open(viz_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {viz_dir}")
    return model, predictor, results

def run_threshold_optimization(config_dict, config_path):
    """
    Подбор порогов на ПОЛНЫХ ТОМАХ через тот же путь, что и inference
    (тайлинг + промпт-ветка). Прежняя версия собирала патчи из val-генератора
    и звала predict_instances без тайлинга — то есть подбирала пороги под
    режим, которого на реальных данных не существует.
    Пороги в конфиг НЕ пишет — впишешь найденное руками.
    """
    import imageio.v2 as imageio
    from csbdeep.utils import normalize

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
        arr = np.asarray(imageio.volread(str(path)))
        if ndim == 3 and arr.ndim == 4 and arr.shape[-1] not in (3, 4) and arr.shape[1] in (3, 4):
            arr = np.moveaxis(arr, 1, -1)
        return arr

    def _to_model_input(img):
        if img.ndim == ndim:
            img = img[..., np.newaxis]
        return normalize(img.astype(np.float32), 1, 99.8, axis=tuple(range(ndim)))

    def find_all_images(path):
        files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.PNG',
                    '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            files.extend(list(path.glob(ext)))
        return sorted(files)

    # На каких сплитах подбирать. По умолчанию val: подбор на test — это
    # подгонка под тестовую выборку, и в статье так писать нельзя.
    splits = config_dict.get('evaluation', {}).get('threshold_opt_splits', ['val'])
    image_files, nuc_files, cyto_files = [], [], []
    for split in splits:
        d = Path(config_dict['data']['dataset_dir']) / split
        if d.exists():
            image_files.extend(find_all_images(d / 'images'))
            nuc_files.extend(find_all_images(d / 'nucleus_masks'))
            cyto_files.extend(find_all_images(d / 'cytoplasm_masks'))
    if len(image_files) == 0:
        print(f"❌ Нет изображений в сплитах {splits}")
        return None
    print(f"📂 Подбор на {len(image_files)} томах из {splits}")

    val_images, val_nuc, val_cyto = [], [], []
    for f_img, f_n, f_c in zip(image_files, nuc_files, cyto_files):
        assert f_img.name == f_n.name == f_c.name, f"Filename mismatch: {f_img.name}"
        val_images.append(_to_model_input(_load_volume(f_img)))
        val_nuc.append(np.asarray(imageio.volread(str(f_n))))
        val_cyto.append(np.asarray(imageio.volread(str(f_c))))
        print(f"   {f_img.name}: {val_images[-1].shape}, "
              f"GT nuc={len(np.unique(val_nuc[-1]))-1} cyto={len(np.unique(val_cyto[-1]))-1}")

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
    )

    thresh_path = (Path(config_dict['output']['basedir']) /
                   config_dict['experiment']['name'] / 'optimized_thresholds.json')
    thresh_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thresh_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"  ✅ Optimized thresholds saved to: {thresh_path}")
    print(f"  ▶ Впиши в config.inference: {best}")
    return best


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
    np.random.seed(42)
    tf.random.set_seed(42)
    # ========== MODE SELECTION ==========
    mode = config_dict.get('mode', 'train')  # По умолчанию training

    if mode == 'inference':
        print("\n🔍 MODE: INFERENCE ONLY")
        result = run_inference_only(config_dict, config_path)  # ← СОХРАНЯЕМ результат
        return result
    elif mode == 'optimize_thresholds':
        print("\n🎯 MODE: THRESHOLD OPTIMIZATION ONLY")
        result = run_threshold_optimization(config_dict, config_path)
        return result
    else:
        print("\n🎓 MODE: TRAINING")
    # ========== Initialize Experiment Tracking ==========
    experiment = None
    if config_dict['experiment']['use_comet']:
        experiment = Experiment(
            api_key=config_dict['experiment']['comet_api_key'],
            project_name=config_dict['experiment']['comet_project'],
            workspace=config_dict['experiment']['comet_workspace'],
            auto_output_logging="simple",
        )
        experiment.set_name(config_dict['experiment']['name'])

    # ========== Preprocessing (if needed) ==========
    if config_dict['preprocessing']['run_preprocessing']:
        print("\nRunning preprocessing...")
        prepare_cytonucs_dataset(
            raw_images_dir=config_dict['preprocessing']['raw_images_dir'],
            raw_nucleus_masks_dir=config_dict['preprocessing']['raw_nucleus_masks_dir'],
            raw_cytoplasm_masks_dir=config_dict['preprocessing']['raw_cytoplasm_masks_dir'],
            output_dir=config_dict['preprocessing']['output_dir'],
            validation_images=config_dict['data'].get('validation_images', []),
            test_split=config_dict['preprocessing'].get('test_split', 0.1),
            val_split=config_dict['preprocessing'].get('val_split', 0.1),
            compute_assignments=config_dict['preprocessing'].get('compute_assignments', True),
            seed=config_dict['preprocessing'].get('seed', 42),
            train_patch_size=tuple(config_dict['training']['train_patch_size']),
            ndim=config_dict['model']['ndim']
        )

    # ========== Create Model Configuration ==========
    model_config = CytoNucsConfig.from_json(config_path)

    print("\nModel Configuration:")
    print(model_config)

    if experiment:
        experiment.log_parameters({
            'n_rays': model_config.n_rays,
            'train_patch_size': model_config.train_patch_size,
            'train_batch_size': model_config.train_batch_size,
            'anisotropy': model_config.anisotropy,
            'grid': model_config.grid,

            # === ЛОГИРОВАНИЕ ВЕСОВ ЛОССОВ ===
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

    # ========== КАНОНИЧЕСКИЙ ЗАМЕР АНИЗОТРОПИИ ==========
    # StarDist 3D выводит anisotropy из МЕДИАННЫХ BBOX ОБЪЕКТОВ, а не из шага
    # между срезами (examples/3D/2_training.ipynb). От неё зависят и лучи, и
    # grid. Делается ПОСЛЕ создания генераторов, потому что нужны загруженные
    # маски, и ДО build_cytonucs_model, потому что модель читает гриды.
    model_config.set_anisotropy_from_labels(
        [s['nucleus_mask'] for s in train_generator.data_pool],
        [s['cytoplasm_mask'] for s in train_generator.data_pool],
        update_grid=True)
    for _gen in (train_generator, val_generator):
        _gen._nuc_grid = tuple(int(g) for g in model_config.nucleus_grid)
        _gen._cyto_grid = tuple(int(g) for g in model_config.cytoplasm_grid)
        _gen._nuc_rays = model_config.rays_nucleus
        _gen._cyto_rays = model_config.rays_cytoplasm
        _gen._nuc_aniso = model_config.rays_anisotropy_nucleus
        _gen._cyto_aniso = model_config.rays_anisotropy_cytoplasm
    print(f"  ✓ Гриды генераторов обновлены: nucleus={train_generator._nuc_grid}, "
          f"cytoplasm={train_generator._cyto_grid}")
    # ============================================================
    # === COMPUTE MEDIAN RADIUS FOR ADAPTIVE FOOTPRINT ===
    # ============================================================

    # print("\n📊 Computing training statistics for adaptive footprint...")
    #
    # all_nuc_radii = []
    # all_cyto_radii = []
    #
    # num_samples = min(10, len(train_generator.data_pool))
    #
    # for i in range(num_samples):
    #     sample = train_generator.data_pool[i]
    #
    #     from skimage.measure import regionprops
    #     from scipy.ndimage import distance_transform_edt
    #
    #     # === NUCLEUS STATISTICS ===
    #     nuc_mask = sample['nucleus_mask']
    #     nuc_props = regionprops(nuc_mask)
    #
    #     for prop in nuc_props:
    #         # Метод 1: Через area (быстро, всегда работает)
    #         if model_config.ndim == 2:
    #             radius_from_area = np.sqrt(prop.area / np.pi)
    #         else:
    #             radius_from_area = (prop.area * 3 / (4 * np.pi)) ** (1 / 3)
    #
    #         # КРИТИЧНО: Пропускаем слишком маленькие объекты
    #         if radius_from_area < 3.0:
    #             continue
    #
    #         # Метод 2: Через StarDist distances (точно, но может fail)
    #         obj_mask = (nuc_mask == prop.label).astype(np.int32)
    #
    #         try:
    #             # Вызываем StarDist для получения distances
    #             if model_config.ndim == 2:
    #                 from stardist.geometry import star_dist
    #                 dist_map = star_dist(obj_mask, model_config.n_rays, mode='cpp')
    #             else:
    #                 from stardist.geometry import star_dist3D
    #                 dist_map = star_dist3D(obj_mask, model_config.rays, mode='cpp')
    #
    #             # Берём distances в центре объекта
    #             center = tuple(int(c) for c in prop.centroid)
    #
    #             # ЗАЩИТА: проверяем, что центр внутри маски
    #             if obj_mask[center] == 0:
    #                 # Центр снаружи - используем центроид маски
    #                 from scipy.ndimage import center_of_mass
    #                 center = tuple(int(c) for c in center_of_mass(obj_mask))
    #
    #             dists_at_center = dist_map[center]
    #
    #             # ЗАЩИТА: фильтруем нулевые/NaN значения
    #             valid_dists = dists_at_center[dists_at_center > 0.1]  # минимум 0.1 px
    #
    #             if len(valid_dists) >= model_config.n_rays * 0.5:  # минимум 50% лучей валидны
    #                 mean_radius = np.mean(valid_dists)
    #
    #                 # ЗАЩИТА: проверяем разумность
    #                 if 1.0 < mean_radius < 200.0:  # разумный диапазон для 2D
    #                     all_nuc_radii.append(mean_radius)
    #                 else:
    #                     # Fallback: используем area-based
    #                     all_nuc_radii.append(radius_from_area)
    #             else:
    #                 # Слишком мало валидных лучей
    #                 all_nuc_radii.append(radius_from_area)
    #
    #         except Exception as e:
    #             # StarDist failed - используем area-based
    #             print(f"    Warning: StarDist failed for nucleus {prop.label}: {e}")
    #             all_nuc_radii.append(radius_from_area)
    #
    #     # === CYTOPLASM STATISTICS (то же самое) ===
    #     cyto_mask = sample['cytoplasm_mask']
    #     cyto_props = regionprops(cyto_mask)
    #
    #     for prop in cyto_props:
    #         if model_config.ndim == 2:
    #             radius_from_area = np.sqrt(prop.area / np.pi)
    #         else:
    #             radius_from_area = (prop.area * 3 / (4 * np.pi)) ** (1 / 3)
    #
    #         # Пропускаем маленькие объекты
    #         if radius_from_area < 5.0:  # cytoplasm обычно больше
    #             continue
    #
    #         obj_mask = (cyto_mask == prop.label).astype(np.int32)
    #
    #         try:
    #             if model_config.ndim == 2:
    #                 from stardist.geometry import star_dist
    #                 dist_map = star_dist(obj_mask, model_config.n_rays, mode='cpp')
    #             else:
    #                 from stardist.geometry import star_dist3D
    #                 dist_map = star_dist3D(obj_mask, model_config.rays, mode='cpp')
    #
    #             center = tuple(int(c) for c in prop.centroid)
    #
    #             if obj_mask[center] == 0:
    #                 from scipy.ndimage import center_of_mass
    #                 center = tuple(int(c) for c in center_of_mass(obj_mask))
    #
    #             dists_at_center = dist_map[center]
    #             valid_dists = dists_at_center[dists_at_center > 0.1]
    #
    #             if len(valid_dists) >= model_config.n_rays * 0.5:
    #                 mean_radius = np.mean(valid_dists)
    #
    #                 if 2.0 < mean_radius < 500.0:  # разумный диапазон для cytoplasm
    #                     all_cyto_radii.append(mean_radius)
    #                 else:
    #                     all_cyto_radii.append(radius_from_area)
    #             else:
    #                 all_cyto_radii.append(radius_from_area)
    #
    #         except Exception as e:
    #             print(f"    Warning: StarDist failed for cytoplasm {prop.label}: {e}")
    #             all_cyto_radii.append(radius_from_area)
    #
    # # === ВЫЧИСЛЯЕМ МЕДИАНЫ с FALLBACK ===
    # if len(all_nuc_radii) > 0:
    #     nuc_median_radius = np.median(all_nuc_radii)
    #     print(f"  ✅ Median nucleus radius: {nuc_median_radius:.1f} px (from {len(all_nuc_radii)} objects)")
    # else:
    #     nuc_median_radius = 10.0
    #     print(f"  ⚠️  No valid nucleus objects, using fallback: {nuc_median_radius} px")
    #
    # if len(all_cyto_radii) > 0:
    #     cyto_median_radius = np.median(all_cyto_radii)
    #     print(f"  ✅ Median cytoplasm radius: {cyto_median_radius:.1f} px (from {len(all_cyto_radii)} objects)")
    # else:
    #     cyto_median_radius = 30.0
    #     print(f"  ⚠️  No valid cytoplasm objects, using fallback: {cyto_median_radius} px")
    #
    # # СОХРАНЯЕМ В CONFIG
    # model_config.median_max_dist_nucleus = float(nuc_median_radius)
    # model_config.median_max_dist_cytoplasm = float(cyto_median_radius)
    # model_config.median_radius_train = float(cyto_median_radius)

    # # Дополнительная статистика
    # if len(all_nuc_radii) > 0:
    #     print(f"    Nucleus radius range: [{np.min(all_nuc_radii):.1f}, {np.max(all_nuc_radii):.1f}] px")
    # if len(all_cyto_radii) > 0:
    #     print(f"    Cytoplasm radius range: [{np.min(all_cyto_radii):.1f}, {np.max(all_cyto_radii):.1f}] px")


    print(f"Steps per epoch: {model_config.train_steps_per_epoch} (from config)")

    # ========== Build Model ==========
    print("\nBuilding model...")

    model = build_cytonucs_model(
        config=model_config,
        name=config_dict['experiment']['name'],
        basedir=config_dict['output']['basedir']
    )
    # Stage 1: загрузить рабочие веса 0.85 Dice

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
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}, starting fresh")
    # ========== КОНЕЦ БЛОКА ==========
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

    class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate, min_lr=1e-6):
            super().__init__()
            self.initial_lr = tf.cast(initial_lr, tf.float32)
            self.warmup_steps = tf.cast(warmup_steps, tf.float32)
            self.decay_steps = tf.cast(decay_steps, tf.float32)
            self.decay_rate = tf.cast(decay_rate, tf.float32)
            self.min_lr = tf.cast(min_lr, tf.float32)

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            # Линейный прогрев от min_lr до initial_lr
            warmup_lr = self.min_lr + (self.initial_lr - self.min_lr) * (step / self.warmup_steps)
            # Экспоненциальное затухание
            decay_step = step - self.warmup_steps
            decay_lr = self.initial_lr * tf.pow(self.decay_rate, decay_step / self.decay_steps)
            return tf.where(step < self.warmup_steps, warmup_lr, tf.maximum(decay_lr, self.min_lr))

        def get_config(self):
            return { "initial_lr": float(self.initial_lr), "warmup_steps": float(self.warmup_steps), "decay_steps": float(self.decay_steps), "decay_rate": float(self.decay_rate), "min_lr": float(self.min_lr) }

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
        min_lr=min_lr
    )
    print(f"  ✓ LR: warmup {lr_warmup_epochs} эпох -> {initial_lr}, "
          f"decay {decay_rate} / {decay_steps} шагов, floor {min_lr}")
    gradient_clip_norm = config_dict['training']['gradient_clip_norm']
    trainer = CytoNucsTrainer(
        model=model,
        config=model_config,
        loss_fn=loss_fn,
        gradient_clip_norm=gradient_clip_norm,
        optimizer=Adam(learning_rate=schedule)
    )
    # Restore history if resuming
    if 'loaded_history' in locals():
        trainer.history = loaded_history
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
            )
        )

        callbacks.append(
            dice_callback
        )

    # ========== Training Loop ==========
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

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
    # ========== Threshold Optimization ==========

    print("\n" + "=" * 60)
    print("Optimizing thresholds...")
    print("=" * 60 + "\n")

    # Create predictor
    predictor = CytoNucsPredictor(
        model=model,
        config=model_config,
        prob_thresh_nucleus=config_dict['inference'].get('prob_thresh_nucleus', 0.3),
        prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),  # ← Проверьте эту строку!
        nms_thresh=config_dict['inference'].get('nms_thresh', 0.3)
    )

    # Collect validation data
    # ============================================================
    # COLLECT FULL VALIDATION VOLUMES
    # ============================================================
    #
    # НЕ используем val_generator[i]:
    # это train_patch_size crops.
    #
    # val_generator.data_pool содержит исходные полные,
    # уже нормализованные volumes.
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

    # ========== Threshold Optimization (опционально) ==========
    if config_dict.get('evaluation', {}).get('optimize_thresholds', False):
        print("\n" + "=" * 60)
        print("Optimizing thresholds...")
        print("=" * 60 + "\n")

        eval_cfg = config_dict['evaluation']
        best_params = predictor.optimize_thresholds(
            val_images, val_nucleus_masks, val_cytoplasm_masks,
            prob_thresh_range=eval_cfg.get('prob_thresh_range', [0.2, 0.3, 0.4, 0.5]),
            nms_thresh_range=eval_cfg.get('nms_thresh_range', [0.2, 0.3, 0.4, 0.5])
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
            prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),  # ← Проверьте эту строку!
            nms_thresh=config_dict['inference'].get('nms_thresh', 0.3)
        )

    # ========== Final Evaluation ==========
    print("\n" + "=" * 60)
    print("Final evaluation on validation set...")
    print("=" * 60 + "\n")

    dice_scores_nucleus = []
    dice_scores_cytoplasm = []
    jtpr_scores = []

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
                assignments_pred=dict(getattr(predictor, '_prompt_assignments', {}) or {}) or None)
            jtpr_scores.append(jtpr)
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
                imageio.imwrite(pred_dir / f'val_{i:03d}_nucleus.tif', nuc_pred.astype(np.uint16))
                imageio.imwrite(pred_dir / f'val_{i:03d}_cytoplasm.tif', cyto_pred.astype(np.uint16))
                imageio.imwrite(
                    pred_dir / f'val_{i:03d}_overlay_iou.png',
                    overlay.astype(np.uint8)
                )
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Оригинал
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
                imageio.volwrite(pred_dir / f'val_{i:03d}_nucleus.tif', nuc_pred.astype(np.uint16))
                imageio.volwrite(pred_dir / f'val_{i:03d}_cytoplasm.tif', cyto_pred.astype(np.uint16))

    # Print results
    print("\n" + "=" * 60)
    print("Final Validation Results:")
    print("=" * 60)
    print(f"Nucleus Dice:    {np.mean(dice_scores_nucleus):.4f} ± {np.std(dice_scores_nucleus):.4f}")
    print(f"Cytoplasm Dice:  {np.mean(dice_scores_cytoplasm):.4f} ± {np.std(dice_scores_cytoplasm):.4f}")
    if config_dict['evaluation']['compute_jtpr']:
        print(f"JTPR:            {np.mean(jtpr_scores):.4f} ± {np.std(jtpr_scores):.4f}")
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
                duration=0.5  # 0.5 секунды на кадр
            )
            print(f"✅ Created training animation: training_progress.gif")
    if config_dict['evaluation']['compute_jtpr']:
        results['jtpr_mean'] = float(np.mean(jtpr_scores))
        results['jtpr_std'] = float(np.std(jtpr_scores))
        results['jtpr_all'] = [float(x) for x in jtpr_scores]

    if 'best_params' in locals():
        results['optimized_thresholds'] = best_params

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
        _out = main(args.config)
        model, predictor, results = (_out if isinstance(_out, tuple) and len(_out) == 3
                                     else (None, None, _out))
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)