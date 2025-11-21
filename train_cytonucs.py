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
    Pure inference mode - loads full images directly, processes with tiling if needed
    """
    import imageio
    from pathlib import Path
    from csbdeep.utils import normalize
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_trainer import compute_dice, _calculate_ap
    from stardist.matching import matching_dataset

    print("=" * 60)
    print("🔍 MODE: INFERENCE ONLY")
    print("=" * 60)

    # ========== Load Model ==========
    model_config = CytoNucsConfig.from_json(config_path)
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

    # ========== Setup Dataset - ПРЯМАЯ ЗАГРУЗКА ==========
    dataset_name = config_dict['inference'].get('dataset', 'train')
    dataset_dir = Path(config_dict['data']['dataset_dir']) / dataset_name

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # ПРЯМОЙ ПОИСК ФАЙЛОВ (без генератора!)
    def find_all_images(path):
        """Находит все изображения в разных форматах"""
        files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            files.extend(list(path.glob(ext)))
        return sorted(files)

    image_files = find_all_images(dataset_dir / 'images')
    nucleus_mask_files = find_all_images(dataset_dir / 'nucleus_masks')
    cytoplasm_mask_files = find_all_images(dataset_dir / 'cytoplasm_masks')

    if len(image_files) == 0:
        print(f"❌ No images found in {dataset_dir / 'images'}")
        sys.exit(1)

    print(f"\n📂 Dataset: {dataset_dir}")
    print(f"Found {len(image_files)} images")

    # Проверка соответствия файлов
    assert len(image_files) == len(nucleus_mask_files) == len(cytoplasm_mask_files), \
        f"Mismatch: {len(image_files)} images, {len(nucleus_mask_files)} nucleus, {len(cytoplasm_mask_files)} cytoplasm"

    print(f"\nTEST Database:")
    for idx, im_file in enumerate(image_files):
        print(f"{idx}\t{im_file.name}")

    # ========== Warm up BatchNorm ==========
    print("\n🔥 Warming up BatchNorm...")

    # Загружаем первое изображение для warmup
    first_img = imageio.volread(str(image_files[0]))

    # КРИТИЧНО: для warmup используем PATCH SIZE модели!
    patch_size = model_config.train_patch_size

    # Обрезаем до patch size
    if first_img.ndim == 3 and first_img.shape[-1] in [3, 4]:
        # RGB: (H, W, 3)
        h, w = first_img.shape[:2]
        crop_h = min(h, patch_size[0])
        crop_w = min(w, patch_size[1])
        img_crop = first_img[:crop_h, :crop_w, :]
        img_for_warmup = img_crop
    else:
        # Grayscale
        slices = tuple(slice(0, min(s, p)) for s, p in zip(first_img.shape, patch_size))
        img_crop = first_img[slices]
        img_for_warmup = img_crop[..., np.newaxis]

    # Normalize
    if img_for_warmup.ndim == 3 and img_for_warmup.shape[-1] in [3, 4]:
        img_normalized = normalize(
            img_for_warmup.astype(np.float32),
            1, 99.8,
            axis=(0, 1)
        )
    else:
        img_normalized = normalize(
            img_for_warmup.astype(np.float32),
            1, 99.8,
            axis=tuple(range(model_config.ndim))
        )

    # Warmup: несколько forward passes
    x_warmup = img_normalized[np.newaxis, ...]
    x_warmup_tf = tf.convert_to_tensor(x_warmup, dtype=tf.float32)

    print(f"  Warmup input shape: {x_warmup.shape}")

    for _ in range(5):
        _ = model.keras_model(x_warmup_tf, training=False)

    print("  ✓ BatchNorm warmed up")

    # ========== Create Predictor ==========
    predictor = CytoNucsPredictor(
        model=model,
        config=model_config,
        prob_thresh_nucleus=config_dict['inference'].get('prob_thresh_nucleus', 0.3),
        prob_thresh_cytoplasm=config_dict['inference'].get('prob_thresh_cytoplasm', 0.4),
        nms_thresh=config_dict['inference'].get('nms_thresh', 0.3)
    )

    print(f"Predictor initialized:")
    print(f"  prob_thresh_nucleus: {predictor.prob_thresh_nucleus}")
    print(f"  prob_thresh_cytoplasm: {predictor.prob_thresh_cytoplasm}")
    print(f"  nms_thresh: {predictor.nms_thresh}")

    # ========== Output Directory ==========
    viz_dir = Path(model.basedir) / model.name / f'inference_{dataset_name}'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ========== Run Inference ==========
    y_true_nucleus, y_pred_nucleus = [], []
    y_true_cytoplasm, y_pred_cytoplasm = [], []

    for i in tqdm(range(len(image_files)), desc=f"Inference on {dataset_name}"):
        # ============================================================
        # ПРЯМАЯ ЗАГРУЗКА ПОЛНЫХ ИЗОБРАЖЕНИЙ
        # ============================================================

        img_path = image_files[i]
        nuc_mask_path = nucleus_mask_files[i]
        cyto_mask_path = cytoplasm_mask_files[i]

        # Проверка совпадения имён
        assert img_path.name == nuc_mask_path.name == cyto_mask_path.name, \
            f"Filename mismatch: {img_path.name}, {nuc_mask_path.name}, {cyto_mask_path.name}"

        # Загрузка ПОЛНОГО изображения
        img = imageio.volread(str(img_path))
        nucleus_gt = imageio.volread(str(nuc_mask_path))
        cytoplasm_gt = imageio.volread(str(cyto_mask_path))

        # Debug первого изображения
        if i == 0:
            print(f"\n  📊 First image stats:")
            print(f"    File: {img_path.name}")
            print(f"    Image shape: {img.shape}, dtype: {img.dtype}")
            print(f"    Range: [{img.min()}, {img.max()}]")
            print(f"    Nucleus GT: {nucleus_gt.shape}, {nucleus_gt.max()} objects")
            print(f"    Cytoplasm GT: {cytoplasm_gt.shape}, {cytoplasm_gt.max()} objects")
            print(f"    Model patch size: {patch_size}")

            # СОХРАНЯЕМ ОРИГИНАЛЬНОЕ ИЗОБРАЖЕНИЕ для проверки
            imageio.imwrite(viz_dir / 'debug_input_raw.png', img)
            print(f"    💾 Saved raw input: {viz_dir / 'debug_input_raw.png'}")

        # ============================================================
        # ПОДГОТОВКА ДЛЯ INFERENCE
        # ============================================================

        # 1. Определяем формат
        if img.ndim == 3 and img.shape[-1] in [3, 4]:
            # 2D RGB: (H, W, 3)
            img_input = img
            is_rgb = True
        elif img.ndim == model_config.ndim:
            # Grayscale без channel dim: (H, W) или (D, H, W)
            img_input = img
            is_rgb = False
        elif img.ndim == model_config.ndim + 1:
            # Grayscale с channel dim: (H, W, 1) или (D, H, W, 1)
            img_input = img[..., 0]
            is_rgb = False
        else:
            print(f"  ⚠️  Unexpected image shape: {img.shape}, skipping")
            continue

        # 2. НОРМАЛИЗАЦИЯ (как в training!)
        if is_rgb:
            # RGB: normalize each channel separately
            img_normalized = normalize(
                img_input.astype(np.float32),
                1, 99.8,
                axis=(0, 1)  # только spatial axes
            )
        else:
            # Grayscale: normalize all spatial dimensions
            img_normalized = normalize(
                img_input.astype(np.float32),
                1, 99.8,
                axis=tuple(range(model_config.ndim))
            )

        if i == 0:
            print(f"    After normalization: [{img_normalized.min():.4f}, {img_normalized.max():.4f}]")

            # СОХРАНЯЕМ НОРМАЛИЗОВАННОЕ для проверки
            img_norm_vis = (img_normalized - img_normalized.min()) / (
                    img_normalized.max() - img_normalized.min() + 1e-8)
            img_norm_vis = (img_norm_vis * 255).astype(np.uint8)
            imageio.imwrite(viz_dir / 'debug_input_normalized.png', img_norm_vis)
            print(f"    💾 Saved normalized: {viz_dir / 'debug_input_normalized.png'}")

        # ============================================================
        # INFERENCE С TILING (если изображение больше patch size)
        # ============================================================

        img_shape = img_normalized.shape[:model_config.ndim]
        needs_tiling = any(s > p for s, p in zip(img_shape, patch_size))

        if needs_tiling:
            print(f"  🔲 Image larger than patch size, using tiling...")

            # Вычисляем количество тайлов
            n_tiles = tuple(int(np.ceil(s / p)) for s, p in zip(img_shape, patch_size))
            print(f"    Grid: {n_tiles} tiles")

            # ИСПОЛЬЗУЕМ встроенный метод predict с tiling
            nucleus_pred, cytoplasm_pred, details = predictor._predict_with_tiling(
                img_normalized,
                n_tiles=n_tiles,
                normalize_img=False
            )
        else:
            # Изображение помещается в patch size
            nucleus_pred, cytoplasm_pred, details = predictor.predict_instances(
                img_normalized,
                normalize_img=False
            )

        # Проверка размеров
        if nucleus_pred.shape != nucleus_gt.shape:
            print(f"  ⚠️  Shape mismatch: pred={nucleus_pred.shape}, gt={nucleus_gt.shape}")
            # Crop/pad if needed
            min_shape = tuple(min(p, g) for p, g in zip(nucleus_pred.shape, nucleus_gt.shape))
            slices = tuple(slice(0, s) for s in min_shape)
            nucleus_pred = nucleus_pred[slices]
            cytoplasm_pred = cytoplasm_pred[slices]
            nucleus_gt = nucleus_gt[slices]
            cytoplasm_gt = cytoplasm_gt[slices]

        y_true_nucleus.append(nucleus_gt)
        y_pred_nucleus.append(nucleus_pred)
        y_true_cytoplasm.append(cytoplasm_gt)
        y_pred_cytoplasm.append(cytoplasm_pred)

        # ============================================================
        # ВИЗУАЛИЗАЦИЯ ПЕРВОГО ИЗОБРАЖЕНИЯ
        # ============================================================


        from cytonucs_trainer import DiceCallback

        dice_callback = DiceCallback(model, None, eval_every=1, save_best=False)

            # Для визуализации используем ОРИГИНАЛЬНОЕ изображение (не нормализованное!)
        img_for_viz = img_input
        output_filename = f"prediction_{i:04d}_{img_path.stem}.png"
        dice_callback._save_epoch_visualization_full(
                epoch=i,  # используем индекс как "эпоху" для уникальности
                img=img_for_viz,
                nuc_gt=nucleus_gt,
                cyto_gt=cytoplasm_gt,
                nuc_pred=nucleus_pred,
                cyto_pred=cytoplasm_pred,
                details=details,
                viz_dir=viz_dir,
                filename=output_filename  # передаём кастомное имя
            )

        print(f"  📷 Saved visualization: {viz_dir / output_filename}")

    # ========== Compute Metrics ==========
    print("\n📊 Computing metrics...")

    # Dice (instance-based)
    mean_dice_nucleus = np.mean([compute_dice(gt, pred) for gt, pred in zip(y_true_nucleus, y_pred_nucleus)])
    mean_dice_cytoplasm = np.mean([compute_dice(gt, pred) for gt, pred in zip(y_true_cytoplasm, y_pred_cytoplasm)])

    # AP через matching_dataset
    taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    print("  Computing AP for nucleus...")
    stats_nucleus = matching_dataset(y_true_nucleus, y_pred_nucleus, thresh=taus, show_progress=False)

    print("  Computing AP for cytoplasm...")
    stats_cytoplasm = matching_dataset(y_true_cytoplasm, y_pred_cytoplasm, thresh=taus, show_progress=False)

    # КРИТИЧНО: matching_dataset возвращает словарь с ключами 'precision', 'recall', etc.
    # Или tuple - нужно проверить формат

    print(f"  DEBUG: stats_nucleus type: {type(stats_nucleus)}")
    if isinstance(stats_nucleus, dict):
        precision_nucleus = stats_nucleus['precision']
        recall_nucleus = stats_nucleus['recall']
        precision_cytoplasm = stats_cytoplasm['precision']
        recall_cytoplasm = stats_cytoplasm['recall']
    elif isinstance(stats_nucleus, tuple):
        # Tuple формат: (tp, fp, fn, precision, recall, ...)
        print(f"  DEBUG: stats_nucleus length: {len(stats_nucleus)}")
        print(f"  DEBUG: stats_nucleus[3] (precision): {stats_nucleus[3]}")
        print(f"  DEBUG: stats_nucleus[4] (recall): {stats_nucleus[4]}")

        precision_nucleus = stats_nucleus[3]
        recall_nucleus = stats_nucleus[4]
        precision_cytoplasm = stats_cytoplasm[3]
        recall_cytoplasm = stats_cytoplasm[4]
    else:
        print(f"  ⚠️  Unexpected stats format: {type(stats_nucleus)}")
        precision_nucleus = [0.0]
        recall_nucleus = [0.0]
        precision_cytoplasm = [0.0]
        recall_cytoplasm = [0.0]

    # Конвертируем в numpy arrays если это не так
    precision_nucleus = np.array(precision_nucleus)
    recall_nucleus = np.array(recall_nucleus)
    precision_cytoplasm = np.array(precision_cytoplasm)
    recall_cytoplasm = np.array(recall_cytoplasm)

    print(f"  Precision nucleus shape: {precision_nucleus.shape}, values: {precision_nucleus}")
    print(f"  Recall nucleus shape: {recall_nucleus.shape}, values: {recall_nucleus}")

    # AP - average precision
    ap_nucleus = _calculate_ap(precision_nucleus, recall_nucleus)
    ap_cytoplasm = _calculate_ap(precision_cytoplasm, recall_cytoplasm)

    def safe_float(val, default=0.0):
        """Безопасное преобразование в float"""
        if val is None:
            return default
        if isinstance(val, str):
            return default
        if isinstance(val, (list, np.ndarray)):
            if len(val) == 0:
                return default
            # Берём первый элемент
            return safe_float(val[0], default)
        try:
            return float(val)
        except:
            return default

    # AP@50 - precision at IoU=0.5 (первый threshold)
    ap50_nucleus = safe_float(precision_nucleus[0] if len(precision_nucleus) > 0 else 0.0)
    ap50_cytoplasm = safe_float(precision_cytoplasm[0] if len(precision_cytoplasm) > 0 else 0.0)

    print(f"\n  Final AP values:")
    print(f"    Nucleus: AP={safe_float(ap_nucleus):.4f}, AP@50={ap50_nucleus:.4f}")
    print(f"    Cytoplasm: AP={safe_float(ap_cytoplasm):.4f}, AP@50={ap50_cytoplasm:.4f}")

    # ============================================================
    # COMPUTE JTPR (Joint True Positive Rate)
    # ============================================================
    print("\n  Computing JTPR (Joint True Positive Rate)...")

    from cytonucs_inference import compute_jtpr

    jtpr_values = []
    joint_tp_total = 0
    total_pairs_total = 0

    for nuc_pred, cyto_pred, nuc_gt, cyto_gt in zip(
            y_pred_nucleus, y_pred_cytoplasm, y_true_nucleus, y_true_cytoplasm
    ):
        jtpr, details = compute_jtpr(
            nucleus_pred=nuc_pred,
            cytoplasm_pred=cyto_pred,
            nucleus_gt=nuc_gt,
            cytoplasm_gt=cyto_gt,
            iou_thresh=0.5
        )
        jtpr_values.append(jtpr)
        joint_tp_total += details['joint_tp']
        total_pairs_total += details['total_pairs']

    # Mean JTPR across all images
    mean_jtpr = np.mean(jtpr_values) if jtpr_values else 0.0

    # Global JTPR (pooled across all images)
    global_jtpr = joint_tp_total / total_pairs_total if total_pairs_total > 0 else 0.0

    print(f"    JTPR (mean): {mean_jtpr:.4f}")
    print(f"    JTPR (global): {global_jtpr:.4f}")
    print(f"    Joint TP: {joint_tp_total} / {total_pairs_total} pairs")

    # ========== Results ==========
    print("\n" + "=" * 60)
    print(f"INFERENCE RESULTS ({dataset_name})")
    print("=" * 60)
    print(f"  Nucleus   | Dice: {mean_dice_nucleus:.4f} | AP: {safe_float(ap_nucleus):.4f} (AP@50: {ap50_nucleus:.4f})")
    print(
        f"  Cytoplasm | Dice: {mean_dice_cytoplasm:.4f} | AP: {safe_float(ap_cytoplasm):.4f} (AP@50: {ap50_cytoplasm:.4f})")
    print(f"  JTPR      | Mean: {mean_jtpr:.4f} | Global: {global_jtpr:.4f} ({joint_tp_total}/{total_pairs_total})")
    print("=" * 60)

    results = {
        'dataset': dataset_name,
        'n_images': len(y_true_nucleus),
        'dice_nucleus_mean': float(mean_dice_nucleus),
        'dice_cytoplasm_mean': float(mean_dice_cytoplasm),
        'ap_nucleus': safe_float(ap_nucleus),
        'ap_cytoplasm': safe_float(ap_cytoplasm),
        'ap50_nucleus': float(ap50_nucleus),
        'ap50_cytoplasm': float(ap50_cytoplasm),
        'jtpr_mean': float(mean_jtpr),
        'jtpr_global': float(global_jtpr),
        'joint_tp': int(joint_tp_total),
        'total_pairs': int(total_pairs_total),
        'checkpoint': str(checkpoint_path),
        'prob_thresh_nucleus': predictor.prob_thresh_nucleus,
        'prob_thresh_cytoplasm': predictor.prob_thresh_cytoplasm,
        'nms_thresh': predictor.nms_thresh,
    }

    with open(viz_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {viz_dir}")
    print(f"✅ Visualization saved to: {viz_dir }")
    print(f"✅ Debug images saved: debug_input_raw.png, debug_input_normalized.png")

    return model, predictor, results

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

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    # ========== MODE SELECTION ==========
    mode = config_dict.get('mode', 'train')  # По умолчанию training

    if mode == 'inference':
        print("\n🔍 MODE: INFERENCE ONLY")
        result = run_inference_only(config_dict, config_path)  # ← СОХРАНЯЕМ результат
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
            validation_images=config_dict['data'].get('validation_images'),
            test_split=config_dict['preprocessing'].get('test_split', 0.1),
            val_split=config_dict['preprocessing'].get('val_split', 0.1),
            compute_assignments=config_dict['preprocessing']['compute_assignments'],
            seed=config_dict['preprocessing'].get('seed', 42)
        )

    # ========== Create Model Configuration ==========
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
    # ============================================================
    # === COMPUTE MEDIAN RADIUS FOR ADAPTIVE FOOTPRINT ===
    # ============================================================

    print("\n📊 Computing training statistics for adaptive footprint...")

    all_nuc_radii = []
    all_cyto_radii = []

    num_samples = min(10, len(train_generator.data_pool))

    for i in range(num_samples):
        sample = train_generator.data_pool[i]

        from skimage.measure import regionprops
        from scipy.ndimage import distance_transform_edt

        # === NUCLEUS STATISTICS ===
        nuc_mask = sample['nucleus_mask']
        nuc_props = regionprops(nuc_mask)

        for prop in nuc_props:
            # Метод 1: Через area (быстро, всегда работает)
            if model_config.ndim == 2:
                radius_from_area = np.sqrt(prop.area / np.pi)
            else:
                radius_from_area = (prop.area * 3 / (4 * np.pi)) ** (1 / 3)

            # КРИТИЧНО: Пропускаем слишком маленькие объекты
            if radius_from_area < 3.0:
                continue

            # Метод 2: Через StarDist distances (точно, но может fail)
            obj_mask = (nuc_mask == prop.label).astype(np.int32)

            try:
                # Вызываем StarDist для получения distances
                if model_config.ndim == 2:
                    from stardist.geometry import star_dist
                    dist_map = star_dist(obj_mask, model_config.n_rays, mode='cpp')
                else:
                    from stardist.geometry import star_dist3D
                    dist_map = star_dist3D(obj_mask, model_config.rays, mode='cpp')

                # Берём distances в центре объекта
                center = tuple(int(c) for c in prop.centroid)

                # ЗАЩИТА: проверяем, что центр внутри маски
                if obj_mask[center] == 0:
                    # Центр снаружи - используем центроид маски
                    from scipy.ndimage import center_of_mass
                    center = tuple(int(c) for c in center_of_mass(obj_mask))

                dists_at_center = dist_map[center]

                # ЗАЩИТА: фильтруем нулевые/NaN значения
                valid_dists = dists_at_center[dists_at_center > 0.1]  # минимум 0.1 px

                if len(valid_dists) >= model_config.n_rays * 0.5:  # минимум 50% лучей валидны
                    mean_radius = np.mean(valid_dists)

                    # ЗАЩИТА: проверяем разумность
                    if 1.0 < mean_radius < 200.0:  # разумный диапазон для 2D
                        all_nuc_radii.append(mean_radius)
                    else:
                        # Fallback: используем area-based
                        all_nuc_radii.append(radius_from_area)
                else:
                    # Слишком мало валидных лучей
                    all_nuc_radii.append(radius_from_area)

            except Exception as e:
                # StarDist failed - используем area-based
                print(f"    Warning: StarDist failed for nucleus {prop.label}: {e}")
                all_nuc_radii.append(radius_from_area)

        # === CYTOPLASM STATISTICS (то же самое) ===
        cyto_mask = sample['cytoplasm_mask']
        cyto_props = regionprops(cyto_mask)

        for prop in cyto_props:
            if model_config.ndim == 2:
                radius_from_area = np.sqrt(prop.area / np.pi)
            else:
                radius_from_area = (prop.area * 3 / (4 * np.pi)) ** (1 / 3)

            # Пропускаем маленькие объекты
            if radius_from_area < 5.0:  # cytoplasm обычно больше
                continue

            obj_mask = (cyto_mask == prop.label).astype(np.int32)

            try:
                if model_config.ndim == 2:
                    from stardist.geometry import star_dist
                    dist_map = star_dist(obj_mask, model_config.n_rays, mode='cpp')
                else:
                    from stardist.geometry import star_dist3D
                    dist_map = star_dist3D(obj_mask, model_config.rays, mode='cpp')

                center = tuple(int(c) for c in prop.centroid)

                if obj_mask[center] == 0:
                    from scipy.ndimage import center_of_mass
                    center = tuple(int(c) for c in center_of_mass(obj_mask))

                dists_at_center = dist_map[center]
                valid_dists = dists_at_center[dists_at_center > 0.1]

                if len(valid_dists) >= model_config.n_rays * 0.5:
                    mean_radius = np.mean(valid_dists)

                    if 2.0 < mean_radius < 500.0:  # разумный диапазон для cytoplasm
                        all_cyto_radii.append(mean_radius)
                    else:
                        all_cyto_radii.append(radius_from_area)
                else:
                    all_cyto_radii.append(radius_from_area)

            except Exception as e:
                print(f"    Warning: StarDist failed for cytoplasm {prop.label}: {e}")
                all_cyto_radii.append(radius_from_area)

    # === ВЫЧИСЛЯЕМ МЕДИАНЫ с FALLBACK ===
    if len(all_nuc_radii) > 0:
        nuc_median_radius = np.median(all_nuc_radii)
        print(f"  ✅ Median nucleus radius: {nuc_median_radius:.1f} px (from {len(all_nuc_radii)} objects)")
    else:
        nuc_median_radius = 10.0
        print(f"  ⚠️  No valid nucleus objects, using fallback: {nuc_median_radius} px")

    if len(all_cyto_radii) > 0:
        cyto_median_radius = np.median(all_cyto_radii)
        print(f"  ✅ Median cytoplasm radius: {cyto_median_radius:.1f} px (from {len(all_cyto_radii)} objects)")
    else:
        cyto_median_radius = 30.0
        print(f"  ⚠️  No valid cytoplasm objects, using fallback: {cyto_median_radius} px")

    # СОХРАНЯЕМ В CONFIG
    model_config.median_max_dist_nucleus = float(nuc_median_radius)
    model_config.median_max_dist_cytoplasm = float(cyto_median_radius)
    model_config.median_radius_train = float(cyto_median_radius)

    # Дополнительная статистика
    if len(all_nuc_radii) > 0:
        print(f"    Nucleus radius range: [{np.min(all_nuc_radii):.1f}, {np.max(all_nuc_radii):.1f}] px")
    if len(all_cyto_radii) > 0:
        print(f"    Cytoplasm radius range: [{np.min(all_cyto_radii):.1f}, {np.max(all_cyto_radii):.1f}] px")

    # Set steps_per_epoch based on repool frequency
    if config_dict['data'].get('repool_freq'):
        model_config.train_steps_per_epoch = train_generator.repool_freq
        print(f"Steps per epoch set to repool frequency: {model_config.train_steps_per_epoch}")

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
    fov = np.array(model._axes_tile_overlap('ZYX') if model_config.ndim == 3 else model._axes_tile_overlap('YX'))
    median_size = extents if 'extents' in locals() else np.array([96, 96])
    print(f"\nMedian object size:     {median_size}")
    print(f"Network field of view:  {fov}")
    if any(np.array(median_size) > fov):
        print("⚠️  WARNING: median object size larger than field of view of the neural network.")

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

    warmup_epochs = 5  # Первые 5 эпох - прогрев
    initial_lr = config_dict['training']['initial_learning_rate']
    steps_per_epoch = model_config.train_steps_per_epoch

    schedule = WarmupExponentialDecay(
        initial_lr=initial_lr,
        warmup_steps=warmup_epochs * steps_per_epoch,
        decay_steps=10000,  # Увеличим для более плавного спада
        decay_rate=0.9,
        min_lr=1e-7
    )

    trainer = CytoNucsTrainer(
        model=model,
        config=model_config,
        loss_fn=loss_fn,
        optimizer=Adam(learning_rate=schedule)
    )
    # Restore history if resuming
    if 'loaded_history' in locals():
        trainer.history = loaded_history
        print(f"✓ Restored training history")
    # ========== КОНЕЦ БЛОКА ==========
    # Callbacks
    callbacks = []

    if config_dict['evaluation']['eval_dice_every'] > 0:
        dice_callback = DiceCallback(
            model=model,
            val_generator=val_generator,
            eval_every=config_dict['evaluation']['eval_dice_every'],
            save_best=config_dict['output']['save_best_dice'],
            experiment=experiment
        )
        callbacks.append(dice_callback)

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

    # ========== Threshold Optimization ==========
    if config_dict['evaluation']['optimize_thresholds']:
        print("\n" + "=" * 60)
        print("Optimizing thresholds...")
        print("=" * 60 + "\n")

        # Create predictor
        predictor = CytoNucsPredictor(
            model=model,
            config=model_config,
            prob_thresh=0.5,
            nms_thresh=0.5
        )

        # Collect validation data
        val_images = []
        val_nucleus_masks = []
        val_cytoplasm_masks = []

        print("Loading validation data...")
        for i in tqdm(range(len(val_generator))):
            x_batch, y_batch = val_generator[i]
            if i >= 1:  # Берем только первый батч для быстрой оптимизации
                break

            for j in range(len(x_batch)):
                # Для RGB берём ВСЕ каналы, для grayscale - один
                if x_batch.shape[-1] == 3:
                    val_images.append(x_batch[j])  # RGB: (H, W, 3)
                else:
                    val_images.append(x_batch[j, ..., 0])  # Grayscale: (H, W)

                val_nucleus_masks.append(y_batch['nucleus_instances'][j])
                val_cytoplasm_masks.append(y_batch['cytoplasm_instances'][j])

        # Optimize
        best_params = predictor.optimize_thresholds(
            val_images, val_nucleus_masks, val_cytoplasm_masks,
            prob_thresh_range=config_dict['evaluation']['prob_thresh_range'],
            nms_thresh_range=config_dict['evaluation']['nms_thresh_range']
        )

        if experiment:
            experiment.log_parameters(best_params)

        # Save optimized thresholds
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
            prob_thresh_nucleus=0.3,
            prob_thresh_cytoplasm=0.35,  # ← Поднять с 0.3!
            nms_thresh=0.5  # ← Поднять с 0.4!
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
        nuc_pred, cyto_pred, details = predictor.predict_instances(
            img,
            normalize_img=False  # Already normalized in generator
        )
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
            jtpr, jtpr_details = compute_jtpr(nuc_pred, cyto_pred,
                                   val_nucleus_masks[i], val_cytoplasm_masks[i])
            jtpr_scores.append(jtpr)
            print(f"    JTPR details: {jtpr_details}")
        # Save predictions if requested
        if config_dict['evaluation']['save_predictions']:
            import imageio
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
        import imageio
        images = []
        for img_path in sorted(viz_dir.glob('epoch_*.png')):
            images.append(imageio.imread(img_path))

        if len(images) > 0:
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
    print(f"\nMedian object size:     {median_size}")
    print(f"Network field of view:  {fov}")

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
        model, predictor, results = main(args.config)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)