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


def main(config_path='config_cytonucs.json'):
    """
    Main training function.
    Follows the same logic as original train.py
    """

    # ========== Load Configuration ==========
    print("=" * 60)
    print("CytoNucs StarDist Training")
    print("=" * 60)

    config_dict = load_config(config_path)
    print(f"\nLoaded config from: {config_path}")

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

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
    model_config = CytoNucsConfig.from_json(config_path)

    # Auto-compute anisotropy and grid if not specified
    if model_config.anisotropy is None or model_config.grid is None:
        # Load training data to compute median cell sizes
        temp_gen = CytoNucsDataGenerator(
            dataset_dir=Path(config_dict['data']['dataset_dir']) / 'train',
            config=model_config,
            rays=model_config.rays,
            subset='train'
        )

        train_medians = temp_gen.get_cell_properties('cells_median_extent_p')
        if train_medians is not None:
            extents = np.median(train_medians, axis=0)
        else:
            # Fallback: estimate from first sample
            print("Warning: No metadata found, estimating from first sample")
            x, y = temp_gen[0]
            extents = np.array(model_config.train_patch_size) / 4  # rough estimate

        print(f"\nMedian cell extents from data: {extents}")

        # Compute anisotropy
        if model_config.anisotropy is None:
            anisotropy = tuple(np.max(extents) / extents)
            model_config.anisotropy = anisotropy
            print(f"Computed anisotropy: {anisotropy}")

        # Compute grid
        if model_config.grid is None:
            grid = tuple(1 if a > 1.5 else 4 for a in model_config.anisotropy)
            model_config.grid = grid
            print(f"Computed grid: {grid}")

        # Recreate rays with anisotropy
        from stardist import Rays_GoldenSpiral
        model_config.rays = Rays_GoldenSpiral(model_config.n_rays, anisotropy=model_config.anisotropy)

        del temp_gen

    print("\nModel Configuration:")
    print(model_config)

    if experiment:
        experiment.log_parameters({
            'n_rays': model_config.n_rays,
            'train_patch_size': model_config.train_patch_size,
            'train_batch_size': model_config.train_batch_size,
            'anisotropy': model_config.anisotropy,
            'grid': model_config.grid,
            'lambda_containment': model_config.lambda_containment,
            'lambda_wbr': model_config.lambda_wbr,
            'lambda_consistency': model_config.lambda_consistency,
            'enable_consistency': model_config.enable_consistency,
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

    print("\nModel Summary:")
    model.summary()

    # Check FOV vs median object size
    fov = np.array(model._axes_tile_overlap('ZYX') if model_config.ndim == 3 else model._axes_tile_overlap('YX'))
    median_size = extents if 'extents' in locals() else model_config.train_patch_size
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

    schedule = ExponentialDecay(
        initial_learning_rate=config_dict['training']['initial_learning_rate'],
        decay_steps=config_dict['training']['lr_decay_steps'],
        decay_rate=config_dict['training']['lr_decay_rate'],
        staircase=config_dict['training']['lr_staircase']
    )

    trainer = CytoNucsTrainer(
        model=model,
        config=model_config,
        loss_fn=loss_fn,
        optimizer=Adam(learning_rate=schedule)
    )

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
        experiment=experiment
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
            nms_thresh=0.4
        )

        # Collect validation data
        val_images = []
        val_nucleus_masks = []
        val_cytoplasm_masks = []

        print("Loading validation data...")
        for i in tqdm(range(len(val_generator))):
            x_batch, y_batch = val_generator[i]
            for j in range(len(x_batch)):
                val_images.append(x_batch[j, ..., 0])
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
            prob_thresh=0.5,
            nms_thresh=0.4
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
            jtpr, _ = compute_jtpr(nuc_pred, cyto_pred,
                                   val_nucleus_masks[i], val_cytoplasm_masks[i])
            jtpr_scores.append(jtpr)

        # Save predictions if requested
        if config_dict['evaluation']['save_predictions']:
            import imageio
            pred_dir = Path(config_dict['output']['basedir']) / config_dict['experiment']['name'] / 'predictions'
            pred_dir.mkdir(parents=True, exist_ok=True)

            imageio.volwrite(pred_dir / f'val_{i:03d}_nucleus.tif', nuc_pred)
            imageio.volwrite(pred_dir / f'val_{i:03d}_cytoplasm.tif', cyto_pred)

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