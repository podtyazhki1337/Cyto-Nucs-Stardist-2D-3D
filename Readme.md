# CytoNucs StarDist: Multi-Nuclear Cell Segmentation

Dual-decoder StarDist for simultaneous nucleus and cytoplasm segmentation in 2D/3D microscopy images. Supports both single-nucleus and multi-nucleus cells.

## Features

- 🔬 **Dual segmentation**: Nuclei + Cytoplasm simultaneously
- 🧬 **Multi-nuclear support**: Handles cells with multiple nuclei
- 📐 **2D/3D unified**: Same codebase for both dimensions
- ⚙️ **JSON configuration**: All parameters in one file
- 🎯 **Advanced losses**: Containment, WBR, optional consistency
- 📊 **Auto-assignment**: Automatic nucleus-cell pairing
- 🔄 **Auto-split**: Automatic train/val/test splitting

## Installation
```bash
pip install tensorflow>=2.8.0 stardist scikit-image scipy numpy imageio tqdm
pip install comet-ml  # optional, for experiment tracking
```

## Quick Start

### 1. Prepare Dataset

Organize your data:
```
raw_data/
├── images/            # Raw microscopy images (.tif)
├── nucleus_masks/     # Nucleus instance labels (.tif)
└── cytoplasm_masks/   # Cytoplasm instance labels (.tif)
```

### 2. Configure

Edit `config_cytonucs.json`:
```json
{
  "experiment": {
    "name": "my_experiment"
  },
  "preprocessing": {
    "run_preprocessing": true,
    "raw_images_dir": "raw_data/images",
    "raw_nucleus_masks_dir": "raw_data/nucleus_masks",
    "raw_cytoplasm_masks_dir": "raw_data/cytoplasm_masks",
    "output_dir": "processed_data/",
    "test_split": 0.1,
    "val_split": 0.1
  },
  "data": {
    "dataset_dir": "processed_data/"
  },
  "model": {
    "ndim": 3,
    "n_rays": 128
  },
  "training": {
    "train_patch_size": [12, 96, 96],
    "train_batch_size": 4,
    "train_epochs": 400
  }
}
```

### 3. Train
```bash
python train_cytonucs.py config_cytonucs.json
```

That's it! The script will:
- ✅ Preprocess data (if enabled)
- ✅ Auto-compute nucleus-cell assignments
- ✅ Split into train/val/test
- ✅ Train the model
- ✅ Optimize thresholds
- ✅ Evaluate and save results

## File Structure
```
cytonucs_stardist/
├── config_cytonucs.json          # Configuration
├── cytonucs_config.py             # Config classes
├── cytonucs_model.py              # Dual-decoder model
├── cytonucs_losses.py             # Loss functions
├── cytonucs_data_generator.py     # Data loading
├── cytonucs_trainer.py            # Training loop
├── cytonucs_inference.py          # Inference
└── train_cytonucs.py              # Main script
```

## Key Configuration Options

### Model
- `ndim`: 2 or 3 (dimensionality)
- `n_rays`: Number of rays for star-convex (default: 128)
- `shared_encoder`: Share encoder between decoders (default: true)

### Loss Weights
- `lambda_containment`: 0.3-0.7 (nuclei inside cells)
- `lambda_wbr`: 0.2-0.5 (within-boundary regularization)
- `lambda_consistency`: 0.0-0.3 (cell center near nuclei, optional)
- `enable_consistency`: false (disable by default)

### Training
- `train_patch_size`: e.g., [12, 96, 96] for 3D
- `train_batch_size`: 2-8 depending on GPU
- `pool_size`: null (all data) or N samples in memory
- `repool_freq`: 10 (reload data every N batches)

## Inference
```python
from cytonucs_config import CytoNucsConfig
from cytonucs_model import build_cytonucs_model
from cytonucs_inference import CytoNucsPredictor
import imageio

# Load model
config = CytoNucsConfig.from_json('config_cytonucs.json')
model = build_cytonucs_model(config, name='my_experiment')
model.load_weights('models/my_experiment/weights_best.h5')

# Predict
predictor = CytoNucsPredictor(model, config)
img = imageio.volread('test_image.tif')
nucleus_pred, cytoplasm_pred, details = predictor.predict_instances(img)

# Save
imageio.volwrite('nucleus_result.tif', nucleus_pred)
imageio.volwrite('cytoplasm_result.tif', cytoplasm_pred)
```

## Output Structure
```
models/my_experiment/
├── weights_best.h5               # Best model (by val loss)
├── weights_best_dice_0.85.h5     # Best model (by Dice)
├── weights_epoch_*.h5            # Checkpoints
├── history.json                  # Training history
├── optimized_thresholds.json     # Optimal thresholds
├── final_results.json            # Final metrics
└── predictions/                  # Validation predictions
```

## Advanced Features

### Automatic Assignment Creation

If assignments are missing, they're computed on-the-fly using 3-stage strategy:
1. Containment (nucleus center inside cell)
2. Overlap (maximum IoU)
3. Proximity (nearest cell within threshold)

### Automatic Train/Val/Test Split

Two modes:
```json
// Mode 1: Automatic split
"preprocessing": {
  "validation_images": null,
  "test_split": 0.1,
  "val_split": 0.1
}

// Mode 2: Manual validation set
"preprocessing": {
  "validation_images": ["sample_007.tif"],
  "test_split": 0.1
}
```

### Repooling

For large datasets that don't fit in memory:
```json
"data": {
  "pool_size": 5,       // Keep 5 samples in RAM
  "repool_freq": 10     // Change pool every 50 batches
}
```

## Metrics

- **Dice**: Standard overlap metric
- **JTPR** (Joint True Positive Rate): Nucleus-cell pair correctness

## Citation

Based on StarDist:
```
@inproceedings{schmidt2018,
  title={Cell Detection with Star-Convex Polygons},
  author={Schmidt, Uwe and Weigert, Martin and Broaddus, Coleman and Myers, Gene},
  booktitle={MICCAI},
  year={2018}
}
```



## Troubleshooting

**Out of memory?**
- Reduce `train_batch_size`
- Reduce `train_patch_size`
- Use `pool_size` parameter

**Poor containment?**
- Increase `lambda_containment`
- Check nucleus-cell assignments

**FOV warning?**
- Increase `train_patch_size`
- Or use smaller `grid`

## Contact

Issues and questions: [GitHub Issues](https://github.com/YOUR_USERNAME/Multi-Nuclear-Stardist-2D-3D/issues)