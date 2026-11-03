# CytoNucs: nested instance segmentation with StarDist and prompted mask refinement

CytoNucs segments objects that lie inside other objects: nuclei inside cells, tumours inside kidneys, or longer
chains of nested levels. It works on 2D and 3D images, including anisotropic volumes. Every object of every
level gets an instance mask, and every child object is linked to its parent (a cell may hold several nuclei).

The method keeps StarDist [1, 2] at its core:

- A shared U-Net predicts, for every level, a StarDist object probability and star-convex ray distances.
  Peaks of the probability are the object proposals.
- Each proposal becomes a prompt: a Gaussian placed at the peak and sized by its rays. A light refinement network
  reads the decoder features gated by the prompt and predicts the mask of that one object, as promptable
  segmentation models do with a point prompt [3]. A full-resolution head sharpens the mask, and a quality head
  scores it. The mask is not limited to a star-convex shape.
- A nucleus prompt yields the nucleus and the cell that contains it. Cell prompts find the remaining cells.
  Masks are written in order of quality, and a child stays inside its parent.
- Three or more levels are run as a chain of adjacent pairs through the same network.
- Levels may be annotated only in parts of the data (partial annotation, unknown labels).

Training adds to the StarDist losses (probability BCE, ray MAE):

- a Polar IoU loss on the rays, from PolarMask [4];
- an IoU-aware probability target (the StarDist target times the Polar IoU of the predicted rays), after
  VarifocalNet [5];
- mask losses (cross-entropy and the Lovász loss [6]) and a quality loss for the refinement;
- a containment term and the within-boundary regularisation (WBR) of HydraStarDist [7], which tie the levels
  together.

> Status: research code of a paper in preparation. Results and the exact training recipes of the paper will be
> added here.

## Repository layout

```
train_cytonucs.py            training entry point (one JSON configuration)
config_cytonucs.json         configuration with every option and its default
cytonucs_config.py           configuration loading, validation, ray sets and grids from the object geometry
cytonucs_model.py            U-Net, StarDist heads, prompted refinement and full-resolution heads
cytonucs_losses.py           losses and the StarDist targets
cytonucs_data_generator.py   tile / region sampling, augmentation, prompts and targets
cytonucs_trainer.py          training loop, checkpoints, in-training full validation
cytonucs_inference.py        proposals, batched / windowed prompt refinement, tiling
cytonucs_postprocess.py      mask candidates and their assembly into label images
cytonucs_hierarchy.py        hierarchies of three or more levels
cytonucs_fine_mask.py, cytonucs_mask_quality.py, cytonucs_geometry.py, cytonucs_containment_helper.py,
cytonucs_refinement_audit.py, cytonucs_runtime.py, cytonucs_gpu_setup.py      helpers

score_hierarchy.py           AP, JTPR and the HydraStarDist-paper metrics of a trained model on a split
score_predictions.py         the same scores for saved label images of any method
semantic_gate.py             threshold of the semantic gate of one level, chosen on saved predictions
recalibrate_gates.py         quality gates of a trained run chosen for AP alone, from its full-validation log
evaluate_blocks.py           scores on whole held-out blocks of a spatially split 3D volume
score_native_resolution.py   scores of a model trained on a downscaled copy, against the native annotation
gpu_monitor.py               GPU and host memory log of a run (resource tables)

baselines/                   StarDist (2D/3D), HydraStarDist, Cellpose (2D/3D) and nnU-Net baselines on the same splits
dataset_prep/                converters of public datasets and split / sparse-annotation tools
```

## Installation

Tested with Python 3.10, TensorFlow 2.18 (Keras 3.12), StarDist 0.9.1, CSBDeep 0.8.1, NumPy 1.26, SciPy 1.15,
scikit-image 0.25, tifffile, imageio and tqdm, on NVIDIA RTX A5000 GPUs (24 GB).

```bash
pip install "tensorflow[and-cuda]==2.18.*" "numpy<2.1" stardist csbdeep scikit-image scipy tifffile imageio tqdm
pip install --no-deps edt      # optional: fast EDT for the StarDist targets (several times faster batches)
```

Install the Cellpose baseline (Cellpose 3, PyTorch) and the nnU-Net baseline (nnU-Net v2) in their own environments.

## Data layout

One folder per split. Each holds the images and, for every level of `model.levels` (inner level first), its
instance label images under the same file names:

```
dataset/
├── train/
│   ├── images/                    2D: YX or YXC; 3D: ZYX or ZYXC (TIFF)
│   ├── nucleus_masks/             one instance label image per image
│   ├── cytoplasm_masks/
│   └── assignments/               optional: {nucleus id: cell id} per image (JSON); derived by overlap if missing
├── val/
└── test/
```

Other level names work the same way, for example `model.levels: ["mass", "kidney"]` reads `mass_masks/`,
`kidney_masks/` and `assignments_mass_kidney/`. `dataset_prep/` writes this layout for TissueNet, HeLaCytoNuc,
BBBC020 and KiTS23, and builds spatially disjoint splits, train regions and sparse-annotation copies of 3D
volumes.

## Training

Copy `config_cytonucs.json`, set `data.dataset_dir`, `model.ndim`, `model.levels` and the patch size, then run:

```bash
python train_cytonucs.py my_config.json
```

Main options:

| Section | Option | Meaning |
|---|---|---|
| model | `ndim`, `levels`, `n_rays` | dimensionality, nested levels (inner first), rays per object |
| model | `unet_n_filter_base`, `unet_n_depth` | width and depth of the shared U-Net |
| training | `train_patch_size`, `image_batch_size`, `prompts_per_crop` | crop size, crops per step, prompts per crop |
| training | `train_epochs`, `steps_per_epoch`, `initial_learning_rate`, `weight_decay`, `ema_decay` | schedule and optimiser |
| training | `object_crop_shares` | [inner, outer]: shares of the crops centred on one object of that level, picked uniformly; the rest at random positions (replaces `fg_crop_prob`) |
| augmentation | `negative_prompt_prob`, `negative_prompt_inside_outer` | share of empty-target (negative) prompts; share of them placed in an outer object where no inner object lies (hard negatives, e.g. kidney tissue) |
| training | `keep_epoch_checkpoints` | how many latest epoch checkpoints to keep (best checkpoints are kept apart) |
| loss_params | `dist_polar_iou_weight`, `prob_iou_aware` | Polar IoU on the rays, IoU-aware probability target |
| inference | `refinement_batch`, `refinement_windows`, `skip_covered_prompts`, `refinement_roi_stride`, `refinement_roi_size_step` | speed of the prompt refinement |
| inference | `inner_min_inside_outer` | an inner object is kept only with at least this share of its voxels inside the kept outer objects (0 = off) |
| inference | `outer_min_inner_share` | an outer object is kept only if at least this share of some inner object lies inside it, e.g. a cell needs a nucleus (0 = off; leave off where cells without a nucleus are real) |
| training | `mask_prompt_fraction`, `mask_prompt_start_epoch` | share of the prompts that also carry the model's own first mask (a pass without gradient), from that epoch: trains the second refinement pass |
| inference | `refinement_iterations` | refinement passes per candidate; from the second on the prompt carries the previous mask (`mask_prompt_fraction` must have trained this) |
| loss_params | `center_offset_weight` | head predicting each voxel's offset to its object's centre (Panoptic-DeepLab), Huber in head-grid nodes (0 = no head) |
| inference | `center_vote_grouping`, `center_vote_radius` | StarDist candidates whose objects vote for one centre (closer than the radius share of the larger mean ray) are prompted once, as the union of their polygons (needs `center_offset_weight`) |
| inference | `tile_blending` | fixed tiling windows (half overlap) added with a Hann weight instead of each writing its centre: no step where two windows meet (each window has its own context and group-norm statistics); same forwards |
| inference | `quality_gates_override` | per-head quality / size gates replacing the calibrated ones (e.g. from `recalibrate_gates.py`) |
| inference | `semantic_scores`, `semantic_min_nucleus`, `semantic_min_cytoplasm` | mean semantic foreground probability of every object (saved with `--predictions` in `semantic/`); objects of a role below its minimum are removed (0 = off) |

The ray geometry and grids are measured on the training labels at the start of a run and saved next to the
weights. The run folder (`output.basedir/experiment.name`) holds the best checkpoints by full validation and by
validation loss, the last epoch checkpoint, the calibrated quality gates and mask thresholds, and the history.

## Evaluation

```bash
python score_hierarchy.py --config my_config.json --weights RUN/weights_best.weights.h5 --split test \
    --output RUN/hierarchy_test.json --predictions RUN/predictions_test
python score_predictions.py --data dataset --predictions OTHER_METHOD/predictions --split test
python evaluate_blocks.py --config my_config.json --weights RUN/weights_best.weights.h5 --splits test val
```

- **AP**: the mean over IoU 0.5–0.95 of TP / (TP + FP + FN), for every level.
- **JTPR**: for every adjacent pair of levels, the share of annotated child–parent pairs whose two objects are
  both found (IoU ≥ 0.5) and linked to each other in the prediction.
- **HydraStarDist-paper metrics** [7]: IoUR, AP and the joint TP rates over children and over parents at
  τ = 0.1–0.9, so that results can be compared with that paper.

Scores are kept per image, so bootstrap intervals and breakdowns can be computed again without predicting.

## Baselines

- `baselines/stardist3d/`: StarDist with one model per level (2D or 3D, files or spatial splits, full resolution).
- `baselines/hydrastardist/`: HydraStarDist [7], the authors' code, vendored with its BSD licence;
  `run_hsd.py --wbr` trains HSD-WBR with the paper's within-boundary loss made differentiable (the released
  WBR has zero gradient; the changes are marked in the code).
- `baselines/cellpose/`: Cellpose [8] (version 3), fine-tuned from the pretrained models and as pretrained;
  `run_cellpose3d.py` for the 3D volume (2D network on XY / XZ / YZ planes, 3D flows or stitched planes chosen on
  the val blocks, scored on whole blocks), `run_cellpose3d_files.py` the same for a dataset of one file per volume
  (KiTS; settings chosen on val16, scored with `score_predictions.py`).
- `baselines/nnunet/`: nnU-Net v2 [9] on KiTS: `prepare_kits.py` writes the CytoNucs conversion as an nnU-Net
  dataset (kidney / tumour / cyst, the same split as one fold), `to_instances.py` turns its predictions into
  kidney and mass instances by the rule of the CytoNucs labels.

## References

1. U. Schmidt, M. Weigert, C. Broaddus, G. Myers. Cell Detection with Star-Convex Polygons. MICCAI 2018.
   [arXiv:1806.03535](https://arxiv.org/abs/1806.03535)
2. M. Weigert, U. Schmidt, R. Haase, K. Sugawara, G. Myers. Star-convex Polyhedra for 3D Object Detection and
   Segmentation in Microscopy. WACV 2020. [arXiv:1908.03636](https://arxiv.org/abs/1908.03636)
3. A. Kirillov et al. Segment Anything. ICCV 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
4. E. Xie et al. PolarMask: Single Shot Instance Segmentation with Polar Representation. CVPR 2020.
   [arXiv:1909.13226](https://arxiv.org/abs/1909.13226)
5. H. Zhang, Y. Wang, F. Dayoub, N. Sünderhauf. VarifocalNet: An IoU-aware Dense Object Detector. CVPR 2021.
   [arXiv:2008.13367](https://arxiv.org/abs/2008.13367)
6. M. Berman, A. Rannen Triki, M. B. Blaschko. The Lovász-Softmax loss: A tractable surrogate for the
   optimization of the intersection-over-union measure in neural networks. CVPR 2018.
   [arXiv:1705.08790](https://arxiv.org/abs/1705.08790)
7. T. De, A. Urbanski, A. Yakimovich. Single-shot Star-convex Polygon-based Instance Segmentation for
   Spatially-correlated Biomedical Objects (HydraStarDist). arXiv 2025.
   [arXiv:2504.12078](https://arxiv.org/abs/2504.12078)
8. C. Stringer, T. Wang, M. Michaelos, M. Pachitariu. Cellpose: a generalist algorithm for cellular
   segmentation. Nature Methods 18, 100–106 (2021). [doi:10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)
9. F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, K. H. Maier-Hein. nnU-Net: a self-configuring method for
   deep learning-based biomedical image segmentation. Nature Methods 18, 203–211 (2021).
   [doi:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)

The paper describing CytoNucs is in preparation.

## Contact

mayurovm@gmail.com
