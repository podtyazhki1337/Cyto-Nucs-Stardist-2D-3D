# Configurations of the models in the paper

One file per trained model, as it was trained, without machine-specific paths. Each file expects its dataset under
`data/` (relative to the folder you run from; see the dataset preparation in the main README) and writes its run to
`runs/<experiment.name>/`. Seeds 1 and 2 of a recipe use the same file with `training.seed` 1 / 2.

```bash
python train_cytonucs.py configs/helacytonuc_e7_clean.json
python score_hierarchy.py --config configs/helacytonuc_e7_clean.json \
    --weights runs/helacytonuc_e7_clean/weights_best.weights.h5 --split test --output runs/helacytonuc_e7_clean/hierarchy_test.json
```

Spatially split 3D volumes (the HeLa spheroid) are scored on whole held-out blocks with `evaluate_blocks.py`.

## Final models

| config | data (`data/...`) | what |
|---|---|---|
| `tissuenet_e7_b32` | `TissueNet/cytonucs` | final recipe on TissueNet (test and held-out tiles) |
| `helacytonuc_e7_clean` | `HeLaCytoNuc/cytonucs_clean` | final recipe on HeLaCytoNuc, cleaned split (test and test_cellprofiler) |
| `bbbc020_e7` | `BBBC020/cytonucs_native` | final recipe on BBBC020 |
| `e_zaug` | `Cytonucs_Emese_Hela/patches_split_hela_xy2` | HeLa spheroid at half XY resolution, initialised from `a1_syn_pre2` (epoch 40), 30 % synthetic crops |
| `p_all` | `Cytonucs_Emese_Hela/patches_split_hela_xy2` | HeLa spheroid at half XY resolution, from scratch |
| `cyto_fullres` | `Cytonucs_Emese_Hela/patches_split_hela` | HeLa spheroid at full resolution |
| `kits_e8` | `KiTS23/cytonucs_3x078_val16` | KiTS23, kidneys and masses |
| `h3l_syn` | `Cytonucs_Synthetic/hela_v1` | synthetic data, three levels (nucleolus, nucleus, cell) |
| `h3l_syn_2l` | `Cytonucs_Synthetic/hela_v1` | synthetic data, two levels |
| `a1_syn_pre2` | `Cytonucs_Synthetic/hela_v1` | synthetic pretraining of `e_zaug` |

## Ablations

| config | what |
|---|---|
| `helacytonuc_e7_clean_nomask` | final recipe without the mask refinement |
| `helacytonuc_e7_clean_nonest` | final recipe without the nesting losses |
| `helacytonuc_e7_clean_k1` | final recipe with one prompt per crop |
| `helacytonuc_e7_clean_nuc_only`, `helacytonuc_e7_clean_cells_only` | separate single-level networks (paired by overlap for the comparison) |
| `tissuenet_e7_b32_nuc_only`, `tissuenet_e7_b32_cells_only` | the same on TissueNet |
| `e_zaug_nuc_only`, `e_zaug_cells_only` | the same on the HeLa spheroid; initialised from `a1_syn_pre2` by layer name where the shapes agree (`training.init_weights_partial`) |
| `tissuenet_e7_b32_nonest` | TissueNet without the nesting losses |

Two ablations change only the inference of a trained model: `inference.outer_from_inner_prompts: false` (cells only
from the cell head's own proposals) and `inference.outer_own_prompts: false` (cells only from nucleus prompts). Set
either in a copy of the model's configuration and score it as usual.

## Partial annotation

| config | data (`data/...`) | what |
|---|---|---|
| `p_k50_s0`, `p_k20_s0`, `p_k5_s0` | `Cytonucs_Emese_Hela/patches_split_hela_xy2_k50` / `_k20` / `_k5` | nuclei annotated in 50 / 20 / 5 training cells (`dataset_prep/sparse_annotation.py`) |
| `p_two` | `Cytonucs_Emese_Hela/patches_split_hela_xy2_two` | nuclei and cells annotated in different blocks (`dataset_prep/hide_level_by_blocks.py`) |
| `p_cells_only` | `Cytonucs_Emese_Hela/patches_split_hela_xy2` | cells only |
| `syn_drop_l3`, `syn_drop_l3_rerun`, `syn_drop_l3_nonest` | `Cytonucs_Synthetic/hela_v1_drop_l3` | nucleoli in 30 of the 60 training volumes (with / without the nesting losses) |
| `syn_sparse_l3` | `Cytonucs_Synthetic/hela_v1_sparse_l3` | nucleoli in 10 % of the nuclei |
| `syn_drop_l2` | `Cytonucs_Synthetic/hela_v1_drop_l2` | nuclei in 30 of the 60 training volumes |
| `syn_sparse_l2_l3` | `Cytonucs_Synthetic/hela_v1_sparse_l2_l3` | nucleoli in 25 % of the nuclei, nuclei in 25 % of the cells |

The synthetic variants are made with `dataset_prep/sparse_tiles.py` (train split only).

## Development of the recipe

| config | what |
|---|---|
| `tissuenet_cyto` | first recipe (A) |
| `tissuenet_e1_prompt` | A with more prompt augmentation (uniform prompts, repeated cell prompts, prompt consistency 0.3) |
| `tissuenet_e2_polar` | A + Polar IoU loss on the rays (B) |
| `tissuenet_e2b_polar3` | B with Polar IoU weight 3 |
| `tissuenet_e3_iouaware` | B + IoU-aware probability target (C) |
| `tissuenet_e5_wide` | C with a U-Net of base width 64 (D) |
| `tissuenet_e6_wide_long` | D for 600 epochs (E) |
| `tissuenet_e7_b32` | D with 32 crops per step, 400 epochs: the final recipe (F) |
| `tissuenet_long`, `tissuenet_e2_long` | A and B trained for 600 epochs |
| `tissuenet_noref` | A without the mask refinement |
| `helacytonuc_cyto`, `helacytonuc_noref`, `helacytonuc_e7` | A, A without refinement and F on the original HeLaCytoNuc split |
| `helacytonuc_e8_iter`, `helacytonuc_e8_rank`, `helacytonuc_e8` | F with a second refinement pass, with quality-ranking weight 0.4, with both |
| `bbbc020_cyto` | A on BBBC020 |
| `cyto_fullres_noref` | spheroid at full resolution without the mask refinement |
| `kits_e7`, `kits_e9` | KiTS23 before `kits_e8` (no object-centred crops, no nesting filter) and after it (more negative prompts inside kidneys) |

The source images and instance annotations of the HeLa-MRC-5 spheroid (`Cytonucs_Emese_Hela`) are available at
https://doi.org/10.6084/m9.figshare.24982227 (Piccinini and Diosdi, 2024); see the datasets of the paper in the main
README. The EA.hy926 volumes are in-house data; their annotations will be released with the paper.
