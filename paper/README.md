# Reproducing the paper

The models of the paper are trained from `configs/` (see `configs/README.md`) and scored with the scripts of the
repository root; the baselines are in `baselines/`. This folder holds the scripts that only the paper's analyses
need. Run them from the repository root (`python paper/<script>.py ...`); each prints its usage with `--help`.

| Paper item | How |
|---|---|
| Main comparison (TissueNet, HeLaCytoNuc, BBBC020) | CytoNucs: `score_hierarchy.py` (BBBC020 with the settings calibrated on val by `calibrate_inference.py`); baselines: `baselines/stardist3d/run_files.py`, `baselines/hydrastardist/run_hsd.py`, `baselines/cellpose/run_cellpose.py`, `baselines/instanseg/run_instanseg.py`, scored with `score_predictions.py` |
| Baselines at the operating point of CytoNucs' calibration | `paper/calibrate_objective.py` (each baseline's thresholds chosen on val for 1/2 linked JTPR + 1/2 mean AP), `paper/calibrate_objective_instanseg.py` (InstanSeg: its seed and mask thresholds, both levels together, with its own links) |
| HeLa spheroid (3D, spatial split) | `evaluate_blocks.py` (whole held-out blocks); baselines `baselines/stardist3d/run_fullres.py`, `baselines/cellpose/run_cellpose3d.py` |
| KiTS23 | `score_hierarchy.py`; baselines `baselines/stardist3d/run_files.py`, `baselines/cellpose/run_cellpose3d_files.py`, `baselines/nnunet/` |
| KiTS23 semantic gate | `paper/semantic_scores.py`, then `paper/semantic_gate.py` |
| How the KiTS23 masses sit in their kidneys | `paper/kits23_nesting.py` |
| Three-level synthetic data | `dataset_prep/synthetic/` (data), `dataset_prep/sparse_tiles.py` (partly annotated variants), `score_hierarchy.py` |
| Partial annotation (spheroid) | `dataset_prep/sparse_annotation.py`, `dataset_prep/hide_level_by_blocks.py`, `evaluate_blocks.py` |
| Ablations; separate single-level networks | `score_hierarchy.py`; the two separate networks are paired by overlap with `score_predictions.py` (their saved labels in one folder per level), and recalibrated jointly for CytoNucs' objective with `paper/recal_single_level_pairs.py`; prompt ablations with `inference.outer_from_inner_prompts` / `inference.outer_own_prompts` false |
| Joint vs separate training at matched calibration (same objective, same threshold budget) | `paper/operating_points.py` (2D), `paper/operating_points_3d.py` (test blocks of the spheroid) |
| Pairs whose cell changes without the cells from nucleus prompts (Figure 6C) | `paper/prompt_source_pairs.py` |
| Inference time and memory | `paper/time_inference.py` (seconds per image and per stage), `paper/time_prompts_and_pair.py` (seconds against the number of prompts; two separate networks plus their pairing), `paper/gpu_monitor.py` (GPU and host memory of a run) |
| Why annotated pairs are missed (inner / outer / link) | `score_predictions.py` and `score_hierarchy.py` (`fail_inner_*`, `fail_outer_*`, `fail_link_*`) |
| Linked JTPR at IoU 0.3 and 0.7 | `score_predictions.py --jtpr-ious 0.3 0.7` |
| Metrics of the HydraStarDist paper | `score_hierarchy.py` / `score_predictions.py` (`paper_<pair>`: IoUR, AP and joint TP rates at τ 0.1-0.9) |
| Bootstrap intervals of linked JTPR | `paper/bootstrap_jtpr.py` (2D test splits, paired over the annotated pairs), `paper/bootstrap_jtpr_blocks.py` (test blocks of the spheroid) |
| Image-level bootstrap intervals of AP and JTPR, seeds averaged | `paper/bootstrap_images.py` (whole images resampled, or whole TissueNet experiments with `--clusters`) |
| Pair precision and F1, states of the annotated pairs, fallback links; cells grouped by their annotated nuclei and the nucleus / cell readouts; readouts on the cells two methods both measure | `paper/pair_metrics.py pairs`, `cells`, `common` |
| Whole-chain recall of the three-level synthetic data | `paper/chain_recall.py` |
| Checkpoint metric: pairs lost to the gates of both levels | `paper/double_subtraction.py` (`history`: bound per logged validation; `gates`: exact re-assembly and gate search) |
| KiTS23 masses found by size | `paper/kits_mass_recall_by_size.py` |
| How children sit in their parents (nesting statistics of the datasets) | `paper/nesting_geometry.py` |
| Operating points: quality gates for AP alone | `paper/recalibrate_gates.py` (gates from the full-validation log of a run, into `inference.quality_gates_override`) |
| Boundary AP and AP by object size | `paper/boundary_ap.py` |
| Learned merge / drop decisions for neighbouring objects | `paper/pair_decisions.py` |
| HeLaCytoNuc clean-up (Supplement S2.1) | `dataset_prep/drop_incomplete_annotation.py` (writes `excluded.json`), `dataset_prep/fix_label_overflow.py` (writes `label_overflow.json`) |
| HydraStarDist+WBR (Supplement S3.1) | `baselines/hydrastardist/run_hsd.py --wbr` (the paper's WBR made differentiable), gradient check `baselines/hydrastardist/check_wbr_gradient.py` |
