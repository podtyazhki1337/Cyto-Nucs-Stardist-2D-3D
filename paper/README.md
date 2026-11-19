# Reproducing the paper

The models of the paper are trained from `configs/` (see `configs/README.md`) and scored with the scripts of the
repository root; the baselines are in `baselines/`. This folder holds the scripts that only the paper's analyses
need. Run them from the repository root (`python paper/<script>.py ...`); each prints its usage with `--help`.

| Paper item | How |
|---|---|
| Main comparison (TissueNet, HeLaCytoNuc, BBBC020) | CytoNucs: `score_hierarchy.py` (BBBC020 with the settings calibrated on val by `calibrate_inference.py`); baselines: `baselines/stardist3d/run_files.py`, `baselines/hydrastardist/run_hsd.py`, `baselines/cellpose/run_cellpose.py`, `baselines/instanseg/run_instanseg.py`, scored with `score_predictions.py` |
| Baselines at the operating point of CytoNucs' calibration | `paper/calibrate_objective.py` (each baseline's thresholds chosen on val for 1/2 linked JTPR + 1/2 mean AP) |
| HeLa spheroid (3D, spatial split) | `evaluate_blocks.py` (whole held-out blocks); baselines `baselines/stardist3d/run_fullres.py`, `baselines/cellpose/run_cellpose3d.py` |
| KiTS23 | `score_hierarchy.py`; baselines `baselines/stardist3d/run_files.py`, `baselines/cellpose/run_cellpose3d_files.py`, `baselines/nnunet/` |
| KiTS23 semantic gate | `paper/semantic_scores.py`, then `paper/semantic_gate.py` |
| How the KiTS23 masses sit in their kidneys | `paper/kits23_nesting.py` |
| Three-level synthetic data | `dataset_prep/synthetic/` (data), `dataset_prep/sparse_tiles.py` (partly annotated variants), `score_hierarchy.py` |
| Partial annotation (spheroid) | `dataset_prep/sparse_annotation.py`, `dataset_prep/hide_level_by_blocks.py`, `evaluate_blocks.py` |
| Ablations; separate single-level networks | `score_hierarchy.py`; the two separate networks are paired by overlap with `score_predictions.py` (their saved labels in one folder per level) |
| Inference time and memory | `paper/time_inference.py` (seconds per image and per stage), `paper/gpu_monitor.py` (GPU and host memory of a run) |
| Why annotated pairs are missed (inner / outer / link) | `score_predictions.py` and `score_hierarchy.py` (`fail_inner_*`, `fail_outer_*`, `fail_link_*`) |
| Linked JTPR at IoU 0.3 and 0.7 | `score_predictions.py --jtpr-ious 0.3 0.7` |
| Metrics of the HydraStarDist paper | `score_hierarchy.py` / `score_predictions.py` (`paper_<pair>`: IoUR, AP and joint TP rates at τ 0.1-0.9) |
| Bootstrap intervals | the per-image records (`records`) of every score file |
| Operating points: quality gates for AP alone | `paper/recalibrate_gates.py` (gates from the full-validation log of a run, into `inference.quality_gates_override`) |
| Boundary AP and AP by object size | `paper/boundary_ap.py` |
| Learned merge / drop decisions for neighbouring objects | `paper/pair_decisions.py` |
| HeLaCytoNuc clean-up (Supplement S2.1) | `dataset_prep/drop_incomplete_annotation.py` (writes `excluded.json`), `dataset_prep/fix_label_overflow.py` (writes `label_overflow.json`) |
| HydraStarDist+WBR (Supplement S3.1) | `baselines/hydrastardist/run_hsd.py --wbr` (the paper's WBR made differentiable), gradient check `baselines/hydrastardist/check_wbr_gradient.py` |
