# Synthetic nested data from annotation statistics

Three-level volumes (nucleolus ⊂ nucleus ⊂ cytoplasm) for CytoNucs. No data set annotates three nested levels;
these volumes are built from the statistics of an annotated two-level set, so that cells and nuclei keep realistic
shapes, nesting, packing and appearance. Only the nucleoli are assumed.

## Steps

1. **Measure** (`measure_gt_statistics.py`): on the training data only (validation and test blocks are masked
   first), per level the size, flatness, elongation, orientation, sphericity, solidity and the outline of every
   slice as a radial function r(φ) with the Fourier modes of log r; nuclei per cell, nucleus / cell volume and
   offset; packing (cell share of windows, density in the tissue envelope, nearest neighbours, contacts); appearance
   per class and channel (means, texture and its correlation length, noise against intensity, blur, profiles at the
   cell border and around the nucleus, depth profile, background haze). Whole objects are kept as donors. Writes
   `gt_statistics.json` and `gt_statistics_report.md`.
2. **Generate** (`make_synthetic_from_stats.py`): every cell and nucleus is resampled from a whole donor object
   (shape from its Fourier modes with the measured spread), placed to reproduce the measured packing, with strict
   nesting (nucleus inside its cell, nucleoli inside their nucleus); nucleoli are assumed (count, size, elongation,
   roughness and darkness ranges in `--help` and `split_info.json`). The label volumes are rendered with intensities,
   texture, blur and noise calibrated to the measured image statistics (`render_params.json`). The output is a
   CytoNucs data set (`train/val/test` with `images`, `<level>_masks`, `assignments*`, `split_info.json`).

`measure_volume()` of step 1 is used again in step 2 to compare the synthetic volumes with the measured numbers
(`synthetic_report.md`, `--preview`).

## The data set of the paper (`hela_v1`)

Source: the in-house HeLa spheroid volume at half XY resolution (`../downscale_split.py`), its 12 training blocks;
the paths of the source data are the constants at the top of `measure_gt_statistics.py`.

```bash
python dataset_prep/synthetic/measure_gt_statistics.py --output STATS --datasets hela
python dataset_prep/synthetic/make_synthetic_from_stats.py --stats STATS/gt_statistics.json --profile hela \
    --output data/Cytonucs_Synthetic/hela_v1 --counts 60 8 8 --shape 32 256 256 --seed 0
```

60 / 8 / 8 volumes of 32 × 256 × 256 voxels, appearance `realistic` (the default), seed 0. The partial-annotation
variants (`hela_v1_drop_l2`, `hela_v1_drop_l3`, `hela_v1_sparse_l3`, `hela_v1_sparse_l2_l3`) are made from it with
`../sparse_tiles.py` (train split only).

Tests: `tests/test_synthetic_from_stats.py`.
