# Splits of the paper

Explicit file lists of the splits the converters in `dataset_prep/` write (they are reproducible from the public data
with the commands of the main README; these lists let you check a conversion, or use the splits with other tools).

| file | dataset | content |
|---|---|---|
| `tissuenet_v1.1.json` | TissueNet v1.1 | every tile of train / val (the 256 validation tiles for checkpoint selection and calibration) / val_all / test / holdout with its source split, index in the published npz and experiment (the unit of the experiment-level bootstrap) |
| `helacytonuc_clean.json` | HeLaCytoNuc, cleaned split | the images kept per split, the excluded images with their annotated and visible nuclei, the relabelled 8-bit test masks |
| `bbbc020.json` | BBBC020 | train 12 / val 4 / test 4 images |
| `kits23.json` | KiTS23 | train 342 / val 73 / test 74 cases, and the 16 validation cases used for checkpoint selection |

The HeLa-MRC-5 spheroid split is `../hela_spheroid_blocks.json` (blocks of the volume, see the main README).
