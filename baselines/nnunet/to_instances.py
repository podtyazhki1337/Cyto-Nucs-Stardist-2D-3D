"""Instances of nnU-Net's KiTS predictions (labels kidney 1 / tumour 2 / cyst 3, prepare_kits.py) by the rule of the
CytoNucs labels (dataset_prep/kits23_to_cytonucs.py): a kidney is a connected region of the foreground, a mass is a
connected tumour or a connected cyst (26-connectivity), kidneys below 1600 mm^3 and masses below 32 mm^3 (or in a
dropped kidney) are removed. Writes <output>/<split>/{mass,kidney}/<case>.tif, scored with score_predictions.py
--levels mass kidney like every other method.

Usage (nnunet env): python to_instances.py --predictions DIR_OF_NIFTI --output DIR --split test
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile
from scipy import ndimage as ndi

MIN_KIDNEY_MM3, MIN_MASS_MM3 = 1600., 32.


def instances(labels, voxel):
    structure = np.ones((3, 3, 3), bool)
    kidneys = ndi.label(labels > 0, structure=structure)[0]
    masses = np.zeros(labels.shape, np.uint16)
    count = 0
    for value in (2, 3):                                               # tumours, then cysts
        parts, n = ndi.label(labels == value, structure=structure)
        masses[parts > 0] = parts[parts > 0] + count
        count += n
    sizes = np.bincount(kidneys.ravel())
    dropped = np.flatnonzero(sizes * voxel < MIN_KIDNEY_MM3)
    dropped = dropped[dropped > 0]
    kidneys[np.isin(kidneys, dropped)] = 0
    sizes = np.bincount(masses.ravel())
    small = np.flatnonzero((sizes > 0) & (sizes * voxel < MIN_MASS_MM3))
    masses[np.isin(masses, small[small > 0]) | (kidneys == 0)] = 0
    return kidneys.astype(np.uint16), masses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True, help='folder of nnU-Net predictions <case>.nii.gz')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', required=True)
    args = parser.parse_args()
    files = sorted(Path(args.predictions).glob('case_*.nii.gz'))
    for level in ('mass', 'kidney'):
        (Path(args.output) / args.split / level).mkdir(parents=True, exist_ok=True)
    for path in files:
        image = sitk.ReadImage(str(path))
        voxel = float(np.prod(image.GetSpacing()))
        kidneys, masses = instances(sitk.GetArrayFromImage(image), voxel)
        case = path.name[:-len('.nii.gz')]
        for level, values in (('mass', masses), ('kidney', kidneys)):
            tifffile.imwrite(Path(args.output) / args.split / level / f'{case}.tif', values, photometric='minisblack',
                             compression='zlib')
    print(len(files), 'cases ->', Path(args.output) / args.split)


if __name__ == '__main__':
    main()
