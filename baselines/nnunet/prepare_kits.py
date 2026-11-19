"""KiTS23 in the CytoNucs conversion (dataset_prep/kits23_to_cytonucs.py, cytonucs_3x078) as an nnU-Net v2 dataset
for the nnU-Net baseline: the same CT volumes on the same 3 x 0.78 x 0.78 mm grid, labels kidney 1 / tumour 2 /
cyst 3 (the KiTS classes). A mass of the CytoNucs labels is a tumour when its number is at most the count of
connected tumours in the KiTS segmentation of the case (the converter numbers the tumours before the cysts, with the
same 26-connectivity); voxels of dropped mass fragments stay kidney, as they are kidney in the CytoNucs labels.

Writes <output>/{imagesTr,labelsTr,imagesTs}, <output>/dataset.json and <preprocessed>/splits_final.json:
imagesTr / labelsTr hold the train and val cases, the one fold trains on the CytoNucs train split and validates on
its val split; imagesTs holds the test cases. The KiTS segmentations are read from the git history of the kits23
repository (git show into a temporary file: the converter's cleanup removed them from the working tree).

Usage (nnunet env):
  python prepare_kits.py --output $nnUNet_raw/Dataset501_KiTSCytoNucs --preprocessed $nnUNet_preprocessed/Dataset501_KiTSCytoNucs
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile
from scipy import ndimage as ndi

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data', default='data/KiTS23/cytonucs_3x078')
parser.add_argument('--kits', default='data/KiTS23/kits23', help='kits23 git repository')
parser.add_argument('--output', required=True)
parser.add_argument('--preprocessed', required=True, help='nnU-Net preprocessed folder of the dataset (splits_final.json)')
args = parser.parse_args()
DATA, OUT = Path(args.data), Path(args.output)
SPACING_ZYX = json.loads((DATA / 'kits23.json').read_text())['spacing']


def tumour_count(case):
    """Connected tumours (label 2, 26-connectivity) of the case's KiTS segmentation."""
    import nibabel as nib
    blob = subprocess.run(['git', '-C', args.kits, 'show', f'HEAD:dataset/{case}/segmentation.nii.gz'],
                          capture_output=True, check=True).stdout
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
        f.write(blob)
        f.flush()
        segmentation = np.asarray(nib.load(f.name).dataobj)
    return int(ndi.label(segmentation == 2, structure=np.ones((3, 3, 3), bool))[1])


def write(array, path):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(float(s) for s in SPACING_ZYX[::-1]))       # SimpleITK: x, y, z
    sitk.WriteImage(image, str(path), useCompression=True)


def main():
    for sub in ('imagesTr', 'labelsTr', 'imagesTs'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    splits, counts = {}, {}
    for split in ('train', 'val', 'test'):
        names = sorted(p.name for p in (DATA / split / 'images').iterdir() if p.suffix == '.tif')
        splits[split] = [Path(n).stem for n in names]
        for name in names:
            case = Path(name).stem
            image = tifffile.imread(DATA / split / 'images' / name)
            write(image, OUT / ('imagesTs' if split == 'test' else 'imagesTr') / f'{case}_0000.nii.gz')
            if split == 'test':
                continue
            if (OUT / 'labelsTr' / f'{case}.nii.gz').is_file():
                continue
            kidneys = tifffile.imread(DATA / split / 'kidney_masks' / name)
            masses = tifffile.imread(DATA / split / 'mass_masks' / name)
            tumours = tumour_count(case)
            labels = np.zeros(kidneys.shape, np.uint8)
            labels[kidneys > 0] = 1
            labels[(masses > 0) & (masses <= tumours)] = 2
            labels[masses > tumours] = 3
            ids = np.unique(masses[masses > 0])
            counts[case] = dict(tumours=int(np.sum(ids <= tumours)), cysts=int(np.sum(ids > tumours)), split=split)
            write(labels, OUT / 'labelsTr' / f'{case}.nii.gz')
            print(case, split, counts[case], flush=True)
    (OUT / 'dataset.json').write_text(json.dumps(dict(
        channel_names={'0': 'CT'}, labels=dict(background=0, kidney=1, tumor=2, cyst=3),
        numTraining=len(splits['train']) + len(splits['val']), file_ending='.nii.gz',
        name='KiTS23 in the CytoNucs conversion (3 x 0.78 x 0.78 mm, HU -200..400)'), indent=1))
    (OUT / 'mass_classes.json').write_text(json.dumps(counts, indent=1))
    Path(args.preprocessed).mkdir(parents=True, exist_ok=True)
    (Path(args.preprocessed) / 'splits_final.json').write_text(json.dumps([dict(train=splits['train'], val=splits['val'])]))
    print('->', OUT, '| fold 0: train', len(splits['train']), 'val', len(splits['val']), '| test', len(splits['test']))


if __name__ == '__main__':
    main()
