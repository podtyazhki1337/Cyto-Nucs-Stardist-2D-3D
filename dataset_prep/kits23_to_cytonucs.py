"""KiTS23 (kidney, renal tumour and cyst CT) as a two-level CytoNucs dataset: kidneys with the masses inside.

Levels (model.levels ["mass", "kidney"]):
  kidney  every connected kidney region of the consensus segmentation (kidney + tumour + cyst voxels), so a
          kidney contains its masses;
  mass    every connected tumour and every connected cyst (labelled separately) inside a kidney.
Images are CT in HU clipped to --window, resampled to --spacing mm ZYX (default the dataset median 3 x 0.78 x
0.78: in-plane resolution kept, slices of 0.5-5 mm brought to one thickness; images linearly, labels by nearest
neighbour) and cropped in XY to the body (HU > -300; no use of the annotation, so the network still has to
find the kidneys). Cases are split at random into train / val / test (--fractions, --seed).

Writes <output>/{train,val,test}/{images,mass_masks,kidney_masks,assignments_mass_kidney}/case_xxxxx.tif(.json),
meta.json per split, split_info.json, kits23.json (spacing, crop and object counts per case).

The imaging is downloaded case by case from the KiTS23 Hugging Face mirror (as kits23/download.py does) into
<kits>/dataset/<case>/imaging.nii.gz; with --delete-raw the downloaded CT is removed once converted (it can be
downloaded again any time; the annotation of the repository is kept).

Usage:
  python dataset_prep/kits23_to_cytonucs.py --kits data/KiTS23/kits23 \\
      --output data/KiTS23/cytonucs_3x078 [--spacing 3.0 0.78 0.78] [--delete-raw] [--limit N]
Data licence: CC BY-NC-SA 4.0 (KiTS23); cite Heller et al., KiTS21/23.
"""
import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

URL = 'https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/{case}.nii.gz'
CASES = list(range(300)) + list(range(400, 589))                     # KiTS23 training cases
ORIENTATION = ('I', 'P', 'L')                                        # axes (Z, Y, X) of every case
MIN_KIDNEY_MM3, MIN_MASS_MM3 = 1600., 32.                           # fragments of the resampling below these are dropped


def download(case, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.parent / f'.partial.{path.name}'
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(URL.format(case=case), str(partial))
            shutil.move(str(partial), str(path))
            return
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == 2:
                raise
            time.sleep(10)


def load(path):
    """Array in (Z, Y, X) = (I, P, L) order and its voxel spacing in mm."""
    import nibabel as nib
    image = nib.load(str(path))
    codes = nib.aff2axcodes(image.affine)
    data = np.asarray(image.dataobj)
    if codes != ORIENTATION:
        transform = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(codes),
                                                    nib.orientations.axcodes2ornt(ORIENTATION))
        data = nib.orientations.apply_orientation(data, transform)
        spacing = np.asarray(image.header.get_zooms()[:3])[np.argsort([int(t[0]) for t in transform])]
    else:
        spacing = np.asarray(image.header.get_zooms()[:3])
    return data, np.asarray(spacing, np.float64)


def instances(segmentation):
    """(kidney labels, mass labels, mass -> kidney) at the resolution of the segmentation."""
    structure = np.ones((3, 3, 3), bool)
    kidneys, n_kidneys = ndi.label(segmentation > 0, structure=structure)
    masses = np.zeros(segmentation.shape, np.uint16)
    count = 0
    for value in (2, 3):                                               # tumours, then cysts
        parts, n = ndi.label(segmentation == value, structure=structure)
        masses[parts > 0] = parts[parts > 0] + count
        count += n
    links = {}
    for mass, box in enumerate(ndi.find_objects(masses), 1):
        if box is not None:
            under = kidneys[box][masses[box] == mass]
            links[mass] = int(np.bincount(under).argmax())            # a mass lies inside one kidney region
    return kidneys.astype(np.uint16), masses, links


def resample(array, zoom, order):
    return ndi.zoom(array, zoom, order=order, mode='nearest', grid_mode=False)


def convert_case(case, kits, spacing, window):
    folder = kits / 'dataset' / case
    image, image_spacing = load(folder / 'imaging.nii.gz')
    segmentation, _ = load(folder / 'segmentation.nii.gz')
    kidneys, masses, links = instances(segmentation)
    zoom = image_spacing / spacing
    # XY crop to the body: largest connected region above -300 HU in the maximum projection along Z, taken
    # from the CT before the window (the window clips air to its lower bound) and scaled to the new grid
    body, n = ndi.label(image.max(axis=0) > -300)
    ct = resample(np.clip(image.astype(np.float32), *window), zoom, 1)
    kidneys, masses = (resample(labels, zoom, 0) for labels in (kidneys, masses))
    if n:
        sizes = ndi.sum(np.ones_like(body), body, range(1, n + 1))
        ys, xs = np.nonzero(body == 1 + int(np.argmax(sizes)))
        y0, x0 = (max(int(np.floor(v.min() * z)) - 4, 0) for v, z in ((ys, zoom[1]), (xs, zoom[2])))
        y1, x1 = (int(np.ceil((v.max() + 1) * z)) + 4 for v, z in ((ys, zoom[1]), (xs, zoom[2])))
    else:
        y0, y1, x0, x1 = 0, ct.shape[1], 0, ct.shape[2]
    crop = (slice(None), slice(y0, y1), slice(x0, x1))
    outside = int(np.count_nonzero(kidneys)) - int(np.count_nonzero(kidneys[crop]))
    if outside:
        raise RuntimeError(f'{case}: the body crop {crop} cuts {outside} kidney voxels')
    ct, kidneys, masses = ct[crop], kidneys[crop], masses[crop]
    # Fragments of the resampling: kidneys below MIN_KIDNEY voxels, masses below MIN_MASS voxels, and the
    # masses of a dropped kidney, are removed.
    voxel = float(np.prod(spacing))
    sizes = dict(zip(*np.unique(kidneys, return_counts=True)))
    dropped_kidneys = {int(v) for v, c in sizes.items() if v and c * voxel < MIN_KIDNEY_MM3}
    kidneys[np.isin(kidneys, sorted(dropped_kidneys))] = 0
    sizes = dict(zip(*np.unique(masses, return_counts=True)))
    dropped_masses = [int(v) for v, c in sizes.items() if v and (c * voxel < MIN_MASS_MM3 or links.get(int(v)) in dropped_kidneys)]
    masses[np.isin(masses, dropped_masses)] = 0
    present = {int(v) for v in np.unique(masses) if v}
    kept = {int(v) for v in np.unique(kidneys) if v}
    links = {m: k for m, k in links.items() if m in present and k in kept}
    info = dict(case=case, source_spacing=image_spacing.tolist(), source_shape=list(image.shape), shape=list(ct.shape),
                crop_yx=[int(y0), int(y1), int(x0), int(x1)], kidneys=len(kept), masses=len(present),
                masses_per_kidney=np.bincount(list(links.values()), minlength=max(kept, default=0) + 1)[1:].tolist())
    return np.round(ct).astype(np.int16), kidneys, masses, links, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kits', required=True, help='clone of github.com/neheller/kits23 (annotation in dataset/)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--spacing', type=float, nargs=3, default=[3.0, 0.78, 0.78],
                        help='target voxel size ZYX in mm (default: the median spacing of KiTS23; slices are 0.5-5 mm thick)')
    parser.add_argument('--window', type=float, nargs=2, default=[-200., 400.])
    parser.add_argument('--fractions', type=float, nargs=3, default=[.7, .15, .15])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--delete-raw', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    kits, output = Path(args.kits), Path(args.output)
    cases = [f'case_{n:05d}' for n in CASES]
    order = np.random.default_rng(args.seed).permutation(len(cases))
    n_train, n_val = int(round(args.fractions[0] * len(cases))), int(round(args.fractions[1] * len(cases)))
    split_of = {cases[i]: ('train' if k < n_train else 'val' if k < n_train + n_val else 'test') for k, i in enumerate(order)}
    if args.limit:
        cases = cases[:args.limit]
    log_path = output / 'kits23.json'
    log = json.loads(log_path.read_text()) if log_path.is_file() else dict(spacing=args.spacing, window=args.window,
                                                                           seed=args.seed, cases={})
    for case in cases:
        split = split_of[case]
        target = output / split / 'images' / f'{case}.tif'
        if target.is_file() and case in log['cases']:
            continue
        raw = kits / 'dataset' / case / 'imaging.nii.gz'
        downloaded = not raw.is_file()
        if downloaded:
            download(case, raw)
        ct, kidneys, masses, links, info = convert_case(case, kits, np.asarray(args.spacing, np.float64), tuple(args.window))
        for sub in ('images', 'mass_masks', 'kidney_masks', 'assignments_mass_kidney'):
            (output / split / sub).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output / split / 'mass_masks' / f'{case}.tif', masses, photometric='minisblack', compression='zlib')
        tifffile.imwrite(output / split / 'kidney_masks' / f'{case}.tif', kidneys, photometric='minisblack', compression='zlib')
        (output / split / 'assignments_mass_kidney' / f'{case}.json').write_text(json.dumps(links))
        tifffile.imwrite(target, ct, photometric='minisblack', compression='zlib')     # last: marks the case done
        log['cases'][case] = dict(info, split=split, downloaded=downloaded)
        log_path.write_text(json.dumps(log, indent=1))
        if args.delete_raw and downloaded:
            raw.unlink()
        print(f"{case} {split}: {info['shape']} | kidneys {info['kidneys']}, masses {info['masses']}", flush=True)
    for split in ('train', 'val', 'test'):
        names = sorted(p.name for p in (output / split / 'images').glob('*.tif')) if (output / split / 'images').is_dir() else []
        meta = [dict(filename=n, n_cells=log['cases'][Path(n).stem]['kidneys'], n_nuclei=log['cases'][Path(n).stem]['masses'])
                for n in names]
        if names:
            (output / split / 'meta.json').write_text(json.dumps(meta, indent=1))
    info = {split: sorted(c + '.tif' for c, v in log['cases'].items() if v['split'] == split) for split in ('train', 'val', 'test')}
    (output / 'split_info.json').write_text(json.dumps(dict(info, seed=args.seed, split='kits23_random_cases', rescale_factor=1.0,
                                                             spacing_mm=args.spacing, levels=['mass', 'kidney']), indent=1))


if __name__ == '__main__':
    main()
