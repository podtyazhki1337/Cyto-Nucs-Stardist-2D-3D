"""How the KiTS23 masses (tumours, cysts) sit in their kidneys, measured on the CytoNucs conversion
(dataset_prep/kits23_to_cytonucs.py, 3 x 0.78 x 0.78 mm): per mass its class, size, the share of its surface that
borders the kidney parenchyma of its kidney, another mass or the outside of the kidney (exophytic part), the share of
its volume inside the convex hull of the parenchyma, its share of the kidney, and its CT contrast to the parenchyma
and to the outside tissue around it (rings of --ring mm); per kidney its parenchyma and masses.

Parenchyma = kidney voxels of no mass. Face areas follow the voxel spacing (a face across Z is 0.78 x 0.78 mm, one
across Y or X 3 x 0.78 mm). The class of a mass comes from the converter's numbering: tumours first, then cysts, so a
mass is a tumour when its number is at most the count of connected tumours (26-connectivity) of the KiTS
segmentation, read from the git history of the kits23 repository as baselines/nnunet/prepare_kits.py does.

Writes <output>/masses.csv, kidneys.csv and summary.json.
Usage: python paper/kits23_nesting.py --data DIR --kits KITS23_REPO --output DIR [--workers 6]
"""
import argparse
import csv
import json
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

STRUCTURE = np.ones((3, 3, 3), bool)


def tumour_count(kits, case):
    import nibabel as nib
    blob = subprocess.run(['git', '-C', kits, 'show', f'HEAD:dataset/{case}/segmentation.nii.gz'],
                          capture_output=True, check=True).stdout
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
        f.write(blob)
        f.flush()
        segmentation = np.asarray(nib.load(f.name).dataobj)
    return int(ndi.label(segmentation == 2, structure=STRUCTURE)[1])


def faces(obj, other, spacing):
    """Area (mm^2) of the faces between the voxels of obj and the voxels of each region in `other` (dict of masks),
    and of the faces on the border of the array (counted as 'outside')."""
    areas = {k: 0.0 for k in other}
    area = (spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1])
    border = 0.0
    for axis in range(3):
        for step in (1, -1):
            shifted = np.roll(obj, step, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if step == 1 else -1
            shifted[tuple(edge)] = False                            # np.roll wraps around: no face there
            face = shifted & ~obj                                   # voxels next to obj along this direction
            for k, region in other.items():
                areas[k] += area[axis] * np.count_nonzero(face & region)
            border += area[axis] * np.count_nonzero(obj[tuple(edge)])
    return areas, border


def hull_share(points_mm, query_mm):
    """Share of query points inside the convex hull of points (both N x 3 in mm)."""
    if len(points_mm) < 10:
        return 0.0
    try:
        hull = ConvexHull(points_mm)
    except Exception:
        return 0.0
    a, b = hull.equations[:, :3], hull.equations[:, 3]
    inside = 0
    for start in range(0, len(query_mm), 20000):
        q = query_mm[start:start + 20000]
        inside += int(np.count_nonzero(np.all(q @ a.T + b <= 1e-6, axis=1)))
    return inside / max(len(query_mm), 1)


def case_rows(job):
    data, split, name, kits, spacing, ring = job
    case = Path(name).stem
    image = tifffile.imread(data / split / 'images' / name).astype(np.float32)
    kidneys = tifffile.imread(data / split / 'kidney_masks' / name).astype(np.int32)
    masses = tifffile.imread(data / split / 'mass_masks' / name).astype(np.int32)
    links = {int(k): int(v) for k, v in json.loads((data / split / 'assignments_mass_kidney' / f'{case}.json').read_text()).items()}
    tumours = tumour_count(kits, case)
    spacing = np.asarray(spacing, float)
    voxel = float(np.prod(spacing))
    pad = np.ceil(ring / spacing).astype(int) + 1
    rows, kidney_rows = [], []
    parenchyma_all = (kidneys > 0) & (masses == 0)
    hull_points = {}
    for k, box in enumerate(ndi.find_objects(kidneys), 1):
        if box is None:
            continue
        par = parenchyma_all[box] & (kidneys[box] == k)
        surface = par & ~ndi.binary_erosion(par)
        hull_points[k] = (np.argwhere(surface) + [s.start for s in box]) * spacing
        in_kidney = masses[box][kidneys[box] == k]
        ids = [int(i) for i in np.unique(in_kidney) if i]
        kidney_rows.append(dict(split=split, case=case, kidney=k, kidney_mm3=np.count_nonzero(kidneys[box] == k) * voxel,
                                parenchyma_mm3=np.count_nonzero(par) * voxel, masses=len(ids),
                                tumours=sum(i <= tumours for i in ids), cysts=sum(i > tumours for i in ids),
                                mass_mm3=np.count_nonzero(in_kidney) * voxel,
                                parenchyma_hu=float(image[box][par].mean()) if par.any() else None))
    kidney_mm3 = {r['kidney']: r['kidney_mm3'] for r in kidney_rows}
    for m, box in enumerate(ndi.find_objects(masses), 1):
        if box is None:
            continue
        big = tuple(slice(max(s.start - p, 0), s.stop + p) for s, p in zip(box, pad))
        obj = masses[big] == m
        k = links.get(m, int(np.bincount(kidneys[big][obj]).argmax()))
        own_par = parenchyma_all[big] & (kidneys[big] == k)
        other_mass = (masses[big] > 0) & ~obj
        outside = kidneys[big] == 0
        other_kidney = (kidneys[big] > 0) & (kidneys[big] != k) & (masses[big] == 0)
        areas, border = faces(obj, dict(parenchyma=own_par, mass=other_mass, outside=outside, other_kidney=other_kidney), spacing)
        areas['outside'] += border
        total = sum(areas.values())
        distance = ndi.distance_transform_edt(~obj, sampling=spacing)
        near = (distance > 0) & (distance <= ring)
        hu = image[big]
        par_ring, out_ring = near & own_par, near & outside
        coords = (np.argwhere(obj) + [s.start for s in big]) * spacing
        zs = np.flatnonzero(obj.any(axis=(1, 2)))
        volume = np.count_nonzero(obj) * voxel
        rows.append(dict(
            split=split, case=case, mass=m, kidney=k, cls='tumour' if m <= tumours else 'cyst', mm3=volume,
            diameter_mm=float(2 * (3 * volume / (4 * np.pi)) ** (1 / 3)), slices=int(zs.max() - zs.min() + 1),
            surface_mm2=total, share_parenchyma=areas['parenchyma'] / total, share_mass=areas['mass'] / total,
            share_outside=areas['outside'] / total, share_other_kidney=areas['other_kidney'] / total,
            in_hull=hull_share(hull_points.get(k, np.zeros((0, 3))), coords),
            share_of_kidney=volume / kidney_mm3[k] if k in kidney_mm3 else None,
            hu=float(hu[obj].mean()), hu_sd=float(hu[obj].std()),
            hu_parenchyma_ring=float(hu[par_ring].mean()) if par_ring.any() else None,
            hu_parenchyma_ring_sd=float(hu[par_ring].std()) if par_ring.any() else None,
            hu_outside_ring=float(hu[out_ring].mean()) if out_ring.any() else None,
            hu_outside_ring_sd=float(hu[out_ring].std()) if out_ring.any() else None))
    return rows, kidney_rows


def describe(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return dict(n=int(len(v)), mean=float(v.mean()), p10=float(np.percentile(v, 10)), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)))


def summary(rows, kidney_rows):
    out = {}
    for cls in ('tumour', 'cyst', 'all'):
        r = [x for x in rows if cls == 'all' or x['cls'] == cls]
        out_share = np.array([x['share_outside'] for x in r])
        par_share = np.array([x['share_parenchyma'] for x in r])
        contrast = [x['hu'] - x['hu_parenchyma_ring'] for x in r if x['hu_parenchyma_ring'] is not None]
        separation = [abs(x['hu'] - x['hu_parenchyma_ring']) / np.sqrt((x['hu_sd'] ** 2 + x['hu_parenchyma_ring_sd'] ** 2) / 2 + 1e-6)
                      for x in r if x['hu_parenchyma_ring'] is not None]
        out[cls] = dict(
            masses=len(r),
            diameter_mm=describe([x['diameter_mm'] for x in r]),
            diameter_bins={b: int(np.sum([(lo <= x['diameter_mm'] < hi) for x in r]))
                           for b, (lo, hi) in {'<10': (0, 10), '10-20': (10, 20), '20-40': (20, 40), '40-70': (40, 70), '>=70': (70, 1e9)}.items()},
            share_outside=describe(out_share), share_parenchyma=describe(par_share),
            share_mass=describe([x['share_mass'] for x in r]),
            enclosed=dict(  # by the exophytic share of the surface
                fully_inside_lt5=int(np.sum(out_share < 0.05)), mostly_inside_5_50=int(np.sum((out_share >= 0.05) & (out_share < 0.5))),
                mostly_outside_ge50=int(np.sum(out_share >= 0.5)), no_parenchyma_contact=int(np.sum(par_share == 0))),
            touches_other_mass=int(np.sum([x['share_mass'] > 0 for x in r])),
            in_hull=describe([x['in_hull'] for x in r]),
            in_hull_ge90=int(np.sum([x['in_hull'] >= 0.9 for x in r])),
            share_of_kidney=describe([x['share_of_kidney'] for x in r]),
            share_of_kidney_ge50=int(np.sum([(x['share_of_kidney'] or 0) >= 0.5 for x in r])),
            hu=describe([x['hu'] for x in r]), hu_parenchyma_ring=describe([x['hu_parenchyma_ring'] for x in r]),
            contrast_to_parenchyma_hu=describe(contrast),
            low_contrast_lt30hu=int(np.sum(np.abs(contrast) < 30)),
            separation_d=describe(separation))
    with_masses = [k for k in kidney_rows if k['masses']]
    out['kidneys'] = dict(
        kidneys=len(kidney_rows), with_masses=len(with_masses),
        masses_per_kidney_with_masses=describe([k['masses'] for k in with_masses]),
        mass_share_of_kidney=describe([k['mass_mm3'] / k['kidney_mm3'] for k in with_masses]),
        parenchyma_below_10pct=int(np.sum([k['parenchyma_mm3'] < 0.1 * k['kidney_mm3'] for k in kidney_rows])))
    per_case = {}
    for k in kidney_rows:
        per_case[k['case']] = per_case.get(k['case'], 0) + 1
    out['kidneys']['cases_by_kidney_count'] = {str(n): int(sum(v == n for v in per_case.values())) for n in sorted(set(per_case.values()))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data/KiTS23/cytonucs_3x078')
    parser.add_argument('--kits', default='data/KiTS23/kits23')
    parser.add_argument('--output', required=True)
    parser.add_argument('--ring', type=float, default=5.0, help='width (mm) of the tissue rings around a mass')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    spacing = json.loads((data / 'kits23.json').read_text())['spacing']
    jobs = [(data, split, name, args.kits, spacing, args.ring) for split in ('train', 'val', 'test')
            for name in sorted(p.name for p in (data / split / 'mass_masks').iterdir() if p.suffix == '.tif')]
    rows, kidney_rows = [], []
    with Pool(args.workers) as pool:
        for k, (r, kr) in enumerate(pool.imap(case_rows, jobs), 1):
            rows += r
            kidney_rows += kr
            if k % 25 == 0:
                print(k, 'of', len(jobs), flush=True)
    for path, table in (('masses.csv', rows), ('kidneys.csv', kidney_rows)):
        with open(out / path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    result = {split: summary([r for r in rows if split == 'all' or r['split'] == split],
                             [k for k in kidney_rows if split == 'all' or k['split'] == split]) for split in ('all', 'val', 'test')}
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result['all'], indent=1))


if __name__ == '__main__':
    main()
