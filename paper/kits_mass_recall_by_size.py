"""KiTS23 test masses by size: the diameter of a sphere of the same volume (voxel 3 x 0.78 x 0.78 mm), bins <= 4 cm,
4-7 cm, > 7 cm and all; per bin the annotated masses and the share matched at IoU 0.5 (StarDist matching, one-to-one,
per volume) for each method's saved test predictions of the mass level.

Usage:
  python paper/kits_mass_recall_by_size.py --data data/KiTS23/cytonucs_3x078_val16/test --output OUT_DIR \\
      --method "CytoNucs=runs/kits_e8/predictions_test/mass" --method "nnU-Net=runs/nnunet/kits/predictions/test/mass" ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from stardist.matching import matching

BINS = [('<= 4 cm', 0, 40), ('4-7 cm', 40, 70), ('> 7 cm', 70, np.inf)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='the test split (images/, mass_masks/)')
    parser.add_argument('--method', action='append', required=True, metavar='NAME=FOLDER',
                        help='saved mass labels of a method, one file per test volume')
    parser.add_argument('--output', required=True)
    parser.add_argument('--voxel', type=float, nargs=3, default=[3.0, 0.78, 0.78], help='voxel size ZYX in mm')
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    methods = dict(m.split('=', 1) for m in args.method)
    voxel_mm3 = float(np.prod(args.voxel))
    names = sorted(p.name for p in (data / 'images').iterdir())
    diameters, matched = [], {m: [] for m in methods}
    for name in names:
        gt = tifffile.imread(data / 'mass_masks' / name)
        ids = np.unique(gt)
        ids = ids[ids > 0]
        if not len(ids):
            continue
        volumes = np.bincount(gt.ravel())[ids] * voxel_mm3
        diameters += list((6 * volumes / np.pi) ** (1 / 3))
        for method, folder in methods.items():
            pred = tifffile.imread(Path(folder) / name)
            result = matching(gt, pred, thresh=0.5, criterion='iou', report_matches=True)
            hit = {int(g) for (g, _), s in zip(result.matched_pairs, result.matched_scores) if s >= 0.5}
            matched[method] += [int(i) in hit for i in ids]
        print(name, len(ids), flush=True)
    diameters = np.array(diameters)
    rows = {}
    for label, lo, hi in BINS + [('all', 0, np.inf)]:
        sel = (diameters > lo) & (diameters <= hi) if label != 'all' else np.ones(len(diameters), bool)
        rows[label] = dict(masses=int(sel.sum()),
                           **{m: (float(np.mean(np.array(v)[sel])) if sel.any() else None) for m, v in matched.items()})
    (out / 'mass_recall_by_size.json').write_text(json.dumps(dict(voxel_mm=args.voxel, diameters_mm=diameters.round(1).tolist(),
                                                                   bins=rows), indent=1))
    lines = ['| mass size (equivalent sphere diameter) | annotated masses | ' + ' | '.join(methods) + ' |',
             '|---|---|' + '---|' * len(methods)]
    for label, r in rows.items():
        lines.append(f'| {label} | {r["masses"]} | ' + ' | '.join('—' if r[m] is None else f'{r[m]:.3f}' for m in methods) + ' |')
    (out / 'mass_recall_by_size.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
