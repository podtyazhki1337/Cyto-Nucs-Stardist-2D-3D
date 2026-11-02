"""Boundary AP and AP by object size of saved 2D predictions, to see whether a method loses at high IoU on the contour
of large objects (LITERATURE_2026-11-01 §1).

Boundary IoU (Cheng et al., CVPR 2021): the IoU of the inner bands of width d of the two masks; as in their
boundary AP, a pair is scored by min(mask IoU, boundary IoU). The band width is given in pixels (--widths), since the
objects of the 2D sets differ by two orders of magnitude in area. AP is the measure of score_hierarchy.py: the mean
over IoU 0.50..0.95 of TP / (TP + FP + FN) with the counts pooled over the images and the matching of
stardist.matching (one-to-one, most matches at the threshold). AP by size: the ground-truth objects split at the
terciles of their area in the split; a bin counts the matches and missed objects of its ground truth and the
unmatched predictions whose area falls in it.

Usage:
  python boundary_ap.py --data DATASET_DIR --split test --method NAME=FOLDER [NAME=FOLDER ...] [--levels nucleus cytoplasm]
                        [--widths 1 2 4] [--output FILE]
FOLDER holds <level>/<name> label images (score_hierarchy.py --predictions and the baseline scorers); predicted
objects lying mostly in <split>/<level>_unknown are left out, as in score_hierarchy.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TAUS = np.arange(0.5, 1.0, 0.05)


def band(labels, width):
    """Inner band of width `width` of every object: its pixels within `width` (chessboard) of another label or
    the background."""
    from scipy import ndimage as ndi
    size = 2 * width + 1
    same = (ndi.maximum_filter(labels, size=size, mode='nearest') == labels) & \
           (ndi.minimum_filter(labels, size=size, mode='nearest') == labels)
    return np.where(same, 0, labels)


def iou_matrix(truth, pred):
    """IoU of every ground-truth object (rows, labels 1..n) with every prediction (columns, labels 1..m)."""
    from stardist.matching import label_overlap, intersection_over_union
    return intersection_over_union(label_overlap(truth, pred, check=False))[1:, 1:]


def assignment(scores, tau):
    """Matched (truth, prediction) index pairs at threshold tau, as stardist.matching."""
    from scipy.optimize import linear_sum_assignment
    n = min(scores.shape)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    costs = -(scores >= tau).astype(float) - scores / (2 * n)
    rows, cols = linear_sum_assignment(costs)
    ok = scores[rows, cols] >= tau
    return rows[ok], cols[ok]


def sequential(labels):
    from skimage.segmentation import relabel_sequential
    return relabel_sequential(np.asarray(labels).astype(np.int64))[0]


def read(path):
    import tifffile
    return sequential(tifffile.imread(path))


def area_edges(data, split, level):
    import tifffile
    areas = []
    for path in sorted((Path(data) / split / f'{level}_masks').iterdir()):
        values = tifffile.imread(path)
        areas += list(np.bincount(values.ravel())[1:][np.bincount(values.ravel())[1:] > 0])
    return np.quantile(areas, [1 / 3, 2 / 3]) if areas else np.array([0, 0])


def score(data, split, level, folder, widths, edges):
    """Pooled counts per tau of mask IoU and of min(mask, boundary) IoU per width, overall and by size bin."""
    from score_hierarchy import without_ignored
    import tifffile
    keys = ['mask'] + [f'boundary_{w}' for w in widths]
    counts = {k: np.zeros((3, len(TAUS))) for k in keys}                       # tp, fp, fn
    sized = {k: np.zeros((3, 3, len(TAUS))) for k in keys}                    # bin, tp/fp/fn, tau
    names = sorted(p.name for p in (Path(data) / split / f'{level}_masks').iterdir())
    for name in names:
        truth = read(Path(data) / split / f'{level}_masks' / name)
        pred = np.asarray(tifffile.imread(Path(folder) / level / name))
        unknown = Path(data) / split / f'{level}_unknown' / name
        if unknown.exists():
            pred = without_ignored(pred, tifffile.imread(unknown) > 0)
        pred = sequential(pred)
        mask = iou_matrix(truth, pred)
        scores = {'mask': mask}
        for w in widths:
            scores[f'boundary_{w}'] = np.minimum(mask, iou_matrix(band(truth, w), band(pred, w)))
        t_area = np.bincount(truth.ravel(), minlength=mask.shape[0] + 1)[1:]
        p_area = np.bincount(pred.ravel(), minlength=mask.shape[1] + 1)[1:]
        t_bin, p_bin = np.digitize(t_area, edges), np.digitize(p_area, edges)
        for key, s in scores.items():
            for k, tau in enumerate(TAUS):
                rows, cols = assignment(s, tau)
                tp = len(rows)
                counts[key][:, k] += (tp, s.shape[1] - tp, s.shape[0] - tp)
                for b in range(3):
                    tp_b = int(np.sum(t_bin[rows] == b))
                    fn_b = int(np.sum(t_bin == b)) - tp_b
                    fp_b = int(np.sum(np.delete(p_bin, cols) == b))
                    sized[key][b, :, k] += (tp_b, fp_b, fn_b)
    return counts, sized, len(names)


def summary(c):
    acc = c[0] / np.maximum(c.sum(0), 1)
    return dict(ap=float(acc.mean()), ap50=float(acc[0]), ap75=float(acc[5]), ap90=float(acc[8]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--method', nargs='+', required=True, help='NAME=FOLDER')
    parser.add_argument('--levels', nargs='+', default=['nucleus', 'cytoplasm'])
    parser.add_argument('--widths', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--output')
    args = parser.parse_args()
    result = dict(data=args.data, split=args.split, widths=args.widths, levels={})
    for level in args.levels:
        edges = area_edges(args.data, args.split, level)
        result['levels'][level] = dict(area_terciles=edges.tolist(), methods={})
        print(f'{level}: area terciles {edges[0]:.0f} / {edges[1]:.0f} px')
        for spec in args.method:
            name, folder = spec.split('=', 1)
            counts, sized, images = score(args.data, args.split, level, folder, args.widths, edges)
            row = {k: summary(c) for k, c in counts.items()}
            row['by_size'] = {k: [summary(s[b]) for b in range(3)] for k, s in sized.items()}
            row['images'] = images
            result['levels'][level]['methods'][name] = row
            b0 = f'boundary_{args.widths[len(args.widths) // 2]}'
            print(f"  {name:12s} mask AP {row['mask']['ap']:.3f} (AP75 {row['mask']['ap75']:.3f}) | " +
                  ' | '.join(f"boundary d={w} AP {row[f'boundary_{w}']['ap']:.3f} (AP75 {row[f'boundary_{w}']['ap75']:.3f})"
                             for w in args.widths) +
                  ' | mask AP small/mid/large ' + ' / '.join(f"{s['ap']:.3f}" for s in row['by_size']['mask']) +
                  f' | {b0} ' + ' / '.join(f"{s['ap']:.3f}" for s in row['by_size'][b0]))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1))
        print('->', args.output)


if __name__ == '__main__':
    main()
