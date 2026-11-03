"""Learned decisions for neighbouring predicted objects of saved 2D predictions: merge the two, drop the smaller, or
keep both (LITERATURE_2026-11-01 §2.2, in the spirit of learned NMS). Targets the extra objects of CytoNucs (halves
of one cell, a small false object next to a true one) without retraining the model.

Pairs: objects of one level that touch (4-neighbourhood). Features of a pair, from the label maps and the image only
(S the smaller, L the larger): areas against the image's median object, area ratio, contact length over each
perimeter, solidity of S, L and their union, centroid distance, image contrast along the contact against both
interiors, and for cells the predicted nuclei: in S, in L, and one nucleus lying across the contact. Targets from the
ground truth: merge when the union matches a ground-truth object (IoU >= 0.5) better than either part and neither
part matches another one; drop S when S matches nothing (IoU < 0.25); keep otherwise.

The classifier (gradient boosting) is trained on the predictions of a training split (val), its thresholds chosen on
out-of-fold probabilities (folds by image) for the AP of the level; then it is applied to the test predictions and
both are scored with score_hierarchy.score_image (links by overlap before and after, so the JTPRs compare).

Usage:
  python pair_decisions.py --data DATASET_DIR --level cytoplasm --train val=PRED_FOLDER [--train ...]
                           --test test=PRED_FOLDER [--output-predictions FOLDER] [--output FILE]
(PRED_FOLDER holds <level>/<name>; several --train sets pool their pairs, with sizes relative to each image.)
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

KEEP, MERGE, DROP = 0, 1, 2
TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def sequential(labels):
    from skimage.segmentation import relabel_sequential
    return relabel_sequential(np.asarray(labels).astype(np.int64))[0]


def normalized_image(image):
    image = np.asarray(image, np.float32)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[0] <= 4 and image.shape[-1] > 4:                       # channels first
        image = np.moveaxis(image, 0, -1)
    lo, hi = np.percentile(image, 1, axis=(0, 1)), np.percentile(image, 99.8, axis=(0, 1))
    return (image - lo) / np.maximum(hi - lo, 1e-6)


def contacts(labels):
    """(pairs (n, 2) with a < b, contact pixel count, per-pixel pair index and position lists) of touching objects."""
    keys, rows, cols = [], [], []
    m = int(labels.max()) + 1
    for axis in (0, 1):
        a = labels[:-1, :] if axis == 0 else labels[:, :-1]
        b = labels[1:, :] if axis == 0 else labels[:, 1:]
        sel = (a != b) & (a > 0) & (b > 0)
        r, c = np.nonzero(sel)
        lo, hi = np.minimum(a[sel], b[sel]), np.maximum(a[sel], b[sel])
        key = lo * m + hi
        keys += [key, key]                                                 # both pixels of the contact
        rows += [r, r + (axis == 0)]
        cols += [c, c + (axis == 1)]
    if not keys:
        return np.zeros((0, 2), int), np.zeros(0), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    keys, rows, cols = np.concatenate(keys), np.concatenate(rows), np.concatenate(cols)
    unique, index, count = np.unique(keys, return_inverse=True, return_counts=True)
    pairs = np.stack([unique // m, unique % m], 1)
    return pairs, count / 2., index, rows, cols


def object_stats(labels):
    """Per object (index = label): area, perimeter (pixels next to another label or the background), centroid,
    convex-hull vertices."""
    from scipy.spatial import ConvexHull
    from skimage.measure import regionprops
    n = int(labels.max())
    area = np.bincount(labels.ravel(), minlength=n + 1).astype(float)
    padded = np.pad(labels, 1, mode='edge')
    edge = np.zeros(labels.shape, bool)
    for dr, dc in ((0, 1), (1, 0), (1, 2), (2, 1)):
        edge |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]] != labels
    perimeter = np.bincount(labels[edge & (labels > 0)], minlength=n + 1).astype(float)
    centroid = np.zeros((n + 1, 2))
    hulls = [None] * (n + 1)
    for region in regionprops(labels):
        centroid[region.label] = region.centroid
        pts = region.coords.astype(float)
        try:
            hulls[region.label] = pts[ConvexHull(pts).vertices] if len(pts) >= 3 else pts
        except Exception:                                                 # collinear pixels
            hulls[region.label] = pts
    return area, perimeter, centroid, hulls


def hull_area(points):
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return float(len(points))
    try:
        hull = ConvexHull(points)
        return hull.volume + hull.area / 2 + 1                            # pixel centres -> pixel area
    except Exception:
        return float(len(points))


def features(image, labels, nuclei=None):
    """(pairs (n, 2) as (S, L), features (n, F)) of the touching objects of `labels`."""
    labels = sequential(labels)
    pairs, contact, index, rows, cols = contacts(labels)
    if not len(pairs):
        return pairs, np.zeros((0, 23 if nuclei is not None else 19)), labels
    area, perimeter, centroid, hulls = object_stats(labels)
    channels = image.shape[-1]
    inside = np.stack([np.bincount(labels.ravel(), weights=image[..., k].ravel(), minlength=len(area))
                       for k in range(channels)], 1) / np.maximum(area[:, None], 1)
    along = np.stack([np.bincount(index, weights=image[rows, cols, k], minlength=len(pairs)) for k in range(channels)], 1) \
        / np.maximum(np.bincount(index, minlength=len(pairs))[:, None], 1)
    small = np.where(area[pairs[:, 0]] <= area[pairs[:, 1]], pairs[:, 0], pairs[:, 1])
    large = np.where(area[pairs[:, 0]] <= area[pairs[:, 1]], pairs[:, 1], pairs[:, 0])
    median = np.median(area[1:][area[1:] > 0]) if np.any(area[1:] > 0) else 1.
    solidity = np.array([area[i] / hull_area(hulls[i]) if hulls[i] is not None else 1. for i in range(len(area))])
    union_solidity = np.array([(area[s] + area[l]) / hull_area(np.vstack([hulls[s], hulls[l]])) for s, l in zip(small, large)])
    cols_ = []
    for k in range(min(channels, 3)):
        cols_ += [along[:, k] - inside[small, k], along[:, k] - inside[large, k], inside[small, k] - inside[large, k]]
    while len(cols_) < 9:
        cols_.append(np.zeros(len(pairs)))
    x = [np.log(area[small] / median), np.log(area[large] / median), area[small] / area[large],
         contact / np.maximum(perimeter[small], 1), contact / np.maximum(perimeter[large], 1),
         solidity[small], solidity[large], union_solidity, union_solidity - np.maximum(solidity[small], solidity[large]),
         np.linalg.norm(centroid[small] - centroid[large], axis=1) / np.sqrt(area[small] + area[large])] + cols_
    if nuclei is not None:
        from stardist.matching import label_overlap
        nuc = sequential(nuclei)
        overlap = label_overlap(labels, nuc, check=False)[:, 1:].astype(float)     # (objects + 1, nuclei)
        nuc_area = np.maximum(np.bincount(nuc.ravel(), minlength=overlap.shape[1] + 1)[1:], 1)
        share = overlap / nuc_area[None, :]
        inside_count = (share >= .5).sum(1)
        across = np.minimum(share[small], share[large]).max(1) if share.shape[1] else np.zeros(len(pairs))
        x += [inside_count[small], inside_count[large], across, (inside_count[small] == 0).astype(float)]
    return np.stack([small, large], 1), np.stack(x, 1), labels


def targets(labels, truth, pairs):
    """KEEP / MERGE / DROP per pair (S, L) from the ground truth."""
    from stardist.matching import label_overlap
    truth = sequential(truth)
    overlap = label_overlap(labels, truth, check=False).astype(float)[:, 1:]       # (objects + 1, truth)
    area = np.bincount(labels.ravel(), minlength=overlap.shape[0]).astype(float)
    t_area = np.bincount(truth.ravel(), minlength=overlap.shape[1] + 1)[1:].astype(float)
    iou = overlap / np.maximum(area[:, None] + t_area[None, :] - overlap, 1)
    best = iou.max(1) if iou.shape[1] else np.zeros(len(area))
    best_g = iou.argmax(1) if iou.shape[1] else np.zeros(len(area), int)
    y = np.full(len(pairs), KEEP)
    for k, (s, l) in enumerate(pairs):
        if iou.shape[1]:
            union = (overlap[s] + overlap[l]) / np.maximum(area[s] + area[l] + t_area - overlap[s] - overlap[l], 1)
            g = int(union.argmax())
            other = lambda i: best[i] >= .5 and best_g[i] != g
            if union[g] >= .5 and union[g] > max(iou[s, g], iou[l, g]) and not other(s) and not other(l):
                y[k] = MERGE
                continue
        if best[s] < .25:
            y[k] = DROP
    return y


def apply(labels, pairs, prob, t_merge, t_drop):
    """Labels after the decisions: merges (union-find) and drops in order of confidence."""
    labels = labels.copy()
    parent = np.arange(int(labels.max()) + 1)
    dropped = np.zeros(len(parent), bool)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    order = np.argsort(-np.maximum(prob[:, MERGE], prob[:, DROP])) if len(pairs) else []
    for k in order:
        s, l = pairs[k]
        if dropped[s] or dropped[l]:
            continue
        if prob[k, MERGE] >= t_merge and prob[k, MERGE] >= prob[k, DROP]:
            parent[root(s)] = root(l)
        elif prob[k, DROP] >= t_drop:
            dropped[s] = True
    lookup = np.array([0 if (i == 0 or dropped[i] or dropped[root(i)]) else root(i) for i in range(len(parent))])
    return sequential(lookup[labels])


def _image_record(args):
    data, split, folder, level, name = args
    import tifffile
    root = Path(data) / split
    image = normalized_image(tifffile.imread(root / 'images' / name))
    labels = np.asarray(tifffile.imread(Path(folder) / level / name))
    nuclei = None
    if level == 'cytoplasm' and (Path(folder) / 'nucleus' / name).exists():
        nuclei = np.asarray(tifffile.imread(Path(folder) / 'nucleus' / name))
    pairs, x, labels = features(image, labels, nuclei)
    truth = tifffile.imread(root / f'{level}_masks' / name)
    y = targets(labels, truth, pairs) if len(pairs) else np.zeros(0, int)
    return dict(name=name, pairs=pairs, x=x, y=y, labels=labels)


def records(data, split, folder, level, workers):
    names = sorted(p.name for p in (Path(data) / split / f'{level}_masks').iterdir())
    with Pool(workers) as pool:
        return pool.map(_image_record, [(data, split, folder, level, n) for n in names])


def level_ap(data, split, level, recs, labels_of):
    """Pooled AP of `level` with the predictions labels_of(record)."""
    import tifffile
    from stardist.matching import matching
    from score_hierarchy import without_ignored
    tp, fp, fn = (np.zeros(len(TAUS)) for _ in range(3))
    root = Path(data) / split
    for r in recs:
        truth = tifffile.imread(root / f'{level}_masks' / r['name'])
        pred = labels_of(r)
        unknown = root / f'{level}_unknown' / r['name']
        if unknown.exists():
            pred = without_ignored(pred, tifffile.imread(unknown) > 0)
        for k, m in enumerate(matching(truth, pred, thresh=TAUS, criterion='iou')):
            tp[k] += m.tp; fp[k] += m.fp; fn[k] += m.fn
    acc = tp / np.maximum(tp + fp + fn, 1)
    return float(acc.mean()), int(fp[0] + tp[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, nargs='+', help='dataset dir(s), one per --train / --test in order, or one for all')
    parser.add_argument('--level', default='cytoplasm')
    parser.add_argument('--train', nargs='+', required=True, help='SPLIT=PRED_FOLDER')
    parser.add_argument('--test', nargs='+', required=True, help='SPLIT=PRED_FOLDER')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output-predictions', help='folder for the test predictions after the decisions (<level>/<name>)')
    parser.add_argument('--output')
    args = parser.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    data = args.data * (len(args.train) + len(args.test)) if len(args.data) == 1 else args.data
    train = [(d, *spec.split('=', 1)) for d, spec in zip(data, args.train)]
    test = [(d, *spec.split('=', 1)) for d, spec in zip(data[len(args.train):], args.test)]
    sets = [(d, split, folder, records(d, split, folder, args.level, args.workers)) for d, split, folder in train]
    x = np.concatenate([r['x'] for *_, recs in sets for r in recs if len(r['x'])])
    y = np.concatenate([r['y'] for *_, recs in sets for r in recs if len(r['x'])])
    groups = np.concatenate([np.full(len(r['x']), i) for i, r in enumerate([r for *_, recs in sets for r in recs]) if len(r['x'])])
    print(f'train pairs {len(y)}: keep {np.sum(y == KEEP)}, merge {np.sum(y == MERGE)}, drop {np.sum(y == DROP)}')
    make = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, class_weight='balanced', random_state=0)
    oof = np.zeros((len(y), 3))
    folds = min(5, len(np.unique(groups)))
    for fit, held in GroupKFold(folds).split(x, y, groups):
        model = make().fit(x[fit], y[fit])
        p = np.zeros((len(held), 3))
        p[:, model.classes_] = model.predict_proba(x[held])
        oof[held] = p
    # thresholds on the out-of-fold probabilities of the training split(s), pooled AP over them
    offset, prob = 0, {}
    for *_, recs in sets:
        for r in recs:
            prob[(id(recs), r['name'])] = oof[offset:offset + len(r['x'])]
            offset += len(r['x'])
    grid = [1.01, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5]

    def train_ap(tm, td):
        aps = [level_ap(d, s, args.level, recs, lambda r, recs=recs: apply(r['labels'], r['pairs'], prob[(id(recs), r['name'])], tm, td))[0]
               for d, s, _, recs in sets]
        return float(np.mean(aps))
    base = train_ap(1.01, 1.01)
    t_merge = max(grid, key=lambda t: train_ap(t, 1.01))
    t_drop = max(grid, key=lambda t: train_ap(t_merge, t))
    print(f'train (out of fold): AP {base:.3f} -> {train_ap(t_merge, t_drop):.3f} at merge {t_merge}, drop {t_drop}')
    model = make().fit(x, y)
    result = dict(level=args.level, train=args.train, test_sets=args.test, thresholds=dict(merge=t_merge, drop=t_drop),
                  train_pairs=dict(keep=int(np.sum(y == KEEP)), merge=int(np.sum(y == MERGE)), drop=int(np.sum(y == DROP))),
                  test={})
    for d, split, folder in test:
        recs = records(d, split, folder, args.level, args.workers)
        for r in recs:
            p = np.zeros((len(r['x']), 3))
            if len(r['x']):
                p[:, model.classes_] = model.predict_proba(r['x'])
            r['prob'] = p
        before = level_ap(d, split, args.level, recs, lambda r: r['labels'])
        decided = {r['name']: apply(r['labels'], r['pairs'], r['prob'], t_merge, t_drop) for r in recs}
        after = level_ap(d, split, args.level, recs, lambda r: decided[r['name']])
        merges = sum(int(np.sum((r['prob'][:, MERGE] >= t_merge) & (r['prob'][:, MERGE] >= r['prob'][:, DROP]))) for r in recs)
        drops = sum(int(np.sum((r['prob'][:, DROP] >= t_drop) & (r['prob'][:, DROP] > r['prob'][:, MERGE]))) for r in recs)
        result['test'][f'{d}:{split}'] = dict(ap_before=before[0], ap_after=after[0], objects_before=before[1],
                                               objects_after=after[1], merge_decisions=merges, drop_decisions=drops)
        print(f'{split} ({Path(d).name}): AP {before[0]:.3f} -> {after[0]:.3f}, objects {before[1]} -> {after[1]} '
              f'({merges} merges, {drops} drops decided)')
        if args.output_predictions:
            import tifffile
            out = Path(args.output_predictions) / args.level
            out.mkdir(parents=True, exist_ok=True)
            for name, labels in decided.items():
                tifffile.imwrite(out / name, labels.astype(np.uint16 if labels.max() < 65536 else np.uint32),
                                 photometric='minisblack', compression='zlib')
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1))
        print('->', args.output)


if __name__ == '__main__':
    main()
