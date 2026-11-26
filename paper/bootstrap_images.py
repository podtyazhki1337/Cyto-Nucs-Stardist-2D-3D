"""Image-level uncertainty of a comparison: paired differences (first method minus each other one) with 95 % bootstrap
intervals, resampling whole test images (with --clusters, also whole groups of images, e.g. the TissueNet experiments,
the source acquisitions of its tiles).

In every iteration the per-image counts of the resampled images are summed and the pooled AP (mean over IoU
0.50..0.95 of TP / (TP + FP + FN)) and the linked JTPR (joint TP / annotated pairs) recomputed, for every seed of both
methods with the same resample; the difference of the seed means is recorded. Seeds are paired by position (the k-th
file of each method); only seeds both methods have are used. The per-image counts are the `records` of the score files
of score_hierarchy.py and score_predictions.py, so nothing is scored again. Point estimate: all images; interval: the
2.5 / 97.5 percentiles of B resamples. The complementary interval over annotated pairs is paper/bootstrap_jtpr.py.

Usage (from the repository root):
  python paper/bootstrap_images.py --split test --output OUT.json \\
      --method "CytoNucs=RUN/hierarchy_test.json,RUN_s1/hierarchy_test.json,RUN_s2/hierarchy_test.json" \\
      --method "StarDist=SD/scores.json,SD_s1/scores.json,SD_s2/scores.json" \\
      [--clusters data/TissueNet/cytonucs/tissuenet.json --cluster-key experiment] [--b 10000]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def records(path, split):
    """The per-image records of a score file (a score_hierarchy.py file, or a score_predictions.py file keyed by split)."""
    d = json.loads(Path(path).read_text())
    x = d.get(split) if isinstance(d.get(split), dict) else d
    if x.get('split', split) != split:
        raise ValueError(f'{path}: split {x.get("split")} != {split}')
    return x['records']


def arrays(recs):
    """Per image: TP, FP, FN (images x 2 levels x (tp, fp, fn) x taus) and joint TP / pairs, in name order."""
    recs = sorted(recs, key=lambda r: r['name'])
    c = np.array([[[r['levels'][lv][k] for k in ('tp', 'fp', 'fn')] for lv in ('nucleus', 'cytoplasm')] for r in recs], np.int64)
    j = np.array([[r['pairs']['nucleus_cytoplasm']['joint_tp'], r['pairs']['nucleus_cytoplasm']['pairs']] for r in recs], np.int64)
    return [r['name'] for r in recs], c, j


def metrics(c, j, weights):
    """AP nucleus, AP cell, JTPR of the images counted with `weights` (B x images)."""
    s = np.einsum('bi,ilkt->blkt', weights, c)                       # B x level x (tp, fp, fn) x tau
    ap = (s[:, :, 0] / np.maximum(s[:, :, 0] + s[:, :, 1] + s[:, :, 2], 1)).mean(-1)
    jj = weights @ j
    return np.column_stack([ap[:, 0], ap[:, 1], jj[:, 0] / np.maximum(jj[:, 1], 1)])


def compare(a, b, units, n_images, B, rng):
    """a, b: {seed: (names, counts, joint)}; units: lists of image indices resampled together."""
    common = sorted(set(a) & set(b))
    a, b = [a[s] for s in common], [b[s] for s in common]
    for x in a + b:
        if x[0] != a[0][0]:
            raise ValueError('the methods must cover the same images')
    full = np.ones((1, n_images))
    idx = rng.integers(0, len(units), size=(B, len(units)))
    w = np.zeros((B, n_images))
    for u, members in enumerate(units):
        w[:, members] += (idx == u).sum(1)[:, None]
    ma = np.mean([metrics(c, j, full) for _, c, j in a], 0)[0]
    mb = np.mean([metrics(c, j, full) for _, c, j in b], 0)[0]
    da = np.mean([metrics(c, j, w) for _, c, j in a], 0) - np.mean([metrics(c, j, w) for _, c, j in b], 0)
    lo, hi = np.percentile(da, [2.5, 97.5], axis=0)
    return dict(seeds=len(common), first=ma.tolist(), other=mb.tolist(), diff=(ma - mb).tolist(), ci_low=lo.tolist(),
                ci_high=hi.tolist(), share_above_zero=(da > 0).mean(0).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', action='append', required=True, metavar='NAME=FILE[,FILE...]',
                        help='score files of a method, one per seed; the first method is the reference')
    parser.add_argument('--split', default='test')
    parser.add_argument('--output', required=True)
    parser.add_argument('--clusters', help='JSON with {"images": {name: {KEY: group}}}: resample whole groups too')
    parser.add_argument('--cluster-key', default='experiment')
    parser.add_argument('--b', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261121)
    args = parser.parse_args()
    if len(args.method) < 2:
        parser.error('give at least two methods')
    methods = []
    for m in args.method:
        name, files = m.split('=', 1)
        methods.append((name, {k: arrays(records(f, args.split)) for k, f in enumerate(files.split(',')) if f}))
    ref_name, ref = methods[0]
    names = next(iter(ref.values()))[0]
    units = {'images': [[i] for i in range(len(names))]}
    if args.clusters:
        meta = json.loads(Path(args.clusters).read_text())['images']
        by = {}
        for i, n in enumerate(names):
            by.setdefault(meta[n][args.cluster_key], []).append(i)
        units = {args.cluster_key: list(by.values()), **units}
    rng = np.random.default_rng(args.seed)
    result = dict(split=args.split, reference=ref_name, B=args.b, comparisons={})
    fmt = lambda r, i: f"{r['diff'][i]:+.3f} [{r['ci_low'][i]:+.3f}, {r['ci_high'][i]:+.3f}]"     # noqa: E731
    lines = [f'{ref_name} minus each method, {args.split}: point estimate [95 % bootstrap interval], B = {args.b}. '
             'Columns: AP nucleus | AP cell | JTPR.', '']
    for unit, members in units.items():
        lines += [f'## resampling {unit} ({len(members)} units)', '',
                  f'| method | seeds | {ref_name} | method | AP nucleus | AP cell | JTPR |', '|---|---|---|---|---|---|---|']
        for name, other in methods[1:]:
            r = compare(ref, other, members, len(names), args.b, rng)
            result['comparisons'].setdefault(unit, {})[name] = r
            lines.append(f"| {name} | {r['seeds']} | {' / '.join(f'{v:.3f}' for v in r['first'])} | "
                         f"{' / '.join(f'{v:.3f}' for v in r['other'])} | {fmt(r, 0)} | {fmt(r, 1)} | {fmt(r, 2)} |")
        lines.append('')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1))
    Path(args.output).with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
