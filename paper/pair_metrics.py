"""Pair-level metrics and annotated cells grouped by their annotated nuclei, from saved 2D predictions of any method.

Each method is a folder of predicted labels, one subfolder per level (nucleus/, cytoplasm/, file names as the images
of the split; or the same under <folder>/<split>/), with the predicted links in parents/<image>.json when the method
writes them (CytoNucs with --predictions, InstanSeg); otherwise the children are linked by overlap with the
nearest-centroid fallback of compute_nucleus_cell_assignments, as the scorers do. Objects are matched one-to-one at
IoU 0.5 (stardist matching), as by the scorers. Predicted objects lying mostly in a region of unknown annotation of
their level (<split>/<level>_unknown, BBBC020) are left out first, as by the scorers.

  pairs  per annotated child -> parent pair (child = nucleus, parent = cell):
           states: both objects matched and linked (joint TP), both matched with a wrong link, only the child matched,
           only the parent matched, neither;
           pair precision = correct predicted links / all predicted links (a link is correct when its child and parent
           are matched to an annotated pair), pair recall = linked JTPR, pair F1;
           for methods linked by overlap: the share of their links made by the fallback (a child with no overlapping
           parent linked to the nearest parent centroid within 100 px) and the JTPR / precision without it.
  cells  annotated cells grouped by their annotated nuclei: one nucleus, several nuclei, none; with an unknown-nucleus
         region also unknown (the cell lies mostly in it). Per group: cell recall (IoU 0.5); duplicates (unmatched
         predicted cells lying mostly in the annotated cell) per cell; for matched cells, the number of predicted nuclei
         linked to the predicted cell against the annotated number. Readouts per annotated cell with at least one
         nucleus: the share where a readout exists (the cell is matched and has a linked predicted nucleus), the error
         of nuclei per cell and of the nucleus / cell area ratio (mean absolute, median relative).
  common the readouts of `cells` on the annotated cells that the first method and each other one both measure, per
         seed and mean ± sd, with the difference of the mean errors and its 95 % image-bootstrap interval.
Output: OUT/<step>.json and OUT/<step>.md.

Usage (from the repository root):
  python paper/pair_metrics.py pairs --data data/HeLaCytoNuc/cytonucs --split test --output OUT \\
      --method "CytoNucs=PRED_DIR" --method "StarDist=PRED_DIR" ...
  python paper/pair_metrics.py cells --data DATA --split test --output OUT --method ...
  python paper/pair_metrics.py common --data DATA --split test --output OUT \\
      --method "CytoNucs=S0_DIR,S1_DIR,S2_DIR" --method "StarDist=S0_DIR,S1_DIR,S2_DIR" [--b 10000]
"""
import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repository root: the CytoNucs modules

STATES = ('linked', 'wrong_link', 'only_child', 'only_parent', 'neither')
GROUPS = ('one', 'several', 'none', 'unknown')
LEVELS = ('nucleus', 'cytoplasm')


def assign(child, parent, max_distance):
    """Links by overlap as the scorers make them (max_distance 0: no nearest-centroid fallback)."""
    from cytonucs_losses import compute_nucleus_cell_assignments
    with contextlib.redirect_stdout(io.StringIO()):
        links, _, _ = compute_nucleus_cell_assignments(child, parent, max_distance=max_distance)
    return {int(k): int(v) for k, v in links.items()}


def images(data, split, pred):
    """(name, gt, truths, ignore, predicted labels, saved links or None) of every image of the split."""
    import tifffile
    from score_hierarchy import ground_truth, without_ignored
    root = Path(pred) / split if (Path(pred) / split).is_dir() else Path(pred)
    if not (root / 'nucleus').is_dir():
        raise FileNotFoundError(f'{root}: no nucleus/ folder of predicted labels')
    folder = Path(data) / split
    for name in sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff')):
        gt, truths, ignore = ground_truth(folder, name, list(LEVELS), [LEVELS])
        pr = {}
        for level in LEVELS:
            labels = tifffile.imread(root / level / name)
            if ignore and ignore.get(level) is not None:
                labels = without_ignored(labels, ignore[level])
            pr[level] = np.asarray(labels)
        pf = root / 'parents' / (Path(name).stem + '.json')
        saved = {int(k): int(v) for k, v in json.loads(pf.read_text()).items()} if pf.is_file() else None
        yield name, gt, truths, ignore, pr, saved


def pair_counts(gt_c, gt_p, pred_c, pred_p, truths, saved_links):
    from cytonucs_inference import match_instances
    cm, pm = match_instances(pred_c, gt_c), match_instances(pred_p, gt_p)             # gt id -> pred id
    present_c = set(int(v) for v in np.unique(gt_c) if v)
    present_p = set(int(v) for v in np.unique(gt_p) if v)
    truths = {c: p for c, p in truths.items() if c in present_c and p in present_p}
    pred_ids_c = set(int(v) for v in np.unique(pred_c) if v)
    pred_ids_p = set(int(v) for v in np.unique(pred_p) if v)
    out = {}
    variants = {'scorer': saved_links if saved_links is not None else assign(pred_c, pred_p, 100.0)}
    if saved_links is None:
        variants['no_fallback'] = assign(pred_c, pred_p, 0.0)
    for name, links in variants.items():
        links = {c: p for c, p in links.items() if c in pred_ids_c and p in pred_ids_p}    # objects left after ignoring
        states = dict.fromkeys(STATES, 0)
        for c, p in truths.items():
            mc, mp = cm.get(c), pm.get(p)
            if mc is not None and mp is not None:
                states['linked' if links.get(mc) == mp else 'wrong_link'] += 1
            elif mc is not None:
                states['only_child'] += 1
            elif mp is not None:
                states['only_parent'] += 1
            else:
                states['neither'] += 1
        out[name] = dict(states, pairs=len(truths), links=len(links))
    return out


def pairs(args, methods):
    result, saved_any = {}, {}
    for method, pred in methods:
        total, n = {}, 0
        for _, gt, truths, _, pr, saved in images(args.data, args.split, pred):
            n += 1
            saved_any[method] = saved_any.get(method, False) or saved is not None
            for variant, c in pair_counts(gt['nucleus'], gt['cytoplasm'], pr['nucleus'], pr['cytoplasm'],
                                          truths['nucleus_cytoplasm'], saved).items():
                acc = total.setdefault(variant, dict.fromkeys(c, 0))
                for k, v in c.items():
                    acc[k] += v
        result[method] = dict(predictions=str(pred), images=n, saved_links=saved_any[method], counts=total)
        print('done', method, flush=True)

    def rates(c):
        recall = c['linked'] / max(c['pairs'], 1)
        precision = c['linked'] / max(c['links'], 1)
        return recall, precision, 2 * precision * recall / max(precision + recall, 1e-12)
    b1 = ['| method | links | pair recall (JTPR) | pair precision | pair F1 |', '|---|---|---|---|---|']
    b3 = ['| method | pairs | ' + ' | '.join(s.replace('_', ' ') for s in STATES) + ' |', '|---|---|' + '---|' * len(STATES)]
    b6 = ['| method | links | by fallback | JTPR | JTPR without fallback | precision without fallback |', '|---|---|---|---|---|---|']
    for method, d in result.items():
        c = d['counts']['scorer']
        r, p, f1 = rates(c)
        d['rates'] = dict(pair_recall=r, pair_precision=p, pair_f1=f1)
        b1.append(f"| {method} | {c['links']} | {r:.3f} | {p:.3f} | {f1:.3f} |")
        b3.append(f"| {method} | {c['pairs']} | " + ' | '.join(f'{c[s] / max(c["pairs"], 1):.3f}' for s in STATES) + ' |')
        if 'no_fallback' in d['counts']:
            nf = d['counts']['no_fallback']
            r0, p0, _ = rates(nf)
            b6.append(f"| {method} | {c['links']} | {(c['links'] - nf['links']) / max(c['links'], 1):.3f} | {r:.3f} | {r0:.3f} | {p0:.3f} |")
    text = (f'# {args.data} {args.split}\n\n## Pair precision and F1\n\n' + '\n'.join(b1) +
            '\n\n## States of the annotated pairs (shares)\n\n' + '\n'.join(b3) +
            '\n\n## Fallback linking (methods linked by overlap)\n\n' + '\n'.join(b6) + '\n')
    return result, text


def cells(args, methods):
    from cytonucs_inference import match_instances
    result = {}
    for method, pred in methods:
        g = {k: defaultdict(float) for k in GROUPS}
        ratio_err, ratio_rel, count_err, readable, eligible, saved_any = [], [], [], 0, 0, False
        for _, gt, truths, ignore, pr, saved in images(args.data, args.split, pred):
            saved_any = saved_any or saved is not None
            links = saved if saved is not None else assign(pr['nucleus'], pr['cytoplasm'], 100.0)
            gc, pc, gn = gt['cytoplasm'], pr['cytoplasm'], gt['nucleus']
            cell_m = match_instances(pc, gc)                                                 # gt cell -> pred cell
            nuclei_of = defaultdict(list)
            for n, c in truths['nucleus_cytoplasm'].items():
                nuclei_of[c].append(n)
            pred_nuclei_of = defaultdict(list)
            for n, c in links.items():
                pred_nuclei_of[c].append(n)
            unknown = ignore.get('nucleus') if ignore else None
            g_area, p_area = np.bincount(gc.ravel()), np.bincount(pc.ravel())
            gn_area, pn_area = np.bincount(gn.ravel()), np.bincount(pr['nucleus'].ravel())
            matched_pred = set(cell_m.values())
            dup = defaultdict(int)                       # unmatched predicted cells mostly inside one annotated cell
            both = (gc > 0) & (pc > 0)
            if both.any():
                width = int(gc.max()) + 1
                keys, inter = np.unique(pc[both].astype(np.int64) * width + gc[both], return_counts=True)
                for k, n in zip(keys, inter):
                    p, c = int(k // width), int(k % width)
                    if p not in matched_pred and n > p_area[p] / 2:
                        dup[c] += 1
            for c in (int(v) for v in np.unique(gc) if v):
                in_unknown = unknown is not None and unknown[gc == c].mean() > 0.5
                k_gt = len(nuclei_of.get(c, []))
                group = 'unknown' if in_unknown else ('none' if k_gt == 0 else ('one' if k_gt == 1 else 'several'))
                s = g[group]
                s['cells'] += 1
                s['duplicates'] += dup.get(c, 0)
                p = cell_m.get(c)
                if p is not None:
                    k_pred = len(pred_nuclei_of.get(p, []))
                    s['matched'] += 1
                    s['count_abs_err'] += abs(k_pred - k_gt)
                    s['count_exact'] += int(k_pred == k_gt)
                if k_gt == 0 or in_unknown:
                    continue
                eligible += 1
                if p is None or not pred_nuclei_of.get(p):
                    continue
                readable += 1
                count_err.append(abs(len(pred_nuclei_of[p]) - k_gt))
                r_gt = sum(gn_area[n] for n in nuclei_of[c]) / max(g_area[c], 1)
                r_pr = sum(pn_area[n] for n in pred_nuclei_of[p]) / max(p_area[p], 1)
                ratio_err.append(abs(r_pr - r_gt))
                ratio_rel.append(abs(r_pr - r_gt) / max(r_gt, 1e-9))
        groups = {k: dict(cells=int(s['cells']), recall=s['matched'] / s['cells'], duplicates_per_cell=s['duplicates'] / s['cells'],
                          count_mae=s['count_abs_err'] / max(s['matched'], 1), count_exact=s['count_exact'] / max(s['matched'], 1))
                  for k, s in g.items() if s['cells']}
        readout = dict(eligible=eligible, measured_share=readable / max(eligible, 1),
                       nuclei_per_cell_mae=float(np.mean(count_err)) if count_err else None,
                       ratio_mae=float(np.mean(ratio_err)) if ratio_err else None,
                       ratio_median_rel=float(np.median(ratio_rel)) if ratio_rel else None)
        result[method] = dict(predictions=str(pred), groups=groups, readout=readout, saved_links=saved_any)
        print(method, json.dumps(dict(groups=groups, readout=readout)), flush=True)
    f = lambda v: '—' if v is None else f'{v:.3f}'                                           # noqa: E731
    lines = [f'# {args.data} {args.split}', '', '| method | group | cells | cell recall | duplicates per cell | nuclei count MAE | count exact |',
             '|---|---|---|---|---|---|---|']
    for method, r in result.items():
        for k, s in r['groups'].items():
            lines.append(f"| {method} | {k} | {s['cells']} | {s['recall']:.3f} | {s['duplicates_per_cell']:.3f} | {s['count_mae']:.3f} | {s['count_exact']:.3f} |")
    lines += ['', '| method | cells with nuclei | readout exists | nuclei per cell MAE | nucleus/cell area ratio MAE | median relative ratio error |',
              '|---|---|---|---|---|---|']
    for method, r in result.items():
        o = r['readout']
        lines.append(f"| {method} | {o['eligible']} | {o['measured_share']:.3f} | {f(o['nuclei_per_cell_mae'])} | {f(o['ratio_mae'])} | {f(o['ratio_median_rel'])} |")
    return result, '\n'.join(lines) + '\n'


def readouts(args, pred):
    """{(image, annotated cell id): (annotated nuclei, count error, area-ratio error, relative ratio error)} of the
    annotated cells with at least one nucleus that a method measures (matched, with a linked predicted nucleus)."""
    from cytonucs_inference import match_instances
    out = {}
    for name, gt, truths, ignore, pr, saved in images(args.data, args.split, pred):
        links = saved if saved is not None else assign(pr['nucleus'], pr['cytoplasm'], 100.0)
        gc, pc, gn = gt['cytoplasm'], pr['cytoplasm'], gt['nucleus']
        cell_m = match_instances(pc, gc)
        nuclei_of, pred_nuclei_of = defaultdict(list), defaultdict(list)
        for n, c in truths['nucleus_cytoplasm'].items():
            nuclei_of[c].append(n)
        for n, c in links.items():
            pred_nuclei_of[c].append(n)
        unknown = ignore.get('nucleus') if ignore else None
        g_area, p_area = np.bincount(gc.ravel()), np.bincount(pc.ravel())
        gn_area, pn_area = np.bincount(gn.ravel()), np.bincount(pr['nucleus'].ravel())
        for c in (int(v) for v in np.unique(gc) if v):
            k_gt = len(nuclei_of.get(c, []))
            if k_gt == 0 or (unknown is not None and unknown[gc == c].mean() > 0.5):
                continue
            p = cell_m.get(c)
            if p is None or not pred_nuclei_of.get(p):
                continue
            r_gt = sum(gn_area[n] for n in nuclei_of[c]) / max(g_area[c], 1)
            r_pr = sum(pn_area[n] for n in pred_nuclei_of[p]) / max(p_area[p], 1)
            out[(name, c)] = (k_gt, abs(len(pred_nuclei_of[p]) - k_gt), abs(r_pr - r_gt), abs(r_pr - r_gt) / max(r_gt, 1e-9))
    return out


COMMON_GROUPS = (('all', lambda k: True), ('one', lambda k: k == 1), ('several', lambda k: k > 1))


def common(args, methods):
    """The readouts of `cells` on the annotated cells that the first method and each other one both measure (their
    own measured sets differ): nuclei per cell MAE, nucleus / cell area ratio MAE and median relative error, for all
    such cells and by group (one / several annotated nuclei); per seed (the k-th folder of each method) and mean ± sd.
    The difference of the mean errors (first minus other) with a 95 % image-bootstrap interval: whole images resampled
    (the same resample for all seeds), the per-image sums over the common cells added up, the difference averaged
    over the seeds."""
    folder = Path(args.data) / args.split
    names = sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    index = {n: i for i, n in enumerate(names)}
    seeds = [[d for d in pred.split(',') if d] for _, pred in methods]
    ref_name = methods[0][0]
    ref = [readouts(args, d) for d in seeds[0]]
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(names), size=(args.b, len(names)))
    w = np.stack([(idx == i).sum(1) for i in range(len(names))], 1).astype(float)          # B x images
    result, lines = {}, [f'# {args.data} {args.split}: {ref_name} vs each method on the cells both measure', '',
                         f'Mean ± sd over the seeds ({ref_name} / method); difference of the mean errors, {ref_name} minus '
                         f'method, point estimate [95 % image-bootstrap interval], B = {args.b}.', '',
                         f'| method | group | common cells | nuclei per cell MAE | area ratio MAE | median relative ratio error | '
                         f'difference: nuclei per cell MAE | difference: area ratio MAE |', '|---|---|---|---|---|---|---|---|']
    ms = lambda v: f'{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}' if len(v) > 1 else (f'{v[0]:.3f}' if v else '—')   # noqa: E731
    for (method, _), dirs in zip(methods[1:], seeds[1:]):
        other = [readouts(args, d) for d in dirs]
        result[method] = {}
        for gname, keep in COMMON_GROUPS:
            per_seed, full, boot = [], [], []
            for a, b in zip(ref, other):
                ks = sorted(c for c in set(a) & set(b) if keep(a[c][0]))
                if not ks:
                    continue
                x = np.zeros((len(names), 5))                       # per image: cells, |count err| a / b, |ratio err| a / b
                for c in ks:
                    x[index[c[0]]] += (1, a[c][1], b[c][1], a[c][2], b[c][2])
                va, vb = np.array([a[c][1:] for c in ks], float), np.array([b[c][1:] for c in ks], float)
                per_seed.append(dict(cells=len(ks), first=dict(count_mae=va[:, 0].mean(), ratio_mae=va[:, 1].mean(), ratio_median_rel=float(np.median(va[:, 2]))),
                                     other=dict(count_mae=vb[:, 0].mean(), ratio_mae=vb[:, 1].mean(), ratio_median_rel=float(np.median(vb[:, 2])))))
                tot = x.sum(0)
                full.append([(tot[1] - tot[2]) / tot[0], (tot[3] - tot[4]) / tot[0]])
                sb = w @ x
                nb = np.maximum(sb[:, 0], 1)
                boot.append(np.column_stack([(sb[:, 1] - sb[:, 2]) / nb, (sb[:, 3] - sb[:, 4]) / nb]))
            if not per_seed:
                continue
            pt = np.mean(full, 0)
            lo, hi = np.percentile(np.mean(boot, 0), [2.5, 97.5], axis=0)
            result[method][gname] = dict(per_seed=per_seed, diff=pt.tolist(), ci_low=lo.tolist(), ci_high=hi.tolist())
            col = lambda k: f"{ms([s['first'][k] for s in per_seed])} / {ms([s['other'][k] for s in per_seed])}"    # noqa: E731
            lines.append(f"| {method} | {gname} | {np.mean([s['cells'] for s in per_seed]):.0f} | {col('count_mae')} | {col('ratio_mae')} | "
                         f"{col('ratio_median_rel')} | {pt[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | {pt[1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}] |")
        print('done', method, flush=True)
    return result, '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('pairs', 'cells', 'common'))
    parser.add_argument('--data', required=True, help='the dataset folder (images and labels of the split)')
    parser.add_argument('--split', default='test')
    parser.add_argument('--method', action='append', required=True, metavar='NAME=PRED_DIR',
                        help='common: NAME=DIR[,DIR...], one folder per seed; the first method is the reference')
    parser.add_argument('--output', required=True, help='output folder')
    parser.add_argument('--b', type=int, default=10000, help='common: bootstrap resamples')
    parser.add_argument('--seed', type=int, default=20261121)
    args = parser.parse_args()
    methods = [tuple(m.split('=', 1)) for m in args.method]
    result, text = dict(pairs=pairs, cells=cells, common=common)[args.step](args, methods)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.step}.json').write_text(json.dumps(dict(data=str(args.data), split=args.split, methods=result), indent=1))
    (out / f'{args.step}.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
