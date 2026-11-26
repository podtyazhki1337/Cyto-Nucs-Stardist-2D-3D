"""The checkpoint metric subtracts a pair twice when both levels' quality gates remove it. How much does that matter?

The metric of a full validation (cytonucs_runtime.deployed_metric) is
    w x (JTPR - (L_N + L_C) / A) + (1 - w) x mean of the gated APs,
with L_h the annotated pairs that the calibrated gate of mask head h removes (calibrate_quality_gates picks each head's
gate on its own) and A the annotated pairs. A pair lost to both gates is subtracted twice, so the metric is too low by
w x |D_N & D_C| / A <= w x min(L_N, L_C) / A.

  history  (CPU) per run and logged full validation: the stored metric, that bound, the selected epoch (best stored
           metric), and whether a bound-sized correction could change the selection -> OUT.md (+ OUT.json)
  gates    (CPU) for a final joint model, from its cached ungated val candidates (paper/operating_points.py cache-joint
           for 2D, paper/operating_points_3d.py cache for the spheroid), re-assembled as its inference does:
             U  no quality or size gate;  N  only the nucleus head's gates;  C  only the cell head's gates;  B  both;
             D_h = the annotated pairs that are joint TP in U but not in h;
             approximate score (the metric's formula): w (JTPR_U - (|D_N| + |D_C|) / A) + (1 - w) (AP_N[N] + AP_C[C]) / 2;
             exact score: w JTPR_B + (1 - w) (AP_N[B] + AP_C[B]) / 2, both gates applied, re-assembled and matched;
             double-counted pairs |D_N & D_C| / A.
           Then an exact search of the four gate values (quality gate and minimum size per head; the coordinate search of
           operating_points.py with the exact objective) from the calibrated ones: does it pick other gates, and by how
           much does the objective change? -> OUT.json
w = the run's checkpoint JTPR weight (loss_params.checkpoint_w_jtpr).

Usage (from the repository root):
  python paper/double_subtraction.py history --configs configs/A.json configs/B.json ... --output OUT.md
  python paper/double_subtraction.py gates --dim 2d --cache CACHE_DIR --data DATA_DIR --output OUT.json [--workers 4]
  python paper/double_subtraction.py gates --dim 3d --cache CACHE_DIR --output OUT.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))         # paper/
import operating_points as op  # noqa: E402
import recal_single_level_pairs as rp  # noqa: E402

ROLES = ('nucleus', 'cytoplasm')
MODE = {}


def joint_tp_set(nuclei, cells, links, gt, truths, tag):
    from cytonucs_inference import match_instances
    mn, mc = match_instances(nuclei, gt['nucleus']), match_instances(cells, gt['cytoplasm'])    # gt id -> pred id
    present_n, present_c = set(np.unique(gt['nucleus']).tolist()), set(np.unique(gt['cytoplasm']).tolist())
    out, pairs = set(), 0
    for n, c in truths['nucleus_cytoplasm'].items():
        if n not in present_n or c not in present_c:          # a pair of the volume outside this block (3D)
            continue
        pairs += 1
        pn, pc = mn.get(n), mc.get(c)
        if pn is not None and pc is not None and links.get(pn) == pc:
            out.add((tag, int(n)))
    return out, pairs


def image_task(task):
    """Counts of both levels and the joint TP set of one image at one setting."""
    from cytonucs_calibration import match_counts
    index, s = task
    gt, truths, _ = op.STATE['gt'][index]
    nuclei, cells, links = op.labels_joint(index, s)
    links = {int(k): int(v) for k, v in (links or {}).items()}
    if MODE['dim'] == '3d':
        import operating_points_3d as op3d
        item = op.STATE['items'][index]
        shape = gt['nucleus'].shape
        nuclei, cells = (op3d.to_block(x, op.STATE['factor'], shape, item['block'], item['window']) for x in (nuclei, cells))
        cn, cc = op3d.counts(gt['nucleus'], nuclei), op3d.counts(gt['cytoplasm'], cells)
    else:
        cn, cc = match_counts(gt['nucleus'], nuclei), match_counts(gt['cytoplasm'], cells)
    tp, pairs = joint_tp_set(nuclei, cells, links, gt, truths, index)
    return cn, cc, tp, pairs


def score(pool, s):
    if MODE['dim'] == '3d':
        from cytonucs_runtime import accuracy_profile
        ap_of = lambda c: float(accuracy_profile(*c).mean())                                     # noqa: E731
    else:
        from cytonucs_calibration import ap_of
    res = pool.map(image_task, [(i, s) for i in range(len(op.STATE['gt']))], chunksize=1)
    tp = set().union(*[r[2] for r in res])
    total = sum(r[3] for r in res)
    return dict(ap_nucleus=ap_of(sum(r[0] for r in res)), ap_cytoplasm=ap_of(sum(r[1] for r in res)), tp=tp, pairs=total,
                jtpr=len(tp) / max(total, 1))


def setup(dim, cache, data):
    import contextlib
    import io
    import pickle
    import tempfile
    from cytonucs_config import CytoNucsConfig, read_config
    MODE['dim'] = dim
    info = json.loads((Path(cache) / 'network.json').read_text())
    raw = read_config(info['config'])
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        p = Path(tmp) / 'c.json'
        p.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(p))
    names = info['images']
    if dim == '3d':
        import operating_points_3d as op3d
        items = [pickle.load(open(Path(cache) / 'val' / f'{n}.pkl', 'rb')) for n in names]
        op.STATE.update(items=items, factor=int(info['factor']), gt=op3d.block_truth(info['native'], items))
    else:
        from score_hierarchy import ground_truth
        items = [pickle.load(open(Path(cache) / 'val' / (Path(n).stem + '.pkl'), 'rb')) for n in names]
        inner, outer = list(raw.get('model', {}).get('levels', ROLES))      # KiTS23: mass, kidney (roles nucleus, cytoplasm)
        def by_role(g):
            gt, truths, ignore = g
            return ({'nucleus': gt[inner], 'cytoplasm': gt[outer]}, {'nucleus_cytoplasm': truths[f'{inner}_{outer}']},
                    {'nucleus': (ignore or {}).get(inner), 'cytoplasm': (ignore or {}).get(outer)})
        op.STATE['gt'] = [by_role(ground_truth(Path(data) / 'val', n, [inner, outer], [(inner, outer)])) for n in names]
    op.STATE.update(arm='joint', joint=dict(cfg=cfg, images=items, mask_default=info['mask_default'],
                                            pair_min_overlap=info['pair_min_overlap'], pair_min_mask_iou=info['pair_min_mask_iou']))
    own = {r: dict(info['own'][r]) for r in ROLES}
    own['filters'] = dict(info['own']['filters'])
    levels = raw.get('model', {}).get('levels', ['nucleus', 'cytoplasm'])
    w = float(raw.get('loss_params', {}).get('checkpoint_w_jtpr', 0.7)) if len(levels) == 2 else 0.0
    return info, own, w


def gates(args):
    import multiprocessing
    dim, out, workers = args.dim, Path(args.output), args.workers
    info, own, w = setup(dim, args.cache, args.data)
    def variant(keep):
        s = json.loads(json.dumps(own))
        for r in ROLES:
            if r not in keep:
                s[r]['gate'], s[r]['area'] = 0.0, 0
        return s
    started = time.time()
    with multiprocessing.get_context('fork').Pool(workers) as pool:
        res = {k: score(pool, variant(keep)) for k, keep in (('U', ()), ('N', ('nucleus',)), ('C', ('cytoplasm',)), ('B', ROLES))}
        A = res['U']['pairs']
        d_n, d_c = res['U']['tp'] - res['N']['tp'], res['U']['tp'] - res['C']['tp']
        approx = w * (res['U']['jtpr'] - (len(d_n) + len(d_c)) / A) + (1 - w) * (res['N']['ap_nucleus'] + res['C']['ap_cytoplasm']) / 2
        exact = w * res['B']['jtpr'] + (1 - w) * (res['B']['ap_nucleus'] + res['B']['ap_cytoplasm']) / 2
        summary = dict(model=info['config'], weights=info['weights'], w=w, pairs=A, own=own,
                       jtpr={k: v['jtpr'] for k, v in res.items()},
                       ap_nucleus={k: v['ap_nucleus'] for k, v in res.items()}, ap_cytoplasm={k: v['ap_cytoplasm'] for k, v in res.items()},
                       lost_nucleus=len(d_n), lost_cell=len(d_c), lost_both_gates=len(res['U']['tp'] - res['B']['tp']),
                       double_counted=len(d_n & d_c), double_counted_share=len(d_n & d_c) / A,
                       approximate=approx, exact=exact, exact_minus_approximate=exact - approx)
        print(json.dumps({k: v for k, v in summary.items() if k not in ('own',)}, default=float), flush=True)
        # exact gate search (quality gate and size per head), the coordinate search of operating_points.py with the exact objective
        grids = {r: dict(gate=sorted(set(rp.GATE_GRID) | {own[r]['gate']}),
                         area=sorted({0, own[r]['area']} | ({int(round(own[r]['area'] * k)) for k in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0)}
                                                            if dim == '3d' else set(rp.AREA_GRID)))) for r in ROLES}
        objective = lambda r: w * r['jtpr'] + (1 - w) * (r['ap_nucleus'] + r['ap_cytoplasm']) / 2      # noqa: E731
        current, cur_obj = json.loads(json.dumps(own)), exact
        for round_ in range(rp.MAX_ROUNDS):
            changed = False
            for r in ROLES:
                for key, values in grids[r].items():
                    rows = []
                    for v in values:
                        s = json.loads(json.dumps(current))
                        s[r][key] = v
                        rows.append(objective(score(pool, s)))
                    k = int(np.argmax(rows))
                    if rows[k] > cur_obj + rp.MIN_GAIN:
                        print(f'round {round_} {r}.{key}: {current[r][key]} -> {values[k]}, {cur_obj:.4f} -> {rows[k]:.4f} '
                              f'({time.time() - started:.0f} s)', flush=True)
                        current[r][key], cur_obj, changed = values[k], rows[k], True
            if not changed:
                break
        summary['exact_gate_search'] = dict(chosen={r: {k: current[r][k] for k in ('gate', 'area')} for r in ROLES},
                                            calibrated={r: {k: own[r][k] for k in ('gate', 'area')} for r in ROLES},
                                            objective_calibrated=exact, objective_chosen=cur_obj, gain=cur_obj - exact)
    summary['seconds'] = round(time.time() - started)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=float))
    print('->', out, json.dumps(summary['exact_gate_search'], default=float), flush=True)


def checkpoint_weight(raw):
    levels = raw.get('model', {}).get('levels', ['nucleus', 'cytoplasm'])
    return float(raw.get('loss_params', {}).get('checkpoint_w_jtpr', 0.7)) if len(levels) == 2 else 0.0


def history(args):
    lines = ['| run | w (JTPR) | full validations | selected epoch | stored metric | max over epochs of the bound w x min(L_N, L_C) / A | '
             'bound at the selected epoch | could the selection change? | margin to the runner-up |',
             '|---|---|---|---|---|---|---|---|---|']
    out = {}
    for config in args.configs:
        raw = json.loads(Path(config).read_text())
        run = raw['experiment']['name']
        f = Path(raw['output']['basedir']) / run / 'history.json'
        if not f.is_file():
            print('no history:', f)
            continue
        h = json.loads(f.read_text())
        metric = h.get('fullval_metric_deployed') or []
        epochs = h.get('fullval_epoch') or list(range(len(metric)))
        if not metric:
            continue
        w = checkpoint_weight(raw)
        rows = []
        for i, m in enumerate(metric):
            get = lambda k: (h.get('fullval_' + k) or [None] * len(metric))[i] if i < len(h.get('fullval_' + k) or []) else None   # noqa: E731
            ln = get('nucleus_quality_gate_pairs_lost') or 0
            lc = get('cytoplasm_quality_gate_pairs_lost') or 0
            a = get('total_pairs') or 0
            bound = w * min(ln, lc) / a if a else 0.0
            rows.append(dict(epoch=epochs[i] if i < len(epochs) else i, metric=m, lost_nucleus=ln, lost_cell=lc, pairs=a, bound=bound))
        valid = [r for r in rows if r['metric'] is not None]
        best = max(valid, key=lambda r: r['metric'])
        others = [r for r in valid if r is not best]
        flip = any(best['metric'] < r['metric'] + r['bound'] for r in others)
        margin = best['metric'] - max((r['metric'] for r in others), default=float('nan'))
        out[run] = dict(w=w, rows=rows, selected=best['epoch'], could_change=flip, margin=margin)
        lines.append(f"| {run} | {w:.2f} | {len(valid)} | {best['epoch']} | {best['metric']:.4f} | {max(r['bound'] for r in valid):.5f} | "
                     f"{best['bound']:.5f} | {'yes' if flip else 'no'} | {margin:.4f} |")
    Path(args.output).write_text('\n'.join(lines) + '\n')
    Path(args.output).with_suffix('.json').write_text(json.dumps(out, indent=1))
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('history', 'gates'))
    parser.add_argument('--configs', nargs='+', help='history: the training configurations of the runs')
    parser.add_argument('--dim', choices=('2d', '3d'), default='2d')
    parser.add_argument('--cache', help='gates: the cached ungated val candidates of the joint model')
    parser.add_argument('--data', help='gates, 2d: the dataset folder (its val split)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root: the CytoNucs modules
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    (history if args.step == 'history' else gates)(args)


if __name__ == '__main__':
    main()
