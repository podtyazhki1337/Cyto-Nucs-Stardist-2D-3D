"""Extra instances on found objects ("pieces"): how many, what they cost, where they come from (read-only).

Usage: python audit/analyze_pieces.py [epoch ...] -> audit/pieces_epoch<last>.json
Input: fixes_v3.2 history.json and epoch_NNNN_refinement.jsonl (written assembly decisions).
- Piece: a written mask with IoU < 0.5 lying mostly (>= 50 % of its voxels) inside a GT that another
  written mask matches (IoU >= 0.5). Counted without and with the calibrated gates of that epoch.
- Oracles (AP after the history gates): pieces removed; pieces joined to the matching mask (written masks are
  disjoint, so intersections and areas add).
- Prompt centres per matched GT from the audit; share of matched GT that have a piece.
- GT-free rule: join a written remainder to its single blocker when the candidate overlapped the blocker by
  >= f of the blocker's written volume; gates re-calibrated without the joint-pair term, compared with the
  baseline re-calibrated the same way.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytonucs_runtime import accuracy_profile, calibrate_quality_gates

PACKAGE = ROOT / 'fixes_v3.2'
TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)


def ap(ious, n_gt):
    ious = np.asarray(ious, np.float64)
    tp = np.array([(ious >= t).sum() for t in TAUS])
    return float(accuracy_profile(tp, len(ious) - tp, n_gt - tp).mean())


def volumes(epoch, head):
    out = []
    for line in (PACKAGE / f'epoch_{epoch:04d}_refinement.jsonl').read_text().splitlines():
        a = json.loads(line)['stage_audit'][head]
        geo = a['gt_geometry']
        objs = []
        for d in a['assembly_decisions']:
            if d['reason'] != 'written':
                continue
            area = d.get('connected_voxels', d['area'])
            inter = {g: v * (area + geo[g]['volume']) / (1 + v) for g, v in d['written_gt_iou'].items() if g in geo and v > 0}
            gt, iou = max(d['written_gt_iou'].items(), key=lambda kv: kv[1], default=(None, 0.))
            objs.append(dict(rank=d['rank'], area=area, candidate=d['area'], free=d['free_voxels'], quality=d['quality'],
                             inter=inter, gt=gt, iou=iou, blockers=d['blockers']))
        out.append(dict(n_gt=a['stages']['final']['gt_count'], geo=geo, objs=objs, prompts=a['prompt_centers_per_gt']))
    return out


def pieces_of(objs):
    main = {o['gt']: o for o in objs if o['iou'] >= .5}
    found = []
    for o in objs:
        if o['iou'] >= .5 or not o['inter']:
            continue
        g, x = max(o['inter'].items(), key=lambda kv: kv[1])
        if x / o['area'] >= .5 and g in main:
            found.append((o, g))
    return main, found


def oracle(vols, keep):
    base, dropped, joined, sizes, overlapped = [], [], [], [], 0
    for v in vols:
        objs = [o for o in v['objs'] if keep(o)]
        main, pieces = pieces_of(objs)
        sizes += [o['area'] for o, _ in pieces]
        overlapped += sum(main[g]['rank'] in o['blockers'] for o, g in pieces)
        ids = {id(o) for o, _ in pieces}
        add = defaultdict(lambda: [0, 0.])
        for o, g in pieces:
            add[g][0] += o['area']; add[g][1] += o['inter'][g]
        for o in objs:
            base.append(o['iou'])
            if id(o) in ids:
                continue
            dropped.append(o['iou'])
            if o['iou'] >= .5 and o['gt'] in add:
                a, x = add[o['gt']]; inter = o['inter'][o['gt']] + x
                joined.append(inter / (o['area'] + a + v['geo'][o['gt']]['volume'] - inter))
            else:
                joined.append(o['iou'])
    n_gt = sum(v['n_gt'] for v in vols)
    return dict(pieces=len(sizes), median_voxels=float(np.median(sizes)) if sizes else None,
                candidate_overlapped_matching_mask=overlapped,
                ap=ap(base, n_gt), ap_pieces_removed=ap(dropped, n_gt), ap_pieces_joined=ap(joined, n_gt))


def by_prompt_count(vols, keep):
    table = defaultdict(lambda: [0, 0])
    for v in vols:
        main, pieces = pieces_of([o for o in v['objs'] if keep(o)])
        with_piece = {g for _, g in pieces}
        for g in main:
            k = str(min(v['prompts'].get(g, 0), 6))
            table[k][0] += 1; table[k][1] += g in with_piece
    return {k: dict(matched_gt=n, with_piece=p, share=p / n) for k, (n, p) in sorted(table.items())}


def overlap_rule(vols, fractions):
    n_gt = sum(v['n_gt'] for v in vols)

    def calibrated(f):
        rows = []
        for v in vols:
            objs = {o['rank']: dict(o, inter=dict(o['inter'])) for o in v['objs']}
            for r in sorted(objs):
                o = objs[r]
                if f is None or len(o['blockers']) != 1 or o['blockers'][0] not in objs:
                    continue
                m = objs[o['blockers'][0]]
                if (o['candidate'] - o['free']) / m['area'] >= f:
                    m['area'] += o['area']
                    for g, x in o['inter'].items():
                        m['inter'][g] = m['inter'].get(g, 0.) + x
                    del objs[r]
            for o in objs.values():
                rows.append((o['quality'], max((x / (o['area'] + v['geo'][g]['volume'] - x) for g, x in o['inter'].items()),
                                               default=0.), o['area']))
        ious = np.array([r[1] for r in rows]); tp = np.array([(ious >= t).sum() for t in TAUS])
        _, stats = calibrate_quality_gates({'h': [(q, i, 0, a) for q, i, a in rows]},
                                           {'h': dict(tp=tp, fp=len(ious) - tp, fn=n_gt - tp)}, TAUS, 1, .7)
        return stats['h']['ap']
    base = calibrated(None)
    return dict(baseline_recalibrated=base, delta_ap={str(f): calibrated(f) - base for f in fractions})


def main(epochs):
    history = json.loads((PACKAGE / 'history.json').read_text(encoding='utf-8'))
    report = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results={})
    for epoch in epochs:
        i = history['fullval_epoch'].index(epoch)
        for head in ('nucleus', 'cytoplasm'):
            gate, min_area = history[f'fullval_{head}_quality_gate'][i], history[f'fullval_{head}_area_gate'][i] or 0
            gated = lambda o: o['quality'] >= gate and o['area'] >= min_area
            vols = volumes(epoch, head)
            r = dict(ungated=oracle(vols, lambda o: True), gated=oracle(vols, gated),
                     share_with_piece_by_prompt_centres=by_prompt_count(vols, gated),
                     overlap_rule=overlap_rule(vols, (0.05, 0.1, 0.2, 0.3)))
            report['results'][f'{epoch}/{head}'] = r
            g = r['gated']
            print(epoch, head, 'pieces ungated/gated', r['ungated']['pieces'], g['pieces'],
                  '| AP %.4f removed %.4f joined %.4f' % (g['ap'], g['ap_pieces_removed'], g['ap_pieces_joined']),
                  '| best overlap rule %+.4f' % max(r['overlap_rule']['delta_ap'].values()))
    (ROOT / 'audit' / f'pieces_epoch{epochs[-1]}.json').write_text(json.dumps(report, indent=2, default=float),
                                                                   encoding='utf-8')


if __name__ == '__main__':
    main([int(e) for e in sys.argv[1:]] or [815, 820])
