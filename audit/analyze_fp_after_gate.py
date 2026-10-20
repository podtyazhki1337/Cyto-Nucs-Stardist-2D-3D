"""What the false positives left after the calibrated gates are, and GT-free rules against them (read-only).

Usage: python audit/analyze_fp_after_gate.py -> audit/fp_after_gate_analysis.json
Input: fixes_v3.2 history.json and epoch_NNNN_refinement.jsonl (written assembly decisions).
- Baseline applies the gates from history; AP is recomputed from the written masks and must match history.
- FP@0.5 kinds: background (best IoU < 0.1), piece of a GT matched by another kept mask, partial mask of an
  unmatched GT (its candidate before assembly already < 0.5, or >= 0.5 and cut by assembly).
- Oracles: AP without the FP@0.5 of each kind; AP if every TP had IoU 1.
- Rules re-calibrate quality/area gates without the joint-pair term (lost = 0), so only differences between
  the rule and the baseline recalibrated the same way are meaningful. Removing or merging a written mask does
  not re-run assembly: a later candidate it blocked is not re-written (conservative for removal).
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytonucs_runtime import accuracy_profile, calibrate_quality_gates

TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)
HEADS = ('nucleus', 'cytoplasm')
PACKAGE = 'fixes_v3.2'
EPOCHS = (730, 745, 750)
FRACTIONS = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def load(epoch, head):
    """Per volume: GT count and written masks with the fields the rules need."""
    volumes = []
    for line in (ROOT / PACKAGE / f'epoch_{epoch:04d}_refinement.jsonl').read_text().splitlines():
        a = json.loads(line)['stage_audit'][head]
        rank_to_obj, objs = {}, []
        for d in a['assembly_decisions']:
            if d['reason'] != 'written':
                continue
            g, v = max(d['written_gt_iou'].items(), key=lambda kv: kv[1], default=(None, 0.))
            area = d.get('connected_voxels', d['area'])
            o = dict(rank=d['rank'], quality=d['quality'], area=area, candidate_area=d['area'],
                     kept=area / max(d['area'], 1), blockers=d['blockers'], gt=g, iou=v,
                     candidate_iou=d['gt_iou'].get(g, 0.) if g is not None else 0.,
                     written_gt_iou={k: float(x) for k, x in d['written_gt_iou'].items()})
            rank_to_obj[d['rank']] = o
            objs.append(o)
        volumes.append(dict(gt=a['stages']['final']['gt_count'],
                            gt_volume={k: v['volume'] for k, v in a['gt_geometry'].items()},
                            objs=objs, by_rank=rank_to_obj))
    return volumes


def profile(ious, n_gt):
    ious = np.asarray(ious, np.float64)
    tp = np.array([(ious >= t).sum() for t in TAUS])
    return tp, len(ious) - tp, n_gt - tp


def ap(ious, n_gt):
    return float(accuracy_profile(*profile(ious, n_gt)).mean())


def kinds(volumes, keep):
    """FP@0.5 kind of every kept mask (None for TP)."""
    out = []
    for vol in volumes:
        kept = [o for o in vol['objs'] if keep(o)]
        matched = {o['gt'] for o in kept if o['iou'] >= .5}
        for o in kept:
            if o['iou'] >= .5:
                k = None
            elif o['iou'] < .1:
                k = 'background'
            elif o['gt'] in matched:
                k = 'piece_of_matched_gt'
            elif o['candidate_iou'] >= .5:
                k = 'partial_cut_by_assembly'
            else:
                k = 'partial_bad_candidate'
            out.append((o, k))
    return out


def calibrated(rows, n_gt):
    """Quality/area gates re-calibrated (lost = 0) and the resulting AP."""
    tp, fp, fn = profile([r[1] for r in rows], n_gt)
    gates, stats = calibrate_quality_gates({'h': [(q, i, 0, a) for q, i, a in rows]},
                                           {'h': dict(tp=tp, fp=fp, fn=fn)}, TAUS, 1, .7)
    return dict(ap=stats['h']['ap'], gate=gates['h'], min_area=stats['h']['min_area'],
                fp_removed=stats['h']['fp_removed'], tp_removed=stats['h']['tp_removed'])


def merged_iou(a, b, gt_volume):
    """Best GT IoU of the union of two disjoint written masks (from their per-GT IoUs)."""
    best = (None, 0.)
    for g in set(a['written_gt_iou']) | set(b['written_gt_iou']):
        if g not in gt_volume:
            continue
        v = gt_volume[g]
        inter = sum(o['written_gt_iou'].get(g, 0.) * (o['area'] + v) / (1 + o['written_gt_iou'].get(g, 0.))
                    for o in (a, b))
        iou = inter / (a['area'] + b['area'] + v - inter)
        if iou > best[1]:
            best = (g, iou)
    return best


def rule_rows(volumes, fraction, action):
    """Written masks after the rule: a mask cut by earlier masks to < fraction of its candidate is
    removed ('drop') or joined to its single blocker ('merge'; several blockers -> removed)."""
    rows = []
    for vol in volumes:
        objs = {o['rank']: dict(o) for o in vol['objs']}
        for o in sorted(vol['objs'], key=lambda o: o['rank']):
            if not o['blockers'] or o['kept'] >= fraction:
                continue
            objs.pop(o['rank'])
            if action == 'merge' and len(o['blockers']) == 1 and o['blockers'][0] in objs:
                b = objs[o['blockers'][0]]
                g, v = merged_iou(b, o, vol['gt_volume'])
                wg = dict(b['written_gt_iou'])
                for k in set(b['written_gt_iou']) | set(o['written_gt_iou']):
                    if k in vol['gt_volume']:
                        V = vol['gt_volume'][k]
                        inter = sum(x['written_gt_iou'].get(k, 0.) * (x['area'] + V)
                                    / (1 + x['written_gt_iou'].get(k, 0.)) for x in (b, o))
                        wg[k] = inter / (b['area'] + o['area'] + V - inter)
                b.update(area=b['area'] + o['area'], gt=g, iou=v, written_gt_iou=wg)
        rows += [(o['quality'], o['iou'], o['area']) for o in objs.values()]
    return rows


def intersections(o, gt_volume):
    """(GT id, |mask ∩ GT|) for every GT the written mask overlaps."""
    return [(g, v * (o['area'] + gt_volume[g]) / (1 + v)) for g, v in o['written_gt_iou'].items()
            if g in gt_volume and v > 0]


def partial_shapes(volumes, keep):
    """Kept masks with 0.1 <= IoU < 0.5: merged over 2+ GT, spilling outside its GT, or a pure part of it."""
    out = Counter()
    for vol in volumes:
        kept = [o for o in vol['objs'] if keep(o)]
        matched = {o['gt'] for o in kept if o['iou'] >= .5}
        for o in kept:
            if not .1 <= o['iou'] < .5:
                continue
            parts = sorted(((i / vol['gt_volume'][g], i / o['area']) for g, i in intersections(o, vol['gt_volume'])),
                           key=lambda p: -p[1])
            coverage, purity = parts[0]
            shape = ('merges_2+_gt' if sum(p[1] > .2 for p in parts) >= 2 else 'spills_outside_gt' if purity < .6
                     else 'pure_part_of_gt' if coverage < .5 else 'other')
            out[f"{shape}/{'gt_matched_by_other_mask' if o['gt'] in matched else 'gt_unmatched'}"] += 1
    return dict(out)


def oracle_merge_parts(volumes, keep, n_gt):
    """AP if the kept masks lying mostly (>= 50 %) inside the same GT were one mask."""
    ious, groups, new_tp = [], 0, 0
    for vol in volumes:
        by_gt = {}
        for o in (o for o in vol['objs'] if keep(o)):
            inside = [(g, i) for g, i in intersections(o, vol['gt_volume']) if i / o['area'] >= .5]
            if inside:
                by_gt.setdefault(inside[0][0], []).append((o, inside[0][1]))
            else:
                ious.append(o['iou'])
        for g, parts in by_gt.items():
            if len(parts) == 1:
                ious.append(parts[0][0]['iou'])
                continue
            groups += 1
            inter = sum(i for _, i in parts)
            iou = inter / (sum(o['area'] for o, _ in parts) + vol['gt_volume'][g] - inter)
            new_tp += iou >= .5 and not any(o['iou'] >= .5 for o, _ in parts)
            ious.append(iou)
    return dict(ap=ap(ious, n_gt), merged_groups=groups, new_tp50=int(new_tp))


def analyse(epoch, head, history, index):
    volumes = load(epoch, head)
    n_gt = sum(v['gt'] for v in volumes)
    gate = history[f'fullval_{head}_quality_gate'][index]
    min_area = history[f'fullval_{head}_area_gate'][index] or 0
    keep = lambda o: o['quality'] >= gate and o['area'] >= min_area
    labelled = kinds(volumes, keep)
    ious = [o['iou'] for o, _ in labelled]
    tp, fp, fn = profile(ious, n_gt)
    result = dict(n_gt=n_gt, gate=gate, min_area=min_area,
                  after_gate=dict(ap=ap(ious, n_gt), history_ap=history[f'fullval_ap_{head}_quality_gated'][index],
                                  tp50=int(tp[0]), fp50=int(fp[0]), fn50=int(fn[0]),
                                  fp_per_tau=[int(x) for x in fp], tp_per_tau=[int(x) for x in tp]))
    # FP at a threshold tau = true extra masks (FP@0.5) + found objects whose IoU is below tau.
    result['after_gate']['fp_share_true_extra_mean_over_tau'] = float(np.mean(fp[0] / np.maximum(fp, 1)))
    counts = Counter(k for _, k in labelled if k)
    fp_kinds = {}
    for k in sorted(counts):
        members = [o for o, kk in labelled if kk == k]
        rest = [o['iou'] for o, kk in labelled if kk != k]
        fp_kinds[k] = dict(n=counts[k], ap_without=ap(rest, n_gt),
                           quality_median=float(np.median([o['quality'] for o in members])),
                           area_median=float(np.median([o['area'] for o in members])),
                           with_blockers=int(sum(bool(o['blockers']) for o in members)),
                           kept_fraction_median=float(np.median([o['kept'] for o in members])))
    result['fp50_kinds'] = fp_kinds
    result['partial_mask_shapes'] = partial_shapes(volumes, keep)
    result['oracle'] = dict(
        no_fp50=ap([o['iou'] for o, k in labelled if k is None], n_gt),
        merge_parts_of_same_gt=oracle_merge_parts(volumes, keep, n_gt),
        tp_iou_1=float(accuracy_profile(np.full(len(TAUS), tp[0]), np.full(len(TAUS), fp[0]),
                                        np.full(len(TAUS), fn[0])).mean()))
    # Written masks cut by an earlier mask: kept fraction of TP vs FP kinds (separability of a rule).
    cut = [(o, k) for o, k in labelled if o['blockers']]
    result['cut_masks_kept_fraction'] = {
        str(k or 'tp'): dict(n=len(v), quantiles_10_25_50_75_90=[float(x) for x in np.quantile(v, [.1, .25, .5, .75, .9])])
        for k in {k for _, k in cut} for v in [[o['kept'] for o, kk in cut if kk == k]] if v}
    all_rows = [(o['quality'], o['iou'], o['area']) for v in volumes for o in v['objs']]
    base = calibrated(all_rows, n_gt)
    rules = dict(baseline_recalibrated=base)
    for action in ('drop', 'merge'):
        for f in FRACTIONS:
            r = calibrated(rule_rows(volumes, f, action), n_gt)
            rules[f'{action}_kept_below_{f}'] = dict(r, delta_ap=r['ap'] - base['ap'])
    result['rules'] = rules
    return result


def main():
    history = json.loads((ROOT / PACKAGE / 'history.json').read_text(encoding='utf-8'))
    report = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  inputs={f'{PACKAGE}/epoch_{e:04d}_refinement.jsonl':
                          hashlib.sha256((ROOT / PACKAGE / f'epoch_{e:04d}_refinement.jsonl').read_bytes()).hexdigest()
                          for e in EPOCHS}, results={})
    for epoch in EPOCHS:
        index = history['fullval_epoch'].index(epoch)
        for head in HEADS:
            r = analyse(epoch, head, history, index)
            report['results'][f'{epoch}/{head}'] = r
            best = max((k for k in r['rules'] if k != 'baseline_recalibrated'), key=lambda k: r['rules'][k]['ap'])
            print(epoch, head, 'AP', round(r['after_gate']['ap'], 4), 'hist', round(r['after_gate']['history_ap'], 4),
                  'TP/FP/FN', r['after_gate']['tp50'], r['after_gate']['fp50'], r['after_gate']['fn50'],
                  '| noFP', round(r['oracle']['no_fp50'], 4), 'merge', r['oracle']['merge_parts_of_same_gt'],
                  'IoU1', round(r['oracle']['tp_iou_1'], 4), r['partial_mask_shapes'],
                  '| kinds', {k: (v['n'], round(v['ap_without'], 4)) for k, v in r['fp50_kinds'].items()},
                  '| best rule', best, round(r['rules'][best]['delta_ap'], 4))
    (ROOT / 'audit' / 'fp_after_gate_analysis.json').write_text(json.dumps(report, indent=2, default=float),
                                                                encoding='utf-8')


if __name__ == '__main__':
    main()
