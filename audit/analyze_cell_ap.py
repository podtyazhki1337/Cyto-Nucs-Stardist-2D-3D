"""Where cell AP is lost (read-only, user JSONL): IoU budget, TP coverage vs precision, Z-face contact,
written-mask kinds behind the size gate, and a kept-fraction quality variant.

Usage: python audit/analyze_cell_ap.py -> audit/cell_ap_analysis.json
- IoU budget: AP of the gated masks vs the AP the same TP/FP/FN at IoU 0.5 would give if every TP had IoU 1.
- Coverage / precision of a TP mask: |mask ∩ GT| / |GT| and / |mask| from its IoU, GT volume and mask volume.
- Gates in the variants are re-calibrated here without the joint-pair term (lost = 0), so absolute AP differs
  slightly from history; only the comparison between variants is meaningful.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytonucs_runtime import calibrate_quality_gates, accuracy_profile

TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)
HEADS = ('cytoplasm', 'nucleus')
RUNS = (('fixes_v3.1', 635), ('fixes_v3.1', 650), ('fixes_v3.2', 730), ('fixes_v3.2', 745), ('fixes_v3.2', 750))
SIZE_BINS = ((0, 10000), (10000, 30000), (30000, 60000), (60000, 10 ** 12))


def best(d):
    return max(d['written_gt_iou'].items(), key=lambda kv: kv[1], default=(None, 0.))


def load(package, epoch, head):
    volumes = []
    for line in (ROOT / package / f'epoch_{epoch:04d}_refinement.jsonl').read_text().splitlines():
        a = json.loads(line)['stage_audit'][head]
        volumes.append((a['gt_geometry'], a['stages']['final']['gt_count'],
                        [d for d in a['assembly_decisions'] if d['reason'] == 'written']))
    return volumes


def iou_budget_and_quality_variant(volumes):
    rows, n_gt = [], 0
    for _, count, written in volumes:
        n_gt += count
        for d in written:
            area = d.get('connected_voxels', d['area'])
            rows.append((d['quality'], best(d)[1], area, area / max(d['area'], 1)))
    iou = np.array([r[1] for r in rows])
    tp = np.array([(iou >= t).sum() for t in TAUS])
    counts = dict(tp=tp, fp=len(iou) - tp, fn=n_gt - tp)
    out = {}
    for name, quality in (('quality', lambda r: r[0]), ('quality_x_kept_fraction', lambda r: r[0] * r[3])):
        gates, stats = calibrate_quality_gates({'h': [(quality(r), r[1], 0, r[2]) for r in rows]}, {'h': counts},
                                               TAUS, 1, .7)
        out[name] = dict(ap=stats['h']['ap'], gate=gates['h'], min_area=stats['h']['min_area'],
                         tp_removed=stats['h']['tp_removed'], fp_removed=stats['h']['fp_removed'])
    gates, stats = calibrate_quality_gates({'h': [(r[0], r[1], 0, r[2]) for r in rows]}, {'h': counts}, TAUS, 1, .7)
    kept = np.array([r[0] >= gates['h'] and r[2] >= stats['h']['min_area'] for r in rows])
    ik = iou[kept]; tpk = np.array([(ik >= t).sum() for t in TAUS]); n = len(ik)
    ap = accuracy_profile(tpk, n - tpk, n_gt - tpk)
    perfect = accuracy_profile(np.full(len(TAUS), tpk[0]), np.full(len(TAUS), n - tpk[0]), np.full(len(TAUS), n_gt - tpk[0]))
    out['iou_budget'] = dict(ap=float(ap.mean()), ap50=float(ap[0]), ap_if_tp_iou_1=float(perfect.mean()),
                             tp_iou_median=float(np.median(ik[ik >= .5])))
    return out


def coverage_precision(volumes):
    rows = []
    for geometry, _, written in volumes:
        for d in written:
            g, v = best(d)
            if v < .5 or g not in geometry:
                continue
            faces = geometry[g]['faces'].split(',') if geometry[g]['faces'] else []
            vol = geometry[g]['volume']; area = d.get('connected_voxels', d['area'])
            inter = v * (area + vol) / (1 + v)
            rows.append((vol, v, inter / vol, inter / area, sum(f in faces for f in ('z0', 'z1')),
                         any(f[0] in 'yx' for f in faces)))
    r = np.array(rows, dtype=float)
    summary = lambda m: dict(n=int(m.sum()), iou_median=float(np.median(r[m, 1])),
                             coverage_median=float(np.median(r[m, 2])), precision_median=float(np.median(r[m, 3])))
    return dict(all=summary(np.ones(len(r), bool)),
                by_volume={f'{lo}-{hi}': summary(m) for lo, hi in SIZE_BINS
                           for m in [(r[:, 0] >= lo) & (r[:, 0] < hi)] if m.sum() >= 10},
                by_faces={f'z_faces_{z}_{"xy_edge" if xy else "xy_inner"}': summary(m) for z in (0, 1, 2) for xy in (0, 1)
                          for m in [(r[:, 4] == z) & (r[:, 5] == xy)] if m.sum() >= 10})


def written_kinds(volumes, area_gate):
    kinds = defaultdict(list)
    for _, _, written in volumes:
        matched = {best(d)[0] for d in written if best(d)[1] >= .5}
        for d in written:
            g, v = best(d)
            kind = 'tp' if v >= .5 else 'background' if v < .1 else \
                'piece_of_matched_gt' if g in matched else 'partial_of_unmatched_gt'
            kinds[kind].append((d.get('connected_voxels', d['area']), d.get('connected_voxels', d['area']) / max(d['area'], 1),
                                np.nan if d['quality'] is None else d['quality']))
    return {k: dict(n=len(v), volume_median=float(np.median([x[0] for x in v])),
                    kept_fraction_median=float(np.median([x[1] for x in v])),
                    quality_median=float(np.nanmedian([x[2] for x in v])),
                    below_area_gate=float(np.mean([x[0] < area_gate for x in v]))) for k, v in kinds.items()}


def main():
    report = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results={})
    for package, epoch in RUNS:
        h = json.loads((ROOT / package / 'history.json').read_text(encoding='utf-8'))
        i = h['fullval_epoch'].index(epoch)
        for head in HEADS:
            volumes = load(package, epoch, head)
            key = f'{package}/{epoch}/{head}'
            report['results'][key] = dict(coverage_precision=coverage_precision(volumes),
                                          written_kinds=written_kinds(volumes, h[f'fullval_{head}_area_gate'][i]),
                                          gates=dict(quality=h[f'fullval_{head}_quality_gate'][i],
                                                     min_area=h[f'fullval_{head}_area_gate'][i]))
            if package == 'fixes_v3.2':
                report['results'][key].update(iou_budget_and_quality_variant(volumes))
            cp = report['results'][key]['coverage_precision']['all']
            print(key, 'cov', round(cp['coverage_median'], 3), 'prec', round(cp['precision_median'], 3),
                  report['results'][key].get('iou_budget'))
    (ROOT / 'audit' / 'cell_ap_analysis.json').write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
