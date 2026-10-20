"""Read-only decomposition of final FP and FN at full validation 455 (fixes_v2.13).

Usage: python audit/analyze_fp_fn_455.py [jsonl] -> audit/fp_fn_455.json

FP: written masks whose best IoU with any GT is < 0.5 (one-to-one at IoU>0.5 is unique).
    background  - best IoU < 0.1 with every GT (spurious or unannotated object);
    fragment    - best GT is matched by another written mask (duplicate / split piece);
    partial     - best GT is unmatched (the same error also counts as that GT's FN);
    merge flag  - >=2 GT with IoU >= 0.2 (one mask spans several objects).
FN: GT whose best final IoU < 0.5, split by the first stage where a >=0.5 candidate is missing.
No weights/TIFF: GT sizes and positions are not in the JSONL, only candidate areas/origins.
"""
import json
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADS = ('nucleus', 'cytoplasm')


def q(values, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    return np.quantile(values, qs).round(4).tolist() if len(values) else None


def analyze(path):
    out = {h: defaultdict(Counter) for h in HEADS}
    sizes = {h: defaultdict(list) for h in HEADS}
    quality = {h: defaultdict(list) for h in HEADS}
    fn_detail = {h: Counter() for h in HEADS}
    fp_rank = {h: [] for h in HEADS}
    per_volume = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        vol = dict(filename=r['filename'])
        for head in HEADS:
            a = r['stage_audit'][head]
            written = [d for d in a['assembly_decisions'] if d['reason'] == 'written']
            matched = {}
            for d in written:
                if d['written_gt_iou'] and max(d['written_gt_iou'].values()) >= 0.5:
                    g = max(d['written_gt_iou'], key=d['written_gt_iou'].get)
                    matched[g] = d
            tp = len(matched); fp = 0
            area_all = [d.get('connected_voxels', d['area']) for d in written]
            median_area = float(np.median(area_all)) if area_all else 0.
            for d in written:
                ious = d['written_gt_iou']
                best = max(ious.values(), default=0.)
                area = d.get('connected_voxels', d['area'])
                if best >= 0.5:
                    sizes[head]['tp'].append(area); quality[head]['tp'].append(d['quality'])
                    continue
                fp += 1
                g = max(ious, key=ious.get) if ious else None
                if best < 0.1:
                    kind = 'background'
                elif g in matched:
                    kind = 'fragment'
                else:
                    kind = 'partial'
                out[head]['fp'][kind] += 1
                if sum(v >= 0.2 for v in ious.values()) >= 2:
                    out[head]['fp']['merge_flag'] += 1
                sizes[head]['fp_' + kind].append(area); quality[head]['fp_' + kind].append(d['quality'])
                fp_rank[head].append(area / max(median_area, 1.))
                if d['origin'][0] == 0:
                    out[head]['fp_at_z0_origin'][kind] += 1
            # FN decomposition
            stages = a['stages']
            gts = list(stages['final']['best_iou_per_gt'])
            lost = {x['gt_id'] for x in a.get('assembly_lost_gt', [])}
            per_gt = a.get('proposals', {}).get('per_gt', {})
            centers = a.get('prompt_centers_per_gt', {})
            parts = a['fragmentation']['final']['parts_per_gt']
            fn = 0
            for g in gts:
                if stages['final']['best_iou_per_gt'][g] >= 0.5:
                    continue
                fn += 1
                before = stages['threshold']['best_iou_per_gt'].get(g, 0.)
                if before >= 0.5:
                    kind = 'lost_in_assembly' if g in lost or stages['after_nms']['best_iou_per_gt'].get(g, 0.) >= .5 \
                        else 'lost_before_assembly'
                elif not centers.get(g):
                    kind = 'no_prompt_center'
                else:
                    kind = 'candidates_below_0.5'
                    b = before
                    fn_detail[head]['best_candidate_iou_0.0-0.1' if b < .1 else
                                    'best_candidate_iou_0.1-0.3' if b < .3 else 'best_candidate_iou_0.3-0.5'] += 1
                    p = parts.get(g, [])
                    if p:
                        cov = max(x['gt_coverage'] for x in p); pur = max(x['prediction_purity'] for x in p)
                        fn_detail[head]['with_dominant_part'] += 1
                        fn_detail[head]['dominant_part_undersized(cov<0.5)'] += cov < .5
                        fn_detail[head]['dominant_part_oversized(pur<0.6,cov>=0.5)'] += (pur < .6) & (cov >= .5)
                        zb = [x['intersection_bounds'][0] for x in p]
                        fn_detail[head]['part_intersection_single_z'] += any(z[1] - z[0] <= 1 for z in zb)
                    else:
                        fn_detail[head]['no_dominant_final_part'] += 1
                if per_gt.get(g):
                    out[head]['fn_proposal_reason'][per_gt[g]['reason']] += 1
                out[head]['fn'][kind] += 1
            vol[head] = dict(tp=tp, fp=fp, fn=fn, gt=len(gts))
        vol['pair_failures'] = r.get('pair_failures')
        per_volume.append(vol)
    result = {}
    for head in HEADS:
        result[head] = dict(
            totals=dict(tp=sum(v[head]['tp'] for v in per_volume), fp=sum(v[head]['fp'] for v in per_volume),
                        fn=sum(v[head]['fn'] for v in per_volume), gt=sum(v[head]['gt'] for v in per_volume)),
            fp=dict(out[head]['fp']), fp_with_z0_origin=dict(out[head]['fp_at_z0_origin']),
            fn=dict(out[head]['fn']), fn_candidates_below_detail=dict(fn_detail[head]),
            fn_proposal_reason=dict(out[head]['fn_proposal_reason']),
            area_quantiles={k: q(v) for k, v in sizes[head].items()},
            quality_quantiles={k: q(v) for k, v in quality[head].items()},
            fp_area_relative_to_volume_median=q(fp_rank[head]))
    return result, per_volume


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'fixes_v2.13' / 'epoch_0455_refinement.jsonl'
    result, per_volume = analyze(path)
    worst = {h: sorted(per_volume, key=lambda v: -(v[h]['fp'] + v[h]['fn']))[:8] for h in HEADS}
    report = dict(scope='Read-only decomposition from the stage audit JSONL; no weights/TIFF.',
                  input=str(path.relative_to(ROOT)).replace('\\', '/'),
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  heads=result,
                  worst_volumes={h: [dict(filename=v['filename'], **v[h]) for v in worst[h]] for h in HEADS})
    target = ROOT / 'audit' / 'fp_fn_455.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=1))
    print(json.dumps(report['worst_volumes'], indent=0)[:3000])


def gate_estimate(path):
    """Joint quality/size gate on the 455 written masks via the production calibration.

    Pair loss per removed TP is an upper bound (1): per-volume joint pairs are not in the JSONL.
    """
    sys.path.insert(0, str(ROOT))
    from cytonucs_runtime import calibrate_quality_gates
    taus = np.arange(0.5, 0.951, 0.05)
    gt = {'nucleus': 1439, 'cytoplasm': 1722}
    rows = {h: [] for h in HEADS}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        for head in HEADS:
            for d in r['stage_audit'][head]['assembly_decisions']:
                if d['reason'] == 'written':
                    iou = max(d['written_gt_iou'].values(), default=0.)
                    rows[head].append((d['quality'], iou, int(iou >= .5), d.get('connected_voxels', d['area'])))
    out = {}
    for head in HEADS:
        iou = np.array([x[1] for x in rows[head]])
        tp = np.array([(iou >= t).sum() for t in taus]); fp = len(iou) - tp
        counts = {head: dict(tp=tp, fp=fp, fn=gt[head] - tp)}
        both = calibrate_quality_gates({head: rows[head]}, counts, taus, 1439, .7)
        quality_only = calibrate_quality_gates({head: [x[:3] for x in rows[head]]}, counts, taus, 1439, .7)
        out[head] = dict(quality_only=quality_only[1][head],
                         quality_and_size=dict(gates=both[0], **both[1][head]))
    return out


def main_with_gates():
    main()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'fixes_v2.13' / 'epoch_0455_refinement.jsonl'
    estimate = gate_estimate(path)
    target = ROOT / 'audit' / 'fp_fn_455.json'
    report = json.loads(target.read_text(encoding='utf-8'))
    report['gate_estimate'] = dict(scope='In-sample on the same validation masks; pair loss upper bound 1 per '
                                         'removed TP.', **estimate)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(estimate, indent=1))


if __name__ == '__main__':
    main_with_gates()
