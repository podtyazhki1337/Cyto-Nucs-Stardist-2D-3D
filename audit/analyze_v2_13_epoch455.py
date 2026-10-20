"""Read-only review of the supplied v2.13 package (full validation 455) against v2.11/445.

Usage: python audit/analyze_v2_13_epoch455.py -> audit/v2_13_epoch455_review.json
Reuses the v2.11 detection review aggregation; adds AP-by-IoU, mask coverage/purity,
objective-optimal quality gates and the monitoring check for object metrics.
"""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'audit'))
from analyze_v2_11_detection import summarize, proposal_counts, fullval_table, digest

V213, V211 = ROOT / 'fixes_v2.13', ROOT / 'fixes_v2.11'
TAUS = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
GT = {'nucleus': 1439, 'cytoplasm': 1722}


def written(path):
    rows = {h: [] for h in GT}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        for head in rows:
            rows[head] += [(d['quality'], max(d['written_gt_iou'].values(), default=0.))
                           for d in r['stage_audit'][head]['assembly_decisions'] if d['reason'] == 'written']
    return rows


def ap_profile(rows, head, keep=None):
    q = np.array([x[0] for x in rows]); iou = np.array([x[1] for x in rows])
    keep = np.ones(len(q), bool) if keep is None else keep
    tp = np.array([(iou[keep] >= t).sum() for t in TAUS]); fp = keep.sum() - tp; fn = GT[head] - tp
    return tp / (tp + fp + fn)


def mask_balance(path):
    out = {}
    for head in GT:
        coverage, purity = [], []
        for line in path.read_text().splitlines():
            a = json.loads(line)['stage_audit'][head]
            final = a['stages']['final']['best_iou_per_gt']
            for g, parts in a['fragmentation']['final']['parts_per_gt'].items():
                if final.get(g, 0) >= .5 and len(parts) == 1:
                    coverage.append(parts[0]['gt_coverage']); purity.append(parts[0]['prediction_purity'])
        coverage, purity = np.array(coverage), np.array(purity)
        out[head] = dict(n=len(coverage), coverage_q10_50_90=np.quantile(coverage, [.1, .5, .9]).round(4).tolist(),
                         purity_q10_50_90=np.quantile(purity, [.1, .5, .9]).round(4).tolist(),
                         oversized_fraction=round(float((coverage > purity).mean()), 4))
    return out


def objective_gates(rows_all, w=0.7):
    """Upper-bound pair loss: every removed matched mask counts as one lost pair."""
    result = {}
    for head, rows in rows_all.items():
        q = np.array([x[0] for x in rows]); iou = np.array([x[1] for x in rows])
        base = ap_profile(rows, head).mean(); best = dict(gate=0., gain=0., ap=float(base), removed_tp=0, removed_fp=0)
        for t in np.unique(np.r_[0., q]):
            keep = q >= t; ap = ap_profile(rows, head, keep).mean(); lost = int(((~keep) & (iou >= .5)).sum())
            gain = w * (-lost / 1439) + (1 - w) * .5 * (ap - base)
            if gain > best['gain'] + 1e-12:
                best = dict(gate=float(t), gain=float(gain), ap=float(ap), removed_tp=lost, removed_fp=int(((~keep) & (iou < .5)).sum()))
        tp_q = q[iou >= .5]
        result[head] = dict(ap_baseline=float(base), objective_optimal=best,
                            matched_quality_q0_1_5=np.quantile(tp_q, [0, .01, .05]).round(4).tolist())
    return result


def main():
    history = json.loads((V213 / 'history.json').read_text())
    epochs = history['fullval_epoch']
    table = [dict(r, **{k: history['fullval_' + k][epochs.index(r['epoch'])] for k in
                        ('ap_nucleus', 'ap50_nucleus', 'ap_cytoplasm', 'ap50_cytoplasm', 'metric')})
             for r in fullval_table(V213 / 'history.json') if r['epoch'] >= 415]
    now, before = written(V213 / 'epoch_0455_refinement.jsonl'), written(V211 / 'epoch_0445_refinement.jsonl')
    object_keys = ('val_nucleus_prob_gt_max_above_020', 'val_nucleus_prob_gt_max_mean', 'val_nucleus_bce_object',
                   'val_nucleus_bce_voxel', 'val_nucleus_bce_foreground', 'val_nucleus_prob_sampled_object_count')
    report = dict(
        scope='Read-only; user package, no weights/TIFF. Upper-bound pair loss for gates; not a rerun.',
        script_sha256=digest(Path(__file__)),
        inputs={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in sorted(V213.iterdir()) if p.is_file()},
        run=dict((k, json.loads((V213 / 'run.json').read_text())['requested_config']['training'][k])
                 for k in ('resume_from_checkpoint', 'reset_best_on_resume')),
        source_matches_local={k: v == digest(ROOT / k) for k, v in
                              json.loads((V213 / 'run.json').read_text())['source_sha256'].items() if (ROOT / k).exists()},
        fullval=table, best=json.loads((V213 / 'best_fullval.json').read_text()),
        proposals=proposal_counts(V213 / 'console.log'),
        epoch455=summarize(V213 / 'epoch_0455_refinement.jsonl'),
        accuracy_by_iou={h: dict(zip(TAUS.round(2).tolist(), ap_profile(now[h], h).round(4).tolist())) for h in GT},
        accuracy_by_iou_v2_11_445={h: dict(zip(TAUS.round(2).tolist(), ap_profile(before[h], h).round(4).tolist())) for h in GT},
        mask_balance_455=mask_balance(V213 / 'epoch_0455_refinement.jsonl'),
        mask_balance_445=mask_balance(V211 / 'epoch_0445_refinement.jsonl'),
        quality_gates=objective_gates(now),
        object_monitoring={k: dict(epochs_445_450=[round(v, 4) for v in history[k][444:450]],
                                   epochs_451_455=[round(v, 4) for v in history[k][450:455]]) for k in object_keys},
    )
    target = ROOT / 'audit' / 'v2_13_epoch455_review.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
