"""Read-only review of fixes_v3.1 (semantic foreground head; full validations 575-650, end calibration).

Usage: python audit/analyze_v3_1_epoch650.py -> audit/v3_1_epoch650_artifacts.json
Reuses the per-epoch statistics of analyze_v3_epoch565 (AP recomputed from written masks, gated by the
epoch's calibrated gates) and adds candidate size vs GT size (precision/coverage) by GT volume.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'audit'))
from analyze_v3_epoch565 import jsonl_stats, timing, gate_of, HEADS
from cytonucs_runtime import history_selection_metrics

V31 = ROOT / 'fixes_v3.1'
V3 = ROOT / 'fixes_v3'
BINS = ((0, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, 10 ** 12))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def size_bins(path):
    """Best candidate per GT (among candidates whose best GT it is): IoU, coverage of GT, precision."""
    out = {}
    for h in HEADS:
        rows = []
        for line in Path(path).read_text().splitlines():
            a = json.loads(line)['stage_audit'][h]; geo = a['gt_geometry']
            best = {}
            for d in a['assembly_decisions']:
                g = d['best_gt']
                if g in geo and (g not in best or d['best_iou'] > best[g][0]):
                    best[g] = (d['best_iou'], d['area'])
            for g, (iou, area) in best.items():
                v = geo[g]['volume']; inter = iou * (area + v) / (1 + iou)
                rows.append((v, iou, inter / v, inter / area))
        rows = np.array(rows)
        out[h] = {f'{lo}-{hi if hi < 10 ** 12 else "inf"}': dict(
            n=int(m.sum()), iou_median=float(np.median(rows[m, 1])), coverage_median=float(np.median(rows[m, 2])),
            precision_median=float(np.median(rows[m, 3])))
            for lo, hi in BINS for m in [(rows[:, 0] >= lo) & (rows[:, 0] < hi)] if m.sum() >= 5}
    return out


def fn_by_size(path):
    out = {}
    for h in HEADS:
        counts = defaultdict(lambda: [0, 0])
        for line in Path(path).read_text().splitlines():
            a = json.loads(line)['stage_audit'][h]
            for g, v in a['stages']['final']['best_iou_per_gt'].items():
                vol = a['gt_geometry'][g]['volume']
                key = next(f'{lo}-{hi if hi < 10 ** 12 else "inf"}' for lo, hi in BINS if lo <= vol < hi)
                counts[key][0] += 1; counts[key][1] += v >= .5
        out[h] = {k: dict(n=n, recall50=tp / n) for k, (n, tp) in counts.items()}
    return out


def source_known(name, sha):
    """The run's source hash equals the current file or an after-version in a local snapshot."""
    candidates = [ROOT / name] + list((ROOT / 'audit' / 'history').glob(f'*/after/{name}'))
    return any(p.exists() and digest(p) == sha for p in candidates)


def main():
    h = json.loads((V31 / 'history.json').read_text(encoding='utf-8'))
    h3 = json.loads((V3 / 'history.json').read_text(encoding='utf-8'))
    run = json.loads((V31 / 'run.json').read_text(encoding='utf-8'))
    deployed = history_selection_metrics(h, 0.7)
    keys = ('jtpr', 'interior_jtpr', 'ap_nucleus', 'ap50_nucleus', 'ap_cytoplasm', 'ap50_cytoplasm',
            'ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated', 'nucleus_tp', 'nucleus_fp', 'nucleus_fn',
            'cytoplasm_tp', 'cytoplasm_fp', 'cytoplasm_fn', 'nucleus_area_gate', 'cytoplasm_area_gate',
            'nucleus_gt_without_prompt_center', 'cytoplasm_gt_without_prompt_center')
    e = h['fullval_epoch']
    table = [dict(epoch=x, deployed_metric=deployed[i], **{k: (h.get('fullval_' + k) or [None] * len(e))[i] for k in keys})
             for i, x in enumerate(e) if x >= 555]
    train_keys = sorted(k for k in h if k.startswith('train_') and any(
        s in k for s in ('semantic', 'nucleus_mask', 'cytoplasm_mask', 'quality_pair_accuracy', 'prob_gt_max_mean')))
    epochs = len(h['train_loss'])
    trend = {k: {f'{a + 1}-{b}': (float(np.nanmean([v for v in h[k][a:b] if v is not None])) if any(
        v is not None for v in h[k][a:b]) else None) for a, b in ((570, 580), (600, 610), (640, epochs))}
        for k in train_keys}
    per_epoch = {}
    for e_ in (635, 640, 645, 650):
        gates = {hd: gate_of(h, e_, hd) for hd in HEADS}
        path = V31 / f'epoch_{e_:04d}_refinement.jsonl'
        per_epoch[e_] = dict(timing=timing(V31, e_), heads=jsonl_stats(path, gates),
                             size_bins=size_bins(path), recall_by_size=fn_by_size(path))
    ref = V3 / 'epoch_0565_refinement.jsonl'
    reference = dict(size_bins=size_bins(ref), recall_by_size=fn_by_size(ref),
                     heads=jsonl_stats(ref, {hd: gate_of(h3, 565, hd) for hd in HEADS}))
    report = dict(
        scope='Read-only; user package fixes_v3.1, no weights/TIFF. Gated = post-hoc removal by the epoch gates.',
        script_sha256=digest(__file__),
        inputs={p.name: digest(p) for p in sorted(V31.iterdir()) if p.is_file()},
        run=dict(resume=run['requested_config']['training']['resume_from_checkpoint'],
                 reset_best=run['requested_config']['training']['reset_best_on_resume'],
                 semantic_foreground_weight=run['requested_config']['loss_params']['semantic_foreground_weight']),
        source_known={k: source_known(k, v) for k, v in run['source_sha256'].items()},
        best=json.loads((V31 / 'best_fullval.json').read_text(encoding='utf-8')),
        mask_calibration=json.loads((V31 / 'mask_calibration.json').read_text(encoding='utf-8')),
        final_results={k: v for k, v in json.loads((V31 / 'final_results.json').read_text(encoding='utf-8')).items()
                       if not isinstance(v, list)},
        fullval=table, train_trend=trend, per_epoch=per_epoch, reference_v3_565=reference)
    (ROOT / 'audit' / 'v3_1_epoch650_artifacts.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(json.dumps(dict(source_known=report['source_known'], run=report['run']), indent=1))
    for r in table:
        print(r['epoch'], round(r['deployed_metric'], 4), round(r['jtpr'], 4),
              [round(r[k] or 0, 3) for k in ('ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated', 'ap_nucleus', 'ap_cytoplasm')],
              [r[k] for k in ('nucleus_tp', 'nucleus_fp', 'nucleus_fn', 'cytoplasm_tp', 'cytoplasm_fp', 'cytoplasm_fn',
                              'nucleus_gt_without_prompt_center', 'cytoplasm_gt_without_prompt_center')])


if __name__ == '__main__':
    main()
