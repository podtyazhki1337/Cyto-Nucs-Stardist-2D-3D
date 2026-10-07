"""Read-only review of fixes_v3 (full validations 555-565) against fixes_v2.17 (540-550).

Usage: python audit/analyze_v3_epoch565.py -> audit/v3_epoch565_artifacts.json
Validation time: file mtimes of epoch_NNNN_fullval.png (written after volume 1) and
epoch_NNNN_refinement.jsonl (written after volume 61) -> time for 60 volumes, scaled to 61.
Strict-IoU accuracy is recomputed from the written masks of the stage audit (as in the v2.17 AP check).
"Gated" = post-hoc removal of written masks below the gate of that epoch (quality < gate or connected
voxels < min area); it approximates the deployed gate (the gate itself runs on the full-resolution labels).
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytonucs_runtime import accuracy_profile, history_selection_metrics

V3 = ROOT / 'fixes_v3'
V217 = ROOT / 'fixes_v2.17'
HEADS = ('nucleus', 'cytoplasm')
TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_of(history, epoch, head):
    i = history['fullval_epoch'].index(epoch)
    q = (history.get(f'fullval_{head}_quality_gate') or [None] * (i + 1))[i]
    a = (history.get(f'fullval_{head}_area_gate') or [None] * (i + 1))[i]
    return q, a


def jsonl_stats(path, gates):
    written = {h: [] for h in HEADS}; gt = Counter(); counters = {h: Counter() for h in HEADS}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        for h in HEADS:
            a = r['stage_audit'][h]
            gt[h] += a['stages']['final']['gt_count']
            counters[h]['attempted'] += r['refinement'][h].get('attempted') or 0
            counters[h]['written'] += r['refinement'][h].get('written') or 0
            for d in a['assembly_decisions']:
                if d['reason'] == 'written':
                    written[h].append((max(d['written_gt_iou'].values(), default=0.), d.get('quality'),
                                       d.get('connected_voxels', d['area']), tuple(d['origin'])))
    out = {}
    for h in HEADS:
        q_gate, a_gate = gates[h]
        rows = written[h]
        variants = dict(raw=rows, gated=[w for w in rows if (q_gate is None or w[1] is None or w[1] >= q_gate)
                                         and (not a_gate or w[2] >= a_gate)])
        res = {}
        for name, rr in variants.items():
            iou = np.array([w[0] for w in rr]); tp = np.array([(iou >= t).sum() for t in TAUS])
            acc = accuracy_profile(tp, len(iou) - tp, gt[h] - tp)
            tp_iou = iou[iou >= .5]
            res[name] = dict(ap=float(acc.mean()), accuracy={f'{t:.2f}': float(v) for t, v in zip(TAUS, acc)},
                             tp=int(tp[0]), fp=int(len(iou) - tp[0]),
                             tp_iou_quantiles_10_25_50_75_90=[float(v) for v in np.quantile(tp_iou, [.1, .25, .5, .75, .9])]
                             if len(tp_iou) else None,
                             tp_share_iou_ge_080=float((tp_iou >= .8).mean()) if len(tp_iou) else None,
                             tp_share_iou_ge_090=float((tp_iou >= .9).mean()) if len(tp_iou) else None)
        bg = [w for w in rows if w[0] < .1]
        res['background_masks'] = dict(count=len(bg), median_voxels=float(np.median([w[2] for w in bg])) if bg else None,
                                       origin_on_low_xy_face=sum(w[3][1] == 0 or w[3][2] == 0 for w in bg),
                                       below_area_gate=sum(bool(a_gate) and w[2] < a_gate for w in bg))
        res['gt'] = int(gt[h]); res['gate'] = dict(quality=q_gate, min_area=a_gate)
        res['prompts_per_volume'] = counters[h]['attempted'] / 61.
        out[h] = res
    return out


def timing(package, epoch):
    png = (package / f'epoch_{epoch:04d}_fullval.png').stat().st_mtime
    jsonl = (package / f'epoch_{epoch:04d}_refinement.jsonl').stat().st_mtime
    return dict(volumes_2_to_61_min=round((jsonl - png) / 60., 1), estimate_61_volumes_min=round((jsonl - png) / 60. * 61 / 60, 1))


def main():
    h3 = json.loads((V3 / 'history.json').read_text(encoding='utf-8'))
    h2 = json.loads((V217 / 'history.json').read_text(encoding='utf-8'))
    run = json.loads((V3 / 'run.json').read_text(encoding='utf-8'))
    keys = ('jtpr', 'interior_jtpr', 'ap_nucleus', 'ap50_nucleus', 'ap_cytoplasm', 'ap50_cytoplasm',
            'ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated', 'nucleus_tp', 'nucleus_fp', 'nucleus_fn',
            'cytoplasm_tp', 'cytoplasm_fp', 'cytoplasm_fn', 'nucleus_quality_gate', 'nucleus_area_gate',
            'cytoplasm_quality_gate', 'cytoplasm_area_gate', 'nucleus_gt_without_prompt_center',
            'cytoplasm_gt_without_prompt_center')

    def table(history, first):
        deployed = history_selection_metrics(history, 0.7)
        e = history['fullval_epoch']
        return [dict(epoch=x, deployed_metric=deployed[i], **{k: (history.get('fullval_' + k) or [None] * len(e))[i]
                                                              for k in keys}) for i, x in enumerate(e) if x >= first]

    per_epoch = {}
    for package, history, epochs in ((V217, h2, (540, 545, 550)), (V3, h3, (555, 560, 565))):
        for e in epochs:
            gates = {h: gate_of(history, e, h) for h in HEADS}
            per_epoch[f'{package.name}/{e}'] = dict(timing=timing(package, e),
                                                    heads=jsonl_stats(package / f'epoch_{e:04d}_refinement.jsonl', gates))
    train_keys = [k for k in h3 if k.startswith('train_') and any(s in k for s in (
        'object_min_voxels', 'quality_pair_accuracy', 'bce_object', 'prob_gt_max_mean', '_mask'))]
    trend = {k: dict(v217_541_550=float(np.nanmean([v for v in h3[k][540:550] if v is not None] or [np.nan])),
                     v3_551_565=float(np.nanmean([v for v in h3[k][550:565] if v is not None] or [np.nan])))
             for k in train_keys}
    report = dict(
        scope='Read-only; user package fixes_v3, no weights/TIFF. Times from file mtimes as copied by the user.',
        script_sha256=digest(__file__),
        inputs={p.name: digest(p) for p in sorted(V3.iterdir()) if p.is_file()},
        run=dict(resume=run['requested_config']['training']['resume_from_checkpoint'],
                 reset_best=run['requested_config']['training']['reset_best_on_resume'],
                 options={k: run['requested_config'][s][k] for s, k in (
                     ('inference', 'refinement_roi_snap'), ('inference', 'residual_proposals'),
                     ('loss_params', 'mask_overlap_loss'), ('loss_params', 'object_losses_min_area_from_gate'),
                     ('loss_params', 'object_losses_include_crop_cut'))}),
        source_matches_local={k: (ROOT / k).exists() and digest(ROOT / k) == v for k, v in run['source_sha256'].items()},
        history_prefix_equal_v217=all(h3[k][:len(h2['fullval_epoch'])] == h2[k][:len(h2['fullval_epoch'])]
                                      for k in ('fullval_epoch', 'fullval_jtpr')),
        best=json.loads((V3 / 'best_fullval.json').read_text(encoding='utf-8')),
        fullval=table(h3, 525), per_epoch=per_epoch, train_trend=trend)
    target = ROOT / 'audit' / 'v3_epoch565_artifacts.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    brief = {k: dict(t=v['timing']['estimate_61_volumes_min'],
                     **{h: dict(raw_ap=round(v['heads'][h]['raw']['ap'], 4), gated_ap=round(v['heads'][h]['gated']['ap'], 4),
                                fp=v['heads'][h]['raw']['fp'], gfp=v['heads'][h]['gated']['fp'],
                                med=round(v['heads'][h]['raw']['tp_iou_quantiles_10_25_50_75_90'][2], 3),
                                ge09=round(v['heads'][h]['raw']['tp_share_iou_ge_090'], 3),
                                bg=v['heads'][h]['background_masks']['count'],
                                ppv=round(v['heads'][h]['prompts_per_volume'], 1)) for h in HEADS})
             for k, v in per_epoch.items()}
    print(json.dumps(dict(source=report['source_matches_local'], prefix=report['history_prefix_equal_v217'],
                          brief=brief), indent=1, default=float))


if __name__ == '__main__':
    main()
