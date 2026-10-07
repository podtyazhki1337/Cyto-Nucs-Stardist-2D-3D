"""Read-only review of fixes_v2.17 (full validations 525-550) and an AP correctness check.

Usage: python audit/analyze_v2_17_epoch550.py -> audit/v2_17_epoch550_artifacts.json
AP check: recompute TP/FP/FN per IoU threshold from the written masks of the stage audit
(independent of the trainer's stardist matching) and compare with history.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'audit'))
from analyze_fp_fn_455 import analyze
from cytonucs_runtime import accuracy_profile, history_selection_metrics

PACKAGE = ROOT / 'fixes_v2.17'
HEADS = ('nucleus', 'cytoplasm')
TAUS = np.arange(0.5, 0.951, 0.05)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def face_group(faces):
    z = 'z' in faces; xy = 'y' in faces or 'x' in faces
    return 'z+xy' if z and xy else 'z' if z else 'xy' if xy else 'interior'


def jsonl_summary(path):
    rows = {h: [] for h in HEADS}; gt = Counter(); faces = {h: Counter() for h in HEADS}
    counters = {h: Counter() for h in HEADS}; background = Counter()
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        for h in HEADS:
            a = r['stage_audit'][h]
            gt[h] += a['stages']['final']['gt_count']
            for g, v in a['stages']['final']['best_iou_per_gt'].items():
                faces[h][face_group(a['gt_geometry'][g]['faces']) + ':' + ('TP' if v >= .5 else 'FN')] += 1
            for key in ('attempted', 'written', 'residual_prompts', 'written_residual'):
                counters[h][key] += r['refinement'][h].get(key) or 0
            for s in a['proposals']['sweeps']:
                if abs(s['threshold'] - a['proposals']['threshold']) < 1e-6:
                    counters[h]['background_local_maxima_at_threshold'] += s['background_local_maxima']
            for d in a['assembly_decisions']:
                if d['reason'] != 'written':
                    continue
                iou = max(d['written_gt_iou'].values(), default=0.)
                rows[h].append(iou)
                if h == 'nucleus' and iou < .1:
                    background['masks'] += 1
                    background['origin_on_low_xy_face'] += d['origin'][1] == 0 or d['origin'][2] == 0
                    background['area_below_200'] += d.get('connected_voxels', d['area']) < 200
    recomputed = {}
    for h in HEADS:
        iou = np.array(rows[h]); tp = np.array([(iou >= t).sum() for t in TAUS])
        acc = accuracy_profile(tp, len(iou) - tp, gt[h] - tp)
        recomputed[h] = dict(ap=float(acc.mean()), ap50=float(acc[0]), tp=int(tp[0]), fp=int(len(iou) - tp[0]))
    recall = {h: {g: faces[h][g + ':TP'] / max(1, faces[h][g + ':TP'] + faces[h][g + ':FN'])
                  for g in ('interior', 'z', 'xy', 'z+xy')} for h in HEADS}
    return recomputed, recall, {h: dict(c) for h, c in counters.items()}, dict(background)


def main():
    history = json.loads((PACKAGE / 'history.json').read_text(encoding='utf-8'))
    run = json.loads((PACKAGE / 'run.json').read_text(encoding='utf-8'))
    epochs = history['fullval_epoch']
    deployed = history_selection_metrics(history, 0.7)
    keys = ('jtpr', 'metric', 'interior_jtpr', 'ap_nucleus', 'ap50_nucleus', 'ap_cytoplasm', 'ap50_cytoplasm',
            'ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated', 'nucleus_tp', 'nucleus_fp', 'nucleus_fn',
            'cytoplasm_tp', 'cytoplasm_fp', 'cytoplasm_fn', 'nucleus_proposal_below_threshold',
            'nucleus_gt_without_prompt_center', 'cytoplasm_gt_without_prompt_center')
    table = [dict(epoch=e, deployed_metric=deployed[i],
                  **{k: (history.get('fullval_' + k) or [None] * len(epochs))[i] for k in keys})
             for i, e in enumerate(epochs) if e >= 465]
    checks, per_epoch = {}, {}
    for e in (540, 545, 550):
        recomputed, recall, counters, background = jsonl_summary(PACKAGE / f'epoch_{e:04d}_refinement.jsonl')
        i = epochs.index(e)
        checks[e] = {h: dict(recomputed=recomputed[h], history=dict(
            ap=history['fullval_ap_' + h][i], ap50=history['fullval_ap50_' + h][i],
            tp=history['fullval_' + h + '_tp'][i], fp=history['fullval_' + h + '_fp'][i]),
            ap_abs_diff=abs(recomputed[h]['ap'] - history['fullval_ap_' + h][i])) for h in HEADS}
        per_epoch[e] = dict(recall_by_face=recall, refinement=counters, nucleus_background_fp=background)
    _, recall515, counters515, background515 = jsonl_summary(ROOT / 'fixes_v2.16' / 'epoch_0515_refinement.jsonl')
    decomposition = {e: analyze(PACKAGE / f'epoch_{e:04d}_refinement.jsonl')[0] for e in (545,)}
    report = dict(
        scope='Read-only; user package, no weights/TIFF. Deployed metric = after calibrated gates, estimate '
              'from post-hoc removal of written masks (pair loss exact per head, double-counted across heads).',
        script_sha256=digest(__file__),
        inputs={p.name: digest(p) for p in sorted(PACKAGE.iterdir()) if p.is_file()},
        run=dict(resume=run['requested_config']['training']['resume_from_checkpoint'],
                 reset_best=run['requested_config']['training']['reset_best_on_resume'],
                 object_losses_include_crop_cut=run['requested_config']['loss_params']['object_losses_include_crop_cut']),
        source_matches_local={k: (ROOT / k).exists() and digest(ROOT / k) == v for k, v in run['source_sha256'].items()},
        best=json.loads((PACKAGE / 'best_fullval.json').read_text(encoding='utf-8')),
        fullval=table, ap_recompute_check=checks, per_epoch=per_epoch,
        reference_515=dict(recall_by_face=recall515, refinement=counters515, nucleus_background_fp=background515),
        decomposition=decomposition,
        epoch_555=dict(jsonl_volumes=len((PACKAGE / 'epoch_0555_refinement.jsonl').read_text().splitlines()),
                       note='incomplete full validation (not in history)'))
    target = ROOT / 'audit' / 'v2_17_epoch550_artifacts.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(json.dumps(dict(checks={e: {h: round(v['ap_abs_diff'], 6) for h, v in c.items()} for e, c in checks.items()},
                          recall_545=per_epoch[545]['recall_by_face'], background_545=per_epoch[545]['nucleus_background_fp']),
                     indent=1, default=float))


if __name__ == '__main__':
    main()
