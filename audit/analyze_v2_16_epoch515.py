"""Read-only review of fixes_v2.16 (full validation 515, first 5 epochs of v2.16).

Usage: python audit/analyze_v2_16_epoch515.py -> audit/v2_16_epoch515_artifacts.json
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'audit'))
from analyze_fp_fn_455 import analyze

PACKAGE = ROOT / 'fixes_v2.16'
HEADS = ('nucleus', 'cytoplasm')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def face_group(faces):
    z = 'z' in faces; xy = 'y' in faces or 'x' in faces
    return 'z+xy' if z and xy else 'z' if z else 'xy' if xy else 'interior'


def main():
    history = json.loads((PACKAGE / 'history.json').read_text(encoding='utf-8'))
    run = json.loads((PACKAGE / 'run.json').read_text(encoding='utf-8'))
    epochs = history['fullval_epoch']
    keys = ('jtpr', 'metric', 'interior_jtpr', 'ap_nucleus', 'ap50_nucleus', 'ap_cytoplasm', 'ap50_cytoplasm',
            'nucleus_tp', 'nucleus_fp', 'nucleus_fn', 'cytoplasm_tp', 'cytoplasm_fp', 'cytoplasm_fn',
            'ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated', 'nucleus_proposal_below_threshold',
            'nucleus_gt_without_prompt_center', 'cytoplasm_gt_without_prompt_center')
    fullval = [dict(epoch=e, **{k: history.get('fullval_' + k, [None] * len(epochs))[i] for k in keys})
               for i, e in enumerate(epochs) if e >= 465]
    train_keys = ('nucleus_bce_object_peak', 'cytoplasm_bce_object_peak', 'cytoplasm_bce_object',
                  'nucleus_quality_pair_accuracy', 'cytoplasm_quality_pair_accuracy',
                  'nucleus_quality_rank_pairs', 'cytoplasm_quality_rank_pairs')
    new_losses = {s + '_' + k: history.get(s + '_' + k, [None] * 515)[510:515]
                  for s in ('train', 'val') for k in train_keys}
    jsonl = PACKAGE / 'epoch_0515_refinement.jsonl'
    decomposition, _ = analyze(jsonl)
    faces = {h: Counter() for h in HEADS}
    volumes = {h: {'TP': [], 'FN': []} for h in HEADS}
    refinement = {h: Counter() for h in HEADS}
    for line in jsonl.read_text().splitlines():
        r = json.loads(line)
        for h in HEADS:
            a = r['stage_audit'][h]
            for g, v in a['stages']['final']['best_iou_per_gt'].items():
                k = 'TP' if v >= .5 else 'FN'
                faces[h][face_group(a['gt_geometry'][g]['faces']) + ':' + k] += 1
                volumes[h][k].append(a['gt_geometry'][g]['volume'])
            for key in ('attempted', 'written', 'residual_peaks', 'residual_prompts', 'written_residual',
                        'area_gate_removed', 'quality_rejected'):
                refinement[h][key] += r['refinement'][h].get(key) or 0
    face_recall = {h: {g: dict(tp=faces[h][g + ':TP'], fn=faces[h][g + ':FN'],
                               recall=faces[h][g + ':TP'] / max(1, faces[h][g + ':TP'] + faces[h][g + ':FN']))
                       for g in ('interior', 'z', 'xy', 'z+xy')} for h in HEADS}
    sizes = {h: {k: dict(q10_50_90=np.quantile(v, [.1, .5, .9]).round(0).tolist(), below_1000=int((np.array(v) < 1000).sum()))
                 for k, v in volumes[h].items()} for h in HEADS}
    check = json.loads((ROOT / 'audit' / 'truncation_flags_check.json').read_text(encoding='utf-8'))
    report = dict(
        scope='Read-only; user package, no weights/TIFF. JTPR changes between neighbouring fullvals are ~±0.02.',
        script_sha256=digest(__file__),
        inputs={p.name: digest(p) for p in sorted(PACKAGE.iterdir()) if p.is_file()},
        run=dict(resume=run['requested_config']['training']['resume_from_checkpoint'],
                 reset_best=run['requested_config']['training']['reset_best_on_resume'],
                 loss_params={k: run['requested_config']['loss_params'][k] for k in
                              ('nucleus_prob_object_weight', 'cytoplasm_prob_object_weight',
                               'prob_object_peak_weight', 'mask_quality_rank_weight')},
                 residual_proposals=run['requested_config']['inference']['residual_proposals']),
        source_matches_local={k: (ROOT / k).exists() and digest(ROOT / k) == v
                              for k, v in run['source_sha256'].items()},
        best=json.loads((PACKAGE / 'best_fullval.json').read_text(encoding='utf-8')),
        fullval=fullval, new_losses_511_515=new_losses, decomposition_515=decomposition,
        recall_by_volume_face=face_recall, gt_volume_by_outcome=sizes, refinement_totals=dict(refinement),
        truncation_flags_check=check['summary'])
    target = ROOT / 'audit' / 'v2_16_epoch515_artifacts.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(face_recall=face_recall, refinement=report['refinement_totals'],
                          sources=report['source_matches_local']), indent=1))


if __name__ == '__main__':
    main()
