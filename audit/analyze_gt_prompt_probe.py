"""Read-only review of audit/gt_prompt_probe_current.json (detector prompts vs GT-prompt oracle).

Usage: python audit/analyze_gt_prompt_probe.py -> audit/gt_prompt_probe_review.json
Per head and volume: one-to-one matches at threshold/before_nms/final stages for both
prompt sources; for detector FN, whether a GT prompt yields a >=0.5 candidate, the best
candidate IoU of both sources, proposal reason and GT geometry (volume, faces).
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADS = ('nucleus', 'cytoplasm')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def face_group(faces):
    z = 'z' in faces; xy = 'y' in faces or 'x' in faces
    return 'z+xy' if z and xy else 'z' if z else 'xy' if xy else 'interior'


def main(path=ROOT / 'audit' / 'gt_prompt_probe_current.json', snapshot=None):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    source = snapshot or ROOT
    matches = {k: (Path(source) / k).exists() and digest(Path(source) / k) == v for k, v in data['source_sha256'].items()}
    volumes, totals = [], {h: Counter() for h in HEADS}
    faces = {h: Counter() for h in HEADS}
    fn_rows = {h: [] for h in HEADS}
    for r in data['results']:
        row = dict(filename=r['filename'], joint_tp=dict(detector=r['pairs']['joint_tp'],
                   oracle=r['gt_prompt_oracle']['pairs']['joint_tp'], total=r['pairs']['total_pairs']))
        for h in HEADS:
            det, orc = r['stage_audit'][h], r['gt_prompt_oracle']['stage_audit'][h]
            stage = lambda a: {s: a['stages'][s]['oracle_one_to_one_tp'] for s in ('threshold', 'before_nms', 'final')}
            row[h] = dict(gt=det['stages']['final']['gt_count'], detector=stage(det), oracle=stage(orc))
            for s, v in stage(det).items():
                totals[h]['detector_' + s] += v
            for s, v in stage(orc).items():
                totals[h]['oracle_' + s] += v
            totals[h]['gt'] += det['stages']['final']['gt_count']
            fd = det['stages']['final']['best_iou_per_gt']; td = det['stages']['threshold']['best_iou_per_gt']
            to = orc['stages']['threshold']['best_iou_per_gt']
            per, geo = det['proposals']['per_gt'], det['gt_geometry']
            for g, v in fd.items():
                faces[h][(face_group(geo[g]['faces']), 'TP' if v >= .5 else 'FN')] += 1
                if v >= .5:
                    continue
                reason = per.get(g, {}).get('reason')
                kind = 'no_center' if reason != 'selected' and not det['prompt_centers_per_gt'].get(g) else 'with_center'
                fn_rows[h].append(dict(kind=kind, reason=reason, max_prob=per.get(g, {}).get('max_prob_grid'),
                                       volume=geo[g]['volume'], faces=geo[g]['faces'],
                                       detector_iou=td.get(g, 0.), oracle_iou=to.get(g, 0.)))
        volumes.append(row)
    summary = {}
    for h in HEADS:
        rows = fn_rows[h]
        by_kind = {}
        for kind in ('no_center', 'with_center'):
            sel = [x for x in rows if x['kind'] == kind]
            by_kind[kind] = dict(
                n=len(sel),
                oracle_candidate_ge_05=sum(x['oracle_iou'] >= .5 for x in sel),
                detector_candidate_iou_median=float(np.median([x['detector_iou'] for x in sel])) if sel else None,
                oracle_candidate_iou_median=float(np.median([x['oracle_iou'] for x in sel])) if sel else None,
                volume_median=float(np.median([x['volume'] for x in sel])) if sel else None,
                max_prob_median=float(np.median([x['max_prob'] for x in sel if x['max_prob'] is not None])) if sel else None)
        summary[h] = dict(stage_totals=dict(totals[h]), fn_by_kind=by_kind,
                          tp_fn_by_face={f'{k[0]}:{k[1]}': v for k, v in sorted(faces[h].items())})
    report = dict(scope='Two validation volumes; GT-prompt oracle uses GT centers/extent, base ROI without '
                        'ray expansion; not validation JTPR.',
                  input=str(Path(path).relative_to(ROOT)).replace('\\', '/'), input_sha256=digest(path),
                  script_sha256=digest(__file__), checkpoint=data['checkpoint'],
                  checkpoint_sha256=data['checkpoint_sha256'], resume_from=data['config']['training']['resume_from_checkpoint'],
                  source_matches=matches, volumes=volumes, summary=summary, fn=fn_rows)
    target = ROOT / 'audit' / 'gt_prompt_probe_review.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(source_matches=matches, volumes=volumes, summary=summary), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main(snapshot=sys.argv[1] if len(sys.argv) > 1 else None)
