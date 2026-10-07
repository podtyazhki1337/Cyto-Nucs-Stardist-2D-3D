"""Metrics after the calibrated gates for whole objects versus objects cut by an XY face of the patch (read-only).

Usage: python audit/analyze_whole_objects.py [epoch ...] -> audit/whole_objects_epoch<last>.json
Input: fixes_v3.2 history.json and epoch_NNNN_refinement.jsonl.
- A GT object is 'cut' when its bounding box touches a Y or X face of the validation volume.
- Scope 'whole' / 'cut': GT of the other kind are ignore regions (COCO style) - not FN, and a written mask
  lying mostly on such a GT is neither TP nor FP. Masks without any GT overlap are counted only in 'all'
  (their scope is unknown), so FP of the two scopes do not add up to 'all'.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytonucs_runtime import accuracy_profile

PACKAGE = ROOT / 'fixes_v3.2'
TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)


def scope_metrics(epoch, head, gate, min_area):
    data = {s: dict(ious=[], n_gt=0) for s in ('all', 'whole', 'cut')}
    for line in (PACKAGE / f'epoch_{epoch:04d}_refinement.jsonl').read_text().splitlines():
        a = json.loads(line)['stage_audit'][head]
        geo = a['gt_geometry']
        cut = {g: any(f[0] in 'yx' for f in v['faces'].split(',') if f) for g, v in geo.items()}
        data['all']['n_gt'] += len(geo)
        data['cut']['n_gt'] += sum(cut.values())
        data['whole']['n_gt'] += len(geo) - sum(cut.values())
        for d in a['assembly_decisions']:
            area = d.get('connected_voxels', d['area'])
            if d['reason'] != 'written' or d['quality'] < gate or area < min_area:
                continue
            inter = {g: v * (area + geo[g]['volume']) / (1 + v) for g, v in d['written_gt_iou'].items() if g in geo and v > 0}
            if not inter:
                data['all']['ious'].append(0.)
                continue
            g = max(inter, key=inter.get)
            iou = d['written_gt_iou'][g]
            data['all']['ious'].append(iou)
            data['cut' if cut[g] else 'whole']['ious'].append(iou)
    out = {}
    for scope, v in data.items():
        ious = np.asarray(v['ious'], np.float64)
        tp = np.array([(ious >= t).sum() for t in TAUS])
        acc = accuracy_profile(tp, len(ious) - tp, v['n_gt'] - tp)
        out[scope] = dict(gt=v['n_gt'], masks=len(ious), tp50=int(tp[0]), ap=float(acc.mean()),
                          accuracy={str(t): float(x) for t, x in zip(TAUS, acc)},
                          recall50=float(tp[0] / max(v['n_gt'], 1)),
                          precision50=float(tp[0] / max(len(ious), 1)),
                          tp_iou_median=float(np.median(ious[ious >= .5])) if tp[0] else None)
    return out


def main(epochs):
    history = json.loads((PACKAGE / 'history.json').read_text(encoding='utf-8'))
    report = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results={})
    for epoch in epochs:
        i = history['fullval_epoch'].index(epoch)
        entry = dict(jtpr=history['fullval_jtpr'][i], interior_jtpr=history['fullval_interior_jtpr'][i],
                     interior_total_pairs=history['fullval_interior_total_pairs'][i],
                     metric_after_gates=history['fullval_metric_deployed'][i])
        for head in ('nucleus', 'cytoplasm'):
            entry[head] = scope_metrics(epoch, head, history[f'fullval_{head}_quality_gate'][i],
                                        history[f'fullval_{head}_area_gate'][i] or 0)
            for scope, m in entry[head].items():
                print(epoch, head, '%-5s' % scope, 'GT %4d recall50 %.3f precision50 %.3f AP %.4f acc@.5/.75/.9 %.3f/%.3f/%.3f IoU %.3f' % (
                    m['gt'], m['recall50'], m['precision50'], m['ap'], m['accuracy']['0.5'], m['accuracy']['0.75'],
                    m['accuracy']['0.9'], m['tp_iou_median']))
        report['results'][str(epoch)] = entry
    (ROOT / 'audit' / f'whole_objects_epoch{epochs[-1]}.json').write_text(json.dumps(report, indent=2, default=float),
                                                                          encoding='utf-8')


if __name__ == '__main__':
    main([int(e) for e in sys.argv[1:]] or [820, 860])
