"""Why AP stops where it does: independent read-outs of the boundary error, its shape, and what is left (read-only).

Usage: python audit/analyze_ap_ceiling.py [epoch] -> audit/ap_ceiling_epoch<epoch>.json
Input: fixes_v3.2 history.json and epoch_NNNN_refinement.jsonl (with the boundary split).
1. readouts: three parts of the network that locate the boundary in different ways - ray distances (regression,
   dominant loss), semantic foreground head (per-voxel classifier with full gradient) and prompted masks - on
   train and validation, averaged over the 10 epochs before `epoch`.
2. shape: least squares over matched masks of  missing voxels = a_z*Z + a_xy*XY + a_rims*rims  (error voxels of
   the three parts of boundary_split); a_part estimates the share of that part's error that is GT not covered.
3. failures: what happened to every GT, for whole objects and for objects cut by an XY face of the patch.
4. gate trade-off: quality/area gates re-calibrated for different JTPR weights, every removed matched mask
   counted as one lost pair (upper bound of the JTPR cost).
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytonucs_runtime import calibrate_quality_gates

PACKAGE = ROOT / 'fixes_v3.2'
TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)
HEADS = ('nucleus', 'cytoplasm')


def mean(history, key, epoch, span=10):
    values = [v for v in history.get(key, [])[epoch - span:epoch] if v is not None]
    return float(np.mean(values)) if values else None


def readouts(history, epoch):
    out = {}
    for head in HEADS:
        out[head] = {f'{split}_{name}': mean(history, f'{split}_{head}_{key}', epoch)
                     for split in ('train', 'val')
                     for name, key in (('ray_distance_mae', 'distance'), ('semantic_dice', 'semantic_dice'),
                                       ('mask_loss', 'mask'))}
    weights = dict(nucleus_distance=0.2, cytoplasm_distance=0.5)      # lambda_dist_* of config_cytonucs_v3_2.json
    total = mean(history, 'train_loss', epoch)
    out['distance_share_of_train_loss'] = sum(w * mean(history, 'train_' + k, epoch) for k, w in weights.items()) / total
    return out


def records(epoch):
    for line in (PACKAGE / f'epoch_{epoch:04d}_refinement.jsonl').read_text().splitlines():
        yield json.loads(line)['stage_audit']


def is_cut(info):
    return any(f[0] in 'yx' for f in info['faces'].split(',') if f)


def shape(epoch, head, gate, min_area):
    x, y, cut = [], [], []
    for audit in records(epoch):
        a = audit[head]; geo = a['gt_geometry']
        for d in a['assembly_decisions']:
            b = d.get('boundary'); area = d.get('connected_voxels', d['area'])
            if d['reason'] != 'written' or b is None or d['quality'] < gate or area < min_area:
                continue
            volume = geo[b['gt']]['volume']; iou = d['written_gt_iou'][b['gt']]
            x.append([b['z_error_voxels'], b['xy_error_voxels'], b['mixed_error_voxels']])
            y.append(volume - iou * (area + volume) / (1 + iou))
            cut.append(is_cut(geo[b['gt']]))
    x, y, cut = np.asarray(x, float), np.asarray(y, float), np.asarray(cut)
    out = {}
    for name, m in (('all', np.ones(len(y), bool)), ('whole', ~cut), ('cut', cut)):
        coef = np.linalg.lstsq(x[m], y[m], rcond=None)[0]
        fit = x[m] @ coef
        out[name] = dict(masks=int(m.sum()), missing_share_z=float(coef[0]), missing_share_xy_projection=float(coef[1]),
                         missing_share_rims=float(coef[2]), missing_share_overall=float(y[m].sum() / x[m].sum()),
                         r2=float(1 - ((y[m] - fit) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()))
    return out


def failures(epoch, head, gate, min_area):
    table = {'whole': Counter(), 'cut': Counter()}
    for audit in records(epoch):
        a = audit[head]; geo = a['gt_geometry']
        before = a['stages']['before_nms']['best_iou_per_gt']; prompts = a['prompt_centers_per_gt']
        best = {}
        for d in a['assembly_decisions']:
            if d['reason'] != 'written':
                continue
            area = d.get('connected_voxels', d['area'])
            for g, v in d['written_gt_iou'].items():
                if v > best.get(g, (0.,))[0]:
                    best[g] = (v, d['quality'], area)
        for g, info in geo.items():
            iou, quality, area = best.get(g, (0., 0., 0))
            if iou >= .5 and quality >= gate and area >= min_area:
                kind = 'found'
            elif info['volume'] < min_area:
                kind = 'gt_smaller_than_area_gate'
            elif iou >= .5:
                kind = 'found_removed_by_gate'
            elif before.get(g, 0.) >= .5:
                kind = 'good_candidate_lost_in_assembly'
            elif not prompts.get(g, 0):
                kind = 'no_prompt_in_object'
            else:
                kind = 'prompt_but_no_mask_at_iou_0.5'
            table['cut' if is_cut(info) else 'whole'][kind] += 1
    return {k: dict(v) for k, v in table.items()}


def gate_tradeoff(epoch, head, total_pairs):
    rows, n_gt = [], 0
    for audit in records(epoch):
        a = audit[head]; n_gt += a['stages']['final']['gt_count']
        for d in a['assembly_decisions']:
            if d['reason'] == 'written':
                iou = max(d['written_gt_iou'].values(), default=0.)
                rows.append((d['quality'], iou, 1 if iou >= .5 else 0, d.get('connected_voxels', d['area'])))
    ious = np.array([r[1] for r in rows]); tp = np.array([(ious >= t).sum() for t in TAUS])
    counts = dict(tp=tp, fp=len(ious) - tp, fn=n_gt - tp)
    out = {}
    for weight in (0.7, 0.5, 0.3, 0.0):
        gates, stats = calibrate_quality_gates({'h': rows}, {'h': counts}, TAUS, total_pairs, weight)
        s = stats['h']
        out[str(weight)] = dict(quality_gate=gates['h'], min_area=s['min_area'], ap=s['ap'],
                                fp_removed=s['fp_removed'], matched_removed=s['tp_removed'])
    return out


def main(epoch):
    history = json.loads((PACKAGE / 'history.json').read_text(encoding='utf-8'))
    i = history['fullval_epoch'].index(epoch)
    report = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), epoch=epoch,
                  readouts=readouts(history, epoch))
    for head in HEADS:
        gate, min_area = history[f'fullval_{head}_quality_gate'][i], history[f'fullval_{head}_area_gate'][i] or 0
        report[head] = dict(ap_after_gates=history[f'fullval_ap_{head}_quality_gated'][i], shape=shape(epoch, head, gate, min_area),
                            failures=failures(epoch, head, gate, min_area), gate_tradeoff=gate_tradeoff(epoch, head, 1439))
        r = report['readouts'][head]; s = report[head]['shape']['all']
        print(head, 'ray MAE train/val %.2f/%.2f px | semantic Dice %.3f/%.3f | missing share Z/XY/rims %.2f/%.2f/%.2f' % (
            r['train_ray_distance_mae'], r['val_ray_distance_mae'], r['train_semantic_dice'], r['val_semantic_dice'],
            s['missing_share_z'], s['missing_share_xy_projection'], s['missing_share_rims']))
        print('   failures', report[head]['failures'])
        print('   gate', {w: (round(v['ap'], 4), v['matched_removed']) for w, v in report[head]['gate_tradeoff'].items()})
    print('distance share of train loss %.2f' % report['readouts']['distance_share_of_train_loss'])
    (ROOT / 'audit' / f'ap_ceiling_epoch{epoch}.json').write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 820)
