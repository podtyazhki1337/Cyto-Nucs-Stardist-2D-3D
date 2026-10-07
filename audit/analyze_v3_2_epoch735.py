"""Read-only review of fixes_v3.2 (full-resolution mask head; full validations 655-735) and of the
hypothesis that the many loss terms fail to suppress FP/FN.

Usage: python audit/analyze_v3_2_epoch735.py -> audit/v3_2_epoch735_artifacts.json
- Deployed errors: written masks after the epoch's calibrated gates (quality, min area), TP at IoU>=0.5.
  FN by first cause: GT smaller than the area gate / good mask removed by the gate / no prompt centre /
  prompted but no candidate with IoU>=0.5 / good candidate lost in assembly. FP by kind: background
  (IoU<0.1), partial mask of an unmatched GT (the same error is that GT's FN), duplicate/piece of a
  matched GT.
- Gate trade-off: written masks the gate removes, TP vs FP; GT below the area gate and whether any
  candidate reached IoU>=0.5 before the gate.
- Loss terms: train/val means of every history component, epochs 651-660 vs 726-735; gradient-norm
  groups from console [GRAD NORMS] lines (parameter-group norms, not per-loss contributions).
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cytonucs_runtime import history_selection_metrics

PACKAGE = ROOT / 'fixes_v3.2'
HEADS = ('nucleus', 'cytoplasm')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def best(d):
    return max(d['written_gt_iou'].items(), key=lambda x: x[1]) if d['written_gt_iou'] else (None, 0.)


def deployed_errors(path, gates):
    out = {}
    for h in HEADS:
        q_gate, a_gate = gates[h]
        fp, fn, trade, small = Counter(), Counter(), Counter(), Counter()
        tp = 0
        for line in Path(path).read_text().splitlines():
            a = json.loads(line)['stage_audit'][h]
            geo, centres = a['gt_geometry'], a['prompt_centers_per_gt']
            candidate = a['stages']['component']['best_iou_per_gt']
            final = a['stages']['final']['best_iou_per_gt']
            written = [d for d in a['assembly_decisions'] if d['reason'] == 'written']
            keep = lambda d: (d.get('quality') is None or d['quality'] >= q_gate) and \
                d.get('connected_voxels', d['area']) >= a_gate
            kept = [d for d in written if keep(d)]
            matched = {best(d)[0] for d in kept if best(d)[1] >= .5}
            tp += sum(best(d)[1] >= .5 for d in kept)
            for d in kept:
                g, v = best(d)
                if v < .5:
                    fp['background' if v < .1 else 'duplicate_or_piece_of_matched_gt' if g in matched
                       else 'partial_of_unmatched_gt'] += 1
            removed_good = {best(d)[0] for d in written if not keep(d) and best(d)[1] >= .5}
            for d in written:
                if not keep(d):
                    trade['removed_TP' if best(d)[1] >= .5 else 'removed_FP'] += 1
            for g in geo:
                if geo[g]['volume'] < a_gate:
                    small['gt'] += 1
                    small['candidate_iou_ge_0.5_before_gate'] += candidate.get(g, 0) >= .5
                    small['final_iou_ge_0.5_ungated'] += final.get(g, 0) >= .5
                if g in matched:
                    continue
                fn['smaller_than_area_gate' if geo[g]['volume'] < a_gate else
                   'good_mask_removed_by_gate' if g in removed_good else
                   'no_prompt_center' if not centres.get(g) else
                   'prompted_bad_mask' if candidate.get(g, 0) < .5 else 'good_candidate_lost_in_assembly'] += 1
        n_fp, n_fn = sum(fp.values()), sum(fn.values())
        out[h] = dict(gates=dict(quality=q_gate, min_area=a_gate), tp=tp, fp=n_fp, fn=n_fn,
                      accuracy_at_05=tp / (tp + n_fp + n_fn), fp_kinds=dict(fp), fn_causes=dict(fn),
                      gate_tradeoff=dict(trade), gt_below_area_gate=dict(small))
    return out


def gate_of(history, epoch, head):
    i = history['fullval_epoch'].index(epoch)
    return history[f'fullval_{head}_quality_gate'][i], history[f'fullval_{head}_area_gate'][i]


def loss_terms(h):
    def mean(k, a, b):
        v = [x for x in h.get(k, [])[a:b] if x is not None]
        return float(np.mean(v)) if v else None
    return {k[6:]: dict(train_651_660=mean(k, 650, 660), train_726_735=mean(k, 725, 735),
                        val_726_735=mean('val_' + k[6:], 725, 735))
            for k in sorted(h) if k.startswith('train_')}


def grad_norms(console):
    pattern = re.compile(r'\[GRAD NORMS\] prob: ([\d.]+) dist: ([\d.]+) backbone: ([\d.]+) cyto_dec: ([\d.]+) '
                         r'\|\| refine: ([\d.]+)')
    values = np.array([[float(x) for x in m.groups()] for m in pattern.finditer(console.read_text(errors='replace'))])
    names = ('prob_head', 'dist_head', 'backbone', 'cytoplasm_decoder', 'refinement')
    return dict(count=len(values), mean={n: float(v) for n, v in zip(names, values.mean(0))},
                median={n: float(v) for n, v in zip(names, np.median(values, 0))})


def main():
    h = json.loads((PACKAGE / 'history.json').read_text(encoding='utf-8'))
    run = json.loads((PACKAGE / 'run.json').read_text(encoding='utf-8'))
    deployed = history_selection_metrics(h, 0.7)
    e = h['fullval_epoch']
    keys = ('jtpr', 'ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated', 'ap_nucleus', 'ap_cytoplasm',
            'nucleus_tp', 'nucleus_fp', 'nucleus_fn', 'cytoplasm_tp', 'cytoplasm_fp', 'cytoplasm_fn',
            'nucleus_gt_without_prompt_center', 'cytoplasm_gt_without_prompt_center')
    table = [dict(epoch=x, deployed_metric=deployed[i], **{k: (h.get('fullval_' + k) or [None] * len(e))[i] for k in keys})
             for i, x in enumerate(e) if x >= 630]
    errors = {ep: deployed_errors(PACKAGE / f'epoch_{ep:04d}_refinement.jsonl', {hd: gate_of(h, ep, hd) for hd in HEADS})
              for ep in (725, 730, 735)}
    report = dict(
        scope='Read-only; user package fixes_v3.2, no weights/TIFF.',
        script_sha256=digest(__file__),
        inputs={p.name: digest(p) for p in sorted(PACKAGE.iterdir()) if p.is_file()},
        run=dict(resume=run['requested_config']['training']['resume_from_checkpoint'],
                 reset_best=run['requested_config']['training']['reset_best_on_resume'],
                 refinement_full_resolution=run['requested_config']['model']['refinement_full_resolution'],
                 mask_ce_class_balance=run['requested_config']['loss_params']['mask_ce_class_balance']),
        source_matches_local={k: (ROOT / k).exists() and digest(ROOT / k) == v for k, v in run['source_sha256'].items()},
        best=json.loads((PACKAGE / 'best_fullval.json').read_text(encoding='utf-8')),
        fullval=table, deployed_errors=errors, loss_terms=loss_terms(h),
        grad_norm_groups=grad_norms(PACKAGE / 'console.log'))
    (ROOT / 'audit' / 'v3_2_epoch735_artifacts.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(json.dumps(dict(run=report['run'], sources=report['source_matches_local'],
                          errors_730=errors[730], grad=report['grad_norm_groups']), indent=1, default=float))


if __name__ == '__main__':
    main()
