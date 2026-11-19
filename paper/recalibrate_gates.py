"""Quality gates of a trained run chosen for AP alone, from the full validation that selected its checkpoint, without
predicting again. Training calibrates the per-head quality and size gates for its checkpoint metric (weight
loss_params.checkpoint_w_jtpr on JTPR, the rest on AP); every written mask of that full validation is logged in
visualizations/epoch_<epoch>_refinement.jsonl (quality, IoU with its ground truth, area), so the gates that maximise
the AP of each head alone are found again with cytonucs_runtime.calibrate_quality_gates and a JTPR weight of 0.
The result goes into inference.quality_gates_override of a scoring config (score_hierarchy.py).

Only the AP objective is offered: the joint pairs a removal would lose are not in the log.

Usage: python paper/recalibrate_gates.py RUN_DIR [--epoch N] [--output FILE]
       (RUN_DIR = output.basedir/experiment.name; epoch default: best_fullval.json)
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repository root: the CytoNucs modules

import numpy as np

TAUS = np.arange(0.5, 1.0, 0.05)
HEADS = ('nucleus', 'cytoplasm')


def decision_rows(log):
    """{head: [(quality, IoU, 0, area)]} of the written masks and {head: ground-truth objects} of a full validation."""
    rows, truth = {h: [] for h in HEADS}, {h: 0 for h in HEADS}
    for line in Path(log).read_text().splitlines():
        audit = json.loads(line).get('stage_audit') or {}
        for head in HEADS:
            if head not in audit:
                continue
            truth[head] += int(audit[head]['monitoring']['gt_count'])
            for d in audit[head]['assembly_decisions']:
                if d['reason'] == 'written':
                    rows[head].append((d['quality'], max(d['written_gt_iou'].values(), default=0.), 0,
                                       d.get('connected_voxels', d['area'])))
    return {h: r for h, r in rows.items() if r}, truth


def gated_ap(rows, truth, gate, min_area):
    from cytonucs_runtime import accuracy_profile
    keep = [r for r in rows if r[0] >= gate and r[3] >= min_area]
    iou = np.array([r[1] for r in keep])
    tp = np.array([(iou >= t).sum() for t in TAUS])
    return float(accuracy_profile(tp, len(keep) - tp, truth - tp).mean())


def recalibrate(run, epoch=None):
    from cytonucs_runtime import calibrate_quality_gates
    run = Path(run)
    best = json.loads((run / 'best_fullval.json').read_text())
    epoch = int(epoch or best['epoch'])
    log = run / 'visualizations' / f'epoch_{epoch:04d}_refinement.jsonl'
    rows, truth = decision_rows(log)
    counts = {}
    for head, r in rows.items():
        iou = np.array([x[1] for x in r])
        tp = np.array([(iou >= t).sum() for t in TAUS])
        counts[head] = dict(tp=tp, fp=len(iou) - tp, fn=truth[head] - tp)
    gates, stats = calibrate_quality_gates(rows, counts, TAUS, 1, 0.0)
    stored = best.get('quality_gates') or {}
    report = dict(run=str(run), epoch=epoch, log=str(log), objective='AP alone (JTPR weight 0)', gates=gates, heads={})
    for head, r in rows.items():
        report['heads'][head] = dict(
            masks=len(r), ground_truth=truth[head], ap_ungated=stats[head]['ap_ungated'],
            ap_stored_gates=gated_ap(r, truth[head], stored.get(head, 0.), stored.get(head + '_min_area', 0)),
            stored_gate=stored.get(head), stored_min_area=stored.get(head + '_min_area'),
            ap_gates=stats[head]['ap'], fp_removed=stats[head]['fp_removed'], tp_removed=stats[head]['tp_removed'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--output', help='default: RUN_DIR/quality_gates_ap.json')
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    report = recalibrate(args.run, args.epoch)
    out = Path(args.output or Path(args.run) / 'quality_gates_ap.json')
    out.write_text(json.dumps(report, indent=1))
    for head, h in report['heads'].items():
        print(f"{head}: val AP ungated {h['ap_ungated']:.3f}, stored gates {h['ap_stored_gates']:.3f}, AP gates "
              f"{h['ap_gates']:.3f} ({h['fp_removed']} FP / {h['tp_removed']} TP removed)")
    print('gates', report['gates'], '->', out)


if __name__ == '__main__':
    main()
