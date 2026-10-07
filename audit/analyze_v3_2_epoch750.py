"""Read-only review of the updated fixes_v3.2 package: the run continued from weights_epoch_740 with the
29.09 code (fine head on a random prompt slot, canonical optimizer order, by-name weights, edge metric).

Usage: python audit/analyze_v3_2_epoch750.py -> audit/v3_2_epoch750_artifacts.json
Deployed errors reuse analyze_v3_2_epoch735.deployed_errors (after the epoch's calibrated gates).
Full-validation time: file mtimes of the first-volume PNG and the final JSONL (60 volumes, scaled to 61).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'audit'))
from cytonucs_runtime import history_selection_metrics
from analyze_v3_2_epoch735 import deployed_errors, gate_of, HEADS

PACKAGE = ROOT / 'fixes_v3.2'
EPOCHS = (730, 735, 745, 750)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    h = json.loads((PACKAGE / 'history.json').read_text(encoding='utf-8'))
    run = json.loads((PACKAGE / 'run.json').read_text(encoding='utf-8'))
    console = (PACKAGE / 'console.log').read_text(encoding='utf-8', errors='replace')
    deployed = history_selection_metrics(h, 0.7)
    e = h['fullval_epoch']
    keys = ('jtpr', 'ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated', 'ap_nucleus_quality_gated_edge_ignored',
            'ap_cytoplasm_quality_gated_edge_ignored', 'nucleus_recall50_quality_gated_edge_ignored',
            'cytoplasm_recall50_quality_gated_edge_ignored', 'nucleus_tp', 'nucleus_fp', 'nucleus_fn', 'cytoplasm_tp',
            'cytoplasm_fp', 'cytoplasm_fn', 'nucleus_area_gate', 'cytoplasm_area_gate',
            'nucleus_gt_without_prompt_center', 'cytoplasm_gt_without_prompt_center')
    table = [dict(epoch=x, deployed_metric=deployed[i], **{k: (h.get('fullval_' + k) or [None] * len(e))[i] for k in keys})
             for i, x in enumerate(e) if x >= 720]
    per_epoch = {}
    for ep in EPOCHS:
        jsonl = PACKAGE / f'epoch_{ep:04d}_refinement.jsonl'
        minutes = (jsonl.stat().st_mtime - (PACKAGE / f'epoch_{ep:04d}_fullval.png').stat().st_mtime) / 60 * 61 / 60
        prompts = {hd: 0 for hd in HEADS}
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            for hd in HEADS:
                prompts[hd] += r['refinement'][hd].get('attempted') or 0
        per_epoch[ep] = dict(fullval_minutes=round(minutes, 1), prompts_per_volume={k: v / 61 for k, v in prompts.items()},
                             deployed_errors=deployed_errors(jsonl, {hd: gate_of(h, ep, hd) for hd in HEADS}))

    def mean(k, a, b):
        v = [x for x in h.get(k, [])[a:b] if x is not None]
        return float(np.mean(v)) if v else None
    losses = {k: dict(train_721_730=mean('train_' + k, 720, 730), train_741_750=mean('train_' + k, 740, 750),
                      val_721_730=mean('val_' + k, 720, 730), val_741_750=mean('val_' + k, 740, 750))
              for k in ('nucleus_fine_mask', 'cytoplasm_fine_mask', 'nucleus_mask', 'cytoplasm_mask', 'loss')}
    resume_lines = [l.strip() for l in console.splitlines()
                    if re.search(r'Resuming from checkpoint|New heads added|restored by name|Restored Adam|'
                                 r'Resume effective|Reset best|refusing', l)]
    report = dict(
        scope='Read-only; user package fixes_v3.2 as updated on 2026-09-29 (console, run, history, effective config '
              'replaced; fullval 745/750 added). No weights/TIFF.',
        script_sha256=digest(__file__),
        inputs={p.name: digest(p) for p in sorted(PACKAGE.iterdir()) if p.is_file()},
        run=dict(resume=run['requested_config']['training']['resume_from_checkpoint'],
                 reset_best=run['requested_config']['training']['reset_best_on_resume'],
                 train_epochs=run['requested_config']['training']['train_epochs']),
        source_matches_local={k: (ROOT / k).exists() and digest(ROOT / k) == v for k, v in run['source_sha256'].items()},
        resume_log=resume_lines,
        best=json.loads((PACKAGE / 'best_fullval.json').read_text(encoding='utf-8')),
        fullval=table, per_epoch=per_epoch, losses=losses)
    (ROOT / 'audit' / 'v3_2_epoch750_artifacts.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(json.dumps(dict(run=report['run'], sources=report['source_matches_local'], resume=resume_lines), indent=1))


if __name__ == '__main__':
    main()
