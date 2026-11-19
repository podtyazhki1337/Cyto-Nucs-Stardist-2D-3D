"""Semantic gate of one level chosen on saved predictions, without predicting again: objects whose mean semantic
foreground probability (score_hierarchy.py --predictions with inference.semantic_scores: <predictions>/semantic/
<stem>.json) is below a threshold are removed. The threshold is the one with the best AP of the level on the
selection volumes (e.g. the val16 cases, as the checkpoints are chosen); each --apply target is then scored with
it and without the gate (score_hierarchy.score_image, links by overlap), and the threshold can go into
inference.semantic_min_<role> of the config.

Usage:
  python paper/semantic_gate.py --data DATASET --level mass --select PRED_DIR:val --select-names VAL16/val/images \\
      --apply PRED_TEST:test PRED_VAL73:val --output FILE
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repository root: the CytoNucs modules

import numpy as np

THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]


def gated(labels, scores, threshold):
    if not threshold:
        return labels
    drop = [int(k) for k, v in scores.items() if v < threshold]
    return np.where(np.isin(labels, drop), 0, labels) if drop else labels


def volume_records(job):
    """Records (score_hierarchy.score_image) of one volume, one per threshold."""
    import tifffile
    from score_hierarchy import ground_truth, score_image
    folder, predictions, name, level, levels, pairs, thresholds = job
    gt, truths, ignore = ground_truth(folder, name, levels, pairs)
    pred = {lv: tifffile.imread(predictions / lv / name) for lv in levels}
    scores = json.loads((predictions / 'semantic' / (Path(name).stem + '.json')).read_text())[level]
    return [score_image(gt, dict(pred, **{level: gated(pred[level], scores, t)}), pairs, truths, ignore=ignore)[0]
            for t in thresholds]


def score_folder(data, predictions, split, level, thresholds, names=None, workers=1):
    """{threshold: summary} of the saved predictions of a split with the gate on `level` (volumes in `workers`
    processes: the matching runs on the CPU)."""
    from multiprocessing import Pool
    from score_hierarchy import summarise
    predictions, folder = Path(predictions), Path(data) / split
    levels = sorted((p.name for p in predictions.iterdir() if p.is_dir() and p.name not in ('semantic', 'parents')),
                    key=lambda v: v != level)                                   # inner level first
    pairs = [(levels[0], levels[1])] if len(levels) == 2 else []
    names = names or sorted(p.name for p in (predictions / level).iterdir())
    jobs = [(folder, predictions, name, level, levels, pairs, list(thresholds)) for name in names]
    if workers > 1:
        with Pool(workers) as pool:
            per_volume = pool.map(volume_records, jobs)
    else:
        per_volume = [volume_records(job) for job in jobs]
    return {t: summarise([records[k] for records in per_volume], levels, pairs) for k, t in enumerate(thresholds)}


def brief(summary, level):
    keys = [f'ap_{level}', f'ap50_{level}', f'recall50_{level}', f'precision50_{level}']
    keys += [k for k in summary if k.startswith('ap_') and k not in keys] + [k for k in summary if k.startswith('jtpr_')]
    return {k: summary[k] for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--level', required=True, help='level whose objects are gated')
    parser.add_argument('--select', required=True, help='PREDICTIONS:SPLIT of the selection volumes')
    parser.add_argument('--select-names', help='folder whose file names restrict the selection volumes')
    parser.add_argument('--apply', nargs='*', default=[], help='PREDICTIONS:SPLIT scored with the chosen threshold')
    parser.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS)
    parser.add_argument('--workers', type=int, default=6, help='processes (one volume each)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    folder, split = args.select.rsplit(':', 1)
    names = sorted(p.name for p in Path(args.select_names).iterdir()) if args.select_names else None
    sweep = score_folder(args.data, folder, split, args.level, args.thresholds, names, args.workers)
    best = max(args.thresholds, key=lambda t: (sweep[t][f'ap_{args.level}'], -t))
    for t in args.thresholds:
        print(f"select t={t:.2f}: " + ' '.join(f'{k} {v:.3f}' for k, v in brief(sweep[t], args.level).items() if v is not None))
    print(f'chosen threshold {best}')
    result = dict(level=args.level, select=args.select, select_volumes=len(names) if names else None, threshold=best,
                  sweep={str(t): brief(s, args.level) for t, s in sweep.items()}, applied={})
    for target in args.apply:
        folder, split = target.rsplit(':', 1)
        scored = score_folder(args.data, folder, split, args.level, sorted({0.0, best}), workers=args.workers)
        result['applied'][target] = {str(t): brief(s, args.level) for t, s in scored.items()}
        for t, s in scored.items():
            print(f"{target} t={t:.2f}: " + ' '.join(f'{k} {v:.3f}' for k, v in brief(s, args.level).items() if v is not None))
    Path(args.output).write_text(json.dumps(result, indent=1))
    print('->', args.output)


if __name__ == '__main__':
    main()
