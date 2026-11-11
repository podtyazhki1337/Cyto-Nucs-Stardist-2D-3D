"""Scores of saved predicted labels of any method against a dataset split, with the scoring of score_hierarchy.py
(per image: TP / FP / FN per level at IoU 0.5..0.95, our JTPR, the HydraStarDist-paper metrics; summaries from
the per-image records). Predicted child -> parent links are taken by overlap (as compute_jtpr does without
predicted pairs) unless <predictions>/<split>/parents/<stem>.json holds them for the inner level.

Predicted labels: <predictions>/<split>/<level>/<name> (as written by the baseline scorers and
baselines/cellpose/run_cellpose.py) or <predictions>/<level>/<name> (score_hierarchy.py --predictions, which also writes
the predicted links in parents/). Predicted objects lying mostly in
<split>/<level>_unknown/<name> are left out. Seconds per image are read from --timing (a JSON of '<split>/<name>'
-> seconds, e.g. the 'seconds' of a Cellpose training.json) when given.

Usage:
  python score_predictions.py --data DIR --predictions DIR [--split test] [--levels nucleus cytoplasm]
      [--timing FILE] [--output FILE]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def score_predictions(data, predictions, split='test', levels=('nucleus', 'cytoplasm'), timing=None, jtpr_ious=()):
    import tifffile
    from score_hierarchy import ground_truth, score_image, summarise
    data, predictions, levels = Path(data), Path(predictions), list(levels)
    # <predictions>/<split>/<level>/<name>, or <predictions>/<level>/<name> (score_hierarchy.py --predictions)
    root = predictions / split if (predictions / split).is_dir() else predictions
    pairs = [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]
    seconds = json.loads(Path(timing).read_text()) if timing else {}
    seconds = seconds.get('seconds', seconds)
    records = []
    names = sorted(p.name for p in (data / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    for name in names:
        pred = {level: tifffile.imread(root / level / name) for level in levels}
        parents_file = root / 'parents' / (Path(name).stem + '.json')
        parents = ({levels[0]: {int(k): int(v) for k, v in json.loads(parents_file.read_text()).items()}}
                   if parents_file.is_file() else None)
        gt, truths, ignore = ground_truth(data / split, name, levels, pairs)
        record, _ = score_image(gt, pred, pairs, truths, pred_parents=parents, ignore=ignore, jtpr_ious=jtpr_ious)
        records.append(dict(record, name=name, seconds=seconds.get(f'{split}/{name}')))
    known = [r['seconds'] for r in records if r['seconds'] is not None]
    result = dict(split=split, data=str(data), predictions=str(predictions / split), volumes=len(records),
                  seconds_per_volume=float(np.mean(known)) if known else None)
    result.update(summarise(records, levels, pairs))
    result['records'] = records
    print(f"{split}: " + ' | '.join(f"{level} {result['objects'][level]}: AP {result[f'ap_{level}']:.3f} "
                                    f"(AP50 {result[f'ap50_{level}']:.3f})" for level in levels) + ' | ' +
          ' | '.join(f"JTPR {a}_{b} {result[f'jtpr_{a}_{b}'] or 0:.3f} of {result[f'pairs_{a}_{b}']}" for a, b in pairs))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--split', nargs='+', default=['test'])
    parser.add_argument('--levels', nargs='+', default=['nucleus', 'cytoplasm'], help='inner level first')
    parser.add_argument('--timing')
    parser.add_argument('--jtpr-ious', type=float, nargs='*', default=[], help='JTPR also at these IoU thresholds, e.g. 0.3 0.7')
    parser.add_argument('--output', help='default: <predictions>/scores.json')
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    results = {split: score_predictions(args.data, args.predictions, split, args.levels, args.timing, args.jtpr_ious)
               for split in args.split}
    output = Path(args.output or Path(args.predictions) / 'scores.json')
    previous = json.loads(output.read_text()) if output.is_file() else {}       # splits scored before stay
    output.write_text(json.dumps({**previous, **results}, indent=1))
    print('->', output)


if __name__ == '__main__':
    main()
