"""Scores of a hierarchy model (any number of levels, model.levels) on the tiles of a split: AP of every level
(mean over IoU 0.5..0.95 of TP / (TP + FP + FN)) and, for every adjacent pair, the share of annotated
child -> parent pairs whose child and parent are both found at IoU 0.5 and linked to each other in the
prediction (JTPR of that pair). Runs cytonucs_hierarchy.predict_hierarchy with the pair calibration saved
next to the weights; also times every volume.

Usage:
  python score_hierarchy.py --config CONFIG --weights WEIGHTS [--split test] [--gpu 0] [--output FILE] [--resume]
Annotated pairs: <split>/assignments/<stem>.json for nucleus -> cytoplasm, assignments_<inner>_<outer>/ for
other pairs; missing files are derived by overlap. A region where a level is not annotated can be given as
<split>/<level>_unknown/<name> (non-zero = unknown): predicted objects of that level lying mostly inside it are
left out of the scores (e.g. cells whose nuclei the annotation lacks).
For every pair it also reports the metrics of the HydraStarDist paper (paper_<pair>: IoUR, AP and the joint TP
rates over children and over parents at tau 0.1..0.9, means over the images), to compare with its tables.
The output keeps one record per image (records: counts per tau, joint TP, paper metrics, seconds), so summaries,
bootstrap intervals and breakdowns are computed again without predicting; --predictions saves the predicted
labels as well. score_image / summarise / ground_truth / save_prediction are shared by the baseline scorers.
"""
import argparse
import contextlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PAPER_TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PAPER_KEYS = ('iour_inner', 'iour_outer', 'ap_inner', 'ap_outer', 'jtpr_inner', 'jtpr_outer')


def pairs_folder(inner, outer):
    return 'assignments' if (inner, outer) == ('nucleus', 'cytoplasm') else f'assignments_{inner}_{outer}'


def without_ignored(labels, ignore):
    """Labels without the objects that lie mostly (more than half of their voxels) in the ignore region."""
    labels = np.asarray(labels).copy()
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    inside = dict(zip(*np.unique(labels[(labels > 0) & ignore], return_counts=True)))
    drop = [int(i) for i, c in zip(ids, counts) if inside.get(i, 0) > c / 2]
    labels[np.isin(labels, drop)] = 0
    return labels


def paper_metrics(gt_inner, gt_outer, pred_inner, pred_outer, truth):
    """Metrics of the HydraStarDist paper (De et al. 2025, arXiv 2504.12078) for one image, at every tau of
    PAPER_TAUS: IoUR (sum of the IoU of the matched objects / ground-truth objects, their Eq. 8) and AP
    (TP / (TP + FP + FN), Eq. 9) of both levels, and the joint TP rate: ground-truth child -> parent pairs (truth)
    whose child and parent are both matched, over all ground-truth children (jtpr_inner, Eq. 12), and the parents
    matched together with at least one matched child over all ground-truth parents (jtpr_outer: Eq. 13 when every
    parent holds one child, at most 1 when it holds several); unlike our JTPR neither asks the matched predictions
    to be linked. None where the image has no ground-truth objects of that level (AP: no objects at all)."""
    from stardist.matching import matching
    out, matched = {}, {}
    for name, gt, pred in (('inner', gt_inner, pred_inner), ('outer', gt_outer, pred_outer)):
        results = matching(np.asarray(gt), np.asarray(pred), thresh=PAPER_TAUS, criterion='iou', report_matches=True)
        n_true = int(results[0].n_true)
        out['n_' + name] = n_true
        out['iour_' + name] = [float(r.mean_true_score) if n_true else None for r in results]
        out['ap_' + name] = [float(r.accuracy) if r.tp + r.fp + r.fn else None for r in results]
        matched[name] = [{r.matched_pairs[i][0] for i in r.matched_tps} for r in results]
    joint = [[(child, parent) for child, parent in truth.items() if child in inner and parent in outer]
             for inner, outer in zip(matched['inner'], matched['outer'])]
    out['jtpr_inner'] = [len(j) / out['n_inner'] if out['n_inner'] else None for j in joint]
    out['jtpr_outer'] = [len({parent for _, parent in j}) / out['n_outer'] if out['n_outer'] else None for j in joint]
    return out


def mean_paper_metrics(images):
    """Mean over the images (as the paper averages over its test images) of every metric of paper_metrics at
    every tau, and its best value over tau (the 'optimal tau' of the paper's tables)."""
    result = dict(taus=PAPER_TAUS, images=len(images))
    for key in PAPER_KEYS:
        per_tau = []
        for k in range(len(PAPER_TAUS)):
            values = [m[key][k] for m in images if m[key][k] is not None]
            per_tau.append(float(np.mean(values)) if values else None)
        result[key] = per_tau
        result[key + '_best'] = max((v for v in per_tau if v is not None), default=None)
    return result


def ground_truth(folder, name, levels, pairs):
    """(labels per level, child -> parent pairs per pair, ignore region per level or None) of one image of a split."""
    import tifffile
    from cytonucs_hierarchy import link_children
    gt = {level: tifffile.imread(folder / f'{level}_masks' / name) for level in levels}
    truths = {}
    for inner, outer in pairs:
        stored = folder / pairs_folder(inner, outer) / (Path(name).stem + '.json')
        truths[f'{inner}_{outer}'] = ({int(k): int(v) for k, v in json.loads(stored.read_text()).items()} if stored.is_file()
                                      else link_children(gt[inner], gt[outer]))
    ignore = {level: (tifffile.imread(folder / f'{level}_unknown' / name) > 0 if (folder / f'{level}_unknown' / name).is_file()
                      else None) for level in levels}
    return gt, truths, ignore


def score_image(gt, pred, pairs, truths, pred_parents=None, ignore=None, jtpr_ious=()):
    """Scores of one image (the record kept per image, so that any summary or interval is computed again
    without predicting): per level the objects and TP / FP / FN at every tau of TAUS; per adjacent pair our
    joint TP over the annotated pairs (predicted links pred_parents[inner] when given, else by overlap as
    compute_jtpr does) and the HydraStarDist-paper metrics. Predicted objects lying mostly in ignore[level]
    are dropped first; jtpr_ious: further IoU thresholds of the joint TP (record pairs[key]['joint_tp_at']);
    returns (record, the predictions as scored)."""
    import contextlib
    import io
    from stardist.matching import matching
    from cytonucs_inference import compute_jtpr
    pred = {level: (without_ignored(labels, ignore[level]) if ignore and ignore.get(level) is not None else np.asarray(labels))
            for level, labels in pred.items()}
    record = dict(levels={}, pairs={})
    for level, labels in gt.items():
        results = matching(labels, pred[level], thresh=TAUS, criterion='iou')
        record['levels'][level] = dict(objects=int(len(np.unique(labels)) - 1), predicted=int(len(np.unique(pred[level])) - 1),
                                       tp=[int(r.tp) for r in results], fp=[int(r.fp) for r in results],
                                       fn=[int(r.fn) for r in results])
    for inner, outer in pairs:
        key = f'{inner}_{outer}'
        with contextlib.redirect_stdout(io.StringIO()):
            _, details = compute_jtpr(pred[inner], pred[outer], gt[inner], gt[outer], iou_thresh=.5,
                                      assignments_pred=(pred_parents or {}).get(inner), assignments_gt=truths[key])
        record['pairs'][key] = dict(joint_tp=int(details['joint_tp']), pairs=int(details['total_pairs']),
                                    fail_inner=int(details['joint_fail_nucleus']), fail_outer=int(details['joint_fail_cell']),
                                    fail_link=int(details['joint_fail_link']),
                                    paper=paper_metrics(gt[inner], gt[outer], pred[inner], pred[outer], truths[key]))
        for iou in jtpr_ious:
            with contextlib.redirect_stdout(io.StringIO()):
                _, at = compute_jtpr(pred[inner], pred[outer], gt[inner], gt[outer], iou_thresh=float(iou),
                                     assignments_pred=(pred_parents or {}).get(inner), assignments_gt=truths[key])
            record['pairs'][key].setdefault('joint_tp_at', {})[str(float(iou))] = int(at['joint_tp'])
    return record, pred


def summarise(records, levels, pairs):
    """Summary of the per-image records of score_image: per level AP (mean over TAUS of TP / (TP + FP + FN) of
    the pooled counts), AP50, recall and precision at 0.5; per pair our JTPR, why the other annotated pairs fail
    (shares of the pairs: the inner object is not found at IoU 0.5, the inner one is but its outer one is not, both
    are but the predicted link joins other objects; compute_jtpr) and the means of the paper metrics."""
    from cytonucs_runtime import accuracy_profile
    result = dict(images=len(records), objects={level: sum(r['levels'][level]['objects'] for r in records) for level in levels})
    for level in levels:
        tp, fp, fn = (np.sum([r['levels'][level][k] for r in records], axis=0) if records else np.zeros(len(TAUS))
                      for k in ('tp', 'fp', 'fn'))
        ap = accuracy_profile(tp, fp, fn)
        result[f'ap_{level}'], result[f'ap50_{level}'] = float(np.mean(ap)), float(ap[0])
        result[f'recall50_{level}'] = float(tp[0] / max(tp[0] + fn[0], 1))
        result[f'precision50_{level}'] = float(tp[0] / max(tp[0] + fp[0], 1))
        result[f'f1_50_{level}'] = float(2 * tp[0] / max(2 * tp[0] + fp[0] + fn[0], 1))
    for inner, outer in pairs:
        key = f'{inner}_{outer}'
        joint, total = (sum(r['pairs'][key][k] for r in records) for k in ('joint_tp', 'pairs'))
        result[f'jtpr_{key}'] = joint / total if total else None
        result[f'pairs_{key}'] = total
        if records and all('fail_link' in r['pairs'][key] for r in records):        # older records lack them
            for kind in ('inner', 'outer', 'link'):
                result[f'fail_{kind}_{key}'] = sum(r['pairs'][key][f'fail_{kind}'] for r in records) / total if total else None
        result[f'paper_{key}'] = mean_paper_metrics([r['pairs'][key]['paper'] for r in records])
        for iou in (records[0]['pairs'][key].get('joint_tp_at', {}) if records else {}):   # JTPR at further IoUs
            if all(iou in r['pairs'][key].get('joint_tp_at', {}) for r in records):
                result[f'jtpr_iou{iou}_{key}'] = sum(r['pairs'][key]['joint_tp_at'][iou] for r in records) / total if total else None
    return result


def save_prediction(folder, name, labels):
    """Predicted labels of every level as <folder>/<level>/<name> (zlib), for scoring again without predicting."""
    import tifffile
    for level, values in labels.items():
        (Path(folder) / level).mkdir(parents=True, exist_ok=True)
        values = np.asarray(values)
        tifffile.imwrite(Path(folder) / level / name, values.astype(np.uint16 if values.max(initial=0) < 65536 else np.uint32),
                         photometric='minisblack', compression='zlib')


def score(config_path, weights, split='test', limit=0, memory_limit=None, predictions=None, resume=None):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        if memory_limit:                                         # a cap, e.g. for a test next to running jobs
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_limit))])
        else:
            tf.config.experimental.set_memory_growth(gpu, True)
    from cytonucs_config import CytoNucsConfig, read_config
    from cytonucs_model import build_cytonucs_model
    from cytonucs_hierarchy import predict_hierarchy, read_calibration
    from cytonucs_runtime import read_model_image

    raw = read_config(config_path, weights)                # 'auto' settings: calibrated for these weights
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(weights)
    folder = Path(raw['data']['dataset_dir']) / split
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='score', basedir=tmp)
            model.load_weights(str(weights))
        calibration = read_calibration(weights)
        inference = raw['inference']
        thresholds = dict(prob_thresh_nucleus=inference['prob_thresh_nucleus'],
                          prob_thresh_cytoplasm=inference['prob_thresh_cytoplasm'], nms_thresh=inference['nms_thresh'])
        levels, pairs = list(cfg.levels), list(cfg.level_pairs)
        records = []
        names = sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
        names = names[:limit] if limit else names
        done = {}
        if resume and Path(resume).exists():                     # records of an earlier, interrupted run
            for line in Path(resume).read_text().splitlines():
                record = json.loads(line)
                done[record['name']] = record
        for name in names:
            if name in done:
                records.append(done[name])
                continue
            image = read_model_image(folder / 'images' / name, int(cfg.ndim), int(cfg.n_image_channels))
            gt, truths, ignore = ground_truth(folder, name, levels, pairs)
            started = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                pred = predict_hierarchy(model, cfg, image, calibration=calibration, thresholds=thresholds)
            seconds = time.time() - started
            record, scored = score_image(gt, pred['levels'], pairs, truths, pred_parents=pred['parents'], ignore=ignore)
            records.append(dict(record, name=name, seconds=seconds))
            if resume:
                with open(resume, 'a') as f:
                    f.write(json.dumps(records[-1]) + '\n')
            if predictions:
                save_prediction(predictions, name, scored)
                if pred.get('parents') and pred['parents'].get(levels[0]) is not None:   # predicted links, for rescoring
                    (Path(predictions) / 'parents').mkdir(parents=True, exist_ok=True)
                    (Path(predictions) / 'parents' / (Path(name).stem + '.json')).write_text(
                        json.dumps({int(k): int(v) for k, v in pred['parents'][levels[0]].items()}))
                if pred.get('semantic'):                                 # per-object semantic scores (paper/semantic_gate.py)
                    (Path(predictions) / 'semantic').mkdir(parents=True, exist_ok=True)
                    (Path(predictions) / 'semantic' / (Path(name).stem + '.json')).write_text(json.dumps(pred['semantic']))
    result = dict(split=split, volumes=len(names), seconds_per_volume=float(np.mean([r['seconds'] for r in records])),
                  weights=str(weights), calibration=calibration, predictions=str(predictions) if predictions else None)
    result.update(summarise(records, levels, pairs))
    result['records'] = records
    print(f"{split}: " + ' | '.join(f"{level} {result['objects'][level]}: AP {result[f'ap_{level}']:.3f} (AP50 {result[f'ap50_{level}']:.3f})"
                                     for level in levels) + ' | ' +
          ' | '.join(f"JTPR {a}_{b} {result[f'jtpr_{a}_{b}'] or 0:.3f} of {result[f'pairs_{a}_{b}']}" for a, b in pairs) +
          f" | {result['seconds_per_volume']:.1f} s per volume")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--gpu')
    parser.add_argument('--output')
    parser.add_argument('--limit', type=int, default=0, help='score only this many volumes (smoke test)')
    parser.add_argument('--memory-limit', type=int, help='GPU memory cap in MB (default: grow as needed)')
    parser.add_argument('--predictions', help='folder for the predicted labels (<level>/<name>), to score again later')
    parser.add_argument('--resume', action='store_true',
                        help='keep each volume\'s record in OUTPUT.records.jsonl and skip the volumes already there '
                             '(run again after a failure, e.g. GPU memory of a long 3D run, in a fresh process)')
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    # cuDNN autotuning stays on (TF_CUDNN_USE_AUTOTUNE=0 is faster on many ROI shapes): without it a convolution has
    # no fallback that needs no scratch memory, and KiTS test volumes failed to allocate the scratch space.
    memory_limit = args.memory_limit or int(os.environ.get('GPU_MEMORY_LIMIT_MB', 0) or 0) or None   # the queue's cap
    output = Path(args.output or str(args.weights) + f'.hierarchy_{args.split}.json')
    resume = str(output) + '.records.jsonl' if args.resume else None
    result = score(args.config, args.weights, args.split, args.limit, memory_limit, args.predictions, resume)
    output.write_text(json.dumps(result, indent=1))
    print('->', output)


if __name__ == '__main__':
    main()
