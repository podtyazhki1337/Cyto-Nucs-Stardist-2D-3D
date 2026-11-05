"""Inference settings calibrated on validation at the end of training, as StarDist does for its thresholds.

Training and the in-training validation use the given values (AUTO_INFERENCE_DEFAULTS for 'auto'). Afterwards the
settings that are 'auto' are chosen on the part of the validation set that fits a time budget (5 minutes by default):

- prob_thresh_nucleus / prob_thresh_cytoplasm: which peaks of each head become prompts. Every volume is predicted once
  with the lowest threshold of evaluation.prob_thresh_range; a higher threshold re-assembles the final labels from the
  cached refinement candidates whose prompt peak reaches it (no new network pass). (The NMS threshold is not
  calibrated: with the peak proposals of the method the StarDist NMS does not run.)
- inner_min_inside_outer / outer_min_inner_share: the nesting filters, applied to the same re-assembled labels.

The objective is the checkpoint metric (JTPR weight of loss_params.checkpoint_w_jtpr, mean AP of the levels). A
coordinate search keeps the given value unless another one is better by more than MIN_GAIN. The chosen values go to
<weights>.inference.json, where cytonucs_config.resolve_auto_inference finds them for that checkpoint.
"""
import json
import time
from types import SimpleNamespace

import numpy as np

from cytonucs_config import calibrated_inference_path

TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MIN_GAIN = 1e-3
PROB_KEYS = ('prob_thresh_nucleus', 'prob_thresh_cytoplasm')
FILTER_KEYS = ('outer_min_inner_share', 'inner_min_inside_outer')
FILTER_SHARES = (0.0, 0.1, 0.25, 0.5, 0.75)


def match_counts(gt, pred):
    """(tp, fp, fn) per IoU threshold of TAUS, shape (len(TAUS), 3)."""
    from stardist.matching import matching
    result = matching(np.asarray(gt), np.asarray(pred), thresh=TAUS, criterion='iou')
    result = result if isinstance(result, (list, tuple)) else [result]
    return np.array([[r.tp, r.fp, r.fn] for r in result], np.int64)


def ap_of(counts):
    """Mean over TAUS of the StarDist accuracy TP / (TP + FP + FN) of summed match_counts."""
    from cytonucs_runtime import accuracy_profile
    return float(accuracy_profile(counts[:, 0], counts[:, 1], counts[:, 2]).mean())


def _scan(current, candidates, score, out_of_time):
    """Best of candidates by score(value), starting from current; a value replaces the best only if it is better by
    more than MIN_GAIN. Stops early when out_of_time()."""
    best, best_score = current, score(current)
    for value in candidates:
        if out_of_time():
            break
        if value == current:
            continue
        s = score(value)
        if s > best_score + MIN_GAIN:
            best, best_score = value, s
    return best, best_score


def calibrate_inference(keys, current, score, prob_range, budget, clock=time.time, start=None):
    """Values of the 'auto' keys (of PROB_KEYS + FILTER_KEYS) and a report. score(values) -> objective of a full set
    of values (current: the values used so far for every key); prob thresholds are searched over prob_range, the
    filters over FILTER_SHARES, one key after the other."""
    start = clock() if start is None else start
    out_of_time = lambda: clock() - start > budget
    cache = {}

    def objective(values):
        key = tuple(sorted(values.items()))
        if key not in cache:
            cache[key] = score(values)
        return cache[key]

    values = dict(current)
    before = objective(values)
    for key in [k for k in PROB_KEYS + FILTER_KEYS if k in keys]:
        candidates = prob_range if key in PROB_KEYS else FILTER_SHARES
        values[key], _ = _scan(values[key], candidates, lambda v, key=key: objective({**values, key: v}), out_of_time)
    report = dict(metric_before=before, metric=objective(values), evaluated=len(cache),
                  seconds=round(clock() - start, 1), budget=budget, out_of_time=bool(out_of_time()))
    return {k: float(values[k]) for k in keys}, report


def reassemble(config, volume, values, quality_gates=None):
    """Final labels and pairs of a predicted volume with other prob thresholds and nesting filters: the assembly of
    cytonucs_inference.predict_instances_prompted on the cached candidates whose prompt peak reaches the threshold of
    its head (window-suppressed residual prompts are kept)."""
    from cytonucs_inference import assemble_head, keep_outer_with_inner, keep_inner_inside_outer
    from cytonucs_postprocess import assign_final_masks, drop_small_instances
    sink, shape = volume['sink'], volume['shape']

    def keep(c):
        return c is not None and (c.get('residual', False) or
                                  c.get('detection_score', 1.0) >= values['prob_thresh_' + c.get('prompt_head', 'nucleus')])
    nuclei, _ = assemble_head(config, 'nucleus', [c for c in sink.get('nucleus', []) if keep(c)], shape)
    cells, _ = assemble_head(config, 'cytoplasm', [c for c in sink.get('cytoplasm', []) if keep(c)], shape)
    if config.mask_quality_min == 'auto' and quality_gates:
        nuclei, _ = drop_small_instances(nuclei, quality_gates.get('nucleus_min_area', 0))
        cells, _ = drop_small_instances(cells, quality_gates.get('cytoplasm_min_area', 0))
    shares = SimpleNamespace(inner_min_inside_outer=values['inner_min_inside_outer'],
                             outer_min_inner_share=values['outer_min_inner_share'])
    cells, _ = keep_outer_with_inner(shares, nuclei, cells)
    nuclei, _ = keep_inner_inside_outer(shares, nuclei, cells)
    evidence = [(n, c if keep(c) else None) for n, c in sink.get('evidence', []) if keep(n)]
    pairs = assign_final_masks(nuclei, cells, getattr(config, 'pair_min_overlap', 0.1), evidence,
                               getattr(config, 'pair_min_mask_iou', 0.25))
    return nuclei, cells, pairs


def checkpoint_score(config, volumes, quality_gates, jtpr_weight):
    """score for calibrate_inference: the checkpoint metric of the re-assembled labels of all volumes."""
    from cytonucs_inference import compute_jtpr
    levels = [h for h in ('nucleus', 'cytoplasm') if config.has_level(h)] if hasattr(config, 'has_level') else ['nucleus', 'cytoplasm']

    def score(values):
        counts = {h: np.zeros((len(TAUS), 3), np.int64) for h in levels}
        joint = pairs_total = 0
        for v in volumes:
            nuclei, cells, pairs = reassemble(config, v, values, quality_gates)
            for h, labels in (('nucleus', nuclei), ('cytoplasm', cells)):
                if h in counts:
                    counts[h] += match_counts(v['gt'][h], labels)
            if len(levels) == 2:
                _, details = compute_jtpr(nuclei, cells, v['gt']['nucleus'], v['gt']['cytoplasm'], iou_thresh=0.5,
                                          assignments_pred=pairs, assignments_gt=v.get('assignments'))
                joint += int(details['joint_tp']); pairs_total += int(details['total_pairs'])
        ap = sum(ap_of(c) for c in counts.values()) / 2
        return jtpr_weight * joint / max(pairs_total, 1) + (1 - jtpr_weight) * ap
    return score


def calibrate_after_training(predictor, samples, keys, budget=300.0, weights_path=None):
    """calibrate_inference on validation samples (dicts with normalised 'image', 'nucleus_mask', 'cytoplasm_mask',
    'assignments', 'filename') with a predictor of the trained model. Volumes are predicted one after another (with the
    lowest prob threshold searched) until 60 % of the budget is spent; the rest is for the search. Writes
    <weights_path>.inference.json (when given) and returns the chosen values and a report."""
    from cytonucs_trainer import DiceCallback
    from cytonucs_runtime import checkpoint_jtpr_weight
    from cytonucs_inference import CytoNucsPredictor
    config = predictor.config
    start = time.time()
    current = {k: float(getattr(predictor, k)) for k in PROB_KEYS}
    current.update({k: float(getattr(config, k)) for k in FILTER_KEYS})
    prob_range = sorted({float(v) for v in config.prob_thresh_range})
    saved = {k: getattr(predictor, k) for k in PROB_KEYS}
    for key in PROB_KEYS:                                   # predict once with the lowest threshold searched
        if key in keys:
            setattr(predictor, key, min(prob_range + [current[key]]))
    volumes = []
    try:
        for sample in samples:
            if volumes and time.time() - start > 0.6 * budget:
                break
            sink = {}
            predictor._refinement_candidate_sink = sink
            try:
                image = np.asarray(sample['image'], np.float32)
                DiceCallback._predict_full_volume(SimpleNamespace(model=predictor.model), predictor, image)
            finally:
                predictor._refinement_candidate_sink = None
            volumes.append(dict(name=sample.get('filename', str(len(volumes))), sink=sink, shape=image.shape[:int(config.ndim)],
                                gt={'nucleus': np.asarray(sample['nucleus_mask']), 'cytoplasm': np.asarray(sample['cytoplasm_mask'])},
                                assignments=sample.get('assignments')))
            print(f"  [inference calibration] predicted {volumes[-1]['name']} ({time.time() - start:.0f} s)")
    finally:
        for key, value in saved.items():
            setattr(predictor, key, value)
    gates = CytoNucsPredictor._active_quality_gates(predictor)
    score = checkpoint_score(config, volumes, gates, checkpoint_jtpr_weight(config))
    values, report = calibrate_inference(keys, current, score, prob_range, budget, start=start)
    report.update(volumes=len(volumes), volume_names=[v['name'] for v in volumes], before={k: current[k] for k in keys})
    print(f"  [inference calibration] {report['before']} -> {values} on {len(volumes)} volumes in {report['seconds']} s, "
          f"metric {report['metric_before']:.4f} -> {report['metric']:.4f}")
    if weights_path is not None:
        calibrated_inference_path(weights_path).write_text(json.dumps(dict(values=values, report=report), indent=2))
        print(f"  [inference calibration] saved to {calibrated_inference_path(weights_path)}")
    return values, report
