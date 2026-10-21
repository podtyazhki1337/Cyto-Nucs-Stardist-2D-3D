"""Reproducible training artifacts and complete console logs."""
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ACTIVE_RUN = None


def align_training_history(history, completed_epochs):
    """Align missing/partial history to checkpoint epoch without inventing values."""
    n = int(completed_epochs)
    if n < 0:
        raise ValueError('completed_epochs must be nonnegative')
    result = {}
    full_epochs = history.get('fullval_epoch', [])
    keep = [i for i, e in enumerate(full_epochs) if e is not None and e <= n]
    for key, values in history.items():
        if not isinstance(values, list):                 # run facts (e.g. trainable_parameters), not per epoch
            result[key] = values
        elif key.startswith('fullval_'):
            result[key] = [values[i] if i < len(values) else None for i in keep]
        else:
            result[key] = list(values[:n]) + [None] * max(0, n - len(values))
    for key in ('train_loss', 'val_loss', 'lr'):
        result.setdefault(key, [None] * n)
    return result


def checkpoint_jtpr_weight(config):
    """Weight of JTPR in the checkpoint metric: loss_params.checkpoint_w_jtpr; 0 for a run with
    one object level (model.levels), which has no nucleus-cell pairs."""
    levels = getattr(config, 'levels', ('nucleus', 'cytoplasm'))
    return float(getattr(config, 'checkpoint_w_jtpr', 0.7)) if len(levels) == 2 else 0.0


def deployed_metric(jtpr, total_pairs, ap_gated, pairs_lost, jtpr_weight, levels=2):
    """Checkpoint metric of what inference outputs: after the calibrated quality/size gates.

    ap_gated: {head: AP after that head's gate}; pairs_lost: joint pairs whose nucleus or
    cell mask the gates remove (a pair removed in both heads counts twice, so this is a
    lower bound on the gated JTPR). levels: object levels of the run; the AP term is their
    mean (the level a run does not have contributes 0).
    """
    lost = sum(pairs_lost.values()) / max(int(total_pairs), 1)
    return jtpr_weight * (jtpr - lost) + (1.0 - jtpr_weight) * sum(ap_gated.values()) / levels


def history_selection_metrics(history, jtpr_weight=0.7):
    """Per full validation: the deployed metric when gates were calibrated, else the raw one.

    Older epochs without 'fullval_metric_deployed' are recomputed from the logged gated AP
    and gate pair losses; the pair count is derived from the pair-failure counters.
    """
    raw = list(history.get('fullval_metric', []))
    stored = list(history.get('fullval_metric_deployed', []))
    out = []
    for i, value in enumerate(raw):
        get = lambda key: (history.get('fullval_' + key) or [None] * (i + 1))[i] \
            if i < len(history.get('fullval_' + key) or []) else None
        if i < len(stored) and stored[i] is not None:
            out.append(stored[i]); continue
        ap_g = {h: get('ap_' + h + '_quality_gated') for h in ('nucleus', 'cytoplasm')}
        jtpr = get('jtpr')
        if value is None or jtpr is None or all(v is None for v in ap_g.values()):
            out.append(value); continue
        total = get('total_pairs')
        if total is None:
            fails = [get(k) for k in ('joint_fail_nucleus', 'joint_fail_cell', 'joint_fail_link')]
            if any(f is None for f in fails) or jtpr >= 1:
                out.append(value); continue
            total = round(sum(fails) / (1.0 - jtpr))
        gated = {h: ap_g[h] if ap_g[h] is not None else get('ap_' + h) for h in ap_g}
        lost = {h: get(h + '_quality_gate_pairs_lost') or 0 for h in ap_g}
        out.append(deployed_metric(jtpr, total, gated, lost, jtpr_weight))
    return out


def restored_best_metric(history, checkpoint_best, selection_start_epoch=0, jtpr_weight=0.7):
    """Use both available sources; partial history must not lower saved best."""
    import math
    metrics = history_selection_metrics(history, jtpr_weight)
    if selection_start_epoch:
        epochs = history.get('fullval_epoch', [])
        metrics = [v for i,v in enumerate(metrics) if i < len(epochs)
                   and epochs[i] is not None and epochs[i] > selection_start_epoch]
    values = [checkpoint_best] + metrics
    return max((float(v) for v in values if v is not None and math.isfinite(float(v))),
               default=float('-inf'))


def percentile_range(x, axis=None, pmin=1.0, pmax=99.8):
    """(low, high) percentiles of x over axis (keepdims) for the percentile normalisation. A channel whose high
    percentile equals its low one while brighter pixels exist (nearly empty: a few hundred pixels above the
    background, e.g. 7 HeLaCytoNuc nucleus channels) takes its maximum as high: dividing by ~0 would blow those
    pixels up to ~1e20. Every other channel gets exactly csbdeep's percentiles."""
    import numpy as np
    x = np.asarray(x, np.float32)
    low = np.percentile(x, pmin, axis=axis, keepdims=True)
    high = np.percentile(x, pmax, axis=axis, keepdims=True)
    top = np.max(x, axis=axis, keepdims=True)
    return low, np.where((high - low < 1e-6) & (top > low), top, high)


def normalize_percentiles(x, pmin=1.0, pmax=99.8, axis=None):
    """csbdeep.utils.normalize(x, pmin, pmax, axis) with the guard of percentile_range for nearly empty channels."""
    import numpy as np
    low, high = percentile_range(x, axis, pmin, pmax)
    return ((np.asarray(x, np.float32) - low) / (high - low + 1e-20)).astype(np.float32)


def read_model_image(path, ndim, channels):
    """Read TIFF axes before falling back to unambiguous channel placement."""
    import numpy as np
    axes = None
    if Path(path).suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray()
            axes = tif.series[0].axes
    else:
        import imageio.v2 as imageio
        arr = imageio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == ndim:
        wanted = 'ZYX' if ndim == 3 else 'YX'
        if axes and len(axes) == ndim and set(axes) == set(wanted):
            arr = np.transpose(arr, [axes.index(a) for a in wanted])
        return arr
    if arr.ndim != ndim + 1:
        raise ValueError(f'Image {path}: expected {ndim} spatial axes and optional channels, got {arr.shape}')
    if axes and len(axes) == arr.ndim:
        channel_axes = [i for i,a in enumerate(axes) if a in ('C','S') and arr.shape[i] == channels]
        if len(channel_axes) == 1:
            arr = np.moveaxis(arr, channel_axes[0], -1)
            remaining = [a for i,a in enumerate(axes) if i != channel_axes[0]]
            wanted = list('ZYX' if ndim == 3 else 'YX')
            if set(remaining) == set(wanted):
                arr = np.transpose(arr, [remaining.index(a) for a in wanted] + [ndim])
            return arr
    if arr.shape[-1] == channels:
        return arr
    if ndim == 3 and arr.shape[1] == channels:
        return np.moveaxis(arr, 1, -1)
    raise ValueError(f'Cannot determine channels for {path}: axes={axes}, shape={arr.shape}, channels={channels}')


def write_instance_mask(path, labels):
    """Lossless instance IDs; uint16 when possible, uint32 for larger volumes."""
    import numpy as np
    import tifffile
    labels = np.asarray(labels)
    if labels.ndim not in (2,3) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Instance masks must be 2D/3D integer labels')
    if labels.size and (labels.min() < 0 or labels.max() > np.iinfo(np.uint32).max):
        raise ValueError('Instance mask IDs must fit nonnegative uint32')
    dtype = np.uint16 if not labels.size or labels.max() <= 65535 else np.uint32
    tifffile.imwrite(path, labels.astype(dtype), photometric='minisblack',
                     metadata={'axes': 'ZYX' if labels.ndim == 3 else 'YX'})


def aggregate_joint_counts(counts):
    """Global pair recall; do not weight tiny and dense volumes equally."""
    tp = sum(int(c['joint_tp']) for c in counts)
    total = sum(int(c['total_pairs']) for c in counts)
    return dict(jtpr_global=tp / total if total else 0., joint_tp=tp, total_pairs=total)


def resolve_resume_epoch(weights_path, restored_epoch=None):
    """State is authoritative; weights-only resume needs an explicit epoch filename."""
    import re
    if restored_epoch is not None:
        epoch = int(restored_epoch)
        if epoch < 0 or epoch != restored_epoch:
            raise ValueError('Checkpoint epoch must be a nonnegative integer')
        return epoch
    if Path(weights_path).is_file():
        import h5py
        with h5py.File(weights_path, 'r') as weights:
            value = weights.attrs.get('cytonucs_epoch')
        if value is not None:
            epoch = int(value)
            if epoch < 0 or epoch != value:
                raise ValueError('Invalid epoch metadata in weights file')
            return epoch
    match = re.search(r'(?:^|_)epoch_(\d+)(?:[_.]|$)', Path(weights_path).name)
    if match is None:
        raise ValueError(
            f'Cannot resume {weights_path}: epoch unknown without .state.* files. '
            'Use the complete checkpoint or an epoch-numbered weights file; '
            'history.json is not evidence of the selected checkpoint epoch.')
    return int(match.group(1))


def accuracy_profile(tp, fp, fn):
    """StarDist accuracy TP/(TP+FP+FN) per IoU threshold, as used for AP."""
    import numpy as np
    tp, fp, fn = (np.asarray(v, np.float64) for v in (tp, fp, fn))
    return tp / np.maximum(tp + fp + fn, 1.0)


def edge_fragment_ignored_ap(masks, fragments, gate, min_area, taus):
    """AP after the calibrated gates with patch-edge fragments as ignore regions (reporting only).

    Fragments are GT objects touching an XY face of the patch and smaller than the calibrated size
    gate: pieces of objects cut by the patch, which the deployed pipeline never outputs. They are
    not FN, and a written mask whose best GT is a fragment is neither TP nor FP (COCO-style
    ignore regions). masks: [(quality, IoU with best GT, area, best GT is a fragment)];
    fragments: [is fragment] for every GT. Written masks are disjoint, so IoU >= tau >= 0.5 is the
    unique match, as in calibrate_quality_gates.
    """
    import numpy as np
    taus = np.asarray(taus, np.float64)
    rows = np.array([(q if q is not None else np.inf, iou, area, frag) for q, iou, area, frag in masks],
                    np.float64).reshape(-1, 4)
    keep = (rows[:, 0] >= gate) & (rows[:, 2] >= min_area) & (rows[:, 3] == 0)
    iou = rows[keep, 1]
    tp = np.array([(iou >= t).sum() for t in taus], np.float64)
    fp = len(iou) - tp
    n_gt = len(fragments) - int(np.sum(fragments))
    acc = accuracy_profile(tp, fp, n_gt - tp)
    return dict(ap=float(acc.mean()), recall50=float(tp[0] / max(n_gt, 1)), tp=int(tp[0]), fp=int(fp[0]),
                fn=int(n_gt - tp[0]), ignored_gt=int(np.sum(fragments)),
                ignored_masks=int(((rows[:, 0] >= gate) & (rows[:, 2] >= min_area) & (rows[:, 3] == 1)).sum()))


def boundary_error_ap(masks, counts, gate, min_area, taus):
    """AP after the calibrated gates with one part of the boundary error removed (reporting only).

    masks: [(quality, IoU with best GT, area, split)] for the written masks, split from
    RefinementStageAudit.boundary_split for masks matched at IoU >= 0.5 (None otherwise);
    counts: dict(tp=, fp=, fn=) per IoU threshold of the ungated masks, as for
    calibrate_quality_gates, so 'ap' equals its gated AP. Only matched masks change IoU,
    FP and FN at IoU 0.5 are unchanged: ap_z_exact / ap_xy_exact / ap_mixed_exact say how
    much AP each part (whole Z slices / XY projection contour / rims) costs among the
    objects already found.
    """
    import numpy as np
    taus = np.asarray(taus, np.float64)
    c = {k: np.asarray(v, np.float64) for k, v in counts.items()}
    quality = np.array([np.inf if m[0] is None else m[0] for m in masks], np.float64)
    iou = np.array([m[1] for m in masks], np.float64)
    area = np.array([m[2] for m in masks], np.float64)
    keep = (quality >= gate) & (area >= min_area)
    hit = lambda values: (np.asarray(values, np.float64)[:, None] >= taus[None]).sum(0)
    removed = hit(iou[~keep]) if (~keep).any() else np.zeros(len(taus))
    tp, fp, fn = c['tp'] - removed, c['fp'] - ((~keep).sum() - removed), c['fn'] + removed
    kept_split = [(i, m[3]) for i, m in enumerate(masks) if keep[i] and m[3] is not None]
    result = dict(ap=float(accuracy_profile(tp, fp, fn).mean()), matched_masks=len(kept_split))
    for part in ('z', 'xy', 'mixed'):
        name = 'iou_' + part + '_exact'
        gain = (hit([s[name] for _, s in kept_split]) - hit([iou[i] for i, _ in kept_split])
                if kept_split else np.zeros(len(taus)))
        result['ap_' + part + '_exact'] = float(accuracy_profile(tp + gain, fp - gain, fn - gain).mean())
    if kept_split:
        splits = [s for _, s in kept_split]
        errors = max(sum(s['error_voxels'] for s in splits), 1)
        result.update(
            z_extent_exact_fraction=float(np.mean([s['extra_slices'] + s['missing_slices'] == 0 for s in splits])),
            z_extent_one_slice_fraction=float(np.mean([s['extra_slices'] + s['missing_slices'] == 1 for s in splits])),
            **{part + '_error_voxel_share': float(sum(s[part + '_error_voxels'] for s in splits) / errors)
               for part in ('z', 'xy', 'mixed')})
    return result


def checkpoint_objective_gain(jtpr_delta, ap_delta, jtpr_weight):
    """Change of the checkpoint metric w*JTPR + (1-w)*mean(AP_nucleus, AP_cytoplasm) for one head."""
    return jtpr_weight * jtpr_delta + (1.0 - jtpr_weight) * 0.5 * ap_delta


def calibrate_quality_gates(decisions, counts, taus, total_pairs, jtpr_weight):
    """Per-head quality gate (and optional size gate) maximizing the checkpoint metric.

    decisions: {head: [(predicted quality, best written GT IoU, joint pairs lost if removed[, area])]}
    for the written masks; counts: {head: dict(tp=, fp=, fn=)} per IoU threshold in taus.
    Removing masks changes TP/FP/FN exactly (written masks are disjoint, so IoU >= 0.5
    identifies the unique match) and loses the listed joint pairs. With areas, a mask is
    kept only if quality >= gate and area >= min_area; both are searched jointly and the
    size gate is reported as '<head>_min_area' only when it is used (> 0).
    Returns gates and per-head statistics; gate 0 keeps everything.
    """
    import numpy as np
    taus = np.asarray(taus, np.float64)
    gates, stats = {}, {}
    for head, rows in decisions.items():
        c = counts[head]
        base = accuracy_profile(c['tp'], c['fp'], c['fn'])
        best = dict(gate=0.0, min_area=0, gain=0.0, ap=float(base.mean()), fp_removed=0, tp_removed=0, pairs_lost=0)
        if rows:
            rows = sorted(rows, key=lambda r: r[0])
            q = np.array([r[0] for r in rows]); iou = np.array([r[1] for r in rows])
            lost_all = np.array([r[2] for r in rows], np.float64)
            area = np.array([r[3] if len(r) > 3 else 0 for r in rows], np.float64)
            hit = (iou[:, None] >= taus[None]).astype(np.int64)
            # Size cutoffs: every area of an unmatched mask (+1 so that mask is removed).
            cutoffs = [0.0] + sorted(set(float(a) + 1 for a in area[iou < taus[0]] if a > 0))
            for cutoff in cutoffs:
                forced = area < cutoff
                m0 = hit[forced].sum(0); u0 = int(forced.sum()) - m0; l0 = lost_all[forced].sum()
                rest = np.flatnonzero(~forced)                        # sorted by quality
                matched = np.vstack([np.zeros((1, len(taus)), np.int64), np.cumsum(hit[rest], axis=0)])
                unmatched = np.arange(len(rest) + 1)[:, None] - matched
                lost = np.r_[0., np.cumsum(lost_all[rest])]
                mk = m0[None] + matched; uk = u0[None] + unmatched     # removal of rest[:k], k = 0..n
                ap = accuracy_profile(c['tp'] - mk, c['fp'] - uk, c['fn'] + mk).mean(axis=1)
                gain = checkpoint_objective_gain(-(l0 + lost) / max(total_pairs, 1), ap - base.mean(), jtpr_weight)
                valid = np.ones(len(rest) + 1, bool)
                qr = q[rest]
                valid[1:len(rest)] = qr[1:] != qr[:-1]                 # a gate cannot split ties
                if cutoff == 0:
                    valid[0] = False
                gain = np.where(valid, gain, -np.inf)
                k = int(np.argmax(gain))
                if gain[k] > best['gain'] + 1e-12:
                    if k == 0:
                        gate = 0.0
                    elif k < len(rest):
                        gate = float(qr[k])
                    else:
                        gate = float(np.nextafter(qr[k - 1], np.inf))
                    best = dict(gate=gate, min_area=int(cutoff), gain=float(gain[k]), ap=float(ap[k]),
                                fp_removed=int(uk[k][0]), tp_removed=int(mk[k][0]), pairs_lost=int(l0 + lost[k]))
        gates[head] = best['gate']
        if best['min_area'] > 0:
            gates[head + '_min_area'] = best['min_area']
        stats[head] = dict(best, ap_ungated=float(base.mean()))
    return gates, stats


def select_mask_thresholds(base, candidates, total_pairs, jtpr_weight):
    """Per-head mask threshold maximizing the checkpoint metric on validation.

    base: {head: dict(threshold=, tp=, fp=, fn=, joint_tp=)} at the default threshold;
    candidates: {head: {threshold: dict(tp=, fp=, fn=, joint_tp=)}} evaluated on the same
    cached candidates with only that head changed. Keeps the default unless the metric improves.
    """
    chosen, stats = {}, {}
    for head, b in base.items():
        ap0 = accuracy_profile(b['tp'], b['fp'], b['fn']).mean()
        best = dict(threshold=float(b['threshold']), gain=0.0, ap=float(ap0),
                    jtpr=b['joint_tp'] / max(total_pairs, 1))
        for threshold, c in sorted(candidates.get(head, {}).items()):
            ap = accuracy_profile(c['tp'], c['fp'], c['fn']).mean()
            gain = checkpoint_objective_gain((c['joint_tp'] - b['joint_tp']) / max(total_pairs, 1), ap - ap0, jtpr_weight)
            if gain > best['gain'] + 1e-12:
                best = dict(threshold=float(threshold), gain=float(gain), ap=float(ap),
                            jtpr=c['joint_tp'] / max(total_pairs, 1))
        chosen[head] = best['threshold']
        stats[head] = dict(best, ap_default=float(ap0), jtpr_default=b['joint_tp'] / max(total_pairs, 1))
    return chosen, stats


def _write_head_values(path, attribute, values):
    import h5py
    with h5py.File(path, 'a') as weights:
        weights.attrs[attribute] = json.dumps(values, sort_keys=True)


def _read_head_values(path, attribute):
    if not Path(path).is_file():
        return None
    import h5py
    try:
        with h5py.File(path, 'r') as weights:
            value = weights.attrs.get(attribute)
    except OSError:
        return None
    if value is None:
        return None
    values = json.loads(value.decode() if isinstance(value, bytes) else value)
    return {k: float(v) for k, v in values.items()
            if k in ('nucleus', 'cytoplasm') or k in ('nucleus_min_area', 'cytoplasm_min_area')}


def write_quality_gates(path, gates):
    _write_head_values(path, 'cytonucs_quality_gates', gates)


def read_quality_gates(path):
    """Calibrated gates stored with a checkpoint, or None (older checkpoints)."""
    return _read_head_values(path, 'cytonucs_quality_gates')


def write_mask_thresholds(path, thresholds):
    _write_head_values(path, 'cytonucs_mask_thresholds', thresholds)


def read_mask_thresholds(path):
    """Calibrated per-head mask thresholds stored with a checkpoint, or None."""
    return _read_head_values(path, 'cytonucs_mask_thresholds')


def sanitized(value):
    if isinstance(value, dict):
        return {k: ('<redacted>' if any(s in k.lower() for s in ('api_key', 'password', 'token', 'secret'))
                    else sanitized(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitized(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_effective_config(model, train_generator=None, val_generator=None):
    target = Path(model.basedir) / model.name
    config = sanitized({k: v for k, v in vars(model.config).items() if not k.startswith('rays')})
    config['rays'] = {h: {'vertices': getattr(model.config, 'rays_' + h).vertices.tolist(),
                         'anisotropy': list(getattr(model.config, 'rays_anisotropy_' + h))}
                      for h in ('nucleus', 'cytoplasm')}
    config['architecture_plan'] = model._architecture_plan
    config['split_files'] = {name: [str(p) for p in gen.image_files] if gen is not None else []
                             for name, gen in (('train', train_generator), ('val', val_generator))}
    for folder in (target, ACTIVE_RUN):
        if folder is not None:
            Path(folder).mkdir(parents=True, exist_ok=True)
            (Path(folder) / 'effective_config.json').write_text(
                json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')


def run_logged(config_path, main):
    global ACTIVE_RUN
    requested = json.loads(Path(config_path).read_text(encoding='utf-8'))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    ACTIVE_RUN = Path(requested['output']['basedir']) / requested['experiment']['name'] / 'runs' / stamp
    ACTIVE_RUN.mkdir(parents=True, exist_ok=False)
    sources = ACTIVE_RUN / 'source'; sources.mkdir()
    hashes = {}
    for path in Path(__file__).parent.glob('*.py'):
        data = path.read_bytes(); hashes[path.name] = hashlib.sha256(data).hexdigest()
        (sources / path.name).write_bytes(data)
    versions = {}
    for package in ('tensorflow', 'keras', 'stardist', 'csbdeep', 'numpy', 'scipy', 'scikit-image'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = dict(started_utc=stamp, python=sys.version, packages=versions,
                    requested_config=sanitized(requested), source_sha256=hashes)
    manifest = ACTIVE_RUN / 'run.json'
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    class Tee:
        def __init__(self, stream, log): self.stream, self.log = stream, log
        # Handlers created during a run (e.g. TensorFlow logging) may outlive it.
        def write(self, text):
            self.stream.write(text)
            if not self.log.closed:
                self.log.write(text); self.log.flush()
        def flush(self):
            self.stream.flush()
            if not self.log.closed:
                self.log.flush()
        def isatty(self): return False
    with (ACTIVE_RUN / 'console.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f'Run artifacts: {ACTIVE_RUN}')
            try:
                result = main(config_path)
                metadata['status'] = 'complete'
                return result
            except BaseException:
                import traceback
                traceback.print_exc()
                metadata['status'] = 'failed_or_interrupted'
                raise
            finally:
                metadata['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
