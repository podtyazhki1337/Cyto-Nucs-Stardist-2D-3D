"""Two separate single-level networks (one trained on the inner level only, one on the outer level only) at the
operating point of CytoNucs' calibration, for the comparison of joint and separate training.
Weights and checkpoints stay; the inference settings of both networks are searched jointly on val with the objective of
CytoNucs' calibration,
1/2 linked JTPR at IoU 0.5 (predicted links by overlap, as the scorers pair them) + 1/2 mean AP of the two levels.
Training calibrates each such network for the AP of its own level only.

Settings of each network (its own level only):
  mask   mask threshold (the candidates of CytoNucs' calibration, 0.25 .. 0.65, and the default 0.45),
  gate   quality gate on the predicted mask quality (a candidate is kept if quality >= gate),
  area   minimum object size after assembly (the size gate),
  prob   detection threshold of the prompts, with --search-prob (where the joint model's configuration calibrates
         it too; otherwise the configuration's value is kept).

  cache:   (GPU) val of one network once, ungated, at the default mask threshold and (with --search-prob) the lowest
           prob threshold of the range; every refinement candidate is saved (it can be re-thresholded, gated and
           re-assembled without a new network pass, as cytonucs_calibration and calibrate_mask_thresholds do).
  search:  (CPU) coordinate search from the current settings of both networks over the grids below until no key
           gains more than MIN_GAIN; writes chosen.json with the objective of every value tried.
  configs: scoring configs of both networks with the chosen settings (<name>.pairrecal.json) and with the current
           ones written out (<name>.pairown.json), for score_hierarchy.py.
The labels re-assembled from ungated candidates are an approximation where a gate removes a first-round candidate
and skip_covered_prompts would then run a second-round prompt (HeLa), so the chosen settings are checked with the
real inference on val (pairown vs pairrecal: score_hierarchy.py, then score_predictions.py) before the test splits.

Usage (from the repository root):
  python paper/recal_single_level_pairs.py cache --config CFG --weights W --out DIR [--gpu 0] [--search-prob] [--limit N]
  python paper/recal_single_level_pairs.py search --data DATA --inner DIR --outer DIR --out chosen.json [--workers 8] [--limit N]
  python paper/recal_single_level_pairs.py configs --chosen chosen.json --inner-config CFG --outer-config CFG --out-dir DIR
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

MASK_GRID = (0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65)
GATE_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
AREA_GRID = (0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200)
JTPR_WEIGHT = 0.5
MIN_GAIN = 1e-4
MAX_ROUNDS = 5
KEPT = ('detection_score', 'prompt_head', 'residual', 'predicted_quality')


def network_config(path, weights=None):
    """(raw config with the 'auto' settings of the weights resolved, CytoNucsConfig, level, role)."""
    import contextlib
    import io
    import tempfile
    from cytonucs_config import CytoNucsConfig, read_config
    raw = read_config(path, weights)
    raw['mode'] = 'inference'
    if weights:
        raw['inference']['checkpoint_path'] = str(weights)
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        p = Path(tmp) / 'config.json'
        p.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(p))
    levels = list(cfg.levels)
    if len(levels) != 1:
        raise SystemExit(f'{path}: a single-level network is expected, got levels {levels}')
    role = next(r for r, level in cfg.level_roles.items() if level == levels[0])     # level_roles: role -> level
    return raw, cfg, levels[0], role


def own_settings(raw, role, weights):
    """The settings the network is scored with now: its config resolved for the weights, the mask threshold and
    the quality / size gates stored with the checkpoint (or the config's quality_gates_override)."""
    from cytonucs_runtime import read_mask_thresholds, read_quality_gates
    inference = raw['inference']
    mask = inference.get(f'mask_thresh_{role}')
    if mask == 'auto':
        mask = (read_mask_thresholds(weights) or {}).get(role, inference.get('mask_thresh', 0.45))
    gates = dict(read_quality_gates(weights) or {}, **(inference.get('quality_gates_override') or {}))
    if inference.get('mask_quality_min') != 'auto':
        gates = {role: float(inference.get('mask_quality_min') or 0.0)}
    return dict(prob=float(inference[f'prob_thresh_{role}']), mask=float(mask if mask is not None else 0.45),
                gate=float(gates.get(role, 0.0)), area=int(gates.get(f'{role}_min_area', 0)))


def cache(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import contextlib
    import io
    import tempfile
    from types import SimpleNamespace
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    from cytonucs_config import CytoNucsConfig
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_model import build_cytonucs_model
    from cytonucs_runtime import normalize_percentiles, read_model_image
    from cytonucs_trainer import DiceCallback
    raw, _, level, role = network_config(args.config, args.weights)
    own = own_settings(raw, role, args.weights)
    inference = raw['inference']
    prob = dict(prob_thresh_nucleus=float(inference['prob_thresh_nucleus']),
                prob_thresh_cytoplasm=float(inference['prob_thresh_cytoplasm']))
    prob_range = sorted({float(v) for v in raw['evaluation']['prob_thresh_range']})
    if args.search_prob:                                   # predict once with the lowest prob threshold searched
        prob[f'prob_thresh_{role}'] = min(prob_range + [own['prob']])
    out = Path(args.out)
    (out / 'val').mkdir(parents=True, exist_ok=True)
    folder = Path(raw['data']['dataset_dir']) / 'val'
    names = sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    names = names[:args.limit] if args.limit else names
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='recal', basedir=tmp)
            model.load_weights(str(args.weights))
        predictor = CytoNucsPredictor(model=model, config=cfg, nms_thresh=inference['nms_thresh'], **prob)
        predictor.calibrating = True                       # ungated candidates at the default mask threshold
        nd = int(cfg.ndim)
        started = time.time()
        for name in names:
            target = out / 'val' / (Path(name).stem + '.pkl')
            if target.is_file():
                continue
            image = np.asarray(read_model_image(folder / 'images' / name, nd, int(cfg.n_image_channels)), np.float32)
            if image.ndim == nd:
                image = image[..., None]
            image = normalize_percentiles(image, 1, 99.8, axis=tuple(range(nd)))      # as predict_hierarchy
            sink = {}
            predictor._refinement_candidate_sink = sink
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    inner, outer, _ = DiceCallback._predict_full_volume(SimpleNamespace(model=model), predictor, image)
            finally:
                predictor._refinement_candidate_sink = None
            candidates = [{k: v for k, v in c.items() if k != 'roi'} for c in sink.get(role, []) if c is not None]
            labels = np.asarray(inner if role == 'nucleus' else outer)
            with open(target, 'wb') as f:
                pickle.dump(dict(name=name, shape=tuple(int(s) for s in image.shape[:nd]), candidates=candidates,
                                 ungated=labels.astype(np.int32)), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'cache {name}: {len(candidates)} candidates ({time.time() - started:.0f} s)', flush=True)
    (out / 'network.json').write_text(json.dumps(dict(
        config=str(args.config), weights=str(args.weights), level=level, role=role, own=own,
        predicted_prob=prob[f'prob_thresh_{role}'], search_prob=bool(args.search_prob), prob_range=prob_range,
        mask_default=float(inference.get('mask_thresh', 0.45)), images=names), indent=1))
    print('->', out)


# ---------------------------------------------------------------- search (CPU; workers inherit the state by fork)
STATE = {}


def labels_of(net, index, setting):
    """Labels of one network on one val image at `setting`: candidates re-thresholded, gated, assembled, size-gated."""
    from cytonucs_inference import assemble_head
    from cytonucs_postprocess import drop_small_instances, rethreshold_candidate
    network = STATE['nets'][net]
    item = network['images'][index]
    kept = []
    for c in item['candidates']:
        if not c.get('residual', False) and c.get('detection_score', 1.0) < setting['prob']:
            continue
        if abs(setting['mask'] - float(c.get('threshold', network['mask_default']))) > 1e-9:
            d = rethreshold_candidate(c, setting['mask'])
            if d is None:
                continue
            for key in KEPT:
                if key in c:
                    d[key] = c[key]
            c = d
        if 'predicted_quality' in c and c['predicted_quality'] < setting['gate']:
            continue
        kept.append(c)
    labels, _ = assemble_head(network['cfg'], network['role'], kept, item['shape'])
    labels, _ = drop_small_instances(np.asarray(labels), int(setting['area']))
    ignore = STATE['gt'][index][2].get(network['level'])
    if ignore is not None:
        from score_hierarchy import without_ignored
        labels = without_ignored(labels, ignore)
    return labels


def scan_image(task):
    """One val image: counts of the varied network at every candidate setting and its joint TP with the other
    network at its current setting; counts of the other network."""
    import contextlib
    import io
    from cytonucs_calibration import match_counts
    from cytonucs_inference import compute_jtpr
    index, varied, settings, fixed = task
    other = 'outer' if varied == 'inner' else 'inner'
    gt, truths, _ = STATE['gt'][index]
    li, lo = STATE['nets']['inner']['level'], STATE['nets']['outer']['level']
    fixed_labels = labels_of(other, index, fixed)
    out = dict(fixed=match_counts(gt[STATE['nets'][other]['level']], fixed_labels), varied=[], joint=[], total=0)
    for setting in settings:
        labels = labels_of(varied, index, setting)
        out['varied'].append(match_counts(gt[STATE['nets'][varied]['level']], labels))
        inner, outer = (labels, fixed_labels) if varied == 'inner' else (fixed_labels, labels)
        with contextlib.redirect_stdout(io.StringIO()):
            _, details = compute_jtpr(inner, outer, gt[li], gt[lo], iou_thresh=0.5, assignments_gt=truths[f'{li}_{lo}'])
        out['joint'].append(int(details['joint_tp']))
        out['total'] = int(details['total_pairs'])
    return out


def evaluate(pool, varied, settings, fixed):
    """Objective (and its parts) of every setting of the varied network with the other one at `fixed`."""
    from cytonucs_calibration import ap_of
    n = len(STATE['gt'])
    results = pool.map(scan_image, [(i, varied, settings, fixed) for i in range(n)], chunksize=1)
    fixed_ap = ap_of(sum(r['fixed'] for r in results))
    total = sum(r['total'] for r in results)
    rows = []
    for k in range(len(settings)):
        ap = ap_of(sum(r['varied'][k] for r in results))
        jtpr = sum(r['joint'][k] for r in results) / max(total, 1)
        ap_inner, ap_outer = (ap, fixed_ap) if varied == 'inner' else (fixed_ap, ap)
        rows.append(dict(objective=JTPR_WEIGHT * jtpr + (1 - JTPR_WEIGHT) * (ap_inner + ap_outer) / 2, jtpr=jtpr,
                         ap_inner=ap_inner, ap_outer=ap_outer))
    return rows


def search(args):
    import multiprocessing
    from score_hierarchy import ground_truth
    nets = {}
    for key, folder in (('inner', args.inner), ('outer', args.outer)):
        info = json.loads((Path(folder) / 'network.json').read_text())
        _, cfg, level, role = network_config(info['config'])
        names = info['images'][:args.limit] if args.limit else info['images']
        images = []
        for name in names:
            with open(Path(folder) / 'val' / (Path(name).stem + '.pkl'), 'rb') as f:
                images.append(pickle.load(f))
        grids = dict(mask=sorted(set(MASK_GRID) | {info['own']['mask']}),
                     gate=sorted(set(GATE_GRID) | {info['own']['gate']}),
                     area=sorted(set(AREA_GRID) | {info['own']['area']}),
                     prob=(sorted(set(info['prob_range']) | {info['own']['prob']}) if info['search_prob']
                           else [info['own']['prob']]))
        nets[key] = dict(info=info, cfg=cfg, level=level, role=role, images=images, names=names, grids=grids,
                         mask_default=info['mask_default'])
    if nets['inner']['names'] != nets['outer']['names']:
        raise SystemExit('the two caches hold different val images')
    data = Path(args.data) / 'val'
    levels = [nets['inner']['level'], nets['outer']['level']]
    STATE['nets'] = nets
    STATE['gt'] = [ground_truth(data, name, levels, [tuple(levels)]) for name in nets['inner']['names']]
    current = {key: dict(nets[key]['info']['own']) for key in nets}
    trace = []
    started = time.time()
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(args.workers) as pool:
        start = evaluate(pool, 'inner', [current['inner']], current['outer'])[0]
        print(f"own settings {current}: objective {start['objective']:.4f} (JTPR {start['jtpr']:.4f}, "
              f"AP {start['ap_inner']:.4f} / {start['ap_outer']:.4f})", flush=True)
        best = start
        for round_ in range(MAX_ROUNDS):
            changed = False
            for net in ('inner', 'outer'):
                for key in ('prob', 'mask', 'gate', 'area'):
                    values = nets[net]['grids'][key]
                    if len(values) < 2:
                        continue
                    settings = [dict(current[net], **{key: v}) for v in values]
                    other = current['outer' if net == 'inner' else 'inner']
                    rows = evaluate(pool, net, settings, other)
                    here = next(r for v, r in zip(values, rows) if v == current[net][key])
                    k = int(np.argmax([r['objective'] for r in rows]))
                    trace.append(dict(round=round_, network=net, key=key, values=values, rows=rows,
                                      before=current[net][key], seconds=round(time.time() - started)))
                    if rows[k]['objective'] > here['objective'] + MIN_GAIN:
                        print(f"round {round_} {net} {key}: {current[net][key]} -> {values[k]}, objective "
                              f"{here['objective']:.4f} -> {rows[k]['objective']:.4f} (JTPR {rows[k]['jtpr']:.4f}, AP "
                              f"{rows[k]['ap_inner']:.4f} / {rows[k]['ap_outer']:.4f}) {time.time() - started:.0f} s",
                              flush=True)
                        current[net][key] = values[k]
                        best = rows[k]
                        changed = True
                    else:
                        best = here
                        print(f"round {round_} {net} {key}: keeps {current[net][key]} ({here['objective']:.4f}) "
                              f"{time.time() - started:.0f} s", flush=True)
            if not changed:
                break
    chosen = dict(objective=dict(own=start, chosen=best), own={k: nets[k]['info']['own'] for k in nets},
                  chosen=current, networks={k: dict(config=nets[k]['info']['config'], weights=nets[k]['info']['weights'],
                                                    level=nets[k]['level'], role=nets[k]['role']) for k in nets},
                  val_images=len(STATE['gt']), jtpr_weight=JTPR_WEIGHT, min_gain=MIN_GAIN,
                  grids={k: nets[k]['grids'] for k in nets}, seconds=round(time.time() - started), trace=trace)
    Path(args.out).write_text(json.dumps(chosen, indent=1, default=float))
    print(f"chosen {current}: objective {start['objective']:.4f} -> {best['objective']:.4f} -> {args.out}")


def configs(args):
    chosen = json.loads(Path(args.chosen).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, path in (('inner', args.inner_config), ('outer', args.outer_config)):
        role = chosen['networks'][key]['role']
        for variant, values in (('pairown', chosen['own'][key]), ('pairrecal', chosen['chosen'][key])):
            d = json.loads(Path(path).read_text())
            i = d['inference']
            i[f'prob_thresh_{role}'] = float(values['prob'])
            i[f'mask_thresh_{role}'] = float(values['mask'])
            i['mask_quality_min'] = 'auto'
            i['quality_gates_override'] = {role: float(values['gate']), f'{role}_min_area': int(values['area'])}
            d['_note'] = d.get('_note', '') + (f' {variant}: settings of the {role} network written out for Table 8 '
                                               f'(paper/recal_single_level_pairs.py, {args.chosen}): {json.dumps(values)}')
            target = out / f'{Path(path).stem}.{variant}.json'
            target.write_text(json.dumps(d, indent=2))
            print('->', target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('cache', 'search', 'configs'))
    parser.add_argument('--config')
    parser.add_argument('--weights')
    parser.add_argument('--out')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--search-prob', action='store_true')
    parser.add_argument('--data')
    parser.add_argument('--inner')
    parser.add_argument('--outer')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='first N val images only (smoke tests)')
    parser.add_argument('--chosen')
    parser.add_argument('--inner-config')
    parser.add_argument('--outer-config')
    parser.add_argument('--out-dir')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root: the CytoNucs modules
    if args.step == 'cache':
        cache(args)
    elif args.step == 'search':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        search(args)
    else:
        configs(args)


if __name__ == '__main__':
    main()
