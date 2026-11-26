"""Operating points of a joint 3D network and of two separate single-level 3D networks on the held-out blocks of a
spatially split volume (the HeLa spheroid), chosen on the val blocks with the same objective and the same budget as
paper/operating_points.py: per level the prompt prob threshold, the mask threshold, the quality gate and the minimum
size, plus the two nesting filters; objective w x linked JTPR + (1 - w) x mean AP. Then both arms at their own
settings and at the chosen ones on the test and val blocks, with real inference.

The 2D search of operating_points.py re-assembles candidates cached per image. Here an "image" is the window of a held-out
block as evaluate_blocks.py predicts it (the block and --margin native px of context, at the network's resolution);
the labels assembled from the window's candidates are enlarged to the native resolution, cut to the objects whose
centroid lies in the block (objects_in_block), and matched with the block's native annotation. The re-assembly,
the links (joint: its own nucleus-cell evidence; separate: by overlap) and the search are operating_points.py's own
functions.
Size grids are relative to the network's own minimum size (3D sizes are voxels): own x AREA_SCALES, and 0.

  cache    (GPU) val windows of one network, ungated, every candidate (both levels and the evidence for the joint network),
           predicted once at the lowest prob threshold of evaluation.prob_thresh_range -> DIR/val/<window>.pkl
  search   (CPU) --arm joint --joint DIR | --arm separate --inner DIR --outer DIR, --w W -> OUT/chosen_<arm>_w<W>.json
  configs  operating_points.configs: the scoring configurations of the chosen settings
  predict  (GPU) real inference of one network on the windows of --split -> DIR/<split>/<window>.npz (window labels at
           the network's resolution, the links)
  score    (CPU) --joint DIR | --inner DIR --outer DIR [--chosen F for the nesting filters of the pair] --split S:
           AP of both levels and linked JTPR on the blocks, as evaluate_blocks.py -> --output JSON
Usage (from the repository root; the separate networks: cache, predict once each with their own --config / --weights):
  python paper/operating_points_3d.py cache --config CFG --weights W --out DIR [--gpu 0] [--margin 192]
  python paper/operating_points_3d.py search --arm joint --joint DIR --w 0.5 --out OUT [--workers 4]
  python paper/operating_points_3d.py search --arm separate --inner DIR --outer DIR --w 0.5 --out OUT
  python paper/operating_points_3d.py configs --chosen OUT/chosen_joint_w0.5.json --config CFG --out-dir DIR
  python paper/operating_points_3d.py predict --config CFG --weights W --out DIR --split test [--gpu 0]
  python paper/operating_points_3d.py score --joint DIR --split test --output S.json
  python paper/operating_points_3d.py score --inner DIR --outer DIR [--chosen OUT/chosen_separate_w0.5.json] --split test --output S.json
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))         # paper/
import operating_points as op  # noqa: E402
import recal_single_level_pairs as rp  # noqa: E402

ROLES = ('nucleus', 'cytoplasm')
LEVEL_FILES = {'nucleus': 'nucleus_masks', 'cytoplasm': 'cytoplasm_masks'}
AREA_SCALES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def resolved(config, weights=None):
    from cytonucs_config import read_config
    raw = read_config(config, weights)
    raw['mode'] = 'inference'
    if weights:
        raw['inference']['checkpoint_path'] = str(weights)
    return raw


def geometry(raw, margin, split):
    """(factor, native folder, [(volume name, block, window)]) of the held-out blocks of a split."""
    from evaluate_blocks import block_windows
    dataset = Path(raw['data']['dataset_dir'])
    manifest = json.loads((dataset / 'split_manifest.json').read_text())
    factor = int(manifest.get('xy_downscale', 1))
    native = Path(manifest['downscaled_from']) if factor > 1 else dataset
    native_manifest = json.loads((native / 'split_manifest.json').read_text())
    return factor, native, list(block_windows(native_manifest, split, margin, factor))


def wname(volume, block, split):
    return f'{split}_{volume}_y{block[0]:04d}_x{block[2]:04d}'


def own_settings(raw, weights, roles):
    """The settings a network is scored with now, per role, and its nesting filters (as operating_points.cache_joint)."""
    from cytonucs_runtime import read_mask_thresholds, read_quality_gates
    inference = raw['inference']
    gates = dict(read_quality_gates(weights) or {}, **(inference.get('quality_gates_override') or {}))
    masks = read_mask_thresholds(weights) or {}
    own = {}
    for role in roles:
        mask = inference.get(f'mask_thresh_{role}')
        if mask == 'auto' or mask is None:
            mask = masks.get(role, inference.get('mask_thresh', 0.45))
        own[role] = dict(prob=float(inference[f'prob_thresh_{role}']), mask=float(mask), gate=float(gates.get(role, 0.0)),
                         area=int(gates.get(f'{role}_min_area', 0)))
    own['filters'] = dict(inner_min_inside_outer=float(inference.get('inner_min_inside_outer', 0.0) or 0.0),
                          outer_min_inner_share=float(inference.get('outer_min_inner_share', 0.0) or 0.0))
    return own


def build(raw, weights, tmp):
    from cytonucs_config import CytoNucsConfig
    from cytonucs_model import build_cytonucs_model
    raw = json.loads(json.dumps(raw))
    raw['output']['basedir'] = tmp
    path = Path(tmp) / 'config.json'
    path.write_text(json.dumps(raw))
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = CytoNucsConfig.from_json(str(path))
        model = build_cytonucs_model(cfg, name='op3d', basedir=tmp)
        model.load_weights(str(weights))
    return cfg, model


def window_images(raw, native, factor, windows, channels):
    """Normalised window images (as evaluate_blocks.evaluate), volume by volume."""
    from cytonucs_runtime import normalize_percentiles as normalize, read_model_image
    dataset = Path(raw['data']['dataset_dir'])
    cache = {}
    for volume, block, window in windows:
        if volume not in cache:
            cache = {volume: read_model_image(dataset / 'train_regions' / 'images' / (volume + '.tiff'), 3, channels)}
        y0, y1, x0, x1 = window
        image = cache[volume][:, y0 // factor:y1 // factor, x0 // factor:x1 // factor]
        yield volume, block, window, normalize(image.astype(np.float32), 1, 99.8, axis=(0, 1, 2)).astype(np.float32)


def tf_setup(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)


def cache(args):
    tf_setup(args.gpu)
    import tempfile
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_trainer import DiceCallback
    raw = resolved(args.config, args.weights)
    levels = list(raw['model']['levels'])
    joint = len(levels) == 2
    if joint:
        roles, level, role = ROLES, None, None
    else:
        _, _, level, role = rp.network_config(args.config, args.weights)
        roles = (role,)
    own = own_settings(raw, args.weights, roles)
    inference = raw['inference']
    prob_range = sorted({float(v) for v in raw['evaluation']['prob_thresh_range']})
    prob = {f'prob_thresh_{r}': float(inference[f'prob_thresh_{r}']) for r in ROLES}
    for r in roles:
        prob[f'prob_thresh_{r}'] = min(prob_range + [own[r]['prob']])
    factor, native, windows = geometry(raw, args.margin, 'val')
    out = Path(args.out)
    (out / 'val').mkdir(parents=True, exist_ok=True)
    names = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg, model = build(raw, args.weights, tmp)
        predictor = CytoNucsPredictor(model=model, config=cfg, nms_thresh=inference['nms_thresh'], **prob)
        predictor.calibrating = True
        runner = SimpleNamespace(model=predictor.model)
        channels = int(getattr(cfg, 'n_image_channels', cfg.n_channel_in))
        started = time.time()
        for volume, block, window, image in window_images(raw, native, factor, windows, channels):
            name = wname(volume, block, 'val')
            names.append(name)
            target = out / 'val' / f'{name}.pkl'
            if target.is_file():
                continue
            sink = {}
            predictor._refinement_candidate_sink = sink
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    DiceCallback._predict_full_volume(runner, predictor, image)
            finally:
                predictor._refinement_candidate_sink = None
            strip = lambda c: {k: v for k, v in c.items() if k != 'roi'}                    # noqa: E731
            cands = {r: [c for c in sink.get(r, []) if c is not None] for r in ROLES}
            item = dict(name=name, volume=volume, block=list(block), window=list(window), shape=tuple(int(s) for s in image.shape[:3]))
            if joint:
                index = {r: {id(c): i for i, c in enumerate(cands[r])} for r in ROLES}
                item['evidence'] = [(index['nucleus'][id(n)], index['cytoplasm'].get(id(c)) if c is not None else None)
                                    for n, c in sink.get('evidence', []) if n is not None and id(n) in index['nucleus']]
                item['candidates'] = {r: [strip(c) for c in cands[r]] for r in ROLES}
            else:
                item['candidates'] = [strip(c) for c in cands[role]]
            with open(target, 'wb') as f:
                pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'cache {name}: ' + ' / '.join(f'{len(cands[r])} {r}' for r in roles) + f' candidates ({time.time() - started:.0f} s)', flush=True)
    (out / 'network.json').write_text(json.dumps(dict(
        config=str(args.config), weights=str(args.weights), joint=joint, roles=list(roles), level=level, role=role, own=own,
        search_prob=True, prob_range=prob_range, predicted_prob={r: prob[f'prob_thresh_{r}'] for r in roles},
        mask_default=float(inference.get('mask_thresh', 0.45)), pair_min_overlap=float(inference.get('pair_min_overlap', 0.1)),
        pair_min_mask_iou=float(inference.get('pair_min_mask_iou', 0.25)), images=names, factor=factor, native=str(native),
        margin=args.margin), indent=1))
    print('->', out)


# ---------------------------------------------------------------- the blocks' annotation and the scoring of a window
def block_truth(native, items):
    """Per window: (GT of both levels cut to the block's objects, {'nucleus_cytoplasm': pairs}, {} ignore), native res."""
    import tifffile
    from evaluate_blocks import objects_in_block
    out, volumes = [], {}
    for it in items:
        v = it['volume']
        if v not in volumes:
            volumes = {v: dict(**{lv: tifffile.imread(Path(native) / 'train_regions' / f / (v + '.tiff')) for lv, f in LEVEL_FILES.items()},
                               pairs={int(k): int(x) for k, x in json.loads((Path(native) / 'train_regions' / 'assignments' / (v + '.json')).read_text()).items()})}
        y0, y1, x0, x1 = it['window']
        gt = {lv: objects_in_block(volumes[v][lv][:, y0:y1, x0:x1], it['block'], it['window']) for lv in LEVEL_FILES}
        out.append((gt, {'nucleus_cytoplasm': volumes[v]['pairs']}, {}))
    return out


def to_block(labels, factor, gt_shape, block, window):
    from evaluate_blocks import objects_in_block
    from score_native_resolution import enlarge_labels
    return objects_in_block(enlarge_labels(np.asarray(labels), factor, gt_shape), block, window)


def counts(gt, pred):
    """TP / FP / FN at the scorer's IoU thresholds (evaluate_blocks.py: stardist matching)."""
    from score_native_resolution import TAUS
    from stardist.matching import matching
    return np.array([(int(r.tp), int(r.fp), int(r.fn)) for r in matching(gt, pred, thresh=TAUS, criterion='iou')]).T


def jtpr_of(nuclei, cells, gt, truths, links):
    from cytonucs_inference import compute_jtpr
    with contextlib.redirect_stdout(io.StringIO()):
        _, d = compute_jtpr(nuclei, cells, gt['nucleus'], gt['cytoplasm'], iou_thresh=0.5, assignments_pred=links,
                            assignments_gt=truths['nucleus_cytoplasm'])
    return int(d['joint_tp']), int(d['total_pairs'])


def score_task(task):
    """One (window, setting): counts of both levels and the joint TP, at the native resolution in the block."""
    index, s = task
    gt, truths, _ = op.STATE['gt'][index]
    item = op.STATE['items'][index]
    nuclei, cells, links = (op.labels_joint if op.STATE['arm'] == 'joint' else op.labels_separate)(index, s)
    shape = gt['nucleus'].shape
    nuclei, cells = (to_block(x, op.STATE['factor'], shape, item['block'], item['window']) for x in (nuclei, cells))
    return counts(gt['nucleus'], nuclei), counts(gt['cytoplasm'], cells), *jtpr_of(nuclei, cells, gt, truths, links)


def evaluate(pool, settings, w):
    from cytonucs_runtime import accuracy_profile
    n = len(op.STATE['gt'])
    res = pool.map(score_task, [(i, s) for s in settings for i in range(n)], chunksize=1)
    rows = []
    for k in range(len(settings)):
        part = res[k * n:(k + 1) * n]
        ap = [float(accuracy_profile(*sum(r[j] for r in part)).mean()) for j in (0, 1)]
        jtpr = sum(r[2] for r in part) / max(sum(r[3] for r in part), 1)
        rows.append(dict(objective=w * jtpr + (1 - w) * (ap[0] + ap[1]) / 2, jtpr=jtpr, ap_nucleus=ap[0], ap_cytoplasm=ap[1]))
    return rows


def load_cache(folder):
    info = json.loads((Path(folder) / 'network.json').read_text())
    items = [pickle.load(open(Path(folder) / 'val' / f'{n}.pkl', 'rb')) for n in info['images']]
    return info, items


def search(args):
    import multiprocessing
    import tempfile
    from cytonucs_config import CytoNucsConfig
    if args.arm == 'joint':
        info, items = load_cache(args.joint)
        raw = resolved(info['config'])
        with tempfile.TemporaryDirectory() as tmp:
            raw['output']['basedir'] = tmp
            p = Path(tmp) / 'c.json'
            p.write_text(json.dumps(raw))
            with contextlib.redirect_stdout(io.StringIO()):
                cfg = CytoNucsConfig.from_json(str(p))
        op.STATE['joint'] = dict(cfg=cfg, images=items, mask_default=info['mask_default'], pair_min_overlap=info['pair_min_overlap'],
                                 pair_min_mask_iou=info['pair_min_mask_iou'])
        own = {r: dict(info['own'][r]) for r in ROLES}
        own['filters'] = dict(info['own']['filters'])
        prob_range, base = {r: info['prob_range'] for r in ROLES}, info
    else:
        nets = {}
        for key, folder in (('inner', args.inner), ('outer', args.outer)):
            i, items = load_cache(folder)
            _, cfg, level, role = rp.network_config(i['config'])
            nets[key] = dict(info=i, cfg=cfg, level=level, role=role, images=items, mask_default=i['mask_default'])
        op.STATE['nets'] = nets
        own = {'nucleus': dict(nets['inner']['info']['own']['nucleus']), 'cytoplasm': dict(nets['outer']['info']['own']['cytoplasm']),
               'filters': dict(inner_min_inside_outer=0.0, outer_min_inner_share=0.0)}
        prob_range = {'nucleus': nets['inner']['info']['prob_range'], 'cytoplasm': nets['outer']['info']['prob_range']}
        items, base = nets['inner']['images'], nets['inner']['info']
        assert [it['name'] for it in items] == [it['name'] for it in nets['outer']['images']], 'the two caches must hold the same windows'
    op.STATE.update(arm=args.arm, items=items, factor=int(base['factor']), gt=block_truth(base['native'], items))
    grids = {r: dict(prob=sorted(set(prob_range[r]) | {own[r]['prob']}), mask=sorted(set(rp.MASK_GRID) | {own[r]['mask']}),
                     gate=sorted(set(rp.GATE_GRID) | {own[r]['gate']}),
                     area=sorted({0, own[r]['area']} | {int(round(own[r]['area'] * k)) for k in AREA_SCALES})) for r in ROLES}
    grids['filters'] = {k: sorted(set(op.FILTER_GRID) | {own['filters'][k]}) for k in own['filters']}
    current = json.loads(json.dumps(own))
    trace, started = [], time.time()
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        start = evaluate(pool, [current], args.w)[0]
        best = start
        print(f'{args.arm} w={args.w}: own {current} -> {start} ({time.time() - started:.0f} s)', flush=True)
        for round_ in range(rp.MAX_ROUNDS):
            changed = False
            for group in ('nucleus', 'cytoplasm', 'filters'):
                for key, values in grids[group].items():
                    if len(values) < 2:
                        continue
                    settings = []
                    for v in values:
                        s = json.loads(json.dumps(current))
                        s[group][key] = v
                        settings.append(s)
                    rows = evaluate(pool, settings, args.w)
                    here = next(r for v, r in zip(values, rows) if v == current[group][key])
                    k = int(np.argmax([r['objective'] for r in rows]))
                    trace.append(dict(round=round_, group=group, key=key, values=values, rows=rows, before=current[group][key]))
                    if rows[k]['objective'] > here['objective'] + rp.MIN_GAIN:
                        print(f"round {round_} {group}.{key}: {current[group][key]} -> {values[k]}, objective {here['objective']:.4f} -> "
                              f"{rows[k]['objective']:.4f} ({time.time() - started:.0f} s)", flush=True)
                        current[group][key] = values[k]
                        best, changed = rows[k], True
                    else:
                        best = here
            if not changed:
                break
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'chosen_{args.arm}_w{args.w}.json'
    target.write_text(json.dumps(dict(arm=args.arm, w=args.w, own=own, chosen=current, objective=dict(own=start, chosen=best),
                                      val_windows=len(items), grids=grids, seconds=round(time.time() - started), trace=trace),
                                 indent=1, default=float))
    print(f'chosen {current}: {start["objective"]:.4f} -> {best["objective"]:.4f} -> {target}', flush=True)


# ---------------------------------------------------------------- real inference and the scores on the blocks
def predict(args):
    tf_setup(args.gpu)
    import tempfile
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_trainer import DiceCallback
    raw = resolved(args.config, args.weights)
    inference = raw['inference']
    factor, native, windows = geometry(raw, args.margin, args.split)
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, model = build(raw, args.weights, tmp)
        predictor = CytoNucsPredictor(model=model, config=cfg, prob_thresh_nucleus=inference['prob_thresh_nucleus'],
                                      prob_thresh_cytoplasm=inference['prob_thresh_cytoplasm'], nms_thresh=inference['nms_thresh'])
        print('gates', getattr(model, 'loaded_quality_gates', None), 'override', inference.get('quality_gates_override'),
              'mask thresholds', getattr(model, 'loaded_mask_thresholds', None), flush=True)
        runner = SimpleNamespace(model=predictor.model)
        channels = int(getattr(cfg, 'n_image_channels', cfg.n_channel_in))
        for volume, block, window, image in window_images(raw, native, factor, windows, channels):
            name = wname(volume, block, args.split)
            if (out / f'{name}.npz').is_file():
                continue
            started = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                nuclei, cells, _ = DiceCallback._predict_full_volume(runner, predictor, image)
            links = getattr(predictor, '_prompt_assignments', None)
            np.savez_compressed(out / f'{name}.npz', nucleus=np.asarray(nuclei, np.int32), cytoplasm=np.asarray(cells, np.int32))
            (out / f'{name}.json').write_text(json.dumps(dict(volume=volume, block=list(block), window=list(window),
                                                              links={int(k): int(v) for k, v in (links or {}).items()},
                                                              seconds=round(time.time() - started, 1))))
            print(f'{args.split} {name}: {len(np.unique(nuclei)) - 1} nuclei / {len(np.unique(cells)) - 1} cells '
                  f'({time.time() - started:.0f} s)', flush=True)
    (Path(args.out) / 'network.json').write_text(json.dumps(dict(config=str(args.config), weights=str(args.weights),
                                                                 factor=factor, native=str(native), margin=args.margin), indent=1))
    print('->', out)


def score(args):
    from cytonucs_inference import keep_inner_inside_outer, keep_outer_with_inner
    from cytonucs_runtime import accuracy_profile
    from score_native_resolution import TAUS
    first = Path(args.joint or args.inner)
    info = json.loads((first / 'network.json').read_text())
    factor, native = int(info['factor']), info['native']
    names = sorted(p.stem for p in (first / args.split).glob('*.npz'))
    shares = (SimpleNamespace(**json.loads(Path(args.chosen).read_text())['chosen']['filters']) if args.chosen
              else SimpleNamespace(inner_min_inside_outer=0.0, outer_min_inner_share=0.0))
    tot = {lv: np.zeros((3, len(TAUS)), np.int64) for lv in ROLES}
    joint, objects, blocks = [0, 0], {lv: 0 for lv in ROLES}, []
    for name in names:
        meta = json.loads((first / args.split / f'{name}.json').read_text())
        if args.joint:
            z = np.load(first / args.split / f'{name}.npz')
            nuclei, cells, links = z['nucleus'], z['cytoplasm'], {int(k): int(v) for k, v in meta['links'].items()}
        else:
            nuclei = np.load(Path(args.inner) / args.split / f'{name}.npz')['nucleus']
            cells = np.load(Path(args.outer) / args.split / f'{name}.npz')['cytoplasm']
            cells, _ = keep_outer_with_inner(shares, nuclei, cells)
            nuclei, _ = keep_inner_inside_outer(shares, nuclei, cells)
            links = None
        item = dict(volume=meta['volume'], block=meta['block'], window=meta['window'])
        gt, truths, _ = block_truth(native, [item])[0]
        shape = gt['nucleus'].shape
        nuclei, cells = (to_block(x, factor, shape, item['block'], item['window']) for x in (nuclei, cells))
        pred = {'nucleus': nuclei, 'cytoplasm': cells}
        for lv in ROLES:
            objects[lv] += int(len(np.unique(gt[lv])) - 1)
            tot[lv] += counts(gt[lv], pred[lv])
        tp, n = jtpr_of(nuclei, cells, gt, truths, links)
        joint[0] += tp
        joint[1] += n
        blocks.append(dict(name=name, jtpr=tp / max(n, 1), pairs=n))
    result = dict(split=args.split, blocks=blocks, jtpr=joint[0] / max(joint[1], 1), pairs=joint[1], objects=objects,
                  filters=vars(shares), links='own' if args.joint else 'overlap')
    for lv in ROLES:
        tp, fp, fn = tot[lv]
        ap = accuracy_profile(tp, fp, fn)
        result.update({f'ap_{lv}': float(ap.mean()), f'ap50_{lv}': float(ap[0]),
                       f'recall50_{lv}': float(tp[0] / max(tp[0] + fn[0], 1)), f'precision50_{lv}': float(tp[0] / max(tp[0] + fp[0], 1))})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(f"{args.split}: AP nucleus {result['ap_nucleus']:.3f} | AP cell {result['ap_cytoplasm']:.3f} | JTPR {result['jtpr']:.3f} "
          f"of {joint[1]} pairs ({len(blocks)} blocks) -> {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('cache', 'search', 'configs', 'predict', 'score'))
    p.add_argument('--config'); p.add_argument('--weights'); p.add_argument('--out'); p.add_argument('--gpu', default='0')
    p.add_argument('--margin', type=int, default=192); p.add_argument('--split', default='test')
    p.add_argument('--arm', choices=('joint', 'separate')); p.add_argument('--w', type=float, default=0.5)
    p.add_argument('--joint'); p.add_argument('--inner'); p.add_argument('--outer'); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chosen'); p.add_argument('--inner-config'); p.add_argument('--outer-config'); p.add_argument('--out-dir')
    p.add_argument('--output')
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root: the CytoNucs modules
    if args.step in ('search', 'score', 'configs'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    dict(cache=cache, search=search, configs=op.configs, predict=predict, score=score)[args.step](args)


if __name__ == '__main__':
    main()
