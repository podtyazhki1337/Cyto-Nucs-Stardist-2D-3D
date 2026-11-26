"""Operating points of a joint two-level network and of two separate single-level networks (one trained on the inner
level only, one on the outer level only), chosen on val with the same objective and the same threshold budget, then
scored on test: the comparison of joint and separate training at matched calibration.

Budget (both arms): per level the prompt probability threshold, the mask threshold, the quality gate and the minimum
size, plus the two nesting filters (inference.inner_min_inside_outer, inference.outer_min_inner_share), which apply to
the two label maps of either arm. Objective: w x linked JTPR + (1 - w) x mean AP of the two levels (w = 0: AP only;
0.5: the objective of CytoNucs' calibration; 0.8: JTPR-heavy). Links: the joint network's own (its nucleus-cell
evidence, as its inference assigns them); the separate pair by overlap (as the scorers pair them). The search
re-assembles candidates cached from one ungated val pass per network (`paper/recal_single_level_pairs.py cache` for
the separate networks, `cache-joint` here for the joint one), with a coordinate search from the networks' own
settings, as recal_single_level_pairs.py does.

  cache-joint  (GPU) val of the joint network once, ungated: every candidate of both levels and their pairing evidence
  search       (CPU) --arm joint --joint DIR | --arm separate --inner DIR --outer DIR, --w W -> OUT/chosen_<arm>_w<W>.json
  configs      scoring configurations of the chosen settings (the joint configuration, or one per separate network)
  pair         (CPU) the separate networks' test labels with the nesting filters of a chosen file applied, in one
               folder per level (for score_predictions.py)
Then score_hierarchy.py (joint) or score_predictions.py (the pair) on the test splits.

Usage (from the repository root):
  python paper/operating_points.py cache-joint --config CFG --weights W --out DIR [--gpu 0] [--search-prob] [--limit N]
  python paper/operating_points.py search --arm joint --joint DIR --data DATA --w 0.5 --out OUT [--workers 4]
  python paper/operating_points.py search --arm separate --inner DIR --outer DIR --data DATA --w 0.5 --out OUT
  python paper/operating_points.py configs --chosen OUT/chosen_joint_w0.5.json --config CFG --out-dir DIR
  python paper/operating_points.py configs --chosen OUT/chosen_separate_w0.5.json --inner-config CFG --outer-config CFG --out-dir DIR
  python paper/operating_points.py pair --chosen OUT/chosen_separate_w0.5.json --inner-pred 'IN/predictions_{split}'
      --outer-pred 'OUT/predictions_{split}' --out-dir DIR --split test
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
import recal_single_level_pairs as rp  # noqa: E402

FILTER_GRID = (0.0, 0.1, 0.25, 0.5)
ROLES = ('nucleus', 'cytoplasm')
STATE = {}


def cache_joint(args):
    """Ungated val pass of the joint network: both levels' candidates and the nucleus-cell evidence (as indices)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import tempfile
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    from cytonucs_config import CytoNucsConfig, read_config
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_model import build_cytonucs_model
    from cytonucs_runtime import normalize_percentiles, read_model_image, read_mask_thresholds, read_quality_gates
    from cytonucs_trainer import DiceCallback
    raw = read_config(args.config, args.weights)
    raw['mode'] = 'inference'
    raw['inference']['checkpoint_path'] = str(args.weights)
    inference = raw['inference']
    gates = dict(read_quality_gates(args.weights) or {}, **(inference.get('quality_gates_override') or {}))
    masks = read_mask_thresholds(args.weights) or {}
    own = {role: dict(prob=float(inference[f'prob_thresh_{role}']),
                      mask=float(masks.get(role, inference.get('mask_thresh', 0.45)) if inference.get(f'mask_thresh_{role}') == 'auto'
                                 else inference[f'mask_thresh_{role}']),
                      gate=float(gates.get(role, 0.0)), area=int(gates.get(f'{role}_min_area', 0))) for role in ROLES}
    own['filters'] = dict(inner_min_inside_outer=float(inference.get('inner_min_inside_outer', 0.0) or 0.0),
                          outer_min_inner_share=float(inference.get('outer_min_inner_share', 0.0) or 0.0))
    prob_range = sorted({float(v) for v in raw['evaluation']['prob_thresh_range']})
    prob = {f'prob_thresh_{r}': (min(prob_range + [own[r]['prob']]) if args.search_prob else own[r]['prob']) for r in ROLES}
    out = Path(args.out)
    (out / 'val').mkdir(parents=True, exist_ok=True)
    folder = Path(raw['data']['dataset_dir']) / 'val'
    names = sorted(p.name for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))[:args.limit or None]
    with tempfile.TemporaryDirectory() as tmp:
        raw['output']['basedir'] = tmp
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(raw))
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = CytoNucsConfig.from_json(str(path))
            model = build_cytonucs_model(cfg, name='op', basedir=tmp)
            model.load_weights(str(args.weights))
        predictor = CytoNucsPredictor(model=model, config=cfg, nms_thresh=inference['nms_thresh'], **prob)
        predictor.calibrating = True
        nd = int(cfg.ndim)
        started = time.time()
        for name in names:
            target = out / 'val' / (Path(name).stem + '.pkl')
            if target.is_file():
                continue
            image = np.asarray(read_model_image(folder / 'images' / name, nd, int(cfg.n_image_channels)), np.float32)
            if image.ndim == nd:
                image = image[..., None]
            image = normalize_percentiles(image, 1, 99.8, axis=tuple(range(nd)))
            sink = {}
            predictor._refinement_candidate_sink = sink
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    DiceCallback._predict_full_volume(SimpleNamespace(model=model), predictor, image)
            finally:
                predictor._refinement_candidate_sink = None
            cands = {r: [c for c in sink.get(r, []) if c is not None] for r in ROLES}
            index = {r: {id(c): i for i, c in enumerate(cands[r])} for r in ROLES}
            evidence = [(index['nucleus'][id(n)], index['cytoplasm'].get(id(c)) if c is not None else None)
                        for n, c in sink.get('evidence', []) if n is not None and id(n) in index['nucleus']]
            strip = lambda c: {k: v for k, v in c.items() if k != 'roi'}                     # noqa: E731
            with open(target, 'wb') as f:
                pickle.dump(dict(name=name, shape=tuple(int(s) for s in image.shape[:nd]),
                                 candidates={r: [strip(c) for c in cands[r]] for r in ROLES}, evidence=evidence), f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            print(f'cache {name}: {len(cands["nucleus"])} / {len(cands["cytoplasm"])} candidates, {len(evidence)} evidence '
                  f'({time.time() - started:.0f} s)', flush=True)
    (out / 'network.json').write_text(json.dumps(dict(config=str(args.config), weights=str(args.weights), own=own,
                                                      search_prob=bool(args.search_prob), prob_range=prob_range,
                                                      mask_default=float(inference.get('mask_thresh', 0.45)),
                                                      pair_min_overlap=float(inference.get('pair_min_overlap', 0.1)),
                                                      pair_min_mask_iou=float(inference.get('pair_min_mask_iou', 0.25)),
                                                      images=names), indent=1))
    print('->', out)


def transformed(cands, setting, mask_default):
    """Candidates of one level at a setting (None where dropped), in the cached order."""
    from cytonucs_postprocess import rethreshold_candidate
    out = []
    for c in cands:
        if not c.get('residual', False) and c.get('detection_score', 1.0) < setting['prob'][c.get('prompt_head', 'nucleus')]:
            out.append(None)
            continue
        if abs(setting['mask'] - float(c.get('threshold', mask_default))) > 1e-9:
            d = rethreshold_candidate(c, setting['mask'])
            if d is not None:
                for key in rp.KEPT:
                    if key in c:
                        d[key] = c[key]
            c = d
        if c is not None and 'predicted_quality' in c and c['predicted_quality'] < setting['gate']:
            c = None
        out.append(c)
    return out


def labels_joint(index, s):
    """Labels of both levels and the links of the joint network at setting s (per-role dicts + filters)."""
    from cytonucs_inference import assemble_head, keep_inner_inside_outer, keep_outer_with_inner
    from cytonucs_postprocess import assign_final_masks, drop_small_instances
    net = STATE['joint']
    item = net['images'][index]
    prob = {r: s[r]['prob'] for r in ROLES}
    kept = {r: transformed(item['candidates'][r], dict(s[r], prob=prob), net['mask_default']) for r in ROLES}
    labels = {}
    for r in ROLES:
        lab, _ = assemble_head(net['cfg'], r, [c for c in kept[r] if c is not None], item['shape'])
        labels[r], _ = drop_small_instances(np.asarray(lab), int(s[r]['area']))
    shares = SimpleNamespace(**s['filters'])
    cells, _ = keep_outer_with_inner(shares, labels['nucleus'], labels['cytoplasm'])
    nuclei, _ = keep_inner_inside_outer(shares, labels['nucleus'], cells)
    evidence = [(kept['nucleus'][n], kept['cytoplasm'][c] if c is not None else None) for n, c in item['evidence']]
    evidence = [(n, c) for n, c in evidence if n is not None]
    links = assign_final_masks(nuclei, cells, net['pair_min_overlap'], evidence, net['pair_min_mask_iou'])
    return nuclei, cells, links


def labels_separate(index, s):
    from cytonucs_inference import keep_inner_inside_outer, keep_outer_with_inner
    rp.STATE['nets'] = STATE['nets']
    rp.STATE['gt'] = STATE['gt']
    nuclei = rp.labels_of('inner', index, dict(s['nucleus'], prob=s['nucleus']['prob']))
    cells = rp.labels_of('outer', index, dict(s['cytoplasm'], prob=s['cytoplasm']['prob']))
    shares = SimpleNamespace(**s['filters'])
    cells, _ = keep_outer_with_inner(shares, nuclei, cells)
    nuclei, _ = keep_inner_inside_outer(shares, nuclei, cells)
    return nuclei, cells, None                                                              # links by overlap


def score_image(task):
    from cytonucs_calibration import match_counts
    from cytonucs_inference import compute_jtpr
    index, settings = task
    gt, truths, _ = STATE['gt'][index]
    out = []
    for s in settings:
        nuclei, cells, links = (labels_joint if STATE['arm'] == 'joint' else labels_separate)(index, s)
        with contextlib.redirect_stdout(io.StringIO()):
            _, d = compute_jtpr(nuclei, cells, gt['nucleus'], gt['cytoplasm'], iou_thresh=0.5, assignments_pred=links,
                                assignments_gt=truths['nucleus_cytoplasm'])
        out.append((match_counts(gt['nucleus'], nuclei), match_counts(gt['cytoplasm'], cells), int(d['joint_tp']), int(d['total_pairs'])))
    return out


def evaluate(pool, settings, w):
    from cytonucs_calibration import ap_of
    res = pool.map(score_image, [(i, settings) for i in range(len(STATE['gt']))], chunksize=1)
    rows = []
    for k in range(len(settings)):
        ap_n = ap_of(sum(r[k][0] for r in res))
        ap_c = ap_of(sum(r[k][1] for r in res))
        jtpr = sum(r[k][2] for r in res) / max(sum(r[k][3] for r in res), 1)
        rows.append(dict(objective=w * jtpr + (1 - w) * (ap_n + ap_c) / 2, jtpr=jtpr, ap_nucleus=ap_n, ap_cytoplasm=ap_c))
    return rows


def search(args):
    import multiprocessing
    from score_hierarchy import ground_truth
    if args.arm == 'joint':
        info = json.loads((Path(args.joint) / 'network.json').read_text())
        from cytonucs_config import CytoNucsConfig, read_config
        import tempfile
        raw = read_config(info['config'])
        with tempfile.TemporaryDirectory() as tmp:
            raw['output']['basedir'] = tmp
            p = Path(tmp) / 'c.json'
            p.write_text(json.dumps(raw))
            with contextlib.redirect_stdout(io.StringIO()):
                cfg = CytoNucsConfig.from_json(str(p))
        names = info['images'][:args.limit or None]
        images = [pickle.load(open(Path(args.joint) / 'val' / (Path(n).stem + '.pkl'), 'rb')) for n in names]
        STATE['joint'] = dict(cfg=cfg, images=images, mask_default=info['mask_default'], pair_min_overlap=info['pair_min_overlap'],
                              pair_min_mask_iou=info['pair_min_mask_iou'])
        own = {r: dict(info['own'][r]) for r in ROLES}
        own['filters'] = dict(info['own']['filters'])
        probs = {r: (sorted(set(info['prob_range']) | {own[r]['prob']}) if info['search_prob'] else [own[r]['prob']]) for r in ROLES}
    else:
        nets = {}
        for key, folder in (('inner', args.inner), ('outer', args.outer)):
            i = json.loads((Path(folder) / 'network.json').read_text())
            _, cfg, level, role = rp.network_config(i['config'])
            names = i['images'][:args.limit or None]
            images = [pickle.load(open(Path(folder) / 'val' / (Path(n).stem + '.pkl'), 'rb')) for n in names]
            nets[key] = dict(info=i, cfg=cfg, level=level, role=role, images=images, mask_default=i['mask_default'])
        STATE['nets'] = nets
        own = {'nucleus': {k: v for k, v in nets['inner']['info']['own'].items()},
               'cytoplasm': {k: v for k, v in nets['outer']['info']['own'].items()},
               'filters': dict(inner_min_inside_outer=0.0, outer_min_inner_share=0.0)}
        probs = {r: (sorted(set(nets[k]['info']['prob_range']) | {own[r]['prob']}) if nets[k]['info']['search_prob'] else [own[r]['prob']])
                 for r, k in (('nucleus', 'inner'), ('cytoplasm', 'outer'))}
    data = Path(args.data) / 'val'
    STATE['arm'] = args.arm
    STATE['gt'] = [ground_truth(data, n, list(ROLES), [tuple(ROLES)]) for n in names]
    grids = {r: dict(prob=probs[r], mask=sorted(set(rp.MASK_GRID) | {own[r]['mask']}), gate=sorted(set(rp.GATE_GRID) | {own[r]['gate']}),
                     area=sorted(set(rp.AREA_GRID) | {own[r]['area']})) for r in ROLES}
    grids['filters'] = {k: sorted(set(FILTER_GRID) | {own['filters'][k]}) for k in own['filters']}
    current = json.loads(json.dumps(own))
    trace, started = [], time.time()
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        start = evaluate(pool, [current], args.w)[0]
        best = start
        print(f'{args.arm} w={args.w}: own {current} -> {start}', flush=True)
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
                                      val_images=len(names), grids=grids, seconds=round(time.time() - started), trace=trace), indent=1, default=float))
    print(f'chosen {current}: {start["objective"]:.4f} -> {best["objective"]:.4f} -> {target}')


def configs(args):
    ch = json.loads(Path(args.chosen).read_text())
    s, out = ch['chosen'], Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{ch['arm']}_w{ch['w']}"
    targets = [('both', args.config)] if ch['arm'] == 'joint' else [('nucleus', args.inner_config), ('cytoplasm', args.outer_config)]
    for role, path in targets:
        d = json.loads(Path(path).read_text())
        i = d['inference']
        roles = ROLES if role == 'both' else (role,)
        override = {}
        for r in roles:
            i[f'prob_thresh_{r}'] = float(s[r]['prob'])
            i[f'mask_thresh_{r}'] = float(s[r]['mask'])
            override.update({r: float(s[r]['gate']), f'{r}_min_area': int(s[r]['area'])})
        i['mask_quality_min'] = 'auto'
        i['quality_gates_override'] = override
        if role == 'both':
            i.update({k: float(v) for k, v in s['filters'].items()})
        d['_note'] = d.get('_note', '') + f' op_{tag}: settings chosen on val for objective w={ch["w"]} (paper/operating_points.py): {json.dumps(s)}'
        target = out / f'{Path(path).stem}.op_{tag}.json'
        target.write_text(json.dumps(d, indent=2))
        print('->', target)


def pair(args):
    """Separate networks' test labels with the chosen nesting filters, as <out>/<split>/{nucleus,cytoplasm}."""
    import tifffile
    from cytonucs_inference import keep_inner_inside_outer, keep_outer_with_inner
    s = json.loads(Path(args.chosen).read_text())['chosen']['filters']
    shares = SimpleNamespace(**s)
    for split in args.split:
        a, b = Path(args.inner_pred.format(split=split)), Path(args.outer_pred.format(split=split))
        out = Path(args.out_dir) / split
        for level in ROLES:
            (out / level).mkdir(parents=True, exist_ok=True)
        for f in sorted((a / 'nucleus').iterdir()):
            nuclei, cells = tifffile.imread(f), tifffile.imread(b / 'cytoplasm' / f.name)
            cells, _ = keep_outer_with_inner(shares, nuclei, cells)
            nuclei, _ = keep_inner_inside_outer(shares, nuclei, cells)
            tifffile.imwrite(out / 'nucleus' / f.name, nuclei.astype(np.uint16), compression='zlib')
            tifffile.imwrite(out / 'cytoplasm' / f.name, cells.astype(np.uint16), compression='zlib')
        print('->', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('cache-joint', 'search', 'configs', 'pair'))
    parser.add_argument('--config'); parser.add_argument('--weights'); parser.add_argument('--out'); parser.add_argument('--gpu', default='0')
    parser.add_argument('--search-prob', action='store_true'); parser.add_argument('--data'); parser.add_argument('--arm', choices=('joint', 'separate'))
    parser.add_argument('--w', type=float, default=0.5); parser.add_argument('--joint'); parser.add_argument('--inner'); parser.add_argument('--outer')
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--chosen'); parser.add_argument('--inner-config'); parser.add_argument('--outer-config'); parser.add_argument('--out-dir')
    parser.add_argument('--inner-pred'); parser.add_argument('--outer-pred'); parser.add_argument('--split', nargs='+', default=['test'])
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root: the CytoNucs modules
    if args.step == 'cache-joint':
        cache_joint(args)
    elif args.step == 'search':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        search(args)
    elif args.step == 'configs':
        configs(args)
    else:
        pair(args)


if __name__ == '__main__':
    main()
