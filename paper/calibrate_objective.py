"""Thresholds of the baselines chosen on validation with the objective of CytoNucs' calibration (the checkpoint metric:
1/2 linked JTPR at IoU 0.5 + 1/2 mean AP of the two levels), so that the methods are compared at the same kind of
operating point. Each baseline keeps its own grid of settings; the pair (setting of the inner level, setting of the
outer level) with the best objective on the val images is taken, and the test splits are predicted with it.

  grid:     predict the val images at every setting of the method -> OUT/val/<level>/<setting>/<name>
  select:   (CytoNucs env) the pair with the best objective, links by overlap as the scorers -> OUT/chosen.json
  predict:  the splits with the chosen pair -> OUT/predictions/<split>/<level>/<name>; then score_predictions.py
Methods (the arguments after `--` are those of the method's scoring call):
  stardist  run_files.py models: prob threshold of each model in PROB_GRID plus its own, its own NMS threshold;
  hsd       run_hsd.py model: prob threshold of each branch in PROB_GRID plus its own, its own NMS thresholds;
  cellpose  run_cellpose.py models (cellpose env): the cellprob x flow grid of run_cellpose.py, per level.

Usage:
  python paper/calibrate_objective.py grid --method stardist --out OUT --gpu 0 -- --data DIR --score --inner ... --outer ...
  python paper/calibrate_objective.py select --out OUT --data DIR
  python paper/calibrate_objective.py predict --method stardist --out OUT --gpu 0 --split test -- <the same arguments>
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                   # the repository root
HERE = ROOT / 'baselines'                                    # the baseline scripts
PROB_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7)        # evaluation.prob_thresh_range of CytoNucs


def split_argv(argv):
    """(own arguments, the scorer's arguments after '--')."""
    if '--' not in argv:
        return argv, []
    i = argv.index('--')
    return argv[:i], argv[i + 1:]


def load_script(path, argv):
    """Import a scoring script with its own command line: its module-level setup reads it."""
    sys.argv = [str(path)] + list(argv)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method(name, gpu, argv):
    """Namespace of a method: data, names(split), read(split, name), levels (inner, outer), settings {level: {key: value}},
    predict(x, {level: value}) -> {level: labels}."""
    import numpy as np
    from types import SimpleNamespace
    if name == 'stardist':
        m = load_script(HERE / 'stardist3d' / 'run_files.py', argv + ['--gpu', gpu])
        kind = m.StarDist3D if m.NDIM == 3 else m.StarDist2D
        levels = dict(level.split(':') for level in (m.args.inner, m.args.outer))
        models = {level: kind(None, name=n, basedir=m.args.basedir) for level, n in levels.items()}
        settings = {level: {f'p{v:.4f}': v for v in sorted(set(PROB_GRID) | {round(float(md.thresholds.prob), 4)})}
                    for level, md in models.items()}

        def predict(x, values):
            return {level: models[level].predict_instances(x, n_tiles=m.n_tiles(x.shape, x.ndim > m.NDIM), prob_thresh=v,
                                                           nms_thresh=models[level].thresholds.nms, show_tile_progress=False)[0]
                    for level, v in values.items()}
        return SimpleNamespace(data=m.DATA, names=m.names, read=lambda s, n: m.read_image(m.DATA / s / 'images' / n),
                               levels=tuple(levels), settings=settings, predict=predict,
                               own={level: f'p{round(float(md.thresholds.prob), 4):.4f}' for level, md in models.items()})
    if name == 'hsd':
        m = load_script(HERE / 'hydrastardist' / 'run_hsd.py', argv + ['--gpu', gpu])
        model = m.StarDist2D(None, name=m.args.name, basedir=m.args.basedir)
        own = {m.args.outer: round(float(model.thresholds1['prob']), 4), m.args.inner: round(float(model.thresholds2['prob']), 4)}
        grid = sorted(set(PROB_GRID) | set(own.values()))
        settings = {level: {f'p{v:.4f}': v for v in grid} for level in (m.args.inner, m.args.outer)}

        def predict(x, values):                            # as run_hsd.score: branch 1 learned the outer level
            first, second = model.predict_instances(
                x, n_tiles=m.n_tiles(x.shape, x.ndim > 2), show_tile_progress=False,
                prob_thresh1=values[m.args.outer], nms_thresh1=model.thresholds1['nms'],
                prob_thresh2=values[m.args.inner], nms_thresh2=model.thresholds2['nms'])
            return {m.args.outer: np.asarray(first[0]), m.args.inner: np.asarray(second[0])}
        return SimpleNamespace(data=m.DATA, names=m.names, read=lambda s, n: m.read_image(m.DATA / s / 'images' / n),
                               levels=(m.args.inner, m.args.outer), settings=settings, predict=predict,
                               own={level: f'p{v:.4f}' for level, v in own.items()})
    if name == 'cellpose':
        m = load_script(HERE / 'cellpose' / 'run_cellpose.py', argv + ['--gpu', gpu])
        import tifffile
        import torch
        from cellpose import models as cp_models
        info = json.loads((m.OUT / 'training.json').read_text())
        chosen = {level: ((cp_models.Cellpose(gpu=torch.cuda.is_available(), model_type=spec['pretrained'])
                           if info.get('pretrained_only') else
                           cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=spec['weights'])),
                          spec['diameter']) for level, spec in info['levels'].items()}
        settings = {level: {f'c{c:+.1f}_f{f:.1f}': [c, f] for c in m.CELLPROB for f in m.FLOW} for level in chosen}

        def predict(x, values):
            return {level: m.predict(chosen[level][0], x, level, chosen[level][1], *values[level]) for level in values}
        return SimpleNamespace(data=m.DATA, names=m.names, read=lambda s, n: tifffile.imread(m.DATA / s / 'images' / n),
                               levels=(m.args.inner, m.args.outer), settings=settings, predict=predict,
                               own={level: f"c{spec['cellprob_threshold']:+.1f}_f{spec['flow_threshold']:.1f}"
                                    for level, spec in info['levels'].items()})
    raise ValueError(name)


def save(path, labels):
    import numpy as np
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    tifffile.imwrite(path, labels.astype(np.uint16 if labels.max(initial=0) < 65536 else np.uint32),
                     photometric='minisblack', compression='zlib')


def grid(args, scorer_argv):
    """Val predictions at every setting; a method whose settings act per level predicts each level at each setting."""
    run = method(args.method, args.gpu, scorer_argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'settings.json').write_text(json.dumps(dict(method=args.method, scorer_args=scorer_argv, levels=run.levels,
                                                       own=run.own, settings=run.settings), indent=1))
    keys = {level: list(run.settings[level]) for level in run.levels}
    rounds = max(len(k) for k in keys.values())
    for name in run.names('val')[:args.limit or None]:
        x = run.read('val', name)
        for r in range(rounds):                            # setting r of every level at once (both at their r-th value)
            values = {level: run.settings[level][keys[level][min(r, len(keys[level]) - 1)]] for level in run.levels}
            for level, labels in run.predict(x, values).items():
                save(out / 'val' / level / keys[level][min(r, len(keys[level]) - 1)] / name, labels)
        print('val', name, flush=True)


def select(args):
    """The pair of settings with the best objective on the val predictions of grid."""
    import contextlib
    import io
    import itertools
    import numpy as np
    import tifffile
    sys.path.insert(0, str(ROOT))
    from cytonucs_calibration import TAUS, ap_of, match_counts
    from cytonucs_inference import compute_jtpr
    from score_hierarchy import ground_truth, without_ignored
    out = Path(args.out)
    info = json.loads((out / 'settings.json').read_text())
    inner, outer = info['levels']
    key = f'{inner}_{outer}'
    data = Path(args.data) / 'val'
    names = sorted(p.name for p in (data / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))[:args.limit or None]
    counts = {level: {s: np.zeros((len(TAUS), 3), np.int64) for s in info['settings'][level]} for level in (inner, outer)}
    joint = {pair: [0, 0] for pair in itertools.product(info['settings'][inner], info['settings'][outer])}
    for name in names:
        gt, truths, ignore = ground_truth(data, name, [inner, outer], [(inner, outer)])
        pred = {}
        for level in (inner, outer):
            for s in info['settings'][level]:
                labels = tifffile.imread(out / 'val' / level / s / name)
                if ignore and ignore.get(level) is not None:
                    labels = without_ignored(labels, ignore[level])
                pred[level, s] = labels
                counts[level][s] += match_counts(gt[level], labels)
        for s_in, s_out in joint:
            with contextlib.redirect_stdout(io.StringIO()):
                _, details = compute_jtpr(pred[inner, s_in], pred[outer, s_out], gt[inner], gt[outer], iou_thresh=0.5,
                                          assignments_gt=truths[key])
            joint[s_in, s_out][0] += int(details['joint_tp'])
            joint[s_in, s_out][1] += int(details['total_pairs'])
        print('select', name, flush=True)
    table = {}
    for (s_in, s_out), (tp, total) in joint.items():
        ap_in, ap_out = ap_of(counts[inner][s_in]), ap_of(counts[outer][s_out])
        jtpr = tp / max(total, 1)
        table[f'{s_in}|{s_out}'] = dict(objective=0.5 * jtpr + 0.5 * (ap_in + ap_out) / 2, jtpr=jtpr, ap_inner=ap_in, ap_outer=ap_out)
    best = max(table, key=lambda k: table[k]['objective'])
    own = f"{info['own'][inner]}|{info['own'][outer]}"
    s_in, s_out = best.split('|')
    chosen = dict(levels=[inner, outer], chosen={inner: s_in, outer: s_out},
                  values={inner: info['settings'][inner][s_in], outer: info['settings'][outer][s_out]},
                  objective=table[best], own_settings={inner: info['own'][inner], outer: info['own'][outer]},
                  own_objective=table.get(own), val_images=len(names), table=table)
    (out / 'chosen.json').write_text(json.dumps(chosen, indent=1))
    print('chosen', chosen['chosen'], 'objective', round(table[best]['objective'], 4), 'own', chosen['own_settings'],
          round(table[own]['objective'], 4) if own in table else None)


def predict(args, scorer_argv):
    """The splits with the chosen settings, saved for score_predictions.py."""
    run = method(args.method, args.gpu, scorer_argv)
    out = Path(args.out)
    chosen = json.loads((out / 'chosen.json').read_text())
    values = {level: run.settings[level][chosen['chosen'][level]] for level in run.levels}
    for split in args.split:
        for name in run.names(split)[:args.limit or None]:
            for level, labels in run.predict(run.read(split, name), values).items():
                save(out / 'predictions' / split / level / name, labels)
        print('->', out / 'predictions' / split, flush=True)


def main():
    own, scorer_argv = split_argv(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('grid', 'select', 'predict'))
    parser.add_argument('--method', choices=('stardist', 'hsd', 'cellpose'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', default='')
    parser.add_argument('--data', help='select: the dataset (its val split)')
    parser.add_argument('--split', nargs='+', default=['test'])
    parser.add_argument('--limit', type=int, default=0, help='first N images only (smoke tests)')
    args = parser.parse_args(own)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.step == 'grid':
        grid(args, scorer_argv)
    elif args.step == 'select':
        select(args)
    else:
        predict(args, scorer_argv)


if __name__ == '__main__':
    main()
