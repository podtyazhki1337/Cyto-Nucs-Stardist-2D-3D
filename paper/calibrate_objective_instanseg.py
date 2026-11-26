"""InstanSeg at the operating point of CytoNucs' calibration, as paper/calibrate_objective.py does for StarDist,
HydraStarDist and Cellpose: the setting with the best 1/2 linked JTPR at IoU 0.5 + 1/2 mean AP of the two levels on
the val images, then the test splits with it.

InstanSeg predicts both levels with one pair of post-processing thresholds (seed_threshold, mask_threshold) and links
them itself (a nucleus and its cell share a label; written to parents/), so the search is over that pair, for both
levels together, with InstanSeg's own links. Grid: SEEDS x MASKS, which holds the grid of
baselines/instanseg/run_instanseg.py (seed 0.5 / 0.7 / 0.9 x mask 0.4 / 0.53, its defaults 0.7 / 0.53).
The run's data, channels and pixel size are those of its training.json. The model is the run's own TorchScript
(model/instanseg.pt), loaded as it is: nothing is written into the run's folder.

  grid     (instanseg env, GPU) val at every setting -> OUT/grid/s<seed>_m<mask>/val/{nucleus,cytoplasm,parents}
  select   (CytoNucs env, CPU) score_predictions.py on every setting, the objective -> OUT/chosen.json (own = the run's
           thresholds in training.json; the own setting's val AP is checked against training.json)
  predict  (instanseg env, GPU) the --split splits at the chosen setting -> OUT/predictions/<split>/...; then
           score_predictions.py on them
Usage (from the repository root; --basedir and --run as the --basedir and --name of run_instanseg.py):
  python paper/calibrate_objective_instanseg.py STEP --basedir runs/instanseg --run NAME --out OUT [--gpu 0]
      [--split test ...] [--workers 4] [--limit N]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                   # the repository root
RUNNER = ROOT / 'baselines' / 'instanseg' / 'run_instanseg.py'
SEEDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MASKS = (0.3, 0.4, 0.45, 0.53, 0.6)


def tag(seed, mask):
    return f's{seed:.2f}_m{mask:.2f}'


def runner(args, split):
    """run_instanseg.py imported with the run's own arguments (its module-level setup reads them)."""
    info = json.loads((Path(args.basedir) / args.run / 'training.json').read_text())
    argv = ['--data', info['data'], '--name', args.run, '--gpu', args.gpu, '--basedir', str(args.basedir),
            '--nucleus-channel', str(info['channels'][0]), '--cytoplasm-channel', str(info['channels'][1]),
            '--split', *split] + (['--pixel-size', str(info['pixel_size'])] if info['pixel_size'] else [])
    argv += ['--limit', str(args.limit)] if args.limit else []
    sys.argv = [str(RUNNER)] + argv
    sys.path.insert(0, str(RUNNER.parent))
    spec = importlib.util.spec_from_file_location('run_instanseg', RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import torch
    from instanseg import InstanSeg
    model = InstanSeg(torch.jit.load(str(Path(args.basedir) / args.run / 'model' / 'instanseg.pt'), map_location='cpu'), verbosity=0)
    return m, model, info


def write(m, root, name, nuclei, cells, links):
    import numpy as np
    import tifffile
    for level, lab in zip(m.LEVELS, (nuclei, cells)):
        assert lab.max() < m.UNKNOWN, f'{name}: {lab.max()} labels'
        tifffile.imwrite(root / level / name, lab.astype(np.uint16), photometric='minisblack', compression='zlib')
    (root / 'parents' / (Path(name).stem + '.json')).write_text(json.dumps(links))


def predict_into(m, model, info, setting, split, root):
    for d in m.LEVELS + ['parents']:
        (root / d).mkdir(parents=True, exist_ok=True)
    started = time.time()
    names = m.names(split)
    for name in names:
        write(m, root, name, *m.predict(model, m.image(split, name), info['pixel_size'], *setting))
    return (time.time() - started) / max(len(names), 1)


def grid(args):
    m, model, info = runner(args, ['val'])
    out = Path(args.out)
    for seed in SEEDS:
        for mask in MASKS:
            folder = out / 'grid' / tag(seed, mask)
            if (folder / 'done').is_file():
                continue
            s = predict_into(m, model, info, (seed, mask), 'val', folder / 'val')
            (folder / 'done').write_text(json.dumps(dict(seconds_per_image=s)))
            print('val', tag(seed, mask), f'{s:.3f} s per image', flush=True)


def objective(d):
    return 0.5 * d['jtpr_nucleus_cytoplasm'] + 0.5 * (d['ap_nucleus'] + d['ap_cytoplasm']) / 2


def select(args):
    out = Path(args.out)
    info = json.loads((Path(args.basedir) / args.run / 'training.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    settings = [(seed, mask) for seed in SEEDS for mask in MASKS]

    def score(setting):
        folder = out / 'grid' / tag(*setting)
        target = folder / 'val.json'
        if not target.is_file():
            subprocess.run([sys.executable, str(ROOT / 'score_predictions.py'), '--data', info['data'], '--predictions', str(folder),
                            '--split', 'val', '--output', str(target)], check=True, env=env, stdout=subprocess.DEVNULL)
        return setting, json.loads(target.read_text())['val']

    with ThreadPoolExecutor(args.workers) as pool:
        scores = dict(pool.map(score, settings))
    row = lambda s: dict(seed_threshold=s[0], mask_threshold=s[1], objective=objective(scores[s]),       # noqa: E731
                         ap_nucleus=scores[s]['ap_nucleus'], ap_cytoplasm=scores[s]['ap_cytoplasm'],
                         jtpr=scores[s]['jtpr_nucleus_cytoplasm'])
    own = (info['seed_threshold'], info['mask_threshold'])
    best = max(settings, key=lambda s: (objective(scores[s]), s == own))
    check = info['val_ap'].get(f"{info['pixel_size']}_{own[0]}_{own[1]}")
    result = dict(run=args.run, data=info['data'], own=row(own), chosen=row(best), kept_own=best == own,
                  own_val_ap_training=check, grid=[row(s) for s in settings])
    (out / 'chosen.json').write_text(json.dumps(result, indent=1))
    print(f"{args.run}: own {own} objective {row(own)['objective']:.4f} -> chosen {best} {row(best)['objective']:.4f}; "
          f"own val AP here {row(own)['ap_nucleus']:.4f} / {row(own)['ap_cytoplasm']:.4f}, training.json "
          f"(pooled AP of run_instanseg.py, another average) {check}", flush=True)


def predict(args):
    out = Path(args.out)
    chosen = json.loads((out / 'chosen.json').read_text())['chosen']
    m, model, info = runner(args, args.split)
    for split in args.split:
        s = predict_into(m, model, info, (chosen['seed_threshold'], chosen['mask_threshold']), split,
                         out / 'predictions' / split)
        print(split, f'{s:.3f} s per image', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('grid', 'select', 'predict'))
    parser.add_argument('--basedir', default='runs/instanseg', help='the folder of the InstanSeg runs')
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--split', nargs='+', default=['test'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    dict(grid=grid, select=select, predict=predict)[args.step](args)


if __name__ == '__main__':
    main()
