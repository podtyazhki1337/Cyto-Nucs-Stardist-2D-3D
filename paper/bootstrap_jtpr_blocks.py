"""Test blocks of a spatially split 3D volume (the HeLa spheroid): per-block AP and JTPR of every method, and bootstrap
intervals of JTPR (IoU 0.5) over the annotated pairs of all test blocks, with the paired difference of the first method
minus each other one. 10 000 resamples of the pairs, the same for all methods.

Labels of a method: the --save folder of evaluate_blocks.py (CytoNucs, with its predicted links) or of
baselines/stardist3d/run_fullres.py --score (paired by overlap, as its scorer), or a Cellpose 3D run folder of
baselines/cellpose/run_cellpose3d.py (its saved window predictions; the objects of each block as its --score).

Usage:
  python paper/bootstrap_jtpr_blocks.py --native data/Cytonucs_Emese_Hela/patches_split_hela --output OUT.json \\
      --saved "CytoNucs=SAVE_DIR" --saved "StarDist-3D=SAVE_DIR" [--cellpose3d "Cellpose 3D=RUN_DIR"]
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from stardist.matching import matching

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repository root: the CytoNucs modules
from cytonucs_inference import compute_jtpr                      # noqa: E402
from cytonucs_runtime import accuracy_profile                    # noqa: E402
from evaluate_blocks import objects_in_block                     # noqa: E402
from score_native_resolution import TAUS                         # noqa: E402

LEVELS = ('nucleus', 'cytoplasm')


def saved(folder):
    """(block key, gt, pred, links) of every test block in a --save folder."""
    folder = Path(folder)
    for f in sorted(folder.glob('test_*_nucleus_gt.tiff')):
        stem = f.name[:-len('_nucleus_gt.tiff')]
        gt = {l: tifffile.imread(folder / f'{stem}_{l}_gt.tiff').astype(np.int32) for l in LEVELS}
        pr = {l: tifffile.imread(folder / f'{stem}_{l}_pred.tiff').astype(np.int32) for l in LEVELS}
        lf = folder / f'{stem}_links.json'
        links = {int(k): int(v) for k, v in json.loads(lf.read_text()).items()} if lf.is_file() else None
        yield stem.split('_y')[-1], gt, pr, links


def cellpose3d(run, native, name):
    """(block key, gt, pred, None) of every test block of a run_cellpose3d.py run folder."""
    run = Path(run)
    info = json.loads((run / 'training.json').read_text())
    masks = {l: tifffile.imread(native / 'train_regions' / f'{l}_masks' / name) for l in LEVELS}
    masks = {l: np.where(v == 65535, 0, v) for l, v in masks.items()}
    for entry in info['windows']['test']:
        block, window = tuple(entry['block']), tuple(entry['window'])
        y0, y1, x0, x1 = window
        gt = {l: objects_in_block(masks[l][:, y0:y1, x0:x1].astype(np.int32), block, window) for l in LEVELS}
        pr = {l: objects_in_block(tifffile.imread(run / 'predictions/test' / l / f'{y0}_{x0}.tif').astype(np.int32),
                                  block, window) for l in LEVELS}
        yield f'{block[0]:04d}_x{block[2]:04d}', gt, pr, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--native', required=True, help='the spatially split dataset (with train_regions/)')
    parser.add_argument('--saved', action='append', default=[], metavar='NAME=SAVE_DIR')
    parser.add_argument('--cellpose3d', action='append', default=[], metavar='NAME=RUN_DIR')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    native = Path(args.native)
    name = next(iter(json.loads((native / 'train_regions/regions.json').read_text())))
    pairs_gt = {int(k): int(v) for k, v in
                json.loads((native / 'train_regions/assignments' / (Path(name).stem + '.json')).read_text()).items()}
    methods = {k: saved(v) for k, v in (s.split('=', 1) for s in args.saved)}
    methods.update({k: cellpose3d(v, native, name) for k, v in (s.split('=', 1) for s in args.cellpose3d)})
    hits, per_block = {}, {}
    for m, blocks in methods.items():
        hits[m] = {}
        per_block[m] = []
        for key, gt, pr, links in blocks:
            with contextlib.redirect_stdout(io.StringIO()):
                _, det = compute_jtpr(pr['nucleus'], pr['cytoplasm'], gt['nucleus'], gt['cytoplasm'], iou_thresh=.5,
                                      assignments_pred=links, assignments_gt=pairs_gt, with_pairs=True)
            joint = {n for n, _ in det['joint_pairs']}
            present = set(int(v) for v in np.unique(gt['nucleus']) if v)
            cells = set(int(v) for v in np.unique(gt['cytoplasm']) if v)
            for n in [n for n, c in pairs_gt.items() if n in present and c in cells]:
                hits[m][(key, n)] = int(n in joint)
            row = dict(block=key, pairs=int(det['total_pairs']), jtpr=det['joint_tp'] / max(det['total_pairs'], 1))
            for l in LEVELS:
                c = np.array([[r.tp, r.fp, r.fn] for r in matching(gt[l], pr[l], thresh=TAUS, criterion='iou')]).T
                row[f'ap_{l}'] = float(accuracy_profile(*c).mean())
            per_block[m].append(row)
    keys = sorted(set.intersection(*(set(h) for h in hits.values())))
    assert all(len(h) == len(keys) for h in hits.values()), {m: len(h) for m, h in hits.items()}
    H = {m: np.array([hits[m][k] for k in keys]) for m in hits}
    n = len(keys)
    idx = np.random.default_rng(0).integers(0, n, (10000, n))
    ci = lambda s: [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]    # noqa: E731
    first = next(iter(H))
    out = dict(pairs=n, resamples=10000, per_block=per_block, methods={}, differences={})
    for m, v in H.items():
        out['methods'][m] = dict(jtpr=float(v.mean()), joint_tp=int(v.sum()), ci95=ci(v[idx].mean(1)))
    for m in list(H)[1:]:
        d = H[first] - H[m]
        b = d[idx].mean(1)
        out['differences'][f'{first} - {m}'] = dict(
            diff=float(d.mean()), ci95=ci(b), share_above_0=float((b > 0).mean()),
            pairs_only_first=int(((H[first] == 1) & (H[m] == 0)).sum()), pairs_only_other=int(((H[first] == 0) & (H[m] == 1)).sum()))
    Path(args.output).write_text(json.dumps(out, indent=1))
    for m, rows in per_block.items():
        print(m, ' | '.join(f"block {r['block']}: AP {r['ap_nucleus']:.3f} / {r['ap_cytoplasm']:.3f}, JTPR {r['jtpr']:.3f} "
                            f"of {r['pairs']}" for r in rows))
    for m, r in out['methods'].items():
        print(f"{m}: JTPR {r['jtpr']:.3f} ({r['joint_tp']}/{n}) CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
    for k, r in out['differences'].items():
        print(f"{k}: {r['diff']:+.3f} CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] P(>0) {r['share_above_0']:.3f} "
              f"discordant {r['pairs_only_first']}/{r['pairs_only_other']}")


if __name__ == '__main__':
    main()
