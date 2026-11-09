"""Cellpose 3D (Stringer et al. 2021; Cellpose 3) as a baseline on the 3D HeLa volume, scored with the CytoNucs
whole-block protocol (evaluate_blocks.py) like run_fullres.py: one model per level, fine-tuned from the pretrained
'cyto3' (cells) and 'nuclei' (nuclei) models as run_cellpose.py does in 2D.

  - training (cellpose env): 2D planes of the train blocks of the source volume (<data>/train_regions with
    split_manifest.json): every XY plane and every --side-step-th XZ and YZ plane, the side planes stretched along Z
    by --anisotropy (Cellpose's 3D recipe: one 2D network for the three orientations); the labels of a plane are
    relabelled by connectivity, so a cell cut twice by a plane gives two objects; Cellpose's train_seg with the
    recipe of run_cellpose.py (SGD, learning rate 0.1, weight decay 1e-4, --epochs x --nimg-per-epoch crops in
    batches of --batch); planes and flows are cached in <basedir>/<name>/cache during training and removed after;
  - inference: every held-out block over the full Z range with --margin px of context; the network runs once per
    window and orientation mode, and the masks of every setting come from its flows: 3D flows (do_3D with
    --anisotropy) or 2D planes stitched across Z (stitch_threshold 0.25 / 0.5), cellprob threshold -1 / 0 / 1; the
    setting of each level is the one with the best pooled AP over IoU 0.5..0.95 on the val blocks (objects by
    centroid, as evaluate_blocks.py), also over two object sizes: the diameter of the training labels (Cellpose's,
    a mean over planes that cut most objects off-centre) and the median diameter of the largest XY section of the
    3D objects of the train blocks;
  - predictions: <basedir>/<name>/predictions/<split>/<level>/<y0>_<x0>.tif (labels of the window) and
    <basedir>/<name>/training.json (settings, windows, seconds per window and level);
  - scoring (CytoNucs env, --score): AP, AP50, recall and precision at 0.5 per level and JTPR (predicted nuclei paired
    with predicted cells by overlap, as cytonucs_inference.compute_jtpr does without predicted pairs) on the val and
    test blocks, as run_fullres.py --score; written to <basedir>/<name>/blocks.json.

Usage:
  python run_cellpose3d.py --name cp3d_hela --gpu 0          (cellpose env)
  python run_cellpose3d.py --name cp3d_hela --score          (CytoNucs env)
  python run_cellpose3d.py --name cp3d_hela --gpu 0 --reuse-models   (settings and predictions again, same models)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data', default='/storage02/miroslavm/data/Cytonucs_Emese_Hela/patches_split_hela')
parser.add_argument('--name', required=True)
parser.add_argument('--gpu', default='0')
parser.add_argument('--cell-channels', type=int, nargs=2, default=[2, 3], help='Cellpose channels [cytoplasm, nucleus] (1=R 2=G 3=B)')
parser.add_argument('--nucleus-channel', type=int, default=3)
parser.add_argument('--anisotropy', type=float, default=5.0, help='Z spacing / XY spacing of the volume')
parser.add_argument('--side-step', type=int, default=8, help='every n-th XZ and YZ plane of a train block is a training image')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--nimg-per-epoch', type=int, default=800)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--margin', type=int, default=192)
parser.add_argument('--score', action='store_true', help='score the saved predictions (CytoNucs env)')
parser.add_argument('--reuse-models', action='store_true',
                    help='skip the training of a level whose model is in <basedir>/<name>/models (its training hours '
                         'and planes are kept from training.json); the settings are chosen and the blocks predicted again')
parser.add_argument('--basedir', default='/storage02/miroslavm/cytonucs_paper_runs/cellpose')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.score else args.gpu
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import tifffile
from evaluate_blocks import block_windows, objects_in_block
from score_native_resolution import TAUS

UNKNOWN = 65535
MASKS = dict(nucleus='nucleus_masks', cytoplasm='cytoplasm_masks')
DATA, OUT = Path(args.data), Path(args.basedir) / args.name
LEVELS = {'cytoplasm': dict(pretrained='cyto3', channels=list(args.cell_channels)),
          'nucleus': dict(pretrained='nuclei', channels=[args.nucleus_channel, 0])}
SETTINGS = [('3d', 0.0, t) for t in (-1.0, 0.0, 1.0)] + [('stitch', s, t) for s in (0.25, 0.5) for t in (-1.0, 0.0, 1.0)]


def volume_name():
    return next(iter(json.loads((DATA / 'train_regions' / 'regions.json').read_text())))


def manifest():
    return json.loads((DATA / 'split_manifest.json').read_text())


def read_volume(level=None):
    path = DATA / 'train_regions' / ('images' if level is None else MASKS[level]) / volume_name()
    array = tifffile.imread(path)
    return array if level is None else np.where(array == UNKNOWN, 0, array)


def train_planes(level, cache, device=None):
    """Images and labels of the training planes written to cache (once); returns (image files, flow files)."""
    from scipy.ndimage import zoom
    from skimage.measure import label
    from cellpose import dynamics
    cache.mkdir(parents=True, exist_ok=True)
    done = cache / 'planes.json'
    if not done.is_file():
        image, labels = read_volume(), read_volume(level)
        names = []
        stretch = (args.anisotropy, 1.0)
        for b in (b for v in manifest()['volumes'] for b in v['blocks'] if b['split'] == 'train'):
            x, y = image[:, b['y0']:b['y1'], b['x0']:b['x1']], labels[:, b['y0']:b['y1'], b['x0']:b['x1']]
            planes = [(f'xy{z}', x[z], y[z]) for z in range(x.shape[0])]
            for k in range(0, x.shape[1], args.side_step):
                planes.append((f'xz{k}', zoom(x[:, k], stretch + (1.0,), order=1), zoom(y[:, k], stretch, order=0)))
            for k in range(0, x.shape[2], args.side_step):
                planes.append((f'yz{k}', zoom(x[:, :, k], stretch + (1.0,), order=1), zoom(y[:, :, k], stretch, order=0)))
            chunk_names, chunk_labels = [], []
            for tag, xi, yi in planes:
                name = f"b{b['y0']}_{b['x0']}_{tag}.tif"
                tifffile.imwrite(cache / name, np.ascontiguousarray(xi), photometric='rgb')
                chunk_names.append(str(cache / name))
                chunk_labels.append(label(yi, background=0, connectivity=1).astype(np.int32))
            for start in range(0, len(chunk_names), 64):
                dynamics.labels_to_flows(chunk_labels[start:start + 64], files=chunk_names[start:start + 64],
                                         device=device, return_flows=False)
            names += chunk_names
        done.write_text(json.dumps(names))
    names = json.loads(done.read_text())
    return names, [str(Path(n).with_name(Path(n).stem + '_flows.tif')) for n in names]


def object_diameter(labels):
    """Median over the 3D objects of the train blocks of the equivalent diameter of their largest XY section."""
    from scipy import ndimage as ndi
    sizes = []
    for b in (b for v in manifest()['volumes'] for b in v['blocks'] if b['split'] == 'train'):
        y = labels[:, b['y0']:b['y1'], b['x0']:b['x1']]
        for k, box in enumerate(ndi.find_objects(y), 1):
            if box is not None:
                sizes.append(max(np.count_nonzero(plane == k) for plane in y[box]))
    return float(2 * np.sqrt(np.median(sizes) / np.pi))


def drop_cache(level):
    import shutil
    cache = (OUT / 'cache' / level).resolve()
    if cache.parent == (OUT / 'cache').resolve() and cache.is_dir():
        shutil.rmtree(cache)


def flows(model, x, level, diameter, mode):
    """Network output of a window: (dP, cellprob) of the 3D run or of the planes."""
    kw = dict(channels=LEVELS[level]['channels'], channel_axis=3, z_axis=0, diameter=diameter, compute_masks=False)
    if mode == '3d':
        out = model.eval(x, do_3D=True, anisotropy=args.anisotropy, **kw)[1]
    else:
        out = model.eval(x, do_3D=False, stitch_threshold=0.5, **kw)[1]     # > 0: normalised over the stack
    return out[1], out[2]


def masks_of(model, output, mode, stitch, cellprob, diameter):
    from cellpose import dynamics, utils
    dP, prob = output
    niter = int(round(200 * diameter / model.diam_mean))                       # Cellpose's 200 / rescale
    device = model.device if model.gpu else None
    if mode == '3d':
        return dynamics.resize_and_compute_masks(dP, prob, niter=niter, cellprob_threshold=cellprob, do_3D=True,
                                                 min_size=15, resize=None, device=device)[0]
    planes = np.array([dynamics.resize_and_compute_masks(dP[:, i], prob[i], niter=niter, cellprob_threshold=cellprob,
                                                         flow_threshold=0.4, min_size=-1, device=device)[0]
                       for i in range(prob.shape[0])])
    return utils.fill_holes_and_remove_small_masks(utils.stitch3D(planes, stitch_threshold=stitch), min_size=15)


def pooled_ap(truth, predicted):
    import fastremap
    from cellpose import metrics
    # Cellpose counts the objects of a mask by its largest label: the block objects keep the labels of the window
    # (of the whole volume for the truth), so they are renumbered 1..n first.
    truth, predicted = ([fastremap.renumber(m, in_place=False)[0] for m in masks] for masks in (truth, predicted))
    _, tp, fp, fn = metrics.average_precision(truth, predicted, threshold=TAUS)
    tp, fp, fn = (np.asarray(v).sum(axis=0) for v in (tp, fp, fn))
    return float(np.mean(tp / np.maximum(tp + fp + fn, 1)))


def main():
    import torch
    from cellpose import models, train
    from importlib.metadata import version
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    image = read_volume()
    windows = {split: list(block_windows(manifest(), split, args.margin)) for split in ('val', 'test')}
    info = dict(name=args.name, data=str(DATA), anisotropy=args.anisotropy, side_step=args.side_step, levels={},
                windows={s: [dict(block=list(b), window=list(w)) for _, b, w in ws] for s, ws in windows.items()},
                cellpose=version('cellpose'), torch=torch.__version__, seconds={})
    previous = json.loads((OUT / 'training.json').read_text())['levels'] if (OUT / 'training.json').is_file() else {}
    for level, spec in LEVELS.items():
        started = time.time()
        torch.cuda.reset_peak_memory_stats() if gpu else None
        path = OUT / 'models' / level
        reused = args.reuse_models and path.is_file() and level in previous
        if reused:
            hours, planes = previous[level]['training_hours'], previous[level]['training_planes']
        else:
            model = models.CellposeModel(gpu=gpu, model_type=spec['pretrained'])
            files, flow_files = train_planes(level, OUT / 'cache' / level, model.device)
            planes = len(files)
            if len(files) < args.nimg_per_epoch:
                repeat = -(-args.nimg_per_epoch // len(files))                   # see run_cellpose.py
                files, flow_files = files * repeat, flow_files * repeat
            path = train.train_seg(model.net, train_files=files, train_labels_files=flow_files, load_files=False,
                                   channels=spec['channels'], normalize=True, n_epochs=args.epochs,
                                   nimg_per_epoch=args.nimg_per_epoch, batch_size=args.batch, learning_rate=0.1,
                                   weight_decay=1e-4, SGD=True, min_train_masks=0, save_path=str(OUT), model_name=level)
            drop_cache(level)
            path = path[0] if isinstance(path, (tuple, list)) else path
            hours = (time.time() - started) / 3600
        model = models.CellposeModel(gpu=gpu, pretrained_model=str(path))
        labels = read_volume(level)
        diameters = dict(labels=float(model.diam_labels), objects=object_diameter(labels))
        # Val blocks: every setting from one network run per window, size and mode.
        settings = [(d,) + s for d in diameters for s in SETTINGS]
        truth, predicted = [], {s: [] for s in settings}
        for _, block, (y0, y1, x0, x1) in windows['val']:
            x = image[:, y0:y1, x0:x1]
            truth.append(objects_in_block(labels[:, y0:y1, x0:x1].astype(np.int32), block, (y0, y1, x0, x1)))
            for size, diameter in diameters.items():
                for mode in ('3d', 'stitch'):
                    output = flows(model, x, level, diameter, mode)
                    for setting in (s for s in settings if s[:2] == (size, mode)):
                        pred = masks_of(model, output, *setting[1:], diameter)
                        predicted[setting].append(objects_in_block(np.asarray(pred, np.int32), block, (y0, y1, x0, x1)))
        scores = {s: pooled_ap(truth, predicted[s]) for s in settings}
        size, *best = max(scores, key=scores.get)
        diameter = diameters[size]
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if gpu else None
        if reused and peak is not None:
            peak = max(peak, previous[level]['gpu_peak_mem_gb'] or 0.0)             # the training's peak
        info['levels'][level] = dict(pretrained=spec['pretrained'], channels=spec['channels'], training_hours=hours,
                                     gpu_peak_mem_gb=peak, diameters=diameters, diameter=diameter, mode=best[0], stitch_threshold=best[1],
                                     cellprob_threshold=best[2], val_ap={'_'.join(map(str, s)): v for s, v in scores.items()},
                                     training_planes=planes, weights=str(path))
        print(level, json.dumps({k: v for k, v in info['levels'][level].items() if k != 'val_ap'}), flush=True)
        for split, ws in windows.items():
            for _, block, (y0, y1, x0, x1) in ws:
                started = time.time()
                pred = masks_of(model, flows(model, image[:, y0:y1, x0:x1], level, diameter, best[0]), *best, diameter)
                info['seconds'][f'{split}/{level}/{y0}_{x0}'] = time.time() - started
                (OUT / 'predictions' / split / level).mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(OUT / 'predictions' / split / level / f'{y0}_{x0}.tif', np.asarray(pred).astype(np.uint16),
                                 photometric='minisblack', compression='zlib')
        del model
    (OUT / 'training.json').write_text(json.dumps(info, indent=2, default=str))
    print('->', OUT / 'predictions')


def score():
    """Metrics of the saved predictions on the val and test blocks, as run_fullres.py --score."""
    import contextlib
    import io
    from stardist.matching import matching
    from cytonucs_inference import compute_jtpr
    from cytonucs_runtime import accuracy_profile
    info = json.loads((OUT / 'training.json').read_text())
    gt_all = {t: read_volume(t) for t in MASKS}
    pairs = {int(k): int(v) for k, v in json.loads((DATA / 'train_regions' / 'assignments' /
                                                    (Path(volume_name()).stem + '.json')).read_text()).items()}
    seconds = info['seconds']
    results = {}
    for split in ('test', 'val'):
        counts = {t: np.zeros((3, len(TAUS)), np.int64) for t in MASKS}
        joint, objects, timing = [0, 0], {t: 0 for t in MASKS}, []
        fails = dict(fail_nucleus=0, fail_cell=0, fail_link=0)    # why annotated pairs are no joint TP
        for entry in info['windows'][split]:
            block, window = tuple(entry['block']), tuple(entry['window'])
            y0, y1, x0, x1 = window
            gt = {t: objects_in_block(gt_all[t][:, y0:y1, x0:x1].astype(np.int32), block, window) for t in MASKS}
            pr = {t: objects_in_block(tifffile.imread(OUT / 'predictions' / split / t / f'{y0}_{x0}.tif').astype(np.int32),
                                      block, window) for t in MASKS}
            for t in MASKS:
                objects[t] += int(len(np.unique(gt[t])) - 1)
                for k, r in enumerate(matching(gt[t], pr[t], thresh=TAUS, criterion='iou')):
                    counts[t][:, k] += (int(r.tp), int(r.fp), int(r.fn))
                timing.append(dict(target=t, block=list(block), seconds=round(seconds[f'{split}/{t}/{y0}_{x0}'], 1)))
            with contextlib.redirect_stdout(io.StringIO()):
                _, details = compute_jtpr(pr['nucleus'], pr['cytoplasm'], gt['nucleus'], gt['cytoplasm'], iou_thresh=.5,
                                          assignments_gt=pairs)
            joint[0] += int(details['joint_tp']); joint[1] += int(details['total_pairs'])
            for key in fails:
                fails[key] += int(details['joint_' + key])
        result = dict(jtpr=joint[0] / max(joint[1], 1), pairs=joint[1], objects=objects, timing=timing, **fails)
        for t in MASKS:
            tp, fp, fn = counts[t]
            ap = accuracy_profile(tp, fp, fn)
            result[f'ap_{t}'], result[f'ap50_{t}'] = float(ap.mean()), float(ap[0])
            result[f'recall50_{t}'] = float(tp[0] / max(tp[0] + fn[0], 1))
            result[f'precision50_{t}'] = float(tp[0] / max(tp[0] + fp[0], 1))
        results[split] = result
        print(f"{split:5s} | nuclei {objects['nucleus']}: AP {result['ap_nucleus']:.3f} (AP50 {result['ap50_nucleus']:.3f}, "
              f"recall {result['recall50_nucleus']:.2f}) | cells {objects['cytoplasm']}: AP {result['ap_cytoplasm']:.3f} "
              f"(AP50 {result['ap50_cytoplasm']:.3f}, recall {result['recall50_cytoplasm']:.2f}) | JTPR {result['jtpr']:.3f} of {joint[1]} pairs")
    (OUT / 'blocks.json').write_text(json.dumps(results, indent=1))
    print('->', OUT / 'blocks.json')


if __name__ == '__main__':
    score() if args.score else main()
