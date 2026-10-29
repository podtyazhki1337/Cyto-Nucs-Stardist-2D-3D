"""Cellpose 3D (Stringer et al. 2021; Cellpose 3) as a baseline on a dataset of one file per volume (KiTS23 in the
CytoNucs conversion), the counterpart of run_cellpose3d.py (one HeLa volume in blocks) with the same recipe: one model
per level, fine-tuned from the pretrained --pretrained model (default 'cyto3', the generalist: CT is one grey channel).

  - data: <data>/{train,val,test}/images/<name> and <data>/<split>/<level>_masks/<name> (unknown label 65535 is
    background);
  - training (cellpose env): 2D planes of the train volumes that hold objects of the level: every such XY plane and
    every --side-step-th such XZ and YZ plane, the side planes stretched along Z by --anisotropy (Cellpose's 3D
    recipe: one 2D network for the three orientations); the labels of a plane are relabelled by connectivity, so an
    object cut twice by a plane gives two objects; planes without objects are left out (Cellpose would enlarge them
    6x, as the diameter of an empty plane is set to 5 px), the background around the objects still fills most crops;
    sections below 4 px are background;
    Cellpose's train_seg with the recipe of run_cellpose.py (SGD, learning rate 0.1, weight decay 1e-4, --epochs x
    --nimg-per-epoch crops in batches of --batch); planes and their flows (zlib) are cached in --cache during
    training and removed after;
  - every predicted object below the size rule of the labels (--min-mm3, dataset_prep/kits23_to_cytonucs.py; as
    baselines/nnunet/to_instances.py for nnU-Net) is removed;
  - settings: the network runs once per volume, object size and mode, and the masks of every setting come from its
    flows: 3D flows (do_3D with --anisotropy) or 2D planes stitched across Z (stitch_threshold 0.25 / 0.5), cellprob
    threshold -1 / 0 / 1; the setting of each level is the one with the best pooled AP over IoU 0.5..0.95 on the val
    volumes of --select-data (the val16 view, as the CytoNucs checkpoints), also over two object sizes: the diameter
    of the training labels (Cellpose's, a mean over planes) and the median diameter of the largest XY section of the
    3D objects of the train volumes;
  - predictions: <basedir>/<name>/predictions/<split>/<level>/<name> (each level as Cellpose gives it) and
    <basedir>/<name>/predictions_in_outer/<split>/... (inner objects kept when at least the share chosen on the val16
    volumes, of 0 / 0.1 / 0.5, of their voxels lie in predicted outer objects: the hierarchy that nnU-Net's labels
    give by construction); <basedir>/<name>/training.json (settings, val AP of every setting) and timing.json
    (seconds per '<split>/<name>', both levels), for score_predictions.py --levels INNER OUTER --timing.

Usage (cellpose env):
  python run_cellpose3d_files.py --name cp3d_kits --gpu 0
  python run_cellpose3d_files.py --name cp3d_kits --gpu 0 --reuse-models   (settings and predictions again, same models)
"""
import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data', default='/storage02/miroslavm/data/KiTS23/cytonucs_3x078')
parser.add_argument('--select-data', default='/storage02/miroslavm/data/KiTS23/cytonucs_3x078_val16',
                    help='dataset whose val volumes choose the settings')
parser.add_argument('--levels', nargs=2, default=['mass', 'kidney'], help='inner level first')
parser.add_argument('--min-mm3', type=float, nargs=2, default=[32., 1600.], help='smallest object of each level')
parser.add_argument('--name', required=True)
parser.add_argument('--gpu', default='0')
parser.add_argument('--pretrained', default='cyto3')
parser.add_argument('--anisotropy', type=float, help='Z spacing / XY spacing (default from <data>/kits23.json)')
parser.add_argument('--side-step', type=int, default=16, help='every n-th XZ and YZ plane of a train volume is a training image')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--nimg-per-epoch', type=int, default=800)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--splits', nargs='+', default=['val', 'test'])
parser.add_argument('--limit', type=int, help='first N volumes of every split (smoke tests)')
parser.add_argument('--reuse-models', action='store_true',
                    help='skip the training of a level whose model is in <basedir>/<name>/models (its training hours '
                         'and planes are kept from training.json); the settings are chosen and the volumes predicted again')
parser.add_argument('--basedir', default='/storage02/miroslavm/cytonucs_paper_runs/cellpose')
parser.add_argument('--cache', help='folder of the training planes (default /storage01/miroslavm/cellpose_cytonucs/cache/<name>)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import tifffile

UNKNOWN = 65535
TAUS = np.arange(0.5, 1.0, 0.05)                                       # as score_hierarchy.TAUS
DATA, SELECT, OUT = Path(args.data), Path(args.select_data), Path(args.basedir) / args.name
CACHE = Path(args.cache or f'/storage01/miroslavm/cellpose_cytonucs/cache/{args.name}')
SPACING = json.loads((DATA / 'kits23.json').read_text())['spacing'] if (DATA / 'kits23.json').is_file() else None
ANISOTROPY = args.anisotropy or SPACING[0] / SPACING[1]
VOXEL_MM3 = float(np.prod(SPACING)) if SPACING else 1.0
INNER, OUTER = args.levels
MIN_VOXELS = {level: mm3 / VOXEL_MM3 for level, mm3 in zip(args.levels, args.min_mm3)}
CHANNELS = [0, 0]
SETTINGS = [('3d', 0.0, t) for t in (-1.0, 0.0, 1.0)] + [('stitch', s, t) for s in (0.25, 0.5) for t in (-1.0, 0.0, 1.0)]
SHARES = (0.0, 0.1, 0.5)


def names(data, split):
    found = sorted(p.name for p in (data / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    return found[:args.limit] if args.limit else found


def read_labels(data, split, level, name):
    labels = tifffile.imread(data / split / f'{level}_masks' / name)
    return np.where(labels == UNKNOWN, 0, labels).astype(np.int32)


def sections_of(plane):
    """Objects of a plane by connectivity; sections below 4 px are background (no flow to learn; Cellpose's GPU flows
    fail on a plane of one labelled pixel)."""
    from skimage.measure import label
    sections = label(plane, background=0, connectivity=1).astype(np.int32)
    sizes = np.bincount(sections.ravel())
    sections[np.isin(sections, np.flatnonzero(sizes < 4))] = 0
    return sections


def train_planes(level, cache, device=None):
    """Training planes and their flows written to cache (once); returns (image files, flow files, object diameter)."""
    from scipy import ndimage as ndi
    from cellpose import dynamics
    cache.mkdir(parents=True, exist_ok=True)
    done = cache / 'planes.json'
    if not done.is_file():
        files, sections = [], []
        stretch = (ANISOTROPY, 1.0)
        for name in names(DATA, 'train'):
            image, labels = tifffile.imread(DATA / 'train' / 'images' / name), read_labels(DATA, 'train', level, name)
            for k, box in enumerate(ndi.find_objects(labels), 1):
                if box is not None:
                    sections.append(max(np.count_nonzero(plane == k) for plane in labels[box]))
            planes = [(f'xy{z}', image[z], labels[z]) for z in range(image.shape[0]) if labels[z].any()]
            for k in range(0, image.shape[1], args.side_step):
                if labels[:, k].any():
                    planes.append((f'xz{k}', ndi.zoom(image[:, k], stretch, order=1), ndi.zoom(labels[:, k], stretch, order=0)))
            for k in range(0, image.shape[2], args.side_step):
                if labels[:, :, k].any():
                    planes.append((f'yz{k}', ndi.zoom(image[:, :, k], stretch, order=1), ndi.zoom(labels[:, :, k], stretch, order=0)))
            planes = [(tag, x, sections_of(y)) for tag, x, y in planes]
            planes = [(tag, x, y) for tag, x, y in planes if y.any()]
            for start in range(0, len(planes), 64):
                chunk = planes[start:start + 64]
                flows = dynamics.labels_to_flows([y for *_, y in chunk], device=device)
                for (tag, x, _), flow in zip(chunk, flows):
                    path = cache / f'{Path(name).stem}_{tag}.tif'
                    tifffile.imwrite(path, np.ascontiguousarray(x))
                    tifffile.imwrite(path.with_name(path.stem + '_flows.tif'), flow, compression='zlib')
                    files.append(str(path))
            print(level, name, len(planes), 'planes', flush=True)
        done.write_text(json.dumps(dict(files=files, object_diameter=float(2 * np.sqrt(np.median(sections) / np.pi)))))
    planes = json.loads(done.read_text())
    files = planes['files']
    return files, [str(Path(n).with_name(Path(n).stem + '_flows.tif')) for n in files], planes['object_diameter']


def drop_cache(cache):
    import shutil
    cache = cache.resolve()
    if cache.parent == CACHE.resolve() and cache.is_dir():
        shutil.rmtree(cache)


def flows(model, x, diameter, mode):
    """Network output of a volume: (dP, cellprob) of the 3D run or of the planes."""
    kw = dict(channels=CHANNELS, z_axis=0, diameter=diameter, compute_masks=False)
    if mode == '3d':
        out = model.eval(x, do_3D=True, anisotropy=ANISOTROPY, **kw)[1]
    else:
        out = model.eval(x, do_3D=False, stitch_threshold=0.5, **kw)[1]     # > 0: normalised over the stack
    return out[1], out[2]


def drop_small(labels, level):
    import fastremap
    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero((sizes > 0) & (sizes < MIN_VOXELS[level]))
    small = small[small > 0]
    labels = np.where(np.isin(labels, small), 0, labels)
    return fastremap.renumber(labels.astype(np.int32), in_place=True)[0]


def masks_of(model, output, level, mode, stitch, cellprob, diameter):
    from cellpose import dynamics, utils
    dP, prob = output
    niter = int(round(200 * diameter / model.diam_mean))                       # Cellpose's 200 / rescale
    device = model.device if model.gpu else None
    if mode == '3d':
        masks = dynamics.resize_and_compute_masks(dP, prob, niter=niter, cellprob_threshold=cellprob, do_3D=True,
                                                  min_size=15, resize=None, device=device)[0]
    else:
        planes = np.array([dynamics.resize_and_compute_masks(dP[:, i], prob[i], niter=niter, cellprob_threshold=cellprob,
                                                             flow_threshold=0.4, min_size=-1, device=device)[0]
                           for i in range(prob.shape[0])])
        masks = utils.fill_holes_and_remove_small_masks(utils.stitch3D(planes, stitch_threshold=stitch), min_size=15)
    return drop_small(np.asarray(masks), level)


def counts(truth, predicted):
    """TP, FP, FN of one volume at IoU TAUS (Cellpose counts the objects by the largest label: renumbered first)."""
    import fastremap
    from cellpose import metrics
    truth, predicted = (fastremap.renumber(np.asarray(m), in_place=False)[0] for m in (truth, predicted))
    _, tp, fp, fn = metrics.average_precision([truth], [predicted], threshold=TAUS)
    return np.stack([tp[0], fp[0], fn[0]]).astype(np.int64)


def pooled_ap(total):
    tp, fp, fn = total
    return float(np.mean(tp / np.maximum(tp + fp + fn, 1)))


def in_outer(inner, outer, share):
    """Inner objects with at least `share` of their voxels in outer objects."""
    if not share:
        return inner
    total = np.bincount(inner.ravel())
    inside = np.bincount(inner[outer > 0].ravel(), minlength=len(total))
    drop = [i for i in np.flatnonzero(total) if i and inside[i] / total[i] < share]
    return np.where(np.isin(inner, drop), 0, inner)


def save(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(labels).astype(np.uint16), photometric='minisblack', compression='zlib')


def main():
    import torch
    from cellpose import models, train
    from importlib.metadata import version
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    select = names(SELECT, 'val')
    info = dict(name=args.name, data=str(DATA), select=str(SELECT), select_volumes=select, anisotropy=ANISOTROPY,
                side_step=args.side_step, min_mm3=dict(zip(args.levels, args.min_mm3)), levels={},
                cellpose=version('cellpose'), torch=torch.__version__)
    previous = json.loads((OUT / 'training.json').read_text())['levels'] if (OUT / 'training.json').is_file() else {}
    seconds = {}
    for level in (OUTER, INNER):
        started = time.time()
        torch.cuda.reset_peak_memory_stats() if gpu else None
        path = OUT / 'models' / level
        reused = args.reuse_models and path.is_file() and level in previous
        if reused:
            hours, planes, object_diameter = (previous[level][k] for k in ('training_hours', 'training_planes', 'diameters'))
            object_diameter = object_diameter['objects']
        else:
            model = models.CellposeModel(gpu=gpu, model_type=args.pretrained)
            files, flow_files, object_diameter = train_planes(level, CACHE / level, model.device)
            planes = len(files)
            if len(files) < args.nimg_per_epoch:
                repeat = -(-args.nimg_per_epoch // len(files))                   # see run_cellpose.py
                files, flow_files = files * repeat, flow_files * repeat
            path = train.train_seg(model.net, train_files=files, train_labels_files=flow_files, load_files=False,
                                   channels=CHANNELS, normalize=True, n_epochs=args.epochs,
                                   nimg_per_epoch=args.nimg_per_epoch, batch_size=args.batch, learning_rate=0.1,
                                   weight_decay=1e-4, SGD=True, min_train_masks=0, save_path=str(OUT), model_name=level)
            drop_cache(CACHE / level)
            path = path[0] if isinstance(path, (tuple, list)) else path
            hours = (time.time() - started) / 3600
        model = models.CellposeModel(gpu=gpu, pretrained_model=str(path))
        diameters = dict(labels=float(model.diam_labels), objects=object_diameter)
        # Val volumes of the selection: every setting from one network run per volume, size and mode.
        settings = [(d,) + s for d in diameters for s in SETTINGS]
        total = {s: 0 for s in settings}
        for name in select:
            x = tifffile.imread(SELECT / 'val' / 'images' / name)
            truth = read_labels(SELECT, 'val', level, name)
            for size, diameter in diameters.items():
                for mode in ('3d', 'stitch'):
                    t0 = time.time()
                    output = flows(model, x, diameter, mode)
                    t1 = time.time()
                    for setting in (s for s in settings if s[:2] == (size, mode)):
                        total[setting] = total[setting] + counts(truth, masks_of(model, output, level, *setting[1:], diameter))
                    del output
                    print(level, 'selection', name, size, mode, f'network {t1 - t0:.0f} s, masks {time.time() - t1:.0f} s', flush=True)
        scores = {s: pooled_ap(total[s]) for s in settings}
        size, *best = max(scores, key=scores.get)
        diameter = diameters[size]
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if gpu else None
        if reused and peak is not None:
            peak = max(peak, previous[level]['gpu_peak_mem_gb'] or 0.0)             # the training's peak
        info['levels'][level] = dict(pretrained=args.pretrained, channels=CHANNELS, training_hours=hours,
                                     gpu_peak_mem_gb=peak, diameters=diameters, diameter=diameter, mode=best[0],
                                     stitch_threshold=best[1], cellprob_threshold=best[2], val_select_ap=scores[(size, *best)],
                                     val_ap={'_'.join(map(str, s)): v for s, v in scores.items()},
                                     training_planes=planes, weights=str(path))
        print(level, json.dumps({k: v for k, v in info['levels'][level].items() if k != 'val_ap'}), flush=True)
        for split in args.splits:
            for name in names(DATA, split):
                started = time.time()
                pred = masks_of(model, flows(model, tifffile.imread(DATA / split / 'images' / name), diameter, best[0]),
                                level, *best, diameter)
                key = f'{split}/{name}'
                seconds[key] = seconds.get(key, 0.0) + time.time() - started
                save(OUT / 'predictions' / split / level / name, pred)
            print(level, split, 'predicted', flush=True)
        del model
        torch.cuda.empty_cache() if gpu else None
    # The hierarchy for the inner level: the share chosen on the selection volumes (all of them are val volumes).
    total = {s: 0 for s in SHARES}
    for name in select:
        truth = read_labels(SELECT, 'val', INNER, name)
        inner, outer = (tifffile.imread(OUT / 'predictions' / 'val' / level / name) for level in (INNER, OUTER))
        for s in SHARES:
            total[s] = total[s] + counts(truth, in_outer(inner, outer, s))
    shares = {s: pooled_ap(total[s]) for s in SHARES}
    share = max(shares, key=shares.get)
    info['in_outer'] = dict(share=share, val_select_ap={str(s): v for s, v in shares.items()})
    print('inner objects in outer ones:', json.dumps(info['in_outer']), flush=True)
    for split in args.splits:
        for name in names(DATA, split):
            inner = tifffile.imread(OUT / 'predictions' / split / INNER / name)
            outer = tifffile.imread(OUT / 'predictions' / split / OUTER / name)
            save(OUT / 'predictions_in_outer' / split / INNER / name, in_outer(inner, outer, share))
            save(OUT / 'predictions_in_outer' / split / OUTER / name, outer)
    (OUT / 'timing.json').write_text(json.dumps(seconds, indent=1))
    (OUT / 'training.json').write_text(json.dumps(info, indent=2, default=str))
    print('->', OUT / 'predictions', OUT / 'predictions_in_outer')


if __name__ == '__main__':
    main()
