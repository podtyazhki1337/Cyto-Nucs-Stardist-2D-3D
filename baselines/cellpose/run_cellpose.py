"""Cellpose (Stringer et al. 2021; Cellpose 3) as a baseline on a 2D dataset of one file per image with two nested
levels: one model per level, fine-tuned from the pretrained 'cyto3' (cells) and 'nuclei' (nuclei) models as the
'FT' variant of the HydraStarDist paper, or the pretrained models as they are (--pretrained-only). Runs in the
cellpose conda env (PyTorch); the predicted labels are written to <basedir>/<name>/predictions/<split>/<level>/
and scored with score_predictions.py in the CytoNucs env, like every other method.

  - data: <data>/{train,val,test}/images/<name> (RGB) and <split>/<level>_masks/<name>; unknown label 65535 in
    train masks: background;
  - channels in Cellpose's convention (1 = red, 2 = green, 3 = blue): --cell-channels CYTO NUCLEUS for the cell
    model, --nucleus-channel for the nucleus model; images normalised by Cellpose (percentiles 1 / 99);
  - training: Cellpose's train_seg with its fine-tuning recipe (SGD, learning rate 0.1, weight decay 1e-4,
    crops of 224 rescaled to the model's object size), --epochs epochs of --nimg-per-epoch crops in batches of
    --batch: the default 300 x 800 crops of batch 8 is the crop budget of the CytoNucs 2D runs (300 x 100 x 8);
    images without objects stay as background examples (min_train_masks 0: removing them breaks the batch sampling
    of Cellpose 3.0 with files); one continuous run per level that reads images and precomputed flows from disk batch by batch (the flows sit
    in <basedir>/<name>/cache during training, about 11 GB for TissueNet or HeLaCytoNuc, and are removed after);
  - thresholds (cellprob, flow) of every level chosen on the val split by the AP over IoU 0.5..0.95 of the pooled
    counts (Cellpose's metrics: TP / (TP + FP + FN), as ours); object size: the diameter of the training labels.

Usage (cellpose env):
  python run_cellpose.py --data DIR --name NAME --gpu 0 --cell-channels 1 3 --nucleus-channel 3 [--split test]
  python run_cellpose.py --data DIR --name NAME --gpu 0 --cell-channels 1 3 --nucleus-channel 3 --pretrained-only
then (CytoNucs env):  python score_predictions.py --data DIR --predictions <basedir>/<name>/predictions --split test
"""
import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--gpu', default='0')
parser.add_argument('--inner', default='nucleus')
parser.add_argument('--outer', default='cytoplasm')
parser.add_argument('--cell-channels', type=int, nargs=2, required=True, help='Cellpose channels [cytoplasm, nucleus] (1=R 2=G 3=B)')
parser.add_argument('--nucleus-channel', type=int, required=True)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--nimg-per-epoch', type=int, default=800)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--split', nargs='+', default=['test'])
parser.add_argument('--pretrained-only', action='store_true', help='no fine-tuning: the pretrained cyto3 / nuclei models')
parser.add_argument('--basedir', default='/storage02/miroslavm/cytonucs_paper_runs/cellpose')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import tifffile
import torch
from cellpose import dynamics, metrics, models, train

UNKNOWN = 65535
TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CELLPROB, FLOW = (-1.0, 0.0, 1.0), (0.4, 0.6)
DATA, OUT = Path(args.data), Path(args.basedir) / args.name
LEVELS = {args.outer: dict(pretrained='cyto3', channels=list(args.cell_channels)),
          args.inner: dict(pretrained='nuclei', channels=[args.nucleus_channel, 0])}


def names(split):
    return sorted(p.name for p in (DATA / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))


def flow_files(split, level, device):
    """Training files read batch by batch (train_seg with load_files=False), so the dataset is never held in
    memory: links to the images in a cache folder of the run and next to them the Cellpose flows of the labels
    (<name>_flows.tif), computed 64 images at a time. Returns (image files, flow files)."""
    cache = OUT / 'cache' / split / level
    cache.mkdir(parents=True, exist_ok=True)
    images, flows, todo = [], [], []
    for name in names(split):
        link = cache / name
        if not link.exists():
            link.symlink_to((DATA / split / 'images' / name).resolve())
        flow = cache / (Path(name).stem + '_flows.tif')
        images.append(str(link))
        flows.append(str(flow))
        if not flow.is_file():
            todo.append(name)
    for start in range(0, len(todo), 64):
        chunk = todo[start:start + 64]
        labels = [np.where(y == UNKNOWN, 0, y).astype(np.int32)
                  for y in (tifffile.imread(DATA / split / f'{level}_masks' / n).astype(np.int64) for n in chunk)]
        dynamics.labels_to_flows(labels, files=[str(cache / n) for n in chunk], device=device, return_flows=False)
    return images, flows


def drop_cache(level):
    import shutil
    cache = (OUT / 'cache' / 'train' / level).resolve()
    if cache.parent.parent == (OUT / 'cache').resolve() and cache.is_dir():
        shutil.rmtree(cache)


def pooled_ap(truth, predicted):
    _, tp, fp, fn = metrics.average_precision(truth, predicted, threshold=TAUS)
    tp, fp, fn = (np.asarray(v).sum(axis=0) for v in (tp, fp, fn))
    return float(np.mean(tp / np.maximum(tp + fp + fn, 1)))


def predict(model, image, level, diameter, cellprob, flow):
    masks = model.eval(image, channels=LEVELS[level]['channels'], diameter=diameter, cellprob_threshold=cellprob,
                       flow_threshold=flow, normalize=True)[0]
    return np.asarray(masks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get('GPU_MEMORY_LIMIT_MB', 0) or 0)         # a cap when the queue shares the GPU
    if limit and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total))
    from importlib.metadata import version
    info = dict(name=args.name, data=str(DATA), pretrained_only=args.pretrained_only, levels={},
                cellpose=version('cellpose'), torch=torch.__version__)
    val_images = [tifffile.imread(DATA / 'val' / 'images' / n) for n in names('val')]
    chosen = {}
    for level, spec in LEVELS.items():
        started = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        gpu = torch.cuda.is_available()
        # Pretrained only: models.Cellpose, whose size model estimates the object diameter of every image.
        model = (models.Cellpose(gpu=gpu, model_type=spec['pretrained']) if args.pretrained_only else
                 models.CellposeModel(gpu=gpu, model_type=spec['pretrained']))
        path = None
        if not args.pretrained_only:
            files, flows = flow_files('train', level, model.device)
            path = train.train_seg(model.net, train_files=files, train_labels_files=flows, load_files=False,
                                   channels=spec['channels'], normalize=True, n_epochs=args.epochs,
                                   nimg_per_epoch=args.nimg_per_epoch, batch_size=args.batch, learning_rate=0.1,
                                   weight_decay=1e-4, SGD=True, min_train_masks=0, save_path=str(OUT), model_name=level)
            drop_cache(level)
            path = path[0] if isinstance(path, (tuple, list)) else path          # the path alone in Cellpose 3.0
            model = models.CellposeModel(gpu=gpu, pretrained_model=str(path))
        hours = (time.time() - started) / 3600
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        diameter = float(model.diam_labels) if not args.pretrained_only else None        # pretrained: Cellpose estimates it
        truth = [np.where(y == UNKNOWN, 0, y) for y in (tifffile.imread(DATA / 'val' / f'{level}_masks' / n) for n in names('val'))]
        scores = {}
        for cellprob in CELLPROB:
            for flow in FLOW:
                scores[(cellprob, flow)] = pooled_ap(truth, [predict(model, x, level, diameter, cellprob, flow) for x in val_images])
        best = max(scores, key=scores.get)
        chosen[level] = (model, diameter, best)
        info['levels'][level] = dict(pretrained=spec['pretrained'], channels=spec['channels'], training_hours=hours,
                                     gpu_peak_mem_gb=peak, diameter=diameter, cellprob_threshold=best[0], flow_threshold=best[1],
                                     val_ap={f'{c}_{f}': v for (c, f), v in scores.items()},
                                     parameters=int(sum(p.numel() for p in (model.cp if args.pretrained_only else model).net.parameters())),
                                     weights=None if path is None else str(path))
        print(level, json.dumps({k: v for k, v in info['levels'][level].items() if k != 'val_ap'}), flush=True)
    timing = {}
    for split in args.split:
        for name in names(split):
            image = tifffile.imread(DATA / split / 'images' / name)
            started = time.time()
            for level, (model, diameter, (cellprob, flow)) in chosen.items():
                labels = predict(model, image, level, diameter, cellprob, flow)
                (OUT / 'predictions' / split / level).mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(OUT / 'predictions' / split / level / name, labels.astype(np.uint16), photometric='minisblack',
                                 compression='zlib')
            timing[f'{split}/{name}'] = time.time() - started
    info['seconds'] = timing
    (OUT / 'training.json').write_text(json.dumps(info, indent=2, default=str))
    print('->', OUT / 'predictions')


if __name__ == '__main__':
    main()
