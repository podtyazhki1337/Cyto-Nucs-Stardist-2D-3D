"""InstanSeg (Goldsborough et al. 2024; package instanseg-torch) as a baseline on a 2D dataset of one file per image with
two nested levels (nucleus inside cell). Unlike Cellpose (one model per level, run_cellpose.py), InstanSeg predicts nuclei
and cells with one model and links them itself (resolve_cell_and_nucleus: a cell and its nucleus share one label). The
predicted links go to parents/, so the JTPR scores InstanSeg's own pairing.

  - --pretrained-only: the published 'fluorescence_nuclei_and_cells' model (channel invariant, trained at 0.5 um/px) as
    it is. It rescales an image from the pixel size it is given to its own. Our HeLaCytoNuc / BBBC020 files carry
    none, so the pixel size is chosen on val among PIXEL_SIZES (0.1..1.0 um/px; TissueNet's 0.5 is in the grid). Its training data
    include TissueNet (Goldsborough et al. 2024), so TissueNet scores of the pretrained model may be optimistic;
  - otherwise trained from scratch on the training split with InstanSeg's own training (instanseg.scripts.train,
    target NC). Its default recipe: InstanSeg_UNet [32, 64, 128, 256], Adam with learning rate 1e-3, batch 3, tiles of
    256 px, 10 hot-start epochs, then --epochs epochs of --len-epoch tiles. It keeps the checkpoint of the best F1 of its
    own validation on the val split. Inputs are the nucleus and cytoplasm channels at native resolution (no rescaling),
    unless --pixel-size gives the files a nominal pixel size: then InstanSeg's own rescaling (requested_pixel_size 0.5
    um/px, as its published models) resizes them by pixel_size / 0.5 in training, in its validation and at prediction,
    and the labels are resized back to the file's resolution. InstanSeg predicts each object inside a window of 128 px
    around its seed, so objects larger than that need it: BBBC020's cells (median largest side 197 px) take 0.125, a
    downscaling by 4;
  - the post-processing thresholds (seed_threshold, mask_threshold; InstanSeg's defaults 0.7 / 0.53 are in the grid) are
    chosen on val. The score is the mean over the two levels of the AP over IoU 0.5..0.95 of the pooled counts. The
    unknown label 65535 counts as background.
Predicted labels: <basedir>/<name>/predictions/<split>/{nucleus,cytoplasm}/<name>, links in parents/<stem>.json; scored
with score_predictions.py in the CytoNucs env, like every other method.

Usage (env instanseg):
  python run_instanseg.py --data DIR --name NAME --gpu 0 --nucleus-channel 3 --cytoplasm-channel 1 --pretrained-only --split test
  python run_instanseg.py --data DIR --name NAME --gpu 0 --nucleus-channel 3 --cytoplasm-channel 1 [--seed 1] --split test
  python run_instanseg.py --data DIR --name NAME --gpu 0 --nucleus-channel 3 --cytoplasm-channel 1 --predict-only --split holdout
  python run_instanseg.py --data DIR --name NAME --gpu 0 --nucleus-channel 3 --cytoplasm-channel 1 --from-checkpoint --split test
then (CytoNucs env):  python score_predictions.py --data DIR --predictions <basedir>/<name>/predictions --split test
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--gpu', default='0')
parser.add_argument('--inner', default='nucleus')
parser.add_argument('--outer', default='cytoplasm')
parser.add_argument('--nucleus-channel', type=int, required=True, help='1 = R, 2 = G, 3 = B (as run_cellpose.py)')
parser.add_argument('--cytoplasm-channel', type=int, required=True)
parser.add_argument('--epochs', type=int, default=500, help="InstanSeg's default")
parser.add_argument('--len-epoch', type=int, default=1000, help="tiles per epoch, InstanSeg's default")
parser.add_argument('--batch', type=int, default=3, help="InstanSeg's default")
parser.add_argument('--split', nargs='+', default=['test'])
parser.add_argument('--limit', type=int, default=0, help='first N images of every split only (smoke tests)')
parser.add_argument('--pretrained-only', action='store_true', help="no training: the 'fluorescence_nuclei_and_cells' model")
parser.add_argument('--from-checkpoint', action='store_true',
                    help='no training: the best checkpoint <basedir>/<name>/model of an earlier training (e.g. one that '
                         'stopped early), then the val grid and --split as after a training')
parser.add_argument('--predict-only', action='store_true',
                    help='no training: the model and settings of <basedir>/<name>/training.json predict --split')
parser.add_argument('--basedir', default='runs/instanseg')
parser.add_argument('--seed', type=int, default=0, help="0: InstanSeg's default (no RNG seed); k > 0: rng_seed k")
parser.add_argument('--pixel-size', type=float, default=None,
                    help='trained model: nominal pixel size of the files (um/px); InstanSeg rescales them to %s um/px. '
                         'Default: native resolution' % 0.5)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import tifffile
import torch

UNKNOWN = 65535
TAUS = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
# Small pixel sizes = strong downscaling: the cells of HeLaCytoNuc / BBBC020 are 5-7x larger in pixels than TissueNet's
# (0301 had its val optimum on BBBC020 at the old lower edge 0.25).
PIXEL_SIZES = (0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.5, 0.7, 1.0)
SEED_THRESHOLDS, MASK_THRESHOLDS = (0.5, 0.7, 0.9), (0.4, 0.53)
MODEL_PIXEL_SIZE = 0.5                  # um/px the trained model works at when --pixel-size is given (InstanSeg's own)
MAX_CANDIDATES = 10000                  # candidate masks of one val image whose merge is tried (guarded_validation_merge)
PRETRAINED = 'fluorescence_nuclei_and_cells'
DATA, OUT = Path(args.data), Path(args.basedir) / args.name
CHANNELS = [args.nucleus_channel - 1, args.cytoplasm_channel - 1]
LEVELS = [args.inner, args.outer]


def names(split):
    found = sorted(p.name for p in (DATA / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    return found[:args.limit] if args.limit else found


def image(split, name):
    """(2, H, W) float32: the nucleus channel, then the cytoplasm channel."""
    x = np.asarray(tifffile.imread(DATA / split / 'images' / name))
    return np.ascontiguousarray(np.moveaxis(x[..., CHANNELS], -1, 0).astype(np.float32))


def labels(split, level, name):
    y = tifffile.imread(DATA / split / f'{level}_masks' / name).astype(np.int64)
    return np.where(y == UNKNOWN, 0, y).astype(np.int32)


def counts(truth, predicted):
    """True positives at every IoU threshold of TAUS (pairs with IoU >= 0.5 are one-to-one), objects in truth and
    prediction."""
    t, p = truth.ravel().astype(np.int64), predicted.ravel().astype(np.int64)
    n_true, n_pred = len(np.unique(t[t > 0])), len(np.unique(p[p > 0]))
    both = (t > 0) & (p > 0)
    if not both.any():
        return np.zeros(len(TAUS)), n_true, n_pred
    key, inter = np.unique(t[both] * (p.max() + 1) + p[both], return_counts=True)
    ti, pi = key // (p.max() + 1), key % (p.max() + 1)
    iou = inter / (np.bincount(t)[ti] + np.bincount(p)[pi] - inter)
    return np.array([(iou >= tau).sum() for tau in TAUS]), n_true, n_pred


def pooled_ap(pairs):
    tp, n_true, n_pred = np.zeros(len(TAUS)), 0, 0
    for truth, predicted in pairs:
        c, a, b = counts(truth, predicted)
        tp, n_true, n_pred = tp + c, n_true + a, n_pred + b
    return float(np.mean(tp / np.maximum(n_true + n_pred - tp, 1)))


def predict(model, x, pixel_size, seed_threshold, mask_threshold):
    """Nucleus and cell labels (H, W) each, and the links nucleus -> cell (shared labels)."""
    out = model.eval_small_image(x, pixel_size=pixel_size, normalise=True, return_image_tensor=False,
                                 target='all_outputs', seed_threshold=seed_threshold, mask_threshold=mask_threshold)
    out = out[0] if isinstance(out, (tuple, list)) else out
    lab = out.squeeze(0).cpu().numpy().astype(np.int64)
    nuclei, cells = lab[0], lab[1]
    shared = np.intersect1d(np.unique(nuclei[nuclei > 0]), np.unique(cells[cells > 0]))
    return nuclei, cells, {int(k): int(k) for k in shared}


def trained_model():
    """InstanSeg's own training (instanseg.scripts.train.instanseg_training) on our train / val splits; the model of
    its best val F1, wrapped for inference."""
    from instanseg.scripts import train as instanseg_train

    def items(split):
        return [dict(image=image(split, n), nucleus_masks=labels(split, args.inner, n),
                     cell_masks=labels(split, args.outer, n), parent_dataset='cytonucs', image_modality='Fluorescence',
                     nuclei_channels=[0], pixel_size=args.pixel_size, name=n) for n in names(split)]

    dataset = {'Train': items('train'), 'Validation': items('val')}
    sys.argv = sys.argv[:1]                               # instanseg_training parses the command line itself
    with guarded_validation_merge():
        instanseg_train.instanseg_training(
            segmentation_dataset=dataset, output_path=str(OUT), experiment_str='model', data_path=str(OUT),
            source_dataset='[cytonucs]', target_segmentation='NC', dim_in=2, num_epochs=args.epochs,
            length_of_epoch=args.len_epoch, batch_size=args.batch, rng_seed=args.seed or None, on_cluster=True,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            **(dict(requested_pixel_size=MODEL_PIXEL_SIZE) if args.pixel_size else {}))
    return load_trained()


@contextlib.contextmanager
def guarded_validation_merge():
    """InstanSeg's validation during training merges the candidate masks of an image by their pairwise IoU, three dense
    C x C matrices on the GPU (instanseg_loss.merge_sparse_predictions). A degenerate prediction of job 0303 had about
    74k candidates at epoch 266 of 500 and asked for 20.5 GB, which ended the training. Above MAX_CANDIDATES (about 1 GB)
    or on a CUDA OOM, that image gets the unmerged labels, as InstanSeg itself does when every candidate fills its
    window. This changes only the val F1 of such an epoch, and it keeps the neighbour job on the GPU from an OOM. The
    original is restored afterwards: the scripted inference model compiles it (and TorchScript cannot compile this)."""
    from instanseg.utils.loss import instanseg_loss
    merge = instanseg_loss.merge_sparse_predictions

    def guarded(x, coords, mask_map, size, mask_threshold=0.5, **kwargs):
        if size[0] <= MAX_CANDIDATES:
            try:
                return merge(x, coords, mask_map, size, mask_threshold=mask_threshold, **kwargs)
            except torch.OutOfMemoryError:
                pass
        torch.cuda.empty_cache()
        print(f'validation merge skipped: {size[0]} candidate masks, unmerged labels used', flush=True)
        return instanseg_loss.convert(x, coords, size=(size[1], size[2]), mask_threshold=mask_threshold)[None]

    instanseg_loss.merge_sparse_predictions = guarded
    try:
        yield
    finally:
        instanseg_loss.merge_sparse_predictions = merge


def load_trained():
    from instanseg import InstanSeg
    from instanseg.utils.loss.instanseg_loss import InstanSeg_Torchscript
    from instanseg.utils.model_loader import load_model
    net, spec = load_model(folder='', path=str(OUT / 'model'))
    net.eval()
    wrapped = InstanSeg_Torchscript(net, cells_and_nuclei=True, pixel_size=spec['pixel_size'], n_sigma=spec['n_sigma'],
                                    feature_engineering_function=str(spec['feature_engineering']), backbone_dim_in=2)
    # scripted as InstanSeg's export_to_torchscript does: the inference class reads the forward's arguments from the
    # TorchScript graph
    scripted = torch.jit.script(wrapped)
    torch.jit.save(scripted, str(OUT / 'model' / 'instanseg.pt'))
    return InstanSeg(scripted, verbosity=0)


def pretrained_model():
    from instanseg import InstanSeg
    return InstanSeg(PRETRAINED, verbosity=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version
    gpu = torch.cuda.is_available()
    started = time.time()
    torch.cuda.reset_peak_memory_stats() if gpu else None
    model = (pretrained_model() if args.pretrained_only else load_trained() if args.from_checkpoint else
             trained_model())
    hours = None if args.from_checkpoint else (time.time() - started) / 3600       # from_checkpoint: see its log
    train_peak = None if args.from_checkpoint or not gpu else torch.cuda.max_memory_allocated() / 2 ** 30
    val = [(image('val', n), labels('val', args.inner, n), labels('val', args.outer, n)) for n in names('val')]
    grid = [(p, s, m) for p in (PIXEL_SIZES if args.pretrained_only else (args.pixel_size,))
            for s in SEED_THRESHOLDS for m in MASK_THRESHOLDS]
    scores = {}
    for setting in grid:
        predicted = [predict(model, x, *setting) for x, _, _ in val]
        scores[setting] = [pooled_ap([(yn, pn) for (_, yn, _), (pn, _, _) in zip(val, predicted)]),
                           pooled_ap([(yc, pc) for (_, _, yc), (_, pc, _) in zip(val, predicted)])]
        print('val', setting, [round(v, 4) for v in scores[setting]], flush=True)
    best = max(scores, key=lambda k: np.mean(scores[k]))
    parameters = sum(p.numel() for p in model.instanseg.parameters())
    info = dict(name=args.name, data=str(DATA), seed=args.seed, pretrained_only=args.pretrained_only,
                from_checkpoint=args.from_checkpoint,
                model=PRETRAINED if args.pretrained_only else str(OUT / 'model'), instanseg=version('instanseg-torch'),
                torch=torch.__version__, channels=[c + 1 for c in CHANNELS], training_hours=hours,
                gpu_peak_mem_gb=train_peak, parameters=int(parameters),
                recipe=None if args.pretrained_only else dict(epochs=args.epochs, len_epoch=args.len_epoch,
                                                             batch=args.batch, target='NC', hotstart_epochs=10,
                                                             requested_pixel_size=MODEL_PIXEL_SIZE if args.pixel_size
                                                             else None),
                pixel_size=best[0], seed_threshold=best[1], mask_threshold=best[2],
                val_ap={f'{p}_{s}_{m}': v for (p, s, m), v in scores.items()})
    print('chosen', json.dumps({k: v for k, v in info.items() if k != 'val_ap'}, default=str), flush=True)
    info['seconds'] = predict_splits(model, best)
    (OUT / 'training.json').write_text(json.dumps(info, indent=2, default=str))
    print('->', OUT / 'predictions')


def predict_splits(model, setting):
    """Labels of both levels and the links for the images of --split into <OUT>/predictions/<split>/; seconds per
    image."""
    timing = {}
    for split in args.split:
        root = OUT / 'predictions' / split
        for d in LEVELS + ['parents']:
            (root / d).mkdir(parents=True, exist_ok=True)
        for name in names(split):
            x = image(split, name)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            started = time.time()
            nuclei, cells, links = predict(model, x, *setting)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timing[f'{split}/{name}'] = time.time() - started
            for level, lab in zip(LEVELS, (nuclei, cells)):
                assert lab.max() < UNKNOWN, f'{name}: {lab.max()} labels'
                tifffile.imwrite(root / level / name, lab.astype(np.uint16), photometric='minisblack', compression='zlib')
            (root / 'parents' / (Path(name).stem + '.json')).write_text(json.dumps(links))
    return timing


def predict_only():
    info = json.loads((OUT / 'training.json').read_text())
    model = pretrained_model() if info['pretrained_only'] else load_trained()
    seconds = predict_splits(model, (info['pixel_size'], info['seed_threshold'], info['mask_threshold']))
    path = OUT / 'predict_seconds.json'
    previous = json.loads(path.read_text()) if path.is_file() else {}
    path.write_text(json.dumps({**previous, **seconds}, indent=1))
    print('->', OUT / 'predictions')


if __name__ == '__main__':
    predict_only() if args.predict_only else main()
