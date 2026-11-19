"""HydraStarDist (De, Urbanski, Yakimovich 2025, arXiv 2504.12078) as a baseline on a 2D dataset of one file per
image with two nested levels, trained with the authors' code: baselines/hydrastardist/hydrastardist is the model
package of github.com/plaque2/napari-hydra (commit c68ae35, BSD-3-Clause, see LICENSE next to it); only its imports
were renamed.

Training as published (paper, Section 4.3, and the defaults of the code): StarDist U-Net with a joint encoder and
one decoder per level, 32 rays, patch 256, batch 32, 400 epochs of 100 steps, Adam 3e-4 with ReduceLROnPlateau
(factor 0.5, patience 40), loss weights (1, 0.2) per branch, background regularisation 1e-4, 90 % of the patches
with foreground; images normalised per channel to their minimum and maximum (load_and_preprocess_data_hydra),
labels with filled holes. Choices of this baseline:
  - grid --grid (2 x 2 as the CytoNucs and StarDist runs; the code's default is 1 x 1);
  - augmentation: the CytoNucs functions with the values of --like, as for the other methods (the paper names
    none): flips, 90 degree rotations, intensity, noise, blur, elastic deformation;
  - WBR: in the released code the WBR layer gets the input image as its mask, multiplies predictions of
    different batch elements, and binarises the sigmoid outputs (always > 0) to 1, so its gradient is zero and
    HSD-WBR trains exactly as HSD. Without --wbr the run is HSD; with --wbr it is HSD-WBR with the loss of the paper
    (Eq. 4-5, lambda_3 = 1, eps = 1e-7) made to work: an output 'wbr' = [inner, outer] probability maps of the two
    branches, trained against the ground-truth outer foreground per image (base_hydra.wbr_loss_paper; its ratio is
    capped at 0.99 so the loss stays below 100, where the paper's form turns negative);
  - the code's data generator trains branch 1 on the second label set it is given and branch 2 on the first
    (StarDistData2D.__getitem__ swaps them), while optimize_thresholds compares branch 1 with the first set; here
    the thresholds are optimised with the label sets in the order the branches learned them, and branch 1 is read
    as the outer level, branch 2 as the inner one;
  - the index cache of valid patch centres is replaced by baselines/patch_sampler.py (same distribution, memory);
  - unknown label 65535 in train masks: background.
Scoring as baselines/stardist3d/run_files.py: AP per level over IoU 0.5..0.95, JTPR (ours, predicted pairs by
overlap), the paper's metrics (score_hierarchy.paper_metrics); predicted objects mostly inside
<split>/<level>_unknown/<name> are left out.

Usage:
  python run_hsd.py --data DIR --gpu 0 --like CONFIG --name NAME [--epochs 400] [--batch 32] [--wbr]
  python run_hsd.py --data DIR --score --name NAME --gpu 0 [--split test val]
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data', required=True)
parser.add_argument('--inner', default='nucleus')
parser.add_argument('--outer', default='cytoplasm')
parser.add_argument('--gpu', default='')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--patch', type=int, nargs=2, default=[256, 256])
parser.add_argument('--grid', type=int, nargs=2, default=[2, 2])
parser.add_argument('--like', help='CytoNucs config whose augmentation values are used')
parser.add_argument('--name', required=True)
parser.add_argument('--seed', type=int, default=42, help='random seed of the run (42: the default of the code)')
parser.add_argument('--wbr', action='store_true', help="HSD-WBR: the paper's within-boundary regularisation, lambda_3 = 1")
parser.add_argument('--basedir', default='runs/hydrastardist')
parser.add_argument('--score', action='store_true')
parser.add_argument('--split', nargs='+', default=['test'])
parser.add_argument('--predictions', help='with --score: folder for the predicted labels (<split>/<level>/<name>)')
parser.add_argument('--fixed-thresholds', type=float, nargs=2, metavar=('PROB', 'NMS'),
                    help='use these thresholds for both branches instead of optimising them (smoke tests)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tifffile
import tensorflow as tf
from csbdeep.utils import normalize
from stardist import fill_label_holes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
from hydrastardist.models.model2d_hydra import Config2D, StarDist2D                     # noqa: E402
from hydrastardist.models.base_hydra import StarDistDataBase                             # noqa: E402
from patch_sampler import light_valid_inds                                               # noqa: E402
from cytonucs_data_generator import cytonucs_augmenter_3d, elastic_displacement, warp_yx  # noqa: E402

for device in tf.config.list_physical_devices('GPU'):
    if int(os.environ.get('GPU_MEMORY_LIMIT_MB', 0) or 0):          # a cap when the queue shares the GPU
        tf.config.set_logical_device_configuration(
            device, [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ['GPU_MEMORY_LIMIT_MB']))])
    else:
        tf.config.experimental.set_memory_growth(device, True)
SEED = args.seed
np.random.seed(SEED); tf.random.set_seed(SEED)
UNKNOWN = 65535
DATA = Path(args.data)
PATCH, GRID = tuple(args.patch), tuple(args.grid)
StarDistDataBase.get_valid_inds = light_valid_inds('Y1')


def names(split):
    return sorted(p.name for p in (DATA / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))


def read_image(path):
    """Channels last, every channel scaled to its minimum and maximum (the authors' normalisation)."""
    with tifffile.TiffFile(path) as tif:
        x, axes = tif.asarray().astype(np.float32), tif.series[0].axes
    channel = [i for i, a in enumerate(axes) if a in 'CS']
    if channel and x.ndim > 2:
        x = np.moveaxis(x, channel[0], -1)
    return normalize(x, 0, 100, axis=(0, 1)).astype(np.float32)


def read_labels(split, level, name):
    y = tifffile.imread(DATA / split / f'{level}_masks' / name)
    return fill_label_holes(np.where(y == UNKNOWN, 0, y).astype(np.uint16))


def load(split):
    xs, inner, outer = [], [], []
    for name in names(split):
        xs.append(read_image(DATA / split / 'images' / name))
        inner.append(read_labels(split, args.inner, name))
        outer.append(read_labels(split, args.outer, name))
    return xs, inner, outer


def make_augmenter(like):
    from cytonucs_config import read_config
    aug = read_config(like)['augmentation'] if like else {}   # the method's augmentation settings included
    cfg = SimpleNamespace(rotation_yx=aug.get('rotation_yx', True), flip_prob=aug.get('flip_prob', .5), flip_z_prob=0.,
                          intensity_range=aug.get('intensity_range', (.8, 1.2)), intensity_offset=aug.get('intensity_offset', (-.1, .1)),
                          gamma_range=aug.get('gamma_range', (.9, 1.1)), noise_sigma=aug.get('noise_sigma', .01),
                          blur_sigma_range=aug.get('blur_sigma_range'), saturation_prob=aug.get('saturation_prob') or 0.)
    elastic = float(aug.get('elastic_prob', 0.)), float(aug.get('elastic_alpha', 0.)), float(aug.get('elastic_sigma', 8.))
    rng = np.random.default_rng(SEED)

    def augment(x, y1, y2):
        x, y1, y2 = np.asarray(x, np.float32)[None], np.asarray(y1)[None], np.asarray(y2)[None]
        if elastic[0] and rng.random() < elastic[0]:
            dy, dx = elastic_displacement(y1.shape[-2:], elastic[1], elastic[2], rng)
            x, y1, y2 = warp_yx(x, dy, dx, 1, 3), warp_yx(y1, dy, dx, 0, 3), warp_yx(y2, dy, dx, 0, 3)
        x, y1, y2 = cytonucs_augmenter_3d(x, y1, y2, rng=rng, config=cfg)
        return x[0], y1[0], y2[0]
    return augment, dict(cfg.__dict__, elastic=elastic)


def n_tiles(shape, channels):
    return tuple(max(1, int(np.ceil(s / (2 * p)))) for s, p in zip(shape[:2], PATCH)) + ((1,) if channels else ())


def train():
    started = time.time()
    x_train, inner_train, outer_train = load('train')
    x_val, inner_val, outer_val = load('val')
    load_s = time.time() - started
    channels = x_train[0].ndim > 2
    augment, aug_values = make_augmenter(args.like)
    conf = Config2D(axes='YXC' if channels else 'YX', n_rays=32, n_channel_in=x_train[0].shape[-1] if channels else 1,
                    grid=GRID, use_gpu=False, train_patch_size=PATCH, train_batch_size=args.batch, train_epochs=args.epochs,
                    train_steps_per_epoch=args.steps, train_learning_rate=3e-4, train_tensorboard=False,
                    train_wbr_loss=args.wbr)
    model = StarDist2D(conf, name=args.name, basedir=args.basedir)
    params = int(model.keras_model.count_params())
    fov = [int(max(a, b)) for a, b in zip(*model._axes_tile_overlap('YX'))]       # per branch in this code: the larger
    print(f'{len(x_train)} train, {len(x_val)} val images, field of view {fov}, {params} parameters, loading {load_s:.0f} s')
    tf.config.experimental.reset_memory_stats('GPU:0') if tf.config.list_logical_devices('GPU') else None
    started = time.time()
    # The code's generator trains branch 1 on the second label set given here (outer), branch 2 on the first (inner).
    history = model.train(x_train, inner_train, outer_train, validation_data=(x_val, inner_val, outer_val), augmenter=augment,
                          epochs=args.epochs, steps_per_epoch=args.steps)
    hours = (time.time() - started) / 3600
    peak = (tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2 ** 30) if tf.config.list_logical_devices('GPU') else None
    started = time.time()
    if args.fixed_thresholds:
        thresholds = [dict(prob=args.fixed_thresholds[0], nms=args.fixed_thresholds[1])] * 2
        for k, t in enumerate(thresholds, 1):
            (Path(args.basedir) / args.name / f'thresholds{k}.json').write_text(json.dumps(t))
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            thresholds = model.optimize_thresholds(x_val, outer_val, inner_val,
                                                   predict_kwargs=dict(n_tiles=n_tiles(np.max([x.shape for x in x_val], axis=0), channels),
                                                                       show_tile_progress=False))
    info = dict(name=args.name, data=str(DATA), seed=SEED, inner=args.inner, outer=args.outer, branches=dict(branch1=args.outer, branch2=args.inner),
                epochs=args.epochs, steps_per_epoch=args.steps, batch=args.batch, patch=list(PATCH), grid=list(GRID), rays=32,
                train_images=len(x_train), val_images=len(x_val), augmentation=aug_values, parameters=params, field_of_view=fov,
                load_s=load_s, training_hours=hours, gpu_peak_mem_gb=peak, threshold_optimisation_s=time.time() - started,
                fixed_thresholds=bool(args.fixed_thresholds),
                thresholds={args.outer: {k: float(v) for k, v in thresholds[0].items()},
                            args.inner: {k: float(v) for k, v in thresholds[1].items()}},
                val_loss=[float(v) for v in history.history.get('val_loss', [])],
                code='baselines/hydrastardist/hydrastardist = napari-hydra c68ae35 (BSD-3-Clause), ' +
                     ('WBR of the paper, made differentiable (wbr_loss_paper)' if args.wbr else 'no WBR'), wbr=args.wbr,
                weights=str(Path(args.basedir) / args.name / 'weights_best.h5'))
    (Path(args.basedir) / args.name / 'training.json').write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps({k: info[k] for k in ('training_hours', 'gpu_peak_mem_gb', 'thresholds', 'parameters')}))


def score():
    from score_hierarchy import ground_truth, save_prediction, score_image, summarise
    inner, outer = args.inner, args.outer
    model = StarDist2D(None, name=args.name, basedir=args.basedir)
    results = {}
    for split in args.split:
        records = []
        for name in names(split):
            x = read_image(DATA / split / 'images' / name)
            started = time.time()
            # Thresholds passed explicitly: without them the code falls back to self.thresholds, which it never sets.
            first, second = model.predict_instances(x, n_tiles=n_tiles(x.shape, x.ndim > 2), show_tile_progress=False,
                                                    prob_thresh1=model.thresholds1['prob'], nms_thresh1=model.thresholds1['nms'],
                                                    prob_thresh2=model.thresholds2['prob'], nms_thresh2=model.thresholds2['nms'])
            seconds = time.time() - started
            pred = {outer: np.asarray(first[0]), inner: np.asarray(second[0])}           # branch 1 learned the outer level
            gt, truths, ignore = ground_truth(DATA / split, name, [inner, outer], [(inner, outer)])
            record, scored = score_image(gt, pred, [(inner, outer)], truths, ignore=ignore)      # pairs by overlap
            records.append(dict(record, name=name, seconds=seconds))
            if args.predictions:
                save_prediction(Path(args.predictions) / split, name, scored)
        result = dict(split=split, images=len(records), seconds_per_image=float(np.mean([r['seconds'] for r in records])),
                      model=args.name, predictions=str(Path(args.predictions) / split) if args.predictions else None)
        result.update(summarise(records, [inner, outer], [(inner, outer)]))
        result['records'] = records
        results[split] = result
        best = result[f'paper_{inner}_{outer}']
        print(f"{split}: {inner} {result['objects'][inner]}: AP {result[f'ap_{inner}']:.3f} | {outer} {result['objects'][outer]}: "
              f"AP {result[f'ap_{outer}']:.3f} | JTPR {result[f'jtpr_{inner}_{outer}'] or 0:.3f} of {result[f'pairs_{inner}_{outer}']}"
              f" | paper JTPR inner/outer {best['jtpr_inner_best'] or 0:.3f} / {best['jtpr_outer_best'] or 0:.3f}"
              f" | {result['seconds_per_image']:.1f} s per image")
    out = Path(args.basedir) / args.name / 'scores.json'
    # Splits scored before (one call per split in the job scripts) stay in the file.
    previous = json.loads(out.read_text()) if out.is_file() else {}
    out.write_text(json.dumps({**previous, **results}, indent=1))
    print('->', out)


if __name__ == '__main__':
    score() if args.score else train()
