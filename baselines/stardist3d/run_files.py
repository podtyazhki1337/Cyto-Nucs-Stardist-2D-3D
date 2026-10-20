"""Plain StarDist (3D, or 2D for 2D data) on a dataset of one file per volume (KiTS23, BBBC020, ...), one model per
object level, trained like the CytoNucs run it is compared with and scored like score_hierarchy.py.

  - data: <data>/{train,val,test}/images/<name> and <data>/<split>/<level>_masks/<name>; every volume normalised by
    its percentiles 1 / 99.8 (per channel for 2D colour images), as the CytoNucs generator does;
  - the train volumes are cached once as float16 / smallest-unsigned .npy files (--cache) and memory-mapped, so a large
    dataset (342 KiTS23 CT volumes) is not held in RAM. StarDist's own sampler keeps every valid patch centre of
    every volume in memory (hundreds of MB per CT volume), so it is replaced by one with the same distribution:
    a centre drawn uniformly among the valid centres, with probability train_foreground_only (0.9) among those
    whose patch contains foreground (rejection sampling on a bit mask of them);
  - patch --patch, --batch patches per step, --steps steps per epoch, --epochs epochs; augmentation: the CytoNucs functions with
    the values of --like (flips and 90 degree rotations in XY, Z flip, intensity scale / offset / gamma, noise,
    blur, elastic XY deformation; no zoom);
  - unknown label 65535 in train masks (sparse annotation): background, StarDist cannot ignore voxels; files
    without masks of the level (partial annotation) are left out of that level's training;
  - thresholds optimised on the val volumes; scoring on --split: AP per level over IoU 0.5..0.95, predicted
    objects mostly inside <split>/<level>_unknown/<name> left out (as score_hierarchy.py), JTPR of the two models
    (predicted inner objects paired with predicted outer objects by overlap), and the HydraStarDist-paper metrics
    (score_hierarchy.paper_metrics).
StarDist's own choices: 96 rays (3D, GoldenSpiral, anisotropy of the training objects) or 32 (2D); U-Net backbone
(depth 3, pool 2, the StarDist default) or the ResNet of the 3d-oids recipe (3D).

Usage:
  python run_files.py --data DIR --target kidney --gpu 0 --like CONFIG --patch 48 192 192 --grid 1 2 2 [--name NAME]
  python run_files.py --data DIR --score --inner mass:NAME --outer kidney:NAME --gpu 0 [--split test val]
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
parser.add_argument('--target', help='object level to train (masks in <split>/<target>_masks)')
parser.add_argument('--gpu', default='')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--batch', type=int, default=2, help='patches per step (as image_batch_size of the CytoNucs run)')
parser.add_argument('--like', help='CytoNucs config whose augmentation values are used')
parser.add_argument('--name')
parser.add_argument('--patch', type=int, nargs='+', required=True, help='training patch (Z)YX; its length sets 2D or 3D')
parser.add_argument('--grid', type=int, nargs='+', help='StarDist grid (default 2 per axis, 1 along Z)')
parser.add_argument('--rays', type=int, help='default 96 (3D), 32 (2D)')
parser.add_argument('--backbone', default='unet', choices=('unet', 'resnet'))
parser.add_argument('--basedir', default='/storage02/miroslavm/cytonucs_paper_runs/stardist')
parser.add_argument('--cache', help='folder of the memory-mapped train volumes (default <basedir>/cache/<parent>_<data name>)')
parser.add_argument('--score', action='store_true')
parser.add_argument('--inner', help='LEVEL:MODEL_NAME of the inner level (with --score)')
parser.add_argument('--outer', help='LEVEL:MODEL_NAME of the outer level (with --score)')
parser.add_argument('--split', nargs='+', default=['test'])
parser.add_argument('--predictions', help='with --score: folder for the predicted labels (<split>/<level>/<name>)')
parser.add_argument('--drop-cache', action='store_true',
                    help='with --score: remove the memory-mapped train volumes of this dataset (--cache) afterwards')
parser.add_argument('--fixed-thresholds', type=float, nargs=2, metavar=('PROB', 'NMS'),
                    help='use these thresholds instead of optimising them on the val volumes (smoke tests)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import resource                                       # every memory-mapped volume holds a file descriptor
resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tifffile
import tensorflow as tf
from scipy.ndimage import find_objects
from stardist import fill_label_holes, Rays_GoldenSpiral
from stardist.models import Config2D, Config3D, StarDist2D, StarDist3D
from stardist.models.base import StarDistDataBase

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from cytonucs_data_generator import cytonucs_augmenter_3d, elastic_displacement, warp_yx
sys.path.insert(0, str(HERE.parent))
from patch_sampler import light_valid_inds
from cytonucs_runtime import normalize_percentiles as normalize

for device in tf.config.list_physical_devices('GPU'):
    if int(os.environ.get('GPU_MEMORY_LIMIT_MB', 0) or 0):          # a cap when the queue shares the GPU
        tf.config.set_logical_device_configuration(
            device, [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ['GPU_MEMORY_LIMIT_MB']))])
    else:
        tf.config.experimental.set_memory_growth(device, True)
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
UNKNOWN = 65535
DATA = Path(args.data)
PATCH = tuple(args.patch)
NDIM = len(PATCH)
GRID = tuple(args.grid) if args.grid else ((1, 2, 2) if NDIM == 3 else (2, 2))
CACHE = Path(args.cache) if args.cache else Path(args.basedir) / 'cache' / f'{DATA.resolve().parent.name}_{DATA.resolve().name}'


def names(split):
    return sorted(p.name for p in (DATA / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))


def read_image(path):
    with tifffile.TiffFile(path) as tif:
        x, axes = tif.asarray().astype(np.float32), tif.series[0].axes
    channel = [i for i, a in enumerate(axes) if a in 'CS']
    if channel and x.ndim > NDIM:
        x = np.moveaxis(x, channel[0], -1)                          # channels last (an ImageJ composite is C, Y, X)
    colour = NDIM == 2 and x.ndim == 3
    return normalize(x, 1, 99.8, axis=(0, 1) if colour else tuple(range(x.ndim))).astype(np.float32)


def read_labels(split, level, name):
    """Labels in the smallest unsigned type that holds them (uint8 for a few objects per volume: half the memory
    of the memory-mapped training data)."""
    y = tifffile.imread(DATA / split / f'{level}_masks' / name)
    y = fill_label_holes(np.where(y == UNKNOWN, 0, y).astype(np.uint16))
    return y.astype(np.uint8) if y.max() < 256 else y


IMAGE_CACHE = 'image_n2'          # n2: cytonucs_runtime.normalize_percentiles (caches of the plain csbdeep one are not reused)
LABEL_CACHE = 'min'               # labels in the smallest unsigned type (uint16 caches of earlier runs are not reused)


def finite_float16(x, name):
    x = np.asarray(x, np.float16)
    if not np.isfinite(x).all():
        raise ValueError(f'{name}: the normalised image does not fit float16 (inf / nan)')
    return x


def cached(split, level):
    """Memory-mapped (images, labels) of a split, written to CACHE on first use."""
    folder = CACHE / split
    folder.mkdir(parents=True, exist_ok=True)
    xs, ys = [], []
    # Files without masks of this level (partial annotation, e.g. 'two datasets') are no training material for it.
    for name in (n for n in names(split) if (DATA / split / f'{level}_masks' / n).is_file()):
        stem = Path(name).stem
        for path, make in ((folder / f'{stem}.{IMAGE_CACHE}.npy', lambda: finite_float16(read_image(DATA / split / 'images' / name), name)),
                           (folder / f'{stem}.{level}.{LABEL_CACHE}.npy', lambda: read_labels(split, level, name))):
            if not path.is_file():
                partial = folder / f'.{stem}.{os.getpid()}.npy'
                np.save(partial, make())
                os.replace(partial, path)
        xs.append(np.load(folder / f'{stem}.{IMAGE_CACHE}.npy', mmap_mode='r'))
        ys.append(np.load(folder / f'{stem}.{level}.{LABEL_CACHE}.npy', mmap_mode='r'))
    return xs, ys


valid_centres = light_valid_inds('Y')                              # same distribution, no index cache
StarDistDataBase.get_valid_inds = valid_centres


def make_augmenter(like):
    aug = json.loads(Path(like).read_text())['augmentation'] if like else {}
    cfg = SimpleNamespace(rotation_yx=aug.get('rotation_yx', True), flip_prob=aug.get('flip_prob', .5),
                          flip_z_prob=(aug.get('flip_z_prob') or 0.) if NDIM == 3 else 0.,
                          intensity_range=aug.get('intensity_range', (.8, 1.2)), intensity_offset=aug.get('intensity_offset', (-.1, .1)),
                          gamma_range=aug.get('gamma_range', (.9, 1.1)), noise_sigma=aug.get('noise_sigma', .01),
                          blur_sigma_range=aug.get('blur_sigma_range'), saturation_prob=aug.get('saturation_prob') or 0.)
    elastic = float(aug.get('elastic_prob', 0.)), float(aug.get('elastic_alpha', 0.)), float(aug.get('elastic_sigma', 8.))
    rng = np.random.default_rng(SEED)

    def augment(x, y):
        x, y = np.asarray(x, np.float32), np.asarray(y)
        if NDIM == 2:
            x, y = x[None], y[None]
        if elastic[0] and rng.random() < elastic[0]:
            dy, dx = elastic_displacement(y.shape[-2:], elastic[1], elastic[2], rng)
            x, y = warp_yx(x, dy, dx, 1, 3), warp_yx(y, dy, dx, 0, 3)
        x, y, _ = cytonucs_augmenter_3d(x, y, y, rng=rng, config=cfg)
        return (x[0], y[0]) if NDIM == 2 else (x, y)
    return augment, dict(cfg.__dict__, elastic=elastic)


def n_tiles(shape, channels):
    """Tiles of about twice the training patch per axis."""
    return tuple(max(1, int(np.ceil(s / (2 * p)))) for s, p in zip(shape[:NDIM], PATCH)) + ((1,) if channels else ())


def train():
    name = args.name or f'stardist{NDIM}d_{DATA.resolve().name}_{args.target}'
    started = time.time()
    x_train, y_train = cached('train', args.target)
    x_val, y_val = (lambda xs, ys: ([np.asarray(x, np.float32) for x in xs], [np.asarray(y) for y in ys]))(*cached('val', args.target))
    cache_s = time.time() - started
    extents = [[s.stop - s.start for s in box] for y in y_train for box in find_objects(np.asarray(y)) if box is not None]
    median = np.median(extents, axis=0)
    channels = x_train[0].ndim > NDIM
    augment, aug_values = make_augmenter(args.like)
    common = dict(n_channel_in=x_train[0].shape[-1] if channels else 1, grid=GRID, use_gpu=False, train_patch_size=PATCH,
                  train_background_reg=1e-4, train_dist_loss='mae', train_loss_weights=(1.0, 0.2), train_epochs=args.epochs,
                  train_steps_per_epoch=args.steps, train_learning_rate=3e-4, train_batch_size=args.batch, train_n_val_patches=None,
                  train_tensorboard=False, train_reduce_lr=dict(factor=0.5, patience=40, min_delta=0))
    if NDIM == 3:
        anisotropy = tuple(float(v) for v in np.max(median) / median)
        backbone = (dict(backbone='resnet', resnet_n_blocks=4, resnet_kernel_size=(3, 3, 3), resnet_kernel_init='he_normal',
                         resnet_n_filter_base=32, resnet_n_conv_per_block=3, resnet_activation='relu', resnet_batch_norm=False,
                         net_conv_after_resnet=128) if args.backbone == 'resnet' else
                    dict(backbone='unet', unet_n_depth=3, unet_kernel_size=(3, 3, 3), unet_n_filter_base=32, unet_pool=(2, 2, 2),
                         net_conv_after_unet=128))
        conf = Config3D(axes='ZYXC' if channels else 'ZYX', rays=Rays_GoldenSpiral(args.rays or 96, anisotropy=anisotropy),
                        anisotropy=anisotropy, **backbone, **common)
        model = StarDist3D(conf, name=name, basedir=args.basedir)
    else:
        anisotropy = None
        conf = Config2D(axes='YXC' if channels else 'YX', n_rays=args.rays or 32, backbone='unet', unet_n_depth=3,
                        unet_kernel_size=(3, 3), unet_n_filter_base=32, unet_pool=(2, 2), net_conv_after_unet=128, **common)
        model = StarDist2D(conf, name=name, basedir=args.basedir)
    params = int(sum(int(np.prod(v.shape)) for v in model.keras_model.trainable_weights))
    fov = [int(v) for v in model._axes_tile_overlap('ZYX' if NDIM == 3 else 'YX')]
    print(f'{args.target}: {len(x_train)} train volumes, {len(x_val)} val volumes, median object extent {median}, '
          f'anisotropy {anisotropy}, field of view {fov}, {params} parameters, cache {cache_s:.0f} s')
    tf.config.experimental.reset_memory_stats('GPU:0') if tf.config.list_logical_devices('GPU') else None
    started = time.time()
    history = model.train(x_train, y_train, validation_data=(x_val, y_val), augmenter=augment,
                          epochs=args.epochs, steps_per_epoch=args.steps)
    hours = (time.time() - started) / 3600
    peak = (tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2 ** 30) if tf.config.list_logical_devices('GPU') else None
    started = time.time()
    tiles = n_tiles(np.max([x.shape for x in x_val], axis=0), channels)
    if args.fixed_thresholds:
        thresholds = dict(prob=args.fixed_thresholds[0], nms=args.fixed_thresholds[1])
        model.thresholds = thresholds
        (Path(args.basedir) / name / 'thresholds.json').write_text(json.dumps(thresholds))
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            thresholds = model.optimize_thresholds(x_val, y_val, predict_kwargs=dict(n_tiles=tiles, show_tile_progress=False))
    info = dict(target=args.target, name=name, data=str(DATA), ndim=NDIM, epochs=args.epochs, steps_per_epoch=args.steps,
                patch=list(PATCH), batch=args.batch, grid=list(GRID), rays=conf.n_rays, backbone=args.backbone,
                anisotropy=None if anisotropy is None else list(anisotropy), median_extent=median.tolist(),
                train_volumes=len(x_train), val_volumes=len(x_val), augmentation=aug_values, parameters=params,
                field_of_view=fov, cache_s=cache_s, training_hours=hours, gpu_peak_mem_gb=peak,
                threshold_optimisation_s=time.time() - started, fixed_thresholds=bool(args.fixed_thresholds), thresholds={k: float(v) for k, v in dict(thresholds).items()},
                val_loss=[float(v) for v in history.history.get('val_loss', [])],
                weights=str(Path(args.basedir) / name / 'weights_best.h5'))
    (Path(args.basedir) / name / 'training.json').write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps({k: info[k] for k in ('training_hours', 'gpu_peak_mem_gb', 'thresholds', 'parameters')}))


def score():
    from score_hierarchy import ground_truth, save_prediction, score_image, summarise
    (inner, inner_name), (outer, outer_name) = (args.inner.split(':'), args.outer.split(':'))
    kind = StarDist3D if NDIM == 3 else StarDist2D
    models = {inner: kind(None, name=inner_name, basedir=args.basedir), outer: kind(None, name=outer_name, basedir=args.basedir)}
    results = {}
    for split in args.split:
        records = []
        for name in names(split):
            x = read_image(DATA / split / 'images' / name)
            pred = {}
            started = time.time()
            for level, model in models.items():
                pred[level] = model.predict_instances(x, n_tiles=n_tiles(x.shape, x.ndim > NDIM), show_tile_progress=False)[0]
            seconds = time.time() - started
            gt, truths, ignore = ground_truth(DATA / split, name, [inner, outer], [(inner, outer)])
            record, scored = score_image(gt, pred, [(inner, outer)], truths, ignore=ignore)      # pairs by overlap
            records.append(dict(record, name=name, seconds=seconds))
            if args.predictions:
                save_prediction(Path(args.predictions) / split, name, scored)
        result = dict(split=split, volumes=len(records), seconds_per_volume=float(np.mean([r['seconds'] for r in records])),
                      models={inner: inner_name, outer: outer_name},
                      predictions=str(Path(args.predictions) / split) if args.predictions else None)
        result.update(summarise(records, [inner, outer], [(inner, outer)]))
        result['records'] = records
        results[split] = result
        print(f"{split}: " + ' | '.join(f"{level} {result['objects'][level]}: AP {result[f'ap_{level}']:.3f} "
                                        f"(AP50 {result[f'ap50_{level}']:.3f})" for level in models) +
              f" | JTPR {result[f'jtpr_{inner}_{outer}'] or 0:.3f} of {result[f'pairs_{inner}_{outer}']}"
              f" | {result['seconds_per_volume']:.1f} s per volume")
    out = Path(args.basedir) / f'files_{inner_name}_{outer_name}.json'
    out.write_text(json.dumps(results, indent=1))
    print('->', out)
    if args.drop_cache:
        drop_cache()


def drop_cache():
    """Removes the cache folder of this dataset, only if it is a folder directly inside a folder named 'cache'."""
    import shutil
    cache = CACHE.resolve()
    if cache.parent.name != 'cache' or not cache.is_dir():
        print(f'cache {cache} not removed (not a dataset folder inside a cache folder)')
        return
    shutil.rmtree(cache)
    print('removed cache', cache)


if __name__ == '__main__':
    score() if args.score else train()
