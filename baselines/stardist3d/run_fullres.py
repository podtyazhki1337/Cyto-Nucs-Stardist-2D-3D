"""Plain StarDist-3D at FULL resolution, trained like CytoNucs, one model per object level, scored with the
CytoNucs whole-block protocol (evaluate_blocks.py).

Same as the CytoNucs run it is compared with:
  - data: the train blocks of the source volume (patches_split_hela/train_regions + regions.json), each
    block a training volume over the full Z range, normalised per block (percentiles 1 / 99.8 per channel);
    validation volumes are the val blocks; nothing is rescaled;
  - patch 16 x 320 x 320, batch 2, 100 steps per epoch, --epochs epochs;
  - augmentation: the CytoNucs functions (flips and 90 degree rotations in XY, intensity scale / offset /
    gamma, noise, blur, elastic XY deformation) with the values of --like (a CytoNucs config);
  - scoring: every held-out block predicted over the full Z range with 192 px of context, objects by
    centroid, AP over IoU 0.5..0.95, JTPR of the two models (predicted nuclei paired with predicted cells by
    overlap, as cytonucs_inference.compute_jtpr does without predicted pairs).
StarDist's own choices: 96 rays (GoldenSpiral) with the anisotropy of the training objects, ResNet backbone of
the user's 3d-oids recipe, grid (1, 4, 4) for objects 50-200 px wide, thresholds optimised on the val blocks.

Usage:
  python run_fullres.py --target nucleus --gpu 0 --like ../../_exp/cyto_fullres.json [--epochs 300]
  python run_fullres.py --score --gpu 0 --nucleus-model NAME --cytoplasm-model NAME
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--target', choices=('nucleus', 'cytoplasm'))
parser.add_argument('--data', default='/storage02/miroslavm/data/Cytonucs_Emese_Hela/patches_split_hela')
parser.add_argument('--gpu', default='')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--like', help='CytoNucs config whose augmentation values are used')
parser.add_argument('--name')
parser.add_argument('--score', action='store_true', help='score two trained models on the val and test blocks')
parser.add_argument('--nucleus-model')
parser.add_argument('--cytoplasm-model')
parser.add_argument('--margin', type=int, default=192)
parser.add_argument('--patch', type=int, nargs=3, default=[16, 320, 320], help='training patch ZYX (160 for half resolution)')
parser.add_argument('--grid', type=int, nargs=3, default=[1, 4, 4], help='StarDist grid ZYX ((1, 2, 2) for half resolution)')
parser.add_argument('--backbone', default='resnet', choices=('resnet', 'unet'),
                    help='resnet: the 3d-oids recipe (field of view 17x49x49 px); unet: depth 3, pool (2,2,2) (32x128x128 px)')
parser.add_argument('--basedir', default='/storage02/miroslavm/cytonucs_paper_runs/stardist',
                    help='where the models are written and read')
parser.add_argument('--unknown-as-background', action='store_true',
                    help='masks with training.unknown_label 65535 (sparse annotation): StarDist cannot ignore voxels, so they become background')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tifffile
import skimage.measure
import tensorflow as tf
from csbdeep.utils import normalize
from stardist import fill_label_holes, Rays_GoldenSpiral
from stardist.matching import matching
from stardist.models import Config3D, StarDist3D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from cytonucs_data_generator import cytonucs_augmenter_3d, elastic_displacement, warp_yx
from cytonucs_runtime import accuracy_profile
from evaluate_blocks import block_windows, objects_in_block

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
PATCH = tuple(args.patch)
GRID = tuple(args.grid)
TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MASKS = dict(nucleus='nucleus_masks', cytoplasm='cytoplasm_masks')
DATA = Path(args.data)


def volume():
    regions = json.loads((DATA / 'train_regions' / 'regions.json').read_text())
    name = next(iter(regions))
    return name, regions[name]


def downscaling():
    """(native dataset, factor): a copy written by dataset_prep/downscale_split.py keeps its native dataset."""
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    factor = int(manifest.get('xy_downscale', 1))
    return (Path(manifest['downscaled_from']) if factor > 1 else DATA), factor


def blocks(split):
    """(y0, y1, x0, x1) of the blocks of a split, in the coordinates of DATA (native blocks / factor)."""
    native, factor = downscaling()
    manifest = json.loads((native / 'split_manifest.json').read_text())
    return [(b['y0'] // factor, b['y1'] // factor, b['x0'] // factor, b['x1'] // factor)
            for v in manifest['volumes'] for b in v['blocks'] if b['split'] == split]


def block_volumes(split, target):
    name, _ = volume()
    image = tifffile.imread(DATA / 'train_regions' / 'images' / name)
    labels = tifffile.imread(DATA / 'train_regions' / MASKS[target] / name)
    xs, ys = [], []
    for y0, y1, x0, x1 in blocks(split):
        xs.append(normalize(image[:, y0:y1, x0:x1].astype(np.float32), 1, 99.8, axis=(0, 1, 2)).astype(np.float32))
        block = labels[:, y0:y1, x0:x1]
        if args.unknown_as_background:
            block = np.where(block == 65535, 0, block)
        ys.append(fill_label_holes(block.astype(np.uint16)))
    return xs, ys


def make_augmenter(like):
    aug = json.loads(Path(like).read_text())['augmentation'] if like else {}
    cfg = SimpleNamespace(rotation_yx=aug.get('rotation_yx', True), flip_prob=aug.get('flip_prob', .5),
                          intensity_range=aug.get('intensity_range', (.8, 1.2)), intensity_offset=aug.get('intensity_offset', (-.1, .1)),
                          gamma_range=aug.get('gamma_range', (.9, 1.1)), noise_sigma=aug.get('noise_sigma', .01),
                          blur_sigma_range=aug.get('blur_sigma_range'))
    elastic = float(aug.get('elastic_prob', 0.)), float(aug.get('elastic_alpha', 0.)), float(aug.get('elastic_sigma', 8.))
    rng = np.random.default_rng(SEED)

    def augment(x, y):
        if elastic[0] and rng.random() < elastic[0]:
            dy, dx = elastic_displacement(y.shape[-2:], elastic[1], elastic[2], rng)
            x, y = warp_yx(x, dy, dx, 1, 3), warp_yx(y, dy, dx, 0, 3)
        x, y, _ = cytonucs_augmenter_3d(x, y, y, rng=rng, config=cfg)
        return x, y
    return augment, dict(cfg.__dict__, elastic=elastic)


def train():
    name = args.name or f'stardist3d_fullres_{args.target}'
    x_train, y_train = block_volumes('train', args.target)
    keep = [i for i, y in enumerate(y_train) if y.max() > 0]
    x_train, y_train = [x_train[i] for i in keep], [y_train[i] for i in keep]
    x_val, y_val = block_volumes('val', args.target)
    extents = [[b[d + 3] - b[d] for d in range(3)] for y in y_train for b in (p.bbox for p in skimage.measure.regionprops(y))]
    median = np.median(extents, axis=0)
    anisotropy = tuple(float(v) for v in np.max(median) / median)
    augment, aug_values = make_augmenter(args.like)
    backbone = (dict(backbone='resnet', resnet_n_blocks=4, resnet_kernel_size=(3, 3, 3), resnet_kernel_init='he_normal',
                     resnet_n_filter_base=32, resnet_n_conv_per_block=3, resnet_activation='relu', resnet_batch_norm=False,
                     net_conv_after_resnet=128) if args.backbone == 'resnet' else
                dict(backbone='unet', unet_n_depth=3, unet_kernel_size=(3, 3, 3), unet_n_filter_base=32, unet_pool=(2, 2, 2),
                     net_conv_after_unet=128))
    conf = Config3D(
        axes='ZYXC', rays=Rays_GoldenSpiral(96, anisotropy=anisotropy), grid=GRID, anisotropy=anisotropy,
        use_gpu=False, n_channel_in=x_train[0].shape[-1], **backbone,
        train_patch_size=PATCH, train_background_reg=1e-4, train_dist_loss='mae', train_loss_weights=(1.0, 0.2),
        train_epochs=args.epochs, train_steps_per_epoch=args.steps, train_learning_rate=3e-4, train_batch_size=2,
        train_n_val_patches=None, train_tensorboard=False, train_reduce_lr=dict(factor=0.5, patience=40, min_delta=0))
    model = StarDist3D(conf, name=name, basedir=args.basedir)
    params = int(sum(int(np.prod(v.shape)) for v in model.keras_model.trainable_weights))
    print(f'{args.target}: {len(x_train)} train blocks {x_train[0].shape}, median object extent {median}, anisotropy '
          f'{np.round(anisotropy, 2)}, field of view {np.array(model._axes_tile_overlap("ZYX"))}, {params} parameters')
    tf.config.experimental.reset_memory_stats('GPU:0') if tf.config.list_logical_devices('GPU') else None
    started = time.time()
    history = model.train(x_train, y_train, validation_data=(x_val, y_val), augmenter=augment,
                          epochs=args.epochs, steps_per_epoch=args.steps)
    hours = (time.time() - started) / 3600
    peak = (tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2 ** 30) if tf.config.list_logical_devices('GPU') else None
    started = time.time()
    thresholds = model.optimize_thresholds(x_val, y_val)
    info = dict(target=args.target, name=name, epochs=args.epochs, steps_per_epoch=args.steps, patch=list(PATCH),
                grid=list(GRID), anisotropy=list(anisotropy), median_extent=median.tolist(), train_blocks=len(x_train),
                augmentation=aug_values, parameters=params, training_hours=hours, gpu_peak_mem_gb=peak,
                threshold_optimisation_s=time.time() - started,
                thresholds={k: float(v) for k, v in dict(thresholds).items()},
                val_loss=[float(v) for v in history.history.get('val_loss', [])])
    info.update(backbone=args.backbone, data=str(DATA), unknown_as_background=args.unknown_as_background,
                field_of_view=[int(v) for v in model._axes_tile_overlap('ZYX')], weights=str(Path(args.basedir) / name / 'weights_best.h5'))
    (Path(args.basedir) / name / 'training.json').write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps({k: info[k] for k in ('training_hours', 'gpu_peak_mem_gb', 'thresholds', 'parameters')}))


def score():
    from cytonucs_inference import compute_jtpr
    import contextlib, io
    models = {t: StarDist3D(None, name=n, basedir=args.basedir)
              for t, n in (('nucleus', args.nucleus_model), ('cytoplasm', args.cytoplasm_model))}
    from score_native_resolution import enlarge_labels
    name, _ = volume()
    native, factor = downscaling()
    image = tifffile.imread(DATA / 'train_regions' / 'images' / name)
    gt_all = {t: tifffile.imread(native / 'train_regions' / MASKS[t] / name) for t in MASKS}
    pairs = {int(k): int(v) for k, v in json.loads((native / 'train_regions' / 'assignments' / (Path(name).stem + '.json')).read_text()).items()}
    manifest = json.loads((native / 'split_manifest.json').read_text())
    results = {}
    for split in ('test', 'val'):
        counts = {t: np.zeros((3, len(TAUS)), np.int64) for t in MASKS}
        joint, objects, timing = [0, 0], {t: 0 for t in MASKS}, []
        for _, block, window in block_windows(manifest, split, args.margin, factor):
            y0, y1, x0, x1 = window
            x = normalize(image[:, y0 // factor:y1 // factor, x0 // factor:x1 // factor].astype(np.float32), 1, 99.8,
                          axis=(0, 1, 2)).astype(np.float32)
            pred = {}
            for t, model in models.items():
                started = time.time()
                n_tiles = (1,) + tuple(int(np.ceil(s / p)) for s, p in zip(x.shape[1:3], PATCH[1:])) + (1,)
                pred[t] = enlarge_labels(model.predict_instances(x, n_tiles=n_tiles, show_tile_progress=False)[0], factor,
                                         (x.shape[0], y1 - y0, x1 - x0)) if factor > 1 else \
                    model.predict_instances(x, n_tiles=n_tiles, show_tile_progress=False)[0]
                timing.append(dict(target=t, block=list(block), seconds=round(time.time() - started, 1)))
            gt = {t: objects_in_block(gt_all[t][:, y0:y1, x0:x1], block, window) for t in MASKS}
            pr = {t: objects_in_block(pred[t].astype(np.int32), block, window) for t in MASKS}
            for t in MASKS:
                objects[t] += int(len(np.unique(gt[t])) - 1)
                for k, r in enumerate(matching(gt[t], pr[t], thresh=TAUS, criterion='iou')):
                    counts[t][:, k] += (int(r.tp), int(r.fp), int(r.fn))
            with contextlib.redirect_stdout(io.StringIO()):
                _, details = compute_jtpr(pr['nucleus'], pr['cytoplasm'], gt['nucleus'], gt['cytoplasm'], iou_thresh=.5,
                                          assignments_gt=pairs)
            joint[0] += int(details['joint_tp']); joint[1] += int(details['total_pairs'])
        result = dict(jtpr=joint[0] / max(joint[1], 1), pairs=joint[1], objects=objects, timing=timing)
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
    out = Path(args.basedir) / f'blocks_{args.nucleus_model}_{args.cytoplasm_model}.json'
    out.write_text(json.dumps(results, indent=1))
    print('->', out)


if __name__ == '__main__':
    score() if args.score else train()
