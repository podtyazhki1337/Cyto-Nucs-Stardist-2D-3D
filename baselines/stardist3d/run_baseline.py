"""Plain StarDist-3D baseline on a CytoNucs spatial split, one model per object level.

Recipe of /storage02/miroslavm/Stardist/3d-oids/train_sd.py + config_3doids.py (official StarDist 3D demo
configuration): ResNet backbone, 96 rays, grid (1, 2, 2), batch 2, 400 epochs x 100 steps, learning rate 3e-4
halved on plateau, one fixed rescale factor estimated from the TRAINING masks (target object size 5 x 28 x 28)
and applied unchanged to train, val and test, percentile normalisation per tile.

What differs from that script, and why:
  - 3 input channels (the tiles are RGB; that script reads one channel);
  - train / val / test are the folders of the spatial split, not a fold file;
  - ray anisotropy is taken from the rescaled training objects (the canonical StarDist rule, as in
    /storage02/miroslavm/Stardist/train.py); config_3doids.py fixes (2, 1, 1) for its own collection;
  - patch (16, 96, 96): the tiles are 16 slices deep, that config uses 48;
  - nothing is sent to Comet;
  - evaluation is the CytoNucs one: predictions are resized back to the tile and scored at native resolution
    with AP = mean over IoU 0.50..0.95 of TP / (TP + FP + FN), over all objects and over whole objects
    (cytonucs_trainer.without_xy_face_objects).

Usage: python run_baseline.py --target nucleus --gpu 1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--target', required=True, choices=('nucleus', 'cytoplasm'))
parser.add_argument('--data', default='/storage02/miroslavm/data/Cytonucs_Emese_Hela/patches_split_hela')
parser.add_argument('--gpu', default='')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--aug', default='LIGHT_AUG', choices=('NO_AUG', 'LIGHT_AUG', 'FULL'))
parser.add_argument('--name', default=None)
parser.add_argument('--limit', type=int, default=0, help='read only this many tiles per split (smoke test)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tifffile
import skimage.measure
from skimage.transform import rescale, resize
import tensorflow as tf
from csbdeep.utils import normalize
from stardist import fill_label_holes, Rays_GoldenSpiral
from stardist.matching import matching
from stardist.models import Config3D, StarDist3D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))                        # the CytoNucs repository: its metric helpers
import augmentations
from cytonucs_runtime import accuracy_profile
from cytonucs_trainer import without_xy_face_objects

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

TARGET_SIZE = np.array((5.0, 28.0, 28.0))                       # config_3doids.target_cell_size
PATCH = (16, 96, 96)
TAUS = np.round(np.arange(0.5, 0.96, 0.05), 2)
MASKS = dict(nucleus='nucleus_masks', cytoplasm='cytoplasm_masks')[args.target]
DATA = Path(args.data)
NAME = args.name or f'stardist3d_hela_{args.target}'
OUT = HERE / 'models' / NAME


def names(split):
    found = sorted(p.name for p in (DATA / split / 'images').iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    return found[:args.limit] if args.limit else found


def mean_extent(mask):
    """Mean bounding-box extent (ZYX) of the objects of one tile, as _measure_cell_means of train_sd.py."""
    boxes = [p.bbox for p in skimage.measure.regionprops(mask)]
    return np.mean([[b[d + 3] - b[d] for d in range(3)] for b in boxes], axis=0) if boxes else None


def load(split, factor):
    """Tiles as the network sees them: rescaled, padded to the patch, normalised; plus the native labels."""
    images, labels, native = [], [], []
    for name in names(split):
        image = tifffile.imread(DATA / split / 'images' / name)
        mask = tifffile.imread(DATA / split / MASKS / name)
        native.append(dict(name=name, mask=mask))
        image = rescale(image, tuple(factor) + (1,), order=1, anti_aliasing=False, preserve_range=True).astype(np.float32)
        small = fill_label_holes(rescale(mask, factor, order=0, anti_aliasing=False, preserve_range=True).astype(np.uint16))
        native[-1]['rescaled_shape'] = small.shape
        pad = [(0, max(0, p - s)) for s, p in zip(small.shape, PATCH)]
        images.append(normalize(np.pad(image, pad + [(0, 0)]), 1, 99.8, axis=(0, 1, 2)))
        labels.append(np.pad(small, pad))
    return images, labels, native


def evaluate(model, split, images, native, folder):
    counts = {part: np.zeros((3, len(TAUS)), np.int64) for part in ('', '_whole')}
    objects = dict(gt=0, gt_whole=0, predicted=0)
    (folder / split).mkdir(parents=True, exist_ok=True)
    for image, item in zip(images, native):
        predicted = model.predict_instances(image, n_tiles=model._guess_n_tiles(image), show_tile_progress=False)[0]
        predicted = predicted[tuple(slice(0, s) for s in item['rescaled_shape'])]
        predicted = resize(predicted, item['mask'].shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)
        tifffile.imwrite(folder / split / item['name'], predicted, photometric='minisblack')
        gt = item['mask']
        whole = without_xy_face_objects(gt, predicted)
        objects['gt'] += len(np.unique(gt)) - 1; objects['gt_whole'] += len(np.unique(whole[0])) - 1
        objects['predicted'] += len(np.unique(predicted)) - 1
        for part, (g, p) in (('', (gt, predicted)), ('_whole', whole)):
            for k, r in enumerate(matching(g, p, thresh=list(TAUS), criterion='iou')):
                counts[part][:, k] += (int(r.tp), int(r.fp), int(r.fn))
    result = dict(tiles=len(images), **objects)
    for part, c in counts.items():
        accuracy = accuracy_profile(*c)
        result['ap' + part] = float(accuracy.mean()); result['ap50' + part] = float(accuracy[0])
        result['tp50' + part], result['fp50' + part], result['fn50' + part] = (int(v) for v in c[:, 0])
    return result


if __name__ == '__main__':
    augmentations.set_config(dict(NO_AUG=augmentations.NO_AUG_CONFIG, LIGHT_AUG=augmentations.LIGHT_AUG_CONFIG,
                                  FULL=augmentations.FULL_CONFIG)[args.aug])
    extents = [e for e in (mean_extent(tifffile.imread(DATA / 'train' / MASKS / n)) for n in names('train')) if e is not None]
    raw_median = np.median(extents, axis=0)
    factor = TARGET_SIZE / raw_median
    print(f'{args.target}: median object extent of the train tiles (ZYX) {raw_median.round(1)}, rescale factor {factor.round(3)}')

    x_train, y_train, _ = load('train', factor)
    x_val, y_val, native_val = load('val', factor)
    x_test, _, native_test = load('test', factor)
    print(f'train {len(x_train)} tiles of {x_train[0].shape}, val {len(x_val)}, test {len(x_test)}')

    rescaled = [e for e in (mean_extent(y) for y in y_train) if e is not None]
    median = np.median(rescaled, axis=0)
    anisotropy = tuple(float(v) for v in np.max(median) / median)
    grid = (1, 2, 2)
    conf = Config3D(
        axes='ZYXC', rays=Rays_GoldenSpiral(96, anisotropy=anisotropy), grid=grid, anisotropy=anisotropy,
        backbone='resnet', use_gpu=False, n_channel_in=x_train[0].shape[-1],
        resnet_n_blocks=4, resnet_kernel_size=(3, 3, 3), resnet_kernel_init='he_normal', resnet_n_filter_base=32,
        resnet_n_conv_per_block=3, resnet_activation='relu', resnet_batch_norm=False, net_conv_after_resnet=128,
        train_patch_size=PATCH, train_background_reg=1e-4, train_dist_loss='mae', train_loss_weights=(1.0, 0.2),
        train_epochs=args.epochs, train_steps_per_epoch=args.steps, train_learning_rate=3e-4, train_batch_size=2,
        train_n_val_patches=None, train_tensorboard=False, train_reduce_lr=dict(factor=0.5, patience=40, min_delta=0))
    print(conf)
    model = StarDist3D(conf, name=NAME, basedir=str(HERE / 'models'))
    print(f'median object extent after rescale {median.round(1)}, anisotropy {np.round(anisotropy, 2)}, '
          f'field of view {np.array(model._axes_tile_overlap("ZYX"))}')

    started = time.time()
    model.train(x_train, y_train, validation_data=(x_val, y_val), augmenter=augmentations.augmenter,
                epochs=args.epochs, steps_per_epoch=args.steps)
    hours = (time.time() - started) / 3600
    thresholds = model.optimize_thresholds(x_val, y_val)
    print('thresholds', thresholds, f'| training {hours:.2f} h')

    result = dict(target=args.target, data=str(DATA), name=NAME, epochs=args.epochs, steps_per_epoch=args.steps,
                  augmentation=args.aug, train_tiles=len(x_train), raw_median_extent_zyx=raw_median.tolist(),
                  rescale_factor_zyx=factor.tolist(), anisotropy=list(anisotropy), grid=list(grid), patch=list(PATCH),
                  thresholds={k: float(v) for k, v in dict(thresholds).items()},
                  training_hours=hours,
                  val=evaluate(model, 'val', x_val, native_val, OUT / 'predictions'),
                  test=evaluate(model, 'test', x_test, native_test, OUT / 'predictions'))
    (OUT / 'result.json').write_text(json.dumps(result, indent=2))
    for split in ('val', 'test'):
        r = result[split]
        print(f"{args.target} {split}: AP {r['ap']:.3f} (AP50 {r['ap50']:.3f}) | whole objects AP {r['ap_whole']:.3f} "
              f"(AP50 {r['ap50_whole']:.3f}) | GT {r['gt']} ({r['gt_whole']} whole), predicted {r['predicted']}")
