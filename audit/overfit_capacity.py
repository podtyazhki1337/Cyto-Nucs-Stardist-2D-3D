"""Can the real pipeline reproduce the boundaries of data it trains on? (synthetic CPU run, not a user-data result)

The real entry point (train_cytonucs.main) with the v3.2 preset scaled down trains on a few fixed synthetic
volumes and validates on the SAME volumes, like the user's validation patches that share their objects with
train. The GT has per-slice contour detail that the blurred image does not show, so the boundaries cannot be
predicted from the image - they can only be fitted.
Variants change one factor at a time: the number of training volumes, whether validation volumes are the
training ones or new ones from the same generator, and the production augmentation / learning-rate schedule.
history.json of each run is summarised into audit/overfit_capacity.json: semantic Dice, ray-distance loss,
mask loss, mean mask IoU of the training prompts (quality target) and the full-validation AP / JTPR.

Usage: python audit/overfit_capacity.py [variant ...]
"""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

PRESET = 'config_cytonucs_v3_2.json'
# XY scale of the objects: 1 -> nuclei ~13 px, cells ~22 px; 3 -> ~40 and ~70 px (half of the user's data).
SCALE = int(os.environ.get('OVERFIT_SCALE', 1))
SHAPE = (16, 64 * SCALE, 64 * SCALE)
EPOCHS, STEPS = int(os.environ.get('OVERFIT_EPOCHS', 40)), int(os.environ.get('OVERFIT_STEPS', 50))
OUT = ROOT / 'audit' / 'overfit_capacity.json'


def wobble(rng, theta, amplitude):
    """Smooth random radial factor of a closed contour."""
    out = np.ones_like(theta)
    for k in (1, 2, 3):
        out += amplitude / k * rng.normal() * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    return out


def volume(rng):
    """Touching cells with one nucleus each, a few slices thick. The image is rendered from smooth
    ellipsoids and blurred; the labels get an independent contour wobble in every slice."""
    zz, yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]].astype(np.float32)
    centers = [(rng.uniform(6.5, 9.5), SCALE * (cy * 21 + 11 + rng.uniform(-3, 3)), SCALE * (cx * 21 + 11 + rng.uniform(-3, 3)))
               for cy in range(3) for cx in range(3)]
    nearest = np.argmin([(yy - c[1]) ** 2 + (xx - c[2]) ** 2 for c in centers], axis=0)
    cells = np.zeros(SHAPE, np.uint16); nuclei = np.zeros(SHAPE, np.uint16)
    smooth_cell = np.zeros(SHAPE, np.float32); smooth_nucleus = np.zeros(SHAPE, np.float32)
    pairs = {}
    for k, c in enumerate(centers, 1):
        theta = np.arctan2(yy[0] - c[1], xx[0] - c[2]); rho = np.hypot(yy[0] - c[1], xx[0] - c[2])
        rz_cell, r_cell = rng.uniform(3.0, 3.6), SCALE * rng.uniform(12, 14)
        rz_nuc, r_nuc = rng.uniform(2.0, 2.5), SCALE * rng.uniform(5.5, 7.5)
        for z in range(SHAPE[0]):
            for rz, r, labels, smooth, amplitude, lab in ((rz_cell, r_cell, cells, smooth_cell, .06, k),
                                                          (rz_nuc, r_nuc, nuclei, smooth_nucleus, .10, 100 + k)):
                h = 1 - ((z - c[0]) / rz) ** 2
                if h <= 0:
                    continue
                radius = r * np.sqrt(h)
                own = nearest[z] == k - 1
                smooth[z][(rho <= radius) & own] = 1.
                labels[z][(rho <= radius * wobble(rng, theta, amplitude)) & own & ((labels[z] == 0) | (labels is nuclei))] = lab
        nuclei[(nuclei == 100 + k) & (cells != k)] = 0          # nuclei stay inside their cell
        pairs[100 + k] = k
    image = .1 + .35 * smooth_cell + .55 * smooth_nucleus
    image = gaussian_filter(image, (0.8, 2.5 * SCALE, 2.5 * SCALE)) + rng.normal(0, .03, SHAPE)
    return image.astype(np.float32), nuclei, cells, pairs


def make_data(root, train, validation, seed=7):
    """validation: 'seen' - the first volumes of train (objects the network trains on);
    'unseen' - new volumes from the same generator."""
    rng = np.random.default_rng(seed)
    items = [volume(rng) for _ in range(train)]
    held_out = [volume(np.random.default_rng(seed + 1000)) for _ in range(VALIDATION)] if validation == 'unseen' else None
    for split, volumes in (('train', items), ('val', held_out or items[:VALIDATION])):
        for sub in ('images', 'nucleus_masks', 'cytoplasm_masks', 'assignments'):
            (root / 'data' / split / sub).mkdir(parents=True)
        for i, (image, nuclei, cells, pairs) in enumerate(volumes):
            for sub, arr in (('images', image), ('nucleus_masks', nuclei), ('cytoplasm_masks', cells)):
                tifffile.imwrite(root / 'data' / split / sub / f'vol{i}.tif', arr, photometric='minisblack',
                                 metadata={'axes': 'ZYX'})
            (root / 'data' / split / 'assignments' / f'vol{i}.json').write_text(json.dumps(pairs))


VALIDATION = 3
PRODUCTION = dict(augmentation=dict(enable=True), training=dict(initial_learning_rate=3e-4, lr_decay_rate=0.85,
                                                                 lr_decay_steps=5000))
# name: (training volumes, validation volumes, config changes)
VARIANTS = {
    'fit': (3, 'seen', {}),                         # no augmentation, constant step: "fit what you see"
    'production_training': (3, 'seen', PRODUCTION),  # production augmentation and learning-rate schedule
    'unseen': (3, 'unseen', {}),                    # same training, validation on volumes never trained on
    'many_seen': (48, 'seen', {}),                  # 16 times more training volumes, same network
    'many_seen_production': (48, 'seen', PRODUCTION),
    'many_unseen': (48, 'unseen', {}),
    # the two production factors separately
    'many_seen_augmentation': (48, 'seen', dict(augmentation=PRODUCTION['augmentation'])),
    'many_seen_schedule': (48, 'seen', dict(training=PRODUCTION['training'])),
}


def config(root, name, changes):
    raw = json.loads((ROOT / PRESET).read_text(encoding='utf-8'))
    raw['experiment'].update(name=name, use_comet=False)
    raw['data']['dataset_dir'] = str(root / 'data')
    raw['output']['basedir'] = str(root / 'models')
    raw['model'].update(n_rays=32, n_channel_in=1, use_gpu=False, unet_n_filter_base=8, net_conv_after_unet=32,
                        mask_refinement_filters=16)
    raw['training'].update(train_patch_size=list(SHAPE), steps_per_epoch=STEPS, train_epochs=EPOCHS,
                           resume_from_checkpoint=None, reset_best_on_resume=False, initial_learning_rate=1e-3,
                           lr_decay_rate=1.0, prefetch_workers=1, prefetch_depth=1)
    raw['augmentation']['enable'] = False
    raw['inference'].update(refinement_patch_size=[0, SHAPE[1], SHAPE[2]], eval_splits=['val'], mask_thresh_nucleus=0.45,
                            mask_thresh_cytoplasm=0.45,
                            checkpoint_path=str(root / 'models' / name / 'weights_best.weights.h5'))
    raw['evaluation'].update(eval_dice_every=10, save_every_n_epochs=EPOCHS, min_eval_epoch=0)
    raw['gpu_config']['visible_devices'] = None
    for section, values in changes.items():
        raw[section].update(values)
    raw['training'].update(train_epochs=EPOCHS, steps_per_epoch=STEPS)
    path = root / f'config_{name}.json'
    path.write_text(json.dumps(raw, indent=2), encoding='utf-8')
    return path


def summary(history):
    def tail(key, span=5):
        values = [v for v in history.get(key, [])[-span:] if v is not None]
        return float(np.mean(values)) if values else None
    out = {k: tail('train_' + k) for k in (
        'nucleus_semantic_dice', 'cytoplasm_semantic_dice', 'nucleus_distance', 'cytoplasm_distance', 'nucleus_mask',
        'cytoplasm_mask', 'nucleus_fine_mask', 'cytoplasm_fine_mask', 'nucleus_quality_target_mean',
        'cytoplasm_quality_target_mean', 'loss')}
    out['curves'] = {k: [None if v is None else round(float(v), 4) for v in history.get('train_' + k, [])[4::5]]
                     for k in ('nucleus_semantic_dice', 'cytoplasm_semantic_dice', 'nucleus_distance', 'nucleus_mask',
                               'cytoplasm_mask', 'nucleus_quality_target_mean', 'cytoplasm_quality_target_mean')}
    out['fullval'] = {k: history.get('fullval_' + k) for k in (
        'epoch', 'jtpr', 'ap_nucleus', 'ap_cytoplasm', 'ap_nucleus_quality_gated', 'ap_cytoplasm_quality_gated',
        'nucleus_tp', 'nucleus_fp', 'nucleus_fn', 'cytoplasm_tp', 'cytoplasm_fp', 'cytoplasm_fn')}
    return out


def main(names):
    from cytonucs_runtime import run_logged
    import train_cytonucs
    (ROOT / '.cache').mkdir(exist_ok=True)
    for name in names:
        train, validation, changes = VARIANTS[name]
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            root = Path(tmp)
            make_data(root, train, validation)
            start = time.time()
            run_logged(str(config(root, name, changes)), train_cytonucs.main)
            history = json.loads((root / 'models' / name / 'history.json').read_text())
            result = dict(summary(history), seconds=round(time.time() - start, 1), training_volumes=train,
                          validation=validation, validation_volumes=VALIDATION, epochs=EPOCHS, steps_per_epoch=STEPS,
                          changes=changes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        # several variants may run at once: merge into the file as it is now
        report = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else dict(results={})
        report.update(scope='Synthetic CPU runs of the real entry point; evidence about fitting capacity, not user-data metrics.',
                      preset=PRESET, shape=SHAPE)
        report['results'][name if SCALE == 1 else f'{name}_scale{SCALE}'] = dict(result, object_scale=SCALE)
        OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
        r = result
        print('RESULT', name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('curves', 'changes')})


if __name__ == '__main__':
    main(sys.argv[1:] or list(VARIANTS))
