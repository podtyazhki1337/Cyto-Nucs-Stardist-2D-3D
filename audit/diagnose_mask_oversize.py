"""Controlled synthetic test: which refinement losses make prompted masks larger than GT.

The project's own loss functions (CytoNucsLoss._refinement_ce_loss / _refinement_dice_loss /
_cell_z_dice) train the same tiny prompted 3D network on identical synthetic data, varying
one factor at a time. Boundaries are ambiguous by construction (blurred, noisy image), which
is where a loss-induced bias can move the decision surface. Not a measurement on user data.

Usage: python audit/diagnose_mask_oversize.py -> audit/mask_oversize_diagnosis.json
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

SHAPE = (16, 64, 64)
STEPS = 500
THRESHOLDS = (0.45, 0.5)


def volume(rng):
    """Anisotropic ellipsoid 'nuclei' (Z radius ~3, XY ~8-11), some touching."""
    zz, yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    labels = np.zeros(SHAPE, np.int32)
    for lab in range(1, 7):
        c = (rng.uniform(4, 12), rng.uniform(10, 54), rng.uniform(10, 54))
        r = (rng.uniform(2.5, 3.5), rng.uniform(8, 11), rng.uniform(8, 11))
        inside = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2 <= 1
        labels[inside & (labels == 0)] = lab
    image = gaussian_filter((labels > 0).astype(np.float32), (0.8, 1.5, 1.5))
    image += rng.normal(0, 0.15, SHAPE).astype(np.float32)
    return image, labels


def sample(rng):
    image, labels = volume(rng)
    ids = [i for i in np.unique(labels) if i]
    target_id = int(rng.choice(ids))
    target = labels == target_id
    pts = np.argwhere(target)
    center = pts.mean(0) + rng.normal(0, 0.1, 3) * (np.ptp(pts, 0) + 1)
    sigma = np.maximum((np.ptp(pts, 0) + 1) / 4., 1.)
    grids = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in SHAPE], indexing='ij')
    prompt = np.exp(-0.5 * sum(((g - c) / s) ** 2 for g, c, s in zip(grids, center, sigma))).astype(np.float32)
    return np.stack([image, prompt], -1), target.astype(np.float32), (labels > 0).astype(np.float32), center, sigma


def batch(rng, n):
    items = [sample(rng) for _ in range(n)]
    return [np.stack([it[k] for it in items]) for k in range(3)], [(it[3], it[4]) for it in items]


def network():
    x = inp = tf.keras.Input(SHAPE + (2,))
    for rate in (1, 2, 4, 1):
        x = tf.keras.layers.Conv3D(16, 3, padding='same', activation='relu', dilation_rate=(1, rate, rate))(x)
    return tf.keras.Model(inp, tf.keras.layers.Conv3D(3, 1)(x))


def train(variant, loss_fn, seed):
    tf.keras.utils.set_random_seed(seed)
    model = network(); optimizer = tf.keras.optimizers.Adam(2e-3)
    rng = np.random.default_rng(seed)
    cfg = loss_fn.config

    @tf.function
    def step(x, target, any_instance):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            probability = tf.nn.softmax(logits)[..., 1:2]
            t = target[..., None]; a = any_instance[..., None]
            loss = 0.
            if variant['ce']:
                loss += 0.5 * loss_fn._refinement_ce_loss(logits, t, a)
            if variant['dice']:
                loss += 0.5 * loss_fn._refinement_dice_loss(probability, t)
            if variant['z_dice']:
                loss += 0.1 * loss_fn._cell_z_dice(probability, t)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    cfg.mask_ce_class_balance = variant['class_balance']
    for _ in range(STEPS):
        (x, target, any_instance), _ = batch(rng, 4)
        step(tf.constant(x), tf.constant(target), tf.constant(any_instance))
    return model


def evaluate(model, seed):
    from cytonucs_postprocess import prompt_component
    rng = np.random.default_rng(seed)
    (x, target, _), prompts = batch(rng, 24)
    probability = tf.nn.softmax(model(x, training=False))[..., 1].numpy()
    out = {}
    for threshold in THRESHOLDS:
        rows = []
        for p, t, (center, sigma) in zip(probability, target > .5, prompts):
            m = prompt_component(p > threshold, center, sigma)
            inter = (m & t).sum()
            rows.append((inter / max(t.sum(), 1), inter / max(m.sum(), 1), m.sum() / max(t.sum(), 1),
                         inter / max((m | t).sum(), 1)))
        r = np.array(rows)
        out[str(threshold)] = dict(coverage=float(np.median(r[:, 0])), purity=float(np.median(r[:, 1])),
                                   size_ratio=float(np.median(r[:, 2])), iou=float(np.median(r[:, 3])),
                                   oversized_fraction=float((r[:, 0] > r[:, 1]).mean()))
    # Size along Z vs XY (does excess come from extra slices?)
    z_extra, xy_extra = [], []
    for p, t in zip(probability, target > .5):
        m = p > 0.45
        z_extra.append(int(np.any(m, (1, 2)).sum()) - int(np.any(t, (1, 2)).sum()))
        xy_extra.append(float(m.max(0).sum() / max(t.max(0).sum(), 1)))
    out['z_slices_extra_median'] = float(np.median(z_extra))
    out['xy_projection_area_ratio_median'] = float(np.median(xy_extra))
    # Size and IoU of the prompt component across thresholds: can a calibrated
    # threshold recover what a different training loss gives?
    ratios, ious = [], []
    for threshold in np.linspace(0.05, 0.95, 19):
        comp = [prompt_component(p > threshold, c, s) for p, (c, s) in zip(probability, prompts)]
        ratios.append(float(np.median([m.sum() / max(t.sum(), 1) for m, t in zip(comp, target > .5)])))
        ious.append(float(np.median([(m & t).sum() / max((m | t).sum(), 1) for m, t in zip(comp, target > .5)])))
    grid = np.round(np.linspace(0.05, 0.95, 19), 2).tolist()
    out['size_ratio_by_threshold'] = dict(zip(grid, ratios))
    out['iou_by_threshold'] = dict(zip(grid, ious))
    return out


def main():
    from test_fixes import config
    from cytonucs_losses import CytoNucsLoss
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = config(train_patch_size=SHAPE, nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1))
    loss_fn = CytoNucsLoss(cfg)
    variants = {
        'current_nucleus: balanced CE + Dice + Z-Dice': dict(ce=True, class_balance=True, dice=True, z_dice=True),
        'balanced CE + Dice': dict(ce=True, class_balance=True, dice=True, z_dice=False),
        'unbalanced CE + Dice': dict(ce=True, class_balance=False, dice=True, z_dice=False),
        'balanced CE only': dict(ce=True, class_balance=True, dice=False, z_dice=False),
        'unbalanced CE only': dict(ce=True, class_balance=False, dice=False, z_dice=False),
        'Dice only': dict(ce=False, class_balance=False, dice=True, z_dice=False),
    }
    selected = sys.argv[1:]  # optional substrings choosing variants
    if selected:
        variants = {k: v for k, v in variants.items() if any(s in k for s in selected)}
    results = {}
    for name, variant in variants.items():
        runs = []
        for seed in (1, 2):
            start = time.time()
            model = train(variant, loss_fn, seed)
            runs.append(dict(seed=seed, seconds=round(time.time() - start, 1), **evaluate(model, 100 + seed)))
        results[name] = dict(variant=variant, runs=runs)
        print(name, [(r['0.45']['coverage'], r['0.45']['purity'], r['0.45']['size_ratio']) for r in runs], flush=True)
    report = dict(scope='Synthetic controlled experiment with the project loss functions; tiny full-resolution '
                        'prompted network, not the CytoNucs architecture or user data.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  shape=SHAPE, steps=STEPS, batch=4, results=results,
                  versions=dict(tensorflow=tf.__version__, numpy=np.__version__))
    target = 'mask_oversize_diagnosis_thresholds.json' if selected else 'mask_oversize_diagnosis.json'
    (ROOT / 'audit' / target).write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
