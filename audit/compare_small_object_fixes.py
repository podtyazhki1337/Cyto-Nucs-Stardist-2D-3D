"""Controlled synthetic comparison of fixes for small / edge-cut prompted masks.

Real-data motivation (fixes_v3/epoch_0565_refinement.jsonl): masks cover the GT (0.84-0.95)
at every size, but a roughly constant excess makes small objects fail (nuclei 300-1000 voxels:
precision 0.52, IoU 0.42). Candidates tested here, one factor at a time:
  - mask output on the detector grid (1,2,2), upsampled at inference (as now) vs full resolution;
  - class-balanced 3-class CE (as now) vs voxel-mean CE; both with the Lovász overlap term (v3);
  - fixed threshold 0.45 vs one calibrated threshold per variant (as mask_thresh='auto').
The project's loss functions, grid subsampling (CytoNucsLoss._subsample), upsample_grid and
prompt_component are used. Tiny networks on synthetic data: evidence for a mechanism and a
ranking of fixes, not a measurement of CytoNucs on user data.

Usage: python audit/compare_small_object_fixes.py -> audit/small_object_fixes.json
"""
import contextlib
import hashlib
import io
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
GRID = (1, 2, 2)
STEPS = 800
SEEDS = (1, 2, 3)
EVAL_PROMPTS = 96
TAUS = np.round(np.arange(0.5, 0.951, 0.05), 2)
SIZE_BINS = ((0, 300), (300, 1000), (1000, 10 ** 9))


def volume(rng):
    """Anisotropic nuclei (Z radius 2.2-3, XY 8-11), densely packed; about a third centred
    outside the XY range so only a sliver remains (like objects cut by a patch face)."""
    zz, yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    labels = np.zeros(SHAPE, np.int32)
    for lab in range(1, 10):
        r = (rng.uniform(2.2, 3.0), rng.uniform(8, 11), rng.uniform(8, 11))
        c = [rng.uniform(3, 13), rng.uniform(6, 58), rng.uniform(6, 58)]
        if rng.random() < 0.35:                       # sliver at an XY face
            axis = int(rng.integers(1, 3)); low = rng.random() < .5
            depth = rng.uniform(1.5, 5.5)             # visible width in voxels
            c[axis] = depth - r[axis] if low else SHAPE[axis] - 1 - depth + r[axis]
        inside = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2 <= 1
        labels[inside & (labels == 0)] = lab
    brightness = np.concatenate([[0.], rng.uniform(0.6, 1.0, labels.max())]).astype(np.float32)
    image = gaussian_filter(brightness[labels], (0.8, 1.5, 1.5))
    image += rng.normal(0, 0.12, SHAPE).astype(np.float32)
    return image.astype(np.float32), labels


def prompt_for(target, rng, jitter=0.1):
    pts = np.argwhere(target)
    extent = np.ptp(pts, 0) + 1
    center = pts.mean(0) + rng.normal(0, jitter, 3) * extent
    sigma = np.maximum(extent / 4., 1.)
    axes = np.ix_(*[np.arange(s, dtype=np.float32) for s in SHAPE])
    prompt = np.exp(-0.5 * sum(((g - c) / s) ** 2 for g, c, s in zip(axes, center, sigma))).astype(np.float32)
    return prompt, center, sigma


def sample(rng, min_voxels=1):
    while True:
        image, labels = volume(rng)
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        ids = ids[counts >= min_voxels]
        if len(ids):
            break
    target = labels == int(rng.choice(ids))
    prompt, center, sigma = prompt_for(target, rng)
    return (np.stack([image, prompt], -1), target.astype(np.float32), (labels > 0).astype(np.float32),
            center, sigma)


def batch(rng, n):
    items = [sample(rng) for _ in range(n)]
    return [np.stack([it[k] for it in items]) for k in range(3)], [(it[3], it[4]) for it in items]


def network(full_resolution):
    """Same trunk; the grid variant stops on the (1,2,2) grid, the full-resolution variant adds
    one upsampling level fused with the first full-resolution features (U-Net final level)."""
    inp = tf.keras.Input(SHAPE + (2,))
    fine = tf.keras.layers.Conv3D(16, (1, 3, 3), padding='same', activation='relu')(inp)
    x = tf.keras.layers.AveragePooling3D(GRID)(fine)
    for rate in (1, 2, 4, 1):
        x = tf.keras.layers.Conv3D(24, 3, padding='same', activation='relu', dilation_rate=(1, rate, rate))(x)
    if full_resolution:
        x = tf.keras.layers.UpSampling3D(GRID)(x)
        x = tf.keras.layers.Concatenate()([x, fine])
        x = tf.keras.layers.Conv3D(16, 3, padding='same', activation='relu')(x)
    return tf.keras.Model(inp, tf.keras.layers.Conv3D(3, 1)(x))


def train(variant, loss_fn, seed):
    tf.keras.utils.set_random_seed(seed)
    model = network(variant['full_resolution']); optimizer = tf.keras.optimizers.Adam(2e-3)
    rng = np.random.default_rng(seed)
    loss_fn.config.mask_ce_class_balance = variant['class_balance']
    grid = (1, 1, 1) if variant['full_resolution'] else GRID

    @tf.function
    def step(x, target, any_instance):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            probability = tf.nn.softmax(logits)[..., 1:2]
            t = loss_fn._subsample(target[..., None], grid); a = loss_fn._subsample(any_instance[..., None], grid)
            loss = 0.5 * loss_fn._refinement_ce_loss(logits, t, a) + 0.5 * loss_fn._refinement_overlap_loss(probability, t)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for _ in range(STEPS):
        (x, target, any_instance), _ = batch(rng, 4)
        step(tf.constant(x), tf.constant(target), tf.constant(any_instance))
    return model


def evaluate(model, variant, seed):
    from cytonucs_postprocess import prompt_component, upsample_grid
    rng = np.random.default_rng(seed)
    (x, target, _), prompts = batch(rng, EVAL_PROMPTS)
    probability = tf.nn.softmax(model(x, training=False))[..., 1].numpy()
    if not variant['full_resolution']:
        probability = np.stack([upsample_grid(p, SHAPE, GRID) for p in probability])
    targets = target > .5

    def ious(threshold):
        out = []
        for p, t, (center, sigma) in zip(probability, targets, prompts):
            m = prompt_component(p > threshold, center, sigma)
            out.append(((m & t).sum() / max((m | t).sum(), 1), m.sum() / max(t.sum(), 1)))
        return np.array(out)

    sizes = targets.reshape(len(targets), -1).sum(1)
    thresholds = np.round(np.arange(0.25, 0.951, 0.05), 2)
    table = {float(th): ious(th) for th in thresholds}
    prompted_ap = {th: float(np.mean([(r[:, 0] >= tau).mean() for tau in TAUS])) for th, r in table.items()}
    calibrated = max(prompted_ap, key=prompted_ap.get)

    def summary(threshold):
        r = table[threshold]
        bins = {}
        for lo, hi in SIZE_BINS:
            m = (sizes >= lo) & (sizes < hi)
            if m.any():
                bins[f'{lo}-{hi if hi < 10 ** 9 else "inf"}'] = dict(
                    n=int(m.sum()), iou_median=float(np.median(r[m, 0])), tp50=float((r[m, 0] >= .5).mean()),
                    size_ratio_median=float(np.median(r[m, 1])))
        return dict(threshold=threshold, prompted_ap=prompted_ap[threshold], iou_median=float(np.median(r[:, 0])),
                    tp50=float((r[:, 0] >= .5).mean()), by_size=bins)

    return dict(fixed_045=summary(0.45), calibrated=summary(calibrated))


def main():
    from test_fixes import config
    from cytonucs_losses import CytoNucsLoss
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = config(train_patch_size=SHAPE, nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1),
                     mask_overlap_loss='lovasz')
    loss_fn = CytoNucsLoss(cfg)
    variants = {
        'grid, balanced CE + Lovász (current v3)': dict(full_resolution=False, class_balance=True),
        'grid, voxel CE + Lovász': dict(full_resolution=False, class_balance=False),
        'full-res, balanced CE + Lovász': dict(full_resolution=True, class_balance=True),
        'full-res, voxel CE + Lovász': dict(full_resolution=True, class_balance=False),
    }
    selected = sys.argv[1:]
    if selected:
        variants = {k: v for k, v in variants.items() if any(s in k for s in selected)}
    results = {}
    for name, variant in variants.items():
        runs = []
        for seed in SEEDS:
            start = time.time()
            model = train(variant, loss_fn, seed)
            runs.append(dict(seed=seed, seconds=round(time.time() - start, 1), **evaluate(model, variant, 100 + seed)))
            r = runs[-1]
            print(name, seed, 'AP@0.45 %.3f calibrated %.3f (thr %.2f)' % (
                r['fixed_045']['prompted_ap'], r['calibrated']['prompted_ap'], r['calibrated']['threshold']),
                {k: round(v['iou_median'], 3) for k, v in r['fixed_045']['by_size'].items()}, flush=True)
        results[name] = dict(variant=variant, runs=runs)
    report = dict(scope='Synthetic controlled experiment; project loss/postprocess functions; tiny networks. '
                        'prompted_ap = mean over IoU 0.5-0.95 of the fraction of prompts reaching that IoU '
                        '(one prompt per GT, no detection/FP).',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  shape=SHAPE, grid=GRID, steps=STEPS, seeds=SEEDS, eval_prompts=EVAL_PROMPTS, results=results,
                  versions=dict(tensorflow=tf.__version__, numpy=np.__version__))
    name = 'small_object_fixes' + ('_' + '_'.join(s.replace(' ', '') for s in selected) if selected else '')
    (ROOT / 'audit' / f'{name}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
