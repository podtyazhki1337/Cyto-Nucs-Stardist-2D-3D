"""Controlled synthetic comparison of fixes for objects cut by the XY faces of a patch.

Real-data motivation (fixes_v3.2/epoch_0730_refinement.jsonl): 80-88% of the prompted bad masks of
medium objects and all GT smaller than the area gate are objects cut by the XY patch faces; their
masks cover about half of the GT (nuclei spill into background, cells merge with a neighbour).
Candidates, one factor at a time, on the v3.2 mask setting (full-resolution output, voxel CE + Lovász):
  - zero padding at the image border (as now) vs mirrored context (input reflected by PAD voxels at the
    XY faces, predictions cropped back; Ronneberger et al. 2015 overlap-tile);
  - training prompts at the centroid of the visible part (as now) vs at the maximum of the object EDT
    (where the detector puts the centre of a cut object: the EDT does not treat the image face as border).
Each model is evaluated with centroid prompts and with detector-like (EDT maximum) prompts.
Tiny networks on synthetic data: a ranking of fixes, not a measurement of CytoNucs on user data.

Usage: python audit/compare_edge_fragment_fixes.py [variant substrings] -> audit/edge_fragment_fixes*.json
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests')); sys.path.insert(0, str(ROOT / 'audit'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt
from compare_small_object_fixes import SHAPE, volume, network, TAUS

STEPS = 800
SEEDS = (1, 2, 3)
EVAL_VOLUMES = 40
PAD = 8
ANISOTROPY = (3., 1., 1.)


def gaussian(center, sigma):
    axes = np.ix_(*[np.arange(s, dtype=np.float32) for s in SHAPE])
    return np.exp(-0.5 * sum(((g - c) / s) ** 2 for g, c, s in zip(axes, center, sigma))).astype(np.float32)


def prompt_center(target, mode, rng, jitter=0.1):
    pts = np.argwhere(target)
    extent = np.ptp(pts, 0) + 1
    if mode == 'centroid':
        center = pts.mean(0) + rng.normal(0, jitter, 3) * extent
    else:  # detector-like: EDT maximum; array faces are not background (as StarDist edt_prob)
        edt = distance_transform_edt(target, sampling=ANISOTROPY)
        center = np.array(np.unravel_index(np.argmax(edt), target.shape), np.float64)
    return center, np.maximum(extent / 4., 1.)


def touches_xy(target):
    return bool(target[:, 0].any() or target[:, -1].any() or target[..., 0].any() or target[..., -1].any())


def sample(rng, prompt_mode):
    while True:
        image, labels = volume(rng)
        ids = [i for i in np.unique(labels) if i]
        if ids:
            break
    target = labels == int(rng.choice(ids))
    mode = prompt_mode if prompt_mode != 'mixed' else ('edt' if rng.random() < .5 else 'centroid')
    center, sigma = prompt_center(target, mode, rng)
    return np.stack([image, gaussian(center, sigma)], -1), target.astype(np.float32), (labels > 0).astype(np.float32)


def wrapped(mirror):
    net = network(True)
    if not mirror:
        return net
    inp = tf.keras.Input(SHAPE + (2,))
    padded = tf.keras.layers.Lambda(
        lambda t: tf.pad(t, [[0, 0], [0, 0], [PAD, PAD], [PAD, PAD], [0, 0]], mode='REFLECT'))(inp)
    inner = network_for((SHAPE[0], SHAPE[1] + 2 * PAD, SHAPE[2] + 2 * PAD))
    out = tf.keras.layers.Lambda(lambda t: t[:, :, PAD:-PAD, PAD:-PAD])(inner(padded))
    return tf.keras.Model(inp, out)


def network_for(shape):
    """compare_small_object_fixes.network(True) for another spatial shape."""
    import compare_small_object_fixes as base
    saved = base.SHAPE
    base.SHAPE = shape
    try:
        return base.network(True)
    finally:
        base.SHAPE = saved


def train(variant, loss_fn, seed):
    tf.keras.utils.set_random_seed(seed)
    model = wrapped(variant['mirror']); optimizer = tf.keras.optimizers.Adam(2e-3)
    rng = np.random.default_rng(seed)

    @tf.function
    def step(x, target, any_instance):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            probability = tf.nn.softmax(logits)[..., 1:2]
            t = target[..., None]; a = any_instance[..., None]
            loss = 0.5 * loss_fn._refinement_ce_loss(logits, t, a) + 0.5 * loss_fn._refinement_overlap_loss(probability, t)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    for _ in range(STEPS):
        items = [sample(rng, variant['train_prompt']) for _ in range(4)]
        step(*[tf.constant(np.stack([it[k] for it in items])) for k in range(3)])
    return model


def evaluate(model, seed):
    """Every object of EVAL_VOLUMES held-out volumes, prompted once per prompt mode."""
    from cytonucs_postprocess import prompt_component
    rng = np.random.default_rng(seed)
    out = {}
    records = {m: [] for m in ('centroid', 'edt')}
    for _ in range(EVAL_VOLUMES):
        image, labels = volume(rng)
        for lab in [i for i in np.unique(labels) if i]:
            target = labels == lab
            for mode in records:
                center, sigma = prompt_center(target, mode, rng)
                x = np.stack([image, gaussian(center, sigma)], -1)[None]
                p = tf.nn.softmax(model(x, training=False))[0, ..., 1].numpy()
                m = prompt_component(p > .45, center, sigma)
                records[mode].append((target.sum(), touches_xy(target), (m & target).sum() / max((m | target).sum(), 1)))
    for mode, rows in records.items():
        r = np.array(rows, dtype=float)
        groups = {'edge<300': (r[:, 1] == 1) & (r[:, 0] < 300), 'edge300+': (r[:, 1] == 1) & (r[:, 0] >= 300),
                  'interior': r[:, 1] == 0}
        out[mode] = dict(prompted_ap=float(np.mean([(r[:, 2] >= t).mean() for t in TAUS])),
                         **{k: dict(n=int(m.sum()), iou_median=float(np.median(r[m, 2])), tp50=float((r[m, 2] >= .5).mean()))
                            for k, m in groups.items() if m.any()})
    return out


def main():
    from test_fixes import config
    from cytonucs_losses import CytoNucsLoss
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = config(train_patch_size=SHAPE, nucleus_grid=(1, 1, 1), cytoplasm_grid=(1, 1, 1),
                     mask_overlap_loss='lovasz', mask_ce_class_balance=False)
    loss_fn = CytoNucsLoss(cfg)
    variants = {
        'zero pad, centroid prompts (v3.2)': dict(mirror=False, train_prompt='centroid'),
        'mirrored context, centroid prompts': dict(mirror=True, train_prompt='centroid'),
        'zero pad, mixed centroid/EDT-max prompts': dict(mirror=False, train_prompt='mixed'),
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
            runs.append(dict(seed=seed, seconds=round(time.time() - start, 1), **evaluate(model, 100 + seed)))
            r = runs[-1]
            print(name, seed, {m: (round(r[m]['prompted_ap'], 3), {k: round(v['iou_median'], 3) for k, v in r[m].items()
                                                                   if isinstance(v, dict)}) for m in ('centroid', 'edt')},
                  flush=True)
        results[name] = dict(variant=variant, runs=runs)
    report = dict(scope='Synthetic controlled experiment; project loss/postprocess functions; tiny networks.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  shape=SHAPE, pad=PAD, steps=STEPS, seeds=SEEDS, eval_volumes=EVAL_VOLUMES, results=results,
                  versions=dict(tensorflow=tf.__version__, numpy=np.__version__))
    name = 'edge_fragment_fixes' + ('_' + '_'.join(s.replace(' ', '').replace(',', '') for s in selected) if selected else '')
    (ROOT / 'audit' / f'{name}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
