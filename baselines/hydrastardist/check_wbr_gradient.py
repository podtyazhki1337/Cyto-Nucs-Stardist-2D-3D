"""Gradient check of the within-boundary regularisation (WBR) of HydraStarDist (De, Urbanski, Yakimovich 2025, Eq. 4-5):
the layer released with the authors' code (napari-hydra c68ae35: blocks_hydra.ActivityRegularizationLayer with
base_hydra.masked_loss_wbr, wired in StarDist2D._build as published) and the differentiable form that the HSD+WBR
baseline trains with (base_hydra.wbr_loss_paper). CPU only, about half a minute.

1. Released layer, wired as published: the probability maps of the two branches concatenated along the image height,
   the input image without its first column passed as the mask and as y_true, the maps of batch element 0 multiplied
   with those of the other elements, the sigmoid outputs binarised by tf.where(p > 0, 1, p). Checked: the loss value
   does not change when the predictions change, and its gradient with respect to the logits of both maps is zero.
2. Differentiable WBR, per image Lambda = 1 / (1 + eps - min(sum((1 - p_out) * p_in) / max(sum(1 - y_out), 1), 0.99)):
   a. its value equals Eq. 5 computed in NumPy (below the cap);
   b. the TensorFlow gradient equals central finite differences (float64) at every pixel of both maps;
   c. it equals the closed form dL/dp_in = L^2 (1 - p_out) / B and dL/dp_out = -L^2 p_in / B, B = ground-truth
      background;
   d. signs: the loss falls when the inner probability falls outside the outer prediction or the outer probability
      rises under the inner prediction (dL/dp_in >= 0, dL/dp_out <= 0);
   e. images are independent: the loss of one image has zero gradient with respect to another image's maps;
   f. above the cap (ratio >= 0.99, e.g. a crop with almost no ground-truth background) the loss is 1 / (0.01 + eps)
      and its gradient is zero.
3. The HSD+WBR model of this baseline (train_wbr_loss=True; output 'wbr' = [inner, outer] probabilities):
   - the gradient of the WBR loss reaches both probability heads (prob1, prob2) and the shared U-Net, and not the ray
     heads (dist1, dist2);
   - with the released layer on the same model's outputs instead, the gradient is zero on every weight.

Usage: python baselines/hydrastardist/check_wbr_gradient.py [--output FILE.json] [--seed 0]
Exit code 1 if a check fails.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np                                                                       # noqa: E402
import tensorflow as tf                                                                  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hydrastardist.models.model2d_hydra import Config2D, StarDist2D                     # noqa: E402  (before blocks_hydra: circular import)
from hydrastardist.blocks_hydra import ActivityRegularizationLayer                       # noqa: E402
from hydrastardist.models.base_hydra import masked_loss_wbr, wbr_loss_paper              # noqa: E402

EPS, CAP = 1e-7, 0.99
CHECKS = []


def check(name, passed, **numbers):
    CHECKS.append(dict(name=name, passed=bool(passed), **{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                                         for k, v in numbers.items()}))
    print(f"{'PASS' if passed else 'FAIL'}  {name}  " + ', '.join(f'{k} {v:.3g}' if isinstance(v, float) else f'{k} {v}'
                                                           for k, v in CHECKS[-1].items() if k not in ('name', 'passed')))


def released(rng):
    """Part 1: the released layer as wired in the published StarDist2D._build."""
    b, h, w, c = 3, 16, 16, 3
    image = tf.constant(rng.uniform(0, 1, (b, h, w, c)), tf.float32)
    mask = image[:, :, 1:]                                         # input_img[:,:,1:] of the published _build
    logits = [tf.Variable(rng.normal(0, 2, (b, h, w, 1)), dtype=tf.float32) for _ in range(2)]

    def loss_of(l1, l2):
        concat = tf.concat([tf.sigmoid(l1), tf.sigmoid(l2)], axis=1)       # Concatenate(axis=1)([prob1, prob2])
        return tf.reduce_mean(masked_loss_wbr(mask)(mask, concat))         # ActivityRegularizationLayer.call
    with tf.GradientTape() as tape:
        loss = loss_of(*logits)
    grads = tape.gradient(loss, logits)
    other = loss_of(*[tf.constant(rng.normal(0, 2, (b, h, w, 1)), tf.float32) for _ in range(2)])
    norm = max(float(tf.reduce_max(tf.abs(g))) if g is not None else 0.0 for g in grads)
    check('released layer: loss independent of the predictions', float(loss) == float(other),
          loss=float(loss), loss_other_predictions=float(other))
    check('released layer: gradient w.r.t. the logits of both maps is zero', norm == 0.0, max_abs_gradient=norm)
    layer = ActivityRegularizationLayer()                          # the layer object itself gives the same loss
    concat = tf.concat([tf.sigmoid(logits[0]), tf.sigmoid(logits[1])], axis=1)
    layer(concat, mask)
    via_layer = float(tf.reduce_mean(layer.losses[-1])) if layer.losses else None
    check('released layer: ActivityRegularizationLayer adds the same loss', via_layer is not None
          and np.isclose(via_layer, float(loss), rtol=1e-6), loss_via_layer=via_layer)


def paper_value(y_out, p_in, p_out):
    """Eq. 5 of the paper per image, NumPy, with the background floor of 1 and the cap of wbr_loss_paper."""
    ratio = ((1 - p_out) * p_in).sum(axis=(1, 2)) / np.maximum((1 - y_out).sum(axis=(1, 2)), 1)
    return 1 / (1 + EPS - np.minimum(ratio, CAP)), ratio


def differentiable(rng):
    """Part 2: wbr_loss_paper in float64 against NumPy, finite differences and its closed form."""
    b, h, w = 2, 12, 12
    y_out = (rng.uniform(0, 1, (b, h, w)) < 0.4).astype(np.float64)
    p = rng.uniform(0.02, 0.98, (b, h, w, 2))                     # [..., 0] inner, [..., 1] outer probability
    loss_fn = wbr_loss_paper(EPS, CAP)
    y = tf.constant(y_out[..., None])

    def per_image(values):
        return loss_fn(y, tf.constant(values) if not isinstance(values, tf.Variable) else values)
    expected, ratio = paper_value(y_out, p[..., 0], p[..., 1])
    got = per_image(p).numpy()
    check('differentiable WBR: value equals Eq. 5 (NumPy) below the cap', np.allclose(got, expected, rtol=1e-12),
          max_rel_error=float(np.max(np.abs(got - expected) / expected)), ratio_min=float(ratio.min()),
          ratio_max=float(ratio.max()))

    var = tf.Variable(p)
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(per_image(var))
    grad = tape.gradient(total, var).numpy()
    step = 1e-6
    numeric = np.zeros_like(p)
    for index in np.ndindex(p.shape):
        up, down = p.copy(), p.copy()
        up[index] += step
        down[index] -= step
        numeric[index] = (per_image(up).numpy().sum() - per_image(down).numpy().sum()) / (2 * step)
    error = np.abs(grad - numeric)
    check('differentiable WBR: gradient equals central finite differences (float64, every pixel)',
          np.max(error / np.maximum(np.abs(numeric), 1e-8)) < 1e-6, entries=int(p.size),
          max_abs_error=float(error.max()), max_rel_error=float(np.max(error / np.maximum(np.abs(numeric), 1e-8))),
          max_abs_gradient=float(np.abs(grad).max()))

    background = np.maximum((1 - y_out).sum(axis=(1, 2)), 1)[:, None, None]
    lam = got[:, None, None]
    closed_in = lam ** 2 * (1 - p[..., 1]) / background
    closed_out = -lam ** 2 * p[..., 0] / background
    closed = np.stack([closed_in, closed_out], axis=-1)
    check('differentiable WBR: gradient equals the closed form L^2 (1 - p_out) / B, -L^2 p_in / B',
          np.allclose(grad, closed, rtol=1e-10, atol=1e-14), max_abs_error=float(np.abs(grad - closed).max()))
    check('differentiable WBR: signs (dL/dp_in > 0, dL/dp_out < 0)',
          bool((grad[..., 0] > 0).all() and (grad[..., 1] < 0).all()),
          min_dL_dp_in=float(grad[..., 0].min()), max_dL_dp_out=float(grad[..., 1].max()))

    with tf.GradientTape() as tape:
        first = per_image(var)[0]
    cross = tape.gradient(first, var).numpy()
    check('differentiable WBR: images are independent (loss of image 0, gradient on image 1 is zero)',
          np.abs(cross[1]).max() == 0.0 and np.abs(cross[0]).max() > 0, max_abs_cross_gradient=float(np.abs(cross[1]).max()))

    y_cap = np.ones((b, h, w))                                     # image 0: no ground-truth background at all
    y_cap[1, :2, :2] = 0                                           # image 1: 4 background pixels
    p_cap = np.stack([np.full((b, h, w), 0.9), np.full((b, h, w), 0.1)], axis=-1)
    var_cap = tf.Variable(p_cap)
    with tf.GradientTape() as tape:
        capped = loss_fn(tf.constant(y_cap[..., None]), var_cap)
        total = tf.reduce_sum(capped)
    grad_cap = tape.gradient(total, var_cap).numpy()
    _, ratio_cap = paper_value(y_cap, p_cap[..., 0], p_cap[..., 1])
    check('differentiable WBR: above the cap the loss is 1 / (0.01 + eps) and the gradient is zero',
          np.allclose(capped.numpy(), 1 / (1 + EPS - CAP)) and np.abs(grad_cap).max() == 0.0,
          ratio_uncapped_min=float(ratio_cap.min()), loss=float(capped.numpy().max()),
          max_abs_gradient=float(np.abs(grad_cap).max()))


def in_model(rng):
    """Part 3: the HSD+WBR model of this baseline, small, random weights."""
    tf.keras.utils.set_random_seed(0)
    conf = Config2D(axes='YXC', n_rays=8, n_channel_in=3, grid=(2, 2), use_gpu=False, train_wbr_loss=True,
                    unet_n_depth=2, unet_n_filter_base=8, net_conv_after_unet=8, train_patch_size=(64, 64))
    model = StarDist2D(conf, name=None, basedir=None).keras_model
    x = tf.constant(rng.uniform(0, 1, (2, 64, 64, 3)), tf.float32)
    y_out = tf.constant((rng.uniform(0, 1, (2, 32, 32, 1)) < 0.4).astype(np.float32))
    weights = model.trainable_variables
    owner = {id(v): layer.name for layer in model.layers for v in layer.trainable_weights}   # variable -> its layer

    def heads(grads):
        out = {}
        for v, g in zip(weights, grads):
            layer = owner[id(v)]
            norm = 0.0 if g is None else float(tf.reduce_max(tf.abs(g)))
            out[layer] = max(out.get(layer, 0.0), norm)
        return out
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        loss = tf.reduce_mean(wbr_loss_paper(EPS, CAP)(y_out, out['wbr']))
    ours = heads(tape.gradient(loss, weights))
    shared = {k: v for k, v in ours.items() if k not in ('prob1', 'prob2', 'dist1', 'dist2')}
    check("HSD+WBR model: WBR gradient reaches both probability heads",
          ours.get('prob1', 0) > 0 and ours.get('prob2', 0) > 0, prob1=ours.get('prob1', 0.0), prob2=ours.get('prob2', 0.0))
    check('HSD+WBR model: no WBR gradient on the ray heads', ours.get('dist1', 0) == 0 and ours.get('dist2', 0) == 0,
          dist1=ours.get('dist1', 0.0), dist2=ours.get('dist2', 0.0))
    check('HSD+WBR model: WBR gradient reaches the shared U-Net', sum(v > 0 for v in shared.values()) > 0,
          layers_with_gradient=int(sum(v > 0 for v in shared.values())), layers=len(shared))
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        concat = tf.concat([out['prob1'], out['prob2']], axis=1)          # wired as the published _build
        mask = x[:, :, 1:]
        released_loss = tf.reduce_mean(masked_loss_wbr(mask)(mask, concat))
    theirs = heads(tape.gradient(released_loss, weights))
    check('HSD model with the released layer: gradient is zero on every weight', max(theirs.values()) == 0.0,
          max_abs_gradient=max(theirs.values()), weights=len(theirs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', help='JSON with every check and its numbers')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    tf.keras.backend.set_floatx('float32')
    rng = np.random.default_rng(args.seed)
    released(rng)
    differentiable(rng)
    in_model(rng)
    failed = [c['name'] for c in CHECKS if not c['passed']]
    print(f'{len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(checks=CHECKS, seed=args.seed, eps=EPS, cap=CAP,
                                                     tensorflow=tf.__version__), indent=1))
        print('->', args.output)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
