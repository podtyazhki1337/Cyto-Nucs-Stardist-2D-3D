"""Full-resolution prompted mask head (model.refinement_full_resolution, v3.2).

The refinement predicts its 3-class mask on the detector grid; inference upsamples it
linearly (upsample_grid). Objects only a few voxels wide (crop-cut slivers, small nuclei)
cannot be represented at that resolution, and the outward bias of the class-balanced CE
hits them hardest (audit/small_object_fixes*.json). The fine head predicts a residual on
the linearly upsampled coarse logits from the first full-resolution backbone features (no
gradient into the backbone), the image, the refinement features and the prompt. It runs in a
window around the prompt (training: one prompt per crop, for memory) or around the coarse mask
(inference); the coarse mask still decides everything outside the window.
"""
import math

import numpy as np
import tensorflow as tf


def head_grid(config, head):
    return tuple(int(g) for g in getattr(config, head + '_grid'))


def training_window(config, head):
    """Window per axis: 3 typical object radii (median longest ray / ray anisotropy), a
    multiple of the head grid, at most the training patch. Derived from data geometry."""
    grid = head_grid(config, head)
    scale = float(getattr(config, 'median_max_dist_' + head, None) or (15.0 if head == 'nucleus' else 60.0))
    anisotropy = getattr(config, 'rays_anisotropy_' + head, None) or config.anisotropy
    window = []
    for a, (size, g) in enumerate(zip(config.train_patch_size, grid)):
        radius = scale / max(float(anisotropy[a]), 1e-6)
        window.append(int(min(size // g * g, g * math.ceil(3.0 * radius / g))))
    return tuple(max(g, w) for w, g in zip(window, grid))


def upsample_linear(t, grid):
    """Linear upsampling equal to upsample_grid on a window whose last node is included:
    [N, *(n), C] -> [N, *((n - 1) * g if g > 1 else n), C]; node k sits at voxel k * g."""
    for axis, g in enumerate(grid, 1):
        if g == 1:
            continue
        n = tf.shape(t)[axis]
        a = tf.gather(t, tf.range(n - 1), axis=axis)
        b = tf.gather(t, tf.range(1, n), axis=axis)
        phases = [a * (1.0 - p / g) + b * (p / g) for p in range(g)]
        stacked = tf.stack(phases, axis=axis + 1)
        shape = tf.shape(a)
        t = tf.reshape(stacked, tf.concat([shape[:axis], [shape[axis] * g], shape[axis + 1:]], 0))
    return t


def pad_last_node(t, grid):
    """Repeat the last grid node on axes with g > 1 (upsample_grid clamps there: mode='nearest')."""
    pads = [[0, 0]] + [[0, 1 if g > 1 else 0] for g in grid] + [[0, 0]]
    return tf.pad(t, pads, mode='SYMMETRIC')


def prompt_window_begin(prompt, window, grid):
    """Window start per item: centred on the prompt maximum, inside the input, grid aligned."""
    nd = len(grid)
    shape = tf.shape(prompt)
    flat = tf.reshape(prompt[..., 0], [shape[0], -1])
    index = tf.argmax(flat, axis=1, output_type=tf.int32)
    dims = [shape[a + 1] for a in range(nd)]
    begin = []
    for a in range(nd):
        stride = 1
        for b in range(a + 1, nd):
            stride = stride * dims[b]
        centre = (index // stride) % dims[a]
        size = tf.minimum(window[a], dims[a])
        start = tf.clip_by_value(centre - size // 2, 0, dims[a] - size)
        begin.append(start // grid[a] * grid[a])
    return tf.stack(begin, axis=1)


def crop(t, begin, size):
    """Per-item crop of [N, *spatial, C] at begin [N, nd] with a common size [nd]."""
    size = tf.concat([tf.cast(size, tf.int32), [-1]], 0)

    def one(args):
        item, start = args
        return tf.slice(item, tf.concat([start, [0]], 0), size)

    return tf.map_fn(one, (t, begin), fn_output_signature=tf.TensorSpec([None] * (len(t.shape) - 1), t.dtype))


def window_inputs(coarse_logits, hidden, full, begin, window, grid):
    """Crop the fine-head inputs: grid tensors with one extra (clamped) node, the list of
    full-resolution tensors at the window. Returns the list fed to the fine head."""
    nd = len(grid)
    dims = tf.shape(full[0])[1:1 + nd]
    size = tf.minimum(tf.constant(window, tf.int32), dims)
    grid_t = tf.constant(grid, tf.int32)
    extra = tf.constant([1 if g > 1 else 0 for g in grid], tf.int32)
    grid_begin = begin // grid_t
    grid_size = size // grid_t + extra
    coarse_w = crop(pad_last_node(coarse_logits, grid), grid_begin, grid_size)
    hidden_w = crop(pad_last_node(hidden, grid), grid_begin, grid_size)
    return [coarse_w, hidden_w] + [crop(t, begin, size) for t in full]


def inference_window(lo, shape, run_shape, net_shape, grid, margin):
    """Full-resolution window [start, stop) around a coarse candidate window, grid aligned.
    `stop` may exceed run_shape up to net_shape (padded features exist there); callers crop."""
    start, stop = [], []
    for a, g in enumerate(grid):
        s = max(0, int(lo[a]) - margin[a]) // g * g
        e = min(int(run_shape[a]), int(lo[a]) + int(shape[a]) + margin[a])
        e = min(int(net_shape[a]), -(-e // g) * g)
        start.append(s); stop.append(max(e, s + g))
    return np.array(start), np.array(stop)


def fine_window_arrays(coarse_logits, hidden, full, start, stop, grid):
    """Numpy/TF windows of the runtime tensors (batch 1) for one inference window."""
    nd = len(grid)
    grid_start = [s // g for s, g in zip(start, grid)]
    grid_stop = [e // g + (1 if g > 1 else 0) for e, g in zip(stop, grid)]
    coarse_p = pad_last_node(coarse_logits, grid)
    hidden_p = pad_last_node(hidden, grid)
    gsl = (slice(None),) + tuple(slice(s, e) for s, e in zip(grid_start, grid_stop))
    fsl = (slice(None),) + tuple(slice(int(s), int(e)) for s, e in zip(start, stop))
    return [coarse_p[gsl], hidden_p[gsl]] + [t[fsl] for t in full]


def component_touches_open_face(mask, start, stop, run_shape):
    """True when the mask reaches a window face that is not the ROI face: the window may cut it."""
    for a in range(mask.ndim):
        if start[a] > 0 and np.take(mask, 0, axis=a).any():
            return True
        if stop[a] < run_shape[a] and np.take(mask, -1, axis=a).any():
            return True
    return False


class PromptSlot(tf.keras.layers.Layer):
    """One prompt slot per training step for the fine head (memory: one window per crop).

    Input: groups of K per-prompt tensors, concatenated (group g = inputs[g*K:(g+1)*K]).
    Training picks a random slot, so the head also sees the detector-derived prompts that
    inference uses (at the border for objects cut by the patch face); otherwise slot 0.
    Returns the picked tensor of every group and the slot index [1] (int32)."""

    def __init__(self, n_prompts, **kwargs):
        super().__init__(**kwargs)
        self.n_prompts = int(n_prompts)

    def call(self, inputs, training=None):
        k = self.n_prompts
        slot = (tf.random.uniform([], 0, k, dtype=tf.int32) if training and k > 1
                else tf.constant(0, tf.int32))
        picked = [tf.gather(tf.stack(inputs[g * k:(g + 1) * k]), slot) for g in range(len(inputs) // k)]
        return picked + [tf.reshape(slot, [1])]

    def compute_output_spec(self, inputs, training=None):
        import keras
        k = self.n_prompts
        return [keras.KerasTensor(inputs[g * k].shape, dtype=inputs[g * k].dtype) for g in range(len(inputs) // k)] + \
            [keras.KerasTensor((1,), dtype='int32')]


class PromptWindowBegin(tf.keras.layers.Layer):
    """Window start [N, nd] (int32) for the training window around each prompt."""

    def __init__(self, window, grid, **kwargs):
        super().__init__(**kwargs)
        self.window, self.grid = tuple(window), tuple(grid)

    def call(self, prompt):
        return prompt_window_begin(prompt, self.window, self.grid)

    def compute_output_spec(self, prompt):
        import keras
        return keras.KerasTensor((prompt.shape[0], len(self.grid)), dtype='int32')


class FineWindow(tf.keras.layers.Layer):
    """Crops [coarse logits, refinement features, *full-resolution tensors] at the window start
    passed as the last input."""

    def __init__(self, window, grid, **kwargs):
        super().__init__(**kwargs)
        self.window, self.grid = tuple(window), tuple(grid)

    def call(self, inputs):
        coarse, hidden, *full, begin = inputs
        return window_inputs(coarse, hidden, full, tf.cast(begin, tf.int32), self.window, self.grid)

    def compute_output_spec(self, inputs):
        import keras
        nd = len(self.grid)
        return [keras.KerasTensor((inputs[0].shape[0],) + (None,) * nd + (t.shape[-1],), dtype='float32')
                for t in inputs[:-1]]
