"""
CytoNucs StarDist Training Loop

Implements custom training with all loss components.
"""
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from cytonucs_inference import CytoNucsPredictor
from stardist.matching import matching_dataset


def _calculate_ap(precision, recall):
    """
    Calculate average precision from precision and recall values.
    Expects precision and recall to be arrays of values in [0, 1].
    """
    # cleaning and conversion
    precision = np.array([p if isinstance(p, (int, float)) and not np.isnan(p) else 0.0 for p in precision])
    recall = np.array([r if isinstance(r, (int, float)) and not np.isnan(r) else 0.0 for r in recall])

    # normalised if the values exceed 1
    if len(precision) > 0 and np.max(precision) > 1.0:
        print(f"⚠️ WARNING: precision contains values > 1.0 (max={np.max(precision)}), normalizing...")
        precision = precision / np.max(precision)

    if len(recall) > 0 and np.max(recall) > 1.0:
        print(f"⚠️ WARNING: recall contains values > 1.0 (max={np.max(recall)}), normalizing...")
        recall = recall / np.max(recall)

    # empty arrays
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    # sorted by recall
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]

    # end points
    precision = np.concatenate(([1.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))

    # interpolated precision (maximum to the right)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    # area under the curve
    x = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[x] - recall[x - 1]) * precision[x])

    return ap
def batch_tensors(batch):
    """A generator batch (x, y) as tensors: floating arrays as float32, integer arrays as int32, other values as they are."""
    x, y = batch
    converted = {}
    for key, value in y.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in 'fiu':
            # numpy casts without the GIL (the ray targets arrive as float16), TensorFlow then only copies.
            value = tf.convert_to_tensor(np.asarray(value, np.float32 if value.dtype.kind == 'f' else np.int32))
        elif isinstance(value, np.ndarray):
            value = tf.convert_to_tensor(value)
        converted[key] = value
    return tf.convert_to_tensor(np.asarray(x, np.float32)), converted


class _Prefetcher:
    """Bounded ordered prefetch: reproducible samples regardless of worker timing.

    convert: applied to every batch in its worker (batch_tensors: the training loop then gets tensors and does not
    spend its own time between two steps on the conversion)."""
    def __init__(self, gen, depth=3, workers=3, count=None, convert=None):
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        self._gen, self._next = gen, 0
        self._item = gen.__getitem__ if convert is None else (lambda index: convert(gen[index]))
        self._count = len(gen) if count is None else int(count)
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers))
        self._futures = deque()
        for _ in range(min(max(1, depth), self._count)):
            self._submit()
    def _submit(self):
        self._futures.append(self._pool.submit(self._item, self._next))
        self._next += 1
    def next(self):
        result = self._futures.popleft().result()
        if self._next < self._count:
            self._submit()
        return result
    def stop(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

def align_fullval_history(history, lengths_before):
    """Keep every fullval_* list index-aligned with fullval_epoch.

    A key first logged now gets leading None for earlier validations; a key not
    logged now gets a trailing None. Values are never moved or invented.
    """
    n = len(history.get('fullval_epoch', []))
    for key, values in history.items():
        if not key.startswith('fullval_') or key == 'fullval_epoch' or len(values) >= n:
            continue
        if len(values) > lengths_before.get(key, 0):
            values[:0] = [None] * (n - len(values))
        else:
            values.extend([None] * (n - len(values)))


def sync_object_min_area(loss_fn, gates):
    """Object-level probability losses skip objects below the calibrated size gates.

    Only with loss_params.object_losses_min_area_from_gate; missing gates keep 0 (no filter).
    """
    if not getattr(loss_fn.config, 'object_losses_min_area_from_gate', False) or not gates:
        return None
    values = [float(gates.get(h + '_min_area', 0.0) or 0.0) for h in ('nucleus', 'cytoplasm')]
    loss_fn.object_min_voxels.assign(values)
    print(f"  Object losses: skip objects below {values[0]:.0f} / {values[1]:.0f} voxels (nucleus / cytoplasm)")
    return values


def _prune_epoch_checkpoints(folder, keep):
    """training.keep_epoch_checkpoints: only the `keep` latest weights_epoch_NNN checkpoints (all their files)
    stay; best and best-val-loss checkpoints have other names and are untouched."""
    import re
    epochs = {}
    for f in Path(folder).glob('weights_epoch_*'):
        m = re.fullmatch(r'weights_epoch_(\d+)\..*', f.name)
        if m:
            epochs.setdefault(int(m.group(1)), []).append(f)
    for epoch in sorted(epochs)[:-keep]:
        for f in epochs[epoch]:
            f.unlink()


def _write_epoch_calibration(model, state_epoch, path):
    """Attach quality gates calibrated by this epoch's full validation, never older ones.

    Mask thresholds are calibrated once after training (calibrate_mask_thresholds).
    """
    gates = getattr(model, 'pending_quality_gates', None)
    if gates is not None and gates['epoch'] == int(state_epoch.numpy()):
        from cytonucs_runtime import write_quality_gates
        write_quality_gates(path, gates)


# Per-object probability diagnostics: a batch without sampled objects reports 0 and
# must not dilute the epoch value, so these are averaged with the object count as weight.
OBJECT_WEIGHTED_PREFIX = 'nucleus_prob_gt_max_'
OBJECT_COUNT_KEY = 'nucleus_prob_sampled_object_count'


OPTIMIZER_PATHS_ATTR = 'cytonucs_optimizer_paths'


def verify_state_order(prefix, ordered_variables, saved_paths=None, weights_loaded=False):
    """The .state checkpoint restores model variables and Adam slots by optimizer index; a
    different variable order would silently put moments (and values) on the wrong variables.

    Checkpoints that record their optimizer variable paths are checked by name (variables added
    later may follow). Older ones are checked by shape and, when the weights of this checkpoint
    were just loaded by name, by the exact saved values."""
    path = lambda v: getattr(v, 'path', None) or v.name
    if saved_paths is not None:
        current = [path(v) for v in ordered_variables]
        if current[:len(saved_paths)] != list(saved_paths):
            first = next(i for i, (a, b) in enumerate(zip(current, saved_paths)) if a != b) \
                if len(current) >= len(saved_paths) else len(current)
            raise ValueError(f'Optimizer state order differs from the checkpoint at index {first}; '
                             'refusing to restore Adam state')
        return
    reader = tf.train.load_checkpoint(prefix)
    for index, variable in enumerate(ordered_variables):
        key = f'optimizer/_trainable_variables/{index}/.ATTRIBUTES/VARIABLE_VALUE'
        if not reader.has_tensor(key):
            continue                               # variable added after this checkpoint
        saved = reader.get_tensor(key)
        if saved.shape != tuple(variable.shape) or (weights_loaded and not np.array_equal(saved, variable.numpy())):
            raise ValueError(f'Optimizer state order differs from the checkpoint at index {index} '
                             f'({path(variable)}); refusing to restore Adam state')


def accumulate_components(sums, components):
    counts = {head: float(components.get(OBJECT_COUNT_KEY.replace('nucleus', head), 0.0))
              for head in ('nucleus', 'cytoplasm')}
    for key, value in components.items():
        weight = next((counts[head] for head in counts
                       if key.startswith(OBJECT_WEIGHTED_PREFIX.replace('nucleus', head))), 1.0)
        total, weights = sums.get(key, (0.0, 0.0))
        sums[key] = (total + float(value) * weight, weights + weight)


def component_means(sums):
    return {key: (total / weights if weights else None) for key, (total, weights) in sums.items()}


def calibrate_mask_thresholds(predictor, samples):
    """Per-head thresholds for mask_thresh_*='auto' from one pass over validation volumes.

    Run once after training (or with mode 'calibrate'), not at every full validation.
    Candidates of each volume are cached; the other thresholds of mask_thresh_range (below
    and above mask_thresh) re-assemble one head without new network passes. The threshold
    maximizing the checkpoint metric is kept (default mask_thresh if nothing improves).
    samples: dicts with normalized 'image', 'nucleus_mask', 'cytoplasm_mask', 'assignments'.
    Returns (thresholds, statistics); empty when no head uses 'auto'.
    """
    from types import SimpleNamespace
    from stardist.matching import matching
    from cytonucs_inference import compute_jtpr, rethreshold_prediction
    from cytonucs_runtime import select_mask_thresholds
    config = predictor.config
    heads = [h for h in ('nucleus', 'cytoplasm') if getattr(config, 'mask_thresh_' + h) == 'auto']
    if not heads:
        return {}, {}
    thresholds = sorted(float(t) for t in set(config.mask_thresh_range) if t != config.mask_thresh)
    taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    def empty():
        return dict(tp=np.zeros(len(taus), np.int64), fp=np.zeros(len(taus), np.int64),
                    fn=np.zeros(len(taus), np.int64), joint_tp=0)

    def add(slot, gt, pred):
        result = matching(gt, pred, thresh=taus, criterion='iou')
        for k, r in enumerate(result if isinstance(result, (list, tuple)) else [result]):
            slot['tp'][k] += int(r.tp); slot['fp'][k] += int(r.fp); slot['fn'][k] += int(r.fn)

    base = {h: dict(empty(), threshold=config.mask_thresh) for h in heads}
    evaluated = {h: {t: empty() for t in thresholds} for h in heads}
    total_pairs = 0
    saved = getattr(predictor, 'calibrating', False)
    predictor.calibrating = True
    runner = SimpleNamespace(model=predictor.model)
    try:
        for index, sample in enumerate(samples):
            sink = {} if thresholds else None
            predictor._refinement_candidate_sink = sink
            try:
                nuclei, cells, _ = DiceCallback._predict_full_volume(
                    runner, predictor, np.asarray(sample['image'], np.float32))
            finally:
                predictor._refinement_candidate_sink = None
            gt = {'nucleus': np.asarray(sample['nucleus_mask']), 'cytoplasm': np.asarray(sample['cytoplasm_mask'])}
            _, joint = compute_jtpr(nuclei, cells, gt['nucleus'], gt['cytoplasm'], iou_thresh=0.5,
                                    assignments_pred=getattr(predictor, '_prompt_assignments', None),
                                    assignments_gt=sample.get('assignments'))
            total_pairs += int(joint['total_pairs'])
            for head in heads:
                add(base[head], gt[head], nuclei if head == 'nucleus' else cells)
                base[head]['joint_tp'] += int(joint['joint_tp'])
                for threshold in thresholds:
                    nuc_t, cell_t, pairs_t = rethreshold_prediction(config, sink, head, threshold, nuclei, cells)
                    slot = evaluated[head][threshold]
                    add(slot, gt[head], nuc_t if head == 'nucleus' else cell_t)
                    _, joint_t = compute_jtpr(nuc_t, cell_t, gt['nucleus'], gt['cytoplasm'], iou_thresh=0.5,
                                              assignments_pred=pairs_t, assignments_gt=sample.get('assignments'))
                    slot['joint_tp'] += int(joint_t['joint_tp'])
            print(f"  [mask calibration {index + 1}/{len(samples)}] {sample.get('filename', index)}")
    finally:
        predictor.calibrating = saved
    from cytonucs_runtime import checkpoint_jtpr_weight
    chosen, stats = select_mask_thresholds(base, evaluated, total_pairs, checkpoint_jtpr_weight(config))
    for head in heads:
        print(f"  Mask threshold {head}: {config.mask_thresh:.3f} -> {chosen[head]:.3f}; "
              f"AP {stats[head]['ap_default']:.4f} -> {stats[head]['ap']:.4f}, "
              f"JTPR {stats[head]['jtpr_default']:.4f} -> {stats[head]['jtpr']:.4f}")
    stats = dict(heads=stats, candidates=thresholds, default=config.mask_thresh, volumes=len(samples),
                 total_pairs=total_pairs)
    return chosen, stats


def source_volume(filename):
    """Source volume of a patch file: its name without the '_zNNN_yNNNN_xNNNN_aNN' patch suffix."""
    import re
    return re.sub(r'_z\d+_y\d+_x\d+_a\d+$', '', Path(str(filename)).stem)


def without_xy_face_objects(gt, pred):
    """Labels restricted to whole objects: GT objects touching an XY face of the tile are removed, and so
    is every predicted object whose best GT (highest IoU) is one of them. As in edge_fragment_ignored_ap,
    such a mask is neither TP nor FP; masks without any GT overlap stay and count as FP."""
    gt, pred = np.asarray(gt).astype(np.int64), np.asarray(pred).astype(np.int64)
    face = np.zeros(gt.shape, bool)
    face[..., 0, :] = face[..., -1, :] = face[..., :, 0] = face[..., :, -1] = True
    cut = np.setdiff1d(np.unique(gt[face]), [0])
    both = (gt > 0) & (pred > 0)
    drop = np.zeros(0, np.int64)
    if both.any():
        width = int(gt.max()) + 1
        keys, intersection = np.unique(pred[both] * width + gt[both], return_counts=True)
        p, g = keys // width, keys % width
        iou = intersection / (np.bincount(pred.ravel())[p] + np.bincount(gt.ravel())[g] - intersection)
        order = np.lexsort((iou, p))                               # the last row of every mask is its best GT
        best = np.r_[p[order][1:] != p[order][:-1], True]
        drop = p[order][best][np.isin(g[order][best], cut)]
    return np.where(np.isin(gt, cut), 0, gt), np.where(np.isin(pred, drop), 0, pred)


def deployed_tile_metrics(predictor, samples, taus):
    """AP and JTPR of what inference outputs (the predictor's gates and mask thresholds) on full tiles.

    'ap_<head>' counts every object; 'ap_<head>_whole' leaves out the objects cut by an XY face of the
    tile (without_xy_face_objects). 'by_volume' repeats the numbers per source volume of the tiles.
    samples: dicts with normalized 'image', the two masks, 'assignments' and 'filename'.
    """
    from types import SimpleNamespace
    from stardist.matching import matching
    from cytonucs_inference import compute_jtpr
    from cytonucs_runtime import accuracy_profile
    runner = SimpleNamespace(model=predictor.model)

    def empty():
        return dict(counts={(head, part): np.zeros((3, len(taus)), np.int64)
                            for head in ('nucleus', 'cytoplasm') for part in ('', '_whole')}, joint=[0, 0], volumes=0)

    def summary(scope):
        result = dict(volumes=scope['volumes'], jtpr=scope['joint'][0] / scope['joint'][1] if scope['joint'][1] else 0.0)
        for (head, part), c in scope['counts'].items():
            accuracy = accuracy_profile(*c)
            result[f'ap_{head}{part}'] = float(accuracy.mean())
            result[f'ap50_{head}{part}'] = float(accuracy[0])
        return result

    total, by_volume = empty(), {}
    for sample in samples:
        scopes = (total, by_volume.setdefault(source_volume(sample.get('filename', 'volume')), empty()))
        nuclei, cells, _ = DiceCallback._predict_full_volume(runner, predictor, np.asarray(sample['image'], np.float32))
        gt = dict(nucleus=np.asarray(sample['nucleus_mask']), cytoplasm=np.asarray(sample['cytoplasm_mask']))
        _, joint = compute_jtpr(nuclei, cells, gt['nucleus'], gt['cytoplasm'], iou_thresh=0.5,
                                assignments_pred=getattr(predictor, '_prompt_assignments', None),
                                assignments_gt=sample.get('assignments'))
        for scope in scopes:
            scope['volumes'] += 1
            scope['joint'][0] += int(joint['joint_tp']); scope['joint'][1] += int(joint['total_pairs'])
        for head, pred in (('nucleus', nuclei), ('cytoplasm', cells)):
            for part, (g, p) in (('', (gt[head], pred)), ('_whole', without_xy_face_objects(gt[head], pred))):
                result = matching(g, p, thresh=taus, criterion='iou')
                for k, r in enumerate(result if isinstance(result, (list, tuple)) else [result]):
                    for scope in scopes:
                        scope['counts'][head, part][:, k] += (int(r.tp), int(r.fp), int(r.fn))
    return dict(summary(total), by_volume={name: summary(scope) for name, scope in sorted(by_volume.items())})


class CytoNucsTrainer:
    """
    Custom trainer for CytoNucs StarDist model.
    """

    def __init__(self, model, config, loss_fn, gradient_clip_norm, optimizer=None):
        """
        Args:
            model: CytoNucsStarDistModel instance
            config: CytoNucsConfig object
            loss_fn: CytoNucsLoss instance
            optimizer: optional optimizer (default: Adam with schedule)
        """
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.prompt_detector = None
        if config.predicted_prompt_fraction or config.detector_prompt_slots:
            self.prompt_detector = tf.keras.Model(model.keras_model.inputs,
                {h+'_'+kind:model.keras_model.output[h+'_'+kind]
                 for h in ('nucleus','cytoplasm') for kind in ('prob','dist')})
        # Detector prompts from the prob / dist of the training pass itself (model.detection_model, then
        # model.prompted_model with the prompts): the separate detector pass computed the same prob / dist once
        # more (no layer of the backbone depends on the training flag). Mask prompts need a complete pass of
        # their own and keep the single call.
        self.two_calls = self.prompt_detector is not None and not config.mask_prompt_fraction
        self.compiled = None
        self.current_step = 0
        self.current_epoch_var = tf.Variable(1, dtype=tf.int32, trainable=False)
        # Setup optimizer
        if optimizer is None:
            lr_decay_steps = getattr(config, 'lr_decay_steps', 700)
            lr_decay_rate = getattr(config, 'lr_decay_rate', 0.5)
            lr_staircase = getattr(config, 'lr_staircase', True)

            schedule = ExponentialDecay(
                initial_learning_rate=config.train_learning_rate,
                decay_steps=lr_decay_steps,
                decay_rate=lr_decay_rate,
                staircase=lr_staircase
            )
            # no clipnorm here: clip_by_global_norm is applied in the training step
            self.optimizer = Adam(learning_rate=schedule)
        else:
            self.optimizer = optimizer


        # Metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }

        # Checkpoint directory
        self.checkpoint_dir = Path(model.basedir) / model.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.loss_metrics_train = {}
        self.loss_metrics_val = {}
        self.state_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.state_best_val = tf.Variable(float('inf'), dtype=tf.float32, trainable=False)
        self.state_best_full = tf.Variable(float('-inf'), dtype=tf.float32, trainable=False)
        self.state_selection_start = tf.Variable(0, dtype=tf.int64, trainable=False)
        # training.ema_decay: exponential moving average of the weights. Validation, full validation,
        # checkpoints and the model after training use the averaged weights (swap_ema); the optimizer
        # keeps stepping the raw ones. Not saved separately: a resume continues from the saved
        # (averaged) weights.
        self.ema_decay = float(getattr(config, 'ema_decay', 0.0) or 0.0)
        if self.ema_decay and int(getattr(config, 'n_pair_channels', 0) or 0):
            raise NotImplementedError('training.ema_decay with three or more levels is not implemented')
        self.ema_variables = ([tf.Variable(v, trainable=False, name='ema_' + v.name.replace(':', '_'))
                               for v in self.model.keras_model.trainable_variables] if self.ema_decay else [])
        self.ema_swapped = False
        # Stable slot order for both fresh training and optimizer restoration.
        self.optimizer.build(self.model.optimizer_variables())
        self.checkpoint = tf.train.Checkpoint(model=self.model.keras_model, optimizer=self.optimizer,
                                             epoch=self.state_epoch, best_val=self.state_best_val,
                                             best_full=self.state_best_full,
                                             selection_start=self.state_selection_start)

    def swap_ema(self):
        """Exchange the raw and the averaged weights (no-op without training.ema_decay)."""
        for average, variable in zip(self.ema_variables, self.model.keras_model.trainable_variables):
            raw = tf.identity(variable)
            variable.assign(average)
            average.assign(raw)
        if self.ema_variables:
            self.ema_swapped = not self.ema_swapped

    def save_checkpoint(self, path):
        self.model.save_weights(path)
        import h5py
        with h5py.File(path, 'a') as weights:
            weights.attrs['cytonucs_epoch'] = int(self.state_epoch.numpy())
            weights.attrs[OPTIMIZER_PATHS_ATTR] = json.dumps(
                [getattr(v, 'path', None) or v.name for v in self.model.optimizer_variables()])
        _write_epoch_calibration(self.model, self.state_epoch, path)
        self.checkpoint.write(str(path) + '.state')

    def restore_training_state(self, weights_path):
        prefix = str(weights_path) + '.state'
        if not Path(prefix + '.index').exists():
            print('WARNING: weights-only resume; Adam state unavailable')
            return False
        self.optimizer.build(self.model.optimizer_variables())
        import h5py
        saved_paths = None
        if Path(weights_path).exists():
            with h5py.File(weights_path, 'r') as weights:
                if OPTIMIZER_PATHS_ATTR in weights.attrs:
                    saved_paths = json.loads(weights.attrs[OPTIMIZER_PATHS_ATTR])
        loaded = str(getattr(self.model, 'loaded_weights_path', '')) == str(weights_path)
        verify_state_order(prefix, self.model.optimizer_variables(), saved_paths, loaded)
        self.checkpoint.restore(prefix).expect_partial()
        if self.config.reset_best_on_resume:
            self.state_best_val.assign(float('inf'))
            self.state_best_full.assign(float('-inf'))
            self.state_selection_start.assign(self.state_epoch)
            print('Reset best selection for changed loss/evaluation; epoch and Adam preserved')
        print(f'Restored Adam and epoch {int(self.state_epoch.numpy())}')
        return True

    def _detection(self, x_batch):
        """model.detection_model in training mode, the first call of the two-call step. training.jit_compile: compiled
        with XLA (the same operations, fused: GroupNorm and the activations are chains of element-wise passes over
        the feature maps otherwise). The prompted networks are not compiled: their windows are cut where the prompts
        lie, and XLA takes such positions, and with them the whole input, as constants of one compilation."""
        net = self.model.detection_model
        if not getattr(self.config, 'jit_compile', False):
            return net(x_batch, training=True)
        if self.compiled is None:
            self.compiled = tf.function(lambda x: net(x, training=True), jit_compile=True)
        return self.compiled(x_batch)

    @tf.function
    def train_step(self, x_batch, y_batch_dict):
        original_prompts = x_batch
        nuclear_cell_fraction = tf.constant(0., tf.float32)
        mask_prompt_share = tf.constant(0., tf.float32)
        if not self.two_calls:
            if self.prompt_detector is not None:
                x_batch, nuclear_cell_fraction = tf.cond(self.current_epoch_var >= self.config.predicted_prompt_start_epoch,
                    lambda: self._predicted_prompts(x_batch, y_batch_dict, return_stats=True),
                    lambda: (x_batch, tf.constant(0., tf.float32)))
            # Quality targets read the prompts' Gaussian geometry: taken before mask prompts are added.
            y_batch_dict = self._mask_quality_inputs(x_batch, y_batch_dict)
            detector_prompts = x_batch
            if self.config.mask_prompt_fraction:
                x_batch, mask_prompt_share = tf.cond(self.current_epoch_var >= self.config.mask_prompt_start_epoch,
                    lambda: self._mask_prompts(x_batch),
                    lambda: (x_batch, tf.constant(0., tf.float32)))
        with tf.GradientTape() as tape:
            if self.two_calls:
                first = self._detection(x_batch)
                y_pred = dict(first['outputs'])
                detector = {key: tf.stop_gradient(y_pred[key]) for key in self.prompt_detector.output}
                x_batch, nuclear_cell_fraction = tf.cond(self.current_epoch_var >= self.config.predicted_prompt_start_epoch,
                    lambda: self._predicted_prompts(original_prompts, y_batch_dict, return_stats=True, pred=detector),
                    lambda: (original_prompts, tf.constant(0., tf.float32)))
                y_batch_dict = self._mask_quality_inputs(x_batch, y_batch_dict)
                detector_prompts = x_batch
                y_pred.update(self.model.prompted_model([x_batch] + list(first['features']), training=True))
            else:
                y_pred = self.model.keras_model(x_batch, training=True)
            loss = self.loss_fn(y_batch_dict, y_pred, current_epoch=self.current_epoch_var)

        for hi, head in enumerate(('nucleus', 'cytoplasm')):
            changed = tf.reduce_any(tf.not_equal(
                detector_prompts[..., self.config.n_channel_in+hi::2],
                original_prompts[..., self.config.n_channel_in+hi::2]),
                axis=list(range(1, self.config.ndim+1)))
            self.loss_fn.last_components[head + '_detector_prompt_changed_fraction'] = tf.reduce_mean(tf.cast(changed, tf.float32))
        self.loss_fn.last_components['cytoplasm_nuclear_detector_prompt_fraction'] = nuclear_cell_fraction
        if self.config.mask_prompt_fraction:
            self.loss_fn.last_components['mask_prompt_fraction'] = mask_prompt_share
        gradients = tape.gradient(loss, self.model.keras_model.trainable_variables)

        # Gradient groups by branch, from the layer names (static at trace time). In Keras 3 var.name is
        # 'kernel' / 'bias' without the layer, so the groups use var.path. The refinement has a group of its own.
        prob_layers = {'nucleus_prob', 'cytoplasm_prob',
                       'nucleus_prob_logit', 'cytoplasm_prob_logit'}
        dist_layers = {'nucleus_dist', 'cytoplasm_dist'}
        refine_subs = ('_refine', 'prompt_gate', 'nucleus_mask', 'cytoplasm_mask',
                       'geo_dist_compress')
        # The cytoplasm decoder is a group of its own (only the encoder is shared), so that its gradient norm is
        # logged apart from the encoder and the nucleus decoder.
        cyto_dec_subs = ('cyto_dec_', 'cyto_features')

        prob_grads, prob_vars = [], []
        dist_grads, dist_vars = [], []
        backbone_grads, backbone_vars = [], []
        refine_grads, refine_vars = [], []
        cdec_grads, cdec_vars = [], []

        for grad, var in zip(gradients, self.model.keras_model.trainable_variables):
            if grad is None:
                continue
            path = getattr(var, 'path', None) or var.name
            layer = path.split('/')[0]
            if any(s in layer for s in refine_subs):
                refine_grads.append(grad);
                refine_vars.append(var)
            elif layer in prob_layers:
                prob_grads.append(grad);
                prob_vars.append(var)
            elif layer in dist_layers:
                dist_grads.append(grad);
                dist_vars.append(var)
            elif any(s in layer for s in cyto_dec_subs):
                cdec_grads.append(grad);
                cdec_vars.append(var)
            else:
                backbone_grads.append(grad);
                backbone_vars.append(var)

        clip_norm = tf.cast(getattr(self.config, 'gradient_clip_norm', 0), tf.float32)

        def _group_norm(grads):
            if len(grads) == 0:
                return tf.constant(0.0, tf.float32)
            return tf.linalg.global_norm(grads)

        # The group norms are for the log only. StarDist (base.py:308) applies Adam without clipping; clipping each
        # group would rescale the gradients by a factor that changes every step and override the loss weights.
        # gradient_clip_norm = 0: no clipping (as StarDist); > 0: one global clip as a safety brake.
        prob_norm = _group_norm(prob_grads)
        dist_norm = _group_norm(dist_grads)
        bb_norm = _group_norm(backbone_grads)
        rf_norm = _group_norm(refine_grads)
        cd_norm = _group_norm(cdec_grads)

        active = [(g, v) for g, v in zip(gradients, self.model.keras_model.trainable_variables)
                  if g is not None]
        all_grads, all_vars = map(list, zip(*active))

        if clip_norm > 0:
            # one global clip for the whole network (as global_clipnorm): the balance between groups is kept
            all_grads, _ = tf.clip_by_global_norm(all_grads, clip_norm)

        self.optimizer.apply_gradients(zip(all_grads, all_vars))
        if self.ema_variables:
            # Bias-corrected warm-up: early steps average over fewer steps than 1 / (1 - decay).
            step = tf.cast(self.optimizer.iterations, tf.float32)
            decay = tf.minimum(tf.constant(self.ema_decay, tf.float32), (1. + step) / (10. + step))
            for average, variable in zip(self.ema_variables, self.model.keras_model.trainable_variables):
                average.assign(decay * average + (1. - decay) * variable)
        self.train_loss_metric.update_state(loss)
        return loss, prob_norm, dist_norm, bb_norm, rf_norm, cd_norm, self.loss_fn.last_components

    def _mask_quality_inputs(self, x, targets):
        if self.config.mask_quality_target != 'component' or not self.config.mask_quality_weight:
            return targets
        result = dict(targets)
        for hi, head in enumerate(('nucleus', 'cytoplasm')):
            result[head + '_quality_prompt'] = tf.concat([
                x[..., self.config.n_channel_in + 2*k + hi]
                for k in range(self.config.prompts_per_crop)], axis=0)
        return result

    def _mask_prompts(self, x):
        """training.mask_prompt_fraction: in that share of the prompt slots (per crop and slot) both prompt
        channels also carry the model's own mask for that prompt, g + 0.5 * m * (1 - g), with g the prompt as given
        and m the mask of a pass without gradient (refinement probability > 0.5 on the head grid, upsampled by
        repetition). The peak stays g's, so the centre the refinement measures its offsets from is unchanged. The
        refinement thus learns the second pass of inference.refinement_iterations, which prompts it with its first
        mask. Returns (prompts, share of slots with a mask prompt)."""
        cfg = self.config
        nd, slots = cfg.ndim, cfg.prompts_per_crop
        pred = self.model.keras_model(x, training=False)
        batch = tf.shape(x)[0]
        spatial = tf.shape(x)[1:nd + 1]
        seed = tf.stack([tf.cast(cfg.seed, tf.int32) + 7919, tf.cast(self.optimizer.iterations, tf.int32)])
        chosen = tf.random.stateless_uniform((slots, batch), seed) < cfg.mask_prompt_fraction
        channels = [x[..., :cfg.n_channel_in]]
        for k in range(slots):
            select = tf.reshape(chosen[k], [-1] + [1] * (nd + 1))
            for hi, head in enumerate(('nucleus', 'cytoplasm')):
                logit = tf.stop_gradient(pred[head + '_mask_logit'][k * batch:(k + 1) * batch])
                mask = tf.cast(tf.nn.softmax(logit, axis=-1)[..., 1:2] > .5, tf.float32)
                for axis, g in enumerate(getattr(cfg, head + '_grid')):
                    if int(g) > 1:
                        mask = tf.repeat(mask, int(g), axis=axis + 1)
                mask = tf.slice(mask, tf.zeros([nd + 2], tf.int32), tf.concat([[-1], spatial, [-1]], 0))
                offset = cfg.n_channel_in + 2 * k + hi
                gaussian = x[..., offset:offset + 1]
                from cytonucs_runtime import mask_prompt
                channels.append(tf.where(select, mask_prompt(gaussian, mask), gaussian))
        return tf.stop_gradient(tf.concat(channels, axis=-1)), tf.reduce_mean(tf.cast(chosen, tf.float32))

    def _predicted_prompts(self, x, targets, return_stats=False, pred=None):
        """Detached detector prompts with existing GT ownership and hard negatives.

        Select a detector peak inside the selected GT target and derive its
        width from predicted rays. Empty targets use a detector peak on GT background.
        No arbitrary positive instance is mislabeled as a negative.
        pred: prob / dist of both heads for x if already computed, else a detector pass computes them.
        """
        if pred is None:
            pred = self.prompt_detector(x, training=False)
        nd, batch = self.config.ndim, tf.shape(x)[0]
        shape = tf.shape(x)[1:nd+1]
        axes = tf.meshgrid(*[tf.cast(tf.range(shape[a]), tf.float32) for a in range(nd)], indexing='ij')
        channels = [x[..., :self.config.n_channel_in]]
        seed = tf.stack([tf.cast(self.config.seed, tf.int32), tf.cast(self.optimizer.iterations, tf.int32)])
        decisions = tf.random.stateless_uniform((self.config.prompts_per_crop, batch, 1, 1), seed)
        def propose(source, head, k, hi):
            grid = getattr(self.config, source + '_grid')
            target = targets[head + '_target'][k*batch:(k+1)*batch]
            sl = (slice(None),) + tuple(slice(None,None,g) for g in grid)
            target_grid = tf.cast(target[sl] > .5, tf.bool)
            # Positive targets may disappear on the detector grid. They must
            # retain their GT prompt, not become background negatives.
            nonempty = tf.reduce_any(target > .5,axis=list(range(1,nd+1)))
            background = targets['nucleus_instances'][sl] == 0
            if 'nucleus_known' in targets:                      # training.unknown_label
                background &= targets['nucleus_known'][sl] > .5
            if head == 'cytoplasm':
                # An unassigned real nucleus is not a reliable cell negative.
                background &= targets['cytoplasm_instances'][sl] == 0
                if 'cytoplasm_known' in targets:
                    background &= targets['cytoplasm_known'][sl] > .5
            valid_points = tf.where(tf.reshape(nonempty,[-1]+[1]*nd), target_grid, background)
            probability = pred[source+'_prob'][...,0]
            scores = tf.where(valid_points, probability, -tf.ones_like(probability))
            valid = tf.reduce_any(valid_points,axis=list(range(1,nd+1)))
            if self.config.predicted_prompt_sample_peaks:
                # Repeated targets should expose different detector hypotheses,
                # not always the same argmax (which trivializes consistency).
                pool = tf.nn.max_pool3d if nd == 3 else tf.nn.max_pool2d
                # Find peaks before GT ownership filtering; otherwise a
                # stronger neighbour outside the target is hidden by GT.
                # Same window/border rule as inference, including null (auto) values.
                from cytonucs_geometry import proposal_peak_settings
                window, border = proposal_peak_settings(
                    self.config, source, [p // g for p, g in zip(self.config.train_patch_size, grid)], peaks=True)
                peak_probability = probability
                if border:
                    coords_grid = tf.meshgrid(*[tf.range(tf.shape(probability)[a+1]) for a in range(nd)], indexing='ij')
                    interior = tf.ones(tf.shape(probability)[1:], tf.bool)
                    for a in range(nd-2, nd):
                        interior &= (coords_grid[a] >= border) & (coords_grid[a] < tf.shape(probability)[a+1]-border)
                    valid_points &= interior[None]
                    peak_probability = tf.where(interior[None], probability, tf.constant(-float('inf'), tf.float32))
                local_max = pool(peak_probability[..., None], ksize=window, strides=1, padding='SAME')[..., 0]
                peaks = valid_points & (probability >= local_max) & (probability > getattr(self.config, 'prob_thresh_'+source))
                random = tf.random.stateless_uniform(tf.shape(scores),
                    tf.random.experimental.stateless_fold_in(seed, 2*k+hi+1))
                has_peaks = tf.reduce_any(peaks, axis=list(range(1, nd+1)), keepdims=True)
                scores = tf.where(peaks, random, -tf.ones_like(random))
                valid = valid & tf.reshape(has_peaks, [-1])
            flat = tf.argmax(tf.reshape(scores, (batch,-1)), axis=1, output_type=tf.int32)
            coords = tf.transpose(tf.unravel_index(flat, tf.shape(scores)[1:]))
            gather = tf.concat([tf.range(batch)[:,None], coords], axis=1)
            distances = tf.maximum(tf.gather_nd(pred[source+'_dist'], gather), 1e-3)
            verts = tf.constant(getattr(self.config,'rays_'+source).vertices, tf.float32)
            points = distances[...,None]*verts[None]
            sigma = tf.maximum((tf.reduce_max(points,axis=1)-tf.reduce_min(points,axis=1))/4.,1.)
            center = tf.cast(coords,tf.float32)*tf.constant(grid,tf.float32)
            d2 = sum(tf.square((ax[None]-tf.reshape(center[:,a],[-1]+[1]*nd)) /
                               tf.reshape(sigma[:,a],[-1]+[1]*nd)) for a,ax in enumerate(axes))
            prompt = tf.exp(-.5*d2)[...,None]
            return prompt, valid

        nuclear_cell_replacements = []
        for k in range(self.config.prompts_per_crop):
            for hi, head in enumerate(('nucleus', 'cytoplasm')):
                prompt, valid = propose(head, head, k, hi)
                use_nucleus = tf.zeros([batch], tf.bool)
                if head == 'cytoplasm' and self.config.cell_predicted_nucleus_fraction:
                    nuclear_prompt, nuclear_valid = propose('nucleus', head, k, hi)
                    source_draw = tf.random.stateless_uniform([batch],
                        tf.random.experimental.stateless_fold_in(seed, 10000+k))
                    use_nucleus = (source_draw < self.config.cell_predicted_nucleus_fraction) & nuclear_valid
                    # Keep the cell-source candidate when there is no owned nuclear
                    # candidate, rather than silently dropping detector exposure.
                    prompt = tf.where(tf.reshape(use_nucleus, [-1]+[1]*(nd+1)), nuclear_prompt, prompt)
                    valid = tf.where(use_nucleus, nuclear_valid, valid)
                fraction = (1.0 if k >= self.config.prompts_per_crop - self.config.detector_prompt_slots
                            else self.config.predicted_prompt_fraction)
                selected = (tf.reshape(decisions[k],[-1]) < fraction) & valid
                if k < 2 and head + '_z_prompt_pair' in targets:
                    selected &= ~tf.cast(targets[head + '_z_prompt_pair'], tf.bool)
                choose = tf.reshape(selected, [-1]+[1]*(nd+1))
                offset = self.config.n_channel_in+2*k+hi
                channels.append(tf.where(choose,prompt,x[...,offset:offset+1]))
                if head == 'cytoplasm':
                    nuclear_cell_replacements.append(tf.cast(selected & use_nucleus, tf.float32))
        result = tf.stop_gradient(tf.concat(channels,axis=-1))
        fraction = tf.reduce_mean(tf.stack(nuclear_cell_replacements))
        return (result, fraction) if return_stats else result

    @tf.function
    def val_step(self, x_batch, y_batch):
        """
        Single validation step.

        Args:
            x_batch: input images (B, H, W, C)
            y_batch: dict with GT data (nucleus_instances, cytoplasm_instances)

        Returns:
            loss: scalar loss value
        """
        # forward pass in inference mode
        y_pred = self.model.keras_model(x_batch, training=False)

        loss = self.loss_fn(self._mask_quality_inputs(x_batch, y_batch), y_pred,
                            current_epoch=self.current_epoch_var)

        # Update validation metric
        self.val_loss_metric.update_state(loss)

        return loss, self.loss_fn.last_components

    def _pair(self, y_batch):
        """Trainer of the batch's pair in a hierarchy of three or more levels (cytonucs_hierarchy:
        one trainer per adjacent pair over the same network and optimizer); otherwise self."""
        index = y_batch.pop('pair_index', None) if isinstance(y_batch, dict) else None
        trainers = getattr(self, 'pair_trainers', None)
        if index is None or not trainers:
            return self
        trainer = trainers[int(index)]
        if trainer is not self:
            trainer.current_epoch_var.assign(self.current_epoch_var)
        return trainer

    # main training loop


    def train(
                self,
                train_generator,
                val_generator=None,
                epochs=None,
                steps_per_epoch=None,
                callbacks=None,
                experiment=None,
                start_epoch=0
        ):
            """
            Train the model.
            """

            epochs = epochs or self.config.train_epochs
            steps_per_epoch = steps_per_epoch or self.config.train_steps_per_epoch

            callbacks = callbacks or []
            from cytonucs_runtime import align_training_history
            self.history = align_training_history(self.history, start_epoch)

            print(f"Training for {epochs} epochs, {steps_per_epoch} steps/epoch")
            print(f"Model: {self.model.name}")
            print(f"Config: {self.config}")

            best_val_loss = float(self.state_best_val.numpy())
            gpu_memory_stats = bool(tf.config.list_logical_devices('GPU'))
            for epoch in range(start_epoch, epochs):
                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"{'=' * 50}")

                self.current_epoch_var.assign(epoch + 1)
                self.state_epoch.assign(epoch + 1)
                epoch_started = time.time()
                if gpu_memory_stats:
                    tf.config.experimental.reset_memory_stats('GPU:0')
                epoch_checkpoint_paths = []
                train_generator.set_epoch(epoch)
                prefetch = _Prefetcher(train_generator,
                    depth=int(getattr(self.config, 'prefetch_depth', 3)),
                    workers=int(getattr(self.config, 'prefetch_workers', 3)), count=steps_per_epoch,
                    convert=batch_tensors)
                component_sums = {}
                # Reset metrics
                self.train_loss_metric.reset_state()
                self.val_loss_metric.reset_state()

                # Training loop
                train_pbar = tqdm(range(steps_per_epoch), desc='Training')
                try:
                    for step in train_pbar:
                        # Get batch: tensors already (converted in the prefetch workers), instance masks included
                        x_batch, y_batch_tf = prefetch.next()
                        stepper = CytoNucsTrainer._pair(self, y_batch_tf)


                        # Train step
                        loss, p_n, d_n, b_n, r_n, c_n, components = stepper.train_step(x_batch, y_batch_tf)
                        self.current_step += 1
                        accumulate_components(component_sums, components)
                        if self.current_step % max(1, int(getattr(self.config, 'log_every_steps', 20))) == 0:
                            print(f"  [GRAD NORMS] prob: {float(p_n):.3f} dist: {float(d_n):.3f} "
                                  f"backbone: {float(b_n):.3f} cyto_dec: {float(c_n):.3f} "
                                  f"|| refine: {float(r_n):.3f} "
                                  f"| clip: {float(self.config.gradient_clip_norm)}")

                        # Update progress bar
                        train_pbar.set_postfix({'loss': f'{loss.numpy():.4f}'})
           
                finally:
                    prefetch.stop()
                for key, value in component_means(component_sums).items():
                    self.history.setdefault('train_' + key, [None] * epoch).append(value)
                # Validation, checkpoints and callbacks of this epoch see the averaged weights.
                if getattr(self, 'ema_variables', None):
                    self.swap_ema()
                train_loss = self.train_loss_metric.result().numpy()
                self.history['train_loss'].append(float(train_loss))
                # Resources of the training part of the epoch (validation is timed separately below).
                train_seconds = time.time() - epoch_started
                self.history.setdefault('time_train_s', [None] * epoch).append(round(train_seconds, 2))
                self.history.setdefault('steps_per_s', [None] * epoch).append(round(steps_per_epoch / max(train_seconds, 1e-6), 3))
                if gpu_memory_stats:
                    self.history.setdefault('gpu_peak_mem_gb', [None] * epoch).append(
                        round(tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2 ** 30, 3))
                self.history['lr'].append(float(self.optimizer.learning_rate.numpy()))

                print(f"Train Loss: {train_loss:.4f}")

                # Log loss components
                if experiment:
                    experiment.log_metric("train_loss", train_loss, epoch=epoch + 1)
                    experiment.log_metric("learning_rate", self.optimizer.learning_rate.numpy(), epoch=epoch + 1)

                # Validation loop: patch losses are a training diagnostic (checkpoints are chosen by full
                # validation), computed every val_loss_every_n_epochs epochs and at the last one.
                val_every = int(getattr(self.config, 'val_loss_every_n_epochs', 1) or 1)
                val_now = (epoch + 1) % val_every == 0 or epoch + 1 == epochs
                if val_generator is not None and not val_now:
                    for key, values in self.history.items():          # keep one entry per epoch
                        if key.startswith('val_') and isinstance(values, list) and len(values) == epoch:
                            values.append(None)
                if val_generator is not None and val_now:
                    val_sums = {}
                    val_pbar = tqdm(range(len(val_generator)), desc='Validation')
                    for step in val_pbar:
                        x_batch, y_batch_tf = batch_tensors(val_generator[step])
                        stepper = CytoNucsTrainer._pair(self, y_batch_tf)

                        _, components = stepper.val_step(x_batch, y_batch_tf)
                        accumulate_components(val_sums, components)

                    for key, value in component_means(val_sums).items():
                        self.history.setdefault('val_' + key, [None] * epoch).append(value)
                    val_loss = self.val_loss_metric.result().numpy()
                    self.history['val_loss'].append(float(val_loss))
                    print(f"Val Loss: {val_loss:.4f}")

                    if experiment:
                        experiment.log_metric("val_loss", val_loss, epoch=epoch + 1)

                    # The loss changes during the warmup, so val_loss is not comparable across it: the best value is
                    # reset when the warmup is complete.
                    _warm = int(getattr(self.config, 'loss_warmup_epochs',
                                        getattr(self.config, 'wbr_warmup_epochs', 20)))
                    if (epoch + 1) == 2 * _warm:
                        best_val_loss = np.inf
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.state_best_val.assign(val_loss)

                        # val_loss is computed on deterministic train-sized patches: a training diagnostic, not the
                        # quality metric (that is the full validation of DiceCallback).
                        best_path = (
                                self.checkpoint_dir
                                / 'weights_best_val_loss.weights.h5'
                        )

                        self.save_checkpoint(
                            best_path
                        )
                        epoch_checkpoint_paths.append(best_path)

                        print(
                            f"  ✅ Best PATCH val-loss checkpoint saved: "
                            f"{best_path}"
                        )

                # Save checkpoints
                save_every = getattr(self.config, 'save_every_n_epochs', 10)
                if (epoch + 1) % save_every == 0:
                    latest_path = self.checkpoint_dir / f'weights_epoch_{epoch + 1:03d}.weights.h5'
                    self.save_checkpoint(latest_path)
                    epoch_checkpoint_paths.append(latest_path)
                    print(f"  💾 Checkpoint saved: {latest_path}")

                # Call callbacks
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, self.history)

                # Keep the early weights checkpoint available if validation is
                # interrupted, but commit its updated best state after callbacks.
                for path in epoch_checkpoint_paths:
                    _write_epoch_calibration(self.model, self.state_epoch, path)
                    self.checkpoint.write(str(path) + '.state')
                keep = int(getattr(self.config, 'keep_epoch_checkpoints', 0) or 0)
                if keep and (epoch + 1) % save_every == 0:
                    _prune_epoch_checkpoints(self.checkpoint_dir, keep)

                if hasattr(train_generator, 'on_epoch_end'):
                    train_generator.on_epoch_end()
                if val_generator is not None and hasattr(val_generator, 'on_epoch_end'):
                    val_generator.on_epoch_end()
                self.history.setdefault('time_epoch_s', [None] * epoch).append(round(time.time() - epoch_started, 2))
                keras_model = getattr(self.model, 'keras_model', None)
                if keras_model is not None:
                    self.history['trainable_parameters'] = int(sum(int(np.prod(v.shape)) for v in keras_model.trainable_variables))
                # Save history
                with open(self.checkpoint_dir / 'history.json', 'w') as f:
                    json.dump(self.history, f, indent=2)
                if epoch + 1 < epochs and getattr(self, 'ema_variables', None):
                    self.swap_ema()                     # back to the raw weights for training
            print(f"\n{'=' * 50}")
            print("Training complete!")
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"Model saved to: {self.checkpoint_dir}")



    def load_weights(self, weights_path):
        """Load model weights"""
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")


class DiceCallback:
    """
    Callback to compute Dice score on validation set.
    Similar to DiceCheckpoint in original code.
    """

    def __init__(
            self,
            model,
            val_generator,
            eval_every=5,
            save_best=True,
            experiment=None,
            prob_thresh_nucleus=None,
            prob_thresh_cytoplasm=None,
            nms_thresh=None,
            train_generator=None
    ):
        """
        Full-volume quality validation.

        val_loss is computed separately, on deterministic patches.

        This callback:
            full validation volume
            -> production-like tiled inference
            -> sparse StarDist NMS
            -> prompted refinement
            -> AP / AP@50 / semantic Dice / JTPR

        The best checkpoint maximises the checkpoint metric: loss_params.checkpoint_w_jtpr x linked JTPR + the rest x
        the mean AP of the levels, AP = mean over IoU 0.50..0.95 of TP / (TP + FP + FN).
        """

        self.model = model
        self.val_generator = val_generator
        # Source of the training volumes for evaluation.fullval_train_volumes (train-side AP).
        self.train_generator = train_generator

        self.eval_every = int(
            eval_every
        )

        self.save_best = bool(
            save_best
        )

        self.experiment = experiment

        self.best_metric = -np.inf

        self.prob_thresh_nucleus = (
            prob_thresh_nucleus
        )

        self.prob_thresh_cytoplasm = (
            prob_thresh_cytoplasm
        )

        self.nms_thresh = (
            nms_thresh
        )

        self.min_eval_epoch = int(
            getattr(
                model.config,
                'min_eval_epoch',
                1
            )
        )

    @staticmethod
    def _semantic_dice(gt, pred):
        """
        Foreground / background Dice over the whole volume.

        Unlike compute_dice() it penalises both FP and FN (computed on the whole binary mask).
        """
        g = np.asarray(gt) > 0
        p = np.asarray(pred) > 0

        if g.shape != p.shape:
            raise ValueError(
                f"Semantic Dice shape mismatch: "
                f"GT={g.shape}, pred={p.shape}"
            )

        if not g.any() and not p.any():
            return 1.0

        denom = int(g.sum()) + int(p.sum())

        if denom == 0:
            return 0.0

        return float(
            2.0
            * np.logical_and(
                g,
                p
            ).sum()
            / denom
        )

    def _train_volumes(self):
        """Fixed training volumes for the train-side metric: originals only (not the offline-augmented
        '_aNN' copies), taken in turn from every source volume, the tiles with most objects first and
        without spatial overlap inside a source volume while such tiles remain."""
        import re
        count = int(getattr(self.model.config, 'fullval_train_volumes', 0))
        pool = getattr(self.train_generator, 'data_pool', None) if count else None
        if not pool:
            return []
        key = tuple(sorted(s['filename'] for s in pool))
        if getattr(self, '_train_volume_cache', (None, None, None))[:2] == (key, count):
            return self._train_volume_cache[2]
        # With training.partial_annotation only tiles that have every level of the run can be scored,
        # with training.unknown_label only tiles without unknown voxels.
        levels = len(getattr(self.model.config, 'levels', ('nucleus', 'cytoplasm')))
        pool = sorted((s for s in pool if len(s.get('annotated', ('nucleus', 'cytoplasm'))) >= levels and
                       s.get('nucleus_unknown') is None and s.get('cytoplasm_unknown') is None), key=lambda s: s['filename'])
        if not pool:
            return []
        originals = [s for s in pool if Path(s['filename']).stem.endswith('_a00')] or pool

        def objects(sample):
            return sum(int(np.count_nonzero(np.unique(sample[m]))) for m in ('nucleus_mask', 'cytoplasm_mask') if m in sample)

        def box(sample):
            found = re.search(r'_z(\d+)_y(\d+)_x(\d+)_a\d+$', Path(sample['filename']).stem)
            shape = np.shape(sample['nucleus_mask']) if 'nucleus_mask' in sample else ()
            if found is None or len(shape) != 3:
                return None
            start = np.array([int(v) for v in found.groups()])
            return start, start + np.array(shape)

        groups = {}
        for sample in originals:
            groups.setdefault(source_volume(sample['filename']), []).append(sample)
        queues = [sorted(group, key=lambda s: (-objects(s), s['filename'])) for group in groups.values()]
        queues.sort(key=lambda q: (-sum(objects(s) for s in q), q[0]['filename']))     # richest source volume first
        taken = [[] for _ in queues]
        picks = []
        while len(picks) < count and any(queues):
            for queue, boxes in zip(queues, taken):
                if not queue or len(picks) >= count:
                    continue
                apart = lambda s: box(s) is None or all(np.any(box(s)[0] >= hi) or np.any(box(s)[1] <= lo) for lo, hi in boxes)
                index = next((i for i, s in enumerate(queue) if apart(s)), 0)
                sample = queue.pop(index)
                if box(sample) is not None:
                    boxes.append(box(sample))
                picks.append(sample)
        self._train_volume_cache = (key, count, picks)
        return picks

    def _predict_full_volume(
            self,
            predictor,
            img
    ):
        """
        The tiled / untiled inference of run_inference_only().

        img is already normalised over the whole volume (val_generator.repool()).
        """
        config = self.model.config

        ndim = int(
            config.ndim
        )

        patch = tuple(
            int(p)
            for p in config.train_patch_size
        )

        spatial = tuple(
            int(s)
            for s in img.shape[:ndim]
        )

        needs_tiling = any(
            s > p
            for s, p in zip(
                spatial,
                patch
            )
        )

        if needs_tiling:

            n_tiles = tuple(
                max(
                    1,
                    int(
                        np.ceil(
                            s / p
                        )
                    )
                )
                for s, p in zip(
                    spatial,
                    patch
                )
            )

            # Z is handled by the fixed-Z windows of _predict_with_tiling().
            if ndim == 3:
                n_tiles = (
                    1,
                    n_tiles[1],
                    n_tiles[2]
                )

            return predictor._predict_with_tiling(
                img,
                n_tiles=n_tiles,
                normalize_img=False
            )

        return predictor.predict_instances(
            img,
            normalize_img=False
        )

    def on_epoch_end(
            self,
            epoch,
            history
    ):
        epoch_num = int(epoch) + 1

        # ============================================================
        # SCHEDULE
        # ============================================================

        if (
                epoch_num
                % self.eval_every
                != 0
        ):
            return

        if epoch_num < self.min_eval_epoch:
            print(
                f"\n  Skipping full-volume validation: "
                f"epoch {epoch_num} < "
                f"{self.min_eval_epoch}"
            )

            return

        if self.val_generator is None:
            print(
                "\n  ⚠️ Full-volume validation skipped: "
                "val_generator=None"
            )

            return

        # ============================================================
        # VALIDATION DATA
        # ============================================================
        #
        # Not self.val_generator[i] (__getitem__ returns a train-sized patch): data_pool holds the whole volumes,
        # normalised once over each volume.
        # ============================================================

        data_pool = getattr(
            self.val_generator,
            'data_pool',
            None
        )

        image_files = getattr(
            self.val_generator,
            'image_files',
            None
        )

        if (
                data_pool is None
                or image_files is None
        ):
            raise RuntimeError(
                "DiceCallback: "
                "val_generator has no "
                "data_pool/image_files"
            )

        if len(data_pool) != len(image_files):
            raise RuntimeError(
                "Full-volume validation needs "
                "the whole validation set in data_pool. "
                f"data_pool={len(data_pool)}, "
                f"image_files={len(image_files)}. "
                "Do not set a smaller pool_size "
                "for val_generator."
            )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"FULL-VOLUME VALIDATION | "
            f"epoch {epoch_num}"
        )

        print(
            "=" * 70
        )

        # ============================================================
        # THRESHOLDS
        # ============================================================

        config = self.model.config

        prob_thresh_nucleus = (
            float(
                self.prob_thresh_nucleus
            )
            if self.prob_thresh_nucleus
               is not None
            else float(
                getattr(
                    config,
                    'prob_thresh_nucleus',
                    0.5
                )
            )
        )

        prob_thresh_cytoplasm = (
            float(
                self.prob_thresh_cytoplasm
            )
            if self.prob_thresh_cytoplasm
               is not None
            else float(
                getattr(
                    config,
                    'prob_thresh_cytoplasm',
                    0.75
                )
            )
        )

        nms_thresh = (
            float(
                self.nms_thresh
            )
            if self.nms_thresh
               is not None
            else float(
                getattr(
                    config,
                    'nms_thresh',
                    0.3
                )
            )
        )

        print(
            f"  thresholds: "
            f"nucleus={prob_thresh_nucleus:.3f}, "
            f"cytoplasm={prob_thresh_cytoplasm:.3f}, "
            f"nms={nms_thresh:.3f}"
        )

        predictor = CytoNucsPredictor(
            model=self.model,
            config=config,
            prob_thresh_nucleus=
            prob_thresh_nucleus,
            prob_thresh_cytoplasm=
            prob_thresh_cytoplasm,
            nms_thresh=
            nms_thresh
        )
        # Calibration must see ungated candidates, whatever the loaded checkpoint says.
        predictor.calibrating = True
        fullval_lengths = {k: len(v) for k, v in history.items() if k.startswith('fullval_')}

        # ============================================================
        # COUNTS ARE ACCUMULATED, NOT MASKS
        # ============================================================
        #
        # The TP / FP / FN of each volume are added up right after it is predicted and its masks are released
        # (the same numbers as matching_dataset's pooled counts, without keeping every mask). JTPR and the semantic
        # Dice are accumulated the same way.
        # ============================================================

        from stardist.matching import matching
        from cytonucs_inference import compute_jtpr
        from types import SimpleNamespace

        taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        cnt = {h: {k: np.zeros(len(taus), np.int64) for k in ('tp', 'fp', 'fn')}
               for h in ('nucleus', 'cytoplasm')}
        # The same counts, written masks and pairs per source volume (reported next to the totals).
        volume_stats = {}

        semantic_nucleus = []
        semantic_cytoplasm = []

        jtpr_values = []
        joint_tp_total = 0
        total_pairs_total = 0
        pair_failures = dict(joint_fail_nucleus=0, joint_fail_cell=0, joint_fail_link=0)
        interior = dict(interior_joint_tp=0, interior_total_pairs=0)
        # Written masks per head: (predicted quality, IoU with best GT, joint pairs lost
        # if removed, written area) for the joint quality/size gate calibration.
        gated_heads = [h for h in ('nucleus', 'cytoplasm')
                       if config.mask_quality_head and
                       getattr(config, 'mask_assembly_' + h).startswith('object_quality')]
        written_quality = {h: [] for h in gated_heads}
        # Same masks and every GT with the patch-face contact of their GT, for the AP that treats
        # patch-edge fragments as ignore regions (reported next to the standard metric).
        edge_rows = {h: dict(masks=[], gt=[]) for h in gated_heads}
        # Same masks with the Z-slice / XY-contour split of their error (matched masks only).
        boundary_rows = {h: [] for h in gated_heads}

        viz_dir = (
                Path(
                    self.model.basedir
                )
                / self.model.name
                / 'visualizations'
        )

        viz_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ============================================================
        # FULL-VOLUME INFERENCE
        # ============================================================

        diagnostic_path = viz_dir / f'epoch_{epoch_num:04d}_refinement.jsonl'
        diagnostic_path.write_text('', encoding='utf-8')
        monitoring_totals = {}
        for i, sample in enumerate(data_pool):

            img = np.asarray(sample['image'], dtype=np.float32)
            nucleus_gt = np.asarray(sample['nucleus_mask'])
            cytoplasm_gt = np.asarray(sample['cytoplasm_mask'])
            filename = sample.get('filename', f'val_{i:03d}')
            per_volume = volume_stats.setdefault(source_volume(filename), dict(
                volumes=0, joint=[0, 0], rows={h: [] for h in cnt},
                cnt={h: {k: np.zeros(len(taus), np.int64) for k in ('tp', 'fp', 'fn')} for h in cnt}))
            per_volume['volumes'] += 1

            from cytonucs_refinement_audit import RefinementStageAudit
            audits = ({'nucleus': RefinementStageAudit(nucleus_gt),
                       'cytoplasm': RefinementStageAudit(cytoplasm_gt)}
                      if config.refinement_stage_audit else None)
            self.model.last_refinement_diagnostics = None
            predictor._refinement_stage_audit = audits
            try:
                nucleus_pred, cytoplasm_pred, details = self._predict_full_volume(predictor, img)
            finally:
                predictor._refinement_stage_audit = None

            # no silent cropping of the validation results
            if nucleus_pred.shape != nucleus_gt.shape:
                raise RuntimeError(
                    f"{filename}: nucleus shape mismatch: "
                    f"pred={nucleus_pred.shape}, GT={nucleus_gt.shape}")
            if cytoplasm_pred.shape != cytoplasm_gt.shape:
                raise RuntimeError(
                    f"{filename}: cytoplasm shape mismatch: "
                    f"pred={cytoplasm_pred.shape}, GT={cytoplasm_gt.shape}")

            sd_n = self._semantic_dice(nucleus_gt, nucleus_pred)
            sd_c = self._semantic_dice(cytoplasm_gt, cytoplasm_pred)
            semantic_nucleus.append(sd_n)
            semantic_cytoplasm.append(sd_c)

            for head, gt_, pr_ in (('nucleus', nucleus_gt, nucleus_pred),
                                   ('cytoplasm', cytoplasm_gt, cytoplasm_pred)):
                res = matching(gt_, pr_, thresh=taus, criterion='iou')
                res = list(res) if isinstance(res, (list, tuple)) else [res]
                for k, r in enumerate(res):
                    cnt[head]['tp'][k] += int(r.tp)
                    cnt[head]['fp'][k] += int(r.fp)
                    cnt[head]['fn'][k] += int(r.fn)
                    for key, value in (('tp', r.tp), ('fp', r.fp), ('fn', r.fn)):
                        per_volume['cnt'][head][key][k] += int(value)

            jtpr, jtpr_details = compute_jtpr(
                nucleus_pred=nucleus_pred,
                cytoplasm_pred=cytoplasm_pred,
                nucleus_gt=nucleus_gt,
                cytoplasm_gt=cytoplasm_gt,
                iou_thresh=0.5,
                assignments_pred=getattr(predictor, '_prompt_assignments', None),
                assignments_gt=sample.get('assignments'), with_pairs=True)
            joint_pairs = jtpr_details.pop('joint_pairs')
            jtpr_values.append(float(jtpr))
            joint_tp_total += int(jtpr_details['joint_tp'])
            per_volume['joint'][0] += int(jtpr_details['joint_tp']); per_volume['joint'][1] += int(jtpr_details['total_pairs'])
            total_pairs_total += int(jtpr_details['total_pairs'])
            for key in pair_failures:
                pair_failures[key] += int(jtpr_details[key])
            for key in interior:
                interior[key] += int(jtpr_details[key])

            print(f"  [{i + 1}/{len(data_pool)}] {filename} | "
                  f"GT nuc={len(np.unique(nucleus_gt)) - 1} "
                  f"cyto={len(np.unique(cytoplasm_gt)) - 1} | "
                  f"pred nuc={len(np.unique(nucleus_pred)) - 1} "
                  f"cyto={len(np.unique(cytoplasm_pred)) - 1} | "
                  f"semantic Dice nuc={sd_n:.3f} cyto={sd_c:.3f} | "
                  f"JTPR={jtpr:.3f}")
            stage_reports = {h:a.report() for h,a in audits.items()} if audits is not None else None
            if stage_reports is not None:
                for head, report in stage_reports.items():
                    for key, value in report['monitoring'].items():
                        name = head + '_' + key
                        monitoring_totals[name] = monitoring_totals.get(name, 0) + value
                    if head in written_quality:
                        joint_owner = [n if head == 'nucleus' else c for n, c in joint_pairs]
                        edge_gt = {str(g): ('y' in v['faces'] or 'x' in v['faces'], v['volume'])
                                   for g, v in report.get('gt_geometry', {}).items()}
                        edge_rows[head]['gt'].extend(edge_gt.values())
                        for d in report['assembly_decisions']:
                            if d['reason'] != 'written':
                                continue
                            gt_id, iou = max(d['written_gt_iou'].items(), key=lambda kv: kv[1],
                                             default=(None, 0.))
                            lost = joint_owner.count(int(gt_id)) if iou >= .5 else 0
                            written_quality[head].append((d['quality'], iou, lost,
                                                          d.get('connected_voxels', d['area'])))
                            edge_rows[head]['masks'].append((d['quality'], iou, d.get('connected_voxels', d['area']),
                                                             edge_gt.get(str(gt_id))))
                            boundary_rows[head].append((d['quality'], iou, d.get('connected_voxels', d['area']),
                                                        d.get('boundary')))
                            per_volume['rows'][head].append((d['quality'], iou, d.get('connected_voxels', d['area']), None))
            with diagnostic_path.open('a', encoding='utf-8') as diagnostic_file:
                json.dump(dict(epoch=epoch_num, filename=filename,
                    refinement=getattr(self.model, 'last_refinement_diagnostics', None),
                    stage_audit=stage_reports,
                    jtpr=float(jtpr), pair_failures=jtpr_details), diagnostic_file, default=int)
                diagnostic_file.write('\n')

            # visualisation of the first volume only (it costs more than the prediction)
            if i == 0:
                self._save_epoch_visualization_full(
                    epoch=epoch_num, img=img,
                    nuc_gt=nucleus_gt, cyto_gt=cytoplasm_gt,
                    nuc_pred=nucleus_pred, cyto_pred=cytoplasm_pred,
                    details=details, viz_dir=viz_dir,
                    filename=f'epoch_{epoch_num:04d}_fullval.png')

            del nucleus_pred, cytoplasm_pred, details

        # ============================================================
        # AGGREGATION
        # ============================================================

        print(f'  Pair failure counts: {pair_failures}')
        if monitoring_totals:
            print(f'  Refinement monitoring: {monitoring_totals}')
            for key, value in monitoring_totals.items():
                name = 'fullval_' + key
                history.setdefault(name, [None] * len(history.get('fullval_epoch', []))).append(value)
        for key, value in pair_failures.items():
            history.setdefault('fullval_' + key, []).append(value)
        interior_jtpr = (interior['interior_joint_tp'] / interior['interior_total_pairs']
                         if interior['interior_total_pairs'] else None)
        from cytonucs_runtime import checkpoint_jtpr_weight
        jtpr_weight = checkpoint_jtpr_weight(config)
        quality_gates, gate_stats = {}, {}
        if config.refinement_stage_audit and written_quality:
            from cytonucs_runtime import calibrate_quality_gates
            quality_gates, gate_stats = calibrate_quality_gates(
                written_quality, {h: cnt[h] for h in written_quality}, taus, total_pairs_total, jtpr_weight)
        padded = dict(interior_jtpr=interior_jtpr, interior_total_pairs=interior['interior_total_pairs'])
        if len(volume_stats) > 1:
            from cytonucs_runtime import accuracy_profile, boundary_error_ap
            by_volume = {}
            for name, s in sorted(volume_stats.items()):
                row = dict(volumes=s['volumes'], jtpr=s['joint'][0] / s['joint'][1] if s['joint'][1] else 0.0)
                for head in cnt:
                    c = s['cnt'][head]
                    row['ap_' + head] = float(accuracy_profile(c['tp'], c['fp'], c['fn']).mean())
                    if head in gate_stats:      # AP after this validation's gates, as for the totals
                        row['ap_' + head + '_quality_gated'] = boundary_error_ap(
                            s['rows'][head], c, quality_gates[head], gate_stats[head]['min_area'], taus)['ap']
                by_volume[name] = row
                gated = [row.get(f'ap_{h}_quality_gated', row[f'ap_{h}']) for h in ('nucleus', 'cytoplasm')]
                print(f"  By source volume | {name}: {s['volumes']} tile(s), AP {'after gates ' if gate_stats else ''}"
                      f"nucleus {gated[0]:.4f} cytoplasm {gated[1]:.4f}, JTPR {row['jtpr']:.4f}")
            padded['by_volume'] = by_volume
        train_samples = self._train_volumes()
        if train_samples:
            # The same deployed pipeline on volumes the network trains on: how well it fits them.
            train_predictor = CytoNucsPredictor(model=self.model, config=config,
                                                prob_thresh_nucleus=prob_thresh_nucleus,
                                                prob_thresh_cytoplasm=prob_thresh_cytoplasm, nms_thresh=nms_thresh)
            if config.mask_quality_min == 'auto':
                train_predictor.quality_gates = quality_gates or None
            train_metrics = deployed_tile_metrics(train_predictor, train_samples, taus)
            padded.update({'train_' + key: value for key, value in train_metrics.items()})
            if len(train_metrics['by_volume']) > 1:
                for name, row in train_metrics['by_volume'].items():
                    print(f"  Train volumes by source | {name}: {row['volumes']} tile(s), AP nucleus {row['ap_nucleus']:.4f} "
                          f"cytoplasm {row['ap_cytoplasm']:.4f}, whole objects {row['ap_nucleus_whole']:.4f} / "
                          f"{row['ap_cytoplasm_whole']:.4f}, JTPR {row['jtpr']:.4f}")
            print(f"  TRAIN VOLUMES ({train_metrics['volumes']}, after gates) | "
                  f"AP nucleus {train_metrics['ap_nucleus']:.4f} (whole objects {train_metrics['ap_nucleus_whole']:.4f}) | "
                  f"AP cytoplasm {train_metrics['ap_cytoplasm']:.4f} (whole objects {train_metrics['ap_cytoplasm_whole']:.4f}) | "
                  f"JTPR {train_metrics['jtpr']:.4f}")
        for head in gate_stats:
            padded[head + '_quality_gate'] = quality_gates[head]
            padded[head + '_area_gate'] = gate_stats[head]['min_area']
            for key in ('fp_removed', 'tp_removed', 'pairs_lost'):
                padded[head + '_quality_gate_' + key] = gate_stats[head][key]
            padded['ap_' + head + '_quality_gated'] = gate_stats[head]['ap']
            print(f"  Quality gate {head}: {quality_gates[head]:.4f}, min area {gate_stats[head]['min_area']} "
                  f"removes {gate_stats[head]['fp_removed']} FP, "
                  f"{gate_stats[head]['tp_removed']} TP ({gate_stats[head]['pairs_lost']} pairs); "
                  f"AP {gate_stats[head]['ap_ungated']:.4f} -> {gate_stats[head]['ap']:.4f}")
            if edge_rows[head]['gt']:
                from cytonucs_runtime import edge_fragment_ignored_ap
                min_area = gate_stats[head]['min_area']
                fragment = lambda info: bool(info and info[0] and info[1] < min_area)
                edge = edge_fragment_ignored_ap(
                    [(q, iou, area, fragment(info)) for q, iou, area, info in edge_rows[head]['masks']],
                    [fragment(info) for info in edge_rows[head]['gt']], quality_gates[head], min_area, taus)
                padded['ap_' + head + '_quality_gated_edge_ignored'] = edge['ap']
                padded[head + '_recall50_quality_gated_edge_ignored'] = edge['recall50']
                padded[head + '_edge_fragments_ignored'] = edge['ignored_gt']
                print(f"  Without patch-edge fragments {head} (reporting only): {edge['ignored_gt']} GT and "
                      f"{edge['ignored_masks']} masks ignored; AP after gates {edge['ap']:.4f}, "
                      f"recall@0.5 {edge['recall50']:.4f}")
                # Whole objects only: every GT cut by an XY face of its tile is an ignore region, whatever
                # its size (the counterpart of fullval_train_ap_<head>_whole on the validation volumes).
                cut = lambda info: bool(info and info[0])
                whole = edge_fragment_ignored_ap(
                    [(q, iou, area, cut(info)) for q, iou, area, info in edge_rows[head]['masks']],
                    [cut(info) for info in edge_rows[head]['gt']], quality_gates[head], min_area, taus)
                padded['ap_' + head + '_quality_gated_whole'] = whole['ap']
                padded[head + '_recall50_quality_gated_whole'] = whole['recall50']
                print(f"  Whole objects {head} (reporting only): {whole['ignored_gt']} cut GT and "
                      f"{whole['ignored_masks']} masks ignored; AP after gates {whole['ap']:.4f}, "
                      f"recall@0.5 {whole['recall50']:.4f}")
            # Written every validation (None when nothing is matched): history keys stay aligned.
            from cytonucs_runtime import boundary_error_ap
            split = boundary_error_ap(boundary_rows[head], cnt[head], quality_gates[head],
                                      gate_stats[head]['min_area'], taus) if boundary_rows[head] else {}
            measured = bool(split.get('matched_masks'))
            for part in ('z', 'xy', 'mixed'):
                padded[f'ap_{head}_quality_gated_{part}_exact'] = split[f'ap_{part}_exact'] if measured else None
                padded[f'{head}_{part}_error_voxel_share'] = split.get(f'{part}_error_voxel_share')
            for key in ('z_extent_exact_fraction', 'z_extent_one_slice_fraction'):
                padded[head + '_' + key] = split.get(key)
            if measured:
                print(f"  Boundary error {head} (reporting only, {split['matched_masks']} matched masks): "
                      f"AP after gates {split['ap']:.4f}; with exact whole Z slices {split['ap_z_exact']:.4f}, "
                      f"exact XY projection {split['ap_xy_exact']:.4f}, exact rims {split['ap_mixed_exact']:.4f}; "
                      f"error voxels Z/XY/rims {split['z_error_voxel_share']:.3f}/{split['xy_error_voxel_share']:.3f}/"
                      f"{split['mixed_error_voxel_share']:.3f}; slice set exact {split['z_extent_exact_fraction']:.3f}, "
                      f"off by one slice {split['z_extent_one_slice_fraction']:.3f}")
        for key, value in padded.items():
            history.setdefault('fullval_' + key, [None] * len(history.get('fullval_epoch', []))).append(value)

        def _agg(head):
            tp = cnt[head]['tp'].astype(np.float64)
            fp = cnt[head]['fp'].astype(np.float64)
            fn = cnt[head]['fn'].astype(np.float64)
            acc = tp / np.maximum(tp + fp + fn, 1.0)
            s = SimpleNamespace(
                tp=int(tp[0]), fp=int(fp[0]), fn=int(fn[0]),
                precision=float(tp[0] / max(tp[0] + fp[0], 1.0)),
                recall=float(tp[0] / max(tp[0] + fn[0], 1.0)))
            return acc, s

        acc_nucleus, s_n = _agg('nucleus')
        acc_cytoplasm, s_c = _agg('cytoplasm')

        for head, stats in (('nucleus', s_n), ('cytoplasm', s_c)):
            for key in ('tp', 'fp', 'fn', 'precision', 'recall'):
                history.setdefault('fullval_' + head + '_' + key, []).append(getattr(stats, key))
        ap_nucleus = float(np.mean(acc_nucleus))
        ap_cytoplasm = float(np.mean(acc_cytoplasm))
        ap50_nucleus = float(acc_nucleus[0])
        ap50_cytoplasm = float(acc_cytoplasm[0])
        # Stored with this epoch's checkpoints for the 'auto' inference settings.
        if quality_gates:
            self.model.pending_quality_gates = dict(epoch=epoch_num, **quality_gates)
            if hasattr(self, 'trainer'):
                sync_object_min_area(self.trainer.loss_fn, quality_gates)

        semantic_dice_nucleus = float(np.mean(semantic_nucleus))
        semantic_dice_cytoplasm = float(np.mean(semantic_cytoplasm))

        jtpr_mean = float(np.mean(jtpr_values)) if jtpr_values else 0.0
        jtpr_global = (float(joint_tp_total / total_pairs_total)
                       if total_pairs_total > 0 else 0.0)

        # ============================================================
        # CHECKPOINT METRIC
        # ============================================================
        #
        # Not compute_dice() (it averages over matched objects only and hardly penalises FP / FN); the AP counts
        # TP, FP and FN.
        # ============================================================
        from cytonucs_runtime import checkpoint_jtpr_weight
        _w_jtpr = checkpoint_jtpr_weight(self.model.config)
        # Mean AP over the levels the run has; a level it does not have scores 0 and is not counted.
        _n_levels = len(getattr(self.model.config, 'levels', ('nucleus', 'cytoplasm')))
        current_metric = float(
            _w_jtpr * jtpr_global
            + (1.0 - _w_jtpr) * (ap_nucleus + ap_cytoplasm) / _n_levels
        )
        # Full validation runs without the gates so they can be calibrated, but inference
        # applies them (mask_quality_min='auto'): select checkpoints by the gated result.
        selection_metric = current_metric
        if config.mask_quality_min == 'auto' and gate_stats:
            from cytonucs_runtime import deployed_metric
            selection_metric = float(deployed_metric(
                jtpr_global, total_pairs_total,
                {h: gate_stats[h]['ap'] if h in gate_stats else ap
                 for h, ap in (('nucleus', ap_nucleus), ('cytoplasm', ap_cytoplasm))},
                {h: gate_stats[h]['pairs_lost'] for h in gate_stats}, _w_jtpr, _n_levels))
        print(
            "\n  FULL-VOLUME METRICS"
        )
        print(
            f"  GT objects: nuc={s_n.tp + s_n.fn} cyto={s_c.tp + s_c.fn} | "
            f"accuracy step ~1/{max(1, s_n.tp + s_n.fp + s_n.fn)}"
        )
        print(
            f"  Nucleus   | "
            f"semantic Dice="
            f"{semantic_dice_nucleus:.4f} "
            f"| AP={ap_nucleus:.4f} "
            f"| AP@50="
            f"{ap50_nucleus:.4f} "
            f"| TP/FP/FN@50="
            f"{s_n.tp}/{s_n.fp}/{s_n.fn}"
        )

        print(
            f"  Cytoplasm | "
            f"semantic Dice="
            f"{semantic_dice_cytoplasm:.4f} "
            f"| AP={ap_cytoplasm:.4f} "
            f"| AP@50="
            f"{ap50_cytoplasm:.4f} "
            f"| TP/FP/FN@50="
            f"{s_c.tp}/{s_c.fp}/{s_c.fn}"
        )

        print(
            f"  JTPR      | "
            f"mean={jtpr_mean:.4f} "
            f"| global={jtpr_global:.4f} "
            f"({joint_tp_total}/"
            f"{total_pairs_total})"
        )
        # Diagnostic only: pairs whose GT nucleus and cell do not touch XY volume faces.
        print(f"  JTPR interior (no XY-face GT) | "
              f"{interior['interior_joint_tp']}/{interior['interior_total_pairs']}")

        print(
            f"  CHECKPOINT METRIC "
            f"(mean AP heads) = "
            f"{current_metric:.4f} without gates"
            + (f", {selection_metric:.4f} after quality/size gates (used for best)"
               if selection_metric != current_metric else "")
        )

        # ============================================================
        # HISTORY
        # ============================================================

        history.setdefault(
            'fullval_epoch',
            []
        ).append(
            epoch_num
        )

        history.setdefault(
            'fullval_semantic_dice_nucleus',
            []
        ).append(
            semantic_dice_nucleus
        )

        history.setdefault(
            'fullval_semantic_dice_cytoplasm',
            []
        ).append(
            semantic_dice_cytoplasm
        )

        history.setdefault(
            'fullval_ap_nucleus',
            []
        ).append(
            ap_nucleus
        )

        history.setdefault(
            'fullval_ap_cytoplasm',
            []
        ).append(
            ap_cytoplasm
        )

        history.setdefault(
            'fullval_ap50_nucleus',
            []
        ).append(
            ap50_nucleus
        )

        history.setdefault(
            'fullval_ap50_cytoplasm',
            []
        ).append(
            ap50_cytoplasm
        )

        history.setdefault(
            'fullval_jtpr',
            []
        ).append(
            jtpr_global
        )

        history.setdefault(
            'fullval_metric',
            []
        ).append(
            current_metric
        )
        history.setdefault('fullval_metric_deployed', []).append(selection_metric)
        history.setdefault('fullval_total_pairs', []).append(int(total_pairs_total))

        align_fullval_history(history, fullval_lengths)

        # ============================================================
        # COMET
        # ============================================================

        if self.experiment:
            self.experiment.log_metrics(
                {
                    'fullval_semantic_dice_nucleus':
                        semantic_dice_nucleus,

                    'fullval_semantic_dice_cytoplasm':
                        semantic_dice_cytoplasm,

                    'fullval_ap_nucleus':
                        ap_nucleus,

                    'fullval_ap_cytoplasm':
                        ap_cytoplasm,

                    'fullval_ap50_nucleus':
                        ap50_nucleus,

                    'fullval_ap50_cytoplasm':
                        ap50_cytoplasm,

                    'fullval_jtpr':
                        jtpr_global,

                    'fullval_metric':
                        current_metric,

                    'fullval_metric_deployed':
                        selection_metric,
                },
                epoch=epoch_num
            )

        # ============================================================
        # BEST MODEL
        # ============================================================

        if (
                self.save_best
                and selection_metric
                > self.best_metric
        ):
            self.best_metric = (
                selection_metric
            )

            # the main checkpoint (best checkpoint metric)
            save_path = (
                    Path(
                        self.model.basedir
                    )
                    / self.model.name
                    / 'weights_best.weights.h5'
            )

            if hasattr(self, 'trainer'):
                self.trainer.state_best_full.assign(selection_metric)
                self.trainer.save_checkpoint(save_path)
            else:
                self.model.save_weights(save_path)

            best_info = {
                'epoch':
                    epoch_num,

                'metric_name':
                    f'{_w_jtpr:g}_jtpr_plus_{1-_w_jtpr:g}_mean_ap'
                    + ('_after_quality_gates' if selection_metric != current_metric else ''),

                'metric':
                    selection_metric,

                'metric_without_gates':
                    current_metric,

                'ap_nucleus':
                    ap_nucleus,

                'ap_cytoplasm':
                    ap_cytoplasm,

                'ap50_nucleus':
                    ap50_nucleus,

                'ap50_cytoplasm':
                    ap50_cytoplasm,

                'semantic_dice_nucleus':
                    semantic_dice_nucleus,

                'semantic_dice_cytoplasm':
                    semantic_dice_cytoplasm,

                'jtpr_global':
                    jtpr_global,

                'prob_thresh_nucleus':
                    prob_thresh_nucleus,

                'prob_thresh_cytoplasm':
                    prob_thresh_cytoplasm,

                'nms_thresh':
                    nms_thresh,

                'quality_gates':
                    quality_gates,
            }

            best_info_path = (
                    Path(
                        self.model.basedir
                    )
                    / self.model.name
                    / 'best_fullval.json'
            )

            with open(
                    best_info_path,
                    'w'
            ) as f:
                json.dump(
                    best_info,
                    f,
                    indent=2
                )

            print(
                f"  ✅ New best FULL-VOLUME model: "
                f"{selection_metric:.4f}"
            )

            print(
                f"     saved to: "
                f"{save_path}"
            )

    def _save_epoch_visualization_full(self, epoch, img, nuc_gt, cyto_gt, nuc_pred, cyto_pred, details, viz_dir, filename=None):
        """
        Save a visualisation with contrasting colours and sharp boundaries.
        """
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
        from matplotlib.colors import ListedColormap

        def _quick_dice(gt, pred):
            g, p = np.asarray(gt) > 0, np.asarray(pred) > 0
            if g.shape != p.shape:
                return 0.0  # shapes differ (empty prediction at grid resolution): skipped
            denom = g.sum() + p.sum()
            return float(2 * np.logical_and(g, p).sum() / denom) if denom else 0.0
        dice_nuc = _quick_dice(nuc_gt, nuc_pred)
        dice_cyto = _quick_dice(cyto_gt, cyto_pred)

        def _n_obj(a):
            """Number of objects = distinct non-zero labels (not .max(): a cropped ground truth keeps the ids of the
            whole volume)."""
            return int(len(np.unique(np.asarray(a))) - (1 if 0 in np.unique(np.asarray(a)) else 0))

        _vmax = {}
        for _k in ('nucleus_prob', 'cytoplasm_prob'):
            _v = details.get(_k)
            _vmax[_k] = float(np.asarray(_v).max()) if _v is not None else 0.0
        if self.model.config.ndim == 3:
            def _mid(a):
                return a[a.shape[0] // 2] if a is not None else None

            img = _mid(img)
            nuc_gt, cyto_gt = _mid(nuc_gt), _mid(cyto_gt)
            nuc_pred, cyto_pred = _mid(nuc_pred), _mid(cyto_pred)
            details = {
                'nucleus_prob': _mid(details.get('nucleus_prob')),
                'cytoplasm_prob': _mid(details.get('cytoplasm_prob')),
                'nucleus_dist': _mid(details.get('nucleus_dist')) if details.get('nucleus_dist') is not None else None,
                'cytoplasm_dist': _mid(details.get('cytoplasm_dist')) if details.get('cytoplasm_dist') is not None else None,
                'cytoplasm_mask': _mid(details.get('cytoplasm_mask')) if details.get(
                    'cytoplasm_mask') is not None else None,

            }
        # Only the image channels are drawn: a hierarchy of three or more levels appends its pair code.
        channels = int(getattr(self.model.config, 'n_image_channels', 0) or 0)
        if channels and img.ndim == 3 and img.shape[-1] > channels:
            img = img[..., :channels]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        # conversion and normalisation
        if img.ndim == 3 and img.shape[-1] in [3, 4]:
            import cv2
            # percentile normalisation for display (as the model input; robust to hot pixels)
            img_display = img.astype(np.float32)
            p_lo = np.percentile(img_display, 1)
            p_hi = np.percentile(img_display, 99.5)
            img_display = (img_display - p_lo) / (p_hi - p_lo + 1e-8)
            img_display = np.clip(img_display, 0, 1)
            img_gray = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.astype(np.float32)
            # Normalize grayscale to [0, 1]
            img_display = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
            img_display = np.clip(img_display, 0, 1)

        # ============================================================
        # A CONTRASTING COLORMAP
        # ============================================================

        # bright, saturated colours

        n_colors = max(nuc_pred.max(), cyto_pred.max(), nuc_gt.max(), cyto_gt.max()) + 1

        # HSV colours for maximal contrast
        from matplotlib.colors import hsv_to_rgb

        colors = []
        colors.append([0, 0, 0])  # Background = black

        for i in range(1, n_colors):
            # high saturation and value
            hue = (i * 137.508) % 360  # golden angle: neighbouring labels far apart in hue
            saturation = 0.8 + 0.2 * (i % 2)  # alternating 0.8 / 1.0
            value = 0.9 + 0.1 * ((i + 1) % 2)  # alternating 0.9 / 1.0

            rgb = hsv_to_rgb([hue / 360.0, saturation, value])
            colors.append(rgb)

        cmap_contrast = ListedColormap(colors)

        # ============================================================
        # CREATE FIGURE: 3 rows x 4 cols
        # ============================================================

        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.patch.set_facecolor('white')

        # ============================================================
        # ROW 1: NUCLEUS
        # ============================================================

        # 1. Original Image
        if (
                img.ndim == 3
                and img.shape[-1] in [3, 4]
        ):
            axes[1, 0].imshow(
                img_display
            )
        else:
            axes[1, 0].imshow(
                img,
                cmap='gray',
                vmin=np.percentile(
                    img,
                    1
                ),
                vmax=np.percentile(
                    img,
                    99
                )
            )
        axes[0, 0].set_title(' Original Image', fontsize=16, fontweight='bold', pad=10)
        axes[0, 0].axis('off')

        # 2. Nucleus GT
        axes[0, 1].imshow(img, cmap='gray', alpha=0.3)
        if nuc_gt.max() > 0:
            axes[0, 1].imshow(nuc_gt, cmap=cmap_contrast, interpolation='nearest',
                              vmin=0, vmax=n_colors - 1, alpha=0.8)
            # boundaries in white
            boundaries = find_boundaries(nuc_gt, mode='thick')
            axes[0, 1].contour(boundaries, colors='white', linewidths=1.5, levels=[0.5])
        axes[0, 1].set_title(f' Nucleus GT\n({_n_obj(nuc_gt)} objects)',
                             fontsize=16, fontweight='bold', pad=10, color='green')
        axes[0, 1].axis('off')

        # 3. Nucleus Prediction
        axes[0, 2].imshow(img, cmap='gray', alpha=0.3)
        if nuc_pred.max() > 0:
            axes[0, 2].imshow(nuc_pred, cmap=cmap_contrast, interpolation='nearest',
                              vmin=0, vmax=n_colors - 1, alpha=0.8)
            boundaries = find_boundaries(nuc_pred, mode='thick')
            axes[0, 2].contour(boundaries, colors='white', linewidths=1.5, levels=[0.5])
        axes[0, 2].set_title(f' Nucleus Pred\n({_n_obj(nuc_pred)} objects) | Dice={dice_nuc:.3f}',
                             fontsize=16, fontweight='bold', pad=10,
                             color='darkblue' if dice_nuc > 0.8 else 'red')
        axes[0, 2].axis('off')

        # 4. Nucleus Overlay (GT vs Pred)
        overlay_nuc = np.zeros((*nuc_gt.shape, 3), dtype=np.float32)

        # GT boundaries - GREEN
        boundaries_gt = find_boundaries(nuc_gt, mode='thick')
        overlay_nuc[boundaries_gt] = [0, 1, 0]

        # Pred boundaries - RED
        boundaries_pred = find_boundaries(nuc_pred, mode='thick')
        overlay_nuc[boundaries_pred] = [1, 0, 0]

        # Overlap - YELLOW
        overlap = boundaries_gt & boundaries_pred
        overlay_nuc[overlap] = [1, 1, 0]

        axes[0, 3].imshow(img, cmap='gray', alpha=0.5)
        axes[0, 3].imshow(overlay_nuc, alpha=0.9)
        axes[0, 3].set_title(' Overlay\nGreen=GT | Red=Pred | Yellow=Match',
                             fontsize=14, fontweight='bold', pad=10)
        axes[0, 3].axis('off')

        # ============================================================
        # ROW 2: CYTOPLASM
        # ============================================================

        # 5. Original (again)
        if img.ndim == 3 and img.shape[-1] in [3, 4]:
            axes[0, 0].imshow(img_display)
        else:
            axes[1, 0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        axes[1, 0].set_title(' Original Image', fontsize=16, fontweight='bold', pad=10)
        axes[1, 0].axis('off')

        # 6. Cytoplasm GT
        axes[1, 1].imshow(img, cmap='gray', alpha=0.3)
        if cyto_gt.max() > 0:
            axes[1, 1].imshow(cyto_gt, cmap=cmap_contrast, interpolation='nearest',
                              vmin=0, vmax=n_colors - 1, alpha=0.8)
            boundaries = find_boundaries(cyto_gt, mode='thick')
            axes[1, 1].contour(boundaries, colors='white', linewidths=1.5, levels=[0.5])
        axes[1, 1].set_title(f' Cytoplasm GT\n({_n_obj(cyto_gt)} objects)',
                             fontsize=16, fontweight='bold', pad=10, color='green')
        axes[1, 1].axis('off')

        # 7. Cytoplasm Prediction
        axes[1, 2].imshow(img, cmap='gray', alpha=0.3)
        if cyto_pred.max() > 0:
            axes[1, 2].imshow(cyto_pred, cmap=cmap_contrast, interpolation='nearest',
                              vmin=0, vmax=n_colors - 1, alpha=0.8)
            boundaries = find_boundaries(cyto_pred, mode='thick')
            axes[1, 2].contour(boundaries, colors='white', linewidths=1.5, levels=[0.5])
        axes[1, 2].set_title(f' Cytoplasm Pred\n({_n_obj(cyto_pred)} objects) | Dice={dice_cyto:.3f}',
                             fontsize=16, fontweight='bold', pad=10,
                             color='darkblue' if dice_cyto > 0.7 else 'red')
        axes[1, 2].axis('off')

        # 8. Cytoplasm Overlay
        overlay_cyto = np.zeros((*cyto_gt.shape, 3), dtype=np.float32)
        boundaries_gt = find_boundaries(cyto_gt, mode='thick')
        overlay_cyto[boundaries_gt] = [0, 1, 0]
        boundaries_pred = find_boundaries(cyto_pred, mode='thick')
        overlay_cyto[boundaries_pred] = [1, 0, 0]
        overlap = boundaries_gt & boundaries_pred
        overlay_cyto[overlap] = [1, 1, 0]

        axes[1, 3].imshow(img, cmap='gray', alpha=0.5)
        axes[1, 3].imshow(overlay_cyto, alpha=0.9)
        axes[1, 3].set_title(' Overlay\nGreen=GT | Red=Pred | Yellow=Match',
                             fontsize=14, fontweight='bold', pad=10)
        axes[1, 3].axis('off')

        # ============================================================
        # ROW 3: PROBABILITY & DISTANCE MAPS
        # ============================================================

        nuc_prob = details.get('nucleus_prob')
        cyto_prob = details.get('cytoplasm_prob')
        nuc_dist = details.get('nucleus_dist')
        cyto_dist = details.get('cytoplasm_dist')
        # the maps may be missing or None (dropped to save memory): defaults
        if nuc_prob is None:  nuc_prob = np.zeros_like(nuc_gt, dtype=float)
        if cyto_prob is None: cyto_prob = np.zeros_like(cyto_gt, dtype=float)
        if nuc_dist is None:  nuc_dist = np.zeros((*nuc_gt.shape, 1), dtype=float)
        if cyto_dist is None: cyto_dist = np.zeros((*cyto_gt.shape, 1), dtype=float)

        # 9. Nucleus Probability
        im1 = axes[2, 0].imshow(nuc_prob, cmap='hot', vmin=0, vmax=1)
        axes[2, 0].set_title(f' Nucleus Prob\nMax(slice)={nuc_prob.max():.3f} '
                             f'Max(volume)={_vmax["nucleus_prob"]:.3f}',
                             fontsize=14, fontweight='bold', pad=10)
        axes[2, 0].axis('off')
        plt.colorbar(im1, ax=axes[2, 0], fraction=0.046, pad=0.04)

        # 10. Cytoplasm Probability
        im2 = axes[2, 1].imshow(cyto_prob, cmap='hot', vmin=0, vmax=1)
        axes[2, 1].set_title(f' Cytoplasm Prob\nMax(slice)={cyto_prob.max():.3f} '
                             f'Max(volume)={_vmax["cytoplasm_prob"]:.3f}',
                             fontsize=14, fontweight='bold', pad=10)
        axes[2, 1].axis('off')
        plt.colorbar(im2, ax=axes[2, 1], fraction=0.046, pad=0.04)

        # 11. Nucleus Distance (mean over rays)
        nuc_dist_mean = nuc_dist.mean(axis=-1) if nuc_dist.ndim > 2 else nuc_dist
        im3 = axes[2, 2].imshow(nuc_dist_mean, cmap='viridis', vmin=0)
        axes[2, 2].set_title(f' Nucleus Dist\nMax={nuc_dist_mean.max():.1f} px',
                             fontsize=14, fontweight='bold', pad=10)
        axes[2, 2].axis('off')
        plt.colorbar(im3, ax=axes[2, 2], fraction=0.046, pad=0.04)

        # 12. Cytoplasm Distance (mean over rays)
        cyto_dist_mean = cyto_dist.mean(axis=-1) if cyto_dist.ndim > 2 else cyto_dist
        im4 = axes[2, 3].imshow(cyto_dist_mean, cmap='viridis', vmin=0)
        axes[2, 3].set_title(f' Cytoplasm Dist\nMax={cyto_dist_mean.max():.1f} px',
                             fontsize=14, fontweight='bold', pad=10)
        axes[2, 3].axis('off')
        plt.colorbar(im4, ax=axes[2, 3], fraction=0.046, pad=0.04)

        # ============================================================
        # MAIN TITLE with metrics
        # ============================================================

        plt.suptitle(
            f' Epoch {epoch} | Nucleus Dice: {dice_nuc:.3f} | Cytoplasm Dice: {dice_cyto:.3f}',
            fontsize=20, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # ============================================================
        # SAVE
        # ============================================================
        if filename is None:
            filename = f'epoch_{epoch:04d}_full.png'

        save_path = viz_dir / filename

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


        print(f"  📷 Saved full visualization: {save_path.name}")




def compute_dice(gt, pred):
    """
    Compute instance-level Dice coefficient.

    Args:
        gt: ground truth instance mask (H, W) with unique ids
        pred: predicted instance mask (H, W) with unique ids

    Returns:
        dice: mean Dice score for matched instances [0, 1]
    """
    from scipy.optimize import linear_sum_assignment
    from skimage.measure import regionprops

    # both empty: perfect
    if gt.max() == 0 and pred.max() == 0:
        return 1.0
    # one empty: failure
    if gt.max() == 0 or pred.max() == 0:
        return 0.0

    # regions
    gt_props = regionprops(gt.astype(np.int32))
    pred_props = regionprops(pred.astype(np.int32))

    if len(gt_props) == 0 and len(pred_props) == 0:
        return 1.0
    if len(gt_props) == 0 or len(pred_props) == 0:
        return 0.0

    # IoU matrix for the matching
    iou_matrix = np.zeros((len(gt_props), len(pred_props)))
    for i, gt_prop in enumerate(gt_props):
        gt_mask = (gt == gt_prop.label)
        for j, pred_prop in enumerate(pred_props):
            pred_mask = (pred == pred_prop.label)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Dice of the matched pairs (IoU > 0.5)
    dice_scores = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] > 0.5:
            gt_mask = (gt == gt_props[i].label)
            pred_mask = (pred == pred_props[j].label)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            dice = 2 * intersection / (gt_mask.sum() + pred_mask.sum())
            dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    # mean Dice over the matched objects
    return np.mean(dice_scores)
