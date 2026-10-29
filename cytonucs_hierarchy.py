"""Hierarchies of three or more nested object levels (model.levels).

The network segments a nested pair: an inner and an outer object role. A hierarchy of N levels is
run as its N - 1 adjacent pairs through that one network, with the same weights: constant input
channels behind the image say which pair the two roles hold (CytoNucsConfig.pair_view). A training
batch holds one pair, full validation runs the two-level validation once per pair, and
predict_hierarchy joins the pairs into one label image per level with child -> parent links.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def pair_name(config):
    """'<inner>_<outer>' of a pair view."""
    return '_'.join(config.levels)


def with_pair_code(image, config):
    """Normalised image with the pair code channels of the pair view `config` appended."""
    image = np.asarray(image, np.float32)
    nd = int(config.ndim)
    if image.ndim == nd:
        image = image[..., None]
    if image.shape[-1] != config.n_image_channels:
        raise ValueError(f'image has {image.shape[-1]} channel(s), the run reads {config.n_image_channels}')
    code = np.zeros(image.shape[:nd] + (config.n_pair_channels,), np.float32)
    code[..., int(config.pair_index)] = 1.
    return np.concatenate([image, code], axis=-1)


class ModelPairView:
    """The network as one pair sees it: the same layers and weights, the pair's config, and the
    pair's own calibration (quality gates, mask thresholds) and caches. Whatever is not set on
    the view is read from the model."""

    def __init__(self, model, config):
        self._model = model
        self.config = config
        self.loaded_quality_gates = None
        self.loaded_mask_thresholds = None
        self.pending_quality_gates = None

    def __getattr__(self, name):
        if name == '_model':
            raise AttributeError(name)
        return getattr(self._model, name)


class _CodedSample(dict):
    """A pool sample whose image carries the pair code (built on access, not stored)."""

    def __init__(self, sample, config):
        super().__init__(sample)
        self._config = config

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return with_pair_code(value, self._config) if key == 'image' else value

    def get(self, key, default=None):
        return self[key] if key in self else default


class PoolView:
    """What full validation reads from a generator, for one pair: its tiles, images with the pair code."""

    def __init__(self, generator):
        self.config = generator.config
        self.subset = generator.subset
        self.image_files = generator.image_files
        self.data_pool = [_CodedSample(sample, generator.config) for sample in generator.data_pool]


class HierarchyGenerator:
    """Batches of a hierarchy: one CytoNucsDataGenerator per adjacent pair, reading the masks of the
    pair's two levels into the inner and outer role. A training batch holds one pair, drawn per
    batch; validation runs through the pairs in turn. Every batch carries 'pair_index'."""

    def __init__(self, dataset_dir, config, augmenter=None, shuffle=True, subset='train', pool_size=None, repool_freq=None):
        from cytonucs_data_generator import CytoNucsDataGenerator
        self.config, self.subset = config, subset
        self.seed = int(getattr(config, 'seed', 42))
        self.epoch = 0
        self.pairs = []
        for index in range(len(config.level_pairs)):
            view = config.pair_view(index)
            self.pairs.append(CytoNucsDataGenerator(dataset_dir, view, view.rays, augmenter=augmenter, shuffle=shuffle,
                                                    subset=subset, pool_size=pool_size, repool_freq=repool_freq))
        self.image_files = self.pairs[0].image_files
        self.data_pool = self.pairs[0].data_pool

    def set_config(self, config):
        """Pair views with the measured geometry: targets of every pair use its own rays."""
        self.config = config
        for index, generator in enumerate(self.pairs):
            view = config.pair_view(index)
            generator.config = view
            generator.rays = view.rays
            generator._nuc_grid = tuple(int(g) for g in view.nucleus_grid)
            generator._cyto_grid = tuple(int(g) for g in view.cytoplasm_grid)
            generator._nuc_rays, generator._cyto_rays = view.rays_nucleus, view.rays_cytoplasm
            generator._nuc_aniso, generator._cyto_aniso = view.rays_anisotropy_nucleus, view.rays_anisotropy_cytoplasm

    def level_labels(self):
        """level -> the label arrays of that level in the loaded tiles (for the geometry measurement)."""
        labels = {}
        for generator in self.pairs:
            for role, key in (('nucleus', 'nucleus_mask'), ('cytoplasm', 'cytoplasm_mask')):
                labels.setdefault(generator.config.level_roles[role], [s[key] for s in generator.data_pool])
        return labels

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        for generator in self.pairs:
            generator.set_epoch(epoch)

    def on_epoch_end(self):
        for generator in self.pairs:
            generator.on_epoch_end()

    def __len__(self):
        if self.subset == 'train':
            return len(self.pairs[0])
        return sum(len(generator) for generator in self.pairs)

    def __getitem__(self, idx):
        if self.subset == 'train':
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(idx), 7]))
            pair = int(rng.integers(len(self.pairs)))
            x, y = self.pairs[pair][idx]
        else:
            pair, local = 0, int(idx) % max(1, len(self))
            while local >= len(self.pairs[pair]):
                local -= len(self.pairs[pair])
                pair += 1
            x, y = self.pairs[pair][local]
        y['pair_index'] = pair
        return x, y


def build_trainers(model, config, gradient_clip_norm, optimizer=None):
    """One trainer per adjacent pair over the same network and optimizer. The first one runs the
    training loop and sends every batch to the trainer of its pair (CytoNucsTrainer._pair)."""
    from cytonucs_losses import CytoNucsLoss
    from cytonucs_trainer import CytoNucsTrainer
    views = [ModelPairView(model, config.pair_view(index)) for index in range(len(config.level_pairs))]
    trainers = []
    for view in views:
        trainer = CytoNucsTrainer(view, view.config, CytoNucsLoss(view.config), gradient_clip_norm,
                                  optimizer=trainers[0].optimizer if trainers else optimizer)
        if trainers:
            trainer.train_loss_metric, trainer.val_loss_metric = trainers[0].train_loss_metric, trainers[0].val_loss_metric
        trainers.append(trainer)
    trainers[0].pair_trainers = trainers
    return trainers, views


class HierarchyCallback:
    """Full validation of a hierarchy: the two-level full validation once per adjacent pair, logged
    under history['pairs'][<inner>_<outer>]. The checkpoint metric is the mean of the pair metrics;
    the best epoch is saved as weights_best with the quality gates of every pair beside it."""

    def __init__(self, callbacks, views, trainer, save_best=True):
        self.callbacks, self.views, self.trainer, self.save_best = callbacks, views, trainer, save_best
        self.names = [pair_name(view.config) for view in views]
        self.best_metric = float('-inf')

    def calibration(self):
        """pair -> quality gates of its last full validation (without the epoch stamp)."""
        result = {}
        for name, view in zip(self.names, self.views):
            gates = getattr(view, 'pending_quality_gates', None)
            result[name] = None if gates is None else {k: v for k, v in gates.items() if k != 'epoch'}
        return result

    def on_epoch_end(self, epoch, history):
        metrics = {}
        for name, callback in zip(self.names, self.callbacks):
            log = history.setdefault('pairs', {}).setdefault(name, {})
            before = len(log.get('fullval_epoch', []))
            print(f"\n=== Pair {name.replace('_', ' in ', 1)} ===")
            callback.on_epoch_end(epoch, log)
            if len(log.get('fullval_epoch', [])) > before:
                deployed = log.get('fullval_metric_deployed') or [None]
                metrics[name] = float(deployed[-1] if deployed[-1] is not None else log['fullval_metric'][-1])
        if len(metrics) < len(self.names):
            return
        metric = float(np.mean(list(metrics.values())))
        history.setdefault('fullval_epoch', []).append(epoch + 1)
        history.setdefault('fullval_metric', []).append(metric)
        print(f"\n  HIERARCHY METRIC (mean over pairs) = {metric:.4f} | " +
              ', '.join(f'{name}: {value:.4f}' for name, value in metrics.items()))
        if self.save_best and metric > self.best_metric:
            self.best_metric = metric
            path = self.trainer.checkpoint_dir / 'weights_best.weights.h5'
            self.trainer.state_best_full.assign(metric)
            self.trainer.save_checkpoint(path)
            write_calibration(path, self.calibration())
            (self.trainer.checkpoint_dir / 'best_fullval.json').write_text(json.dumps(dict(
                epoch=epoch + 1, metric_name='mean_over_pairs_of_the_pair_checkpoint_metric', metric=metric,
                pairs=metrics, quality_gates=self.calibration()), indent=2))
            print(f"  ✅ New best hierarchy model: {metric:.4f} -> {path}")


def calibration_path(weights_path):
    return Path(str(weights_path) + '.pairs.json')


def write_calibration(weights_path, calibration):
    calibration_path(weights_path).write_text(json.dumps(calibration, indent=2))


def read_calibration(weights_path):
    path = calibration_path(weights_path)
    return json.loads(path.read_text()) if path.is_file() else {}


def link_children(children, parents):
    """child id -> parent id by the largest overlap (children outside every parent are left out)."""
    children, parents = np.asarray(children), np.asarray(parents)
    both = (children > 0) & (parents > 0)
    links = {}
    if both.any():
        width = int(parents.max()) + 1
        keys, counts = np.unique(children[both].astype(np.int64) * width + parents[both], return_counts=True)
        for key, count in sorted(zip(keys.tolist(), counts.tolist()), key=lambda kc: kc[1]):
            links[int(key // width)] = int(key % width)         # the largest overlap is written last
    return links


def predict_hierarchy(model, config, image, calibration=None, normalize=True, thresholds=None):
    """One label image per level of a hierarchy, with child -> parent links.

    The two-level pipeline runs once per adjacent pair on the image with that pair's code. Every
    level but the outermost is taken from the pair where it is the inner object (found by its
    own peaks); the outermost one from the last pair. Links follow the largest overlap.

    calibration: pair -> quality gates (read_calibration of the checkpoint); thresholds:
    dict(prob_thresh_nucleus, prob_thresh_cytoplasm, nms_thresh) for the predictors.
    Returns dict(levels={level: labels}, parents={child level: {child id: parent id}}, semantic={level: {id:
    mean semantic foreground probability}} (with inference.semantic_scores or semantic_min_*, else empty)).
    """
    from cytonucs_runtime import normalize_percentiles as percentile_normalize
    from cytonucs_inference import CytoNucsPredictor
    from cytonucs_trainer import DiceCallback
    nd = int(config.ndim)
    image = np.asarray(image, np.float32)
    if image.ndim == nd:
        image = image[..., None]
    if normalize:
        image = percentile_normalize(image, 1, 99.8, axis=tuple(range(nd)))
    if not config.n_pair_channels:
        # One or two levels: the two-level pipeline itself, with the quality gates and mask thresholds
        # loaded with the weights and the child -> parent pairs it chose, as the full validation of
        # training scores them.
        predictor = CytoNucsPredictor(model=model, config=config, **(thresholds or {}))
        inner, outer, _ = DiceCallback._predict_full_volume(SimpleNamespace(model=model), predictor, image)
        by_role = dict(nucleus=np.asarray(inner), cytoplasm=np.asarray(outer))
        levels = {level: by_role[role] for role, level in config.level_roles.items()}
        parents = {}
        if len(config.levels) == 2:
            from cytonucs_losses import compute_nucleus_cell_assignments
            chosen = getattr(predictor, '_prompt_assignments', None)
            if chosen is None:                                   # as compute_jtpr does without predicted pairs
                chosen = compute_nucleus_cell_assignments(levels[config.levels[0]], levels[config.levels[1]])[0]
            parents[config.levels[0]] = {int(k): int(v) for k, v in chosen.items()}
        scores = getattr(predictor, '_last_semantic_scores', None) or {}
        semantic = {level: scores[role] for role, level in config.level_roles.items() if role in scores}
        return dict(levels=levels, parents=parents, semantic=semantic)
    levels, semantic = {}, {}
    outer = None
    for index in range(len(config.level_pairs)):
        view = ModelPairView(model, config.pair_view(index))
        view.loaded_quality_gates = (calibration or {}).get(pair_name(view.config))
        predictor = CytoNucsPredictor(model=view, config=view.config, **(thresholds or {}))
        inner, outer, _ = DiceCallback._predict_full_volume(SimpleNamespace(model=view), predictor,
                                                            with_pair_code(image, view.config))
        levels[view.config.levels[0]] = np.asarray(inner)
        scores = getattr(predictor, '_last_semantic_scores', None) or {}
        if 'nucleus' in scores:
            semantic[view.config.levels[0]] = scores['nucleus']
        if 'cytoplasm' in scores:
            semantic[config.levels[-1]] = scores['cytoplasm']                    # the last pair's stays
    levels[config.levels[-1]] = np.asarray(outer)
    parents = {child: link_children(levels[child], levels[parent]) for child, parent in config.level_pairs}
    return dict(levels=levels, parents=parents, semantic=semantic)


def train_hierarchy(config_dict, config, experiment=None):
    """Training run of a hierarchy of three or more levels (called by train_cytonucs.main)."""
    import tensorflow as tf
    from cytonucs_data_generator import cytonucs_augmenter_2d, cytonucs_augmenter_3d
    from cytonucs_model import build_cytonucs_model
    from cytonucs_trainer import DiceCallback
    from train_cytonucs import WarmupExponentialDecay

    training, evaluation, inference = config_dict['training'], config_dict['evaluation'], config_dict['inference']
    if training.get('resume_from_checkpoint'):
        raise NotImplementedError('Resuming is not implemented for three or more levels')
    print(f"\nHierarchy of {len(config.levels)} levels: " + ' in '.join(config.levels) +
          f" | pairs through one network: {[pair_name(config.pair_view(i)) for i in range(len(config.level_pairs))]}")
    dataset = Path(config_dict['data']['dataset_dir'])
    augmenter = None
    if config_dict['augmentation']['enable']:
        augmenter = cytonucs_augmenter_3d if config.ndim == 3 else cytonucs_augmenter_2d
    train = HierarchyGenerator(dataset / 'train', config, augmenter, shuffle=True, subset='train',
                               pool_size=config_dict['data'].get('pool_size'), repool_freq=config_dict['data'].get('repool_freq', 10))
    val = HierarchyGenerator(dataset / 'val', config, None, shuffle=False, subset='val')
    config.set_hierarchy_geometry(train.level_labels())
    for generator in (train, val):
        generator.set_config(config)

    model = build_cytonucs_model(config=config, name=config_dict['experiment']['name'], basedir=config_dict['output']['basedir'])
    steps = config.train_steps_per_epoch
    schedule = WarmupExponentialDecay(
        initial_lr=training['initial_learning_rate'], warmup_steps=max(1, int(training.get('lr_warmup_epochs', 3) * steps)),
        decay_steps=training.get('lr_decay_steps', 10000), decay_rate=training.get('lr_decay_rate', 0.9),
        min_lr=training.get('min_learning_rate', 1e-7), step_offset=0, staircase=training.get('lr_staircase', False))
    trainers, views = build_trainers(model, config, training['gradient_clip_norm'],
                                     tf.keras.optimizers.Adam(learning_rate=schedule))
    callbacks = []
    if evaluation['eval_dice_every'] > 0:
        pair_callbacks = []
        for view, trainer, train_pair, val_pair in zip(views, trainers, train.pairs, val.pairs):
            callback = DiceCallback(model=view, val_generator=PoolView(val_pair), eval_every=evaluation['eval_dice_every'],
                                    save_best=False, experiment=None,
                                    prob_thresh_nucleus=inference.get('prob_thresh_nucleus', 0.5),
                                    prob_thresh_cytoplasm=inference.get('prob_thresh_cytoplasm', 0.75),
                                    nms_thresh=inference.get('nms_thresh', 0.3), train_generator=PoolView(train_pair))
            callback.trainer = trainer
            pair_callbacks.append(callback)
        callbacks.append(HierarchyCallback(pair_callbacks, views, trainers[0], save_best=config_dict['output'].get('save_best_dice', True)))
    trainers[0].train(train, val, epochs=config.train_epochs, steps_per_epoch=steps, callbacks=callbacks, experiment=experiment)
    final = trainers[0].checkpoint_dir / 'weights_final.weights.h5'
    trainers[0].save_checkpoint(final)
    if callbacks:
        write_calibration(final, callbacks[0].calibration())
    print(f"\n🎉 Hierarchy training complete: {trainers[0].checkpoint_dir}")
    return dict(history=trainers[0].history, model=model, config=config,
                calibration=callbacks[0].calibration() if callbacks else {})
