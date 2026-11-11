# cytonucs_losses.py - ФИНАЛЬНАЯ версия со ВСЕМИ лоссами

import tensorflow as tf
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

class CytoNucsLoss:
    """
    Полный loss с:
    - StarDist BCE + MAE (через C++)
    - WBR (Within-Boundary Regularization)
    - Containment (geometric + distance field)
    - Consistency (single object, no fragmentation)
    """

    def __init__(self, config):
        self.config = config
        self.ndim = config.ndim
        self.epsilon = 1e-6
        self.model = None
        # Существующие веса
        self.lambda_wbr = config.lambda_wbr
        self.lambda_containment = config.lambda_containment
        self.lambda_consistency = config.lambda_consistency if config.enable_consistency else 0.0
        self._loss_call_count = 0
        self._last_epoch = -1
        self._debug_count = {'nucleus': 0, 'cytoplasm': 0}
        # НОВЫЕ веса (с fallback на 0 если не заданы)
        self.lambda_fragmentation = getattr(config, 'lambda_fragmentation', 0.0)
        self.lambda_background_penalty = getattr(config, 'lambda_background_penalty', 0.0)
        self.dist_loss = getattr(config, 'dist_loss', 'mae')  # 'mae' or 'huber' or 'mse'
        self.huber_delta = getattr(config, 'huber_delta', 1.0)
        self.wbr_warmup_epochs = getattr(config, 'wbr_warmup_epochs', 10)
        self.current_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='current_epoch')
        # Smallest object (voxels) kept in object-level probability losses, per head
        # [nucleus, cytoplasm]; synced from the calibrated size gates by the trainer.
        self.object_min_voxels = tf.Variable([0., 0.], dtype=tf.float32, trainable=False,
                                             name='object_min_voxels')

        # Refinement heads weights
        self.lambda_cyto_mask = getattr(config, 'lambda_cyto_mask', 0.0)


        self.lambda_nuc_mask = getattr(config, 'lambda_nuc_mask', 0.0)
        self.n_prompts = max(1, int(getattr(config, 'prompts_per_crop', config.train_batch_size)))


        # Anisotropy для анизотропных окон max-pool в лоссах (универсально 2D/3D)
        self.anisotropy = tuple(getattr(config, 'anisotropy', (1,) * self.ndim))
        if len(self.anisotropy) != self.ndim:
            self.anisotropy = (1,) * self.ndim
    def _warmup_factor(self, current_epoch):

        ramp = float(getattr(self.config, 'loss_warmup_epochs',
                             getattr(self.config, 'wbr_warmup_epochs', 20)))
        ramp = max(ramp, 1.0)
        e = tf.cast(current_epoch, tf.float32)
        return tf.clip_by_value((e - ramp) / ramp, 0.0, 1.0)

    def _match_to_cytoplasm(self, t):
        """Приводит тензор nucleus-разрешения к cytoplasm-разрешению max-пулингом.
        Фактор из соотношения nucleus_grid/cytoplasm_grid (целочисленный)."""
        ng = self._get_nucleus_grid()
        cg = self._get_cytoplasm_grid()
        for axis, (n, c) in enumerate(zip(ng, cg), 1):
            if n > c:
                if n % c:
                    raise ValueError('Head grids must have integer ratios')
                t = tf.repeat(t, n // c, axis=axis)
        factor = tuple(max(1, int(round(c / n))) for n, c in zip(ng, cg))
        if all(f == 1 for f in factor):
            return t
        if self.ndim == 3:
            return tf.nn.max_pool3d(t, ksize=[1, *factor, 1],
                                    strides=[1, *factor, 1], padding='VALID')
        else:
            return tf.nn.max_pool2d(t, ksize=[1, *factor, 1],
                                    strides=[1, *factor, 1], padding='VALID')
    def _get_nucleus_grid(self):
        grid = tuple(getattr(self.config, 'nucleus_grid', (1,) * self.ndim))
        if len(grid) != self.ndim or any(int(g) < 1 for g in grid):
            grid = (1,) * self.ndim
        return tuple(int(g) for g in grid)

    def _get_cytoplasm_grid(self):
        grid = tuple(getattr(self.config, 'cytoplasm_grid', (1,) * self.ndim))
        if len(grid) != self.ndim or any(int(g) < 1 for g in grid):
            grid = (1,) * self.ndim
        return tuple(int(g) for g in grid)

    def _subsample(self, t, grid):
        """Прореживает spatial-оси тензора (B, spatial..., C) на заданный grid."""
        if all(g == 1 for g in grid):
            return t
        if self.ndim == 3:
            gz, gy, gx = grid
            return t[:, ::gz, ::gy, ::gx, ...]
        else:
            gy, gx = grid
            return t[:, ::gy, ::gx, ...]

    def _grid_subsample_nucleus(self, t):
        return self._subsample(t, self._get_nucleus_grid())

    def _grid_subsample_cytoplasm(self, t):
        return self._subsample(t, self._get_cytoplasm_grid())

    def _aniso_ksize(self, base):
        """Per-axis нечётный kernel для max-pool окон поиска пиков, масштабируемый
        по anisotropy: грубая (низкоразрешённая) ось получает меньше вокселей,
        чтобы физическое окно оставалось ~изотропным.
        2D / изотропия → base по всем осям (поведение не меняется)."""
        min_a = min(self.anisotropy)
        ks = []
        for a in self.anisotropy:
            k = max(1, int(round(base * min_a / a)))
            if k % 2 == 0:
                k += 1
            ks.append(k)
        return ks

    def __call__(self, y_true_raw, y_pred, current_epoch=None):
        cfg = self.config

        current_epoch = (self.current_epoch if current_epoch is None else current_epoch) - 1              # для warmup (0-based)

        y_true = self._create_gt_from_instances(y_true_raw)
        # training.partial_annotation: 'levels_present' is 1 for a level annotated in this batch
        # (or not a level of the run) and 0 for a level whose masks the files of the batch lack;
        # the terms of such a level are switched off, since empty masks there mean "unknown".
        # The terms tying nuclei to cells (WBR, containment) stay: they read whatever
        # annotation exists and never ask for an object.
        if 'levels_present' in y_true_raw:
            present = tf.cast(tf.reshape(y_true_raw['levels_present'], [-1]), tf.float32)
            on = {'nucleus': present[0], 'cytoplasm': present[1]}
        else:
            on = {'nucleus': 1.0, 'cytoplasm': 1.0}
        # training.unknown_label: '<role>_known' is 1 where the annotation of the level is known and
        # '<role>_target_known' is 1 for a prompt whose target is known. Voxel terms of a level are
        # averaged over its known voxels; an unknown empty target is skipped by the prompt terms.
        known, slot_known = {}, {}
        for head in ('nucleus', 'cytoplasm'):
            given = head + '_known' in y_true_raw
            known[head] = tf.stop_gradient(self._subsample(
                tf.cast(y_true_raw[head + '_known'], tf.float32)[..., None], getattr(cfg, head + '_grid'))) if given else None
            slot_known[head] = tf.reshape(tf.cast(y_true_raw[head + '_target_known'], tf.float32), [-1]) if given else None
        # Objects cut by the training crop (generator masks: fewer voxels than in the
        # source volume); older inputs carry only per-axis face flags.
        crop_cut = {h: y_true_raw.get(h + '_crop_cut') for h in ('nucleus', 'cytoplasm')}
        truncating_axes = y_true_raw.get('crop_truncation_axes')
        if cfg.object_losses_include_crop_cut:
            crop_cut = {h: None for h in crop_cut}
            truncating_axes = None

        # === 1. StarDist Loss ===
        # Objects cut by the crop would get full object weight from a sliver.
        # Voxel BCE still supervises them; only the object mean skips them.
        def object_labels(head):
            if not (getattr(cfg, head + '_prob_object_weight') or cfg.prob_object_peak_weight):
                return None
            grid = getattr(cfg, head + '_grid')
            labels = self._subsample(y_true_raw[head + '_instances'], grid)
            if crop_cut[head] is not None:
                keep = tf.cast(self._subsample(crop_cut[head], grid), labels.dtype) == 0
                labels = tf.where(keep, labels, tf.zeros_like(labels))
            elif truncating_axes is not None:
                labels = without_truncated_objects(y_true_raw[head + '_instances'], labels,
                                                   truncating_axes, self.ndim)
            if cfg.object_losses_min_area_from_gate:
                # Objects smaller than the calibrated size gate are removed at inference
                # anyway; asking for a confident peak on them only teaches peaks on specks.
                labels = without_small_objects(y_true_raw[head + '_instances'], labels,
                                               self.object_min_voxels[0 if head == 'nucleus' else 1])
            return labels
        warmup = self._warmup_factor(current_epoch)
        loss_nucleus = self._stardist_loss(
            y_true['nucleus_prob'], y_pred['nucleus_prob'],
            y_true['nucleus_dist'], y_pred['nucleus_dist'],
            name='nucleus', prob_logit=y_pred.get('nucleus_prob_logit'),
            instances=object_labels('nucleus'), known=known['nucleus'], quality_ramp=warmup)
        loss_cytoplasm = self._stardist_loss(
            y_true['cytoplasm_prob'], y_pred['cytoplasm_prob'],
            y_true['cytoplasm_dist'], y_pred['cytoplasm_dist'],
            name='cytoplasm', prob_logit=y_pred.get('cytoplasm_prob_logit'),
            instances=object_labels('cytoplasm'), known=known['cytoplasm'], quality_ramp=warmup)

        # Бинарный GT-контейнер клеток на цито-гриде — общий для WBR
        # Контейнер для WBR и containment: клетка ИЛИ ядро. Ядро по определению
        # принадлежит клетке, даже если разметка цитоплазмы его не покрывает
        # (свободное ядро, или срезы, где ядро толще плоского цито-слоя).
        cyto_fg_gt = self._grid_subsample_cytoplasm(
            tf.expand_dims(tf.cast(y_true_raw['cytoplasm_instances'] > 0,
                                   tf.float32), axis=-1))
        nuc_fg_at_cyto = self._match_to_cytoplasm(self._grid_subsample_nucleus(
            tf.expand_dims(tf.cast(y_true_raw['nucleus_instances'] > 0,
                                   tf.float32), axis=-1)))
        container_fg = tf.stop_gradient(tf.maximum(cyto_fg_gt, nuc_fg_at_cyto))
        if known['cytoplasm'] is not None:
            # Where the cells are not annotated, a nucleus may lie inside one.
            container_fg = tf.maximum(container_fg, 1. - known['cytoplasm'])

        # === 2. WBR ===
        wbr_raw = self._wbr_loss(y_pred['nucleus_prob'], container_fg)
        loss_wbr = self.lambda_wbr * warmup * wbr_raw

        # === 3. Containment ===
        loss_containment = self._containment_loss(y_true, y_pred, y_true_raw)

        # === 4. Fragmentation ===
        loss_fragmentation = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_fragmentation > 0:
            nuc_inst_g = self._grid_subsample_nucleus(
                tf.expand_dims(tf.cast(y_true_raw['nucleus_instances'], tf.float32), -1))[..., 0]
            cyto_inst_g = self._grid_subsample_cytoplasm(
                tf.expand_dims(tf.cast(y_true_raw['cytoplasm_instances'], tf.float32), -1))[..., 0]
            loss_frag_nuc = self._fragmentation_loss(
                y_pred['nucleus_prob'], nuc_inst_g, name='nucleus')
            loss_frag_cyto = self._fragmentation_loss(
                y_pred['cytoplasm_prob'], cyto_inst_g, name='cytoplasm')
            loss_fragmentation = self.lambda_fragmentation * (
                on['nucleus'] * loss_frag_nuc + on['cytoplasm'] * loss_frag_cyto) / 2.0

        # === 5. Background Penalty ===
        loss_background = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_background_penalty > 0:
            loss_bg_nuc = self._background_penalty_loss(
                y_pred['nucleus_prob'], y_true['nucleus_prob'], name='nucleus', known=known['nucleus'])
            loss_bg_cyto = self._background_penalty_loss(
                y_pred['cytoplasm_prob'], y_true['cytoplasm_prob'], name='cytoplasm', known=known['cytoplasm'])
            loss_background = self.lambda_background_penalty * (
                on['nucleus'] * loss_bg_nuc + on['cytoplasm'] * loss_bg_cyto) / 2.0


        # === 6. Mask refinement — БЕЗ warmup.
        # Decoder gradient is controlled by refinement_gradient_scale.
        # Refinement produces inference instances and trains from the first epoch.
        def _tile_prompts(t):
            """Батч B -> K*B в порядке k*B + b, как склеивает модель."""
            if self.n_prompts == 1:
                return t
            return tf.tile(t, [self.n_prompts] + [1] * (self.ndim + 1))

        loss_cyto_mask = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_cyto_mask > 0 and 'cytoplasm_mask' in y_pred:
            gt_mask_semantic = self._grid_subsample_cytoplasm(tf.expand_dims(
                tf.cast(y_true_raw['cytoplasm_target'], tf.float32), axis=-1))
            _cyto_any = tf.stop_gradient(_tile_prompts(
                self._grid_subsample_cytoplasm(
                    tf.expand_dims(tf.cast(y_true_raw['cytoplasm_instances'] > 0,
                                           tf.float32), axis=-1))))
            loss_cyto_mask = on['cytoplasm'] * self.lambda_cyto_mask * (
                    0.5 * self._refinement_ce_loss(
                y_pred['cytoplasm_mask_logit'], gt_mask_semantic, _cyto_any,
                None if known['cytoplasm'] is None else _tile_prompts(known['cytoplasm']), slot_known['cytoplasm']) +
                    0.5 * self._refinement_overlap_loss(
                y_pred['cytoplasm_mask'], gt_mask_semantic))

        loss_nuc_mask = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_nuc_mask > 0 and 'nucleus_mask' in y_pred:
            gt_nuc_semantic = self._grid_subsample_nucleus(tf.expand_dims(
                tf.cast(y_true_raw['nucleus_target'], tf.float32), axis=-1))
            _nuc_any = tf.stop_gradient(_tile_prompts(
                self._grid_subsample_nucleus(
                    tf.expand_dims(tf.cast(y_true_raw['nucleus_instances'] > 0,
                                           tf.float32), axis=-1))))
            loss_nuc_mask = on['nucleus'] * self.lambda_nuc_mask * (
                    0.5 * self._refinement_ce_loss(
                y_pred['nucleus_mask_logit'], gt_nuc_semantic, _nuc_any,
                None if known['nucleus'] is None else _tile_prompts(known['nucleus']), slot_known['nucleus']) +
                    0.5 * self._refinement_overlap_loss(
                y_pred['nucleus_mask'], gt_nuc_semantic))

        fine_mask = tf.constant(0., tf.float32)
        fine_monitoring = {}
        if cfg.refinement_full_resolution:
            # Full-resolution head: the same CE + overlap as the coarse mask, on the window the
            # model cropped around the prompt slot it picked (rows k*B..(k+1)*B of the K-major
            # targets; full-resolution targets, no subsampling).
            from cytonucs_fine_mask import crop
            for head, weight in (('nucleus', self.lambda_nuc_mask), ('cytoplasm', self.lambda_cyto_mask)):
                if not weight or head + '_fine_logit' not in y_pred:
                    continue
                logit = tf.cast(y_pred[head + '_fine_logit'], tf.float32)
                begin = tf.cast(y_pred[head + '_fine_begin'], tf.int32)
                size = tf.shape(logit)[1:1 + self.ndim]
                first = tf.shape(logit)[0]
                slot = tf.cast(tf.reshape(y_pred['fine_prompt_slot'], [-1])[0], tf.int32) \
                    if 'fine_prompt_slot' in y_pred else 0
                target = crop(tf.cast(y_true_raw[head + '_target'], tf.float32)[slot * first:(slot + 1) * first, ..., None],
                              begin, size)
                any_instance = crop(tf.cast(y_true_raw[head + '_instances'] > 0, tf.float32)[..., None], begin, size)
                voxels, prompts = None, None
                if known[head] is not None:
                    voxels = crop(tf.cast(y_true_raw[head + '_known'], tf.float32)[..., None], begin, size)
                    prompts = slot_known[head][slot * first:(slot + 1) * first]
                probability = tf.nn.softmax(logit, axis=-1)[..., 1:2]
                head_loss = on[head] * weight * (0.5 * self._refinement_ce_loss(logit, target, any_instance, voxels, prompts) +
                                                 0.5 * self._refinement_overlap_loss(probability, target))
                fine_monitoring[head + '_fine_mask'] = head_loss
                fine_mask += head_loss
        consistency = tf.constant(0., tf.float32)
        if self.lambda_consistency:
            consistency = self.lambda_consistency * warmup * 0.5 * sum(
                on[h] * self._consistency_loss(y_pred[h + '_prob'], y_true[h + '_prob'])
                for h in ('nucleus', 'cytoplasm'))
        prompt_consistency = tf.constant(0., tf.float32)
        if cfg.prompt_consistency_weight and self.n_prompts > 1:
            for head in ('nucleus', 'cytoplasm'):
                target = self._subsample(tf.cast(y_true_raw[head + '_target'][..., None], tf.float32),
                                         getattr(cfg, head + '_grid'))
                prompt_consistency += on[head] * cfg.prompt_consistency_weight * 0.5 * self._prompt_consistency(
                    y_pred[head + '_mask'], target)
        cell_prompt_separation = tf.constant(0., tf.float32)
        if cfg.cell_prompt_separation_weight and self.n_prompts > 1:
            target = self._subsample(tf.cast(y_true_raw['cytoplasm_target'][..., None], tf.float32),
                                     cfg.cytoplasm_grid)
            cell_prompt_separation = on['cytoplasm'] * cfg.cell_prompt_separation_weight * self._cell_prompt_separation(
                y_pred['cytoplasm_mask_logit'], target)
        cell_z_dice = tf.constant(0., tf.float32)
        if cfg.cell_z_dice_weight:
            target = self._subsample(tf.cast(y_true_raw['cytoplasm_target'][..., None], tf.float32),
                                     cfg.cytoplasm_grid)
            cell_z_dice = on['cytoplasm'] * cfg.cell_z_dice_weight * self._cell_z_dice(y_pred['cytoplasm_mask'], target)
        nucleus_z_dice = tf.constant(0., tf.float32)
        if cfg.nucleus_z_dice_weight:
            target = self._subsample(tf.cast(y_true_raw['nucleus_target'][..., None], tf.float32),
                                     cfg.nucleus_grid)
            nucleus_z_dice = on['nucleus'] * cfg.nucleus_z_dice_weight * self._cell_z_dice(y_pred['nucleus_mask'], target)
        semantic_foreground = tf.constant(0., tf.float32)
        semantic_monitoring = {}
        if cfg.semantic_foreground_weight:
            # Voxel truth, so crop-cut objects count as foreground like any other.
            for head in ('nucleus', 'cytoplasm'):
                target = self._subsample(tf.cast(y_true_raw[head + '_instances'] > 0, tf.float32)[..., None],
                                         getattr(cfg, head + '_grid'))
                logit = tf.cast(y_pred[head + '_semantic_logit'], tf.float32)
                probability = tf.sigmoid(logit)
                if known[head] is None:
                    bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logit))
                    overlap = self._refinement_overlap_loss(probability, target)
                else:
                    bce = tf.math.divide_no_nan(tf.reduce_sum(known[head] * tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=target, logits=logit)), tf.reduce_sum(known[head]))
                    overlap = self._refinement_overlap_loss(probability * known[head], target * known[head])
                binary = tf.cast(probability > 0.5, tf.float32)
                semantic_monitoring[head + '_semantic_bce'] = bce
                semantic_monitoring[head + '_semantic_overlap'] = overlap
                semantic_monitoring[head + '_semantic_dice'] = tf.math.divide_no_nan(
                    2. * tf.reduce_sum(binary * target), tf.reduce_sum(binary) + tf.reduce_sum(target))
                semantic_foreground += on[head] * cfg.semantic_foreground_weight * 0.5 * (0.5 * bce + 0.5 * overlap)
        center_offset = tf.constant(0., tf.float32)
        offset_monitoring = {}
        if cfg.center_offset_weight:
            # Offset to the object's centre at every voxel of an object (Huber, in head-grid nodes); the centre is
            # where the object's prob target peaks, on the object's part inside the crop.
            for head in ('nucleus', 'cytoplasm'):
                labels = self._subsample(y_true_raw[head + '_instances'], getattr(cfg, head + '_grid'))
                prob_target = y_true[head + '_prob']
                if prob_target.shape.rank == self.ndim + 2:
                    prob_target = prob_target[..., 0]
                target = tf.stop_gradient(center_offset_targets(labels, prob_target, self.ndim))
                weight = tf.cast(labels > 0, tf.float32)
                if known[head] is not None:
                    weight *= known[head][..., 0]
                difference = tf.cast(y_pred[head + '_center_offset'], tf.float32) - target
                error = tf.abs(difference)
                huber = tf.reduce_sum(tf.where(error < 1., .5 * tf.square(error), error - .5), axis=-1)
                offset_monitoring[head + '_center_offset_error'] = tf.math.divide_no_nan(
                    tf.reduce_sum(tf.norm(difference, axis=-1) * weight), tf.reduce_sum(weight))
                center_offset += on[head] * cfg.center_offset_weight * .5 * tf.math.divide_no_nan(
                    tf.reduce_sum(huber * weight), tf.reduce_sum(weight))
        mask_quality = tf.constant(0., tf.float32)
        quality_monitoring = {}
        if cfg.mask_quality_weight:
            axes = list(range(1, self.ndim + 2))
            for head in ('nucleus', 'cytoplasm'):
                target = self._subsample(tf.cast(y_true_raw[head + '_target'][..., None], tf.float32),
                                         getattr(cfg, head + '_grid'))
                # 'auto' is calibrated after training; quality targets use the default threshold.
                threshold = cfg.mask_threshold(head)
                binary = tf.cast(y_pred[head + '_mask'] > threshold, tf.float32)
                inter = tf.reduce_sum(binary * target, axis=axes)
                union = tf.reduce_sum(tf.maximum(binary, target), axis=axes)
                quality_target = tf.stop_gradient(tf.math.divide_no_nan(inter, union))
                grid_quality_target = quality_target
                if cfg.mask_quality_target == 'component':
                    from cytonucs_mask_quality import component_quality_targets
                    prompt_key = head + '_quality_prompt'
                    if prompt_key not in y_true_raw:
                        raise ValueError('component quality target requires actual prompts from trainer')
                    grid = tuple(getattr(cfg, head + '_grid'))
                    quality_target = tf.numpy_function(
                        lambda p, t, pr, g=grid, th=threshold: component_quality_targets(p, t, pr, g, th),
                        [tf.stop_gradient(y_pred[head + '_mask']), y_true_raw[head + '_target'],
                         y_true_raw[prompt_key]], tf.float32, stateful=False)
                    quality_target.set_shape([None])
                quality_monitoring[head + '_quality_target_mean'] = tf.reduce_mean(quality_target)
                quality_monitoring[head + '_quality_target_shift'] = tf.reduce_mean(
                    tf.abs(quality_target - grid_quality_target))
                error = tf.square(tf.reshape(y_pred[head + '_quality'], [-1]) - quality_target)
                if cfg.mask_quality_ignore_boundary_targets:
                    if crop_cut[head] is not None:
                        # Targets are K-major (k*B + b): repeat the per-image masks K times.
                        eligible, positive = quality_supervision_weights(
                            y_true_raw[head + '_target'], self.ndim,
                            crop_cut=tf.tile(crop_cut[head], [self.n_prompts] + [1] * self.ndim))
                    elif cfg.object_losses_include_crop_cut:
                        eligible, positive = quality_supervision_weights(
                            y_true_raw[head + '_target'], self.ndim, crop_cut=tf.zeros_like(
                                tf.cast(y_true_raw[head + '_target'], tf.uint8)))
                    else:
                        eligible, positive = quality_supervision_weights(
                            y_true_raw[head + '_target'], self.ndim,
                            None if truncating_axes is None else
                            tf.tile(tf.cast(truncating_axes, tf.float32), [self.n_prompts, 1]))
                    if slot_known[head] is not None:
                        eligible *= slot_known[head]
                    quality_monitoring[head + '_quality_supervised_fraction'] = tf.reduce_mean(eligible)
                    quality_monitoring[head + '_quality_complete_positive_fraction'] = tf.reduce_mean(eligible * positive)
                    error_mean = tf.math.divide_no_nan(tf.reduce_sum(error * eligible), tf.reduce_sum(eligible))
                elif slot_known[head] is not None:
                    error_mean = tf.math.divide_no_nan(tf.reduce_sum(error * slot_known[head]), tf.reduce_sum(slot_known[head]))
                else:
                    error_mean = tf.reduce_mean(error)
                mask_quality += on[head] * cfg.mask_quality_weight * .5 * error_mean
                if cfg.mask_quality_rank_weight:
                    # Order over every prompt: a crop-cut object looks like a volume-cut one at
                    # inference, and background prompts (target 0) must rank below real masks.
                    rank, pair_accuracy, pair_count = quality_rank_loss(
                        tf.reshape(y_pred[head + '_quality'], [-1]), quality_target,
                        tf.ones_like(error) if slot_known[head] is None else slot_known[head])
                    quality_monitoring[head + '_quality_rank_loss'] = rank
                    quality_monitoring[head + '_quality_pair_accuracy'] = pair_accuracy
                    quality_monitoring[head + '_quality_rank_pairs'] = pair_count
                    mask_quality += on[head] * cfg.mask_quality_rank_weight * rank
        # === TOTAL ===
        total_loss = (
                on['nucleus'] * cfg.lambda_prob_nucleus * loss_nucleus['bce'] +
                on['nucleus'] * cfg.lambda_dist_nucleus * loss_nucleus['distance'] +
                on['cytoplasm'] * cfg.lambda_prob_cytoplasm * loss_cytoplasm['bce'] +
                on['cytoplasm'] * cfg.lambda_dist_cytoplasm * loss_cytoplasm['distance'] +
                loss_wbr +
                self.lambda_containment * warmup * loss_containment +
                loss_fragmentation +
                loss_background +
                loss_cyto_mask +
                loss_nuc_mask + consistency + prompt_consistency + cell_prompt_separation + cell_z_dice + nucleus_z_dice + mask_quality +
                semantic_foreground + fine_mask + center_offset
        )

        self.last_components = {
            'nucleus_bce': loss_nucleus['bce'], 'nucleus_distance': loss_nucleus['distance'],
            'cytoplasm_bce': loss_cytoplasm['bce'], 'cytoplasm_distance': loss_cytoplasm['distance'],
            'wbr': loss_wbr, 'containment': self.lambda_containment * warmup * loss_containment,
            'nucleus_mask': loss_nuc_mask, 'cytoplasm_mask': loss_cyto_mask,
            'consistency': consistency, 'prompt_consistency': prompt_consistency,
            'mask_quality': mask_quality, 'cell_prompt_separation': cell_prompt_separation,
            'cell_z_dice': cell_z_dice,
            'nucleus_z_dice': nucleus_z_dice,
        }
        if cfg.semantic_foreground_weight:
            self.last_components['semantic_foreground'] = semantic_foreground
            self.last_components.update(semantic_monitoring)
        if cfg.center_offset_weight:
            self.last_components['center_offset'] = center_offset
            self.last_components.update(offset_monitoring)
        self.last_components.update(fine_monitoring)

        for head in ('nucleus', 'cytoplasm'):
            head_loss = loss_nucleus if head == 'nucleus' else loss_cytoplasm
            for metric in ('bce_foreground', 'bce_background', 'prob_foreground_fraction'):
                self.last_components[head + '_' + metric] = head_loss[metric]
            flag = head + '_z_prompt_pair'
            self.last_components[head + '_z_pair_fraction'] = (
                tf.reduce_mean(tf.cast(y_true_raw[flag], tf.float32)) if flag in y_true_raw
                else tf.constant(0., tf.float32))
        self.last_components.update({'nucleus_' + k: v for k, v in loss_nucleus['probability_metrics'].items()})
        self.last_components.update({'cytoplasm_' + k: v for k, v in loss_cytoplasm['probability_metrics'].items()})
        if cfg.object_losses_min_area_from_gate:
            self.last_components['nucleus_object_min_voxels'] = tf.identity(self.object_min_voxels[0])
            self.last_components['cytoplasm_object_min_voxels'] = tf.identity(self.object_min_voxels[1])
        self.last_components.update(quality_monitoring)
        return total_loss

    # ============================================================
    # === СОЗДАНИЕ GT через StarDist C++ ===
    # ============================================================
    def _create_gt_from_instances(self, y_true_raw):
        """
        Вызывает StarDist C++ для создания GT prob/dist maps.
        GRID: wrapper прореживает выход на grid соответствующей головы
        (nucleus/cytoplasm раздельно) -> spatial-форма = вход/grid головы.
        set_shape по spatial делаем динамическим (None), фиксируем только каналы.
        """
        # Быстрый путь: GT уже посчитан генератором в фоновом потоке,
        # параллельно с шагом GPU. py_function ниже — фолбэк (старый генератор,
        # отдельные вызовы из трейнера).
        _keys = ('nucleus_prob', 'nucleus_dist', 'cytoplasm_prob', 'cytoplasm_dist')
        if all(k in y_true_raw for k in _keys):
            return {k: tf.cast(y_true_raw[k], tf.float32) for k in _keys}
        nucleus_instances = y_true_raw['nucleus_instances']
        cytoplasm_instances = y_true_raw['cytoplasm_instances']

        nucleus_prob, nucleus_dist = tf.py_function(
            func=lambda m: self._star_dist_wrapper_nucleus(m),
            inp=[nucleus_instances],
            Tout=[tf.float32, tf.float32]
        )
        cytoplasm_prob, cytoplasm_dist = tf.py_function(
            func=lambda m: self._star_dist_wrapper_cytoplasm(m),
            inp=[cytoplasm_instances],
            Tout=[tf.float32, tf.float32]
        )

        # spatial-форма динамическая (вход/grid), фиксируем только канальную ось
        spatial_none = [None] * self.ndim
        nucleus_prob.set_shape([None, *spatial_none, 1])
        nucleus_dist.set_shape([None, *spatial_none, self.config.n_rays])
        cytoplasm_prob.set_shape([None, *spatial_none, 1])
        cytoplasm_dist.set_shape([None, *spatial_none, self.config.n_rays])

        return {
            'nucleus_prob': nucleus_prob,
            'nucleus_dist': nucleus_dist,
            'cytoplasm_prob': cytoplasm_prob,
            'cytoplasm_dist': cytoplasm_dist
        }
    def _star_dist_wrapper_nucleus(self, instances):
        """Wrapper для nucleus: свой набор лучей и своя анизотропия."""
        expected_max_radius = getattr(self.config, 'median_max_dist_nucleus', None)
        if expected_max_radius is None:
            expected_max_radius = 20.0
        return self._star_dist_wrapper_internal(
            instances, expected_max_radius, self._get_nucleus_grid(),
            getattr(self.config, 'rays_nucleus', self.config.rays),
            getattr(self.config, 'rays_anisotropy_nucleus', self.config.anisotropy),
            normalize_prob=self.config.nucleus_prob_grid_normalize)

    def _star_dist_wrapper_cytoplasm(self, instances):
        """Wrapper для cytoplasm: свой набор лучей и своя анизотропия."""
        expected_max_radius = getattr(self.config, 'median_max_dist_cytoplasm', None)
        if expected_max_radius is None:
            expected_max_radius = 50.0
        return self._star_dist_wrapper_internal(
            instances, expected_max_radius, self._get_cytoplasm_grid(),
            getattr(self.config, 'rays_cytoplasm', self.config.rays),
            getattr(self.config, 'rays_anisotropy_cytoplasm', self.config.anisotropy))

    def _star_dist_wrapper_internal(self, instances, expected_max_radius, grid,
                                    rays, aniso, normalize_prob=False):
        """Фолбэк-путь, когда GT не пришёл из генератора. Логика вынесена в
        compute_stardist_gt, чтобы генератор и лосс считали одно и то же."""
        instances = instances.numpy()
        prob_list, dist_list = [], []
        for b in range(instances.shape[0]):
            p, d = compute_stardist_gt(instances[b], self.config, grid,
                                       rays=rays, aniso=aniso, normalize_prob=normalize_prob)
            prob_list.append(p)
            dist_list.append(d)
        return (np.stack(prob_list, axis=0).astype(np.float32),
                np.stack(dist_list, axis=0).astype(np.float32))

    # ============================================================
    # === STARDIST LOSS ===
    # ============================================================

    def _stardist_loss(self, prob_true, prob_pred, dist_true, dist_pred,
                       name='object', prob_logit=None, instances=None, known=None, quality_ramp=None):

        prob_true_squeeze = tf.squeeze(prob_true, axis=-1)
        prob_pred_squeeze = tf.squeeze(prob_pred, axis=-1)
        foreground_mask = tf.cast(prob_true_squeeze > 0.01, tf.float32)
        background_mask = 1.0 - foreground_mask
        iou_power = float(getattr(self.config, 'prob_iou_aware', 0.0) or 0.0)
        if iou_power and quality_ramp is not None:
            # IoU-aware detection (VarifocalNet, Zhang et al. CVPR 2021): the probability target is
            # the object-centre value times the Polar IoU of the rays predicted at the voxel, so
            # peaks prefer voxels whose polygon fits. Ramped in like WBR: early rays fit nothing.
            prob_true_squeeze = prob_true_squeeze * self._ray_quality(
                dist_true, dist_pred, prob_true_squeeze, iou_power, quality_ramp)


        eps = 1e-7
        if prob_logit is not None:
            z = tf.squeeze(tf.cast(prob_logit, tf.float32), axis=-1)
            bce_raw = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=prob_true_squeeze, logits=z)
        else:
            z = None
            p = tf.clip_by_value(prob_pred_squeeze, eps, 1.0 - eps)
            bce_raw = -(prob_true_squeeze * tf.math.log(p) +
                        (1.0 - prob_true_squeeze) * tf.math.log(1.0 - p))
        if known is None:
            loss_bce = tf.reduce_mean(bce_raw)
        else:
            # Voxel BCE over the voxels whose annotation is known (training.unknown_label).
            known = tf.squeeze(known, axis=-1)
            loss_bce = tf.math.divide_no_nan(tf.reduce_sum(bce_raw * known), tf.reduce_sum(known))
            foreground_mask *= known
            background_mask *= known
        probability_metrics = {}
        object_weight = getattr(self.config, name + '_prob_object_weight', 0.0)
        peak_weight = self.config.prob_object_peak_weight if name in ('nucleus', 'cytoplasm') else 0.0
        if object_weight or peak_weight:
            if instances is None:
                raise ValueError('Object-level probability losses require sampled instance labels')
            object_bce, probability_metrics = object_balanced_probability(
                bce_raw, prob_pred_squeeze, instances)
            probability_metrics['bce_voxel'] = loss_bce
            if object_weight:
                probability_metrics['bce_object'] = object_bce
                probability_metrics['bce_object_weighted'] = object_weight * object_bce
                loss_bce += probability_metrics['bce_object_weighted']
            if peak_weight:
                # Detection needs one confident peak per object, not only a good mean.
                peak = object_peak_probability(prob_pred_squeeze, prob_true_squeeze, instances)
                probability_metrics['bce_object_peak'] = peak
                probability_metrics['bce_object_peak_weighted'] = peak_weight * peak
                loss_bce += probability_metrics['bce_object_peak_weighted']

        num_fg = tf.reduce_sum(foreground_mask) + self.epsilon
        num_bg = tf.reduce_sum(background_mask) + self.epsilon
        loss_bce_fg = tf.stop_gradient(tf.reduce_sum(bce_raw * foreground_mask) / num_fg)
        loss_bce_bg = tf.stop_gradient(tf.reduce_sum(bce_raw * background_mask) / num_bg)

        # ============================================================
        # === 2. Distance loss — канонический StarDist (mask = prob_true) ===
        # Ошибка в СЫРЫХ пикселях, как в StarDist. Нормировка на median_max_dist_*
        # в градиент НЕ идёт: у MAE производная = ±1 независимо от масштаба, поэтому
        # деление на dist_scale не выравнивает головы (они и так сбалансированы),
        # а лишь делит градиент на 15/60 — и требует раздутых lambda для компенсации.
        # dist_scale используем ТОЛЬКО чтобы напечатать относительную ошибку.
        if name == 'nucleus':
            dist_scale = float(getattr(self.config, 'median_max_dist_nucleus', None) or 15.0)
        else:
            dist_scale = float(getattr(self.config, 'median_max_dist_cytoplasm', None) or 60.0)
        dist_scale = tf.cast(max(dist_scale, 1.0), tf.float32)

        dist_mask = tf.cast(prob_true, tf.float32)
        stride = getattr(self.config, 'dist_target_stride', None)
        if stride is not None:
            dist_pred = self._subsample(dist_pred, tuple(int(s) for s in stride))
            dist_mask = self._subsample(dist_mask, tuple(int(s) for s in stride))

        dist_diff = tf.cast(dist_true, tf.float32) - tf.cast(dist_pred, tf.float32)

        # Абсолютная ошибка, РАВНЫЙ вес на каждый луч (d/dPred = 1).
        # Ни log-weighting, ни relative: оба перекашивают баланс лучей.
        # log усиливал длинные -> раздувал их градиент (выброс backbone);
        # relative делил на GT -> длинные лучи в 55x слабее -> max_ray не добирается.
        # Разброс длин лучей (Z~2px vs XY~110px) — это про АМПЛИТУДУ скаляра loss,
        # его нормируем ниже на dist_scale, а НЕ через вес на луч.
        if self.dist_loss == 'huber':
            abs_diff = tf.abs(dist_diff)
            delta = tf.cast(self.huber_delta, tf.float32)
            dist_error_per_ray = tf.where(
                abs_diff <= delta,
                0.5 * tf.square(dist_diff),
                delta * (abs_diff - 0.5 * delta)
            )
        elif self.dist_loss == 'mse':
            dist_error_per_ray = tf.square(dist_diff)
        else:  # 'mae'
            dist_error_per_ray = tf.abs(dist_diff)

        weighted_error = dist_mask * dist_error_per_ray
        loss_distance = tf.reduce_mean(weighted_error) / (
                tf.reduce_mean(dist_mask) + self.epsilon)
        polar_weight = float(getattr(self.config, 'dist_polar_iou_weight', 0.0) or 0.0)
        if polar_weight:
            # Polar IoU (PolarMask, Xie et al. CVPR 2020): all rays of a voxel as one shape,
            # -log(sum_k min(d, d*) / sum_k max(d, d*)), weighted like the ray error. Scaled by the
            # median object size so a weight of 1 is comparable to the ray error in pixels.
            true_rays = tf.cast(dist_true, tf.float32)
            pred_rays = tf.maximum(tf.cast(dist_pred, tf.float32), 0.)
            overlap = tf.reduce_sum(tf.minimum(true_rays, pred_rays), axis=-1)
            union = tf.reduce_sum(tf.maximum(true_rays, pred_rays), axis=-1)
            polar = -tf.math.log((overlap + 1.) / (union + 1.))
            weight = tf.squeeze(dist_mask, axis=-1)
            loss_polar = tf.reduce_sum(weight * polar) / (tf.reduce_sum(weight) + self.epsilon)
            loss_distance = loss_distance + polar_weight * dist_scale * loss_polar

        bg_reg_weight = tf.cast(
            getattr(self.config, 'train_background_reg', 1e-4), tf.float32)
        if bg_reg_weight > 0:
            bg_penalty = tf.reduce_mean((1.0 - dist_mask) * tf.abs(dist_pred))
            loss_distance = loss_distance + bg_reg_weight * bg_penalty

        # относительная ошибка — только для лога, в градиент не идёт
        loss_distance_rel = tf.stop_gradient(loss_distance / dist_scale)

        return {'bce': loss_bce, 'distance': loss_distance, 'distance_rel': loss_distance_rel,
                'bce_foreground': loss_bce_fg, 'bce_background': loss_bce_bg,
                'prob_foreground_fraction': tf.stop_gradient(tf.reduce_mean(foreground_mask)),
                'probability_metrics': probability_metrics}

    def _ray_quality(self, dist_true, dist_pred, like, power, ramp):
        """Polar IoU of predicted and true rays per voxel of `like` (no gradient), as
        1 - ramp + ramp * IoU**power; rays on the dist_target_stride grid are repeated back."""
        stride = getattr(self.config, 'dist_target_stride', None)
        stride = tuple(int(v) for v in stride) if stride is not None else (1,) * self.ndim
        dist_pred = self._subsample(dist_pred, stride)
        true_rays = tf.cast(dist_true, tf.float32)
        pred_rays = tf.maximum(tf.cast(dist_pred, tf.float32), 0.)
        iou = ((tf.reduce_sum(tf.minimum(true_rays, pred_rays), axis=-1) + 1.)
               / (tf.reduce_sum(tf.maximum(true_rays, pred_rays), axis=-1) + 1.))
        for axis, step in enumerate(stride):
            if step > 1:
                iou = tf.repeat(iou, step, axis=axis + 1)
        size = tf.shape(like)
        iou = iou[(slice(None),) + tuple(slice(0, size[a + 1]) for a in range(self.ndim))]
        ramp = tf.cast(ramp, tf.float32)
        return tf.stop_gradient(1. - ramp + ramp * tf.pow(iou, power))

    # ============================================================
    # === WBR LOSS ===
    # ============================================================

    def _wbr_loss(self, nucleus_prob_pred, container_fg):
        """
        Within-Boundary Regularization: nucleus_prob не должен жить вне клетки.

        Нормировка на ЧИСЛО outside-вокселей, а не на массу prob.

        Прежняя форма Σ(p·o)/Σp однородна степени 0 по p: при p -> c·p значение
        не меняется, то есть уронить карту целиком она не могла. Замена
        знаменателя на stop_gradient это свойство ломала — оставался чистый
        снос вниз с множителем 1/Σp, который растёт по мере падения prob.
        Положительная обратная связь: при mean_p=0.001 снос в 80 раз сильнее,
        чем при 0.05. Итог — насыщение сигмоиды и Max(том)=0.000 у ядра при
        целой цитоплазме (WBR действует только на nucleus_prob).

        Здесь знаменатель от предсказания не зависит вообще, градиент на воксель
        равен lam·o_i/N_out и ограничен. Численно лосс теперь ЛЕЖИТ В ДРУГОМ
        МАСШТАБЕ, чем в старых логах (это средняя вероятность снаружи, а не доля
        массы) — сравнивать значения между прогонами нельзя.

        Контейнер = клетка ИЛИ ядро (строится в __call__): ядро вне разметки
        цитоплазмы иначе считалось бы «снаружи» и давилось.
        """
        nucleus_mask = self._match_to_cytoplasm(nucleus_prob_pred)
        outside = tf.stop_gradient(1.0 - tf.cast(container_fg, tf.float32))
        return tf.reduce_sum(nucleus_mask * outside) / (
                tf.reduce_sum(outside) + self.epsilon)

    # ============================================================
    # === CONTAINMENT LOSS ===
    # ============================================================

    def _containment_loss(self, y_true, y_pred, y_true_raw):
        """
        Multi-nuclear containment:
        1. Geometric — ядро внутри GT-клеток (контейнер бинарный, не prob)
        2. Distance field — nuc_dist <= cyto_dist
        3. Cyto support — односторонний hinge к СВОЕМУ GT, а не к единице
        4. Center separation — по флагу w_center_separation (по умолчанию 0)
        GRID: nucleus-карты приводятся к cytoplasm-разрешению.
        """
        if self.lambda_containment == 0:
            return tf.constant(0.0, dtype=tf.float32)

        nuc_prob = self._match_to_cytoplasm(y_pred['nucleus_prob'])
        cyto_prob = y_pred['cytoplasm_prob']
        nuc_dist = self._match_to_cytoplasm(y_pred['nucleus_dist'])
        cyto_dist = y_pred['cytoplasm_dist']

        nuc_prob_gt = self._match_to_cytoplasm(y_true['nucleus_prob'])
        cyto_prob_gt = y_true['cytoplasm_prob']

        # БИНАРНЫЙ GT-контейнер клеток на цито-гриде. Именно он, а не
        # предсказанный cyto_prob (edt-карта), задаёт «внутри клетки».
        cyto_fg_gt = self._grid_subsample_cytoplasm(
            tf.expand_dims(tf.cast(y_true_raw['cytoplasm_instances'] > 0,
                                   tf.float32), axis=-1))
        # БИНАРНЫЙ GT-контейнер: клетка ИЛИ ядро, на цито-гриде.
        cyto_fg_gt = tf.stop_gradient(tf.maximum(
            self._grid_subsample_cytoplasm(
                tf.expand_dims(tf.cast(y_true_raw['cytoplasm_instances'] > 0,
                                       tf.float32), axis=-1)),
            self._match_to_cytoplasm(self._grid_subsample_nucleus(
                tf.expand_dims(tf.cast(y_true_raw['nucleus_instances'] > 0,
                                       tf.float32), axis=-1)))))

        if 'cytoplasm_known' in y_true_raw:
            # Where the cells are not annotated, a nucleus may lie inside one (training.unknown_label).
            cyto_fg_gt = tf.maximum(cyto_fg_gt, 1. - self._grid_subsample_cytoplasm(
                tf.cast(y_true_raw['cytoplasm_known'], tf.float32)[..., None]))

        # 1. GEOMETRIC CONTAINMENT
        loss_geometric = tf.reduce_mean(nuc_prob * (1.0 - cyto_fg_gt))

        # 2. DISTANCE FIELD CONTAINMENT (raw pixels)
        nuc_fg = tf.cast(nuc_prob_gt > 0.5, tf.float32)
        w_distance = float(getattr(self.config, 'w_containment_distance', 0.0))
        if w_distance:
            nr = np.asarray(self.config.rays_nucleus.vertices)
            cr = np.asarray(self.config.rays_cytoplasm.vertices)
            if nr.shape != cr.shape or not np.allclose(nr, cr):
                raise ValueError('Distance containment requires identical ray directions; set w_containment_distance=0')
            shared_fg = nuc_fg * self._grid_subsample_cytoplasm(tf.expand_dims(
                tf.cast(y_true_raw['cytoplasm_instances'] > 0, tf.float32), -1))
            loss_distance_containment = tf.reduce_mean(tf.nn.relu(nuc_dist - cyto_dist) * shared_fg)
        else:
            loss_distance_containment = tf.constant(0.0, tf.float32)

        # 3. CYTOPLASM SUPPORT — односторонний hinge к СОБСТВЕННОМУ GT.
        # Прежнее (1 - cyto_prob) требовало cyto_prob = 1 на всей площади ядра,
        # что прямо противоречит edt-таргету (там 1 только в центре клетки) и
        # сплющивало пик. Штрафуем только НЕДОБОР относительно GT.
        loss_cyto_support = tf.reduce_mean(
            tf.nn.relu(tf.stop_gradient(cyto_prob_gt) - cyto_prob) * nuc_fg)

        # 4. CENTER SEPARATION — выключено по умолчанию: оценка ожидаемого
        # числа ядер через изотропный объём шара (4/3·pi·r^3) при anisotropy
        # [5,1,1] занижает счёт в ~5 раз, из-за чего target_density оказывается
        # ниже реальной и штраф давит настоящие пики.
        w_separation = float(getattr(self.config, 'w_center_separation', 0.0))
        if w_separation > 0:
            if getattr(self.config, 'median_radius_train', None) is not None:
                median_radius = self.config.median_radius_train
            else:
                nuc_dist_gt = self._match_to_cytoplasm(y_true['nucleus_dist'])
                # nuc_fg broadcast'ится по лучам; sum(tile(m)) = n_rays*sum(m),
                # множитель сокращается в отношении, результат тот же.
                num_fg = (tf.reduce_sum(nuc_fg) *
                          tf.cast(tf.shape(nuc_dist)[-1], tf.float32) + self.epsilon)
                median_radius = tf.maximum(
                    tf.reduce_sum(nuc_dist_gt * nuc_fg) / num_fg, 1.0)
            loss_separation = self._compute_center_separation(
                nuc_prob=nuc_prob, cyto_prob_gt=cyto_prob_gt,
                nuc_prob_gt=nuc_prob_gt, median_radius=median_radius)
        else:
            loss_separation = tf.constant(0.0, dtype=tf.float32)

        w_geometric = getattr(self.config, 'w_containment_geometric', 1.0)
        w_distance = getattr(self.config, 'w_containment_distance', 0.0)
        w_cyto_support = getattr(self.config, 'w_cyto_support', 0.1)

        return (w_geometric * loss_geometric +
                w_distance * loss_distance_containment +
                w_separation * loss_separation +
                w_cyto_support * loss_cyto_support)

    # ============================================================
    # === FRAGMENTATION LOSS (Pure TensorFlow) ===
    # ============================================================

    def _fragmentation_loss(self, prob_pred, instances_gt, name='object'):
        """
        Штрафует за множественные prob peaks внутри одного GT объекта.
        Заставляет prob head давать ОДИН максимум на instance, даже multi-nuclear.

        Подход:
        1. Находим локальные максимумы (peaks) через maxpool
        2. Через unsorted_segment_sum считаем число peaks в каждом GT instance
        3. Penalty = ReLU(peaks_per_instance - 1) — превышение над одним peak

        Pure TF, без python loop по labels, без хардкода диапазона.
        Совместимо с @tf.function (динамический max_label через segment ops).

        Args:
            prob_pred: (B, H, W, 1) или (B, D, H, W, 1) предсказанная вероятность
            instances_gt: (B, H, W) или (B, D, H, W) GT instance mask
            name: 'nucleus' или 'cytoplasm'

        Returns:
            loss: скаляр, среднее превышение числа peaks над 1 на instance
        """
        # ============================================================
        # 1. Локальные максимумы (peaks) через maxpool
        # ============================================================
        kernel_size = 7

        if self.ndim == 2:
            ksize = kernel_size
            prob_maxpool = tf.nn.max_pool2d(prob_pred, ksize=ksize, strides=1, padding='SAME')
            prob_avgpool = tf.nn.avg_pool2d(prob_pred, ksize=ksize, strides=1, padding='SAME')
        else:
            ksize = self._aniso_ksize(kernel_size)
            prob_maxpool = tf.nn.max_pool3d(prob_pred, ksize=ksize, strides=1, padding='SAME')
            prob_avgpool = tf.nn.avg_pool3d(prob_pred, ksize=ksize, strides=1, padding='SAME')

        prob_squeeze = tf.squeeze(prob_pred, axis=-1)
        prob_maxpool_squeeze = tf.squeeze(prob_maxpool, axis=-1)
        prob_avgpool_squeeze = tf.squeeze(prob_avgpool, axis=-1)

        # Строгий пик: (а) равен локальному максимуму в окне,
        # (б) уверенно высокий, (в) ВЫРАЖЕННО выше среднего по окну (не шум/плато).
        # Жёсткий порог контраста (peak - avg) убирает ложные пики на необученном
        # plateau prob≈0.5, которые в 3D на большом объёме взрывали лосс.
        peak_thresh = getattr(self.config, 'fragmentation_peak_thresh', 0.5)
        peak_contrast = getattr(self.config, 'fragmentation_peak_contrast', 0.1)
        is_peak = tf.cast(
            tf.logical_and(
                tf.logical_and(
                    (prob_maxpool_squeeze - prob_squeeze) < 1e-4,
                    prob_squeeze > peak_thresh
                ),
                (prob_squeeze - prob_avgpool_squeeze) > peak_contrast
            ),
            tf.float32
        )

        # ============================================================
        # 2. Считаем число peaks в каждом GT instance через segment_sum
        # ============================================================
        instances_int = tf.cast(instances_gt, tf.int32)  # (B, ...spatial)

        # Уникальные ID per-batch могут совпадать между картинками батча,
        # поэтому делаем глобально уникальные label через offset = b * (max_label + 1)
        batch_size = tf.shape(instances_int)[0]
        max_label = tf.reduce_max(instances_int) + 1  # +1 чтобы вместить сам max
        max_label = tf.maximum(max_label, 1)

        # Offset для каждого элемента батча
        # batch_index broadcast по spatial dims
        batch_range = tf.range(batch_size, dtype=tf.int32)
        # reshape (B, 1, 1[, 1]) для broadcast
        reshape_shape = [batch_size] + [1] * self.ndim
        batch_index = tf.reshape(batch_range, reshape_shape)
        batch_index = tf.broadcast_to(batch_index, tf.shape(instances_int))

        # Глобальный label: только для foreground (instances_int > 0)
        # background (label 0) маппим в отдельный segment 0 и игнорируем потом
        is_fg = tf.cast(instances_int > 0, tf.int32)
        global_label = (batch_index * max_label + instances_int) * is_fg
        # foreground получает global_label >= 1 (т.к. instances_int >= 1 для fg),
        # background → 0

        flat_global_label = tf.reshape(global_label, [-1])
        flat_peaks = tf.reshape(is_peak, [-1])

        n_segments = batch_size * max_label
        peaks_per_segment = tf.math.unsorted_segment_sum(
            flat_peaks, flat_global_label, n_segments
        )

        # Сегмент 0 = background → игнорируем
        peaks_per_instance = peaks_per_segment[1:]

        # ============================================================
        # 3. Penalty = превышение над одним peak, только для существующих instance
        # ============================================================
        # Маска существующих instance: те segments в которых есть хоть один пиксель GT
        flat_fg = tf.reshape(tf.cast(is_fg, tf.float32), [-1])
        area_per_segment = tf.math.unsorted_segment_sum(
            flat_fg, flat_global_label, n_segments
        )
        instance_exists = tf.cast(area_per_segment[1:] > 0.5, tf.float32)

        excess_peaks = tf.nn.relu(peaks_per_instance - 1.0) * instance_exists

        # Среднее по реально существующим instance (не по всем n_segments)
        n_instances = tf.reduce_sum(instance_exists) + self.epsilon
        loss = tf.reduce_sum(excess_peaks) / n_instances

        return loss

    # ============================================================
    # === BACKGROUND PENALTY LOSS (уже Pure TensorFlow) ===
    # ============================================================

    def _background_penalty_loss(self, prob_pred, prob_gt, name='object', known=None):
        """
        Усиленный штраф за высокие predictions на фоне.
        Pure TensorFlow - без изменений, эта функция уже корректна.
        """

        prob_pred_squeeze = tf.squeeze(prob_pred, axis=-1)  # (B, H, W)
        prob_gt_squeeze = tf.squeeze(prob_gt, axis=-1)

        # Фон = где GT prob < 0.1 (чистый background)
        background_mask = tf.cast(prob_gt_squeeze < 0.1, tf.float32)
        if known is not None:
            background_mask *= tf.squeeze(known, axis=-1)

        # Predictions на фоне
        pred_on_background = prob_pred_squeeze * background_mask

        # КВАДРАТИЧНЫЙ штраф
        penalty = tf.square(pred_on_background)

        # Дополнительный множитель для очень высоких prob (> 0.5)
        high_confidence_mask = tf.cast(pred_on_background > 0.5, tf.float32)
        extra_penalty = high_confidence_mask * tf.square(pred_on_background) * 2.0

        total_penalty = penalty + extra_penalty

        # Нормализуем по площади фона
        num_bg_pixels = tf.reduce_sum(background_mask) + self.epsilon
        mean_penalty = tf.reduce_sum(total_penalty) / num_bg_pixels

        return mean_penalty


    def _compute_center_separation(self, nuc_prob, cyto_prob_gt, nuc_prob_gt, median_radius):
        """
        Ensure nucleus centers are well-separated within same cytoplasm.
        Использует max-pooling для поиска локальных максимумов.
        """
        # Fixed kernel size (упрощение для TensorFlow eager mode)
        kernel_size_fixed = 9 if self.ndim == 2 else 7

        if self.ndim == 2:
            nuc_maxpool = tf.nn.max_pool2d(
                nuc_prob,
                ksize=kernel_size_fixed,
                strides=1,
                padding='SAME'
            )
        else:
            nuc_maxpool = tf.nn.max_pool3d(
                nuc_prob,
                ksize=self._aniso_ksize(kernel_size_fixed),
                strides=1,
                padding='SAME'
            )
        # Peaks: where prob ≈ maxpool and prob > threshold
        is_peak = tf.cast(
            tf.logical_and(
                tf.abs(nuc_prob - nuc_maxpool) < 0.05,
                nuc_prob > 0.5
            ),
            tf.float32
        )

        cyto_fg_gt = tf.cast(cyto_prob_gt > 0.5, tf.float32)
        nuc_fg_gt = tf.cast(nuc_prob_gt > 0.5, tf.float32)

        # Peaks within cytoplasm
        peaks_in_cyto = is_peak * cyto_fg_gt

        # Векторизованная статистика per-batch
        if self.ndim == 2:
            reduction_axes = [1, 2, 3]
            expected_area_per_nucleus = 3.14159 * tf.square(median_radius)
        else:
            reduction_axes = [1, 2, 3, 4]
            expected_area_per_nucleus = 4.0 / 3.0 * 3.14159 * tf.pow(median_radius, 3.0)

        gt_nucleus_area = tf.reduce_sum(nuc_fg_gt, axis=reduction_axes)
        cyto_area = tf.reduce_sum(cyto_fg_gt, axis=reduction_axes) + self.epsilon

        expected_num_nuclei = gt_nucleus_area / (expected_area_per_nucleus + self.epsilon)
        target_density = expected_num_nuclei / cyto_area

        num_peaks = tf.reduce_sum(peaks_in_cyto, axis=reduction_axes)
        peak_density = num_peaks / cyto_area

        # L1 loss к GT-based target density
        density_error = tf.abs(peak_density - target_density)

        # Tolerance ±20%
        tolerance = target_density * 0.2 + self.epsilon
        penalty = tf.nn.relu(density_error - tolerance)

        return tf.reduce_mean(penalty)

    # ============================================================
    # === CONSISTENCY LOSS ===
    # ============================================================

    def _consistency_loss(self, prob_pred, prob_gt):
        """Variance of probability residual, not variance of the soft EDT target.

        Zero at the correct prediction; does not force a flat foreground map.
        This is a foreground regularizer, not instance/prompt consistency.
        """
        axes = list(range(1, self.ndim + 2))
        mask = tf.cast(prob_gt > 0.01, tf.float32)
        residual = tf.cast(prob_pred, tf.float32) - tf.cast(prob_gt, tf.float32)
        count = tf.reduce_sum(mask, axis=axes, keepdims=True)
        mean = tf.math.divide_no_nan(tf.reduce_sum(residual * mask, axis=axes, keepdims=True), count)
        var = tf.math.divide_no_nan(tf.reduce_sum(tf.square(residual - mean) * mask, axis=axes, keepdims=True), count)
        valid = tf.cast(count > 1, tf.float32)
        return tf.math.divide_no_nan(tf.reduce_sum(var * valid), tf.reduce_sum(valid))

    def _prompt_consistency(self, pred, target):
        """Same GT instance, different prompts, same image (K-major ordering)."""
        ps, ts = tf.split(pred, self.n_prompts, axis=0), tf.split(target, self.n_prompts, axis=0)
        axes = list(range(1, self.ndim + 2))
        total = tf.constant(0., tf.float32)
        count = tf.constant(0., tf.float32)
        for i in range(self.n_prompts):
            for j in range(i + 1, self.n_prompts):
                same = tf.cast((tf.reduce_sum(tf.abs(ts[i]-ts[j]), axis=axes) < .5) &
                               (tf.reduce_sum(ts[i], axis=axes) > .5), tf.float32)
                weight = tf.stop_gradient(tf.maximum(ts[i], tf.maximum(ps[i], ps[j])))
                error = tf.math.divide_no_nan(tf.reduce_sum(tf.square(ps[i]-ps[j])*weight, axis=axes),
                                             tf.reduce_sum(weight, axis=axes))
                total += tf.reduce_sum(error * same)
                count += tf.reduce_sum(same)
        return tf.math.divide_no_nan(total, count)

    def _cell_z_dice(self, pred, target):
        """Equal weight to every occupied Z plane, then to every positive prompt.

        Full-volume Dice can underweight thin ends. No smoothing across Z and
        no supervision to fill GT gaps: empty planes remain covered by CE and
        volume Dice. This term supervises the selected instance only.
        """
        p, t = tf.cast(pred, tf.float32), tf.cast(target, tf.float32)
        axes = [2, 3, 4]  # B,Z,Y,X,C -> B,Z
        area = tf.reduce_sum(t, axis=axes)
        dice = (2 * tf.reduce_sum(p*t, axis=axes) + 1.) / (tf.reduce_sum(p, axis=axes) + area + 1.)
        valid = tf.cast(area > .5, tf.float32)
        n = tf.reduce_sum(valid, axis=1)
        per_prompt = tf.math.divide_no_nan(tf.reduce_sum((1-dice)*valid, axis=1), n)
        return tf.math.divide_no_nan(tf.reduce_sum(per_prompt), tf.reduce_sum(tf.cast(n > 0, tf.float32)))

    def _cell_prompt_separation(self, logits, target):
        """Different GT cells in the same image must prefer their own prompt.

        Compare foreground log-odds across prompts on each GT cell, averaging
        cells equally, not by volume. Ignore empty/repeated/overlapping targets.
        K-major split keeps images separate when image_batch_size > 1.
        Ordinary CE/Dice still supervise absolute foreground/background values.
        """
        z = tf.cast(logits, tf.float32)
        odds = z[..., 1:2] - tf.reduce_logsumexp(tf.gather(z, [0, 2], axis=-1), axis=-1, keepdims=True)
        ps = tf.split(odds, self.n_prompts, axis=0)
        ts = tf.split(tf.cast(target > .5, tf.float32), self.n_prompts, axis=0)
        axes = list(range(1, self.ndim + 2))
        total = tf.constant(0., tf.float32)
        count = tf.constant(0., tf.float32)
        for i in range(self.n_prompts):
            for j in range(i + 1, self.n_prompts):
                ni, nj = (tf.reduce_sum(ts[k], axis=axes) for k in (i, j))
                valid = tf.cast((ni > 0) & (nj > 0) &
                                (tf.reduce_sum(ts[i] * ts[j], axis=axes) == 0), tf.float32)
                li = tf.math.divide_no_nan(tf.reduce_sum(tf.nn.softplus(ps[j]-ps[i]) * ts[i], axis=axes), ni)
                lj = tf.math.divide_no_nan(tf.reduce_sum(tf.nn.softplus(ps[i]-ps[j]) * ts[j], axis=axes), nj)
                total += tf.reduce_sum(.5 * (li + lj) * valid)
                count += tf.reduce_sum(valid)
        return tf.math.divide_no_nan(total, count)

    def _refinement_overlap_loss(self, pred, target):
        """Per-prompt overlap term next to CE: soft Dice (legacy) or Lovász (IoU surrogate)."""
        if self.config.mask_overlap_loss == 'lovasz':
            return lovasz_jaccard_loss(pred, target)
        return self._refinement_dice_loss(pred, target)

    def _refinement_dice_loss(self, pred, target):
        """
        Soft Dice по классу 1 (целевой инстанс), с ОТБРОСОМ пустых сэмплов.

        При target == 0 формула вырождается: dice = 1/(pred_sum+1), и минимум
        достигается при pred_sum = 0, то есть лосс требует занулить маску по
        ВСЕМУ патчу. Градиент на воксель мал, но размазан по всей карте и
        всегда направлен вниз — маска систематически подавляется.

        Пустой таргет возникает штатно: генератор выбирает клетку, у которой
        в патче нет ядра (обрезана границей тайла), тогда nucleus_target нулевой.
        На инференсе выход nucleus-рефайна для таких клеток не используется
        (ветка «клетки без ядер» берёт только cytoplasm_mask), поэтому учить на
        них нечему — сэмпл просто исключается из среднего.
        """
        pred_squeeze = tf.squeeze(pred, axis=-1)
        target_squeeze = tf.cast(tf.squeeze(target, axis=-1), tf.float32)

        spatial_dims = list(range(1, len(pred_squeeze.shape)))

        intersection = tf.reduce_sum(pred_squeeze * target_squeeze, axis=spatial_dims)
        pred_sum = tf.reduce_sum(pred_squeeze, axis=spatial_dims)
        target_sum = tf.reduce_sum(target_squeeze, axis=spatial_dims)

        dice_per_image = (2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)

        valid = tf.cast(target_sum > 0.5, tf.float32)  # (B,)
        n_valid = tf.reduce_sum(valid)
        loss = tf.where(
            n_valid > 0.5,
            1.0 - tf.reduce_sum(dice_per_image * valid) / (n_valid + self.epsilon),
            tf.constant(0.0, tf.float32))
        return loss

    def _refinement_ce_loss(self, logit, target_fg, inst_any, known=None, prompt_known=None):
        """
        Трёхклассовая CE для промпт-маски: 0 фон, 1 целевой инстанс, 2 другие.

        Таргет собирается здесь из того, что уже есть у генератора:
          class 1 = *_target          (выбранный инстанс)
          class 2 = (*_instances > 0) И НЕ class 1
          class 0 = остальное

        mask_ce_class_balance=True усредняет CE внутри каждого присутствующего
        класса и затем между классами; выбранный объект не теряет вес при
        росте числа окружающих инстансов. False сохраняет старое усреднение
        по всем вокселям. Пустые target получают mask_ce_empty_weight.
        known (rows aligned with the targets, 1 = annotation known) leaves out the voxels whose
        class is unknown; prompt_known (one flag per row) the prompts whose target is unknown.
        """
        z = tf.cast(logit, tf.float32)  # (B, ..., 3)
        c1 = tf.cast(tf.squeeze(target_fg, axis=-1) > 0.5, tf.float32)
        c2 = tf.cast(tf.squeeze(inst_any, axis=-1) > 0.5, tf.float32) * (1.0 - c1)
        c0 = 1.0 - c1 - c2
        labels = tf.stack([c0, c1, c2], axis=-1)

        spatial_dims = list(range(1, len(c1.shape)))
        valid = tf.cast(tf.reduce_sum(c1, axis=spatial_dims) > 0.5, tf.float32)  # (B,)
        valid = tf.where(valid > 0, valid, float(getattr(self.config, 'mask_ce_empty_weight', 0.25)))
        if prompt_known is not None:
            valid *= prompt_known
        n_valid = tf.reduce_sum(valid)
        if known is not None:
            labels *= tf.cast(known, tf.float32)                # unknown voxels belong to no class

        ce_per_voxel = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=z)
        if self.config.mask_ce_class_balance:
            # One selected instance must not lose its CE weight as ROI/density grows.
            # Average within each present class, then across present classes per image.
            class_count = tf.reduce_sum(labels, axis=spatial_dims)
            class_loss = tf.math.divide_no_nan(
                tf.reduce_sum(ce_per_voxel[..., None] * labels, axis=spatial_dims), class_count)
            ce_per_image = tf.math.divide_no_nan(tf.reduce_sum(class_loss, axis=-1),
                tf.reduce_sum(tf.cast(class_count > 0, tf.float32), axis=-1))
        elif known is not None:
            weight = tf.reduce_sum(labels, axis=-1)
            ce_per_image = tf.math.divide_no_nan(tf.reduce_sum(ce_per_voxel * weight, axis=spatial_dims),
                                                 tf.reduce_sum(weight, axis=spatial_dims))
        else:
            ce_per_image = tf.reduce_mean(ce_per_voxel, axis=spatial_dims)
        return tf.where(
            n_valid > 0,
            tf.reduce_sum(ce_per_image * valid) / (n_valid + self.epsilon),
            tf.constant(0.0, tf.float32))



def compute_stardist_gt(instances, config, grid, rays=None, aniso=None, normalize_prob=False):
    """
    GT prob/dist для ОДНОГО сэмпла. Канон StarDist 3D (model3d.py:86,91):
        prob = edt_prob(lbl, anisotropy=anisotropy)[ss_grid]
        dist = star_dist3D(lbl, rays, grid=grid)

    rays и aniso передаются ЯВНО: наборы лучей у ядра и цитоплазмы разные
    (config.rays — это набор ЯДРА, для обратной совместимости), а anisotropy в
    каноне — аспект объекта, тоже свой на голову. Оба параметра опциональны:
    None -> config.rays / config.anisotropy, то есть прежнее поведение.
    """
    from stardist.geometry import star_dist, star_dist3D
    from stardist.utils import edt_prob

    ndim = int(config.ndim)
    grid = tuple(int(g) for g in grid)
    if len(grid) != ndim or any(g < 1 for g in grid):
        grid = (1,) * ndim

    if rays is None:
        rays = config.rays
    if aniso is None:
        aniso = getattr(config, 'anisotropy', (1.0,) * ndim)
    aniso = tuple(float(v) for v in aniso)
    if len(aniso) != ndim:
        aniso = (1.0,) * ndim

    patch = np.asarray(config.train_patch_size, dtype=np.float64)
    max_clip = float(np.sqrt(np.sum(patch ** 2)))

    # Labels as annotated: no fill_label_holes (a fix for holes in poorly annotated masks that cost a
    # third of the batch time under the prefetch threads' GIL).
    lbl = np.asarray(instances).astype(np.int32)

    # Ray targets dominate the CPU cost of a batch; dist_target_stride computes them on every
    # stride-th node of the head grid (the loss reads the prediction at the same nodes).
    stride = getattr(config, 'dist_target_stride', None) or (1,) * ndim
    dist_grid = tuple(g * int(s) for g, s in zip(grid, stride))
    if ndim == 3:
        dist = star_dist3D(lbl, rays, mode='cpp', grid=dist_grid)
    else:
        dist = star_dist(lbl, config.n_rays, mode='cpp', grid=dist_grid)
    dist = np.clip(dist, 1e-3, max_clip).astype(np.float16)

    ss = tuple(slice(None, None, g) for g in grid)
    prob = edt_prob(lbl, anisotropy=aniso)[ss][..., np.newaxis].astype(np.float32)
    if normalize_prob:
        # Full-resolution EDT maxima may fall between grid points. Normalize
        # per sampled instance, retaining background and relative EDT shape.
        from scipy.ndimage import maximum
        sampled = lbl[ss]
        ids, inverse = np.unique(sampled, return_inverse=True)
        peaks = np.asarray(maximum(prob[..., 0], labels=sampled, index=ids))
        peaks[ids == 0] = 1.
        scales = peaks[inverse].reshape(sampled.shape)
        prob[..., 0] = np.divide(prob[..., 0], scales, out=np.zeros_like(scales, dtype=np.float32), where=scales > 0)

    return prob, dist
# ============================================================
# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (numpy, для monitoring) ===
# ============================================================

def compute_nucleus_cell_assignments(nucleus_instances, cytoplasm_instances, max_distance=100.0):
    """Smart overlap-based assignment для смешанных контуров"""
    from skimage.measure import regionprops
    from scipy.spatial.distance import cdist
    from collections import defaultdict

    if nucleus_instances.shape != cytoplasm_instances.shape:
        min_shape = tuple(min(n, c) for n, c in zip(nucleus_instances.shape, cytoplasm_instances.shape))
        nucleus_instances = nucleus_instances[tuple(slice(0, s) for s in min_shape)]
        cytoplasm_instances = cytoplasm_instances[tuple(slice(0, s) for s in min_shape)]

    nuc_props = {p.label: p for p in regionprops(nucleus_instances)}
    cyto_props = {p.label: p for p in regionprops(cytoplasm_instances)}

    assignments = {}
    unassigned = []

    # СТРАТЕГИЯ 1: Maximum overlap (bbox-прунинг: пересечения только внутри
    # общего bbox пары, вместо полнообъёмного logical_and на каждую пару)
    ndim = nucleus_instances.ndim

    def _mins_maxs(prop):
        bb = prop.bbox
        return np.array(bb[:ndim]), np.array(bb[ndim:])

    cyto_bb = {cid: _mins_maxs(p) for cid, p in cyto_props.items()}

    for nuc_id, nuc_prop in nuc_props.items():
        n_lo, n_hi = _mins_maxs(nuc_prop)
        nuc_area = nuc_prop.area

        overlaps = {}
        for cyto_id, (c_lo, c_hi) in cyto_bb.items():
            lo = np.maximum(n_lo, c_lo)
            hi = np.minimum(n_hi, c_hi)
            if np.any(hi <= lo):
                continue  # bbox не пересекаются → overlap = 0
            sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(ndim))
            ov = int(np.logical_and(
                nucleus_instances[sl] == nuc_id,
                cytoplasm_instances[sl] == cyto_id
            ).sum())
            if ov > 0:
                overlaps[cyto_id] = ov

        if len(overlaps) > 0:
            best_cell = max(overlaps, key=overlaps.get)
            overlap_fraction = overlaps[best_cell] / max(nuc_area, 1)
            if overlap_fraction > 0.01:
                assignments[int(nuc_id)] = int(best_cell)
                continue

    # СТРАТЕГИЯ 2: Proximity
    unassigned_nuclei = [nid for nid in nuc_props.keys() if int(nid) not in assignments]

    if len(unassigned_nuclei) > 0 and len(cyto_props) > 0:
        cyto_centers = np.array([p.centroid for p in cyto_props.values()])
        cyto_ids = list(cyto_props.keys())

        for nuc_id in unassigned_nuclei:
            nuc_center = nuc_props[nuc_id].centroid
            dists = cdist([nuc_center], cyto_centers)[0]
            nearest_idx = np.argmin(dists)

            if dists[nearest_idx] <= max_distance:
                assignments[int(nuc_id)] = int(cyto_ids[nearest_idx])
            else:
                unassigned.append(int(nuc_id))
    else:
        unassigned.extend([int(nid) for nid in unassigned_nuclei])

    # Статистика
    print(f"  Assigned: {len(assignments)}/{len(nuc_props)} (overlap+proximity)")

    cell_to_nuclei = defaultdict(list)
    for nuc_id, cell_id in assignments.items():
        cell_to_nuclei[cell_id].append(nuc_id)

    multi_nuclear = {cid: len(nids) for cid, nids in cell_to_nuclei.items() if len(nids) > 1}
    if len(multi_nuclear) > 0:
        print(f"    🔬 Multi-nuclear: {len(multi_nuclear)} cells (max {max(multi_nuclear.values())} nuclei/cell)")

    stats = {'assigned': len(assignments), 'unassigned': len(unassigned)}

    assert len(assignments) <= len(nuc_props), f"Assignments overflow!"

    return assignments, unassigned, stats


def quality_supervision_weights(target, ndim, truncating_axes=None, crop_cut=None):
    """Exclude potentially truncated positive masks from quality regression only.

    A target touching a crop face that cut its object (the object continues in
    the source volume) is not evidence of a complete instance. truncating_axes
    (rows aligned with target, one flag per spatial axis) marks such faces;
    faces equal to volume faces truncate image and GT identically at inference
    and stay supervised. None keeps the conservative every-face rule. When no
    complete positive exists, skip this head's quality update, including
    negatives, to avoid a negative-only calibration bias. Mask CE/Dice stay active.
    crop_cut (rows aligned with target) marks voxels of objects the crop cut; when
    given it replaces the face rule: a target overlapping it is not complete.
    """
    target = tf.cast(target > .5, tf.bool)
    positive = tf.reduce_any(target, axis=list(range(1, ndim+1)))
    if crop_cut is not None:
        boundary = tf.reduce_any(target & (tf.cast(crop_cut, tf.int32) > 0), axis=list(range(1, ndim+1)))
        complete = positive & ~boundary
        eligible = ~boundary & tf.reduce_any(complete)
        return tf.cast(eligible, tf.float32), tf.cast(positive, tf.float32)
    boundary = tf.zeros_like(positive)
    for axis in range(1, ndim+1):
        for face in (0, -1):
            edge = tf.gather(target, face if face == 0 else tf.shape(target)[axis]-1, axis=axis)
            touches = tf.reduce_any(edge, axis=list(range(1, ndim)))
            if truncating_axes is not None:
                touches &= tf.cast(truncating_axes, tf.float32)[:, axis-1] > .5
            boundary = boundary | touches
    complete = positive & ~boundary
    eligible = ~boundary & tf.reduce_any(complete)
    return tf.cast(eligible, tf.float32), tf.cast(positive, tf.float32)


def without_truncated_objects(full_labels, grid_labels, truncating_axes, ndim):
    """Drop objects cut by an artificial crop face from sampled labels.

    Faces are checked at full resolution (the grid may skip the last row);
    volume faces (flag 0) keep their objects: inference sees them identically.
    """
    def per_image(args):
        full, sampled, flags = args
        face = tf.zeros(tf.shape(full), tf.bool)
        for axis in range(ndim):
            n = tf.shape(full)[axis]
            index = tf.reshape(tf.range(n), [-1 if a == axis else 1 for a in range(ndim)])
            face |= ((index == 0) | (index == n - 1)) & (flags[axis] > .5)
        cut = tf.unique(tf.boolean_mask(full, face & (full > 0)))[0]
        drop = tf.reduce_any(tf.equal(sampled[..., None], cut), axis=-1)
        return tf.where(drop, tf.zeros_like(sampled), sampled)
    return tf.map_fn(per_image, (tf.cast(full_labels, tf.int32), tf.cast(grid_labels, tf.int32),
                                 tf.cast(truncating_axes, tf.float32)),
                     fn_output_signature=tf.int32)


def lovasz_jaccard_loss(probability, target):
    """Lovász-Softmax loss for the target class (Berman et al., 2018), per prompt.

    A convex surrogate of 1 - IoU of the thresholded mask: the sorted absolute errors
    |target - p| are weighted by the Lovász gradient of the Jaccard index. For a binary
    prediction it equals 1 - IoU exactly. Like the Dice term, prompts with an empty
    target are excluded (their CE term still applies); mean over the remaining prompts.
    """
    batch = tf.shape(probability)[0]
    p = tf.reshape(tf.cast(probability, tf.float32), [batch, -1])
    t = tf.reshape(tf.cast(target > .5, tf.float32), [batch, -1])
    # All prompts in one sort (rows are independent): a loop over prompts costs more than the sort itself.
    errors, order = tf.math.top_k(tf.abs(t - p), k=tf.shape(p)[1], sorted=True)
    fg = tf.gather(t, order, batch_dims=1)
    total = tf.reduce_sum(fg, axis=1, keepdims=True)
    intersection = total - tf.cumsum(fg, axis=1)
    union = total + tf.cumsum(1. - fg, axis=1)
    jaccard = 1. - intersection / tf.maximum(union, 1e-6)
    grad = tf.concat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], 1)
    losses = tf.reduce_sum(errors * tf.stop_gradient(grad), axis=1)
    valid = tf.cast(tf.reduce_sum(t, axis=1) > .5, tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(losses * valid), tf.reduce_sum(valid))


def batch_segments(labels):
    """Objects of a batch of label images as one set of segments, without a loop over the images.

    IDs are local to each image and may be sparse. Returns (segment index of every voxel, flattened image by image;
    number of segments; True for the segments of objects, i.e. label > 0)."""
    labels = tf.cast(labels, tf.int64)
    batch = tf.shape(labels, out_type=tf.int64)[0]
    flat = tf.reshape(labels, [batch, -1])
    low = tf.minimum(tf.reduce_min(flat), 0)
    width = tf.maximum(tf.reduce_max(flat), 0) - low + 1
    ids, segment = tf.unique(tf.reshape(flat - low + width * tf.range(batch)[:, None], [-1]))
    return segment, tf.size(ids), ids % width + low > 0


def without_small_objects(full_labels, grid_labels, min_voxels):
    """Drop from sampled labels the objects with fewer than min_voxels voxels in the crop.

    Counts are taken at full resolution, the unit of the calibrated inference size gate.
    """
    full = tf.cast(full_labels, tf.int32)
    sampled = tf.cast(grid_labels, tf.int32)
    batch = tf.shape(full)[0]
    full = tf.reshape(full, [batch, -1])
    flat = tf.reshape(sampled, [batch, -1])
    in_full = tf.concat([tf.ones_like(full, tf.float32), tf.zeros_like(flat, tf.float32)], axis=1)
    segment, n, _ = batch_segments(tf.concat([full, flat], axis=1))
    counts = tf.math.unsorted_segment_sum(tf.reshape(in_full, [-1]), segment, n)
    count = tf.reshape(tf.gather(counts, segment), tf.shape(in_full))[:, tf.shape(full)[1]:]
    # count 0: a sampled label absent at full resolution is kept, as before.
    drop = tf.reshape((flat > 0) & (count > 0) & (count < min_voxels), tf.shape(sampled))
    return tf.where(drop, tf.zeros_like(sampled), sampled)


def object_peak_probability(probability, target, instances):
    """BCE of each object's maximum probability against its maximum target, mean over objects.

    Proposals are local maxima: an object is detected only if its peak clears the
    threshold, while voxel/object-mean BCE can stay low with a flat, subthreshold
    interior. The target is the object's own maximum soft EDT value (1 with grid
    normalization), so the soft-target optimum is unchanged. Gradient reaches only
    the arg-max voxel of each object; background keeps the voxel BCE.
    """
    eps = 1e-7
    segment, n, keep = batch_segments(instances)
    peak = tf.math.unsorted_segment_max(tf.reshape(tf.cast(probability, tf.float32), [-1]), segment, n)
    wanted = tf.stop_gradient(tf.math.unsorted_segment_max(tf.reshape(tf.cast(target, tf.float32), [-1]), segment, n))
    peak = tf.clip_by_value(tf.boolean_mask(peak, keep), eps, 1. - eps)
    wanted = tf.boolean_mask(wanted, keep)
    error = -(wanted * tf.math.log(peak) + (1. - wanted) * tf.math.log(1. - peak))
    return tf.math.divide_no_nan(tf.reduce_sum(error), tf.cast(tf.size(error), tf.float32))


def quality_rank_loss(predicted, target, eligible):
    """Order-consistency of predicted mask quality over all supervised prompts of a batch.

    Object assembly commits masks by predicted quality, so the order matters, not only
    the absolute value. For every pair with target_i > target_j the predicted gap should
    be at least the true gap: hinge max(0, (t_i - t_j) - (q_i - q_j)). The margin is
    the measured quality difference itself, so there is no scale constant. Returns the
    mean over ordered pairs, the fraction of such pairs already in the right order and
    the number of pairs (0 pairs: loss and fraction are 0).
    """
    q = tf.reshape(tf.cast(predicted, tf.float32), [-1])
    t = tf.stop_gradient(tf.reshape(tf.cast(target, tf.float32), [-1]))
    w = tf.reshape(tf.cast(eligible, tf.float32), [-1])
    gap_t = t[:, None] - t[None, :]
    gap_q = q[:, None] - q[None, :]
    pairs = tf.cast(gap_t > 0, tf.float32) * w[:, None] * w[None, :]
    count = tf.reduce_sum(pairs)
    loss = tf.math.divide_no_nan(tf.reduce_sum(tf.nn.relu(gap_t - gap_q) * pairs), count)
    accuracy = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(gap_q > 0, tf.float32) * pairs), count)
    return loss, tf.stop_gradient(accuracy), tf.stop_gradient(count)


def object_balanced_probability(bce, probability, instances):
    """Mean BCE per sampled GT object, then across objects (not across voxels).

    IDs are local to each image. Sparse IDs never allocate max(label) segments.
    Background remains supervised by the original voxel BCE. Diagnostics are
    GT-interior maxima, not local-peak recall or instance matching.
    """
    tf.debugging.assert_equal(tf.shape(bce), tf.shape(instances))
    tf.debugging.assert_equal(tf.shape(probability), tf.shape(instances))
    segment, n, foreground = batch_segments(instances)
    error = tf.reshape(bce, [-1])
    total = tf.math.unsorted_segment_sum(error, segment, n)
    count = tf.math.unsorted_segment_sum(tf.ones_like(error), segment, n)
    object_means = tf.boolean_mask(tf.math.divide_no_nan(total, count), foreground)
    maxima = tf.boolean_mask(tf.math.unsorted_segment_max(
        tf.reshape(tf.stop_gradient(probability), [-1]), segment, n), foreground)
    totals = tf.stack([tf.reduce_sum(object_means), tf.cast(tf.size(object_means), tf.float32),
                       tf.reduce_sum(maxima)] + [tf.reduce_sum(tf.cast(maxima > threshold, tf.float32))
                                                for threshold in (.2, .3, .4, .5)])
    means = tf.math.divide_no_nan(totals, totals[1])
    monitoring = {'prob_sampled_object_count': totals[1], 'prob_gt_max_mean': means[2]}
    monitoring.update({'prob_gt_max_above_' + key: means[i+3]
                       for i, key in enumerate(('020', '030', '040', '050'))})
    return means[0], {k: tf.stop_gradient(v) for k, v in monitoring.items()}


def center_offset_targets(instances, target, ndim):
    """Offset from every voxel of an object to the object's centre, in grid nodes ((B, *S, ndim); 0 on background).

    The centre is the mean position of the voxels where the object's soft-EDT prob target is largest, so it lies
    inside the object also for a non-convex one; it is taken on the object's voxels in the crop, which is what the
    network sees. IDs are local to each image."""
    def per_image(args):
        labels, values = args
        shape = tf.shape(labels)
        flat = tf.reshape(labels, [-1])
        ids, segment = tf.unique(flat)
        n = tf.size(ids)
        values = tf.reshape(values, [-1])
        top = tf.math.unsorted_segment_max(values, segment, n)
        at_top = tf.cast(values >= tf.gather(top, segment) - 1e-6, tf.float32)
        axes = tf.meshgrid(*[tf.range(shape[d]) for d in range(ndim)], indexing='ij')
        coords = tf.cast(tf.stack([tf.reshape(a, [-1]) for a in axes], 1), tf.float32)
        centre = tf.math.divide_no_nan(tf.math.unsorted_segment_sum(coords * at_top[:, None], segment, n),
                                       tf.math.unsorted_segment_sum(at_top, segment, n)[:, None])
        offset = (tf.gather(centre, segment) - coords) * tf.cast(flat > 0, tf.float32)[:, None]
        return tf.reshape(offset, tf.concat([shape, [ndim]], 0))
    return tf.map_fn(per_image, (tf.cast(instances, tf.int32), tf.cast(target, tf.float32)),
                     fn_output_signature=tf.TensorSpec([None] * ndim + [ndim], tf.float32))
