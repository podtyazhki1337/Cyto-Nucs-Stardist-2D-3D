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

    def _get_wbr_weight(self):
        """Постепенное включение WBR loss"""
        if self.current_epoch < self.wbr_warmup_epochs:
            # Linear warmup: 0 → lambda_wbr за wbr_warmup_epochs
            warmup_factor = self.current_epoch / self.wbr_warmup_epochs
            return self.lambda_wbr * warmup_factor
        return self.lambda_wbr

    def __call__(self, y_true_raw, y_pred, current_epoch=None):
        cfg = self.config

        current_epoch_display = current_epoch  # Для печати (1-based)
        current_epoch = current_epoch - 1  # Для WBR warmup (0-based)

        y_true = self._create_gt_from_instances(y_true_raw)

        # === 1. StarDist Loss ===
        loss_nucleus = self._stardist_loss(
            y_true['nucleus_prob'], y_pred['nucleus_prob'],
            y_true['nucleus_dist'], y_pred['nucleus_dist'],
            name='nucleus'
        )

        loss_cytoplasm = self._stardist_loss(
            y_true['cytoplasm_prob'], y_pred['cytoplasm_prob'],
            y_true['cytoplasm_dist'], y_pred['cytoplasm_dist'],
            name='cytoplasm'
        )

        # === 2. WBR ===
        wbr_raw = self._wbr_loss(y_pred['nucleus_prob'], y_pred['cytoplasm_prob'])
        warmup_epochs = getattr(cfg, 'wbr_warmup_epochs', 5)
        current_epoch_float = tf.cast(current_epoch, tf.float32)
        warmup_epochs_float = tf.cast(warmup_epochs, tf.float32)
        if current_epoch < warmup_epochs:
            wbr_factor = current_epoch_float / tf.maximum(warmup_epochs_float, 1.0)
        else:
            wbr_factor = 1.0

        loss_wbr = self.lambda_wbr * wbr_factor * wbr_raw

        # === 3. Containment ===
        loss_containment = self._containment_loss(y_true, y_pred, y_true_raw)

        # === 4. Fragmentation Loss (среднее по nucleus + cytoplasm) ===
        loss_fragmentation = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_fragmentation > 0:
            loss_frag_nuc = self._fragmentation_loss(
                y_pred['nucleus_prob'],
                y_true_raw['nucleus_instances'],
                name='nucleus'
            )
            loss_frag_cyto = self._fragmentation_loss(
                y_pred['cytoplasm_prob'],
                y_true_raw['cytoplasm_instances'],
                name='cytoplasm'
            )
            loss_fragmentation = self.lambda_fragmentation * (loss_frag_nuc + loss_frag_cyto) / 2.0

        # === 5. Background Penalty (среднее по nucleus + cytoplasm) ===
        loss_background = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_background_penalty > 0:
            loss_bg_nuc = self._background_penalty_loss(
                y_pred['nucleus_prob'],
                y_true['nucleus_prob'],
                name='nucleus'
            )
            loss_bg_cyto = self._background_penalty_loss(
                y_pred['cytoplasm_prob'],
                y_true['cytoplasm_prob'],
                name='cytoplasm'
            )
            loss_background = self.lambda_background_penalty * (loss_bg_nuc + loss_bg_cyto) / 2.0

        # === TOTAL LOSS ===
        total_loss = (
                cfg.lambda_prob_nucleus * loss_nucleus['bce'] +
                cfg.lambda_dist_nucleus * loss_nucleus['distance'] +
                cfg.lambda_prob_cytoplasm * loss_cytoplasm['bce'] +
                cfg.lambda_dist_cytoplasm * loss_cytoplasm['distance'] +
                loss_wbr +
                self.lambda_containment * loss_containment +
                loss_fragmentation +
                loss_background
        )

        # Debug

        def print_debug(epoch_val, nuc_bce, nuc_dist, cyto_bce, cyto_dist,
                        wbr_val, wbr_f, cont_val, frag_val, bg_val, tot_val):
            # Конвертируем в numpy внутри py_function
            if self._loss_call_count % 5 == 0:
                print(f"\n=== LOSS (epoch {int(epoch_val)}, global_step {self._loss_call_count}) ===")
                print(f"  Nucleus BCE: {float(nuc_bce):.4f} | Dist: {float(nuc_dist):.2f}")
                print(f"  Cytoplasm BCE: {float(cyto_bce):.4f} | Dist: {float(cyto_dist):.2f}")
                print(f"  WBR: {float(wbr_val):.4f} (factor {float(wbr_f):.2f})")
                print(f"  Containment: {float(cont_val):.4f}")
                print(f"  Fragmentation: {float(frag_val):.4f}")
                print(f"  Background: {float(bg_val):.4f}")
                print(f"  TOTAL: {float(tot_val):.4f}")

            self._loss_call_count += 1
            return 0

        # Передаём все значения как аргументы
        tf.py_function(
            print_debug,
            [current_epoch_display, loss_nucleus['bce'], loss_nucleus['distance'],
             loss_cytoplasm['bce'], loss_cytoplasm['distance'],
             loss_wbr, wbr_factor, loss_containment,
             loss_fragmentation, loss_background, total_loss],
            [tf.int32]
        )

        return total_loss

    # ============================================================
    # === СОЗДАНИЕ GT через StarDist C++ ===
    # ============================================================

    def _create_gt_from_instances(self, y_true_raw):
        """
        Вызывает StarDist C++ для создания GT prob/dist maps.
        ИСПРАВЛЕНО: Раздельная нормализация для nucleus и cytoplasm.
        """
        nucleus_instances = y_true_raw['nucleus_instances']
        cytoplasm_instances = y_true_raw['cytoplasm_instances']

        # === ВЫЗОВ StarDist C++ через py_function ===
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

        # Set shapes
        batch_size = tf.shape(nucleus_instances)[0]

        if self.ndim == 3:
            spatial_shape = nucleus_instances.shape[1:4]
        else:
            spatial_shape = nucleus_instances.shape[1:3]

        nucleus_prob.set_shape([None, *spatial_shape, 1])
        nucleus_dist.set_shape([None, *spatial_shape, self.config.n_rays])
        cytoplasm_prob.set_shape([None, *spatial_shape, 1])
        cytoplasm_dist.set_shape([None, *spatial_shape, self.config.n_rays])

        return {
            'nucleus_prob': nucleus_prob,
            'nucleus_dist': nucleus_dist,
            'cytoplasm_prob': cytoplasm_prob,
            'cytoplasm_dist': cytoplasm_dist
        }

    def _star_dist_wrapper_nucleus(self, instances):
        """Wrapper для nucleus с нормализацией"""
        expected_max_radius = getattr(self.config, 'median_max_dist_nucleus', None)
        if expected_max_radius is None:
            # Auto-detect from data
            expected_max_radius = 20.0  # fallback
        return self._star_dist_wrapper_internal(instances, expected_max_radius)

    def _star_dist_wrapper_cytoplasm(self, instances):
        """Wrapper для cytoplasm с нормализацией"""
        expected_max_radius = getattr(self.config, 'median_max_dist_cytoplasm', None)
        if expected_max_radius is None:
            # Auto-detect from data
            expected_max_radius = 50.0  # fallback
        return self._star_dist_wrapper_internal(instances, expected_max_radius)

    def _star_dist_wrapper_internal(self, instances, expected_max_radius):
        """
        Обёртка для вызова StarDist C++ (КАК В train.py!).
        ИСПРАВЛЕНО: Возвращаем RAW PIXEL DISTANCES (без нормализации)

        Args:
            instances: instance mask (H, W) or (D, H, W)
            expected_max_radius: НЕ ИСПОЛЬЗУЕТСЯ (оставлен для совместимости)

        Returns:
            prob_map: probability map [0, 1]
            dist_map: RAW distance map в ПИКСЕЛЯХ
        """
        from stardist.geometry import star_dist, star_dist3D

        instances = instances.numpy()
        batch_size = instances.shape[0]

        prob_list = []
        dist_list = []

        for b in range(batch_size):
            mask = instances[b]

            # Fill holes (как в train.py!)
            mask_filled = np.zeros_like(mask, dtype=np.int32)
            for label_id in np.unique(mask):
                if label_id == 0:
                    continue
                mask_i = (mask == label_id)
                mask_i_filled = binary_fill_holes(mask_i)
                mask_filled[mask_i_filled] = label_id

            # === ВЫЗОВ StarDist C++ (как в train.py!) ===
            if self.ndim == 3:
                dist_map_raw = star_dist3D(mask_filled, self.config.rays, mode='cpp')
            else:
                dist_map_raw = star_dist(mask_filled, self.config.n_rays, mode='cpp')

            # === КРИТИЧНО: БЕЗ НОРМАЛИЗАЦИИ! ===
            # StarDist работает с RAW PIXEL DISTANCES
            # Model будет учиться предсказывать абсолютные расстояния

            # Только clip для безопасности (убираем inf/nan)
            dist_map_clean = np.clip(dist_map_raw, 0, 500.0)

            # Prob map через EDT + tanh (как в оригинале)
            prob_map = np.zeros(mask.shape, dtype=np.float32)
            for label_id in np.unique(mask_filled):
                if label_id == 0:
                    continue
                m = (mask_filled == label_id)
                edt = distance_transform_edt(m)
                if edt.max() > 0:
                    prob_map[m] = np.maximum(prob_map[m], (edt[m] / edt.max()))

            prob_list.append(prob_map[..., np.newaxis])
            dist_list.append(dist_map_clean)  # ← RAW PIXELS!

        prob_batch = np.stack(prob_list, axis=0)
        dist_batch = np.stack(dist_list, axis=0)

        return prob_batch.astype(np.float32), dist_batch.astype(np.float32)

    # ============================================================
    # === STARDIST LOSS ===
    # ============================================================

    def _stardist_loss(self, prob_true, prob_pred, dist_true, dist_pred, name='object'):

        # ============================================================
        # === 1. BALANCED BCE ===
        # ============================================================

        prob_true_squeeze = tf.squeeze(prob_true, axis=-1)
        prob_pred_squeeze = tf.squeeze(prob_pred, axis=-1)

        foreground_mask = tf.cast(prob_true_squeeze > 0.01, tf.float32)
        background_mask = 1.0 - foreground_mask

        epsilon = 1e-7
        prob_pred_clipped = tf.clip_by_value(prob_pred_squeeze, epsilon, 1.0 - epsilon)

        bce_raw = -(
                prob_true_squeeze * tf.math.log(prob_pred_clipped) +
                (1.0 - prob_true_squeeze) * tf.math.log(1.0 - prob_pred_clipped)
        )

        fg_weight = 1.0
        bg_weight = 1.0

        weight_map = foreground_mask * fg_weight + background_mask * bg_weight
        bce_weighted = bce_raw * weight_map

        num_fg = tf.reduce_sum(foreground_mask) + self.epsilon
        num_bg = tf.reduce_sum(background_mask) + self.epsilon

        loss_bce_fg = tf.reduce_sum(bce_weighted * foreground_mask) / num_fg
        loss_bce_bg = tf.reduce_sum(bce_weighted * background_mask) / num_bg
        loss_bce = (loss_bce_fg + loss_bce_bg) / 2.0

        # ============================================================
        # === 2. Distance loss ===
        # ============================================================

        n_rays = tf.shape(dist_true)[-1]

        # Binary spatial mask для ОБОИХ (стандартный StarDist)
        spatial_weight = tf.cast(
            tf.reduce_max(dist_true, axis=-1, keepdims=True) > 0,
            tf.float32
        )

        if name == 'nucleus':
            scale = getattr(self.config, 'dist_nucleus_norm', 20.0)
        else:
            scale = getattr(self.config, 'dist_cytoplasm_norm', 50.0)

        scale = tf.cast(scale, tf.float32)
        dist_true_norm = dist_true / (scale + self.epsilon)
        dist_pred_norm = dist_pred / (scale + self.epsilon)

        # Per-ray weighting: только для cytoplasm (форма важна)
        # Nucleus почти круглые → uniform weights
        if name == 'nucleus':
            ray_weight = tf.ones_like(dist_true)
        else:
            max_dist_per_pixel = tf.reduce_max(dist_true, axis=-1, keepdims=True)
            reg_weight = 1e-4
            normalizer = max_dist_per_pixel + reg_weight * tf.reduce_mean(dist_true, axis=-1, keepdims=True)
            ray_weight = dist_true / (normalizer + self.epsilon)

        # MAE / Huber / MSE
        if self.dist_loss == 'mse':
            dist_error = tf.square(dist_true_norm - dist_pred_norm)
        elif self.dist_loss == 'huber':
            err = dist_true_norm - dist_pred_norm
            abs_err = tf.abs(err)
            delta = tf.cast(self.huber_delta, tf.float32)
            quadratic = tf.minimum(abs_err, delta)
            linear = abs_err - quadratic
            dist_error = 0.5 * tf.square(quadratic) + delta * linear
        else:
            dist_error = tf.abs(dist_true_norm - dist_pred_norm)

        if self.ndim == 2:
            spatial_weight_exp = tf.tile(spatial_weight, [1, 1, 1, n_rays])
        else:
            spatial_weight_exp = tf.tile(spatial_weight, [1, 1, 1, 1, n_rays])

        weighted_dist_error = dist_error * ray_weight * spatial_weight_exp


        total_weight = tf.reduce_sum(ray_weight * spatial_weight_exp) + self.epsilon
        loss_distance = tf.reduce_sum(weighted_dist_error) / total_weight

        # ============================================================
        # === Debug ===
        # ============================================================

        if not hasattr(self, '_debug_count'):
            self._debug_count = {}
        if name not in self._debug_count:
            self._debug_count[name] = 0
        self._debug_count[name] += 1

        if self._debug_count[name] < 5:
            mean_pred_fg = tf.reduce_sum(prob_pred_squeeze * foreground_mask) / num_fg
            mean_pred_bg = tf.reduce_sum(prob_pred_squeeze * background_mask) / num_bg

            bg_probs_only = tf.where(background_mask > 0.5, prob_pred_squeeze, -1.0)
            max_pred_bg = tf.reduce_max(bg_probs_only)

            tf.print("\n  [STARDIST LOSS -", name, "]")
            tf.print("    BCE (balanced):     ", loss_bce)
            tf.print("      └─ fg component:  ", loss_bce_fg)
            tf.print("      └─ bg component:  ", loss_bce_bg)
            tf.print("    Mean prob (fg):     ", mean_pred_fg)
            tf.print("    Mean prob (bg):     ", mean_pred_bg, "← should be LOW!")
            tf.print("    Max prob (bg):      ", max_pred_bg)
            tf.print("    Dist (fg):      ", loss_distance)

            mean_dist_fg = tf.reduce_sum(dist_pred * spatial_weight_exp) / (
                    tf.reduce_sum(spatial_weight_exp) + self.epsilon
            )
            tf.print("    Mean dist (fg):     ", mean_dist_fg, "px")

        return {'bce': loss_bce, 'distance': loss_distance}

    # ============================================================
    # === WBR LOSS ===
    # ============================================================

    def _wbr_loss(self, nucleus_prob_pred, cytoplasm_prob_pred):
        """
        Within-Boundary Regularization.
        Nucleus должно быть внутри cytoplasm.
        """
        nucleus_mask = nucleus_prob_pred
        cytoplasm_inverted = 1.0 - cytoplasm_prob_pred

        # Nucleus predictions OUTSIDE cytoplasm
        nucleus_outside = nucleus_mask * cytoplasm_inverted
        numerator = tf.reduce_sum(nucleus_outside)

        # Нормализуем по NUCLEUS area
        denominator = tf.reduce_sum(nucleus_mask) + self.epsilon

        ratio = numerator / denominator

        return ratio

    # ============================================================
    # === CONTAINMENT LOSS ===
    # ============================================================

    def _containment_loss(self, y_true, y_pred, y_true_raw):
        """
        Multi-nuclear containment loss:
        1. Geometric: nucleus inside cytoplasm
        2. Distance field: nuc_dist <= cyto_dist
        3. Center separation: multi-nuclear cells должны иметь разделённые центры
        """
        if self.lambda_containment == 0:
            return tf.constant(0.0, dtype=tf.float32)

        nuc_prob = y_pred['nucleus_prob']
        cyto_prob = y_pred['cytoplasm_prob']
        nuc_dist = y_pred['nucleus_dist']
        cyto_dist = y_pred['cytoplasm_dist']

        nuc_prob_gt = y_true['nucleus_prob']
        cyto_prob_gt = y_true['cytoplasm_prob']

        # 1. GEOMETRIC CONTAINMENT: nucleus inside cytoplasm
        nuc_outside_cyto = nuc_prob * (1.0 - cyto_prob)
        loss_geometric = tf.reduce_mean(nuc_outside_cyto)

        # 2. DISTANCE FIELD CONTAINMENT
        nuc_fg = tf.cast(nuc_prob_gt > 0.5, tf.float32)
        n_rays = tf.shape(nuc_dist)[-1]

        if self.ndim == 2:
            nuc_fg_exp = tf.tile(nuc_fg, [1, 1, 1, n_rays])
        else:
            nuc_fg_exp = tf.tile(nuc_fg, [1, 1, 1, 1, n_rays])

        distance_violation = tf.nn.relu(nuc_dist - cyto_dist) * nuc_fg_exp
        loss_distance_containment = tf.reduce_mean(distance_violation)

        # 3. CENTER SEPARATION (для multi-nuclear cells)
        # Получаем median_radius из config или вычисляем
        if hasattr(self.config, 'median_radius_train') and self.config.median_radius_train is not None:
            median_radius = self.config.median_radius_train
        else:
            # Fallback: вычисляем из GT
            nuc_dist_fg = y_true['nucleus_dist'] * nuc_fg_exp
            num_fg = tf.reduce_sum(nuc_fg_exp) + self.epsilon
            median_radius = tf.reduce_sum(nuc_dist_fg) / num_fg
            median_radius = tf.maximum(median_radius, 1.0)

        loss_separation = self._compute_center_separation(
            nuc_prob=nuc_prob,
            cyto_prob_gt=cyto_prob_gt,
            nuc_prob_gt=nuc_prob_gt,
            median_radius=median_radius
        )
        # 4. CYTOPLASM SUPPORT: push cyto_prob up on GT nucleus regions
        # Penalizes low cytoplasm probability where GT nuclei exist
        loss_cyto_support = tf.reduce_mean((1.0 - cyto_prob) * nuc_fg)
        # COMBINE
        w_geometric = getattr(self.config, 'w_containment_geometric', 1.0)
        w_distance = getattr(self.config, 'w_containment_distance', 0.5)
        w_separation = getattr(self.config, 'w_center_separation', 0.2)
        w_cyto_support = getattr(self.config, 'w_cyto_support', 0.1)
        total_containment_loss = (
                w_geometric * loss_geometric +
                w_distance * loss_distance_containment +
                w_separation * loss_separation +
                w_cyto_support * loss_cyto_support
        )

        return total_containment_loss

    # ============================================================
    # === FRAGMENTATION LOSS (Pure TensorFlow) ===
    # ============================================================

    def _fragmentation_loss(self, prob_pred, instances_gt, name='object'):
        """
        Штрафует за предсказание нескольких peaks внутри одного GT объекта.
        ИСПРАВЛЕНО: Pure TensorFlow без py_function для совместимости с @tf.function

        Упрощённая версия:
        1. Находим peaks в probability map
        2. Для каждого GT объекта считаем суммарную вероятность peaks
        3. Штрафуем, если суммарная prob сильно превышает ожидаемую (1.0)

        Args:
            prob_pred: (B, H, W, 1) предсказанная вероятность
            instances_gt: (B, H, W) GT instance mask
            name: 'nucleus' или 'cytoplasm'

        Returns:
            loss: скаляр, штраф за фрагментацию
        """

        prob_squeeze = tf.squeeze(prob_pred, axis=-1)  # (B, H, W)

        # ============================================================
        # 1. Находим локальные максимумы (peaks)
        # ============================================================

        kernel_size = 5 if self.ndim == 2 else 5

        if self.ndim == 2:
            prob_maxpool = tf.nn.max_pool2d(
                prob_pred,
                ksize=kernel_size,
                strides=1,
                padding='SAME'
            )
        else:
            prob_maxpool = tf.nn.max_pool3d(
                prob_pred,
                ksize=kernel_size,
                strides=1,
                padding='SAME'
            )

        prob_maxpool_squeeze = tf.squeeze(prob_maxpool, axis=-1)

        # Peaks: где prob = maxpool и prob > 0.3
        is_peak = tf.logical_and(
            tf.abs(prob_squeeze - prob_maxpool_squeeze) < 0.01,
            prob_squeeze > 0.3
        )

        # Peak strength (вероятность в точках peaks)
        peak_strength = tf.where(is_peak, prob_squeeze, 0.0)

        # ============================================================
        # 2. Для каждого GT объекта считаем сумму peak strengths
        # ============================================================

        instances_gt_float = tf.cast(instances_gt, tf.float32)

        # Получаем unique labels (без 0)
        # КРИТИЧНО: используем tf ops вместо numpy

        # Простой подход: перебираем возможные labels от 1 до max_label
        max_label = tf.cast(tf.reduce_max(instances_gt_float), tf.int32)
        max_label = tf.minimum(max_label, 200)  # Ограничение для эффективности

        penalties = []

        # Векторизованный подход для каждого label
        for label_id in range(1, 201):  # Фиксированный диапазон для @tf.function
            label_id_float = tf.cast(label_id, tf.float32)

            # Маска объекта с этим label
            obj_mask = tf.equal(instances_gt_float, label_id_float)  # (B, H, W)
            obj_mask_float = tf.cast(obj_mask, tf.float32)

            # Сумма area объекта
            obj_area = tf.reduce_sum(obj_mask_float)

            # Skip если объекта нет
            obj_exists = tf.greater(obj_area, 0.5)

            # Сумма peak strengths внутри объекта
            peaks_in_obj = peak_strength * obj_mask_float
            total_peak_strength = tf.reduce_sum(peaks_in_obj)

            # Идеальное значение = 1.0 (один peak с prob=1.0)
            # Если total > 1.5, значит есть фрагментация
            fragmentation_penalty = tf.nn.relu(total_peak_strength - 1.5)

            # Применяем penalty только если объект существует
            penalty = tf.where(obj_exists, fragmentation_penalty, 0.0)

            penalties.append(penalty)

        # Среднее по всем возможным labels
        mean_penalty = tf.reduce_mean(tf.stack(penalties))

        # ИСПРАВЛЕНО: Нормализация по batch size И масштабирование
        batch_size = tf.cast(tf.shape(prob_squeeze)[0], tf.float32)

        # КРИТИЧНО: Делим на большое число для масштаба
        mean_penalty = mean_penalty / (batch_size * 100.0 + self.epsilon)  # ← ДОБАВЛЕНО /100

        return mean_penalty

    # ============================================================
    # === BACKGROUND PENALTY LOSS (уже Pure TensorFlow) ===
    # ============================================================

    def _background_penalty_loss(self, prob_pred, prob_gt, name='object'):
        """
        Усиленный штраф за высокие predictions на фоне.
        Pure TensorFlow - без изменений, эта функция уже корректна.
        """

        prob_pred_squeeze = tf.squeeze(prob_pred, axis=-1)  # (B, H, W)
        prob_gt_squeeze = tf.squeeze(prob_gt, axis=-1)

        # Фон = где GT prob < 0.1 (чистый background)
        background_mask = tf.cast(prob_gt_squeeze < 0.1, tf.float32)

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
                ksize=kernel_size_fixed,
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
        """
        Consistency within single GT object.
        Penalizes high variance of predicted prob within GT foreground.
        Предотвращает фрагментацию объектов.
        """
        if self.ndim == 2:
            prob_pred_map = tf.squeeze(prob_pred, axis=-1)  # (B, H, W)
            fg_mask = tf.squeeze(prob_gt, axis=-1)
        else:
            prob_pred_map = tf.squeeze(prob_pred, axis=-1)  # (B, H, W, D)
            fg_mask = tf.squeeze(prob_gt, axis=-1)

        fg_mask = tf.cast(fg_mask > 0.5, tf.float32)

        # Masked predictions
        prob_on_fg = prob_pred_map * fg_mask

        min_fg_pixels = 10.0 if self.ndim == 2 else 50.0

        # Векторизованная версия
        if self.ndim == 2:
            num_fg = tf.reduce_sum(fg_mask, axis=[1, 2])  # (B,)
            mean_prob = tf.reduce_sum(prob_on_fg, axis=[1, 2]) / (num_fg + self.epsilon)

            prob_centered = (prob_pred_map - tf.expand_dims(tf.expand_dims(mean_prob, -1), -1)) * fg_mask
            variance = tf.reduce_sum(tf.square(prob_centered), axis=[1, 2]) / (num_fg + self.epsilon)
        else:  # 3D
            num_fg = tf.reduce_sum(fg_mask, axis=[1, 2, 3])
            mean_prob = tf.reduce_sum(prob_on_fg, axis=[1, 2, 3]) / (num_fg + self.epsilon)

            prob_centered = (
                prob_pred_map -
                tf.expand_dims(tf.expand_dims(tf.expand_dims(mean_prob, -1), -1), -1)
            ) * fg_mask
            variance = tf.reduce_sum(tf.square(prob_centered), axis=[1, 2, 3]) / (num_fg + self.epsilon)

        # Adaptive threshold по размеру объекта
        size_factor = tf.minimum(num_fg / 200.0, 1.0)
        adaptive_threshold = 0.15 - 0.05 * size_factor

        # Penalty за высокую variance
        penalty = tf.nn.relu(variance - adaptive_threshold)

        # Маска для валидных объектов
        valid_mask = tf.cast(num_fg >= min_fg_pixels, tf.float32)

        weighted_penalty = penalty * valid_mask
        num_valid = tf.reduce_sum(valid_mask) + self.epsilon

        return tf.reduce_sum(weighted_penalty) / num_valid


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

    # СТРАТЕГИЯ 1: Maximum overlap
    for nuc_id, nuc_prop in nuc_props.items():
        nuc_mask = (nucleus_instances == nuc_id)

        overlaps = {}
        for cyto_id in cyto_props.keys():
            cyto_mask = (cytoplasm_instances == cyto_id)
            overlap_pixels = np.logical_and(nuc_mask, cyto_mask).sum()
            if overlap_pixels > 0:
                overlaps[cyto_id] = overlap_pixels

        if len(overlaps) > 0:
            best_cell = max(overlaps, key=overlaps.get)
            best_overlap = overlaps[best_cell]
            overlap_fraction = best_overlap / nuc_mask.sum()

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