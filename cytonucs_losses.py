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

        # Refinement heads weights
        self.lambda_cyto_mask = getattr(config, 'lambda_cyto_mask', 0.0)


        self.lambda_nuc_mask = getattr(config, 'lambda_nuc_mask', 0.0)



        # Anisotropy для анизотропных окон max-pool в лоссах (универсально 2D/3D)
        self.anisotropy = tuple(getattr(config, 'anisotropy', (1,) * self.ndim))
        if len(self.anisotropy) != self.ndim:
            self.anisotropy = (1,) * self.ndim
    def _warmup_factor(self, current_epoch):
        """
        Общий warmup для вспомогательных лоссов (WBR, containment, refine).
        Схема: hold(0) -> линейный ramp -> 1.0.
          factor = clip((epoch - hold) / ramp, 0, 1),  hold = ramp = warmup_epochs

        Кросс-головные лоссы (WBR, containment) связывают nucleus и cytoplasm
        через градиент. Пока обе головы не сошлись независимо, они тянут prob
        в стороны: WBR давит nucleus_prob вниз везде, где cyto_prob < 1, а
        w_cyto_support тянет cyto_prob вверх на GT-ядрах. Поэтому держим их
        в нуле, пока головы учатся, и вводим плавно.
        Refine (mask/boundary) не трогает backbone (stop_gradient), но учиться
        на несошедшихся фичах декодера ему тоже нечему -> тот же график.
        Новых параметров нет: берётся существующий wbr_warmup_epochs.
        """
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

        current_epoch_display = current_epoch          # для печати (1-based)
        current_epoch = current_epoch - 1              # для warmup (0-based)

        y_true = self._create_gt_from_instances(y_true_raw)

        # === 1. StarDist Loss ===
        loss_nucleus = self._stardist_loss(
            y_true['nucleus_prob'], y_pred['nucleus_prob'],
            y_true['nucleus_dist'], y_pred['nucleus_dist'],
            name='nucleus', prob_logit=y_pred.get('nucleus_prob_logit'))
        loss_cytoplasm = self._stardist_loss(
            y_true['cytoplasm_prob'], y_pred['cytoplasm_prob'],
            y_true['cytoplasm_dist'], y_pred['cytoplasm_dist'],
            name='cytoplasm', prob_logit=y_pred.get('cytoplasm_prob_logit'))
        warmup = self._warmup_factor(current_epoch)

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
            loss_fragmentation = self.lambda_fragmentation * (loss_frag_nuc + loss_frag_cyto) / 2.0

        # === 5. Background Penalty ===
        loss_background = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_background_penalty > 0:
            loss_bg_nuc = self._background_penalty_loss(
                y_pred['nucleus_prob'], y_true['nucleus_prob'], name='nucleus')
            loss_bg_cyto = self._background_penalty_loss(
                y_pred['cytoplasm_prob'], y_true['cytoplasm_prob'], name='cytoplasm')
            loss_background = self.lambda_background_penalty * (loss_bg_nuc + loss_bg_cyto) / 2.0

        # === 6. Mask refinement — БЕЗ warmup.
        # Ветка стоит за stop_gradient и контаминировать backbone не может,
        # поэтому держать её выключенной 2*warmup эпох — чистая потеря времени,
        # тем более что именно она производит инстансы на инференсе.
        loss_cyto_mask = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_cyto_mask > 0 and 'cytoplasm_mask' in y_pred:
            gt_mask_semantic = self._grid_subsample_cytoplasm(tf.expand_dims(
                tf.cast(y_true_raw['cytoplasm_target'], tf.float32), axis=-1))
            _cyto_any = tf.stop_gradient(self._grid_subsample_cytoplasm(
                tf.expand_dims(tf.cast(y_true_raw['cytoplasm_instances'] > 0,
                                       tf.float32), axis=-1)))
            loss_cyto_mask = self.lambda_cyto_mask * (
                    0.5 * self._refinement_ce_loss(
                y_pred['cytoplasm_mask_logit'], gt_mask_semantic, _cyto_any) +
                    0.5 * self._refinement_dice_loss(
                y_pred['cytoplasm_mask'], gt_mask_semantic))

        loss_nuc_mask = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_nuc_mask > 0 and 'nucleus_mask' in y_pred:
            gt_nuc_semantic = self._grid_subsample_nucleus(tf.expand_dims(
                tf.cast(y_true_raw['nucleus_target'], tf.float32), axis=-1))
            _nuc_any = tf.stop_gradient(self._grid_subsample_nucleus(
                tf.expand_dims(tf.cast(y_true_raw['nucleus_instances'] > 0,
                                       tf.float32), axis=-1)))
            loss_nuc_mask = self.lambda_nuc_mask * (
                    0.5 * self._refinement_ce_loss(
                y_pred['nucleus_mask_logit'], gt_nuc_semantic, _nuc_any) +
                    0.5 * self._refinement_dice_loss(
                y_pred['nucleus_mask'], gt_nuc_semantic))

        # === TOTAL ===
        total_loss = (
                cfg.lambda_prob_nucleus * loss_nucleus['bce'] +
                cfg.lambda_dist_nucleus * loss_nucleus['distance'] +
                cfg.lambda_prob_cytoplasm * loss_cytoplasm['bce'] +
                cfg.lambda_dist_cytoplasm * loss_cytoplasm['distance'] +
                loss_wbr +
                self.lambda_containment * warmup * loss_containment +
                loss_fragmentation +
                loss_background +
                loss_cyto_mask +
                loss_nuc_mask
        )

        def print_debug(epoch_val, nuc_bce, nuc_dist, cyto_bce, cyto_dist,
                        wbr_val, wbr_f, cont_val, frag_val, bg_val,
                        cyto_mask_val, nuc_mask_val, tot_val):
            if self._loss_call_count % 5 == 0:
                print(f"\n=== LOSS (epoch {int(epoch_val)}, global_step {self._loss_call_count}) ===")
                print(f"  Nucleus BCE: {float(nuc_bce):.4f} | Dist: {float(nuc_dist):.2f}")
                print(f"  Cytoplasm BCE: {float(cyto_bce):.4f} | Dist: {float(cyto_dist):.2f}")
                print(f"  WBR: {float(wbr_val):.4f} | warmup factor: {float(wbr_f):.2f}")
                print(f"  Containment: {float(cont_val):.4f} (raw, ×warmup в total)")
                print(f"  Fragmentation: {float(frag_val):.4f}")
                print(f"  Background: {float(bg_val):.4f}")
                print(f"  CytoMask: {float(cyto_mask_val):.4f} ")
                print(f"  NucMask: {float(nuc_mask_val):.4f} ")

                print(f"  TOTAL: {float(tot_val):.4f}")
            self._loss_call_count += 1
            return 0

        tf.py_function(
            print_debug,
            [current_epoch_display, loss_nucleus['bce'], loss_nucleus['distance'],
             loss_cytoplasm['bce'], loss_cytoplasm['distance'],
             loss_wbr, warmup, loss_containment,
             loss_fragmentation, loss_background,
             loss_cyto_mask, loss_nuc_mask, total_loss],
            [tf.int32]
        )

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
            getattr(self.config, 'rays_anisotropy_nucleus', self.config.anisotropy))

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
                                    rays, aniso):
        """Фолбэк-путь, когда GT не пришёл из генератора. Логика вынесена в
        compute_stardist_gt, чтобы генератор и лосс считали одно и то же."""
        instances = instances.numpy()
        prob_list, dist_list = [], []
        for b in range(instances.shape[0]):
            p, d = compute_stardist_gt(instances[b], self.config, grid,
                                       rays=rays, aniso=aniso)
            prob_list.append(p)
            dist_list.append(d)
        return (np.stack(prob_list, axis=0).astype(np.float32),
                np.stack(dist_list, axis=0).astype(np.float32))

    # ============================================================
    # === STARDIST LOSS ===
    # ============================================================

    def _stardist_loss(self, prob_true, prob_pred, dist_true, dist_pred,
                       name='object', prob_logit=None):

        prob_true_squeeze = tf.squeeze(prob_true, axis=-1)
        prob_pred_squeeze = tf.squeeze(prob_pred, axis=-1)
        foreground_mask = tf.cast(prob_true_squeeze > 0.01, tf.float32)
        background_mask = 1.0 - foreground_mask

        # Канонический StarDist (base.py:77): BCE минус энтропия таргета —
        # prob_true мягкая EDT-карта, без вычитания лосс имеет ненулевой минимум.
        # ПЛЮС адаптивный pos_weight: на 3D-патче доля fg порядка 0.5-1 %, и
        # чистый BCE топит fg-градиент в фоне. Вес выводится из данных
        # (num_bg/num_fg на текущем батче), потолок 50 — от вырождения на
        # почти пустых патчах. Энтропия взвешивается тем же множителем, чтобы
        # вычитаемая константа осталась согласованной с числителем.
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
        loss_bce = tf.reduce_mean(bce_raw)

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
        n_rays = tf.shape(dist_true)[-1]

        if name == 'nucleus':
            dist_scale = float(getattr(self.config, 'median_max_dist_nucleus', 15.0))
        else:
            dist_scale = float(getattr(self.config, 'median_max_dist_cytoplasm', 60.0))
        dist_scale = tf.cast(max(dist_scale, 1.0), tf.float32)

        dist_mask = tf.cast(prob_true, tf.float32)  # канон: мягкая edt-маска
        if self.ndim == 2:
            dist_mask_exp = tf.tile(dist_mask, [1, 1, 1, n_rays])
        else:
            dist_mask_exp = tf.tile(dist_mask, [1, 1, 1, 1, n_rays])

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

        weighted_error = dist_mask_exp * dist_error_per_ray
        loss_distance = tf.reduce_mean(weighted_error) / (
                tf.reduce_mean(dist_mask_exp) + self.epsilon)

        bg_reg_weight = tf.cast(
            getattr(self.config, 'train_background_reg', 1e-4), tf.float32)
        if bg_reg_weight > 0:
            bg_penalty = tf.reduce_mean((1.0 - dist_mask) * tf.abs(dist_pred))
            loss_distance = loss_distance + bg_reg_weight * bg_penalty

        # относительная ошибка — только для лога, в градиент не идёт
        loss_distance_rel = tf.stop_gradient(loss_distance / dist_scale)

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
            max_pred_fg = tf.reduce_max(prob_pred_squeeze * foreground_mask)
            tf.print("\n  [STARDIST LOSS -", name, "]")
            tf.print("    BCE :     ", loss_bce)
            tf.print("      └─ fg component:  ", loss_bce_fg)
            tf.print("      └─ bg component:  ", loss_bce_bg)
            tf.print("    Mean prob (fg):     ", mean_pred_fg, " Max prob (fg):", max_pred_fg)
            tf.print("    Mean prob (bg):     ", mean_pred_bg, " Max prob (bg):", max_pred_bg)
            if z is not None:
                # Кайма шириной b узлов против внутренней области. Если логит на
                # кайме систематически выше, это отклик на zero-padding, а не на
                # данные: там признаки после relu близки к нулю, и при суммарно
                # отрицательных весах головы логит выходит максимальным.
                _b = 8
                _e = tf.zeros_like(z)
                if self.ndim == 3:
                    _inner = tf.pad(tf.ones_like(z[:, :, _b:-_b, _b:-_b]),
                                    [[0, 0], [0, 0], [_b, _b], [_b, _b]])
                else:
                    _inner = tf.pad(tf.ones_like(z[:, _b:-_b, _b:-_b]),
                                    [[0, 0], [_b, _b], [_b, _b]])
                _rim = 1.0 - _inner
                tf.print("    logit кайма/внутри (mean):",
                         tf.reduce_sum(z * _rim) / (tf.reduce_sum(_rim) + self.epsilon),
                         "/",
                         tf.reduce_sum(z * _inner) / (tf.reduce_sum(_inner) + self.epsilon),
                         " max кайма:", tf.reduce_max(tf.where(_rim > 0.5, z, -1e9)))
                _big = tf.constant(1e9, tf.float32)
                _has_fg = tf.reduce_sum(foreground_mask) > 0.5
                tf.print("    logit bg  min/mean/max:",
                         tf.reduce_min(tf.where(background_mask > 0.5, z, _big)),
                         tf.reduce_sum(tf.where(background_mask > 0.5, z, 0.0)) / num_bg,
                         tf.reduce_max(tf.where(background_mask > 0.5, z, -_big)))
                tf.cond(_has_fg,
                        lambda: tf.print("    logit fg  min/mean/max:",
                                         tf.reduce_min(tf.where(foreground_mask > 0.5, z, _big)),
                                         tf.reduce_sum(tf.where(foreground_mask > 0.5, z, 0.0)) / num_fg,
                                         tf.reduce_max(tf.where(foreground_mask > 0.5, z, -_big)),
                                         "  разрыв fg-bg:",
                                         tf.reduce_sum(tf.where(foreground_mask > 0.5, z, 0.0)) / num_fg -
                                         tf.reduce_sum(tf.where(background_mask > 0.5, z, 0.0)) / num_bg),
                        lambda: tf.print("    logit fg: в патче НЕТ переднего плана"))
            tf.print("    Dist loss (", name, "):", loss_distance, " | rel:", loss_distance_rel)
            tf.print("    neg_ray frac:", tf.reduce_mean(tf.cast(dist_pred < 0, tf.float32)))
            fg_binary = tf.cast(
                tf.reduce_max(dist_true, axis=-1, keepdims=True) > 0, tf.float32
            )
            if self.ndim == 2:
                fg_binary_exp = tf.tile(fg_binary, [1, 1, 1, n_rays])
            else:
                fg_binary_exp = tf.tile(fg_binary, [1, 1, 1, 1, n_rays])
            fg_pixel_count = tf.reduce_sum(fg_binary) + self.epsilon
            n_rays_f = tf.cast(n_rays, tf.float32)
            fg_mask_nd = tf.squeeze(fg_binary, axis=-1)

            mean_dist_fg = tf.reduce_sum(dist_pred * fg_binary_exp) / (
                    tf.reduce_sum(fg_binary_exp) + self.epsilon)
            tf.print("    Mean dist (fg):     ", mean_dist_fg, "px")

            # --- SHAPE DIAG (для ОБЕИХ голов) ---
            large_val = 1e6
            pred_for_min = dist_pred + (1.0 - fg_binary_exp) * large_val
            pred_min_ray = tf.reduce_min(pred_for_min, axis=-1)
            pred_max_ray = tf.reduce_max(dist_pred * fg_binary_exp, axis=-1)
            valid = fg_mask_nd * tf.cast(pred_min_ray < large_val * 0.5, tf.float32)
            _floor = tf.cast(0.01 * dist_scale, tf.float32)
            valid = valid * tf.cast(pred_min_ray > _floor, tf.float32)
            n_valid = tf.reduce_sum(valid) + self.epsilon

            gt_for_min = dist_true + (1.0 - fg_binary_exp) * large_val
            gt_min_ray = tf.reduce_min(gt_for_min, axis=-1)
            gt_max_ray = tf.reduce_max(dist_true * fg_binary_exp, axis=-1)
            gt_valid = fg_mask_nd * tf.cast(gt_min_ray < large_val * 0.5, tf.float32)
            gt_valid = gt_valid * tf.cast(gt_min_ray > 0.5, tf.float32)
            n_gt_valid = tf.reduce_sum(gt_valid) + self.epsilon

            tf.print("    --- SHAPE DIAG (", name, ") ---")
            tf.print("    Pred min_ray:", tf.reduce_sum(pred_min_ray * valid) / n_valid,
                     "px  max_ray:", tf.reduce_sum(pred_max_ray * valid) / n_valid, "px")
            tf.print("    GT   min_ray:", tf.reduce_sum(gt_min_ray * gt_valid) / n_gt_valid,
                     "px  max_ray:", tf.reduce_sum(gt_max_ray * gt_valid) / n_gt_valid, "px")

            # --- ЛУЧИ В ЦЕНТРЕ (argmax GT prob) — ключевая диагностика формы ---
            pt0 = prob_true_squeeze[0]
            center_idx = tf.argmax(tf.reshape(pt0, [-1]))
            dt0 = tf.reshape(tf.cast(dist_true[0], tf.float32), [-1, n_rays])
            dp0 = tf.reshape(tf.cast(dist_pred[0], tf.float32), [-1, n_rays])
            gt_rays = tf.gather(dt0, center_idx)
            pr_rays = tf.gather(dp0, center_idx)
            tf.print("    [CENTER] GT   min/mean/max:", tf.reduce_min(gt_rays),
                     tf.reduce_mean(gt_rays), tf.reduce_max(gt_rays))
            tf.print("    [CENTER] PRED min/mean/max:", tf.reduce_min(pr_rays),
                     tf.reduce_mean(pr_rays), tf.reduce_max(pr_rays))
            _pr = tf.reduce_sum(pred_max_ray * valid) / n_valid
            _gr = tf.reduce_sum(gt_max_ray * gt_valid) / n_gt_valid
            tf.print("    max_ray pred/GT:", _pr / (_gr + self.epsilon))

            # prob В ИСТИННОМ ЦЕНТРЕ объекта. Max prob(fg) считается по маске
            # prob_true > 0.01, куда не попадает внешняя оболочка объекта
            # (edt_prob там ~0), поэтому высокий Max prob(bg) может быть как
            # ложным пиком в пустоте, так и римом настоящего объекта. Значение
            # в argmax GT prob различает эти два случая однозначно.
            tf.print("    [CENTER] prob GT:", tf.gather(tf.reshape(pt0, [-1]), center_idx),
                     " PRED:", tf.gather(tf.reshape(prob_pred_squeeze[0], [-1]), center_idx))
        return {'bce': loss_bce, 'distance': loss_distance, 'distance_rel': loss_distance_rel}

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

        # 1. GEOMETRIC CONTAINMENT
        loss_geometric = tf.reduce_mean(nuc_prob * (1.0 - cyto_fg_gt))

        # 2. DISTANCE FIELD CONTAINMENT (raw pixels)
        nuc_fg = tf.cast(nuc_prob_gt > 0.5, tf.float32)
        n_rays = tf.shape(nuc_dist)[-1]
        if self.ndim == 2:
            nuc_fg_exp = tf.tile(nuc_fg, [1, 1, 1, n_rays])
        else:
            nuc_fg_exp = tf.tile(nuc_fg, [1, 1, 1, 1, n_rays])
        distance_violation = tf.nn.relu(nuc_dist - cyto_dist) * nuc_fg_exp
        loss_distance_containment = tf.reduce_mean(distance_violation)

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
                num_fg = tf.reduce_sum(nuc_fg_exp) + self.epsilon
                median_radius = tf.maximum(
                    tf.reduce_sum(nuc_dist_gt * nuc_fg_exp) / num_fg, 1.0)
            loss_separation = self._compute_center_separation(
                nuc_prob=nuc_prob, cyto_prob_gt=cyto_prob_gt,
                nuc_prob_gt=nuc_prob_gt, median_radius=median_radius)
        else:
            loss_separation = tf.constant(0.0, dtype=tf.float32)

        w_geometric = getattr(self.config, 'w_containment_geometric', 1.0)
        w_distance = getattr(self.config, 'w_containment_distance', 0.5)
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
        if self.ndim == 2:
            spatial_dims = [1, 2]
        else:
            spatial_dims = [1, 2, 3]

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
    # === REFINEMENT BCE LOSS (NEW) ===
    # ============================================================
    def _refinement_dice_loss(self, pred, target):
        """
        Soft Dice loss для mask head.
        Напрямую оптимизирует IoU pixel-wise, заставляет model выдавать
        полное покрытие клеток (vs BCE которая может выучить только cores).
        """
        pred_squeeze = tf.squeeze(pred, axis=-1)
        target_squeeze = tf.cast(tf.squeeze(target, axis=-1), tf.float32)

        # Per-image Dice (averaging по batch)
        # Расчёт по spatial dims, leaving batch dim
        spatial_dims = list(range(1, len(pred_squeeze.shape)))

        intersection = tf.reduce_sum(pred_squeeze * target_squeeze, axis=spatial_dims)
        pred_sum = tf.reduce_sum(pred_squeeze, axis=spatial_dims)
        target_sum = tf.reduce_sum(target_squeeze, axis=spatial_dims)

        dice_per_image = (2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)
        loss = 1.0 - tf.reduce_mean(dice_per_image)

        return loss


    def _refinement_ce_loss(self, logit, target_fg, inst_any):
        """
        Трёхклассовая CE для промпт-маски: 0 фон, 1 целевой инстанс, 2 другие.

        Таргет собирается здесь из того, что уже есть у генератора:
          class 1 = *_target          (выбранный инстанс)
          class 2 = (*_instances > 0) И НЕ class 1
          class 0 = остальное

        БЕЗ взвешивания классов. Обоснование: фон — лёгкий класс, его CE падает
        почти до нуля за первые эпохи, после чего лосс определяется путаницей
        1<->2 на стыке инстансов, то есть ровно тем, что нужно. Плюс рядом
        стоит soft Dice по классу 1, который смотрит только на передний план и
        не даёт ему утонуть. Подобранных множителей в схеме нет — это
        стандартная многоклассовая постановка.

        Возвращает CE. Диагностика долей классов печатается первые шаги: если
        доля class 1 в предсказании уйдёт в ноль, значит фон всё-таки давит и
        балансировку придётся вводить — но тогда уже по измерению, а не заранее.
        """
        z = tf.cast(logit, tf.float32)                      # (B, ..., 3)
        c1 = tf.cast(tf.squeeze(target_fg, axis=-1) > 0.5, tf.float32)
        c2 = tf.cast(tf.squeeze(inst_any, axis=-1) > 0.5, tf.float32) * (1.0 - c1)
        c0 = 1.0 - c1 - c2
        labels = tf.stack([c0, c1, c2], axis=-1)
        if self._debug_count.get('_refine', 0) < 5:
            self._debug_count['_refine'] = self._debug_count.get('_refine', 0) + 1
            _p = tf.nn.softmax(z, axis=-1)
            _n = tf.cast(tf.size(c1), tf.float32)
            tf.print("    [refine 3cls] GT доли 0/1/2:",
                     tf.reduce_sum(c0) / _n, tf.reduce_sum(c1) / _n,
                     tf.reduce_sum(c2) / _n,
                     " PRED доли:", tf.reduce_mean(_p[..., 0]),
                     tf.reduce_mean(_p[..., 1]), tf.reduce_mean(_p[..., 2]),
                     " P(class1) на GT-цели:",
                     tf.reduce_sum(_p[..., 1] * c1) / (tf.reduce_sum(c1) + self.epsilon),
                     " на СОСЕДЯХ:",
                     tf.reduce_sum(_p[..., 1] * c2) / (tf.reduce_sum(c2) + self.epsilon))
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=z))



def compute_stardist_gt(instances, config, grid, rays=None, aniso=None):
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
    from stardist.utils import fill_label_holes, edt_prob

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

    lbl = fill_label_holes(np.asarray(instances).astype(np.int32))

    if ndim == 3:
        dist = star_dist3D(lbl, rays, mode='cpp', grid=grid)
    else:
        dist = star_dist(lbl, config.n_rays, mode='cpp', grid=grid)
    dist = np.clip(dist, 1e-3, max_clip).astype(np.float32)

    ss = tuple(slice(None, None, g) for g in grid)
    prob = edt_prob(lbl, anisotropy=aniso)[ss][..., np.newaxis].astype(np.float32)

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