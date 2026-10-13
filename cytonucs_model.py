import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from cytonucs_geometry import architecture_plan, refinement_pool_schedule


class MaskQualityPool(layers.Layer):
    """Pool over the predicted instance, with no quality gradient into its mask.

    Squared foreground probabilities suppress the large low-probability background.
    No weights: switching from global pooling retains all checkpoint tensor shapes.
    """
    def call(self, inputs):
        hidden, logits = inputs
        weight = tf.stop_gradient(tf.square(tf.nn.softmax(logits, axis=-1)[..., 1:2]))
        axes = tuple(range(1, len(hidden.shape) - 1))
        return tf.math.divide_no_nan(tf.reduce_sum(hidden * weight, axis=axes),
                                     tf.reduce_sum(weight, axis=axes))


class ScalarGate(layers.Layer):
    """Обучаемый скалярный гейт в (0,1) через sigmoid. Масштабирует вход.
    Используется чтобы цитоплазма получала контекст ядра ОСЛАБЛЕННЫМ —
    модель сама калибрует силу влияния ядра."""

    def __init__(self, init_value=0.3, **kwargs):
        super().__init__(**kwargs)
        self.init_value = float(init_value)

    def build(self, input_shape):
        logit = float(np.log(self.init_value / (1.0 - self.init_value)))
        self.gate = self.add_weight(
            name='gate', shape=(),
            initializer=tf.keras.initializers.Constant(logit),
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x * tf.sigmoid(self.gate)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'init_value': self.init_value})
        return cfg


FINE_CHANNELS = 8


def _fine_head(ConvLayer, ndim, grid, name, feature_channels, image_channels):
    """Residual full-resolution mask head: linear upsampling of the coarse logits (as
    upsample_grid) plus a correction from the window's full-resolution backbone features
    (projected here, inside the window), the image and the prompt; the output layer is
    zero-initialised, so the head starts as the upsampled coarse mask."""
    from cytonucs_fine_mask import upsample_linear
    spatial = (None,) * ndim
    coarse = layers.Input(spatial + (3,)); hidden = layers.Input(spatial + (FINE_CHANNELS,))
    features = layers.Input(spatial + (feature_channels,)); image = layers.Input(spatial + (image_channels,))
    prompt = layers.Input(spatial + (1,))
    up_coarse = layers.Lambda(lambda t: upsample_linear(t, grid), name=f'{name}_up_coarse')(coarse)
    up_hidden = layers.Lambda(lambda t: upsample_linear(t, grid), name=f'{name}_up_hidden')(hidden)
    fine = ConvLayer(FINE_CHANNELS, 1, padding='same', activation='relu', kernel_initializer='he_normal',
                     name=f'{name}_feature_proj')(features)
    x = layers.Concatenate(name=f'{name}_concat')([up_coarse, up_hidden, fine, image, prompt])
    for i in (1, 2):
        x = ConvLayer(16, 3, padding='same', activation='relu', kernel_initializer='he_normal',
                      name=f'{name}_c{i}')(x)
    delta = ConvLayer(3, 1, padding='same', kernel_initializer='zeros', bias_initializer='zeros',
                      name=f'{name}_delta')(x)
    return Model([coarse, hidden, features, image, prompt], layers.Add(name=f'{name}_logits')([up_coarse, delta]),
                 name=name)


class CytoNucsStarDistModel:

    def __init__(self, config, name='cytonucs_stardist', basedir='models', fine_wiring='random_slot'):
        self.config = config
        self.name = name
        self.basedir = basedir
        # 'slot0' rebuilds the graph of v3.2 checkpoints (fine head on the first prompt) only to
        # load them: Keras .weights.h5 files address layers by graph order, not by layer name.
        self.fine_wiring = fine_wiring
        self.ndim = config.ndim
        # Инициализатор backbone-свёрток; выставляется в _build_model,
        # дефолт нужен на случай раннего вызова _build_block.
        self._kinit = 'he_normal'
        self.keras_model = self._build_model()

    def _build_block(self, x, n_filters, ConvLayer, prefix):
        """UNet conv block: 2x Conv3x3 ReLU; with model.unet_norm_groups each convolution is
        followed by GroupNorm before its activation (a batch is one or a few crops).
        Инициализатор — self._kinit, обоснование в _build_model."""
        _k = getattr(self, '_kinit', 'he_normal')
        import re
        level = re.search(r'_l(\d+)', prefix)
        schedule = self.config.unet_kernel_schedule
        kernel = tuple(schedule[int(level.group(1))]) if schedule is not None and level else 3
        activation = getattr(self.config, 'unet_activation', 'relu')
        groups = int(getattr(self.config, 'unet_norm_groups', 0) or 0)
        for j in (1, 2):
            if groups:
                x = ConvLayer(n_filters, kernel, padding='same', activation=None,
                              kernel_initializer=_k, name=f'{prefix}_c{j}')(x)
                x = layers.GroupNormalization(groups=groups, name=f'{prefix}_n{j}')(x)
                x = layers.Activation(activation, name=f'{prefix}_a{j}')(x)
            else:
                x = ConvLayer(n_filters, kernel, padding='same', activation=activation,
                              kernel_initializer=_k,
                              name=f'{prefix}_c{j}')(x)
        return x

    def _build_model(self):
        cfg = self.config

        if self.ndim == 3:
            ConvLayer = layers.Conv3D
            PoolLayer = layers.MaxPooling3D
            UpSampleLayer = layers.UpSampling3D
        else:
            ConvLayer = layers.Conv2D
            PoolLayer = layers.MaxPooling2D
            UpSampleLayer = layers.UpSampling2D

        unet_n_depth = getattr(cfg, 'unet_n_depth', 4)
        unet_n_filter_base = getattr(cfg, 'unet_n_filter_base', 32)
        net_conv_after_unet = getattr(cfg, 'net_conv_after_unet', 128)
        unet_activation = getattr(cfg, 'unet_activation', 'relu')
        mask_n_filter = getattr(cfg, 'mask_refinement_filters', 64)

        # ============================================================
        # ИНИЦИАЛИЗАЦИЯ СВЁРТОК С ReLU
        # ============================================================
        # Preserve legacy initializers; the complete activation path is now configurable.
        self._kinit = ('he_normal'
                       if str(unet_activation).lower() in
                          ('relu', 'leaky_relu', 'relu6', 'gelu')
                       else 'glorot_uniform')
        print(f"  ✓ Инициализация backbone-свёрток: {self._kinit} "
              f"(активация {unet_activation})")

        _aniso = tuple(getattr(cfg, 'anisotropy', (1,) * self.ndim))
        if len(_aniso) != self.ndim:
            _aniso = (1,) * self.ndim

        def _read_grid(name):
            g = tuple(getattr(cfg, name, (1,) * self.ndim))
            if len(g) != self.ndim or any(int(v) < 1 for v in g):
                g = (1,) * self.ndim
            return tuple(int(v) for v in g)

        nucleus_grid = _read_grid('nucleus_grid')
        cytoplasm_grid = _read_grid('cytoplasm_grid')
        self._nucleus_grid = nucleus_grid
        self._cytoplasm_grid = cytoplasm_grid

        # ============================================================
        # ДВА РАЗНЫХ ПУЛИНГА — КАК В КАНОНЕ (model3d.py:365-380)
        # ============================================================
        # Explicit schedules override the legacy shape-derived pooling plan.
        # Kernel/pooling ablations change the architecture and need separate checkpoints.

        plan = architecture_plan(cfg)
        base_grid = plan['base_grid']
        base_steps = plan['base_steps']
        nuc_skip = plan['nucleus_skip']
        cyto_skip = plan['cytoplasm_skip']

        unet_pools = plan['pools']
        _sz = plan['bottleneck']
        self._architecture_plan = plan
        self._unet_pools = unet_pools
        self._base_steps = base_steps
        self._input_multiple = plan['input_multiple']

        print(f"  ✓ Прореживание входа до grid {base_grid}: {base_steps}")
        print(f"  ✓ UNet pooling по уровням: {unet_pools}")
        print(f"  ✓ Бутылочное горлышко: {tuple(_sz)}; "
              f"вход должен быть кратен {self._input_multiple}")
        print(f"  ✓ Головы: nucleus {nucleus_grid} (skip {nuc_skip}), "
              f"cytoplasm {cytoplasm_grid} (skip {cyto_skip})")

        # ============================================================
        # ВХОД
        # ============================================================
        # K наборов промптов лежат в КАНАЛАХ, а не в копиях батча.
        #
        # Legacy refinement uses stop_gradient; v2 can train decoder features.
        # (nuc/cyto_decoder_stop_grad_for_refine, cyto_ctx_stop_grad_for_refine,
        # _geo_prior), поэтому backbone можно посчитать ОДИН раз, а ветку
        # применить K раз с общими весами: память растёт только на её
        # активации (2 conv по mask_refinement_filters), а не на весь UNet.
        #
        # K is prompts_per_crop; image_batch_size controls independent crops.
        # ============================================================
        print(f"  ✓ Refinement: depth={cfg.refinement_depth}, feature gradient={cfg.refinement_gradient_scale}, quality={cfg.mask_quality_head}")
        n_img_ch = int(cfg.n_channel_in)
        n_pr = max(1, int(getattr(cfg, 'prompts_per_crop', cfg.train_batch_size)))
        self._n_prompts = n_pr

        input_shape = (None,) * self.ndim + (n_img_ch + 2 * n_pr,)
        inputs_all = layers.Input(shape=input_shape, name='input')
        inputs = layers.Lambda(lambda t: t[..., :n_img_ch],
                               name='img_channels')(inputs_all)

        prompt_nuc_full, prompt_cyto_full = [], []
        for k in range(n_pr):
            _o = n_img_ch + 2 * k
            prompt_nuc_full.append(layers.Lambda(
                lambda t, _a=_o: t[..., _a:_a + 1],
                name=f'prompt_nuc_{k}')(inputs_all))
            prompt_cyto_full.append(layers.Lambda(
                lambda t, _a=_o: t[..., _a + 1:_a + 2],
                name=f'prompt_cyto_{k}')(inputs_all))

        _AvgPool = layers.AveragePooling3D if self.ndim == 3 else layers.AveragePooling2D

        def _prompt_with_coords(p_full, steps, tag, grid):
            """Промпт + ОТНОСИТЕЛЬНЫЕ КООРДИНАТЫ до его центра.
            Рефайн-ветка это несколько conv3, её собственное рецептивное поле
            мало по сравнению с объектом, поэтому воксель на краю не видит
            гауссиану в центре. Смещение до центра он знает локально."""
            p = p_full
            for _i, st in enumerate(steps):
                if any(v > 1 for v in st):
                    p = _AvgPool(st, name=f'prompt_pool_{tag}_{_i}')(p)

            # Смещения в ФИЗИЧЕСКИХ единицах: шаг вокселя на своём разрешении
            # равен anisotropy[a]*grid[a], нормируем на физический размер
            # самой длинной оси патча.
            _step = [float(_aniso[a]) * float(grid[a]) for a in range(self.ndim)]
            _extent = max(float(cfg.train_patch_size[a]) * float(_aniso[a])
                          for a in range(self.ndim))
            _w = [s / _extent for s in _step]

            def _rel(t):
                sh = tf.shape(t)
                flat = tf.reshape(t[..., 0], [sh[0], -1])
                amax = tf.argmax(flat, axis=1, output_type=tf.int32)
                dims = [sh[a + 1] for a in range(self.ndim)]
                idx, rem = [], amax
                for a in range(self.ndim):
                    stride = 1
                    for b in range(a + 1, self.ndim):
                        stride = stride * dims[b]
                    idx.append(rem // stride)
                    rem = rem % stride
                out = []
                for a in range(self.ndim):
                    n = dims[a]
                    ax = tf.cast(tf.range(n), tf.float32)
                    shape = [1] * (self.ndim + 2)
                    shape[a + 1] = -1
                    ax = tf.reshape(ax, shape)
                    c = tf.reshape(tf.cast(idx[a], tf.float32),
                                   [sh[0]] + [1] * (self.ndim + 1))
                    out.append((ax - c) * _w[a] * tf.ones_like(t))
                return tf.concat(out, axis=-1)

            rel = layers.Lambda(_rel, name=f'prompt_rel_{tag}')(p)
            return layers.Concatenate(name=f'prompt_full_{tag}')([p, rel])

        # Промпт живёт на разрешении СВОЕЙ головы: сначала прореживание входа
        # до base_grid, затем столько же уровней UNet, сколько пропускает голова.
        _nuc_prompt_steps = base_steps + unet_pools[:nuc_skip]
        _cyto_prompt_steps = base_steps + unet_pools[:cyto_skip]

        p_at_nuc = [_prompt_with_coords(prompt_nuc_full[k], _nuc_prompt_steps,
                                        f'nuc{k}', nucleus_grid)
                    for k in range(n_pr)]
        p_at_cyto = [_prompt_with_coords(prompt_cyto_full[k], _cyto_prompt_steps,
                                         f'cyto{k}', cytoplasm_grid)
                     for k in range(n_pr)]

        # ============================================================
        # ПРОРЕЖИВАНИЕ ВХОДА ДО base_grid (канон model3d.py:365-375)
        # ============================================================
        pooled = inputs
        full_res_features = None
        first_kernel = tuple(cfg.unet_kernel_schedule[0]) if cfg.unet_kernel_schedule else 3
        for i, st in enumerate(base_steps):
            pooled = ConvLayer(unet_n_filter_base, first_kernel, padding='same',
                               activation=unet_activation,
                               kernel_initializer=self._kinit,
                               name=f'pre_pool{i}_c1')(pooled)
            pooled = ConvLayer(unet_n_filter_base, first_kernel, padding='same',
                               activation=unet_activation,
                               kernel_initializer=self._kinit,
                               name=f'pre_pool{i}_c2')(pooled)
            if full_res_features is None:
                full_res_features = pooled
            pooled = PoolLayer(st, name=f'pre_pool{i}')(pooled)

        # ============================================================
        # ENCODER (общий) — ПОВЕРХ прореженного входа, пулинг по ВСЕМ осям
        # ============================================================
        encoder_features = []
        x = pooled
        for i in range(unet_n_depth + 1):
            n_filters = unet_n_filter_base * (2 ** i)
            x = self._build_block(x, n_filters, ConvLayer, prefix=f'enc_l{i}')
            encoder_features.append(x)
            if i < unet_n_depth:
                x = PoolLayer(unet_pools[i], name=f'enc_l{i}_pool')(x)
        if full_res_features is None:
            full_res_features = encoder_features[0]

        cell_encoder_features = encoder_features
        if not cfg.shared_encoder:
            x = inputs
            for i, st in enumerate(base_steps):
                for j in (1, 2):
                    x = ConvLayer(unet_n_filter_base, first_kernel, padding='same',
                                  activation=unet_activation, kernel_initializer=self._kinit,
                                  name=f'cell_pre_pool{i}_c{j}')(x)
                x = PoolLayer(st, name=f'cell_pre_pool{i}')(x)
            cell_encoder_features = []
            for i in range(unet_n_depth + 1):
                x = self._build_block(x, unet_n_filter_base * 2**i, ConvLayer, f'cell_enc_l{i}')
                cell_encoder_features.append(x)
                if i < unet_n_depth:
                    x = PoolLayer(unet_pools[i], name=f'cell_enc_l{i}_pool')(x)

        def _decode(bottleneck, enc_feats, skip, prefix, ctx=None, ctx_skip=0):
            """Симметричный декодер: с уровня unet_n_depth обратно до `skip`.
            Пулинг/апсемплинг на уровне i — unet_pools[i], тот же, что в
            энкодере, поэтому skip-connection сходится по форме на всех осях."""
            x = bottleneck
            for i in reversed(range(skip, unet_n_depth)):
                n_filters = unet_n_filter_base * (2 ** i)
                x = UpSampleLayer(unet_pools[i], name=f'{prefix}_l{i}_up')(x)
                concat_list = [x, enc_feats[i]]
                if ctx is not None:
                    c = ctx
                    for j in range(ctx_skip, i):
                        if any(v > 1 for v in unet_pools[j]):
                            c = PoolLayer(unet_pools[j])(c)
                    concat_list.append(c)
                x = layers.Concatenate(name=f'{prefix}_l{i}_concat')(concat_list)
                x = self._build_block(x, n_filters, ConvLayer,
                                      prefix=f'{prefix}_l{i}')
            return x

        # ============================================================
        # ГОЛОВЫ prob/dist. Отклонение от канона (model3d.py:379-386, где обе
        # растут из ОДНОГО features): слои раздельные, активация dist задаётся
        # конфигом (dist_activation).
        #
        # prob отдаётся И логитом, И вероятностью: BCE считается из логита
        # (sigmoid_cross_entropy_with_logits), где производная (sigmoid(z) - t)
        # не обнуляется при насыщении. Прежний путь через clip_by_value давал
        # РОВНО НОЛЬ градиента ниже 1e-7, и голова замерзала необратимо.
        #
        # Выходные слои — glorot, за ними ReLU нет.
        # ============================================================
        _dist_act = str(getattr(cfg, 'dist_activation', 'linear')).lower()
        _n_geo = int(getattr(cfg, 'refine_geom_channels', 8))

        def _dist_head(feat, name, bias_key, fallback_key, default):
            """softplus с bias = масштаб объекта, либо канонический linear.

            ВНИМАНИЕ: dist_bias_init_* сейчас НЕ заполняется замером геометрии
            (расчёт был удалён при переписывании set_anisotropy_from_labels),
            поэтому softplus-ветка падает на fallback median_max_dist_*, а это
            САМЫЙ ДЛИННЫЙ луч (полудиагональ bbox) — втрое выше медианы длин
            лучей, замеренной по star_dist3D. Пока замер не восстановлен,
            корректен только dist_activation='linear'. Печать ниже показывает,
            откуда взято значение.
            """
            if _dist_act == 'softplus':
                _measured = getattr(cfg, bias_key, None)
                b = float(_measured or getattr(cfg, fallback_key, default))
                print(f"  {'✓' if _measured else '⚠️ '} {name}: softplus, "
                      f"bias={b:.1f} px "
                      f"({'замер' if _measured else 'ФОЛБЭК на max-луч'})")
                return ConvLayer(
                    cfg.n_rays, 1, padding='same', activation='softplus',
                    kernel_initializer='glorot_uniform',
                    bias_initializer=tf.keras.initializers.Constant(
                        float(np.log(np.expm1(max(b, 1e-3))))),
                    name=name)(feat)
            return ConvLayer(
                cfg.n_rays, 1, padding='same', activation='linear',
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                name=name)(feat)

        def _geo_prior(prob_t, dist_t, scale, tag):
            """Coarse-оценка StarDist на вход рефайна, ЗА stop_gradient.
            Рефайн иначе сегментирует с нуля, игнорируя уже посчитанную
            геометрию. dist нормируется на масштаб объекта и сжимается с
            n_rays до _n_geo: 96 сырых каналов забили бы вход числом каналов."""
            _p = layers.Lambda(lambda t: tf.stop_gradient(t),
                               name=f'{tag}_geo_prob_sg')(prob_t)
            _s = max(float(scale), 1e-3)
            _d = layers.Lambda(lambda t, _k=_s: tf.stop_gradient(t) / _k,
                               name=f'{tag}_geo_dist_sg')(dist_t)
            _d = ConvLayer(_n_geo, 1, padding='same', activation=unet_activation,
                           kernel_initializer=self._kinit,
                           name=f'{tag}_geo_dist_compress')(_d)
            return _p, _d

        def _stack_prompts(tensors, name):
            """K выходов рефайна -> ось БАТЧА. Строка k*B + b отвечает промпту k
            сэмпла b; в том же порядке генератор обязан сложить *_target."""
            if len(tensors) == 1:
                return layers.Lambda(lambda t: t, name=name)(tensors[0])
            return layers.Lambda(lambda ts: tf.concat(ts, axis=0),
                                 name=name)(tensors)

        def _feature_gradient(t):
            frozen = tf.stop_gradient(t)
            if not cfg.refinement_gradient_scale:
                return frozen
            return frozen + float(cfg.refinement_gradient_scale) * (t - frozen)

        def _multiscale_refiner(channels, tag, head):
            # Pooling has no weights: the schedule changes features, not tensor shapes.
            schedule = refinement_pool_schedule(cfg, head)
            inp = layers.Input(shape=(None,) * self.ndim + (channels,))
            t, skips = inp, []
            for level in range(cfg.refinement_depth):
                t = ConvLayer(mask_n_filter, 3, padding='same', activation='relu',
                              kernel_initializer='he_normal', name=f'{tag}_refine_down{level}')(t)
                skips.append(t)
                t = PoolLayer(schedule[level])(t)
            t = ConvLayer(mask_n_filter, 3, padding='same', activation='relu',
                          kernel_initializer='he_normal', name=f'{tag}_refine_bottleneck')(t)
            for level in reversed(range(cfg.refinement_depth)):
                t = UpSampleLayer(schedule[level])(t)
                t = layers.Concatenate()([t, skips[level]])
                t = ConvLayer(mask_n_filter, 3, padding='same', activation='relu',
                              kernel_initializer='he_normal', name=f'{tag}_refine_up{level}')(t)
            return Model(inp, t, name=f'{tag}_refine_multiscale')

        # ============================================================
        # DECODER 1: NUCLEUS
        # ============================================================
        x_nucleus = _decode(encoder_features[-1], encoder_features,
                            nuc_skip, 'nuc_dec')

        x_nuc_feat_prob = ConvLayer(
            net_conv_after_unet, 3, padding='same', activation=unet_activation,
            kernel_initializer=self._kinit, name='nuc_features_prob'
        )(x_nucleus)
        x_nuc_feat_dist = ConvLayer(
            net_conv_after_unet, 3, padding='same', activation=unet_activation,
            kernel_initializer=self._kinit, name='nuc_features_dist'
        )(x_nucleus)

        _prob_logit_nuc = ConvLayer(
            1, 1, padding='same', activation=None,
            kernel_initializer='glorot_uniform', name='nucleus_prob_logit'
        )(x_nuc_feat_prob)
        nucleus_prob = layers.Activation('sigmoid',
                                         name='nucleus_prob')(_prob_logit_nuc)
        nucleus_dist = _dist_head(x_nuc_feat_dist, 'nucleus_dist',
                                  'dist_bias_init_nucleus',
                                  'median_max_dist_nucleus', 15.0)
        print(f"  ✓ Nucleus head: split prob/dist features, dist={_dist_act}")

        # Decoder gradient is controlled by refinement_gradient_scale; geometry remains detached.
        # Слои создаются ОДИН раз, применяются к каждому из K промптов: веса
        # общие, а примеров за шаг в K раз больше.
        x_nucleus_for_refine = layers.Lambda(
            _feature_gradient, name='nuc_decoder_stop_grad_for_refine'
        )(x_nucleus)
        _gp_n, _gd_n = _geo_prior(
            nucleus_prob, nucleus_dist,
            getattr(cfg, 'median_max_dist_nucleus', 15.0) or 15.0, 'nuc')

        _nuc_gate_l = ConvLayer(int(x_nucleus_for_refine.shape[-1]), 1,
                                padding='same', activation='sigmoid',
                                kernel_initializer='glorot_uniform',
                                name='nuc_prompt_gate')
        _nuc_c1 = ConvLayer(mask_n_filter, 3, padding='same', activation='relu',
                            kernel_initializer='he_normal',
                            name='nuc_refine_conv1')
        _nuc_c2 = ConvLayer(mask_n_filter, 3, padding='same', activation='relu',
                            kernel_initializer='he_normal',
                            name='nuc_refine_conv2')
        _nuc_out = ConvLayer(3, 3, padding='same', activation=None,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros', name='nucleus_mask_out')

        _nuc_logits = []
        _nuc_hidden = []
        _nuc_body = None
        _nuc_quality = []
        _global_pool = layers.GlobalAveragePooling3D if self.ndim == 3 else layers.GlobalAveragePooling2D
        if cfg.mask_quality_head:
            _nq_pool = (MaskQualityPool if cfg.mask_quality_pooling == 'mask' else _global_pool)(name='nuc_refine_quality_pool')
            _nq_dense = layers.Dense(1, activation='sigmoid', name='nuc_refine_quality')
        for k in range(n_pr):
            _mod = layers.Multiply(name=f'nuc_prompt_mod_{k}')(
                [x_nucleus_for_refine, _nuc_gate_l(p_at_nuc[k])])
            _in = layers.Concatenate(name=f'nuc_refine_concat_{k}')(
                [_mod, p_at_nuc[k], _gp_n, _gd_n])
            if cfg.refinement_depth:
                if _nuc_body is None:
                    _nuc_body = _multiscale_refiner(int(_in.shape[-1]), 'nuc', 'nucleus')
                hidden = _nuc_body(_in)
            else:
                hidden = _nuc_c2(_nuc_c1(_in))
            _nuc_hidden.append(hidden)
            _nuc_logits.append(_nuc_out(hidden))
            if cfg.mask_quality_head:
                pooled = _nq_pool([hidden, _nuc_logits[-1]]) if cfg.mask_quality_pooling == 'mask' else _nq_pool(hidden)
                _nuc_quality.append(_nq_dense(pooled))

        _nuc_mask_logit = _stack_prompts(_nuc_logits, 'nucleus_mask_logit')
        nucleus_mask = layers.Lambda(
            lambda t: tf.nn.softmax(t, axis=-1)[..., 1:2], name='nucleus_mask'
        )(_nuc_mask_logit)
        print(f"  ✓ Nucleus refinement: 3-class prompted mask ({mask_n_filter} ch) "
              f"+ geometry prior ({_n_geo} ch), промптов за форвард: {n_pr}")

        # ============================================================
        # КОНТЕКСТ ЯДРА -> цитоплазма. Только в ДЕКОДЕР, не в головы и не в
        # рефайн. Вход контекста застопорен, поэтому декодер ядра от
        # цитоплазмы не зависит.
        # ============================================================
        x_nucleus_for_ctx = layers.Lambda(
            lambda t: tf.stop_gradient(t), name='nucleus_ctx_input_stop_grad'
        )(x_nucleus)
        nucleus_prob_context = ConvLayer(
            1, 3, padding='same', activation='sigmoid',
            kernel_initializer='glorot_uniform', name='nucleus_prob_ctx_src'
        )(x_nucleus_for_ctx)
        nucleus_context_gated = ScalarGate(
            init_value=getattr(cfg, 'nucleus_context_init', 0.3),
            name='nucleus_context_gate'
        )(nucleus_prob_context)

        # Приведение с уровня nuc_skip на уровень cyto_skip теми же пулингами,
        # что использует UNet на этих уровнях.
        nucleus_context_at_cyto = nucleus_context_gated
        for j in range(nuc_skip, cyto_skip):
            if any(v > 1 for v in unet_pools[j]):
                nucleus_context_at_cyto = PoolLayer(
                    unet_pools[j])(nucleus_context_at_cyto)
        for j in range(cyto_skip, nuc_skip):
            if any(v > 1 for v in unet_pools[j]):
                nucleus_context_at_cyto = UpSampleLayer(
                    unet_pools[j])(nucleus_context_at_cyto)

        # ============================================================
        # DECODER 2: CYTOPLASM
        # ============================================================
        x_cytoplasm = _decode(cell_encoder_features[-1], cell_encoder_features, cyto_skip,
                              'cyto_dec', ctx=nucleus_context_at_cyto,
                              ctx_skip=cyto_skip)
        x_cytoplasm_decoder_for_refine = layers.Lambda(
            _feature_gradient, name='cyto_decoder_stop_grad_for_refine'
        )(x_cytoplasm)

        x_cyto_feat_prob = ConvLayer(
            net_conv_after_unet, 3, padding='same', activation=unet_activation,
            kernel_initializer=self._kinit, name='cyto_features_prob'
        )(x_cytoplasm)
        x_cyto_feat_dist = ConvLayer(
            net_conv_after_unet, 3, padding='same', activation=unet_activation,
            kernel_initializer=self._kinit, name='cyto_features_dist'
        )(x_cytoplasm)

        _prob_logit_cyto = ConvLayer(
            1, 1, padding='same', activation=None,
            kernel_initializer='glorot_uniform', name='cytoplasm_prob_logit'
        )(x_cyto_feat_prob)
        cytoplasm_prob = layers.Activation('sigmoid',
                                           name='cytoplasm_prob')(_prob_logit_cyto)
        cytoplasm_dist = _dist_head(x_cyto_feat_dist, 'cytoplasm_dist',
                                    'dist_bias_init_cytoplasm',
                                    'median_max_dist_cytoplasm', 30.0)
        print(f"  ✓ Cytoplasm head: split prob/dist features, dist={_dist_act}")

        # Cell refinement uses the same configurable decoder-gradient path.
        _ctx_for_refine = layers.Lambda(
            lambda t: tf.stop_gradient(t), name='cyto_ctx_stop_grad_for_refine'
        )(nucleus_context_at_cyto)
        _gp_c, _gd_c = _geo_prior(
            cytoplasm_prob, cytoplasm_dist,
            getattr(cfg, 'median_max_dist_cytoplasm', 30.0) or 30.0, 'cyto')

        _cyto_gate_l = ConvLayer(int(x_cytoplasm_decoder_for_refine.shape[-1]), 1,
                                 padding='same', activation='sigmoid',
                                 kernel_initializer='glorot_uniform',
                                 name='cyto_prompt_gate')
        _cyto_c1 = ConvLayer(mask_n_filter, 3, padding='same', activation='relu',
                             kernel_initializer='he_normal',
                             name='cyto_refine_conv1')
        _cyto_c2 = ConvLayer(mask_n_filter, 3, padding='same', activation='relu',
                             kernel_initializer='he_normal',
                             name='cyto_refine_conv2')
        _cyto_out = ConvLayer(3, 3, padding='same', activation=None,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros', name='cytoplasm_mask_out')

        _cyto_logits = []
        _cyto_hidden = []
        _cyto_body = None
        _cyto_quality = []
        if cfg.mask_quality_head:
            _cq_pool = (MaskQualityPool if cfg.mask_quality_pooling == 'mask' else _global_pool)(name='cyto_refine_quality_pool')
            _cq_dense = layers.Dense(1, activation='sigmoid', name='cyto_refine_quality')
        for k in range(n_pr):
            _mod = layers.Multiply(name=f'cyto_prompt_mod_{k}')(
                [x_cytoplasm_decoder_for_refine, _cyto_gate_l(p_at_cyto[k])])
            _in = layers.Concatenate(name=f'cyto_refine_concat_{k}')(
                [_mod, _ctx_for_refine, p_at_cyto[k], _gp_c, _gd_c])
            if cfg.refinement_depth:
                if _cyto_body is None:
                    _cyto_body = _multiscale_refiner(int(_in.shape[-1]), 'cyto', 'cytoplasm')
                hidden = _cyto_body(_in)
            else:
                hidden = _cyto_c2(_cyto_c1(_in))
            _cyto_hidden.append(hidden)
            _cyto_logits.append(_cyto_out(hidden))
            if cfg.mask_quality_head:
                pooled = _cq_pool([hidden, _cyto_logits[-1]]) if cfg.mask_quality_pooling == 'mask' else _cq_pool(hidden)
                _cyto_quality.append(_cq_dense(pooled))

        _cyto_mask_logit = _stack_prompts(_cyto_logits, 'cytoplasm_mask_logit')
        cytoplasm_mask = layers.Lambda(
            lambda t: tf.nn.softmax(t, axis=-1)[..., 1:2], name='cytoplasm_mask'
        )(_cyto_mask_logit)
        print(f"  ✓ Cytoplasm refinement: 3-class prompted mask ({mask_n_filter} ch) "
              f"+ geometry prior ({_n_geo} ch), промптов за форвард: {n_pr}")

        outputs = {
            'nucleus_prob': nucleus_prob,
            'nucleus_dist': nucleus_dist,
            'nucleus_mask': nucleus_mask,
            'cytoplasm_prob': cytoplasm_prob,
            'cytoplasm_dist': cytoplasm_dist,
            'cytoplasm_mask': cytoplasm_mask,
            'nucleus_prob_logit': _prob_logit_nuc,
            'cytoplasm_prob_logit': _prob_logit_cyto,
            'nucleus_mask_logit': _nuc_mask_logit,
            'cytoplasm_mask_logit': _cyto_mask_logit,
        }
        if cfg.mask_quality_head:
            outputs['nucleus_quality'] = _stack_prompts(_nuc_quality, 'nucleus_quality')
            outputs['cytoplasm_quality'] = _stack_prompts(_cyto_quality, 'cytoplasm_quality')
        if cfg.semantic_foreground_weight:
            # Binary foreground of each head, read linearly from the decoder features the
            # refinement consumes. prob (soft EDT, centre-weighted) and dist (EDT-weighted
            # rays) give these features almost no boundary signal; this head does, with
            # full gradient (Panoptic-DeepLab semantic branch, HoVer-Net NP branch).
            # Created last: optimizer slots are restored by variable index.
            for head, feat in (('nucleus', x_nucleus), ('cytoplasm', x_cytoplasm)):
                outputs[head + '_semantic_logit'] = ConvLayer(
                    1, 1, padding='same', activation=None, kernel_initializer='glorot_uniform',
                    name=head + '_semantic_logit')(feat)
        fine_runtime = {}
        if cfg.refinement_full_resolution:
            # Full-resolution residual on the linearly upsampled coarse mask (cytonucs_fine_mask).
            # Memory: no gradient into the full-resolution backbone features, projection only
            # inside the window, and one prompt slot per crop in training (random, so detector-
            # derived prompts are seen too; shared weights serve every prompt at inference).
            # Created after all other layers.
            from cytonucs_fine_mask import training_window, PromptWindowBegin, FineWindow, PromptSlot
            fine_source = layers.Lambda(tf.stop_gradient, name='fine_features_stop_gradient')(full_res_features)
            self.fine_head_models = {}
            if self.fine_wiring == 'slot0':
                picked = [_nuc_logits[0], _nuc_hidden[0], prompt_nuc_full[0],
                          _cyto_logits[0], _cyto_hidden[0], prompt_cyto_full[0]]
            else:
                *picked, fine_slot = PromptSlot(n_pr, name='fine_prompt_slot')(
                    _nuc_logits + _nuc_hidden + prompt_nuc_full + _cyto_logits + _cyto_hidden + prompt_cyto_full)
                outputs['fine_prompt_slot'] = fine_slot
            for head, tag, logits_k, hidden_k, prompt_k, hidden_runtime, grid in (
                    ('nucleus', 'nuc', picked[0], picked[1], picked[2], _nuc_hidden[0], nucleus_grid),
                    ('cytoplasm', 'cyto', picked[3], picked[4], picked[5], _cyto_hidden[0], cytoplasm_grid)):
                window = training_window(cfg, head)
                hidden_proj = ConvLayer(FINE_CHANNELS, 1, padding='same', activation='relu',
                                        kernel_initializer='he_normal', name=f'{tag}_fine_hidden_proj')
                head_model = _fine_head(ConvLayer, self.ndim, grid, f'{tag}_fine_head',
                                        int(full_res_features.shape[-1]), n_img_ch)
                self.fine_head_models[head] = head_model
                hidden_train = hidden_proj(hidden_k)
                begin = PromptWindowBegin(window, grid, name=f'{tag}_fine_begin')(prompt_k)
                windowed = FineWindow(window, grid, name=f'{tag}_fine_window')(
                    [logits_k, hidden_train, fine_source, inputs, prompt_k, begin])
                outputs[head + '_fine_logit'] = head_model(windowed)
                outputs[head + '_fine_begin'] = begin
                fine_runtime[head] = hidden_train if self.fine_wiring == 'slot0' else hidden_proj(hidden_runtime)
        # Reuse the exact trained layers; no additional weights or dummy prompts.
        feature_tensors = [x_nucleus_for_refine, x_cytoplasm_decoder_for_refine,
                           _ctx_for_refine, _gp_n, _gd_n, _gp_c, _gd_c]
        self.refinement_features_model = Model(inputs=inputs, outputs=feature_tensors,
                                               name='refinement_features')
        self.refinement_model = Model(
            inputs=feature_tensors + [prompt_nuc_full[0], prompt_cyto_full[0]],
            outputs=[_nuc_logits[0], _cyto_logits[0]], name='refinement_runtime')
        if cfg.mask_quality_head:
            self.refinement_with_quality_model = Model(
                inputs=feature_tensors + [prompt_nuc_full[0], prompt_cyto_full[0]],
                outputs=[_nuc_logits[0], _cyto_logits[0], _nuc_quality[0], _cyto_quality[0]])
        # The cell head uses shared nuclear context, but no nuclear mask decoder.
        self.cytoplasm_refinement_model = Model(
            inputs=[feature_tensors[i] for i in (1, 2, 5, 6)] + [prompt_cyto_full[0]],
            outputs=[_cyto_logits[0], _cyto_quality[0]] if cfg.mask_quality_head else _cyto_logits[0],
            name='cytoplasm_refinement_runtime')
        if fine_runtime:
            # Inference: full-resolution features once per ROI; coarse logits, quality and the
            # projected refinement features per prompt; the fine head per window.
            self.fine_features_model = Model(inputs=inputs, outputs=fine_source, name='fine_features_runtime')
            quality = [_nuc_quality[0], _cyto_quality[0]] if cfg.mask_quality_head else []
            self.refinement_fine_model = Model(
                inputs=feature_tensors + [prompt_nuc_full[0], prompt_cyto_full[0]],
                outputs=[_nuc_logits[0], _cyto_logits[0]] + quality + [fine_runtime['nucleus'], fine_runtime['cytoplasm']],
                name='refinement_fine_runtime')
            self.cytoplasm_fine_model = Model(
                inputs=[feature_tensors[i] for i in (1, 2, 5, 6)] + [prompt_cyto_full[0]],
                outputs=[_cyto_logits[0]] + quality[1:] + [fine_runtime['cytoplasm']],
                name='cytoplasm_fine_runtime')
        return Model(inputs=inputs_all, outputs=outputs, name=self.name)

    def summary(self):
        self.keras_model.summary()

    BY_NAME = 'cytonucs_by_name'

    def save_weights(self, filepath):
        self.keras_model.save_weights(filepath)
        import json
        import h5py
        from pathlib import Path
        # Keras stores layers under generic names in graph order; a copy keyed by the layer names
        # keeps the checkpoint loadable after graph changes that reorder layers.
        with h5py.File(filepath, 'a') as store:
            if self.BY_NAME in store:
                del store[self.BY_NAME]
            group = store.create_group(self.BY_NAME)
            for path, layer in self._weighted_layers():
                for index, value in enumerate(layer.get_weights()):
                    group.create_dataset(f'{path}/{index}', data=value)
        Path(str(filepath) + '.architecture.json').write_text(
            json.dumps(self.architecture_signature(), indent=2), encoding='utf-8')

    def _weighted_layers(self):
        """(name path, layer) of every layer owning weights, nested models expanded."""
        def walk(model, prefix):
            for layer in model.layers:
                if isinstance(layer, Model):
                    yield from walk(layer, prefix + layer.name + '/')
                elif layer.weights:
                    yield prefix + layer.name, layer
        return list(walk(self.keras_model, ''))

    def _load_by_name(self, filepath, allowed_missing):
        import h5py
        missing = []
        with h5py.File(filepath, 'r') as store:
            group = store[self.BY_NAME]
            for path, layer in self._weighted_layers():
                if f'{path}/0' not in group:
                    missing.append(path)
                    continue
                values = [group[f'{path}/{i}'][()] for i in range(len(layer.weights))]
                if any(v.shape != tuple(w.shape) for v, w in zip(values, layer.weights)):
                    raise ValueError(f'Checkpoint weights of {path} have another shape')
                layer.set_weights(values)
        if any(not allowed_missing(path) for path in missing):
            raise ValueError(f'Layers absent from the checkpoint: {missing}')
        if missing:
            print(f'New heads added: checkpoint weights restored by name, {missing} initialized')

    def architecture_signature(self):
        import json
        keys = ('ndim', 'n_rays', 'n_channel_in', 'shared_encoder', 'unet_n_depth',
                'unet_n_filter_base', 'net_conv_after_unet', 'mask_refinement_filters',
                'unet_pool_schedule', 'unet_kernel_schedule', 'refinement_depth', 'mask_quality_head',
                'train_patch_size', 'nucleus_grid', 'cytoplasm_grid')
        result = {k:getattr(self.config, k) for k in keys}
        result['unet_activation'] = getattr(self.config, 'unet_activation', 'relu')
        result['dist_activation'] = getattr(self.config, 'dist_activation', 'linear')
        result['mask_quality_pooling'] = self.config.mask_quality_pooling
        result['refinement_pool'] = self.config.refinement_pool
        result['semantic_foreground_head'] = bool(self.config.semantic_foreground_weight)
        result['refinement_full_resolution'] = bool(self.config.refinement_full_resolution)
        if self.config.refinement_full_resolution:
            result['fine_wiring'] = self.fine_wiring
        if getattr(self.config, 'unet_norm_groups', 0):
            result['unet_norm_groups'] = int(self.config.unet_norm_groups)
        return json.loads(json.dumps(result))

    # Order of the added heads in every checkpoint saved so far (v3.1 semantic, v3.2 fine).
    SEMANTIC_ORDER = ('cytoplasm_semantic_logit', 'nucleus_semantic_logit')
    FINE_ORDER = ('cyto_fine_hidden_proj', 'nuc_fine_hidden_proj',
                  'cyto_fine_head_feature_proj', 'cyto_fine_head_c1', 'cyto_fine_head_c2', 'cyto_fine_head_delta',
                  'nuc_fine_head_feature_proj', 'nuc_fine_head_c1', 'nuc_fine_head_c2', 'nuc_fine_head_delta')

    def optimizer_variables(self):
        """Trainable variables in the canonical optimizer order: the base network in the Keras order
        of the graph without the added heads, then the semantic head, then the fine head.

        Checkpoint state restores model variables and Adam slots by index. Keras orders variables by
        graph traversal, which new layers can reshuffle (the fine prompt slot did), so the base
        order comes from a model built without the heads and the heads use fixed name orders."""
        variables = self.keras_model.trainable_variables
        path = lambda v: getattr(v, 'path', None) or v.name
        layer = lambda v: path(v).split('/')[0]
        semantic = [v for v in variables if layer(v) in self.SEMANTIC_ORDER]
        fine = [v for v in variables if '_fine_' in path(v)]
        if not semantic and not fine:
            return list(variables)
        by_path = {path(v): v for v in variables}
        base = [by_path[p] for p in self._base_variable_paths()]
        if len(base) + len(semantic) + len(fine) != len(variables):
            raise ValueError('Unexpected trainable variables outside the base network and the added heads')
        rank = lambda order: (lambda v: (order.index(layer(v)), path(v).endswith('/bias')))
        if any(layer(v) not in self.FINE_ORDER for v in fine):
            raise ValueError(f'Fine-head variable outside FINE_ORDER: {[path(v) for v in fine]}')
        return base + sorted(semantic, key=rank(self.SEMANTIC_ORDER)) + sorted(fine, key=rank(self.FINE_ORDER))

    def _base_variable_paths(self):
        if not hasattr(self, '_cached_base_paths'):
            import copy, contextlib, io
            config = copy.copy(self.config)
            config.semantic_foreground_weight = 0.0
            config.refinement_full_resolution = False
            with contextlib.redirect_stdout(io.StringIO()):
                base = CytoNucsStarDistModel(config, name=self.name, basedir=self.basedir)
            self._cached_base_paths = [getattr(v, 'path', None) or v.name for v in base.keras_model.trainable_variables]
            del base
        return self._cached_base_paths

    def load_weights(self, filepath):
        import json
        from pathlib import Path
        signature = Path(str(filepath) + '.architecture.json')
        if signature.exists():
            saved = json.loads(signature.read_text(encoding='utf-8'))
            current = self.architecture_signature()
            # Pooling has no variables; the Dense weights and Adam slots are identical.
            # Old v2 sidecars predate this field and mean global pooling.
            old_pool = saved.pop('mask_quality_pooling', 'global')
            new_pool = current.pop('mask_quality_pooling')
            # Refiner pooling has no weights either; sidecars before v2.13 mean isotropic.
            old_refiner = saved.pop('refinement_pool', 'isotropic')
            new_refiner = current.pop('refinement_pool')
            # Sidecars before v3.1 have no semantic head, before v3.2 no full-resolution head.
            old_semantic = saved.pop('semantic_foreground_head', False)
            new_semantic = current.pop('semantic_foreground_head')
            old_fine = saved.pop('refinement_full_resolution', False)
            new_fine = current.pop('refinement_full_resolution')
            # v3.2 sidecars have no wiring field: their fine head was wired to the first prompt.
            old_wiring = saved.pop('fine_wiring', 'slot0' if old_fine else None)
            current.pop('fine_wiring', None)
            if saved != current or (old_semantic and not new_semantic) or (old_fine and not new_fine):
                raise ValueError('Checkpoint architecture differs from config; use its original config or start a new run')
            if old_pool != new_pool and self.config.mask_quality_head:
                print(f'Quality pooling: {old_pool} -> {new_pool}; restoring compatible weights, quality requires recalibration')
            if old_refiner != new_refiner and self.config.refinement_depth:
                print(f'Refiner pooling: {old_refiner} -> {new_refiner}; restoring compatible weights, '
                      'refinement requires fine-tuning')
            import h5py
            with h5py.File(filepath, 'r') as store:
                by_name = self.BY_NAME in store
            head_layer = lambda path: path.split('/')[0]
            if by_name:
                self._load_by_name(filepath, lambda path: (
                    (new_semantic and not old_semantic and head_layer(path).endswith('_semantic_logit')) or
                    (new_fine and not old_fine and '_fine_' in path)))
                self._read_calibration(filepath)
                return
            if (new_semantic and not old_semantic) or (new_fine and not old_fine) or \
                    (old_fine and old_wiring != self.fine_wiring):
                self._load_via_saved_graph(filepath, old_semantic, old_fine, old_wiring)
                return
        elif (self.config.unet_pool_schedule is not None or self.config.unet_kernel_schedule is not None
              or self.config.refinement_depth or self.config.mask_quality_head or not self.config.shared_encoder
              or self.config.refinement_pool != 'isotropic'):
            raise ValueError('Architecture metadata missing for non-legacy checkpoint; cannot safely resume changed architecture')
        self.keras_model.load_weights(filepath)
        self._read_calibration(filepath)

    def _read_calibration(self, filepath):
        self.loaded_weights_path = str(filepath)
        from cytonucs_runtime import read_quality_gates, read_mask_thresholds
        self.loaded_quality_gates = read_quality_gates(filepath)
        self.loaded_mask_thresholds = read_mask_thresholds(filepath)

    def _load_via_saved_graph(self, filepath, semantic, fine, wiring):
        """Checkpoint without layer-name keys whose graph differs from this model (older heads or
        wiring): load it into a model built with the saved graph, copy every weighted layer by
        name; heads the checkpoint does not have keep their initialisation."""
        import copy
        legacy_config = copy.copy(self.config)
        if not semantic:
            legacy_config.semantic_foreground_weight = 0.0
        if not fine:
            legacy_config.refinement_full_resolution = False
        legacy = CytoNucsStarDistModel(legacy_config, name=self.name, basedir=self.basedir,
                                       fine_wiring=wiring or 'random_slot')
        legacy.keras_model.load_weights(filepath)
        copied = set()
        for layer in legacy.keras_model.layers:
            if layer.weights:
                self.keras_model.get_layer(layer.name).set_weights(layer.get_weights())
                copied.add(layer.name)
        missing = [l.name for l in self.keras_model.layers if l.weights and l.name not in copied]
        allowed = lambda name: (not semantic and name.endswith('_semantic_logit')) or (not fine and '_fine_' in name)
        if any(not allowed(name) for name in missing):
            raise ValueError(f'Layers absent from the checkpoint: {missing}')
        print(f'Checkpoint loaded through its saved graph (semantic={semantic}, full_resolution={fine}, '
              f'wiring={wiring}); weights copied by name, new layers initialized: {missing}')
        self._read_calibration(filepath)


def build_cytonucs_model(config, name='cytonucs_stardist', basedir='models'):
    return CytoNucsStarDistModel(config, name=name, basedir=basedir)
