import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np


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


class CytoNucsStarDistModel:

    def __init__(self, config, name='cytonucs_stardist', basedir='models'):
        self.config = config
        self.name = name
        self.basedir = basedir
        self.ndim = config.ndim
        self.keras_model = self._build_model()

    def _build_block(self, x, n_filters, ConvLayer, prefix):
        """UNet conv block: 2x Conv3x3 ReLU"""
        x = ConvLayer(n_filters, 3, padding='same', activation='relu',
                      kernel_initializer='glorot_uniform',
                      name=f'{prefix}_c1')(x)
        x = ConvLayer(n_filters, 3, padding='same', activation='relu',
                      kernel_initializer='glorot_uniform',
                      name=f'{prefix}_c2')(x)
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

        _aniso = tuple(getattr(cfg, 'anisotropy', (1,) * self.ndim))
        if len(_aniso) != self.ndim:
            _aniso = (1,) * self.ndim
        _min_aniso = min(_aniso)
        pool_size = tuple(2 if a < 2.0 * _min_aniso else 1 for a in _aniso)
        print(f"  ✓ Pool/upsample size (anisotropy {_aniso}): {pool_size}")

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
        # GRID КАК В КАНОНЕ: вход прореживается ДО UNet (model3d.py:365-375),
        # и весь UNet строится ПОВЕРХ прореженного входа — симметрично, все
        # unet_n_depth уровней лежат НИЖЕ grid-разрешения. Прежняя схема
        # (энкодер от full-res, декодер обрывается на уровне skip) давала
        # рецептивное поле вдвое меньше при большей стоимости.
        #
        # base_grid = поэлементный минимум гридов голов: до него прореживаем
        # общий ствол, дальше каждая голова останавливает свой декодер на
        # СВОЁМ уровне (если гриды различаются).
        # ============================================================
        base_grid = tuple(min(n, c) for n, c in zip(nucleus_grid, cytoplasm_grid))

        def _levels_between(coarse, fine):
            """Сколько раз надо применить pool_size, чтобы из fine получить coarse."""
            n = 0
            cur = list(fine)
            while tuple(cur) != tuple(coarse):
                nxt = [cur[a] * pool_size[a] if coarse[a] > cur[a] else cur[a]
                       for a in range(self.ndim)]
                if nxt == cur:
                    raise ValueError(f"grid {coarse} недостижим из {fine} "
                                     f"шагами {pool_size}")
                cur = nxt
                n += 1
            return n

        n_pool_base = _levels_between(base_grid, (1,) * self.ndim)
        nuc_skip = _levels_between(nucleus_grid, base_grid)
        cyto_skip = _levels_between(cytoplasm_grid, base_grid)
        print(f"  ✓ Canonical grid — base {base_grid} ({n_pool_base} пулингов входа), "
              f"nucleus {nucleus_grid} (skip {nuc_skip}), "
              f"cytoplasm {cytoplasm_grid} (skip {cyto_skip})")

        # ============================================================
        # ВХОД
        # ============================================================
        n_img_ch = int(cfg.n_channel_in)
        input_shape = (None,) * self.ndim + (n_img_ch + 2,)
        inputs_all = layers.Input(shape=input_shape, name='input')
        inputs = layers.Lambda(lambda t: t[..., :n_img_ch], name='img_channels')(inputs_all)
        prompt_nuc_full = layers.Lambda(lambda t: t[..., n_img_ch:n_img_ch + 1],
                                        name='prompt_nuc')(inputs_all)
        prompt_cyto_full = layers.Lambda(lambda t: t[..., n_img_ch + 1:n_img_ch + 2],
                                         name='prompt_cyto')(inputs_all)

        _AvgPool = layers.AveragePooling3D if self.ndim == 3 else layers.AveragePooling2D

        def _prompt_with_coords(p_full, n_pool, tag, grid):
            """Промпт + ОТНОСИТЕЛЬНЫЕ КООРДИНАТЫ до его центра.
            Рефайн-ветка это несколько conv3, её собственное рецептивное поле
            мало по сравнению с объектом, поэтому воксель на краю не видит
            гауссиану в центре. Смещение до центра он знает локально."""
            p = p_full
            for _i in range(n_pool):
                p = _AvgPool(pool_size, name=f'prompt_pool_{tag}_{_i}')(p)

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

        p_at_nuc = _prompt_with_coords(prompt_nuc_full, n_pool_base + nuc_skip,
                                       'nuc', nucleus_grid)
        p_at_cyto = _prompt_with_coords(prompt_cyto_full, n_pool_base + cyto_skip,
                                        'cyto', cytoplasm_grid)

        # ============================================================
        # ПРОРЕЖИВАНИЕ ВХОДА ДО base_grid (канон model3d.py:365-375)
        # ============================================================
        pooled = inputs
        for i in range(n_pool_base):
            pooled = ConvLayer(unet_n_filter_base, 3, padding='same',
                               activation=unet_activation,
                               kernel_initializer='glorot_uniform',
                               name=f'pre_pool{i}_c1')(pooled)
            pooled = ConvLayer(unet_n_filter_base, 3, padding='same',
                               activation=unet_activation,
                               kernel_initializer='glorot_uniform',
                               name=f'pre_pool{i}_c2')(pooled)
            pooled = PoolLayer(pool_size, name=f'pre_pool{i}')(pooled)

        # ============================================================
        # ENCODER (общий) — ПОВЕРХ прореженного входа
        # ============================================================
        encoder_features = []
        x = pooled
        for i in range(unet_n_depth + 1):
            n_filters = unet_n_filter_base * (2 ** i)
            x = self._build_block(x, n_filters, ConvLayer, prefix=f'enc_l{i}')
            encoder_features.append(x)
            if i < unet_n_depth:
                x = PoolLayer(pool_size, name=f'enc_l{i}_pool')(x)

        def _decode(bottleneck, enc_feats, skip, prefix, ctx=None, ctx_skip=0):
            """Симметричный декодер: с уровня unet_n_depth обратно до `skip`.
            При равных гридах голов skip=0, то есть декодер возвращается ровно
            на base_grid — как в каноне. ctx подмешивается на каждом уровне."""
            x = bottleneck
            for i in reversed(range(skip, unet_n_depth)):
                n_filters = unet_n_filter_base * (2 ** i)
                x = UpSampleLayer(pool_size, name=f'{prefix}_l{i}_up')(x)
                concat_list = [x, enc_feats[i]]
                if ctx is not None:
                    c = ctx
                    for _ in range(i - ctx_skip):
                        c = PoolLayer(pool_size)(c)
                    concat_list.append(c)
                x = layers.Concatenate(name=f'{prefix}_l{i}_concat')(concat_list)
                x = self._build_block(x, n_filters, ConvLayer, prefix=f'{prefix}_l{i}')
            return x

        # ============================================================
        # ГОЛОВЫ prob/dist. Отклонение от канона (model3d.py:379-386, где обе
        # растут из ОДНОГО features): слои раздельные, dist по умолчанию
        # softplus с bias = медиана длин лучей.
        # Причина: у канона лучи 15-30 px, здесь у цитоплазмы 110-160. Чтобы
        # 96-канальная dist-голова с glorot-весами выдала нужный сдвиг, признаки
        # обязаны иметь амплитуду в десятки, а 1-канальная prob-голова на тех же
        # признаках получает логит порядка сотни и насыщается. Линейный dist с
        # нулевым bias при таких целях доезжает до медианы за ~170 эпох из 200.
        # dist_activation='linear' в конфиге возвращает канонический вариант.
        #
        # prob отдаётся И логитом, И вероятностью: BCE считается из логита
        # (sigmoid_cross_entropy_with_logits), где производная (sigmoid(z) - t)
        # не обнуляется при насыщении. Прежний путь через clip_by_value давал
        # РОВНО НОЛЬ градиента ниже 1e-7, и голова замерзала необратимо.
        # ============================================================
        _dist_act = str(getattr(cfg, 'dist_activation', 'softplus')).lower()
        _n_geo = int(getattr(cfg, 'refine_geom_channels', 8))

        def _dist_head(feat, name, bias_key, fallback_key, default):
            if _dist_act == 'softplus':
                b = float(getattr(cfg, bias_key, None) or
                          getattr(cfg, fallback_key, default))
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
            геометрию. dist нормируется на масштаб объекта (лучи до ~174 px
            против O(1) у остальных входов) и сжимается с n_rays до _n_geo:
            96 сырых каналов забили бы вход числом каналов."""
            _p = layers.Lambda(lambda t: tf.stop_gradient(t),
                               name=f'{tag}_geo_prob_sg')(prob_t)
            _s = max(float(scale), 1e-3)
            _d = layers.Lambda(lambda t, _k=_s: tf.stop_gradient(t) / _k,
                               name=f'{tag}_geo_dist_sg')(dist_t)
            _d = ConvLayer(_n_geo, 1, padding='same', activation=unet_activation,
                           kernel_initializer='glorot_uniform',
                           name=f'{tag}_geo_dist_compress')(_d)
            return _p, _d

        # ============================================================
        # DECODER 1: NUCLEUS
        # ============================================================
        x_nucleus = _decode(encoder_features[-1], encoder_features, nuc_skip, 'nuc_dec')

        x_nuc_feat_prob = ConvLayer(
            net_conv_after_unet, 3, padding='same', activation=unet_activation,
            kernel_initializer='glorot_uniform', name='nuc_features_prob'
        )(x_nucleus)
        x_nuc_feat_dist = ConvLayer(
            net_conv_after_unet, 3, padding='same', activation=unet_activation,
            kernel_initializer='glorot_uniform', name='nuc_features_dist'
        )(x_nucleus)

        _prob_logit_nuc = ConvLayer(
            1, 1, padding='same', activation=None,
            kernel_initializer='glorot_uniform', name='nucleus_prob_logit'
        )(x_nuc_feat_prob)
        nucleus_prob = layers.Activation('sigmoid', name='nucleus_prob')(_prob_logit_nuc)
        nucleus_dist = _dist_head(x_nuc_feat_dist, 'nucleus_dist',
                                  'dist_bias_init_nucleus',
                                  'median_max_dist_nucleus', 15.0)
        print(f"  ✓ Nucleus head: split prob/dist features, dist={_dist_act}")

        # РЕФАЙН ЯДРА — за stop_gradient, backbone не трогает
        x_nucleus_for_refine = layers.Lambda(
            lambda t: tf.stop_gradient(t), name='nuc_decoder_stop_grad_for_refine'
        )(x_nucleus)
        _pn_gate = ConvLayer(int(x_nucleus_for_refine.shape[-1]), 1, padding='same',
                             activation='sigmoid', name='nuc_prompt_gate')(p_at_nuc)
        _xn_mod = layers.Multiply(name='nuc_prompt_mod')(
            [x_nucleus_for_refine, _pn_gate])
        _gp_n, _gd_n = _geo_prior(
            nucleus_prob, nucleus_dist,
            getattr(cfg, 'median_max_dist_nucleus', 15.0) or 15.0, 'nuc')
        x_nuc_refine_in = layers.Concatenate(name='nuc_refine_concat')(
            [_xn_mod, p_at_nuc, _gp_n, _gd_n])
        x_nucleus_refine = ConvLayer(
            mask_n_filter, 3, padding='same', activation='relu',
            kernel_initializer='he_normal', name='nuc_refine_conv1'
        )(x_nuc_refine_in)
        x_nucleus_refine = ConvLayer(
            mask_n_filter, 3, padding='same', activation='relu',
            kernel_initializer='he_normal', name='nuc_refine_conv2'
        )(x_nucleus_refine)
        _nuc_mask_logit = ConvLayer(
            3, 3, padding='same', activation=None,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            name='nucleus_mask_logit'
        )(x_nucleus_refine)
        nucleus_mask = layers.Lambda(
            lambda t: tf.nn.softmax(t, axis=-1)[..., 1:2], name='nucleus_mask'
        )(_nuc_mask_logit)
        print(f"  ✓ Nucleus refinement: 3-class prompted mask ({mask_n_filter} ch) "
              f"+ geometry prior ({_n_geo} ch)")

        # ============================================================
        # КОНТЕКСТ ЯДРА -> цитоплазма. Только в ДЕКОДЕР, не в головы и не в
        # рефайн: рядом с головами тот же 96-канальный dist переписывал бы
        # веса, задающие вход всего декодера. Вход контекста застопорен,
        # поэтому декодер ядра от цитоплазмы не зависит.
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

        _ctx_delta = cyto_skip - nuc_skip
        nucleus_context_at_cyto = nucleus_context_gated
        for _ in range(max(0, _ctx_delta)):
            nucleus_context_at_cyto = PoolLayer(pool_size)(nucleus_context_at_cyto)
        for _ in range(max(0, -_ctx_delta)):
            nucleus_context_at_cyto = UpSampleLayer(pool_size)(nucleus_context_at_cyto)

        # ============================================================
        # DECODER 2: CYTOPLASM
        # ctx подаём УЖЕ приведённым к цито-разрешению и с соответствующим
        # ctx_skip: при nuc_skip > cyto_skip прежний вариант давал
        # range(отрицательное) -> пулинга нет -> Concatenate разных разрешений.
        # ============================================================
        x_cytoplasm = _decode(encoder_features[-1], encoder_features, cyto_skip,
                              'cyto_dec', ctx=nucleus_context_at_cyto,
                              ctx_skip=cyto_skip)
        x_cytoplasm_decoder_for_refine = layers.Lambda(
            lambda t: tf.stop_gradient(t), name='cyto_decoder_stop_grad_for_refine'
        )(x_cytoplasm)

        x_cyto_feat_prob = ConvLayer(
            net_conv_after_unet, 3, padding='same', activation=unet_activation,
            kernel_initializer='glorot_uniform', name='cyto_features_prob'
        )(x_cytoplasm)
        x_cyto_feat_dist = ConvLayer(
            net_conv_after_unet, 3, padding='same', activation=unet_activation,
            kernel_initializer='glorot_uniform', name='cyto_features_dist'
        )(x_cytoplasm)

        _prob_logit_cyto = ConvLayer(
            1, 1, padding='same', activation=None,
            kernel_initializer='glorot_uniform', name='cytoplasm_prob_logit'
        )(x_cyto_feat_prob)
        cytoplasm_prob = layers.Activation('sigmoid', name='cytoplasm_prob')(_prob_logit_cyto)
        cytoplasm_dist = _dist_head(x_cyto_feat_dist, 'cytoplasm_dist',
                                    'dist_bias_init_cytoplasm',
                                    'median_max_dist_cytoplasm', 30.0)
        print(f"  ✓ Cytoplasm head: split prob/dist features, dist={_dist_act}")

        # РЕФАЙН ЦИТОПЛАЗМЫ — за stop_gradient по всем входам
        _pc_gate = ConvLayer(int(x_cytoplasm_decoder_for_refine.shape[-1]), 1,
                             padding='same', activation='sigmoid',
                             name='cyto_prompt_gate')(p_at_cyto)
        _xc_mod = layers.Multiply(name='cyto_prompt_mod')(
            [x_cytoplasm_decoder_for_refine, _pc_gate])
        _ctx_for_refine = layers.Lambda(
            lambda t: tf.stop_gradient(t), name='cyto_ctx_stop_grad_for_refine'
        )(nucleus_context_at_cyto)
        _gp_c, _gd_c = _geo_prior(
            cytoplasm_prob, cytoplasm_dist,
            getattr(cfg, 'median_max_dist_cytoplasm', 30.0) or 30.0, 'cyto')
        x_cytoplasm_refine_in = layers.Concatenate(name='cyto_refine_concat')(
            [_xc_mod, _ctx_for_refine, p_at_cyto, _gp_c, _gd_c])
        x_cytoplasm_refine = ConvLayer(
            mask_n_filter, 3, padding='same', activation='relu',
            kernel_initializer='he_normal', name='cyto_refine_conv1'
        )(x_cytoplasm_refine_in)
        x_cytoplasm_refine = ConvLayer(
            mask_n_filter, 3, padding='same', activation='relu',
            kernel_initializer='he_normal', name='cyto_refine_conv2'
        )(x_cytoplasm_refine)
        _cyto_mask_logit = ConvLayer(
            3, 3, padding='same', activation=None,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            name='cytoplasm_mask_logit'
        )(x_cytoplasm_refine)
        cytoplasm_mask = layers.Lambda(
            lambda t: tf.nn.softmax(t, axis=-1)[..., 1:2], name='cytoplasm_mask'
        )(_cyto_mask_logit)
        print(f"  ✓ Cytoplasm refinement: 3-class prompted mask ({mask_n_filter} ch) "
              f"+ geometry prior ({_n_geo} ch)")

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
        return Model(inputs=inputs_all, outputs=outputs, name=self.name)

    def summary(self):
        self.keras_model.summary()

    def save_weights(self, filepath):
        self.keras_model.save_weights(filepath)

    def load_weights(self, filepath):
        self.keras_model.load_weights(filepath)



def build_cytonucs_model(config, name='cytonucs_stardist', basedir='models'):
    return CytoNucsStarDistModel(config, name=name, basedir=basedir)