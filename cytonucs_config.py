"""
CytoNucs StarDist Configuration
Unified config for 2D/3D cytoplasm + nucleus segmentation
"""
from stardist.models import Config2D, Config3D

import numpy as np
import json
from pathlib import Path


class CytoNucsConfig:
    """
    Configuration for CytoNucs StarDist model.

    Key features:
    - Dual decoders: one for nuclei, one for cytoplasm
    - Multi-nuclear support: cells can contain multiple nuclei
    - Additional loss components: containment, WBR, consistency (optional)
    - Per-head ray sets and grids, derived canonically from object extents
    """

    def __init__(
            self,
            ndim=3,  # 2 or 3
            n_rays=128,
            grid=(1, 1),  # subsampling grid
            anisotropy=None,
            n_channel_in=1,
            unet_n_depth=4,
            unet_n_filter_base=32,
            net_conv_after_unet=128,

            # Architecture

            shared_encoder=True,

            # Training patches
            train_patch_size=(512, 512),  # (Z,Y,X) for 3D or (Y,X) for 2D
            train_batch_size=2,
            mode='train',
            fg_crop_prob=0.5,
            # Loss weights
            lambda_prob_nucleus=1.0,
            lambda_dist_nucleus=1.0,
            lambda_prob_cytoplasm=1.0,
            lambda_dist_cytoplasm=1.0,
            lambda_containment=0.5,
            w_containment_geometric=1.0,  # nucleus inside cell
            w_containment_distance=0.5,
            w_center_separation=0.2,
            w_cyto_support=0.1,
            lambda_wbr=0.3,  # within-boundary regularization
            lambda_variance=0.5,
            lambda_consistency=0.0,
            enable_consistency=False,  # explicitly enable/disable
            lambda_area_cytoplasm=0.1,
            lambda_fragmentation=0.0,
            lambda_background_penalty=0.0,

            # Containment loss params
            containment_margin=2.0,  # allow small margin (voxels)
            max_assignment_distance=100.0,  # max distance for nucleus-cell pairing
            prob_thresh_cytoplasm=0.4,
            prob_thresh_nucleus=0.5,
            nms_thresh=0.3,
            # WBR params
            wbr_sigma=2.0,  # gaussian smoothing for boundary detection
            wbr_warmup_epochs=5,
            min_eval_epoch=1,  # start evaluating after this epoch
            # Distance normalization params
            median_radius_train=None,
            scale_factor_nucleus=None,
            scale_factor_cytoplasm=None,
            median_max_dist_nucleus=None,
            median_max_dist_cytoplasm=None,
            mask_thresh_range=(0.1, 0.9),
            threshold_opt_splits="val",
            dist_loss='mae',
            huber_delta=1.0,
            # Training
            use_gpu=False,
            train_epochs=400,
            train_steps_per_epoch=100,
            train_learning_rate=3e-4,
            lr_decay_steps=700,
            lr_decay_rate=0.5,
            lr_staircase=True,
            save_every_n_epochs=50,
            gradient_clip_norm=10,
            save_pred_tiff=True,
            lambda_cyto_mask=0.0,
            lambda_nuc_mask=0.0,
            lr_warmup_epochs=0,
            dist_bias_init_nucleus=None,
            dist_bias_init_cytoplasm=None,
            mask_refinement_filters=64,
            mask_thresh=0.5,

            bbox_padding_factor=1.5,

            **kwargs
    ):
        self.ndim = ndim
        self.n_rays = n_rays
        self.unet_n_depth = unet_n_depth
        self.unet_n_filter_base = unet_n_filter_base
        self.net_conv_after_unet = net_conv_after_unet

        def _coerce_grid(g):
            g = tuple(g) if g is not None else (1,) * ndim
            return g if len(g) == ndim else (1,) * ndim

        self.grid = _coerce_grid(grid)
        self.nucleus_grid = _coerce_grid(kwargs.get('nucleus_grid', grid))
        self.cytoplasm_grid = _coerce_grid(kwargs.get('cytoplasm_grid', grid))
        self.anisotropy = anisotropy if anisotropy is not None else (1,) * ndim
        self.n_channel_in = n_channel_in
        self.gpu_config = kwargs.get('gpu_config', {
            'allow_growth': True,
            'memory_limit': None,
            'visible_devices': None
        })
        # Проверка и исправление anisotropy
        if len(self.anisotropy) != self.ndim:
            print(f"⚠️  Fixing anisotropy: {self.anisotropy} -> {(1,) * self.ndim}")
            self.anisotropy = (1,) * self.ndim

        # Architecture
        self.shared_encoder = shared_encoder
        # Patches
        self.train_patch_size = train_patch_size
        self.train_batch_size = train_batch_size
        self.gradient_clip_norm = gradient_clip_norm
        self.threshold_opt_splits = threshold_opt_splits
        self.mask_thresh_range = mask_thresh_range
        # Loss weights
        self.lambda_prob_nucleus = lambda_prob_nucleus
        self.lambda_dist_nucleus = lambda_dist_nucleus
        self.lambda_prob_cytoplasm = lambda_prob_cytoplasm
        self.lambda_dist_cytoplasm = lambda_dist_cytoplasm
        self.lambda_containment = lambda_containment
        self.lambda_wbr = lambda_wbr
        self.lambda_area_cytoplasm = lambda_area_cytoplasm
        self.lambda_variance = lambda_variance
        self.lambda_consistency = lambda_consistency if enable_consistency else 0.0
        self.enable_consistency = enable_consistency
        self.lambda_fragmentation = lambda_fragmentation
        self.lambda_background_penalty = lambda_background_penalty
        self.dist_loss = dist_loss
        self.huber_delta = huber_delta
        self.mode = mode
        self.lr_warmup_epochs = lr_warmup_epochs
        # Containment
        self.containment_margin = containment_margin
        self.w_containment_geometric = w_containment_geometric
        self.w_containment_distance = w_containment_distance
        self.w_center_separation = w_center_separation
        self.w_cyto_support = w_cyto_support
        self.max_assignment_distance = max_assignment_distance
        # Refinement params
        self.lambda_cyto_mask = lambda_cyto_mask
        self.lambda_nuc_mask = lambda_nuc_mask

        self.dist_bias_init_nucleus = dist_bias_init_nucleus
        self.dist_bias_init_cytoplasm = dist_bias_init_cytoplasm

        self.mask_refinement_filters = mask_refinement_filters
        self.mask_thresh = mask_thresh
        self.bbox_padding_factor = bbox_padding_factor

        # WBR
        self.wbr_sigma = wbr_sigma
        self.wbr_warmup_epochs = wbr_warmup_epochs

        # Distance normalization
        self.median_radius_train = median_radius_train
        self.scale_factor_nucleus = scale_factor_nucleus
        self.scale_factor_cytoplasm = scale_factor_cytoplasm
        self.median_max_dist_nucleus = median_max_dist_nucleus
        self.median_max_dist_cytoplasm = median_max_dist_cytoplasm

        # Training
        self.use_gpu = use_gpu
        self.train_epochs = train_epochs
        self.train_steps_per_epoch = train_steps_per_epoch
        self.train_learning_rate = train_learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_staircase = lr_staircase
        self.save_every_n_epochs = save_every_n_epochs
        self.min_eval_epoch = min_eval_epoch
        self.fg_crop_prob = fg_crop_prob
        # Inference thresholds
        self.prob_thresh_cytoplasm = prob_thresh_cytoplasm
        self.prob_thresh_nucleus = prob_thresh_nucleus
        self.nms_thresh = nms_thresh
        self.save_pred_tiff = save_pred_tiff

        # ============================================================
        # ЛУЧИ
        # ============================================================
        # Канон StarDist 3D (examples/3D/2_training.ipynb):
        #     extents    = calculate_extents(Y)            # медиана bbox объектов
        #     anisotropy = tuple(np.max(extents)/extents)  # АСПЕКТ ОБЪЕКТА
        #     grid       = tuple(1 if a > 1.5 else 2 for a in anisotropy)
        #     rays       = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)
        # То есть anisotropy у StarDist — форма объектов в вокселях, а НЕ
        # физический шаг между срезами.
        #
        # Rays_GoldenSpiral делает verts = dir/anisotropy, поэтому она
        # распределяет направления: сколько лучей ляжет в плоскость объекта, а
        # сколько упрётся в его верх и низ. При физической [5,1,1] на плоской
        # 2D-культуре контур цитоплазмы (полутолщина ~2.5 вокселя, радиус
        # ~100 px) задают 12 лучей из 96 — остальные 84 меряют толщину слоя и
        # забирают 2/3 MAE. При анизотропии, равной аспекту (100/2.5 = 40),
        # контур задают 68 лучей и все длины лежат в одном масштабе.
        #
        # Ядро и цитоплазма — объекты разной формы, наборы лучей раздельные.
        # n_rays общий, поэтому формы выходов голов не меняются.
        self.rays_anisotropy_nucleus = self._coerce_aniso(
            kwargs.get('rays_anisotropy_nucleus'), self.anisotropy)
        self.rays_anisotropy_cytoplasm = self._coerce_aniso(
            kwargs.get('rays_anisotropy_cytoplasm'), self.anisotropy)

        self._build_rays()

        if kwargs:
            print(f"⚠️ DEBUG: kwargs contains: {list(kwargs.keys())}")
        # Store additional kwargs
        for k, v in kwargs.items():
            if k not in ['prob_thresh_nucleus', 'prob_thresh_cytoplasm',
                         'nms_thresh', 'nucleus_grid', 'cytoplasm_grid',
                         'rays_anisotropy_nucleus', 'rays_anisotropy_cytoplasm']:
                setattr(self, k, v)

    # ============================================================
    # === ЛУЧИ И АНИЗОТРОПИЯ ===
    # ============================================================

    def _coerce_aniso(self, value, fallback):
        """Приводит анизотропию к кортежу длины ndim. None и мусор -> fallback."""
        if value is None:
            return tuple(float(v) for v in fallback)
        try:
            a = tuple(float(v) for v in value)
        except TypeError:
            a = (float(value),) + (1.0,) * (self.ndim - 1)
        if len(a) != self.ndim or any(v <= 0 for v in a):
            return tuple(float(v) for v in fallback)
        return a

    def _poolable_axes(self):
        """Оси, по которым декодер реально делает pooling.

        Модель считает pool_size как `2 if a < 2*min(a) else 1` по
        self.anisotropy. Grid по НЕпулящейся оси обязан оставаться 1: иначе GT
        прорежен, а выход сети нет, и формы разъедутся. Дублируем правило
        здесь, чтобы канонический grid не сломал сеть."""
        a = tuple(float(v) for v in self.anisotropy)
        m = min(a)
        return tuple(v < 2.0 * m for v in a)

    def canonical_grid(self, aniso):
        """grid = tuple(1 if a > 1.5 else 2 for a in anisotropy) — канон
        StarDist, с поправкой на непулящиеся оси сети."""
        ok = self._poolable_axes()
        return tuple(2 if (a <= 1.5 and ok[i]) else 1
                     for i, a in enumerate(aniso))

    def _build_rays(self):
        """Строит наборы лучей из текущих rays_anisotropy_*.
        Вызывается из __init__ и повторно из set_anisotropy_from_labels."""
        n_rays = int(self.n_rays)

        if self.ndim == 2:
            # 2D: направления равномерно по окружности, анизотропия не нужна.
            # Первый луч вверх, как в StarDist 2D.
            phi = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
            vertices_2d = np.stack([np.sin(phi), -np.cos(phi)],
                                   axis=-1).astype(np.float32)

            class Rays2D:
                def __init__(self, vertices):
                    self.vertices = np.array(vertices, dtype=np.float32)
                    self.ndim = 2

                def __len__(self):
                    return len(self.vertices)

                def dist_loss_weights(self, anisotropy=None):
                    return np.ones(len(self.vertices), dtype=np.float32)

            self.rays_nucleus = Rays2D(vertices_2d)
            self.rays_cytoplasm = Rays2D(vertices_2d)
            self.rays = self.rays_nucleus
            print(f"✅ 2D rays: {n_rays}, ray 0 = {self.rays.vertices[0]}")
            return

        from stardist import Rays_GoldenSpiral

        self.rays_nucleus = Rays_GoldenSpiral(
            n_rays, anisotropy=self.rays_anisotropy_nucleus)
        self.rays_cytoplasm = Rays_GoldenSpiral(
            n_rays, anisotropy=self.rays_anisotropy_cytoplasm)
        # Обратная совместимость: старый код, читающий config.rays, получает
        # набор ЯДРА. Наборы больше не совпадают — все новые обращения должны
        # брать rays_nucleus / rays_cytoplasm явно.
        self.rays = self.rays_nucleus

        print(f"✅ 3D rays: {n_rays} на голову, наборы РАЗНЫЕ")
        print(f"   nucleus   anisotropy={tuple(round(v, 1) for v in self.rays_anisotropy_nucleus)}")
        print(f"   cytoplasm anisotropy={tuple(round(v, 1) for v in self.rays_anisotropy_cytoplasm)}")

    def set_anisotropy_from_labels(self, nucleus_labels, cytoplasm_labels,
                                   update_grid=True):
        """Канонический замер по обучающим маскам: анизотропия -> лучи -> гриды
        -> масштаб dist. В конце пишет geometry.json рядом с чекпоинтом.

        Повторяет examples/3D/2_training.ipynb:
            extents    = calculate_extents(Y)
            anisotropy = tuple(np.max(extents)/extents)
            grid       = tuple(1 if a > 1.5 else 2 for a in anisotropy)

        МАСШТАБ DIST пересчитывается ВСЕГДА, даже если задан в конфиге. Он
        производный от набора лучей: Rays_GoldenSpiral хранит verts =
        dir/anisotropy, поэтому dist — множитель вдоль такого вектора, и при
        смене анизотропии все длины меняются. Аналитика по box-приближению
        объекта: t_k = min_a(h_a/|v_ka|), h = медианный bbox / 2 (воксели).
        Для клетки 5x200x200 при аспекте 40 это max_ray ~161, тогда как
        прежний dist_bias_init_cytoplasm=35 был подобран под физическую
        anisotropy [5,1,1] — голова стартовала втрое ниже цели и много эпох
        выдавала ровную заливку.

        Явно заданные rays_anisotropy_* и nucleus_grid/cytoplasm_grid
        приоритетнее: измерение не должно молча отменять ручную настройку.
        """
        if self.ndim != 3:
            self._build_rays()
            return self.rays_anisotropy_nucleus, self.rays_anisotropy_cytoplasm

        from stardist import calculate_extents

        base = tuple(float(v) for v in self.anisotropy)
        base_grid = tuple(int(v) for v in self.grid)
        measured_ext = {}

        def _measure(lbls):
            if lbls is None or len(lbls) == 0:
                return None
            ext = np.asarray(calculate_extents(list(lbls)), dtype=np.float64)
            if ext.size != self.ndim or np.any(ext <= 0):
                return None
            return tuple(float(v) for v in (np.max(ext) / ext)), ext

        print("\n📐 Канонический замер геометрии по GT:")
        for name, lbls in (('nucleus', nucleus_labels),
                           ('cytoplasm', cytoplasm_labels)):
            key = f'rays_anisotropy_{name}'
            if getattr(self, key) != base:
                print(f"   {key}: задана в конфиге "
                      f"{tuple(round(v, 1) for v in getattr(self, key))}, "
                      f"измерение не применяется")
                a = getattr(self, key)
                res = _measure(lbls)
                if res is not None:
                    measured_ext[name] = res[1]
            else:
                res = _measure(lbls)
                if res is None:
                    print(f"   {key}: объектов не найдено, оставляю "
                          f"{tuple(round(v, 1) for v in base)}")
                    a = getattr(self, key)
                else:
                    a, ext = res
                    measured_ext[name] = ext
                    setattr(self, key, a)
                    print(f"   {name}: медианный bbox {tuple(round(v, 1) for v in ext)} "
                          f"-> anisotropy {tuple(round(v, 1) for v in a)}")

            if update_grid:
                cur = tuple(int(v) for v in getattr(self, f'{name}_grid'))
                if cur != base_grid:
                    print(f"   {name}_grid: {cur} (из конфига, замер не применён)")
                else:
                    g = self.canonical_grid(a)
                    setattr(self, f'{name}_grid', g)
                    print(f"   {name}_grid: {g} (канон, с поправкой на pooling сети)")

        self._build_rays()

        # --- МАСШТАБ DIST под НОВЫЕ лучи ---
        for name in ('nucleus', 'cytoplasm'):
            ext = measured_ext.get(name)
            if ext is None:
                continue
            # rays.vertices ЕДИНИЧНЫЕ (rays3d.py:371 нормирует после деления на
            # анизотропию), поэтому t_k = min_a(h_a/|u_ka|) — это евклидово
            # расстояние до границы box вдоль луча k, в вокселях.
            verts = np.asarray(getattr(self, f'rays_{name}').vertices, np.float64)
            h = np.asarray(ext, np.float64) / 2.0
            t = np.min(h[None, :] / np.maximum(np.abs(verts), 1e-12), axis=1)
            # bias = МЕДИАНА длин, а не максимум: для MAE оптимальная константа
            # это медиана распределения таргета. Разброс длин большой (два
            # чисто-Z луча всегда дают полутолщину, экваториальные — радиус),
            # и максимум завышает старт вдвое-втрое: у цитоплазмы стартовый MAE
            # 48.6 px при bias=152 против 22.2 px при bias=110.
            scale = float(np.median(t))
            old_b = getattr(self, f'dist_bias_init_{name}')
            setattr(self, f'dist_bias_init_{name}', scale)
            # median_max_dist_* идёт в dist_scale (лог относительной ошибки) и
            # в r_hint промптов — там нужен именно РАЗМЕР объекта, максимум.
            setattr(self, f'median_max_dist_{name}', float(np.max(t)))
            print(f"   {name}: лучи max={t.max():.1f} median={scale:.1f} min={t.min():.1f} px "
                  f"-> dist_bias_init {old_b} -> {scale:.1f}, "
                  f"median_max_dist -> {t.max():.1f}")

        # --- СОХРАНЕНИЕ: без него inference соберёт лучи из физической
        # anisotropy и интерпретирует dist чужой геометрией ---
        geo_path = getattr(self, 'geometry_path', None)
        if geo_path:
            p = Path(geo_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                json.dump({
                    'n_rays': int(self.n_rays),
                    'rays_anisotropy_nucleus': list(self.rays_anisotropy_nucleus),
                    'rays_anisotropy_cytoplasm': list(self.rays_anisotropy_cytoplasm),
                    'nucleus_grid': list(self.nucleus_grid),
                    'cytoplasm_grid': list(self.cytoplasm_grid),
                    'median_max_dist_nucleus': float(self.median_max_dist_nucleus),
                    'median_max_dist_cytoplasm': float(self.median_max_dist_cytoplasm),
                    'dist_bias_init_nucleus': float(self.dist_bias_init_nucleus),
                    'dist_bias_init_cytoplasm': float(self.dist_bias_init_cytoplasm),
                }, f, indent=2)
            print(f"   💾 geometry.json сохранён: {p}")

        return self.rays_anisotropy_nucleus, self.rays_anisotropy_cytoplasm

    # ============================================================
    # === ЗАГРУЗКА ===
    # ============================================================

    @classmethod
    def from_json(cls, json_path):
        """Load configuration from JSON file.

        Плюс подхватывает geometry.json рядом с чекпоинтом, если он есть.
        Без этого inference/optimize_thresholds строили лучи из ФИЗИЧЕСКОЙ
        anisotropy, тогда как веса обучены под ЗАМЕРЕННУЮ (аспект объекта):
        формы совпадают, n_rays тот же, ошибок нет — но dist интерпретируется
        другим набором направлений, и полиэдры выходят чужой геометрии.
        Файл пишет set_anisotropy_from_labels в конце обучающего замера.
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        model_cfg = config_dict.get('model', {})
        training_cfg = config_dict.get('training', {})
        loss_weights = config_dict.get('loss_weights', {})
        loss_params = config_dict.get('loss_params', {})
        inference_cfg = config_dict.get('inference', {})
        gpu_cfg = config_dict.get('gpu_config', {})
        evaluation_cfg = config_dict.get('evaluation', {})
        # augmentation и data читались через getattr с дефолтами, но в params
        # не попадали — zoom_range/intensity_range/pool_size молча игнорировались.
        augmentation_cfg = config_dict.get('augmentation', {})
        data_cfg = config_dict.get('data', {})

        params = {
            **model_cfg,
            **training_cfg,
            **loss_weights,
            **loss_params,
            **evaluation_cfg,
            **{k: v for k, v in augmentation_cfg.items() if k != 'enable'},
            **{k: v for k, v in data_cfg.items()
               if k in ('pool_size', 'repool_freq')},
        }

        params['mode'] = config_dict.get('mode', 'train')

        if gpu_cfg:
            params['gpu_config'] = gpu_cfg

        # JSON uses initial_learning_rate, trainer reads train_learning_rate.
        if 'initial_learning_rate' in training_cfg:
            params['train_learning_rate'] = float(
                training_cfg['initial_learning_rate'])

        if 'steps_per_epoch' in training_cfg:
            params['train_steps_per_epoch'] = int(training_cfg['steps_per_epoch'])

        # Все inference-настройки кроме физического пути к чекпоинту.
        for key, value in inference_cfg.items():
            if key == 'checkpoint_path':
                continue
            params[key] = value

        # --- ГЕОМЕТРИЯ ОБУЧЕНИЯ ---
        geo_path = (Path(config_dict.get('output', {}).get('basedir', 'models')) /
                    config_dict.get('experiment', {}).get('name', 'model') /
                    'geometry.json')
        params['geometry_path'] = str(geo_path)
        if geo_path.exists():
            geo = json.load(open(geo_path))
            if int(geo.get('n_rays', params.get('n_rays', 0))) != int(params.get('n_rays', 0)):
                raise ValueError(
                    f"n_rays в {geo_path} ({geo.get('n_rays')}) не совпадает с "
                    f"конфигом ({params.get('n_rays')}) — веса и лучи от разных прогонов")
            for k in ('rays_anisotropy_nucleus', 'rays_anisotropy_cytoplasm',
                      'nucleus_grid', 'cytoplasm_grid',
                      'median_max_dist_nucleus', 'median_max_dist_cytoplasm',
                      'dist_bias_init_nucleus', 'dist_bias_init_cytoplasm'):
                if geo.get(k) is not None:
                    params[k] = geo[k]
            print(f"✅ Геометрия обучения подхвачена из {geo_path}")
        elif params.get('mode', 'train') != 'train':
            print(f"⚠️  {geo_path} не найден — лучи строятся из "
                  f"anisotropy={params.get('anisotropy')}. Если модель обучалась "
                  f"с замером, предсказания будут интерпретированы НЕВЕРНО.")

        for key in ('train_patch_size', 'anisotropy', 'grid',
                    'nucleus_grid', 'cytoplasm_grid',
                    'rays_anisotropy_nucleus', 'rays_anisotropy_cytoplasm'):
            if params.get(key) is not None:
                params[key] = tuple(params[key])

        return cls(**params)

    def to_base_config(self, head='nucleus'):
        """Convert to base StarDist Config2D/Config3D for compatibility."""
        ConfigClass = Config3D if self.ndim == 3 else Config2D
        rays = self.rays_cytoplasm if head == 'cytoplasm' else self.rays_nucleus
        grid = self.cytoplasm_grid if head == 'cytoplasm' else self.nucleus_grid

        return ConfigClass(
            rays=rays,
            grid=grid,
            anisotropy=self.anisotropy,
            use_gpu=self.use_gpu,
            n_channel_in=self.n_channel_in,
            train_patch_size=self.train_patch_size,
            train_batch_size=self.train_batch_size,
        )

    def __repr__(self):
        attrs = [
            f"ndim={self.ndim}",
            f"n_rays={self.n_rays}",
            f"nuc_grid={self.nucleus_grid}",
            f"cyto_grid={self.cytoplasm_grid}",
            f"rays_aniso_nuc={tuple(round(v, 1) for v in self.rays_anisotropy_nucleus)}",
            f"rays_aniso_cyto={tuple(round(v, 1) for v in self.rays_anisotropy_cytoplasm)}",
            f"patch_size={self.train_patch_size}",
            f"batch_size={self.train_batch_size}",
            f"consistency={'ON' if self.enable_consistency else 'OFF'}",
        ]
        return f"CytoNucsConfig({', '.join(attrs)})"