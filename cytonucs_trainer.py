"""
CytoNucs StarDist Training Loop

Implements custom training with all loss components.
"""
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
    # Очистка и преобразование
    precision = np.array([p if isinstance(p, (int, float)) and not np.isnan(p) else 0.0 for p in precision])
    recall = np.array([r if isinstance(r, (int, float)) and not np.isnan(r) else 0.0 for r in recall])

    # === ВАЖНАЯ ПРОВЕРКА: нормализация, если данные > 1 ===
    if len(precision) > 0 and np.max(precision) > 1.0:
        print(f"⚠️ WARNING: precision contains values > 1.0 (max={np.max(precision)}), normalizing...")
        precision = precision / np.max(precision)

    if len(recall) > 0 and np.max(recall) > 1.0:
        print(f"⚠️ WARNING: recall contains values > 1.0 (max={np.max(recall)}), normalizing...")
        recall = recall / np.max(recall)

    # Проверка на пустые массивы
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    # Сортировка по recall
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]

    # Добавление конечных точек
    precision = np.concatenate(([1.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))

    # Интерполяция precision (максимум справа)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    # Вычисление площади под кривой
    x = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[x] - recall[x - 1]) * precision[x])

    return ap
class _Prefetcher:
    """Готовит батчи в фоновых потоках. Несколько воркеров, потому что одного
    не хватало: T_gen ~50 с против T_gpu ~21 с, очередь пустела и шаги шли
    вперемешку 21/76 с."""

    def __init__(self, gen, depth=3, workers=3):
        import threading, queue
        self._gen = gen
        self._q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._err = None
        self._times = []
        self._lock = threading.Lock()
        self._n = max(1, int(workers))
        self._threads = [threading.Thread(target=self._work, args=(w,), daemon=True)
                         for w in range(self._n)]
        for t in self._threads:
            t.start()

    def _work(self, w):
        import queue, time, traceback
        i = w
        while not self._stop.is_set():
            try:
                t0 = time.time()
                batch = self._gen[i % max(1, len(self._gen))]
                dt = time.time() - t0
            except Exception as e:
                # НЕ глотаем: прежний `except: continue` оставлял поток крутиться
                # вхолостую, очередь пустела и next() висел без трейсбека.
                self._err = e
                traceback.print_exc()
                return
            with self._lock:
                self._times.append(dt)
                if len(self._times) > 50:
                    self._times.pop(0)
            while not self._stop.is_set():
                try:
                    self._q.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    continue
            i += self._n

    def gen_time(self):
        with self._lock:
            return float(np.mean(self._times)) if self._times else 0.0

    def next(self):
        import queue
        while True:
            try:
                return self._q.get(timeout=1.0)
            except queue.Empty:
                if self._err is not None:
                    raise RuntimeError("Prefetcher: ошибка в генераторе") from self._err
                if self._stop.is_set():
                    raise RuntimeError("Prefetcher остановлен")

    def stop(self):
        self._stop.set()
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
            # ИСПРАВЛЕНО: БЕЗ clipnorm (будет применяться clip_by_global_norm вручную)
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

        # cytonucs_trainer.py - ЛУЧШИЙ ВАРИАНТ с tf.py_function


    @tf.function
    def train_step(self, x_batch, y_batch_dict):

        with tf.GradientTape() as tape:
            y_pred = self.model.keras_model(x_batch, training=True)
            loss = self.loss_fn(y_batch_dict, y_pred, current_epoch=self.current_epoch_var)

        gradients = tape.gradient(loss, self.model.keras_model.trainable_variables)

        # Имена слоёв статичны на этапе трейса — группируем по веткам.
        # ВАЖНО: в Keras 3 var.name = 'kernel'/'bias' БЕЗ префикса слоя, полный путь
        # в var.path -> группируем по var.path, иначе всё валится в один backbone.
        # refinement (mask/boundary) stop_gradient'нут от декодера -> его градиент
        # НЕ течёт в общий encoder/decoder, клипаем его ОТДЕЛЬНОЙ группой (когда он
        # снова включён), чтобы его Dice-градиент не масштабировал вниз prob/dist.
        prob_layers = {'nucleus_prob', 'cytoplasm_prob',
                       'nucleus_prob_logit', 'cytoplasm_prob_logit'}
        dist_layers = {'nucleus_dist', 'cytoplasm_dist'}
        refine_subs = ('_refine', 'prompt_gate', 'nucleus_mask', 'cytoplasm_mask',
                       'geo_dist_compress')
        # Декодер цитоплазмы — ОТДЕЛЬНАЯ группа. Он не разделяется с ядром
        # (общий только энкодер), а его градиент пришёл к насыщению обеих
        # цито-голов при целом ядре. В общей группе backbone его норма
        # усреднялась с энкодером и декодером ядра, и всплеск был не виден.
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

        # Нормы — ТОЛЬКО для лога. Канон (base.py:308) применяет Adam без
        # клиппинга вообще. Клип по группам множит градиент на 20/norm, а norm
        # скачет по шагам (prob 499 -> 68 -> 1.4): постоянный множитель Adam
        # сокращает, ПЕРЕМЕННЫЙ — нет, потому что v накоплен по одному масштабу,
        # а m по другому. Плюс группы режутся независимо, и заданный конфигом
        # баланс lambda_prob/lambda_dist = 1/0.2 динамически переопределяется
        # каждым шагом. gradient_clip_norm=0 отключает клип (каноническое
        # поведение); >0 оставлен как аварийный тормоз.
        prob_norm = _group_norm(prob_grads)
        dist_norm = _group_norm(dist_grads)
        bb_norm = _group_norm(backbone_grads)
        rf_norm = _group_norm(refine_grads)
        cd_norm = _group_norm(cdec_grads)

        all_grads = prob_grads + dist_grads + backbone_grads + refine_grads + cdec_grads
        all_vars = prob_vars + dist_vars + backbone_vars + refine_vars + cdec_vars

        if clip_norm > 0:
            # ОДИН глобальный клип на всю сеть, как делал бы Adam(clipnorm=...):
            # относительный баланс между группами сохраняется.
            all_grads, _ = tf.clip_by_global_norm(all_grads, clip_norm)

        self.optimizer.apply_gradients(zip(all_grads, all_vars))
        self.train_loss_metric.update_state(loss)
        return loss, prob_norm, dist_norm, bb_norm, rf_norm, cd_norm




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
        # Forward pass (БЕЗ training=True)
        y_pred = self.model.keras_model(x_batch, training=False)

        # ИСПРАВЛЕНИЕ: убрать assignments=None
        loss = self.loss_fn(y_batch, y_pred, current_epoch=self.current_epoch_var)

        # Update validation metric
        self.val_loss_metric.update_state(loss)

        return loss

    # Основной цикл тренировки


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

            print(f"Training for {epochs} epochs, {steps_per_epoch} steps/epoch")
            print(f"Model: {self.model.name}")
            print(f"Config: {self.config}")

            best_val_loss = np.inf
            prefetch = _Prefetcher(
                train_generator,
                depth=int(getattr(self.config, 'prefetch_depth', 3)),
                workers=int(getattr(self.config, 'prefetch_workers', 3)))
            for epoch in range(start_epoch, epochs):
                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"{'=' * 50}")

                self.current_epoch_var.assign(epoch + 1)
                # Reset metrics
                self.train_loss_metric.reset_state()
                self.val_loss_metric.reset_state()

                # Training loop
                train_pbar = tqdm(range(steps_per_epoch), desc='Training')
                for step in train_pbar:
                    # Get batch
                    x_batch, y_batch = prefetch.next()

                    # Конвертируем x в tensor
                    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)

                    # Конвертируем y_batch ЦЕЛИКОМ (включая instance masks)
                    y_batch_tf = {}
                    for k, v in y_batch.items():
                        if isinstance(v, np.ndarray):
                            if v.dtype.kind == 'f':  # float
                                y_batch_tf[k] = tf.convert_to_tensor(v, dtype=tf.float32)
                            elif v.dtype.kind in ['i', 'u']:  # int/uint
                                y_batch_tf[k] = tf.convert_to_tensor(v, dtype=tf.int32)
                            else:
                                y_batch_tf[k] = tf.convert_to_tensor(v)
                        else:
                            y_batch_tf[k] = v

                    # Train step
                    loss, p_n, d_n, b_n, r_n, c_n = self.train_step(x_batch, y_batch_tf)
                    self.current_step += 1
                    if self.current_step % 1 == 0:
                        print(f"  [GRAD NORMS] prob: {float(p_n):.3f} dist: {float(d_n):.3f} "
                              f"backbone: {float(b_n):.3f} cyto_dec: {float(c_n):.3f} "
                              f"|| refine: {float(r_n):.3f} "
                              f"| clip: {float(self.config.gradient_clip_norm)}")

                    # Update progress bar
                    train_pbar.set_postfix({'loss': f'{loss.numpy():.4f}'})

                # Record training loss
                train_loss = self.train_loss_metric.result().numpy()
                self.history['train_loss'].append(float(train_loss))
                self.history['lr'].append(float(self.optimizer.learning_rate.numpy()))

                print(f"Train Loss: {train_loss:.4f}")

                # Log loss components
                if (epoch + 1) % 1 == 0:
                    try:
                        x_sample, y_sample = train_generator[0]
                        x_sample_tf = tf.convert_to_tensor(x_sample, dtype=tf.float32)

                        y_sample_tf = {}
                        for k, v in y_sample.items():
                            if isinstance(v, np.ndarray):
                                if v.dtype.kind == 'f':
                                    y_sample_tf[k] = tf.convert_to_tensor(v, dtype=tf.float32)
                                elif v.dtype.kind in ['i', 'u']:
                                    y_sample_tf[k] = tf.convert_to_tensor(v, dtype=tf.int32)
                                else:
                                    y_sample_tf[k] = tf.convert_to_tensor(v)
                            else:
                                y_sample_tf[k] = v

                        y_pred_sample = self.model.keras_model(x_sample_tf, training=True)
                        y_true_sample = self.loss_fn._create_gt_from_instances(y_sample_tf)

                        loss_nuc = self.loss_fn._stardist_loss(
                            y_true_sample['nucleus_prob'], y_pred_sample['nucleus_prob'],
                            y_true_sample['nucleus_dist'], y_pred_sample['nucleus_dist'],
                            name='nucleus'
                        )
                        loss_cyto = self.loss_fn._stardist_loss(
                            y_true_sample['cytoplasm_prob'], y_pred_sample['cytoplasm_prob'],
                            y_true_sample['cytoplasm_dist'], y_pred_sample['cytoplasm_dist'],
                            name='cytoplasm'
                        )
                        _cyto_fg = self.loss_fn._grid_subsample_cytoplasm(
                            tf.expand_dims(tf.cast(
                                y_sample_tf['cytoplasm_instances'] > 0, tf.float32), -1))
                        loss_wbr_val = self.loss_fn._wbr_loss(
                            y_pred_sample['nucleus_prob'], _cyto_fg
                        )
                        loss_containment_val = self.loss_fn._containment_loss(
                            y_true_sample, y_pred_sample, y_sample_tf
                        )

                        print(f"   Loss components:")
                        print(
                            f"     Nucleus BCE: {loss_nuc['bce'].numpy():.4f}, Distance: {loss_nuc['distance'].numpy():.4f}")
                        print(
                            f"     Cytoplasm BCE: {loss_cyto['bce'].numpy():.4f}, Distance: {loss_cyto['distance'].numpy():.4f}")
                        print(f"     WBR penalty: {loss_wbr_val.numpy():.4f}")
                        print(f"     Containment: {loss_containment_val.numpy():.4f}")

                        if experiment:
                            experiment.log_metrics({
                                'nucleus_bce': float(loss_nuc['bce'].numpy()),
                                'nucleus_distance': float(loss_nuc['distance'].numpy()),
                                'cytoplasm_bce': float(loss_cyto['bce'].numpy()),
                                'cytoplasm_distance': float(loss_cyto['distance'].numpy()),
                                'wbr_penalty': float(loss_wbr_val.numpy()),
                                'containment': float(loss_containment_val.numpy()),
                            }, epoch=epoch + 1)
                    except Exception as e:
                        print(f"  └─ Could not compute loss components: {e}")

                # Log to Comet
                if experiment:
                    experiment.log_metric("train_loss", train_loss, epoch=epoch + 1)
                    experiment.log_metric("learning_rate", self.optimizer.learning_rate.numpy(), epoch=epoch + 1)

                # Validation loop
                if val_generator is not None:
                    val_pbar = tqdm(range(len(val_generator)), desc='Validation')
                    for step in val_pbar:
                        x_batch, y_batch = val_generator[step]

                        x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)

                        y_batch_tf = {}
                        for k, v in y_batch.items():
                            if isinstance(v, np.ndarray):
                                if v.dtype.kind == 'f':
                                    y_batch_tf[k] = tf.convert_to_tensor(v, dtype=tf.float32)
                                elif v.dtype.kind in ['i', 'u']:
                                    y_batch_tf[k] = tf.convert_to_tensor(v, dtype=tf.int32)
                                else:
                                    y_batch_tf[k] = tf.convert_to_tensor(v)
                            else:
                                y_batch_tf[k] = v

                        self.val_step(x_batch, y_batch_tf)

                    val_loss = self.val_loss_metric.result().numpy()
                    self.history['val_loss'].append(float(val_loss))
                    print(f"Val Loss: {val_loss:.4f}")

                    if experiment:
                        experiment.log_metric("val_loss", val_loss, epoch=epoch + 1)

                        # СНАРУЖИ if experiment: при use_comet=false чекпоинт
                        # не сохранялся вовсе.
                        # Состав лосса меняется на warmup -> val_loss несравним
                        # между регимами. Сбрасываем отсчёт, когда warmup доехал
                        # до 1.0, иначе best замерзает на предпоследней эпохе hold.
                    _warm = int(getattr(self.config, 'loss_warmup_epochs',
                                        getattr(self.config, 'wbr_warmup_epochs', 20)))
                    if (epoch + 1) == 2 * _warm:
                        best_val_loss = np.inf
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                        # val_loss считается на deterministic train-sized patches.
                        # Это полезная training diagnostic, но НЕ production quality metric.
                        best_path = (
                                self.checkpoint_dir
                                / 'weights_best_val_loss.weights.h5'
                        )

                        self.model.save_weights(
                            best_path
                        )

                        print(
                            f"  ✅ Best PATCH val-loss checkpoint saved: "
                            f"{best_path}"
                        )

                # Save checkpoints
                save_every = getattr(self.config, 'save_every_n_epochs', 10)
                if (epoch + 1) % save_every == 0:
                    latest_path = self.checkpoint_dir / f'weights_epoch_{epoch + 1:03d}.weights.h5'
                    self.model.save_weights(latest_path)
                    print(f"  💾 Checkpoint saved: {latest_path}")

                # Call callbacks
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, self.history)

                if hasattr(train_generator, 'on_epoch_end'):
                    train_generator.on_epoch_end()
                if val_generator is not None and hasattr(val_generator, 'on_epoch_end'):
                    val_generator.on_epoch_end()
                # Save history
                with open(self.checkpoint_dir / 'history.json', 'w') as f:
                    json.dump(self.history, f, indent=2)
            prefetch.stop()

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
            nms_thresh=None
    ):
        """
        Full-volume quality validation.

        val_loss по-прежнему считается отдельно на deterministic patches.

        Этот callback:
            full validation volume
            -> production-like tiled inference
            -> sparse StarDist NMS
            -> prompted refinement
            -> AP / AP@50 / semantic Dice / JTPR

        Лучший checkpoint выбирается по:
            mean(AP_nucleus, AP_cytoplasm)

        где AP = mean accuracy TP/(TP+FP+FN)
        по IoU thresholds 0.50...0.95.
        """

        self.model = model
        self.val_generator = val_generator

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
        Foreground/background Dice по полному volume.

        В отличие от compute_dice(), штрафует и FP, и FN,
        потому что считается по всей бинарной маске.
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

    def _predict_full_volume(
            self,
            predictor,
            img
    ):
        """
        Тот же выбор tiled/non-tiled inference,
        который используется в run_inference_only().

        img уже нормализован на полном volume внутри
        val_generator.repool().
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

            # Z полностью обрабатывается fixed-Z
            # логикой внутри _predict_with_tiling().
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
        # ВАЖНО:
        # НЕ используем:
        #
        #     self.val_generator[i]
        #
        # потому что __getitem__ возвращает train-sized patch.
        #
        # data_pool содержит исходные ПОЛНЫЕ volumes,
        # нормализованные один раз по всему тому.
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
                "val_generator не содержит "
                "data_pool/image_files"
            )

        if len(data_pool) != len(image_files):
            raise RuntimeError(
                "Full-volume validation требует "
                "ВСЮ validation выборку в data_pool. "
                f"data_pool={len(data_pool)}, "
                f"image_files={len(image_files)}. "
                "Для val_generator не задавайте "
                "уменьшенный pool_size."
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

        # ============================================================
        # STORAGE
        # ============================================================

        y_true_nucleus = []
        y_pred_nucleus = []

        y_true_cytoplasm = []
        y_pred_cytoplasm = []

        semantic_nucleus = []
        semantic_cytoplasm = []

        pred_assignments = []

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

        for i, sample in enumerate(
                data_pool
        ):

            img = np.asarray(
                sample['image'],
                dtype=np.float32
            )

            nucleus_gt = np.asarray(
                sample['nucleus_mask']
            )

            cytoplasm_gt = np.asarray(
                sample['cytoplasm_mask']
            )

            filename = sample.get(
                'filename',
                f'val_{i:03d}'
            )

            print(
                f"\n  [{i + 1}/"
                f"{len(data_pool)}] "
                f"{filename}"
            )

            print(
                f"    img={img.shape} | "
                f"GT nuc="
                f"{len(np.unique(nucleus_gt)) - 1} "
                f"cyto="
                f"{len(np.unique(cytoplasm_gt)) - 1}"
            )

            (
                nucleus_pred,
                cytoplasm_pred,
                details
            ) = self._predict_full_volume(
                predictor,
                img
            )

            # Никакого silent cropping validation результатов.
            # Shape mismatch здесь означает реальную ошибку pipeline.
            if (
                    nucleus_pred.shape
                    != nucleus_gt.shape
            ):
                raise RuntimeError(
                    f"{filename}: "
                    f"nucleus shape mismatch: "
                    f"pred={nucleus_pred.shape}, "
                    f"GT={nucleus_gt.shape}"
                )

            if (
                    cytoplasm_pred.shape
                    != cytoplasm_gt.shape
            ):
                raise RuntimeError(
                    f"{filename}: "
                    f"cytoplasm shape mismatch: "
                    f"pred={cytoplasm_pred.shape}, "
                    f"GT={cytoplasm_gt.shape}"
                )

            sd_n = self._semantic_dice(
                nucleus_gt,
                nucleus_pred
            )

            sd_c = self._semantic_dice(
                cytoplasm_gt,
                cytoplasm_pred
            )

            semantic_nucleus.append(
                sd_n
            )

            semantic_cytoplasm.append(
                sd_c
            )

            y_true_nucleus.append(
                nucleus_gt
            )

            y_pred_nucleus.append(
                nucleus_pred
            )

            y_true_cytoplasm.append(
                cytoplasm_gt
            )

            y_pred_cytoplasm.append(
                cytoplasm_pred
            )

            pred_assignments.append(
                dict(
                    getattr(
                        predictor,
                        '_prompt_assignments',
                        {}
                    ) or {}
                )
            )

            print(
                f"    semantic Dice: "
                f"nuc={sd_n:.3f} "
                f"cyto={sd_c:.3f} | "
                f"pred nuc="
                f"{len(np.unique(nucleus_pred)) - 1} "
                f"cyto="
                f"{len(np.unique(cytoplasm_pred)) - 1}"
            )

            # Один полный validation example
            # на каждую evaluation epoch.
            if i == 0:
                self._save_epoch_visualization_full(
                    epoch=epoch_num,
                    img=img,
                    nuc_gt=nucleus_gt,
                    cyto_gt=cytoplasm_gt,
                    nuc_pred=nucleus_pred,
                    cyto_pred=cytoplasm_pred,
                    details=details,
                    viz_dir=viz_dir,
                    filename=(
                        f'epoch_'
                        f'{epoch_num:04d}_'
                        f'fullval.png'
                    )
                )

        # ============================================================
        # INSTANCE METRICS
        # ============================================================

        taus = [
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95
        ]

        stats_nucleus = matching_dataset(
            y_true_nucleus,
            y_pred_nucleus,
            thresh=taus,
            show_progress=False
        )

        stats_cytoplasm = matching_dataset(
            y_true_cytoplasm,
            y_pred_cytoplasm,
            thresh=taus,
            show_progress=False
        )

        acc_nucleus = np.asarray(
            [
                s.accuracy
                for s in stats_nucleus
            ],
            dtype=np.float64
        )

        acc_cytoplasm = np.asarray(
            [
                s.accuracy
                for s in stats_cytoplasm
            ],
            dtype=np.float64
        )

        ap_nucleus = float(
            np.mean(
                acc_nucleus
            )
        )

        ap_cytoplasm = float(
            np.mean(
                acc_cytoplasm
            )
        )

        ap50_nucleus = float(
            acc_nucleus[0]
        )

        ap50_cytoplasm = float(
            acc_cytoplasm[0]
        )

        semantic_dice_nucleus = float(
            np.mean(
                semantic_nucleus
            )
        )

        semantic_dice_cytoplasm = float(
            np.mean(
                semantic_cytoplasm
            )
        )

        s_n = stats_nucleus[0]
        s_c = stats_cytoplasm[0]

        # ============================================================
        # JTPR
        # ============================================================

        from cytonucs_inference import (
            compute_jtpr
        )

        jtpr_values = []

        joint_tp_total = 0
        total_pairs_total = 0

        for idx_img, (
                nucleus_pred,
                cytoplasm_pred,
                nucleus_gt,
                cytoplasm_gt
        ) in enumerate(
            zip(
                y_pred_nucleus,
                y_pred_cytoplasm,
                y_true_nucleus,
                y_true_cytoplasm
            )
        ):
            assignments_pred = (
                pred_assignments[idx_img]
                if idx_img
                   < len(pred_assignments)
                else None
            )

            jtpr, jtpr_details = (
                compute_jtpr(
                    nucleus_pred=
                    nucleus_pred,

                    cytoplasm_pred=
                    cytoplasm_pred,

                    nucleus_gt=
                    nucleus_gt,

                    cytoplasm_gt=
                    cytoplasm_gt,

                    iou_thresh=0.5,

                    assignments_pred=(
                        assignments_pred
                        if assignments_pred
                        else None
                    )
                )
            )

            jtpr_values.append(
                float(jtpr)
            )

            joint_tp_total += int(
                jtpr_details[
                    'joint_tp'
                ]
            )

            total_pairs_total += int(
                jtpr_details[
                    'total_pairs'
                ]
            )

        jtpr_mean = (
            float(
                np.mean(
                    jtpr_values
                )
            )
            if jtpr_values
            else 0.0
        )

        jtpr_global = (
            float(
                joint_tp_total
                / total_pairs_total
            )
            if total_pairs_total > 0
            else 0.0
        )

        # ============================================================
        # CHECKPOINT METRIC
        # ============================================================
        #
        # compute_dice() здесь СПЕЦИАЛЬНО не используем:
        # он считает Dice только по matched instances и
        # не штрафует достаточным образом FP/FN.
        #
        # AP/accuracy учитывает:
        # TP, FP и FN.
        # ============================================================

        current_metric = float(
            0.5
            * (
                    ap_nucleus
                    + ap_cytoplasm
            )
        )

        print(
            "\n  FULL-VOLUME METRICS"
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

        print(
            f"  CHECKPOINT METRIC "
            f"(mean AP heads) = "
            f"{current_metric:.4f}"
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
                },
                epoch=epoch_num
            )

        # ============================================================
        # BEST MODEL
        # ============================================================

        if (
                self.save_best
                and current_metric
                > self.best_metric
        ):
            self.best_metric = (
                current_metric
            )

            # Главный production checkpoint.
            save_path = (
                    Path(
                        self.model.basedir
                    )
                    / self.model.name
                    / 'weights_best.weights.h5'
            )

            self.model.save_weights(
                save_path
            )

            best_info = {
                'epoch':
                    epoch_num,

                'metric_name':
                    'mean_fullval_ap',

                'metric':
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
                f"{current_metric:.4f}"
            )

            print(
                f"     saved to: "
                f"{save_path}"
            )

    def _save_epoch_visualization_full(self, epoch, img, nuc_gt, cyto_gt, nuc_pred, cyto_pred, details, viz_dir, filename=None):
        """
        Save visualization with КОНТРАСТНЫМИ цветами и чёткими границами.
        """
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
        from matplotlib.colors import ListedColormap

        def _quick_dice(gt, pred):
            g, p = np.asarray(gt) > 0, np.asarray(pred) > 0
            if g.shape != p.shape:
                return 0.0  # размеры разошлись (пустой pred в grid-разрешении) — пропускаем
            denom = g.sum() + p.sum()
            return float(2 * np.logical_and(g, p).sum() / denom) if denom else 0.0
        dice_nuc = _quick_dice(nuc_gt, nuc_pred)
        dice_cyto = _quick_dice(cyto_gt, cyto_pred)

        def _n_obj(a):
            """Число объектов = уникальные ненулевые лейблы.
            `.max()` даёт максимальный ID: GT-маска после кропа сохраняет
            нумерацию исходного тома (ядра 3, 7, 10 -> max 10 при трёх
            объектах), а предсказание перенумеровано подряд -> панели GT и
            Pred считали разное, и «10 GT против 3 Pred» выглядело как
            недодетекция там, где было попадание 3/3."""
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
        # Конвертируем и нормализуем
        if img.ndim == 3 and img.shape[-1] in [3, 4]:
            import cv2
            # перцентильная нормализация для показа (как вход модели, устойчиво
            # к горячим пикселям) — иначе один яркий выброс гасит всю картинку.
            img_display = img.astype(np.float32)
            p_lo = np.percentile(img_display, 1)
            p_hi = np.percentile(img_display, 99.5)
            img_display = (img_display - p_lo) / (p_hi - p_lo + 1e-8)
            img_display = np.clip(img_display, 0, 1)
            img_gray = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            is_rgb = True
        else:
            img_gray = img.astype(np.float32)
            # Normalize grayscale to [0, 1]
            img_display = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
            img_display = np.clip(img_display, 0, 1)
            is_rgb = False

        # ============================================================
        # СОЗДАЁМ КОНТРАСТНУЮ COLORMAP
        # ============================================================

        # Генерируем яркие, насыщенные цвета

        n_colors = max(nuc_pred.max(), cyto_pred.max(), nuc_gt.max(), cyto_gt.max()) + 1

        # HSV colorspace для максимальной контрастности
        from matplotlib.colors import hsv_to_rgb

        colors = []
        colors.append([0, 0, 0])  # Background = black

        for i in range(1, n_colors):
            # Фиксированная saturation и value для яркости
            hue = (i * 137.508) % 360  # Golden angle для максимальной separation
            saturation = 0.8 + 0.2 * (i % 2)  # Чередуем 0.8 и 1.0
            value = 0.9 + 0.1 * ((i + 1) % 2)  # Чередуем 0.9 и 1.0

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
            mask = axes[0, 1].imshow(nuc_gt, cmap=cmap_contrast, interpolation='nearest',
                                     vmin=0, vmax=n_colors - 1, alpha=0.8)
            # Границы белым
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
        # карты могут отсутствовать или быть None (убраны для экономии RAM) — дефолты
        if nuc_prob is None:  nuc_prob = np.zeros_like(nuc_gt, dtype=float)
        if cyto_prob is None: cyto_prob = np.zeros_like(cyto_gt, dtype=float)
        if nuc_dist is None:  nuc_dist = np.zeros((*nuc_gt.shape, 1), dtype=float)
        if cyto_dist is None: cyto_dist = np.zeros((*cyto_gt.shape, 1), dtype=float)

        # 9. Nucleus Probability
        im1 = axes[2, 0].imshow(nuc_prob, cmap='hot', vmin=0, vmax=1)
        axes[2, 0].set_title(f' Nucleus Prob\nMax(срез)={nuc_prob.max():.3f} '
                             f'Max(том)={_vmax["nucleus_prob"]:.3f}',
                             fontsize=14, fontweight='bold', pad=10)
        axes[2, 0].axis('off')
        plt.colorbar(im1, ax=axes[2, 0], fraction=0.046, pad=0.04)

        # 10. Cytoplasm Probability
        im2 = axes[2, 1].imshow(cyto_prob, cmap='hot', vmin=0, vmax=1)
        axes[2, 1].set_title(f' Cytoplasm Prob\nMax(срез)={cyto_prob.max():.3f} '
                             f'Max(том)={_vmax["cytoplasm_prob"]:.3f}',
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
        gt: ground truth instance mask (H, W) с unique IDs
        pred: predicted instance mask (H, W) с unique IDs

    Returns:
        dice: mean Dice score for matched instances [0, 1]
    """
    from scipy.optimize import linear_sum_assignment
    from skimage.measure import regionprops

    # Если обе пустые - идеально
    if gt.max() == 0 and pred.max() == 0:
        return 1.0
    # Если одна пустая - провал
    if gt.max() == 0 or pred.max() == 0:
        return 0.0

    # Получаем регионы
    gt_props = regionprops(gt.astype(np.int32))
    pred_props = regionprops(pred.astype(np.int32))

    if len(gt_props) == 0 and len(pred_props) == 0:
        return 1.0
    if len(gt_props) == 0 or len(pred_props) == 0:
        return 0.0

    # IoU matrix для matching
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

    # Dice для matched pairs (IoU > 0.5)
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

    # Mean Dice по matched instances
    return np.mean(dice_scores)