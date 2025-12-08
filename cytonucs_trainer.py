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
    This is the implementation from the stardist library.
    """
    # === ИСПРАВЛЕНИЕ: Очистка данных от нечисловых значений ===
    # Заменяем все, что не является числом (например, строки 'n/a'), на 0.0
    precision = np.array([p if isinstance(p, (int, float)) else 0.0 for p in precision])
    recall = np.array([r if isinstance(r, (int, float)) else 0.0 for r in recall])
    # ==========================================================

    # Append end points
    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))
    # Preprocess precision
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])
    # Compute area under curve
    x = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[x] - recall[x - 1]) * precision[x])
    return ap

class CytoNucsTrainer:
    """
    Custom trainer for CytoNucs StarDist model.
    """

    def __init__(self, model, config, loss_fn, optimizer=None):
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
        # Setup optimizer
        if optimizer is None:
            schedule = ExponentialDecay(
                initial_learning_rate=config.train_learning_rate,
                decay_steps=700,
                decay_rate=0.5,
                staircase=True
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

    # cytonucs_trainer.py - train_step БЕЗ промежуточного create_targets
    @tf.function
    def train_step(self, x_batch, y_batch_dict):

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model.keras_model(x_batch, training=True)

            # Loss (внутри: StarDist C++ + WBR + Containment + Consistency!)
            loss = self.loss_fn(y_batch_dict, y_pred)

        # Backward pass
        gradients = tape.gradient(loss, self.model.keras_model.trainable_variables)

        valid_gradients = []
        valid_vars = []
        for grad, var in zip(gradients, self.model.keras_model.trainable_variables):
            if grad is not None:
                valid_gradients.append(grad)
                valid_vars.append(var)

        # Gradient clipping
        clip_norm = 5.0
        clipped_gradients, global_norm = tf.clip_by_global_norm(valid_gradients, clip_norm)

        # Debug
        if self.current_step < 10 or self.current_step % 100 == 0:
            tf.print("  [GRADIENTS] Global norm:", global_norm, "Clipped:", global_norm > clip_norm)

        # Apply
        self.optimizer.apply_gradients(zip(clipped_gradients, valid_vars))

        # Update metrics
        self.train_loss_metric.update_state(loss)
        self.current_step += 1

        return loss


    def _create_targets_from_masks(self, y_batch_dict):
        """
        Конвертирует instance masks в prob/dist targets.
        Вызывается на CPU в Python для каждого батча.
        """
        nucleus_instances = y_batch_dict['nucleus_instances'].numpy()
        cytoplasm_instances = y_batch_dict['cytoplasm_instances'].numpy()

        batch_size = nucleus_instances.shape[0]

        # Инициализация
        if self.config.ndim == 3:
            prob_shape = (*nucleus_instances.shape[1:], 1)
            dist_shape = (*nucleus_instances.shape[1:], self.config.n_rays)
        else:
            prob_shape = (*nucleus_instances.shape[1:], 1)
            dist_shape = (*nucleus_instances.shape[1:], self.config.n_rays)

        nucleus_prob_batch = np.zeros((batch_size, *prob_shape), dtype=np.float32)
        nucleus_dist_batch = np.zeros((batch_size, *dist_shape), dtype=np.float32)
        cytoplasm_prob_batch = np.zeros((batch_size, *prob_shape), dtype=np.float32)
        cytoplasm_dist_batch = np.zeros((batch_size, *dist_shape), dtype=np.float32)

        # Обработка каждого образца в батче
        from cytonucs_losses import create_distance_maps_with_custom_rays

        for b in range(batch_size):
            nuc_prob, nuc_dist = create_distance_maps_with_custom_rays(
                nucleus_instances[b], self.config.rays
            )
            cyto_prob, cyto_dist = create_distance_maps_with_custom_rays(
                cytoplasm_instances[b], self.config.rays
            )

            nucleus_prob_batch[b] = nuc_prob
            nucleus_dist_batch[b] = nuc_dist
            cytoplasm_prob_batch[b] = cyto_prob
            cytoplasm_dist_batch[b] = cyto_dist

        return {
            'nucleus_prob': tf.convert_to_tensor(nucleus_prob_batch),
            'nucleus_dist': tf.convert_to_tensor(nucleus_dist_batch),
            'cytoplasm_prob': tf.convert_to_tensor(cytoplasm_prob_batch),
            'cytoplasm_dist': tf.convert_to_tensor(cytoplasm_dist_batch)
        }

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
        loss = self.loss_fn(y_batch, y_pred)  # ← БЕЗ assignments!

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

            for epoch in range(start_epoch, epochs):
                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"{'=' * 50}")

                if hasattr(self.loss_fn, 'current_epoch'):
                    self.loss_fn.current_epoch = epoch + 1

                # Reset metrics
                self.train_loss_metric.reset_state()
                self.val_loss_metric.reset_state()

                # Training loop
                train_pbar = tqdm(range(steps_per_epoch), desc='Training')
                for step in train_pbar:
                    # Get batch
                    batch_idx = step % len(train_generator)
                    x_batch, y_batch = train_generator[batch_idx]

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
                    loss = self.train_step(x_batch, y_batch_tf)

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
                        loss_wbr_val = self.loss_fn._wbr_loss(
                            y_pred_sample['nucleus_prob'], y_pred_sample['cytoplasm_prob']
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

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = self.checkpoint_dir / 'weights_best.weights.h5'
                        self.model.save_weights(best_path)
                        print(f"  ✅ Best model saved: {best_path}")

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

                # Save history
                with open(self.checkpoint_dir / 'history.json', 'w') as f:
                    json.dump(self.history, f, indent=2)

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

    def __init__(self, model, val_generator, eval_every=5, save_best=True, experiment=None):
        """
        Args:
            model: CytoNucsStarDistModel
            val_generator: validation data generator
            eval_every: evaluate every N epochs
            save_best: save best model by Dice
            experiment: Comet ML experiment
        """
        self.model = model
        self.val_generator = val_generator
        self.eval_every = eval_every
        self.save_best = save_best
        self.best_dice = -np.inf
        self.best_metric = -np.inf  # ИСПРАВЛЕНО: добавлена инициализация
        self.experiment = experiment

    def on_epoch_end(self, epoch, history):
        if (epoch + 1) % self.eval_every != 0:
            return

        print(f"\n Computing Dice & AP scores...")
        viz_dir = Path(self.model.basedir) / self.model.name / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        print("  Warming up BatchNorm statistics...")

        # Прогоняем несколько батчей validation в inference mode
        # чтобы обновить moving average
        for i in range(min(3, len(self.val_generator))):
            x_batch, y_batch = self.val_generator[i]
            x_batch_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)

            # Forward pass в inference mode (обновляет moving average!)
            _ = self.model.keras_model(x_batch_tf, training=False)

        print("  ✓ BatchNorm warmed up")

        # Теперь создаём predictor
        config = self.model.config
        predictor = CytoNucsPredictor(
            model=self.model,
            config=config,
            prob_thresh_nucleus=0.3,
            prob_thresh_cytoplasm=0.35,
            nms_thresh=0.5
        )

        y_true_nucleus, y_pred_nucleus = [], []
        y_true_cytoplasm, y_pred_cytoplasm = [], []

        for i in range(len(self.val_generator)):
            x_batch, y_batch = self.val_generator[i]
            for j in range(len(x_batch)):
                img = x_batch[j]
                img_input = img[..., 0] if img.shape[-1] == 1 else img
                nucleus_pred, cytoplasm_pred, details = predictor.predict_instances(img_input, normalize_img=False)
                y_true_nucleus.append(y_batch['nucleus_instances'][j])
                y_pred_nucleus.append(nucleus_pred)
                y_true_cytoplasm.append(y_batch['cytoplasm_instances'][j])
                y_pred_cytoplasm.append(cytoplasm_pred)
                if i == 0 and j == 0:
                    self._save_epoch_visualization_full(epoch + 1, img_input, y_batch['nucleus_instances'][j],
                                                        y_batch['cytoplasm_instances'][j], nucleus_pred, cytoplasm_pred,
                                                        details, viz_dir)

        mean_dice_nucleus = np.mean([compute_dice(gt, pred) for gt, pred in zip(y_true_nucleus, y_pred_nucleus)])
        mean_dice_cytoplasm = np.mean([compute_dice(gt, pred) for gt, pred in zip(y_true_cytoplasm, y_pred_cytoplasm)])

        taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        stats_nucleus_tuple = matching_dataset(y_true_nucleus, y_pred_nucleus, thresh=taus, show_progress=False)
        stats_cytoplasm_tuple = matching_dataset(y_true_cytoplasm, y_pred_cytoplasm, thresh=taus, show_progress=False)

        precision_nucleus, recall_nucleus = stats_nucleus_tuple[3], stats_nucleus_tuple[4]
        precision_cytoplasm, recall_cytoplasm = stats_cytoplasm_tuple[3], stats_cytoplasm_tuple[4]

        ap_nucleus = _calculate_ap(precision_nucleus, recall_nucleus)
        ap_cytoplasm = _calculate_ap(precision_cytoplasm, recall_cytoplasm)

        # ИСПРАВЛЕНО: Проверяем, что значения - это числа перед конвертацией
        def safe_float(val, default=0.0):
            """Безопасное преобразование в float с проверкой типа"""
            if isinstance(val, str):
                return default
            if isinstance(val, (list, np.ndarray)):
                if len(val) > 0:
                    return safe_float(val[0], default)
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        # Используем индекс 0 для AP@50 (первый порог tau=0.5)
        ap50_nucleus = safe_float(precision_nucleus[0])
        ap50_cytoplasm = safe_float(precision_cytoplasm[0])

        ap_nucleus = safe_float(ap_nucleus)
        ap_cytoplasm = safe_float(ap_cytoplasm)
        print(f"  Nucleus   | Dice: {mean_dice_nucleus:.4f} | AP: {ap_nucleus:.4f} (AP@50: {ap50_nucleus:.4f})")
        print(f"  Cytoplasm | Dice: {mean_dice_cytoplasm:.4f} | AP: {ap_cytoplasm:.4f} (AP@50: {ap50_cytoplasm:.4f})")

        if self.experiment:
            self.experiment.log_metrics(
                {"dice_nucleus": mean_dice_nucleus, "dice_cytoplasm": mean_dice_cytoplasm, "ap_nucleus": ap_nucleus,
                 "ap_cytoplasm": ap_cytoplasm, "ap50_nucleus": ap50_nucleus, "ap50_cytoplasm": ap50_cytoplasm},
                epoch=epoch + 1)
        mean_dice= (mean_dice_nucleus + mean_dice_cytoplasm) / 2.0
        current_metric = mean_dice
        if self.save_best and current_metric > self.best_metric:
            self.best_metric = current_metric
            save_path = Path(self.model.basedir) / self.model.name / f"weights_best_dice_{current_metric:.4f}.weights.h5"
            self.model.save_weights(save_path)
            print(f"  ✅ New best DICE! Saved to {save_path}")

    def _save_epoch_visualization_full(self, epoch, img, nuc_gt, cyto_gt, nuc_pred, cyto_pred, details, viz_dir, filename=None):
        """
        Save visualization with КОНТРАСТНЫМИ цветами и чёткими границами.
        """
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as mpatches

        # Compute Dice
        dice_nuc = compute_dice(nuc_gt, nuc_pred)
        dice_cyto = compute_dice(cyto_gt, cyto_pred)
        # Конвертируем и нормализуем
        if img.ndim == 3 and img.shape[-1] in [3, 4]:
            import cv2
            # КРИТИЧНО: нормализуем RGB к [0, 1] для matplotlib
            img_display = img.astype(np.float32)
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
            img_display = np.clip(img_display, 0, 1)  # ← ИСПРАВЛЕНИЕ

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
        np.random.seed(42)  # Фиксируем для воспроизводимости
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
        if img.ndim == 3 and img.shape[-1] in [3, 4]:
            axes[0, 0].imshow(img)
        else:
            axes[0, 0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
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
        axes[0, 1].set_title(f' Nucleus GT\n({nuc_gt.max()} objects)',
                             fontsize=16, fontweight='bold', pad=10, color='green')
        axes[0, 1].axis('off')

        # 3. Nucleus Prediction
        axes[0, 2].imshow(img, cmap='gray', alpha=0.3)
        if nuc_pred.max() > 0:
            axes[0, 2].imshow(nuc_pred, cmap=cmap_contrast, interpolation='nearest',
                              vmin=0, vmax=n_colors - 1, alpha=0.8)
            boundaries = find_boundaries(nuc_pred, mode='thick')
            axes[0, 2].contour(boundaries, colors='white', linewidths=1.5, levels=[0.5])
        axes[0, 2].set_title(f' Nucleus Pred\n({nuc_pred.max()} objects) | Dice={dice_nuc:.3f}',
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
            axes[1, 0].imshow(img)
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
        axes[1, 1].set_title(f' Cytoplasm GT\n({cyto_gt.max()} objects)',
                             fontsize=16, fontweight='bold', pad=10, color='green')
        axes[1, 1].axis('off')

        # 7. Cytoplasm Prediction
        axes[1, 2].imshow(img, cmap='gray', alpha=0.3)
        if cyto_pred.max() > 0:
            axes[1, 2].imshow(cyto_pred, cmap=cmap_contrast, interpolation='nearest',
                              vmin=0, vmax=n_colors - 1, alpha=0.8)
            boundaries = find_boundaries(cyto_pred, mode='thick')
            axes[1, 2].contour(boundaries, colors='white', linewidths=1.5, levels=[0.5])
        axes[1, 2].set_title(f' Cytoplasm Pred\n({cyto_pred.max()} objects) | Dice={dice_cyto:.3f}',
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

        nuc_prob = details.get('nucleus_prob', np.zeros_like(nuc_gt, dtype=float))
        cyto_prob = details.get('cytoplasm_prob', np.zeros_like(cyto_gt, dtype=float))
        nuc_dist = details.get('nucleus_dist', np.zeros((*nuc_gt.shape, 1)))
        cyto_dist = details.get('cytoplasm_dist', np.zeros((*cyto_gt.shape, 1)))

        # 9. Nucleus Probability
        im1 = axes[2, 0].imshow(nuc_prob, cmap='hot', vmin=0, vmax=1)
        axes[2, 0].set_title(f' Nucleus Prob\nMax={nuc_prob.max():.3f}',
                             fontsize=14, fontweight='bold', pad=10)
        axes[2, 0].axis('off')
        plt.colorbar(im1, ax=axes[2, 0], fraction=0.046, pad=0.04)

        # 10. Cytoplasm Probability
        im2 = axes[2, 1].imshow(cyto_prob, cmap='hot', vmin=0, vmax=1)
        axes[2, 1].set_title(f' Cytoplasm Prob\nMax={cyto_prob.max():.3f}',
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

    def _save_epoch_visualization(self, epoch, img, nuc_gt, cyto_gt,
                                  nuc_pred_prob, cyto_pred_prob, viz_dir):
        """Save visualization with GT vs Pred overlay + DISTANCE MAPS"""
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries

        # Threshold predictions
        nuc_pred = (nuc_pred_prob > 0.5).astype(np.uint8)
        cyto_pred = (cyto_pred_prob > 0.5).astype(np.uint8)

        # Compute Dice
        from cytonucs_trainer import compute_dice
        dice_nuc = compute_dice(nuc_gt, nuc_pred)
        dice_cyto = compute_dice(cyto_gt, cyto_pred)

        # === РАСШИРЕННАЯ ВИЗУАЛИЗАЦИЯ: 3 ряда ===
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        # === ROW 1: NUCLEUS ===
        if img.ndim == 3 and img.shape[-1] in [3, 4]:
            axes[0, 0].imshow(img)
            img_for_bottom = img
        else:
            axes[0, 0].imshow(img, cmap='gray')
            img_for_bottom = img

        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(nuc_gt, cmap='tab20')
        axes[0, 1].set_title(f'Nucleus GT ({nuc_gt.max()} objects)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(nuc_pred_prob, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Nucleus Pred (Dice={dice_nuc:.3f})')
        axes[0, 2].axis('off')

        # Overlay nucleus
        overlay_nuc = np.zeros((*nuc_gt.shape, 3))
        boundaries_gt = find_boundaries(nuc_gt, mode='inner')
        overlay_nuc[boundaries_gt] = [0, 1, 0]
        boundaries_pred = find_boundaries(nuc_pred, mode='inner')
        overlay_nuc[boundaries_pred] = [1, 0, 0]
        overlap = boundaries_gt & boundaries_pred
        overlay_nuc[overlap] = [1, 1, 0]

        axes[0, 3].imshow(overlay_nuc)
        axes[0, 3].set_title('Overlay (G=GT, R=Pred, Y=Match)')
        axes[0, 3].axis('off')

        # === ROW 2: CYTOPLASM ===
        if img_for_bottom.ndim == 3 and img_for_bottom.shape[-1] in [3, 4]:
            axes[1, 0].imshow(img_for_bottom)
        else:
            axes[1, 0].imshow(img_for_bottom, cmap='gray')

        axes[1, 0].set_title('Input Image')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cyto_gt, cmap='tab20')
        axes[1, 1].set_title(f'Cytoplasm GT ({cyto_gt.max()} objects)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(cyto_pred_prob, cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Cytoplasm Pred (Dice={dice_cyto:.3f})')
        axes[1, 2].axis('off')

        # Overlay cytoplasm
        overlay_cyto = np.zeros((*cyto_gt.shape, 3))
        boundaries_gt = find_boundaries(cyto_gt, mode='inner')
        overlay_cyto[boundaries_gt] = [0, 1, 0]
        boundaries_pred = find_boundaries(cyto_pred, mode='inner')
        overlay_cyto[boundaries_pred] = [1, 0, 0]
        overlap = boundaries_gt & boundaries_pred
        overlay_cyto[overlap] = [1, 1, 0]

        axes[1, 3].imshow(overlay_cyto)
        axes[1, 3].set_title('Overlay (G=GT, R=Pred, Y=Match)')
        axes[1, 3].axis('off')

        # === ROW 3: DISTANCE MAPS (НОВОЕ!) ===
        # Нужно получить distance predictions из модели
        # Это временный workaround - в следующей версии сохраняй dist в callback

        axes[2, 0].text(0.5, 0.5, 'Distance maps\nnot available\nin this version',
                        ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].axis('off')

        axes[2, 1].text(0.5, 0.5, f'WBR: ???\nContainment: ???',
                        ha='center', va='center', transform=axes[2, 1].transAxes,
                        fontsize=12)
        axes[2, 1].axis('off')

        axes[2, 2].text(0.5, 0.5, 'See console\nfor loss\ncomponents',
                        ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].axis('off')

        axes[2, 3].axis('off')

        plt.suptitle(f'Epoch {epoch} - Nucleus Dice: {dice_nuc:.3f}, Cytoplasm Dice: {dice_cyto:.3f}')
        plt.tight_layout()

        plt.savefig(viz_dir / f'epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  📷 Saved visualization: epoch_{epoch:04d}.png")


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