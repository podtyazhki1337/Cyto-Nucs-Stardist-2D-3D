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

        # Setup optimizer
        if optimizer is None:
            schedule = ExponentialDecay(
                initial_learning_rate=config.train_learning_rate,
                decay_steps=700,
                decay_rate=0.5,
                staircase=True
            )
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

    @tf.function
    def train_step(self, x_batch, y_batch):
        """
        Single training step.

        Args:
            x_batch: (B, Z, Y, X, C) input images
            y_batch: dict with target maps and instances

        Returns:
            loss: scalar loss value
        """
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model.keras_model(x_batch, training=True)

            # Compute loss (assignments handled separately in numpy)
            loss = self.loss_fn(y_batch, y_pred, assignments=None)

        # Backward pass
        gradients = tape.gradient(loss, self.model.keras_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.keras_model.trainable_variables)
        )

        # Update metrics
        self.train_loss_metric.update_state(loss)

        return loss

    @tf.function
    def val_step(self, x_batch, y_batch):
        """
        Single validation step.
        """
        y_pred = self.model.keras_model(x_batch, training=False)
        loss = self.loss_fn(y_batch, y_pred, assignments=None)
        self.val_loss_metric.update_state(loss)
        return loss

    def train(
            self,
            train_generator,
            val_generator=None,
            epochs=None,
            steps_per_epoch=None,
            callbacks=None,
            experiment=None
    ):
        """
        Train the model.

        Args:
            train_generator: CytoNucsDataGenerator for training
            val_generator: optional CytoNucsDataGenerator for validation
            epochs: number of epochs (default from config)
            steps_per_epoch: steps per epoch (default from config)
            callbacks: list of callbacks
            experiment: Comet ML experiment object
        """
        epochs = epochs or self.config.train_epochs
        steps_per_epoch = steps_per_epoch or self.config.train_steps_per_epoch
        callbacks = callbacks or []

        print(f"Training for {epochs} epochs, {steps_per_epoch} steps/epoch")
        print(f"Model: {self.model.name}")
        print(f"Config: {self.config}")

        best_val_loss = np.inf

        for epoch in range(epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 50}")

            # Reset metrics
            self.train_loss_metric.reset_states()
            self.val_loss_metric.reset_states()

            # Training loop
            train_pbar = tqdm(range(steps_per_epoch), desc='Training')
            for step in train_pbar:
                # Get batch
                batch_idx = step % len(train_generator)
                x_batch, y_batch = train_generator[batch_idx]

                # Convert to tensors
                x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                y_batch_tf = {
                    k: tf.convert_to_tensor(v, dtype=tf.float32)
                    for k, v in y_batch.items()
                    if k not in ['nucleus_instances', 'cytoplasm_instances', 'assignments']
                }

                # Train step
                loss = self.train_step(x_batch, y_batch_tf)

                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.numpy():.4f}'})

            # Record training loss
            train_loss = self.train_loss_metric.result().numpy()
            self.history['train_loss'].append(float(train_loss))
            self.history['lr'].append(float(self.optimizer.learning_rate.numpy()))

            print(f"Train Loss: {train_loss:.4f}")

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
                    y_batch_tf = {
                        k: tf.convert_to_tensor(v, dtype=tf.float32)
                        for k, v in y_batch.items()
                        if k not in ['nucleus_instances', 'cytoplasm_instances', 'assignments']
                    }

                    self.val_step(x_batch, y_batch_tf)

                val_loss = self.val_loss_metric.result().numpy()
                self.history['val_loss'].append(float(val_loss))
                print(f"Val Loss: {val_loss:.4f}")

                # Log to Comet
                if experiment:
                    experiment.log_metric("val_loss", val_loss, epoch=epoch + 1)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = self.checkpoint_dir / 'weights_best.h5'
                    self.model.save_weights(best_path)
                    print(f"✓ Best model saved: {best_path}")

            # Save latest model
            latest_path = self.checkpoint_dir / f'weights_epoch_{epoch + 1:03d}.h5'
            self.model.save_weights(latest_path)

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
        self.experiment = experiment

    def on_epoch_end(self, epoch, history):
        """Called at end of each epoch"""
        if (epoch + 1) % self.eval_every != 0:
            return

        print(f"\n📊 Computing Dice scores...")

        dice_scores_nucleus = []
        dice_scores_cytoplasm = []

        # Evaluate on validation set
        for i in range(len(self.val_generator)):
            x_batch, y_batch = self.val_generator[i]

            # Get predictions
            x_batch_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_pred = self.model.keras_model(x_batch_tf, training=False)

            # Convert to numpy
            nucleus_prob_pred = y_pred['nucleus_prob'].numpy()
            cytoplasm_prob_pred = y_pred['cytoplasm_prob'].numpy()

            # For each sample in batch
            for j in range(len(x_batch)):
                # Ground truth
                nucleus_gt = y_batch['nucleus_instances'][j]
                cytoplasm_gt = y_batch['cytoplasm_instances'][j]

                # Predictions (threshold probability maps for quick Dice)
                nucleus_pred = (nucleus_prob_pred[j, ..., 0] > 0.5).astype(np.uint8)
                cytoplasm_pred = (cytoplasm_prob_pred[j, ..., 0] > 0.5).astype(np.uint8)

                # Compute Dice
                dice_nuc = compute_dice(nucleus_gt, nucleus_pred)
                dice_cyto = compute_dice(cytoplasm_gt, cytoplasm_pred)

                dice_scores_nucleus.append(dice_nuc)
                dice_scores_cytoplasm.append(dice_cyto)

        mean_dice_nucleus = np.mean(dice_scores_nucleus)
        mean_dice_cytoplasm = np.mean(dice_scores_cytoplasm)
        mean_dice_combined = (mean_dice_nucleus + mean_dice_cytoplasm) / 2

        print(f"  Nucleus Dice: {mean_dice_nucleus:.4f}")
        print(f"  Cytoplasm Dice: {mean_dice_cytoplasm:.4f}")
        print(f"  Combined Dice: {mean_dice_combined:.4f}")

        # Log to Comet
        if self.experiment:
            self.experiment.log_metric("dice_nucleus", mean_dice_nucleus, epoch=epoch + 1)
            self.experiment.log_metric("dice_cytoplasm", mean_dice_cytoplasm, epoch=epoch + 1)
            self.experiment.log_metric("dice_combined", mean_dice_combined, epoch=epoch + 1)

        # Save if best
        if self.save_best and mean_dice_combined > self.best_dice:
            self.best_dice = mean_dice_combined
            save_path = Path(self.model.basedir) / self.model.name / \
                        f"weights_best_dice_{mean_dice_combined:.4f}.h5"
            self.model.save_weights(save_path)
            print(f"  🟢 New best Dice! Saved to {save_path}")


def compute_dice(gt, pred):
    """
    Compute Dice coefficient.

    Args:
        gt: ground truth binary mask
        pred: predicted binary mask

    Returns:
        dice: Dice score [0, 1]
    """
    intersection = np.logical_and(gt > 0, pred > 0).sum()
    volume = (gt > 0).sum() + (pred > 0).sum()
    return (2 * intersection) / volume if volume > 0 else 1.0