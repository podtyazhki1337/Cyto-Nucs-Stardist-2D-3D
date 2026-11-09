"""StarDist-3D trained on partial annotation with the unknown voxels left out of every loss term ('masked loss', a
modification of the baseline for the partial-annotation comparison; the unmodified alternative takes the unknown
voxels as background).

Unknown voxels carry the label -1. StarDist 0.9.1 already handles negative labels in part: its data generator
computes the targets with them set to background and then sets the probability target there to -1, which its
probability loss leaves out; its patch sampler looks for foreground with label > 0, so unknown regions (and the
objects hidden in them) never attract a training crop. Two terms still see the unknown voxels:
  - the probability loss averages over the known voxels of the batch, so a batch without any known voxel gives NaN;
  - the distance loss takes the probability target before the -1 as its mask, so unknown voxels count as background
    in its background regulariser (train_background_reg), which pulls the predicted distances there to 0.
Here the generator also marks the unknown voxels in the mask channel of the distance target (-1), and the losses
    probability: sum over known voxels of BCE / number of known voxels (0 without known voxels);
    distance: StarDist's masked MAE / MSE over the object voxels, normalised by their share as StarDist does, plus the
              background regulariser over the known background only, normalised by the known share;
so that with every voxel known both equal StarDist's own losses, and the targets and predictions at unknown voxels
do not enter the loss or its gradient.
"""
import numpy as np
import tensorflow as tf
from csbdeep.utils.tf import BACKEND as K                                 # the backend of StarDist's own losses
from stardist.models import model3d
from stardist.models.model3d import StarDist3D, StarDistData3D

UNKNOWN = -1


def known_prob_loss(y_true, y_pred):
    """Binary cross-entropy averaged over the known voxels (target >= 0) of the batch; 0 if there are none."""
    y_true = tf.cast(y_true, K.floatx())
    known = tf.cast(y_true >= 0, K.floatx())
    bce = K.binary_crossentropy(tf.maximum(y_true, 0.0), y_pred)
    return tf.reduce_sum(known * bce) / tf.maximum(tf.reduce_sum(known), 1.0)


def known_dist_loss(n_rays, penalty='mae', reg_weight=0.0):
    """Distance loss on [dist (n_rays), mask (1)] targets whose mask is the object probability (>= 0) at known voxels
    and -1 at unknown ones."""
    penalty = {'mae': tf.abs, 'mse': tf.square}[penalty]

    def loss(dist_true_mask, dist_pred):
        dist_true, mask = tf.split(dist_true_mask, num_or_size_splits=[n_rays, -1], axis=-1)
        known = tf.cast(mask >= 0, K.floatx())
        mask = tf.maximum(tf.cast(mask, K.floatx()), 0.0)
        actual = K.mean(mask * penalty(tf.cast(dist_true, K.floatx()) - dist_pred), axis=-1)
        out = actual / (K.mean(mask) + K.epsilon())                      # as stardist generic_masked_loss
        if reg_weight > 0:
            reg = K.mean(known * (1 - mask) * tf.abs(dist_pred), axis=-1)
            out = out + reg_weight * reg / tf.maximum(K.mean(known), K.epsilon())
        return out
    return loss


def known_dist_mae(n_rays):
    """Metric: mean absolute distance error over the object voxels."""
    def relevant_mae(dist_true_mask, dist_pred):
        dist_true, mask = tf.split(dist_true_mask, num_or_size_splits=[n_rays, -1], axis=-1)
        mask = tf.maximum(tf.cast(mask, K.floatx()), 0.0)
        return K.mean(mask * tf.abs(tf.cast(dist_true, K.floatx()) - dist_pred), axis=-1) / (K.mean(mask) + K.epsilon())
    return relevant_mae


class MaskedStarDistData3D(StarDistData3D):
    """StarDist's generator, with the unknown voxels also marked (-1) in the mask channel of the distance target."""

    def __getitem__(self, i):
        x, (prob, dist) = super().__getitem__(i)
        unknown = prob[..., 0] < 0
        if unknown.any():
            dist = dist.copy()
            dist[..., -1][unknown] = UNKNOWN
        return x, (prob, dist)


class MaskedStarDist3D(StarDist3D):
    """StarDist3D whose training leaves voxels labelled -1 out of every loss term."""

    def prepare_for_training(self, optimizer=None):
        super().prepare_for_training(optimizer)                        # callbacks, optimizer
        n_rays, conf = self.config.n_rays, self.config
        self.keras_model.compile(self.keras_model.optimizer,
                                 loss=[known_prob_loss, known_dist_loss(n_rays, conf.train_dist_loss, conf.train_background_reg)],
                                 loss_weights=list(conf.train_loss_weights),
                                 metrics={'dist': [known_dist_mae(n_rays)]})

    def train(self, *args, **kwargs):
        model3d.StarDistData3D = MaskedStarDistData3D                    # StarDist3D.train builds its generators by name
        try:
            return super().train(*args, **kwargs)
        finally:
            model3d.StarDistData3D = StarDistData3D


def with_unknown(labels, unknown_value=65535):
    """Labels for MaskedStarDist3D: unknown_value -> -1 (int32), the rest as given."""
    labels = np.asarray(labels)
    return np.where(labels == unknown_value, UNKNOWN, labels).astype(np.int32)
