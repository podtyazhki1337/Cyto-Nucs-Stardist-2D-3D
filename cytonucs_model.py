"""
CytoNucs StarDist Model Architecture
Shared encoder + dual decoders (nucleus + cytoplasm)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from stardist.models import StarDist2D, StarDist3D
import numpy as np


class CytoNucsStarDistModel:
    """
    Dual-head StarDist model for simultaneous nucleus and cytoplasm segmentation.

    Architecture:
    - Shared encoder (U-Net or ResNet backbone)
    - Nucleus decoder: probability + distance maps
    - Cytoplasm decoder: probability + distance maps
    """

    def __init__(self, config, name='cytonucs_stardist', basedir='models'):
        self.config = config
        self.name = name
        self.basedir = basedir
        self.ndim = config.ndim

        # Build the model
        self.keras_model = self._build_model()

    def _build_model(self):
        """Build the full CytoNucs model"""
        cfg = self.config

        # Input
        if self.ndim == 3:
            input_shape = (*cfg.train_patch_size, cfg.n_channel_in)
        else:
            input_shape = (*cfg.train_patch_size, cfg.n_channel_in)

        inputs = layers.Input(shape=input_shape, name='input')

        # Shared encoder
        if cfg.shared_encoder:
            encoder_features = self._build_encoder(inputs)
        else:
            encoder_features_nucleus = self._build_encoder(inputs, name_prefix='nucleus_')
            encoder_features_cytoplasm = self._build_encoder(inputs, name_prefix='cytoplasm_')

        # Nucleus decoder
        if cfg.shared_encoder:
            nucleus_prob, nucleus_dist = self._build_decoder(
                encoder_features, name_prefix='nucleus_'
            )
        else:
            nucleus_prob, nucleus_dist = self._build_decoder(
                encoder_features_nucleus, name_prefix='nucleus_'
            )

        # Cytoplasm decoder
        if cfg.shared_encoder:
            cytoplasm_prob, cytoplasm_dist = self._build_decoder(
                encoder_features, name_prefix='cytoplasm_'
            )
        else:
            cytoplasm_prob, cytoplasm_dist = self._build_decoder(
                encoder_features_cytoplasm, name_prefix='cytoplasm_'
            )

        # Outputs: [nucleus_prob, nucleus_dist, cytoplasm_prob, cytoplasm_dist]
        outputs = {
            'nucleus_prob': nucleus_prob,
            'nucleus_dist': nucleus_dist,
            'cytoplasm_prob': cytoplasm_prob,
            'cytoplasm_dist': cytoplasm_dist,
        }

        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model

    def _build_encoder(self, inputs, name_prefix=''):
        """
        Build U-Net style encoder.
        Returns feature maps at multiple scales.
        """
        cfg = self.config
        ConvLayer = layers.Conv3D if self.ndim == 3 else layers.Conv2D
        PoolLayer = layers.MaxPooling3D if self.ndim == 3 else layers.MaxPooling2D

        # Encoder levels
        features = []
        x = inputs

        # Level 0 (highest resolution)
        x = ConvLayer(32, 3, padding='same', activation='relu',
                      name=f'{name_prefix}enc_l0_c1')(x)
        x = ConvLayer(32, 3, padding='same', activation='relu',
                      name=f'{name_prefix}enc_l0_c2')(x)
        features.append(x)
        x = PoolLayer(2, name=f'{name_prefix}enc_l0_pool')(x)

        # Level 1
        x = ConvLayer(64, 3, padding='same', activation='relu',
                      name=f'{name_prefix}enc_l1_c1')(x)
        x = ConvLayer(64, 3, padding='same', activation='relu',
                      name=f'{name_prefix}enc_l1_c2')(x)
        features.append(x)
        x = PoolLayer(2, name=f'{name_prefix}enc_l1_pool')(x)

        # Level 2
        x = ConvLayer(128, 3, padding='same', activation='relu',
                      name=f'{name_prefix}enc_l2_c1')(x)
        x = ConvLayer(128, 3, padding='same', activation='relu',
                      name=f'{name_prefix}enc_l2_c2')(x)
        features.append(x)
        x = PoolLayer(2, name=f'{name_prefix}enc_l2_pool')(x)

        # Bottleneck
        x = ConvLayer(256, 3, padding='same', activation='relu',
                      name=f'{name_prefix}enc_bottleneck_c1')(x)
        x = ConvLayer(256, 3, padding='same', activation='relu',
                      name=f'{name_prefix}enc_bottleneck_c2')(x)
        features.append(x)

        return features

    def _build_decoder(self, encoder_features, name_prefix=''):
        """
        Build decoder for one head (nucleus or cytoplasm).
        Returns probability map and distance map.
        """
        cfg = self.config
        ConvLayer = layers.Conv3D if self.ndim == 3 else layers.Conv2D
        UpSampleLayer = layers.UpSampling3D if self.ndim == 3 else layers.UpSampling2D

        # Start from bottleneck
        x = encoder_features[-1]

        # Level 2 (upsample + skip connection)
        x = UpSampleLayer(2, name=f'{name_prefix}dec_l2_up')(x)
        x = layers.Concatenate(name=f'{name_prefix}dec_l2_concat')([x, encoder_features[2]])
        x = ConvLayer(128, 3, padding='same', activation='relu',
                      name=f'{name_prefix}dec_l2_c1')(x)
        x = ConvLayer(128, 3, padding='same', activation='relu',
                      name=f'{name_prefix}dec_l2_c2')(x)

        # Level 1
        x = UpSampleLayer(2, name=f'{name_prefix}dec_l1_up')(x)
        x = layers.Concatenate(name=f'{name_prefix}dec_l1_concat')([x, encoder_features[1]])
        x = ConvLayer(64, 3, padding='same', activation='relu',
                      name=f'{name_prefix}dec_l1_c1')(x)
        x = ConvLayer(64, 3, padding='same', activation='relu',
                      name=f'{name_prefix}dec_l1_c2')(x)

        # Level 0
        x = UpSampleLayer(2, name=f'{name_prefix}dec_l0_up')(x)
        x = layers.Concatenate(name=f'{name_prefix}dec_l0_concat')([x, encoder_features[0]])
        x = ConvLayer(32, 3, padding='same', activation='relu',
                      name=f'{name_prefix}dec_l0_c1')(x)
        x = ConvLayer(32, 3, padding='same', activation='relu',
                      name=f'{name_prefix}dec_l0_c2')(x)

        # Output heads
        # Probability (1 channel, sigmoid activation)
        prob = ConvLayer(1, 1, padding='same', activation='sigmoid',
                         name=f'{name_prefix}prob')(x)

        # Distance (n_rays channels, linear activation)
        dist = ConvLayer(cfg.n_rays, 1, padding='same', activation='linear',
                         name=f'{name_prefix}dist')(x)

        return prob, dist

    def summary(self):
        """Print model summary"""
        self.keras_model.summary()

    def save_weights(self, filepath):
        """Save model weights"""
        self.keras_model.save_weights(filepath)

    def load_weights(self, filepath):
        """Load model weights"""
        self.keras_model.load_weights(filepath)

    def _axes_tile_overlap(self, axes):
        """Helper for FOV computation"""
        # Simplified version
        return self.config.train_patch_size

    def _guess_n_tiles(self, img):
        """Helper for tiling"""
        return None


# Convenience function to create model from config
def build_cytonucs_model(config, name='cytonucs_stardist', basedir='models'):
    """
    Factory function to create CytoNucs StarDist model.

    Args:
        config: CytoNucsConfig object
        name: model name
        basedir: base directory for saving

    Returns:
        CytoNucsStarDistModel instance
    """
    return CytoNucsStarDistModel(config, name=name, basedir=basedir)