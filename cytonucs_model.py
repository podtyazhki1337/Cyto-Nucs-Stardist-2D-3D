# cytonucs_model.py - ПРАВИЛЬНАЯ иерархия

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np


class CytoNucsStarDistModel:
    """
    Иерархическая модель как в train.py, но с двумя decoder'ами.
    """

    def __init__(self, config, name='cytonucs_stardist', basedir='models'):
        self.config = config
        self.name = name
        self.basedir = basedir
        self.ndim = config.ndim
        self.keras_model = self._build_model()

    # cytonucs_model.py - ПРАВИЛЬНАЯ логика pooling контекста
        # cytonucs_model.py - ИСПРАВЛЕННЫЙ _build_model
    def _build_model(self):
        """Build Keras model"""
        cfg = self.config

        # Выбор слоёв
        if self.ndim == 3:
            ConvLayer = layers.Conv3D
            PoolLayer = layers.MaxPooling3D
            UpSampleLayer = layers.UpSampling3D
        else:
            ConvLayer = layers.Conv2D
            PoolLayer = layers.MaxPooling2D
            UpSampleLayer = layers.UpSampling2D

        input_shape = (*cfg.train_patch_size, cfg.n_channel_in)
        inputs = layers.Input(shape=input_shape, name='input')

        # ========== ENCODER (общий) ==========
        features = []
        x = inputs

        for i, n_filters in enumerate([32, 64, 128, 256]):
            x = ConvLayer(n_filters, 3, padding='same', activation='relu', name=f'enc_l{i}_c1')(x)
            x = ConvLayer(n_filters, 3, padding='same', activation='relu', name=f'enc_l{i}_c2')(x)
            features.append(x)
            if i < 3:
                x = PoolLayer(2, name=f'enc_l{i}_pool')(x)

        # ========== DECODER 1: NUCLEUS ==========
        x_nuc = features[-1]

        for i, n_filters in reversed(list(enumerate([32, 64, 128]))):
            x_nuc = UpSampleLayer(2, name=f'nuc_dec_l{i}_up')(x_nuc)
            x_nuc = layers.Concatenate(name=f'nuc_dec_l{i}_concat')([x_nuc, features[i]])
            x_nuc = ConvLayer(n_filters, 3, padding='same', activation='relu', name=f'nuc_dec_l{i}_c1')(x_nuc)
            x_nuc = ConvLayer(n_filters, 3, padding='same', activation='relu', name=f'nuc_dec_l{i}_c2')(x_nuc)

        # Output heads NUCLEUS
        nucleus_prob = ConvLayer(1, 1, padding='same', activation='sigmoid', name='nucleus_prob')(x_nuc)

        # ИСПРАВЛЕНО: Правильная инициализация для абсолютных расстояний
        # Bias ~15 (типичный radius nucleus), weights малые для стабильности
        nucleus_dist = ConvLayer(
            cfg.n_rays, 1,
            padding='same',
            activation='linear',
            name='nucleus_dist',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.keras.initializers.Constant(15.0)  # ← Начальный radius ~15px
        )(x_nuc)
        # ========== DECODER 2: CYTOPLASM (С КОНТЕКСТОМ) ==========
        # Stop gradient для контекста
        nucleus_prob_context = layers.Lambda(
            lambda x: tf.stop_gradient(x),
            name='nucleus_prob_stop_grad'
        )(nucleus_prob)

        x_cyto = features[-1]  # Bottleneck

        # Decoder с контекстом на каждом уровне
        for i, n_filters in reversed(list(enumerate([32, 64, 128]))):
            # 1. Upsample decoder
            x_cyto = UpSampleLayer(2, name=f'cyto_dec_l{i}_up')(x_cyto)

            # 2. Создаём контекст правильного размера
            remaining_upsamples = i  # Сколько ещё upsampling'ов после этого
            context_at_level = nucleus_prob_context

            for _ in range(remaining_upsamples):
                context_at_level = PoolLayer(2)(context_at_level)

            # 3. Concatenate
            x_cyto = layers.Concatenate(name=f'cyto_dec_l{i}_concat')([
                x_cyto,
                features[i],
                context_at_level
            ])

            # 4. Convolutions
            x_cyto = ConvLayer(n_filters, 3, padding='same', activation='relu', name=f'cyto_dec_l{i}_c1')(x_cyto)
            x_cyto = ConvLayer(n_filters, 3, padding='same', activation='relu', name=f'cyto_dec_l{i}_c2')(x_cyto)

        # Final context fusion
        x_cyto = layers.Concatenate(name='cyto_final_concat')([x_cyto, nucleus_prob_context])
        x_cyto = ConvLayer(32, 3, padding='same', activation='relu', name='cyto_final_conv')(x_cyto)

        # Output heads CYTOPLASM
        cytoplasm_prob = ConvLayer(1, 1, padding='same', activation='sigmoid', name='cytoplasm_prob')(x_cyto)

        # ИСПРАВЛЕНО: Bias ~40 (типичный radius cytoplasm)
        cytoplasm_dist = ConvLayer(
            cfg.n_rays, 1,
            padding='same',
            activation='linear',
            name='cytoplasm_dist',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.keras.initializers.Constant(40.0)  # ← Начальный radius ~40px
        )(x_cyto)

        outputs = {
            'nucleus_prob': nucleus_prob,
            'nucleus_dist': nucleus_dist,
            'cytoplasm_prob': cytoplasm_prob,
            'cytoplasm_dist': cytoplasm_dist,
        }

        return Model(inputs=inputs, outputs=outputs, name=self.name)


    def summary(self):
        self.keras_model.summary()

    def save_weights(self, filepath):
        self.keras_model.save_weights(filepath)

    def load_weights(self, filepath):
        self.keras_model.load_weights(filepath)

    def _axes_tile_overlap(self, query_axes):
        if self.ndim == 3:
            return tuple(self.config.train_patch_size[i] for i, ax in enumerate('ZYX') if ax in query_axes)
        else:
            return tuple(self.config.train_patch_size[i] for i, ax in enumerate('YX') if ax in query_axes)

    def _guess_n_tiles(self, img):
        axes = 'ZYX' if self.ndim == 3 else 'YX'
        return tuple(max(1, int(np.ceil(s / p)))
                     for s, p in zip(img.shape[:self.ndim], self.config.train_patch_size))


def build_cytonucs_model(config, name='cytonucs_stardist', basedir='models'):
    return CytoNucsStarDistModel(config, name=name, basedir=basedir)