import tensorflow as tf
from tensorflow.keras import layers

from .layer_utils import conv2d_bn


def build_model(x, n_classes, multilabel=False, data_format='channels_first'):

    # format must be channels_first (faster on NVIDIA GPUs)

    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dense(n_classes, activation='sigmoid')(x)

    return x
