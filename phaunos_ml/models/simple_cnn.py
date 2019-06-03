import tensorflow as tf
from tensorflow.keras import layers

from .layer_utils import conv2d_bn


def build_model(x, n_classes, data_format='channels_first'):

    # format must be channels_first (faster on NVIDIA GPUs)

    x = conv2d_bn(x, 32, (3, 3), data_format=data_format, name='l1')
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), data_format=data_format, name='l1_mp')(x)
    x = conv2d_bn(x, 32, (5, 5), data_format=data_format, name='l2')
    x = layers.MaxPooling2D(pool_size=(5, 5), strides=(3,3), data_format=data_format, name='l2_mp')(x)
    x = conv2d_bn(x, 64, (5, 5), data_format=data_format, name='l3')
    x = layers.MaxPooling2D(pool_size=(5, 5), strides=(3,3), data_format=data_format, name='l3_mp')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='elu')(x)
    x = layers.Dense(n_classes, activation='softmax')(x)

    return x