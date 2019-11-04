import tensorflow as tf
from tensorflow.keras import layers

from .layer_utils import conv2d_bn


def build_model(x, n_classes, multilabel=False, data_format='channels_first'):

    # format must be channels_first (faster on NVIDIA GPUs)

    x = conv2d_bn(x, 32, (3, 3), data_format=data_format, name='l1')
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), data_format=data_format, name='l1_mp')(x)
    x = conv2d_bn(x, 32, (5, 5), data_format=data_format, name='l2')
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), data_format=data_format, name='l2_mp')(x)
    x = conv2d_bn(x, 64, (5, 5), data_format=data_format, name='l3')
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), data_format=data_format, name='l3_mp')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='elu', name='logits')(x)
    if multilabel:
        x = layers.Dense(n_classes, activation='sigmoid')(x)
    else:
        x = layers.Dense(n_classes, activation='softmax')(x)

    return x
