import tensorflow as tf
from tensorflow.keras import layers

from .layer_utils import conv2d_bn


def build_model(x, n_classes, data_format='channels_first'):

    # First layer
    x = layers.Conv2D(
        32, (3,3),
        strides=(1,1),
        padding='valid',
        use_bias=True,
        data_format=data_format,
        name='conv_1')(x)
    x = layers.MaxPooling2D(
        pool_size=(1, 2),
        strides=(1,2),
        data_format=data_format,
        name='mp_1')(x)
    x = layers.Activation('relu')(x)

    # Second layer
    x = layers.Conv2D(
        32, (3,3),
        strides=(1,1),
        padding='valid',
        use_bias=True,
        data_format=data_format,
        name='conv_2')(x)
    x = layers.MaxPooling2D(
        pool_size=(1, 2),
        strides=(1,2),
        data_format=data_format,
        name='mp_2')(x)
    x = layers.Activation('relu')(x)

    # Third layer
    x = layers.Conv2D(
        32, (3,3),
        strides=(1,1),
        padding='valid',
        use_bias=True,
        data_format=data_format,
        name='conv_3')(x)
    x = layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2,2),
        data_format=data_format,
        name='mp_3')(x)
    x = layers.Activation('relu')(x)

    # Fourth layer
    x = layers.Conv2D(
        64, (3,3),
        strides=(1,1),
        padding='valid',
        use_bias=True,
        data_format=data_format,
        name='conv_4')(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D(
        data_format=data_format,
        name='av_pool'
    )(x)

    # Classification
    x = layers.Dense(n_classes, activation='softmax')(x)

    return x
