import tensorflow as tf
from tensorflow.keras import layers


def conv2d_bn(x,
              filters,
              filter_shape,
              padding='valid',
              strides=(1, 1),
              data_format='channels_first',
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        num_filters: num filters in `Conv2D`.
        filter_shape: shape of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        data_format: 'channels_first' or 'channels_last'
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 1 if data_format=='channels_first' else -1
    x = layers.Conv2D(
        filters, filter_shape,
        strides=strides,
        padding=padding,
        use_bias=False, # batch norm already does some shifting
        data_format=data_format,
        name=conv_name)(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        scale=False, # scaling will be done by the next layer (only for linear activation)
        name=bn_name)(x)
    return layers.Activation('relu', name=name)(x)
