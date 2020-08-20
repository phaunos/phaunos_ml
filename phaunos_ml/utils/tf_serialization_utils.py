import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow import keras


"""
Tensorflow serialization utils
"""


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_data(filename, start_time, end_time, data, labels):
    if not data.dtype == np.float32:
        # TODO: add dtype to serialized data to allow 16 bits
        data = data.astype(np.float32)
    feature = {
        'filename': _bytes_feature([filename.encode()]),
        'times': _float_feature([start_time, end_time]),
        'shape': _int64_feature(data.shape),
        'data': _bytes_feature([data.tobytes()]),
        'labels': _bytes_feature(["#".join(str(l) for l in labels).encode()]),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def labelstr2onehot(labelstr, class_list):
    """One-hot label encoding
    
    labelstr: label, encoded as 'id1#...#idN', where idn is an int
    class_list: list of classes used as the reference for the one hot encoding    
    """

    # parse string
    labels = tf.cond(
        tf.equal(tf.strings.length(labelstr), 0),
        true_fn=lambda:tf.constant([], dtype=tf.int32),
        false_fn=lambda:tf.strings.to_number(
            tf.strings.split(labelstr, '#'),
            out_type=tf.int32
        )
    )

    # sort class_list and get indices of labels in class_list
    class_list = tf.sort(class_list)
    labels = tf.where(
        tf.equal(
            tf.expand_dims(labels, axis=1),
            class_list)
    )[:,1]

    return tf.cond(
        tf.equal(tf.size(labels), 0),
        true_fn=lambda: tf.zeros(tf.size(class_list)),
        false_fn=lambda: tf.reduce_max(tf.one_hot(labels, tf.size(class_list)), 0)
    )


def serialized2example(serialized_data):
    features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'times': tf.io.FixedLenFeature([2], tf.float32),
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(serialized_data, features) # TODO parse_example on batch


def serialized2data(
        serialized_data,
        class_list,
        one_hot_label=True
):
    """Generates data, filename, segment time bounds and binary (if one_hot_label=False)
    or one hot encoded (if one_hot_label=True) labels.

    Args:
        serialized_data: data serialized using utils.tf_utils.serialize_data
        class_list: list of class ids (used for one-hot encoding the labels)
        one_hot_label (bool): whether to return the label as one-hot vector
            (multi-class classification) or binary value (binary classification)
    """

    example = serialized2example(serialized_data)

    # reshape data to original shape
    data = tf.reshape(tf.io.decode_raw(example['data'], tf.float32), example['shape'])

    if one_hot_label:
        label = labelstr2onehot(example['labels'], class_list)
    else:
        tf.debugging.assert_equal(tf.convert_to_tensor(class_list).shape, (2,))
        tf.debugging.Assert(
            tf.math.logical_or(
                tf.math.equal(
                    tf.strings.to_number(example['labels'], out_type=tf.int32),
                    class_list[0]),
                tf.math.equal(
                    tf.strings.to_number(example['labels'], out_type=tf.int32),
                    class_list[1])
            ),
            [example['labels'], class_list]
        )
        label = tf.cond(
            tf.strings.to_number(example['labels'], out_type=tf.int32)==class_list[0],
            true_fn=lambda:tf.constant(0, dtype=tf.int32),
            false_fn=lambda:tf.constant(1, dtype=tf.int32),
        )

    return (data, label, example['filename'], example['times'])
