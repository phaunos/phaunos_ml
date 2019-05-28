import tensorflow as tf
from tensorflow import keras


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
    feature = {
        'filename': _bytes_feature([filename.encode()]),
        'times': _float_feature([start_time, end_time]),
        'data': _float_feature(data.flatten()),
        'labels': _bytes_feature(["#".join(str(l) for l in labels).encode()]),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def tfrecord2example(tfrecord_filename, feature_extractor):
    dataset = tf.data.TFRecordDataset([tfrecord_filename])
    dataset = dataset.map(lambda x: serialized2example(x, feature_extractor.example_shape))
    it = dataset.make_one_shot_iterator()
    return [ex for ex in it]


def tfrecord2data(tfrecord_filename, feature_extractor, n_classes):
    dataset = tf.data.TFRecordDataset([tfrecord_filename])
    dataset = dataset.map(lambda x: serialized2data(x, feature_extractor.example_shape, n_classes))
    it = dataset.make_one_shot_iterator()
    return [ex for ex in it]


def serialized2example(serialized_data, feature_shape):
    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'times': tf.FixedLenFeature([2], tf.float32),
        'data': tf.FixedLenFeature(feature_shape, tf.float32),
        'labels': tf.FixedLenFeature([], tf.string),
    }
    return tf.parse_single_example(serialized_data, features)


def serialized2data(serialized_data, feature_shape, n_classes):

    features = {
        'filename': tf.FixedLenFeature([], tf.string),
        'times': tf.FixedLenFeature([2], tf.float32),
        'data': tf.FixedLenFeature(feature_shape, tf.float32),
        'labels': tf.FixedLenFeature([], tf.string),
    }
    example = tf.parse_single_example(serialized_data, features)

    # reshape data to channels_first format
    data = tf.reshape(example['data'], (1, feature_shape[0], feature_shape[1]))

    # one-hot encode labels
    labels = tf.strings.to_number(
        tf.string_split([example['labels']], '#').values,
        out_type=tf.int64
    )

    one_hot = tf.cond(
        tf.equal(tf.size(labels), 0),
        lambda: tf.zeros(n_classes),
        lambda: tf.reduce_max(tf.one_hot(labels, n_classes), 0)
    )

    return (data, one_hot)
