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


# Can't get tf.logging.warning to work...
def myprint(message):
    tf.print(message)
    return 1


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

    if tf.executing_eagerly():
        return [ex for ex in it]
    else:
        next_example = it.get_next()
        examples = []
        with tf.Session() as sess:
            try:
                while True:
                    examples.append(sess.run(next_example))
            except tf.errors.OutOfRangeError:
                pass
        return examples


def tfrecord2data(tfrecord_filename, feature_extractor, class_list):
    dataset = tf.data.TFRecordDataset([tfrecord_filename])
    dataset = dataset.map(lambda x: serialized2data(x, feature_extractor.example_shape, class_list))
    it = dataset.make_one_shot_iterator()

    if tf.executing_eagerly():
        return [ex for ex in it]
    else:
        next_example = it.get_next()
        examples = []
        with tf.Session() as sess:
            try:
                while True:
                    examples.append(sess.run(next_example))
            except tf.errors.OutOfRangeError:
                pass
        return examples


def serialized2example(serialized_data, feature_shape):
    features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'times': tf.io.FixedLenFeature([2], tf.float32),
        'data': tf.io.FixedLenFeature(feature_shape, tf.float32),
        'labels': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(serialized_data, features)


def serialized2data(
        serialized_data,
        feature_shape,
        class_list,
        training=True):
    """Generate features, labels and, if training is False, filenames and times.
    Labels are indices of original label in class_list.
    """

    features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'times': tf.io.FixedLenFeature([2], tf.float32),
        'data': tf.io.FixedLenFeature(feature_shape, tf.float32),
        'labels': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_data, features)

    # reshape data to channels_first format
    data = tf.reshape(example['data'], (1, feature_shape[0], feature_shape[1]))

    # one-hot encode labels
    labels = tf.strings.to_number(
        tf.string_split([example['labels']], '#').values,
        out_type=tf.int32
    )

    # get intersection of class_list and labels
    labels = tf.squeeze(
        tf.sparse.to_dense(
            tf.sets.intersection(
                tf.expand_dims(labels, axis=0),
                tf.expand_dims(class_list, axis=0)
            )
        ),
        axis=0
    )

    # sort class_list and get indices of labels in class_list
    class_list = tf.sort(class_list)
    labels = tf.where(
        tf.equal(
            tf.expand_dims(labels, axis=1),
            class_list)
    )[:,1]

    tf.cond(
        tf.math.logical_and(training, tf.equal(tf.size(labels), 0)),
        true_fn=lambda:myprint(tf.strings.format('File {} has no label', example['filename'])),
        false_fn=lambda:1
    )

    one_hot = tf.cond(
        tf.equal(tf.size(labels), 0),
        true_fn=lambda: tf.zeros(tf.size(class_list)),
        false_fn=lambda: tf.reduce_max(tf.one_hot(labels, tf.size(class_list)), 0)
    )

    if training:
        return (data, one_hot)
    else:
        return (data, one_hot, example['filename'], example['times'])


def tfrecords2tfdataset(
        files,
        example_shape,
        class_list,
        training=True,
        batch_size=32
):
    """Returns a tensorflow's dataset from a list of tfrecords."""

    if training:
        files = tf.convert_to_tensor(files, dtype=dtypes.string)
        files = tf.data.Dataset.from_tensor_slices(files)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=8)
        # dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(100), cycle_length=8)
        dataset = dataset.map(lambda x: serialized2data(x, example_shape, class_list, training))
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()  # Repeat the input indefinitely.
    else:
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(lambda x: serialized2data(x, example_shape, class_list, training))

    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


