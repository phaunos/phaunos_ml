import os
import json
from time import time
import pathlib
import librosa

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

from phaunos_ml.utils import tf_utils, feature_utils, dataug_utils, tf_feature_utils, dataset_utils
from phaunos_ml.models import simple_cnn


# Path to data
DATASET_ROOT = '/home/jul/data'
DATASET_FILE = '/home/jul/data/ingerop/subset_1572008350/subset_1572008350.csv'
DATASET_DIR = os.path.dirname(DATASET_FILE)
FEATEX_CFG = os.path.join(DATASET_DIR, 'features/featex_config.json')
AUDIO_DIRNAME = 'audio/wav_22050hz_MLR'
ANNOTATION_DIRNAME = 'annotations_ingerop'

# tf.data.Dataset pipeline config
BATCH_SIZE = 32
SHUFFLE_FILES = True
INTERLEAVE_FILES = True
SHUFFLE_EXAMPLES = True
PREFETCH_DATA = True
PARALLEL_CALLS = True

# Mel spectrogram config
N_FFT = 512
HOP_LENGTH = 128
FMIN = 500
FMAX = 8000
N_MELS = 64
N_TIME_BINS = 169 # inspect the data to find out

# Data Augmentation
DA_MIXUP = True
DA_TIME_WARP = False
DA_MASKING = True


def tfrecords2dataset(
        tfrecords,
        feature_extractor,
        class_list,
        batch_size = 8,
        shuffle_files=True,
        interleave_files=False,
        shuffle_examples=True
):

    # Get list of files and shuffle
    files = tf.convert_to_tensor(tfrecords, dtype=dtypes.string)
    files = tf.data.Dataset.from_tensor_slices(files)

    if shuffle_files:
        files = files.shuffle(5000, reshuffle_each_iteration=True)

    # Read TFrecords
    if not interleave_files:
        dataset = tf.data.TFRecordDataset(files)
    else:
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=10)

    # deserialize to feature (with shape (1,1,22050)) and one-hot encoded labels
    dataset = dataset.map(lambda x: tf_utils.serialized2data(
        x,
        feature_extractor.feature_shape,
        class_list,
        training=True),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if PARALLEL_CALLS else None)

    # shuffle, repeat and batch
    if shuffle_examples:
        dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    return dataset.batch(batch_size, drop_remainder=True)


def run():

    # Get class list
    class_list = sorted([int(i) for i in next(open(DATASET_FILE, 'r')).strip().split(':')[1].split(',')])
    print("\nClass list:")
    print(class_list)

    # Create feature extractor from config
    feature_extractor = feature_utils.AudioSegmentExtractor.from_config(FEATEX_CFG)
    print("\nFeature extractor config:")
    print(json.load(open(FEATEX_CFG, 'r')))


    #################
    # Get TFRecords #
    #################

    # Get training TFRecords
    training_dataset_file = DATASET_FILE.replace('.csv', '.train.csv')
    tfrecord_path = os.path.dirname(training_dataset_file)
    training_tfrecords = [os.path.join(tfrecord_path, 'features/positive', line.strip().replace('.wav', '.tf')) \
                          for line in open(training_dataset_file, 'r') \
                          if not line.startswith('#')]

    # Get validation TFRecords
    valid_dataset_file = DATASET_FILE.replace('.csv', '.test.csv')
    valid_tfrecords = [os.path.join(tfrecord_path, 'features/positive', line.strip().replace('.wav', '.tf')) \
                          for line in open(valid_dataset_file, 'r') \
                          if not line.startswith('#')]

    # Get TFRecords used for data augmentation (Audioset data from training dataset)
    dataug_tfrecords = [os.path.join(tfrecord_path, 'features/positive', line.strip().replace('.wav', '.tf')) \
                          for line in open(training_dataset_file, 'r') \
                          if not line.startswith('#') and 'audioset' in line]


    ###################
    # Set up datasets #
    ###################

    training_dataset = tfrecords2dataset(
        training_tfrecords,
        feature_extractor,
        class_list,
        batch_size=BATCH_SIZE,
        shuffle_files=SHUFFLE_FILES,
        interleave_files=INTERLEAVE_FILES,
        shuffle_examples=SHUFFLE_EXAMPLES
    )

    valid_dataset = tfrecords2dataset(
        valid_tfrecords,
        feature_extractor,
        class_list,
        batch_size=BATCH_SIZE,
        shuffle_files=SHUFFLE_FILES,
        interleave_files=INTERLEAVE_FILES,
        shuffle_examples=SHUFFLE_EXAMPLES
    )

    dataug_dataset = tfrecords2dataset(
        dataug_tfrecords,
        feature_extractor,
        class_list,
        batch_size=BATCH_SIZE,
        shuffle_files=SHUFFLE_FILES,
        interleave_files=INTERLEAVE_FILES,
        shuffle_examples=SHUFFLE_EXAMPLES
    )


    #####################################
    # Data augmentation / preprocessing #
    #####################################

    # data augmentation on time-domain signal (waveform)
    if DA_MIXUP:
        mixup = dataug_utils.Mixup(max_weight=0.4)
        dataset = tf.data.Dataset.zip((training_dataset, dataug_dataset))
        dataset = dataset.map(lambda dataset1, dataset2: (
            mixup.process(dataset1[0], dataset1[1], dataset2[0], dataset2[1], BATCH_SIZE)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE if PARALLEL_CALLS else None)
    else:
        dataset = dataset_training

    # compute mel spectrogram
    melspec_ex = tf_feature_utils.MelSpectrogram(feature_extractor.sr, N_FFT, HOP_LENGTH, N_MELS, fmin=FMIN, fmax=FMAX, log=False)
    dataset = dataset.map(lambda data, labels: (
        tf.expand_dims(melspec_ex.process(tf.squeeze(data)), 1),
        labels),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if PARALLEL_CALLS else None)
    valid_dataset = valid_dataset.map(lambda data, labels: (
        tf.expand_dims(melspec_ex.process(tf.squeeze(data)), 1),
        labels),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if PARALLEL_CALLS else None)
    
    # data augmentation on time-frequency-domain signal (spectrogram)
    if DA_TIME_WARP or DA_MASKING:
        # Convert from channels_first (NCHW) to channels_last (NHWC) format
        # to match SpecAugment requirement
        dataset = dataset.map(lambda data, labels: (tf.transpose(data, [0,2,3,1]), labels))

        if DA_TIME_WARP:
            dataset = dataset.map(lambda data, labels: (tf.py_function(dataug_utils.time_warp, [data, 5], tf.float32), labels),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE if PARALLEL_CALLS else None)

        if DA_MASKING:
            # Time masking
            dataset = dataset.map(lambda data, labels: (tf.py_function(dataug_utils.time_mask, [data, 10], tf.float32), labels),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE if PARALLEL_CALLS else None)
            # Frequency masking
            dataset = dataset.map(lambda data, labels: (tf.py_function(dataug_utils.frequency_mask, [data, 5], tf.float32), labels),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE if PARALLEL_CALLS else None)

        # Back to channels_first (NCHW)
        dataset = dataset.map(lambda data, labels: (tf.transpose(data, [0,3,1,2]), labels))

    # Take the log of the mel-spectrogram
    dataset = dataset.map(lambda data, labels: (tf.math.log(data + tf_feature_utils.LOG_OFFSET), labels))
    valid_dataset = valid_dataset.map(lambda data, labels: (tf.math.log(data + tf_feature_utils.LOG_OFFSET), labels))

    if PREFETCH_DATA:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    ################
    # Set up model #
    ################

    # build model
    inputs = tf.keras.Input(shape=(1, N_MELS, N_TIME_BINS), batch_size=BATCH_SIZE, name='mels')
    outputs = simple_cnn.build_model(inputs, len(class_list), multilabel=True)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # compile model
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    loss = tf.keras.losses.binary_crossentropy
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy()])


    ################################
    # Define Tensorboard callbacks #
    ################################

    out_dir = os.path.join(DATASET_DIR, 'run_' + str(int(time())))
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Learning rate scheduler
    #    lr_interpolate = interpolate.interp1d([0, 34, 54], [0.01, 0.0001, 1e-05], kind='linear')
    #    lr_scheduler = LearningRateScheduler(
    #        lambda x:float(lr_interpolate(x)),
    #        verbose=1)

    # Tensorboard
    tb_log_dir = os.path.join(out_dir, "tb_logs")
    pathlib.Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
    tb = TensorBoard(log_dir=tb_log_dir)

    # Model checkpoint
    model_dir = os.path.join(out_dir, "models")
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    mc = ModelCheckpoint(
        os.path.join(model_dir, 'model.{epoch:02d}-' +
                     '{val_binary_accuracy:.2f}.h5'),
        monitor='val_binary_accuracy',
        verbose=1,
        save_best_only=True)

    #    callback_list = [lr_scheduler, tb, mc]
    callback_list = [tb, mc]


    #########################
    # Get number of batches #
    #########################

    # get numbers of batches in training dataset
    n_train_batches, n_train_examples_per_class = dataset_utils.dataset_stat_per_example(
        DATASET_ROOT,
        DATASET_FILE.replace('.csv', '.train.csv'),
        os.path.join(DATASET_DIR, 'features/positive/'),
        feature_extractor.feature_shape,
        class_list,
        batch_size=BATCH_SIZE,
        audio_dirname=AUDIO_DIRNAME,
        annotation_dirname=ANNOTATION_DIRNAME
    )
    n_valid_batches, n_train_examples_per_class = dataset_utils.dataset_stat_per_example(
        DATASET_ROOT,
        DATASET_FILE.replace('.csv', '.test.csv'),
        os.path.join(DATASET_DIR, 'features/positive/'),
        feature_extractor.feature_shape,
        class_list,
        batch_size=BATCH_SIZE,
        audio_dirname=AUDIO_DIRNAME,
        annotation_dirname=ANNOTATION_DIRNAME
    )

    print(f'Number of training batches: {n_train_batches}')
    print(f'Number of validation batches: {n_valid_batches}')

    #########
    # Train #
    #########

    model.fit(
        dataset,
        steps_per_epoch=n_train_batches,
        validation_data=valid_dataset,
        validation_steps=n_valid_batches,
        epochs=50,
        callbacks=callback_list,
        verbose=2)


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

#    parser = argparse.ArgumentParser()
#    parser.add_argument("config_filename", type=str)
#    args = parser.parse_args()

    run()

