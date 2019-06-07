import os
import argparse
import pathlib
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.python.framework import dtypes
from scipy import interpolate


from phaunos_ml.utils.tf_utils import serialized2data
from phaunos_ml.models import simple_cnn


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def filelist2dataset(files, example_shape, class_list, batch_size=32, nolabel_warning=True):
    files = tf.convert_to_tensor(files, dtype=dtypes.string)
    files = tf.data.Dataset.from_tensor_slices(files)
#    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(100), cycle_length=8)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=8)
    dataset = dataset.map(lambda x: serialized2data(x, example_shape, class_list, nolabel_warning))
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def train(
        dataset_train,
        n_train_batches,
        feature_extractor,
        out_dir,
        n_classes,
        multilabel=False,
        batch_size=32,
        epochs=10,
        dataset_valid=None,
        n_valid_batches=None
):


    ###############
    # build model #
    ###############

    h = feature_extractor.example_shape[0]
    w = feature_extractor.example_shape[1]

    inputs = keras.Input(shape=(1, h, w), batch_size=batch_size, name='mels')
    outputs = simple_cnn.build_model(inputs, n_classes, multilabel=multilabel)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # define optimizer
    optimizer = keras.optimizers.Adam(
        lr=0.01, beta_1=0.5, beta_2=0.999)

    if multilabel:
        loss = keras.losses.binary_crossentropy
        print("yeah")
    else:
        loss = keras.losses.categorical_crossentropy
        print("oh")

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])


    ########################
    # Define the callbacks #
    ########################

    write_dir = os.path.join(out_dir, str(int(time())))

    # Learning rate scheduler
#    lr_interpolate = interpolate.interp1d([0, 34, 54], [0.01, 0.0001, 1e-05], kind='linear')
#    lr_scheduler = LearningRateScheduler(
#        lambda x:float(lr_interpolate(x)),
#        verbose=1)
    
    # Tensorboard
    tb_log_dir = os.path.join(write_dir, "tb_logs")
    pathlib.Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
    tb = TensorBoard(log_dir=tb_log_dir)

    # Model checkpoint
#    model_dir = join(write_dir, "models")
#    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
#    mc = ModelCheckpoint(
#        join(model_dir, "model.{epoch:02d}-{val_acc:.2f}.h5"),
#        monitor='val_acc',
#        verbose=1,
#        save_best_only=True)

#    callback_list = [lr_scheduler, tb, mc]
    callback_list = [tb]


    #########
    # train #
    #########

    model.fit(
        dataset_train,
        steps_per_epoch=n_train_batches,
        validation_data=dataset_valid,
        validation_steps=n_valid_batches,
        epochs=epochs,
        callbacks=callback_list,
        verbose=2)
