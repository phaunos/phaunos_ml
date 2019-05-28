import os
import argparse
import pathlib
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from scipy import interpolate


from phaunos_ml.utils.tf_utils import serialized2featurelabel
from phaunos_ml.models import simple_cnn


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


MEL_SHAPE = [83, 128]
BATCH_SIZE = 32
N_CLASSES = 50


def input_fn(tfrecords_path):
    files = tf.data.Dataset.list_files(os.path.join(tfrecords_path, '*.tf'), shuffle=True) 
#    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(100), cycle_length=8)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=8)
    dataset = dataset.map(lambda x: serialized2featurelabel(x, MEL_SHAPE, N_CLASSES))
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('tfrecords_train_path', metavar='tfrecords_train_path', type=str,
                        help='Path to the training tfrecord files.')
    parser.add_argument('tfrecords_valid_path', metavar='tfrecords_valid_path', type=str,
                        help='Path to the validating tfrecord files.')
    parser.add_argument('out_path', metavar='out_path', type=str,
                        help='Path to write output files to.')
    args = parser.parse_args()

    # data iterator
    dataset_train = input_fn(args.tfrecords_train_path)

    # build model
    inputs = keras.Input(shape=(1, MEL_SHAPE[0], MEL_SHAPE[1]), batch_size=BATCH_SIZE, name='mels')
    outputs = simple_cnn.build_model(inputs, N_CLASSES)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    ########################
    # Define the callbacks #
    ########################

    write_dir = os.path.join(args.out_path, str(int(time())))

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


    # define optimizer
    optimizer = keras.optimizers.Adam(
        lr=0.01, beta_1=0.5, beta_2=0.999)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(
        dataset_train,
        steps_per_epoch=3754,
#        validation_data=data_gen.generate_forever("validation", with_filenames=False),
#        validation_steps=num_steps_valid,
        epochs=100,
        callbacks=callback_list,
        verbose=2)



