import os
import argparse
import json
import pathlib
from time import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from scipy import interpolate

from phaunos_ml.utils.feature_utils import MelSpecExtractor
from phaunos_ml.utils.dataset_utils import read_dataset_file
from phaunos_ml.utils.tf_utils import tfrecords2tfdataset
from phaunos_ml.models import simple_cnn


def process(config_filename):

    """Training

    Args:
        config_filename: json config file. Must contain the following key:
            "feature_path" : path to the TFRecord files (should contain a feature
                extractor config file named "featex_config.json"
            "train_set_file" : training dataset file, as defined in phaunos_ml.utils.dataset_utils
            "n_train_batches" : number of training batches (can be counted by phaunos_ml.utils.dataset_utils.dataset_stat_per_example)
            "multilabel" : 'true' or 'false'. Whether an example can have multiple labels.
            "batch_size"
            "epochs" : number of training epochs
            "out_dir" : output directory to write log files, models...
            "valid_set_file" (optional) : validation dataset file, as defined in phaunos_ml.utils.dataset_utils
            "n_valid_batches" (optional) : number of validation batches (can be counted by phaunos_ml.utils.dataset_utils.dataset_stat_per_example)
            
    """

    with open(config_filename, "r") as config_file:
        config = json.load(config_file)

    ########################################
    # create feature extractor from config #
    ########################################

    try:
        featex_config = os.path.join(config['feature_path'], 'featex_config.json')
        feature_extractor = MelSpecExtractor.from_config(featex_config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'File {config} not found. Config files must be named "featex_config.json" and located in <feature_path>') from e

    ##############################################
    # get training and valid (optional) datasets #
    ##############################################

    train_files, labels = read_dataset_file(
        config['train_set_file'],
        prepend_path=config['feature_path'],
        replace_ext='.tf'
    )
    class_list = sorted(list(set.union(*labels)))
    train_dataset = tfrecords2tfdataset(
        train_files,
        feature_extractor.feature_shape,
        class_list,
        batch_size=config['batch_size']
    )

    valid_dataset = None
    if config['valid_set_file']:
        valid_files, _ = read_dataset_file(
            config['valid_set_file'],
            prepend_path=config['feature_path'],
            replace_ext='.tf'
        )
        valid_dataset = tfrecords2tfdataset(
            valid_files,
            feature_extractor.feature_shape,
            class_list,
            batch_size=config['batch_size']
        )

    ###############
    # build model #
    ###############

    h = feature_extractor.feature_shape[0]
    w = feature_extractor.feature_shape[1]

    inputs = keras.Input(shape=(1, h, w), batch_size=config['batch_size'], name='mels')
    outputs = simple_cnn.build_model(inputs, len(class_list), multilabel=config['multilabel'])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    # define optimizer
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    if config['multilabel']:
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.categorical_crossentropy

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[keras.metrics.BinaryAccuracy() \
                           if config['multilabel'] else keras.metrics.CategoricalAccuracy()])


    ########################
    # Define the callbacks #
    ########################

    out_dir = os.path.join(config['out_dir'], 'run_' + str(int(time())))
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
                     '{val_binary_accuracy:.2f}.h5' if config['multilabel'] else \
                     '{val_categorical_accuracy:.2f}.h5'),
        monitor='val_binary_accuracy' if config['multilabel'] \
            else 'val_categorical_accuracy',
        verbose=1,
        save_best_only=True)

#    callback_list = [lr_scheduler, tb, mc]
    callback_list = [tb, mc]


    #########
    # train #
    #########

    model.fit(
        train_dataset,
        steps_per_epoch=config['n_train_batches'],
        validation_data=valid_dataset,
        validation_steps=config.get('n_valid_batches', None),
        epochs=config['epochs'],
        callbacks=callback_list,
        verbose=2)


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", type=str)
    args = parser.parse_args()

    process(args.config_filename)
