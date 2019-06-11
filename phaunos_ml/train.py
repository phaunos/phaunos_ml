import os
import argparse
import pathlib
from time import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from scipy import interpolate
import git

from phaunos_ml.utils.feature_utils import MelSpecExtractor
from phaunos_ml.utils.dataset_utils import read_dataset_file
from phaunos_ml.utils.tf_utils import filelist2dataset
from phaunos_ml.models import simple_cnn


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def process(
        feature_extractor,
        train_dataset,
        n_train_batches,
        out_dir,
        n_classes,
        multilabel=False,
        batch_size=32,
        epochs=10,
        valid_dataset=None,
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

    model.summary()

    # define optimizer
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    if multilabel:
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.categorical_crossentropy

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])


    ########################
    # Define the callbacks #
    ########################


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
        os.path.join(model_dir, "model.{epoch:02d}-{val_acc:.2f}.h5"),
        monitor='val_acc',
        verbose=1,
        save_best_only=True)

#    callback_list = [lr_scheduler, tb, mc]
    callback_list = [tb, mc]


    #########
    # train #
    #########

    model.fit(
        train_dataset,
        steps_per_epoch=n_train_batches,
        validation_data=valid_dataset,
        validation_steps=n_valid_batches,
        epochs=epochs,
        callbacks=callback_list,
        verbose=2)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("feature_path", type=str,
                        help="Path to the feature extractor config " +
                        " and the tfrecord files.")
    parser.add_argument("train_set_file", type=str,
                        help="File containing the list of " +
                        "the training audio files.")
    parser.add_argument("n_train_batches", type=int,
                        help="Number of training batches.")
    parser.add_argument("out_dir", type=str,
                        help="Output dir to write history, models and Tensorboard logs.")
    parser.add_argument("--valid_set_file", type=str,
                        help="File containing the list of " +
                        "the validation audio files.")
    parser.add_argument("--n_valid_batches", type=int,
                        help="Number of valid batches.")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size.", default=32)
    args = parser.parse_args()

    ########################################
    # create feature extractor from config #
    ########################################

    try:
        config = os.path.join(args.feature_path, 'config.json')
        feature_extractor = MelSpecExtractor.from_config(config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'File {config} not found. Config files must be named "config.json" and located in <feature_path>') from e

    ##############################################
    # get training and valid (optional) datasets #
    ##############################################

    train_files, labels = read_dataset_file(
        args.train_set_file,
        prepend_path=args.feature_path,
        replace_ext='.tf'
    )
    class_list = sorted(list(set.union(*labels)))
    train_dataset = filelist2dataset(
        train_files,
        feature_extractor.example_shape,
        class_list,
        batch_size=args.batch_size
    )

    valid_dataset = None
    if args.valid_set_file:
        valid_files, _ = read_dataset_file(
            args.valid_set_file,
            prepend_path=args.feature_path,
            replace_ext='.tf'
        )
        valid_dataset = filelist2dataset(
            valid_files,
            feature_extractor.example_shape,
            class_list,
            batch_size=args.batch_size
        )

    ##################################
    # write commit sha in out_dir #
    ##################################
    
    out_dir = os.path.join(args.out_dir, str(int(time())))
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "git_commit_sha.txt"), "w") as f:
        repo = git.Repo(
            os.path.dirname(os.path.abspath(__file__)),
            search_parent_directories=True
        )
        f.write(repo.head.object.hexsha)

    #########
    # train #
    #########

    process(
        feature_extractor,
        train_dataset,
        args.n_train_batches,
        out_dir,
        len(class_list),
        multilabel=True,
        valid_dataset=valid_dataset,
        n_valid_batches=args.n_valid_batches,
        batch_size=args.batch_size,
    )
