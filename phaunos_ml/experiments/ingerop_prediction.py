import os
import json

import tensorflow as tf

from nsb_aad.frame_based_detectors.mario_detector import MarioDetector
from phaunos_ml.utils import feature_utils, audio_utils, tf_utils, tf_feature_utils


SR = 22050

MIN_ACTIVITY_DUR = 0.05

TMP_DIR = '/home/jul/tmp_tfrecords'

# Mel spectrogram config
N_FFT = 512
HOP_LENGTH = 128
FMIN = 500
FMAX = 8000
N_MELS = 64
N_TIME_BINS = 169 # inspect the data to find out

def predict(audio_filename, actdet_cfg_file, featex_cfg_file):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # TODO write am audiofile2example function so
    # that we do not have to write/read a TFRecord

    #################################################
    # Run activity detection and feature extraction #
    #################################################
    

    # create activity_detector from config
    actdet_cfg = json.load(open(actdet_cfg_file, 'r'))
    activity_detector = MarioDetector(actdet_cfg)

    # create feature extractor from config
    feature_extractor = feature_utils.AudioSegmentExtractor.from_config(featex_cfg_file)

    # write tfrecord
    audio_utils.audiofile2tfrecord(
        os.path.dirname(audio_filename),
        os.path.basename(audio_filename),
        TMP_DIR,
        feature_extractor,
        annotation_filename=None,
        activity_detector=activity_detector,
        min_activity_dur=actdet_cfg['min_activity_dur']
    )

    tfrecord_file = os.path.join(
        TMP_DIR,
        'positive',
        os.path.basename(audio_filename).replace('.wav', '.tf')
    )

    ##########################
    # Create tf.data.Dataset # 
    ##########################

    # Because phaunos_ml was primarily written for training,
    # we need to provide some class_list and make batches of 1
    # to run the prediction

    class_list = [0,1,2,3] # random class list which is a mandatory argument
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(lambda x: tf_utils.serialized2data(
        x,
        feature_extractor.feature_shape,
        class_list,
        training=True))
    dataset = dataset.batch(1)

    print(dataset)

    melspec_ex = tf_feature_utils.MelSpectrogram(
        feature_extractor.sr, N_FFT, HOP_LENGTH, N_MELS, fmin=FMIN, fmax=FMAX, log=True)
    dataset = dataset.map(lambda data, labels: (
        tf.expand_dims(melspec_ex.process(tf.squeeze(data, axis=[1,2])), 1),
        labels))

    print(dataset)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    somedata = []
    try:
        while True:
            somedata.append(session.run(next_batch))
    except tf.errors.OutOfRangeError:
        pass


    return somedata


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

#    parser = argparse.ArgumentParser()
#    parser.add_argument("config_filename", type=str)
#    args = parser.parse_args()

    print("Not implemented")
    #run()

