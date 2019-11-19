import os
import json
import numpy as np
import librosa

import tensorflow as tf

from nsb_aad.frame_based_detectors.mario_detector import MarioDetector
from phaunos_ml.utils import feature_utils, audio_utils, tf_utils, tf_feature_utils
from phaunos_ml.utils.audio_utils import audio2data

from phaunos_ml.models import simple_cnn2 as simple_cnn

SR = 22050

TMP_DIR = '/home/jul/tmp_tfrecords'

# Mel spectrogram config
N_FFT = 512
HOP_LENGTH = 128
FMIN = 500
FMAX = 10000
N_MELS = 64
N_TIME_BINS = 341 # inspect the data to find out

N_CLASSES = 3



class Prediction:

    def __init__(self, actdet_cfg_file, featex_cfg_file, model_weights_file):

        # create activity_detector from config
        actdet_cfg = json.load(open(actdet_cfg_file, 'r'))
        self.activity_detector = MarioDetector(actdet_cfg)
        self.min_activity_dur = actdet_cfg['min_activity_dur']

        # create feature extractor from config
        self.feature_extractor = feature_utils.AudioSegmentExtractor.from_config(featex_cfg_file)

        # build model and set weights
        inputs = tf.keras.Input(shape=(1, N_MELS, N_TIME_BINS), batch_size=1, name='mels')
        outputs = simple_cnn.build_model(inputs, N_CLASSES)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.load_weights(model_weights_file)
        self.model._make_predict_function()

        # create TF mel extractor
        self.melspec_ex = tf_feature_utils.MelSpectrogram(
            self.feature_extractor.sr, N_FFT, HOP_LENGTH, N_MELS, fmin=FMIN, fmax=FMAX, log=True)


    def process(self, audio, sr):

        # Because phaunos_ml was primarily written for training,
        # we need to provide some class_list and make batches of 1
        # to run the prediction

        data, _ = audio2data(
            audio,
            sr,
            self.feature_extractor,
            self.activity_detector,
            class_list=np.arange(N_CLASSES),
            mask_min_dur=self.min_activity_dur)

        features = np.asarray([d[0] for d in data])
        labels = np.asarray([d[1] for d in data])

        features = tf.reshape(features, (features.shape[0], 1, features.shape[1], features.shape[2]))

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(1)

        dataset = dataset.map(lambda data, labels: (
            tf.expand_dims(self.melspec_ex.process(tf.squeeze(data, axis=[1,2])), 1)))

        predictions = self.model.predict(dataset)

        # integrate predictions over time
        return np.mean(predictions, axis=0)


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

#    parser = argparse.ArgumentParser()
#    parser.add_argument("config_filename", type=str)
#    args = parser.parse_args()

    print("Not implemented")
    #run()

