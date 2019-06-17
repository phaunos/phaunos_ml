import pytest
import os
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import librosa

from phaunos_ml.utils.feature_utils import MelSpecExtractor
from phaunos_ml.utils.annotation_utils import read_annotation_file
from phaunos_ml.utils.audio_utils import audiofile2tfrecord
from phaunos_ml.utils.tf_utils import serialized2example, serialized2data


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
WAV_FILE = 'chirp.wav'
ANNOTATION_FILE = os.path.join(DATA_PATH, 'chirp.ann')
TFRECORD_FILE = os.path.join(DATA_PATH, 'chirp.tf')


class TestTFRecord:

    @pytest.fixture(scope="class")
    def audio_data(self):
        return librosa.load(os.path.join(DATA_PATH, WAV_FILE), sr=None)
    
    @pytest.fixture(scope="class")
    def annotation_set(self):
        return read_annotation_file(ANNOTATION_FILE)

    def test_data(self, audio_data, annotation_set):
        """Arbitrary sanity checks"""

        # compute features
        audio, sr = audio_data
        feature_extractor = MelSpecExtractor(
            n_fft=512,
            hop_length=128,
            example_duration=0.4,
            example_hop_duration=0.1
        )
        features = feature_extractor.process(audio, sr)

        # write tfrecord file
        audiofile2tfrecord(
            DATA_PATH,
            WAV_FILE,
            DATA_PATH,
            feature_extractor,
            annotation_filename=ANNOTATION_FILE
        )

        # read tfrecord to example
        dataset = tf.data.TFRecordDataset([TFRECORD_FILE])
        dataset = dataset.map(lambda x: serialized2example(x, feature_extractor.example_shape))
        it = dataset.make_one_shot_iterator()
        examples = [ex for ex in it]

        # check data
        assert np.array_equal(features, np.array([ex['data'].numpy() for ex in examples]))

        # check labels
        labels_str = np.array([ex['labels'].numpy() for ex in examples])
        assert labels_str[0] == b''
        assert np.all(labels_str[1:7]==b'6')
        assert np.all(labels_str[9:19]==b'5')
        assert np.all(labels_str[19:31]==b'2#3#5')
        assert np.all(labels_str[31:41]==b'2#3')
        assert labels_str[41] == b'3'
        assert np.all(labels_str[42:]==b'')

        # read tfrecord to data and label (model input)
        dataset = tf.data.TFRecordDataset([TFRECORD_FILE])
        dataset = dataset.map(lambda x: serialized2data(x, feature_extractor.example_shape, list(range(10))))
        it = dataset.make_one_shot_iterator()
        data_label_list = np.array([dl[1].numpy() for dl in it])

        #check labels
        assert np.all(data_label_list[0] == np.array([0,0,0,0,0,0,0,0,0,0]))
        assert np.all(data_label_list[1:7] == np.array([0,0,0,0,0,0,1,0,0,0]))
        assert np.all(data_label_list[9:19] == np.array([0,0,0,0,0,1,0,0,0,0]))
        assert np.all(data_label_list[19:31] == np.array([0,0,1,1,0,1,0,0,0,0]))
        assert np.all(data_label_list[31:41] == np.array([0,0,1,1,0,0,0,0,0,0]))
        assert np.all(data_label_list[41] == np.array([0,0,0,1,0,0,0,0,0,0]))
        assert np.all(data_label_list[42:] == np.array([0,0,0,0,0,0,0,0,0,0]))

        os.remove(TFRECORD_FILE)
