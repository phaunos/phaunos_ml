import pytest
import os
import tensorflow as tf
import numpy as np
import soundfile as sf

from phaunos_ml.utils.feature_utils import MelSpecExtractor
from phaunos_ml.utils.annotation_utils import read_annotation_file
from phaunos_ml.utils.audio_utils import audiofile2tfrecord
from phaunos_ml.utils.tf_serialization_utils import serialized2example, serialized2data


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
AUDIOFILE_RELPATH = 'chirp.wav'
ANNFILE_RELPATH = 'chirp.ann'
TFRECORD_FILE = os.path.join(DATA_PATH, 'positive', 'chirp.tf')


class TestTFRecord:

    @pytest.fixture(scope="class")
    def audio_data(self):
        y, sr = sf.read(os.path.join(DATA_PATH, AUDIOFILE_RELPATH))
        y = np.atleast_2d(np.asfortranarray(y).T) # y.shape is [n_channels, n_samples]

        return y, sr
    
    @pytest.fixture(scope="class")
    def annotation_set(self):
        return read_annotation_file(os.path.join(DATA_PATH, ANNFILE_RELPATH))

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
        features, _, _ = feature_extractor.process(audio, sr)

        # write tfrecord file
        audiofile2tfrecord(
            DATA_PATH,
            AUDIOFILE_RELPATH,
            DATA_PATH,
            feature_extractor,
            annfile_relpath=ANNFILE_RELPATH
        )

        # read tfrecord to example
        dataset = tf.data.TFRecordDataset([TFRECORD_FILE])
        dataset = dataset.map(lambda x: serialized2example(x))
        examples = [ex for ex in dataset]

        # check data
        assert np.array_equal(features, np.array([tf.reshape(tf.io.decode_raw(ex['data'], tf.float32), ex['shape']).numpy() for ex in examples]))

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
        dataset = dataset.map(lambda x: serialized2data(x, list(range(10))))
        data_label_list = np.array([d[1].numpy() for d in dataset])

        #check labels
        assert np.all(data_label_list[0] == np.array([0,0,0,0,0,0,0,0,0,0]))
        assert np.all(data_label_list[1:7] == np.array([0,0,0,0,0,0,1,0,0,0]))
        assert np.all(data_label_list[9:19] == np.array([0,0,0,0,0,1,0,0,0,0]))
        assert np.all(data_label_list[19:31] == np.array([0,0,1,1,0,1,0,0,0,0]))
        assert np.all(data_label_list[31:41] == np.array([0,0,1,1,0,0,0,0,0,0]))
        assert np.all(data_label_list[41] == np.array([0,0,0,1,0,0,0,0,0,0]))
        assert np.all(data_label_list[42:] == np.array([0,0,0,0,0,0,0,0,0,0]))

        os.remove(TFRECORD_FILE)
