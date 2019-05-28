import pytest
import os
import librosa
import numpy as np

from phaunos_ml.utils.feature_utils import MelSpecExtractor


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
WAV_FILE = os.path.join(DATA_PATH, 'chirp.wav')


class TestMelSpecExtractor:
    
    @pytest.fixture(scope="class")
    def audio_data(self):
        return librosa.load(WAV_FILE, sr=None)

    def test_init(self, audio_data):

        audio, sr = audio_data

        with pytest.raises(ValueError):
            ex = MelSpecExtractor(sr=44100)
            ex.process(audio, sr)

        with pytest.raises(ValueError):
            ex = MelSpecExtractor()
            ex.process(np.tile(audio, (2, 1)), sr)

    def test_stride(self, audio_data):

        ex = MelSpecExtractor(
            n_fft=512,
            hop_length=128,
            example_duration=0.4,
            example_hop_duration=0.1
        )
        audio, sr = audio_data
        features = ex.process(audio, sr)

        assert np.array_equal(features[10,:,ex.example_hop_size], features[11,:,0])


