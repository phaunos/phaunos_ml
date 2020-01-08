"""
Test Tensorflow mel-spectrogram from raw audio in tf.data.Dataset
by comparing to Librosa mel spectrogram.
The results should be the same within a MAX_ERROR margin.
"""

import pytest
import numpy as np
import librosa
import tensorflow as tf

from phaunos_ml.utils import tf_feature_utils


N_EXAMPLES = 50
AUDIO_LENGTH = 50000
SR = 22050
N_CHANNELS = 3
BATCH_SIZE = 8

N_FFT = 512
HOP_LENGTH = 128
FMIN = 50
FMAX = 8000
N_MELS = 64

MAX_ERROR = 0.02


class TestMelSpectrogram:

    @pytest.fixture(scope="class")
    def data(self):
        # generate data in format NCHW
        return (np.random.rand(N_EXAMPLES, N_CHANNELS, 1, AUDIO_LENGTH) * 2 - 1).astype(np.float32)

    @pytest.fixture(scope="class")
    def melspec(self):
        return tf_feature_utils.MelSpectrogram(
            SR,
            N_FFT,
            HOP_LENGTH,
            N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            log=False
        )


    def test_melspectrogram(self, data, melspec):

        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(lambda data: melspec.process(tf.squeeze(data,2)))

        count = 0
        for batch in dataset:
            for d in batch:
                for c in range(N_CHANNELS):
                    librosa_mel_spec = librosa.feature.melspectrogram(                
                        y=np.squeeze(data,2)[count,c],
                        sr=SR,
                        n_fft=N_FFT,
                        hop_length=HOP_LENGTH,
                        win_length=N_FFT,
                        n_mels=N_MELS,
                        fmin=FMIN,
                        fmax=FMAX,
                        center=False,
                        power=1.0,
                        norm=None,
                        htk=True

                    )

                    assert (np.all(np.abs(d[c].numpy() - librosa_mel_spec) < np.abs(librosa_mel_spec * MAX_ERROR)))
                count += 1
            
