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


NUM_EXAMPLES = 50
AUDIO_LENGTH = 50000
SR = 22050
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
        return (np.random.rand(NUM_EXAMPLES, AUDIO_LENGTH) * 2 - 1).astype(np.float32)

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
        dataset = dataset.map(lambda data: melspec.process(data))

        count = 0
        for batch in dataset:
            for d in batch:
                librosa_mel_spec = librosa.feature.melspectrogram(                
                    y=data[count],
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

                for t in range(d.shape[1]):
                    assert (np.all(np.abs(d[:,t].numpy() - librosa_mel_spec[:,t]) < np.abs(librosa_mel_spec[:,t] * MAX_ERROR)))
                count += 1
        
