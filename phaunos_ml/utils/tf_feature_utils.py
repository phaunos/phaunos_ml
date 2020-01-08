import tensorflow as tf

from .feature_utils import LOG_OFFSET


class MelSpectrogram:

    def __init__(self,
                 sr,
                 n_fft,
                 hop_length,
                 n_mels,
                 fmin=0,
                 fmax=0,
                 log=False):

        self.n_fft = tf.constant(n_fft)
        self.hop_length = tf.constant(hop_length)
        self.log = tf.constant(log)

        # Mel filter weights
        self.mel_filters = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=n_mels,
            num_spectrogram_bins=int(n_fft / 2 + 1),
            sample_rate=sr,
            lower_edge_hertz=fmin,
            upper_edge_hertz=fmax if fmax > 0 else int(sr / 2)
        )

    def process(self, data):
        """
        Compute mel-spectrogram from tf.Tensor.
        See tests/test_feature_utils.py for an example.
        Args:
            data: Tensor with shape (batch_size, n_channels, audio_length).
        """

        # compute spectogram
        spec = tf.abs(tf.signal.stft(
            data,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft))

        # apply mel filtering (+ log)
        melspec = tf.cond(
            self.log,
            lambda:tf.math.log(tf.tensordot(spec, self.mel_filters, 1) + LOG_OFFSET),
            lambda:tf.tensordot(spec, self.mel_filters, 1)
        )

        # transpose to get the frequency along the height and time along the width,
        # because height is first in Tensorflow's convention (input formats are NHWC or NCHW)
        return tf.transpose(melspec, [0, 1, 3, 2])
