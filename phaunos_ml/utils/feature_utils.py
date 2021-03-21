import numpy as np
from enum import Enum
import json

from .librosa_lite import melspectrogram


"""
Utils for extracting features and making fixed-sized examples
in NCHW format, where:
    N = number of examples
    C = number of channels
    H = dimension 1 of the feature
    W = dimension 2 of the feature
"""


LOG_OFFSET = 1e-8


class NP_DTYPE(Enum):
    F16 = np.float16
    F32 = np.float32
    F64 = np.float64


class MelSpecExtractor:
    """
    Log mel spectrogram extractor.
    See https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
    for features parameters.

    If example_duration == -1, it is set to the audio signal duration.
    """

    def __init__(
            self, 
            sr=22050,
            n_fft=2048,
            hop_length=512,
            fmin=50,
            fmax=None,
            log=True,
            n_mels=128,
            example_duration=-1,
            example_hop_duration=-1,
            dtype=np.float32):

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax if fmax else sr / 2
        self.log = log
        self.n_mels = n_mels
        self.example_duration = example_duration
        self.example_hop_duration = example_hop_duration
        self.dtype = dtype
        
        if self.example_duration != -1:
            self.feature_size = int(self.example_duration * self.feature_rate) + 1
        else:
            self.feature_size = -1

    @classmethod
    def from_config(cls, config_file):
        config = json.load(open(config_file, 'r'))
        obj = cls(**config)
        obj.dtype = NP_DTYPE[config['dtype']].value
        return obj

    @property
    def feature_rate(self):
        return self.sr / self.hop_length

    @property
    def feature_shape(self):
        return [self.n_mels, self.feature_size]
    
    @property
    def example_hop_size(self):
        return int(self.example_hop_duration * self.feature_rate)

    @property
    def actual_example_duration(self):
        return self.feature_size / self.feature_rate

    @property
    def actual_example_hop_duration(self):
        return self.example_hop_size / self.feature_rate
    
    def config2file(self, filename):
        with open(filename, 'w') as f:
            d = self.__dict__.copy()
            d['dtype'] = NP_DTYPE(self.dtype).name
            del d['feature_size']
            json.dump(d, f)

    def process(self, audio, sr):
        """Compute mel spectrogram.

        Args:
            audio: [n_channels, n_samples]
            sr: sample rate

        Returns a list of feature arrays representing the fixed-sized examples
        (in format NCHW) and the times boundaries of the examples.
        """

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')

        n_channels = audio.shape[0]

        # Compute mel spectrogram
        mel_sp = np.array([melspectrogram(
            y=audio[c],
            sr=sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True
        ).astype(self.dtype) for c in range(n_channels)])
        
        if self.example_duration != -1:
            # Create overlapping examples
            segments = seq2frames(
                mel_sp,
                self.feature_size,
                self.example_hop_size,
                center=False)
        else:
            # Create just one segment
            self.feature_size = mel_sp.shape[-1]
            segments = np.expand_dims(mel_sp, 0)

        if self.log:
            segments = np.log(segments + LOG_OFFSET)

        # Build times arrays
        times = [
            (
                i * self.example_hop_size / self.feature_rate,
                (i * self.example_hop_size + self.feature_size) / self.feature_rate
            ) for i in range(segments.shape[0])
        ]
        
        return segments, times

    def __repr__(self):
        t = type(self)        
        return '{}.{}. Config: {}'.format(t.__module__, t.__qualname__, self.__dict__)


class CorrelogramExtractor:
    """Extractor of correlogram series, preminarily used for vehicle detection.

    If example_duration == -1, it is set to the audio signal duration.
    """

    def __init__(
            self, 
            max_delay,
            sr=48000,
            n_fft=1024,
            hop_length=1024,
            example_duration=-1,
            example_hop_duration=-1,
            gcc_norm=False,
            dtype=np.float32):

        """
        Args:
            (Default values are those of the first version of vehicle detection.)
            max_delay (float):              max delay between the microphones
                                            (i.e. distance between the microphone / 340)
            sr (int):                       sample rate, in Hertz
            n_fft (int):                    analysis window size
            hop_length (int):               analysis window hop size
            example_duration (float):       example duration. -1 to set it to the audio signal duration.
            example_hop_duration (float):   example hop duration. -1 to set it to the audio signal duration.
            gcc_norm (bool):                whether to normalize every gcc independently in the correlogram

        Initializes the extractor.
        """

        self.max_delay = max_delay
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.example_duration = example_duration
        self.example_hop_duration = example_hop_duration
        self.gcc_norm = gcc_norm
        self.dtype = dtype

        if self.ind_min < 0 or self.ind_max >= n_fft:
            raise ValueError(f'n_fft duration ({n_fft/sr:.3f}) must' +
                             ' be larger than 2 * max_delay ({2*max_delay})')
        
        if self.example_duration != -1:
            self.feature_size = int(self.example_duration * self.feature_rate) + 1
        else:
            self.feature_size = -1

    @classmethod
    def from_config(cls, config_file):
        config = json.load(open(config_file, 'r'))
        obj = cls(
            config['max_delay'],
            config['sr'],
            config['n_fft'],
            config['hop_length'],
            config['example_duration'],
            config['example_hop_duration'],
            config['gcc_norm']
        )
        obj.dtype = NP_DTYPE[config['dtype']].value
        return obj

    # Indices of the correlogram corresponding to
    # -max_delay and +max_delay
    @property
    def ind_min(self):
        return int(self.n_fft / 2 - self.max_delay * self.sr)
    @property
    def ind_max(self):
        return int(self.n_fft / 2 + self.max_delay * self.sr)

    @property
    def feature_rate(self):
        return self.sr / self.hop_length

    @property
    def feature_shape(self):
        return [self.ind_max-self.ind_min, self.feature_size]
    
    @property
    def example_hop_size(self):
        return int(self.example_hop_duration * self.feature_rate)

    @property
    def actual_example_duration(self):
        return self.feature_size / self.feature_rate

    @property
    def actual_example_hop_duration(self):
        return self.example_hop_size / self.feature_rate
    
    def config2file(self, filename):
        with open(filename, 'w') as f:
            d = self.__dict__.copy()
            d['dtype'] = NP_DTYPE(self.dtype).name
            del d['feature_size']
            json.dump(d, f)

    def gccphat(self, a1, a2, norm=False, fftshift=True, min_d=1e-6):
        """
        Computes GCC-PHAT.

        Args:
            a1 (np.array):      First array, with shape (n_frames, n_fft)
            a2 (np.array):      Second array, with shape (n_frames, n_fft)
            norm (bool):        Normalize to [-1,1]
            fftshift (bool):    Shift the zero-frequency component to the center of the spectrum.
            offset (float):     Min value of d in the calculus below, to avoid division by 0.
        
        Returns an array of correlograms.
        """

        c = np.fft.rfft(a1) * np.conj(np.fft.rfft(a2))
        d = np.abs(c)
        d[d<min_d] = min_d
        c = np.fft.irfft(c / d)
        if norm:
            d = np.max(np.abs(c), axis=1)
            d[d<min_d] = min_d
            c /= d[:,np.newaxis]
        if fftshift:
            c = np.fft.fftshift(c, axes=1)
        return c

    def process(self, audio, sr):
        """Computes series of Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

        Args:
            audio: [2, n_samples].
            sr: sample rate

        Returns a list of feature arrays representing the fixed-sized examples
        (in format NCHW) and the times boundaries of the examples.
        """

        n_channels = audio.shape[0]

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')
        if n_channels != 2:
            raise ValueError(f'Audio must have two channels (found {n_channels})')
        
        # Create overlapping frames
        frames = seq2frames(
            np.reshape(audio, (n_channels, 1, audio.shape[1])),
            self.n_fft,
            self.hop_length,
            center=True)

        # Compute GCC-PHAT and get correlograms corresponding to [-max_delay,max_delay[
        gccs = self.gccphat(frames[:,0,0], frames[:,1,0], norm=self.gcc_norm)[:,self.ind_min:self.ind_max]

        # Reshape to match seq2frames input format
        gccs = gccs.swapaxes(0, 1)[np.newaxis,:]

        if self.example_duration != -1:
            # Create overlapping examples
            segments = seq2frames(
                gccs,
                self.feature_size,
                self.example_hop_size,
                center=False)
        else:
            # Create just one segment
            self.feature_size = gccs.shape[-1]
            segments = np.expand_dims(gccs, 0)

        # Build times arrays
        times = [
            (
                i * self.example_hop_size / self.feature_rate,
                (i * self.example_hop_size + self.feature_size) / self.feature_rate
            ) for i in range(segments.shape[0])
        ]
        
        return segments, times

    def __repr__(self):
        t = type(self)        
        return '{}.{}. Config: {}'.format(t.__module__, t.__qualname__, self.__dict__)


class AudioSegmentExtractor:
    """
    Raw audio segment extractor.
    
    If example_duration == -1, it is set to the audio signal duration.
    """

    def __init__(
            self, 
            sr=22050,
            example_duration=-1,
            example_hop_duration=-1,
            dtype=np.float32):

        self.sr = sr
        self.example_duration = example_duration
        self.example_hop_duration = example_hop_duration
        self.dtype = dtype
        
        if self.example_duration != -1:
            self.feature_size = int(self.example_duration * self.sr)
        else:
            self.feature_size = -1
        
    @classmethod
    def from_config(cls, config_file):
        config = json.load(open(config_file, 'r'))
        obj = cls(**config)
        obj.dtype = NP_DTYPE[config['dtype']].value
        return obj

    @property
    def feature_shape(self):
        return [1, self.feature_size]
    
    @property
    def example_hop_size(self):
        return int(self.example_hop_duration * self.sr)
    
    @property
    def actual_example_duration(self):
        return self.feature_size / self.sr

    @property
    def actual_example_hop_duration(self):
        return self.example_hop_size / self.sr

    def config2file(self, filename):
        with open(filename, 'w') as f:
            d = self.__dict__.copy()
            d['dtype'] = NP_DTYPE(self.dtype).name
            del d['feature_size']
            json.dump(d, f)

    def process(self, audio, sr):
        """Compute fixed-sized audio chunks.

        Args:
            audio: [n_channels, n_samples]
            sr: sample rate

        Returns a list of feature arrays representing the fixed-sized examples
        (in format NCHW) and the times boundaries of the examples.
        """

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')
        
        audio = np.expand_dims(audio, 1) # to CHW

        if self.example_duration != -1:
            # Create overlapping examples
            segments = seq2frames(
                audio,
                self.feature_size,
                self.example_hop_size,
                center=False)
        else:
            # Create just one segment
            self.feature_size = audio.shape[-1]
            segments = np.expand_dims(audio, 0)

        # Build times arrays
        times = [
            (
                i * self.example_hop_size / self.feature_rate,
                (i * self.example_hop_size + self.feature_size) / self.feature_rate
            ) for i in range(segments.shape[0])
        ]

        return segments, times
        
    def __repr__(self):
        t = type(self)        
        return '{}.{}. Config: {}'.format(t.__module__, t.__qualname__, self.__dict__)


def seq2frames(data, frame_len, frame_hop_len, center=False):
    """Reorganize sequence data into frames.

    Args:
        data: sequence of data with shape (C, H, T), where
            C in the number of channels, H the size of the first dimension of the feature
            (e.g. H=1 for audio and H=num_mel_bands for mel spectrograms) and T the number
            of time bins in the sequence.
        frame_len: length of each frame
        frame_hop_len: hop length between frames
        center (bool): pad the time series so that frames are centered

    Returns:
        Data frames with shape (n_frames, C, H, frame_len).
        Last example is 0-padded to cover the whole sequence
    """
    
    C, H, T = data.shape

    n_frames = T // frame_hop_len + 1

    if center:
        pad_before = frame_len // 2
        pad_after = max(0, (n_frames - 1) * frame_hop_len + frame_len // 2 - T)
    else:
        pad_before = 0
        pad_after = max(0, (n_frames - 1) * frame_hop_len + frame_len - T)

    data = np.pad(
        data,
        ((0,0),(0,0),(pad_before,pad_after)),
        mode='constant',
        constant_values=0
    )

    shape = (n_frames, C, H, frame_len)
    strides = (frame_hop_len*data.strides[-1],) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
