import numpy as np
from enum import Enum
import librosa
from librosa.display import specshow
import json
import matplotlib.pyplot as plt


"""
Utils for extracting features and making fixed-sized examples.
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
            example_duration=2,
            example_hop_duration=1.5,
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
    def feature_size(self):
        return int(self.example_duration * self.feature_rate)
    
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
            json.dump(d, f)

    def process(self, audio, sr, mask=None, mask_sr=None, mask_min_dur=None):
        """
        If mask, mask_sr (the sampling rate of the mask) and mask_min_dur
        (minimum total duration, in seconds, of positive mask values in a segment) are set,
        two arrays are returned: one with segments containing True (or 1s) mask value
        and one with the remaining segments.
        """

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')
        if len(audio.shape) != 1:
            raise ValueError('Only mono audio files are allowed')
        if not ((mask is None) == (mask_sr is None) == (mask_min_dur is None)):
            raise ValueError("mask, mask_sr and mask_min_dur parameters must be all set or all not set.")

        # Compute mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
            n_fft=self.n_fft, hop_length=self.hop_length
        ).astype(self.dtype)
        
        # Create overlapping examples. Pad last example to cover the whole signal.
        n_frames = mel_sp.shape[1]
        num_examples = int(np.ceil(max(0, (n_frames - self.feature_size)) / self.example_hop_size) + 1)
        pad_size = (num_examples - 1) * self.example_hop_size + self.feature_size - n_frames
        mel_sp = np.pad(
            mel_sp,
            ((0, 0), (0, pad_size)),
            mode='constant',
            constant_values=0
        )
        if self.log:
            mel_sp = np.log(mel_sp + LOG_OFFSET)

        shape = (num_examples, mel_sp.shape[0], self.feature_size)
        strides = (mel_sp.strides[1] * self.example_hop_size,) + mel_sp.strides

        segments = np.lib.stride_tricks.as_strided(mel_sp, shape=shape, strides=strides)
        mask_segments = np.ones(num_examples, dtype=np.bool)

        times = []
        start = 0
        for i in range(num_examples):
            end = start + self.feature_size - 1
            times.append((start/self.feature_rate, end/self.feature_rate))
            if not (mask is None):
                start_mask = int(start / self.feature_rate * mask_sr)
                end_mask = int(min(len(mask) - 1, end / self.feature_rate * mask_sr))
                # count positive mask values in the segment
                n_pos = np.count_nonzero(mask[start_mask:end_mask]) if start_mask < len(mask) else 0
                # if the total duration of the positive mask frames is above the threshold, set segment mask to True
                mask_segments[i] = True if n_pos / mask_sr > mask_min_dur else False
            start += self.example_hop_size
        
        return segments, mask_segments, times


    def plot(self, mel_sp):
        return specshow(
            mel_sp,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            cmap='gray_r',
            x_axis='time',
            y_axis='mel'
        )


    def __repr__(self):
        t = type(self)        
        return '{}.{}. Config: {}'.format(t.__module__, t.__qualname__, self.__dict__)


class AudioSegmentExtractor:
    """
    Raw audio segment extractor.
    """

    def __init__(
            self, 
            sr=22050,
            example_duration=2,
            example_hop_duration=1.5,
            dtype=np.float32):

        self.sr = sr
        self.example_duration = example_duration
        self.example_hop_duration = example_hop_duration
        self.dtype = dtype

    @classmethod
    def from_config(cls, config_file):
        config = json.load(open(config_file, 'r'))
        obj = cls(**config)
        obj.dtype = NP_DTYPE[config['dtype']].value
        return obj

    @property
    def feature_size(self):
        return int(self.example_duration * self.sr)
    
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
            json.dump(d, f)

    def process(self, audio, sr, mask=None, mask_sr=None, mask_min_dur=None):
        """
        If mask, mask_sr (the sampling rate of the mask) and mask_min_dur
        (minimum total duration, in seconds, of positive mask values in a segment) are set,
        two arrays are returned: one with segments containing True (or 1s) mask value
        and one with the remaining segments.
        """

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')
        if len(audio.shape) != 1:
            raise ValueError('Only mono audio files are allowed')
        if not ((mask is None) == (mask_sr is None) == (mask_min_dur is None)):
            raise ValueError("mask, mask_sr and mask_min_dur parameters must be all set or all not set.")
        

        # Create overlapping segments. Pad last example to cover the whole signal.

        num_segments = int(np.ceil(max(0, (audio.size - self.feature_size)) / self.example_hop_size) + 1)
        pad_size = (num_segments - 1) * self.example_hop_size + self.feature_size - audio.size
        audio = np.pad(
            audio,
            (0, pad_size),
            mode='constant',
            constant_values=0
        )

        # reshape audio to (1, len(audio))
        audio = np.expand_dims(audio, 0)

        shape = (num_segments, audio.shape[0], self.feature_size)
        strides = (audio.strides[1] * self.example_hop_size,) + audio.strides

        segments =  np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

        mask_segments = np.ones(num_segments, dtype=np.bool)
        times = []
        start = 0
        for i in range(num_segments):
            end = start + self.feature_size - 1
            times.append((start/self.sr, end/self.sr))
            if not (mask is None):
                start_mask = int(start / self.sr * mask_sr)
                end_mask = int(min(len(mask) - 1, end / self.sr * mask_sr))
                # count positive mask values in the segment
                n_pos = np.count_nonzero(mask[start_mask:end_mask]) if start_mask < len(mask) else 0
                # if the total duration of the positive mask frames is above the threshold, set segment mask to True
                mask_segments[i] = True if n_pos / mask_sr > mask_min_dur else False
            start += self.example_hop_size

        return segments, mask_segments, times
        
    def plot(self, data):
        return plt.plot(np.arange(len(data))/self.sr, data)

    def __repr__(self):
        t = type(self)        
        return '{}.{}. Config: {}'.format(t.__module__, t.__qualname__, self.__dict__)
