import numpy as np
from enum import Enum
import librosa
from librosa.display import specshow
import json
import matplotlib.pyplot as plt


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
        """Compute mel spectrogram.

        Args:
            audio: [n_channels, n_samples]
            sr: sample rate
            mask: boolean mask
            mask_sr: mask sample rate
            mask_min_dur: minimum total duration, in seconds, of
                positive mask values in a segment

        Returns a list of feature arrays representing the fixed-sized examples
        (in format NCHW), a boolean mask and the times boundaries of the examples.
        """

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')
        if not ((mask is None) == (mask_sr is None) == (mask_min_dur is None)):
            raise ValueError("mask, mask_sr and mask_min_dur parameters must be all set or all not set.")

        n_channels = audio.shape[0]

        # Compute mel spectrogram
        mel_sp = np.array([librosa.feature.melspectrogram(
            y=audio[c], sr=sr,
            n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
            n_fft=self.n_fft, hop_length=self.hop_length
        ).astype(self.dtype) for c in range(n_channels)])
        
        # Create overlapping examples
        segments = seq2frames(mel_sp, self.feature_size, self.example_hop_size)
        if self.log:
            segments = np.log(segments + LOG_OFFSET)

        # Build mask and times arrays
        n_segments = segments.shape[0]
        mask_segments = np.ones(n_segments, dtype=np.bool)
        times = []
        start = 0
        for i in range(n_segments):
            end = start + self.feature_size - 1
            times.append((start/self.feature_rate, (end+1)/self.feature_rate))
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


class CorrelogramExtractor:
    """Extractor of correlogram series, preminarily used for vehicle detection.
    """

    def __init__(
            self, 
            max_delay,
            sr=48000,
            n_fft=1024,
            hop_length=1024,
            example_duration=5*8192./48000,
            example_hop_duration=5*8192./48000,
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
            example_duration (float):       example duration
            example_hop_duration (float):   example hop duration
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

        # Indices of the correlogram corresponding to
        # -max_delay and +max_delay
        self.ind_min = int(n_fft / 2 - max_delay * sr)
        self.ind_max = int(n_fft / 2 + max_delay * sr)

        if self.ind_min < 0 or self.ind_max >= n_fft:
            raise ValueError(f'n_fft duration ({n_fft/sr:.3f}) must' +
                             ' be larger than 2 * max_delay ({2*max_delay})')


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

    @property
    def feature_rate(self):
        return self.sr / self.hop_length

    @property
    def feature_size(self):
        return int(self.example_duration * self.feature_rate)
    
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

    def process(self, audio, sr, mask=None, mask_sr=None, mask_min_dur=None):
        """Computes series of Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

        Args:
            audio: [2, n_samples].
            sr: sample rate
            mask: boolean mask
            mask_sr: mask sample rate
            mask_min_dur: minimum total duration, in seconds, of
                positive mask values in a segment

        Returns a list of feature arrays representing the fixed-sized examples
        (in format NCHW), a boolean mask and the times boundaries of the examples.
        """

        n_channels = audio.shape[0]

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')
        if n_channels != 2:
            raise ValueError(f'Audio must have two channels (found {n_channels})')
        if not ((mask is None) == (mask_sr is None) == (mask_min_dur is None)):
            raise ValueError("mask, mask_sr and mask_min_dur parameters must be all set or all not set.")
        
        # Create overlapping frames
        frames = seq2frames(
            np.reshape(audio, (n_channels, 1, audio.shape[1])),
            self.n_fft,
            self.hop_length)

        # Compute GCC-PHAT and get correlograms corresponding to [-max_delay,max_delay[
        frames = self.gccphat(frames[:,0,0], frames[:,1,0], norm=self.gcc_norm)[:,self.ind_min:self.ind_max]

        # Reshape to match seq2frames input format
        frames = frames.swapaxes(0, 1)[np.newaxis,:]

        # Create overlapping examples
        segments = seq2frames(frames, self.feature_size, self.example_hop_size)

        # Build mask and times arrays
        n_segments = segments.shape[0]
        mask_segments = np.ones(n_segments, dtype=np.bool)
        times = []
        start = 0
        for i in range(n_segments):
            end = start + self.feature_size - 1
            times.append((start/self.feature_rate, (end+1)/self.feature_rate))
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
        """Compute fixed-sized audio chunks.

        Args:
            audio: [n_channels, n_samples]
            sr: sample rate
            mask: boolean mask
            mask_sr: mask sample rate
            mask_min_dur: minimum total duration, in seconds, of
                positive mask values in a segment

        Returns a list of feature arrays representing the fixed-sized examples
        (in format NCHW), a boolean mask and the times boundaries of the examples.
        """

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')
        if not ((mask is None) == (mask_sr is None) == (mask_min_dur is None)):
            raise ValueError("mask, mask_sr and mask_min_dur parameters must be all set or all not set.")
        
        audio = np.expand_dims(audio, 1) # to CHW

        # Create overlapping segments
        segments = seq2frames(audio, self.feature_size, self.example_hop_size)

        # Build mask and times arrays
        n_segments = segments.shape[0]
        mask_segments = np.ones(n_segments, dtype=np.bool)
        times = []
        start = 0
        for i in range(n_segments):
            end = start + self.feature_size - 1
            times.append((start/self.sr, (end+1)/self.sr))
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


def seq2frames(data, frame_len, frame_hop_len):
    """Reorganize sequence data into frames.

    Args:
        data: sequence of data with shape (C, H, T), where
            C in the number of channels, H the size of the first dimension of the feature
            (e.g. H=1 for audio and H=num_mel_bands for mel spectrograms) and T the number
            of time bins in the sequence.
        frame_len: length of each frame
        frame_hop_len: hop length between frames

    Returns:
        Data frames with shape (n_frames, C, H, frame_len).
        Last example is 0-padded to cover the whole sequence
    """
    
    C, H, T = data.shape
    
    # Pad last example to cover the whole sequence
    n_frames = int(np.ceil(max(0, (T - frame_len)) / frame_hop_len) + 1)
    pad_size = (n_frames - 1) * frame_hop_len + frame_len - T
    data = np.pad(
        data,
        ((0, 0),(0,0),(0,pad_size)),
        mode='constant',
        constant_values=0
    )

    shape = (n_frames, C, H, frame_len)
    strides = (frame_hop_len*data.strides[-1],) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


