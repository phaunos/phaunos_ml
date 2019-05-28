import numpy as np
import librosa
from librosa.display import specshow
import json


LOG_OFFSET = 1e-8
MIN_LAST_CHUNK_DURATION = 0.2


class MelSpecExtractor:

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

    @property
    def example_size(self):
        return int((self.example_duration * self.sr - self.n_fft) / self.hop_length + 1)
    
    @property
    def example_shape(self):
        return [self.n_mels, self.example_size]
    
    @property
    def example_hop_size(self):
        return int(self.example_hop_duration * self.sr / self.hop_length)

    @property
    def actual_example_duration(self):
        return ((self.example_size - 1) * self.hop_length + self.n_fft) / self.sr

    @property
    def actual_example_hop_duration(self):
        return self.example_hop_size * self.hop_length / self.sr
    
    def config2file(self, filename):
        with open(filename, 'w') as f:
            d = self.__dict__.copy()
            d['dtype'] = str(d['dtype'])
            json.dump(d, f)

    def process(self, audio, sr):

        if sr != self.sr:
            raise ValueError(f'Sample rate must be {self.sr} ({sr} detected)')
        if len(audio.shape) != 1:
            raise ValueError('Only mono audio files are allowed')

        # Compute mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
            n_fft=self.n_fft, hop_length=self.hop_length
        ).astype(self.dtype)
        
        # Create overlapping examples. Pad last example to cover the whole signal.
        n_frames = mel_sp.shape[1]
        num_examples = int(np.ceil(max(0, (n_frames - self.example_size)) / self.example_hop_size) + 1)
        pad_size = (num_examples - 1) * self.example_hop_size + self.example_size - n_frames
        mel_sp = np.pad(
            mel_sp,
            ((0, 0), (0, pad_size)),
            mode='constant',
            constant_values=0
        )
        if self.log:
            mel_sp = np.log(mel_sp + LOG_OFFSET)

        shape = (num_examples, mel_sp.shape[0], self.example_size)
        strides = (mel_sp.strides[1] * self.example_hop_size,) + mel_sp.strides

        return np.lib.stride_tricks.as_strided(mel_sp, shape=shape, strides=strides)


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
