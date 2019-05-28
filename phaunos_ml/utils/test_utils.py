import os
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow


from .audio_utils import FeatureExtractor, plot_mel_spectrogram
from .tf_utils import serialized2example


tf.enable_eager_execution() 


def generate_images_from_audio(feature_extractor, filename, out_dir):

    y, sr = librosa.load(audio_filename, sr=None)

    features = feature_extractor.process(y, sr)

    for i in range(features.shape[0]):
        start_time = feature_extractor.actual_example_hop_duration * i
        specshow(features[i],
                 x_axis='time', y_axis='mel',
                 sr=22050,
                 hop_length=feature_extractor.hop_length,
                 fmin=feature_extractor.fmin, fmax=feature_extractor.fmax,
                 cmap='gray_r')
        plt.savefig(os.path.join(out_dir, filename.split('/')[-1].replace('.wav', '')) +
                    '_{:.3f}_{:.3f}'.format(
                        start_time,
                        start_time + feature_extractor.actual_example_duration) + '.png'
                    )


def generate_images_from_tfrecord(feature_extractor, filename, out_dir):

    dataset = tf.data.TFRecordDataset(filename).shuffle(10000)

    parsed_dataset = dataset.map(lambda x: serialized2example(x, [83,128]))

    it = parsed_dataset.make_one_shot_iterator()

    while True:
        try:
            example = it.get_next()
            data = example['data'].numpy()
            times = example['times'].numpy()
            specshow(data,
                     x_axis='time', y_axis='mel',
                     sr=22050,
                     hop_length=feature_extractor.hop_length,
                     fmin=feature_extractor.fmin, fmax=feature_extractor.fmax,
                     cmap='gray_r')
            plt.savefig(os.path.join(out_dir, filename.split('/')[-1].replace('.tf', '')) +
                        '_{:.3f}_{:.3f}'.format(times[0], times[1]) + '.png')
        except tf.errors.OutOfRangeError:
            break




