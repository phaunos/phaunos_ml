import os
from collections import defaultdict
import tensorflow as tf 
tf.enable_eager_execution() 
import numpy as np


import librosa
from phaunos_ml.utils.tf_utils import serialized2example



def sanity_check(root_path, feature_path, dataset_file, feature_extractor, label_files, class_list):

    class_list = sorted(class_list)

    f2l = dict()
    for label_file in label_files:
        for line in open(label_file, 'r'):
            if line.startswith('fname'):
                continue
            i = line.find(',')
            filename = line[:i]
            label_str = line[i+1:].strip().replace('"', '').split(',')
            f2l[filename] = set([class_list.index(l) for l in label_str])


    filelist = [line.split(',')[0] for line in open(dataset_file, 'r') if not line.startswith('#')]
    dataset = tf.data.TFRecordDataset(
        [os.path.join(feature_path, f.replace('.wav', '.tf')) for f in filelist]
    )
    dataset = dataset.map(lambda x: serialized2example(x, feature_extractor.feature_shape))
    it = dataset.make_one_shot_iterator()

    file_dict = defaultdict(list)
    for example in it:
        filename = example['filename'].numpy().decode()
        labels = set([int(l) for l in example['labels'].numpy().decode().split('#')])
        times = example['times'].numpy()
        file_dict[filename].append(times)

        if not f2l[filename.replace('.tf', '.wav')] == labels:
            print(f'Label issue in {filename}')

    for filename, times in file_dict.items(): 
        try:
            audio, sr = librosa.load(
                os.path.join(
                    root_path,
                    'train_curated/audio_22050hz',
                    filename.replace('.tf', '.wav')
                ), sr=None
            )
        except:
            audio, sr = librosa.load(
                os.path.join(
                    root_path,
                    'train_noisy/audio_22050hz',
                    filename.replace('.tf', '.wav')
                ), sr=None
            )

        duration = len(audio) / sr
        actual_n_examples = len(times)

        wanted_n_examples = int(
            np.ceil(max(0, (duration - feature_extractor.actual_example_duration)) / feature_extractor.actual_example_hop_duration) + 1
        )
        if not wanted_n_examples == actual_n_examples:
            print(f'Num examples issue in {filename}:')
            print(f'{actual_n_examples} / {wanted_n_examples}')
            print(f'duration: {duration}')


        ok = True
        start, end = times[-1]
        if abs(end - start - feature_extractor.actual_example_duration) > 1.5:
            print("***********************************")
            print(f'Times issue in {filename}:')
            print(f'duration: {end - start}')




    print(f'Num files: {len(file_dict)}')








