import os

from phaunos_ml.utils.annotation_utils import Annotation, write_annotation_file, ANN_EXT


"""
Utils for the Freesound challenge 2019.
https://www.kaggle.com/c/freesound-audio-tagging-2019/
"""


def get_class_list(sample_submission_filename):
    first_line = open(sample_submission_filename, 'r').readline()
    labels = first_line.strip().split(',')[1:]
    assert len(labels) == 80, 'Wrong number of classes (must be 80)'
    return sorted(labels)
 

def generate_ann_files(train_path, class_list):
    
    datasets = ['curated', 'noisy']

    class_list = sorted(class_list)

    for dataset in datasets:

        label_filename = os.path.join(train_path, f'train_{dataset}', f'train_{dataset}.csv')
        lines = open(label_filename, 'r').readlines()[1:]
        for line in lines:
            cols = line.strip().replace('"', '').split(',')
            audio_filename = cols[0]
            label_set = set([class_list.index(s) for s in cols[1:]])

            ann_filename = os.path.join(
                train_path,
                f'train_{dataset}',
                'annotations',
                audio_filename.replace('.wav', ANN_EXT)
            )

            write_annotation_file([Annotation(label_set=label_set)], ann_filename)
