import os
from collections import defaultdict
import random
import time
from tqdm import tqdm

from .audio_utils import audiofile2tfrecord
from .annotation_utils import read_annotation_file


def audiolist2tfrecords(
        audio_path,
        filelist,
        out_dir,
        feature_extractor,
        annotation_path=None
):
    for line in tqdm(open(filelist, 'r').readlines()):
        if line.startswith('#'):
            continue
        audio_filename = line.strip()
        annotation_filename = os.path.join(
            annotation_path,
            audio_filename.replace('.wav', '.ann')
        ) if annotation_path else None
        audiofile2tfrecord(
            audio_path,
            audio_filename,
            out_dir,
            feature_extractor,
            annotation_filename=annotation_filename
        )


def create_filename_list(audio_path, out_path, annotation_path=None, label_set=None):
    """label_ind: take only file with at least one label from label_ind"""
    out_filename = os.path.join(
        out_path,
        'subset_{}.csv'.format(str(int(time.time())))
    )
    with open(out_filename, 'w') as out_file:
        if annotation_path and label_set:
            out_file.write('#class subset: {}\n'.format(','.join([str(i) for i in sorted(list(label_set))])))
        for file_path, _, filenames in os.walk(audio_path):
            for filename in filenames:
                rel_path = os.path.relpath(file_path, audio_path)
                audio_filename = os.path.join(rel_path, filename)
                add_file = True
                if annotation_path and label_set:
                    annotation_set = read_annotation_file(
                        os.path.join(annotation_path, audio_filename.replace('.wav', '.ann'))
                    )
                    file_label_set = set()
                    for ann in annotation_set:
                        file_label_set.update(ann.label_set)

                    if not file_label_set.intersection(label_set):
                        add_file = False
                if add_file:
                    out_file.write(f'{audio_filename}\n')


#def split_dataset(dataset, valid=0.2, test=0):
#    """Stratified split of dataset file
#        <filename>, <label_id>
#        ...
#        <filename>, <label_id>
#    into train, valid and test dataset files."""
#
#    if valid + test >= 1:
#        raise ValueError('The sum of valid and test ratio must be smaller than 1')
#
#    d = defaultdict(set)
#
#    for line in open(dataset, 'r'):
#        filename, label_id = line.strip().split(',')
#        d[label_id].add(filename)
#
#    valid_dataset = defaultdict(set) 
#    test_dataset = defaultdict(set) 
#    train_dataset = defaultdict(set) 
#    for label_id, filenames in d.items():
#        if valid > 0:
#            n_valid = int(len(filenames) * valid)
#            if n_valid == 0:
#                raise ValueError(f'The number of instances of label {label_id} is too small.')
#            valid_dataset[label_id] = set(random.sample(d[label_id], n_valid))
#        if test > 0:
#            n_test = int(len(filenames) * test)
#            if n_test == 0:
#                raise ValueError(f'The number of instances of label {label_id} is too small.')
#            test_dataset[label_id] = set(random.sample(d[label_id]-valid_dataset[label_id], n_test))
#        train_dataset[label_id] = d[label_id] - valid_dataset[label_id] - test_dataset[label_id]
#
#    return train_dataset, valid_dataset, test_dataset
#
#
#
#
#
#
