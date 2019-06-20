import os
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
import time
from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split

from .audio_utils import audiofile2tfrecord
from .annotation_utils import read_annotation_file, ANN_EXT
from .tf_utils import filelist2tfdataset


def dataset2tfrecords(
        root_path,
        dataset_file,
        out_dir,
        feature_extractor,
        audio_dirname='audio',
        annotation_dirname='annotations',
        with_labels=True
):
    for line in tqdm(open(dataset_file, 'r').readlines()):
        if line.startswith('#'):
            continue
        audio_filename = line.strip().split(',')[0]
        if with_labels:
            annotation_filename = audio_filename.replace(audio_dirname, annotation_dirname) \
                .replace('.wav', ANN_EXT)
        else:
            annotation_filename = None
        audiofile2tfrecord(
            root_path,
            audio_filename,
            out_dir,
            feature_extractor,
            annotation_filename=annotation_filename
        )


def create_subset(root_path, subset_path_list, out_path, audio_dirname='audio', ann_dirname='annotations', label_set=None):
    """Create a file with a list of audio_file.
    If label_set is set, only files having at least one label from label_set are kept."""


    # create a folder for this subset
    subset_name = 'subset_{}'.format(str(int(time.time())))
    subset_filename = os.path.join(out_path, subset_name, f'{subset_name}.csv')
    os.makedirs(os.path.dirname(subset_filename), exist_ok=True)

    with open(subset_filename, 'w') as out_file:
        if label_set:
            out_file.write('#class subset: {}\n'.format(','.join([str(i) for i in sorted(list(label_set))])))
        for subset_path in subset_path_list:
            audio_path = os.path.join(root_path, subset_path, audio_dirname)
            ann_path = os.path.join(root_path, subset_path, ann_dirname)
            for file_path, _, filenames in os.walk(audio_path):
                for filename in filenames:
                    add_file = True

                    # get file labels
                    ann_filename = os.path.join(
                        ann_path,
                        os.path.relpath(file_path, audio_path),
                        filename.replace('.wav', ANN_EXT))
                    ann_set = read_annotation_file(ann_filename)
                    file_label_set = set()
                    for ann in ann_set:
                        file_label_set.update(ann.label_set)

                    # get intersection
                    if label_set:
                        file_label_set = file_label_set.intersection(label_set)
                        if not file_label_set:
                            add_file = False

                    # write file
                    if add_file:
                        audio_filename = os.path.join(
                            os.path.relpath(file_path, root_path),
                            filename
                        )
                        file_label_set_str = '#'.join(str(i) for i in file_label_set)
                        out_file.write(f'{audio_filename},{file_label_set_str}\n')


def read_dataset_file(dataset_file, prepend_path='', replace_ext=''):

    filenames = []
    labels = []

    for line in open(dataset_file, 'r'):
        if line.startswith('#'):
            continue
        filename, file_label_set_str = line.strip().split(',')
        if replace_ext:
            filename = filename.replace('.wav', replace_ext)
        filenames.append(os.path.join(prepend_path, filename))
        file_label_set = set([int(i) for i in file_label_set_str.split('#')])
        labels.append(file_label_set)

    return filenames, labels


def split_dataset(dataset_file, test_size=0.2):
    """Split dataset in train and test sets (stratified)."""

    filenames, labels = read_dataset_file(dataset_file)
    label_set = set.union(*labels)
    label_list = sorted(list(label_set))

    multilabel = False
    for file_label_set in labels:
        if len(file_label_set) > 1:
            multilabel = True

    if multilabel:

        # adapt data to iterative_train_test_split input
        filenames = np.expand_dims(np.array(filenames), axis=1)
        sparse_labels = lil_matrix((len(filenames), len(label_list)))
        for i, file_label_set in enumerate(labels):
            file_label_ind = [label_list.index(l) for l in file_label_set]
            sparse_labels[i, file_label_ind] = 1

        # multi-label stratified data split
        X_train, y_train, X_test, y_test = iterative_train_test_split(np.array(filenames), sparse_labels, test_size=test_size)

        # write dataset files
        for set_name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            set_filename = dataset_file.replace('.csv', f'.{set_name}.csv')
            with open(set_filename, 'w') as set_file:
                set_file.write('#class subset: {}\n'.format(','.join([str(i) for i in sorted(list(label_set))])))
                for i in range(X.shape[0]):
                    file_label_list = sparse.find(y[i])[1]
                    file_label_str = '#'.join([str(label_list[ind]) for ind in file_label_list])
                    set_file.write(f'{X[i,0]},{file_label_str}\n')
        
    else:
        print("Not implemented")


def dataset_stat_per_file(dataset_file):

    filenames, labels = read_dataset_file(dataset_file)
    label_set = set.union(*labels)

    return Counter([l for label_set in labels for l in label_set])


def dataset_stat_per_example(dataset_file, tfrecord_path, example_shape, class_list, batch_size=32):

    files, labels = read_dataset_file(dataset_file, prepend_path=tfrecord_path, replace_ext='.tf')
    dataset = filelist2tfdataset(
        files,
        example_shape,
        class_list,
        training=False,
        batch_size=batch_size
    )
    it = dataset.make_one_shot_iterator()

    n_batches = 0
    n_examples_per_class = np.zeros((len(class_list),))

    for _, one_hot, _, _  in it:
        n_examples_per_class += np.count_nonzero(one_hot, axis=0)
        n_batches += 1

    return n_batches, n_examples_per_class




