import os
import time
from collections import Counter, defaultdict
from tqdm import tqdm
import audioread
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import tensorflow as tf

from .audio_utils import audiofile2tfrecord
from .annotation_utils import read_annotation_file, ANN_EXT
from .tf_utils import tfrecords2tfdataset


"""
Utils for handling dataset, being defined as a file containing a list of
audio files and, optionally, annotation files.
"""


def dataset2tfrecords(
        root_path,
        dataset_file,
        out_dir,
        feature_extractor,
        activity_detector=None,
        min_activity_dur=None,
        audio_dirname='audio',
        annotation_dirname='annotations',
        with_labels=True
):
    """ Compute fixed-size examples with features (and optionally labels)
    for all audio files in the dataset file and write to tfrecords.

    Args:
        root_path: root path of the audio files
        dataset_file: file containing a list of audio filenames, relative to root_path
        out_dir: path of the output directory
        feature_extractor: see :func:`.feature_utils`
        audio_dirname: name of the directory containing audio files (see below)
        annotation_dirname: name of the directory containing annotation files. Annotation files must
            have the same path as the audio files, just replacing audio_dirname by annotation_dirname.
        with_labels: whether to include labels in the tfrecords.
            
    Returns:
        Write tfrecords in out_dir.
    """

    for line in tqdm(open(dataset_file, 'r').readlines()):
        if line.startswith('#'):
            continue
        audio_filename = line.strip()
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
            annotation_filename=annotation_filename,
            activity_detector=activity_detector,
            min_activity_dur=min_activity_dur
        )


def create_subset(
        root_path,
        subset_path_list,
        out_dir,
        audio_dirname='audio',
        annotation_dirname='annotations',
        label_set=None
):
    """Create a file with a list of audio_file.
    
    Args:
        root_path: root path of the audio files
        subset_path_list: list of directories containing audio and annotation files
        out_dir: path of the output directory
        audio_dirname: name of the directory containing audio files (see below)
        annotation_dirname: name of the directory containing annotation files. Annotation files must
            have the same path as the audio files, just replacing audio_dirname by annotation_dirname.
        label_set: if set, only files having at least one label from label_set are kept.

    Returns:
        Writes a dataset file of all audio files in subset_path_list with at least one label from label_set (if specified).
    """

    # create a folder for this subset
    subset_name = 'subset_{}'.format(str(int(time.time())))
    subset_filename = os.path.join(out_dir, subset_name, f'{subset_name}.csv')
    os.makedirs(os.path.dirname(subset_filename), exist_ok=True)

    with open(subset_filename, 'w') as out_file:
        if label_set:
            out_file.write('#class subset: {}\n'.format(','.join([str(i) for i in sorted(list(label_set))])))
        for subset_path in subset_path_list:
            audio_path = os.path.join(root_path, subset_path, audio_dirname)
            ann_path = os.path.join(root_path, subset_path, annotation_dirname)
            for file_path, _, filenames in os.walk(audio_path):
                for filename in filenames:
                    add_file = True

                    # get file labels
                    ann_filename = os.path.join(
                        ann_path,
                        os.path.relpath(file_path, audio_path),
                        filename.replace('.wav', ANN_EXT))
                    ann_set = read_annotation_file(ann_filename)
                    file_label_set = set.union(*map(lambda x:set(x.label_set), ann_set))

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
                        out_file.write(f'{audio_filename}\n')

    return subset_filename


def read_dataset_file(
        root_path,
        dataset_file,
        audio_dirname='audio',
        annotation_dirname='annotations',
        replace_ext=''):

    """Read dataset file"""

    audio_filenames = []
    labels = []

    for line in open(dataset_file, 'r'):
        if line.startswith('#') or not line.strip():
            continue
        audio_filename = line.strip().split(',')[0]

        # get annotation labels
        ann_filename = os.path.join(
            root_path,
            audio_filename.replace(audio_dirname, annotation_dirname).replace('.wav', ANN_EXT))
        ann_set = read_annotation_file(ann_filename)
        labels.append(set.union(*map(lambda x:set(x.label_set), ann_set)))

        if replace_ext:
            audio_filename = audio_filename.replace('.wav', replace_ext)
        audio_filenames.append(audio_filename)

    return audio_filenames, labels


def split_dataset(
        root_path,
        dataset_file,
        audio_dirname='audio',
        annotation_dirname='annotations',
        test_size=0.2):
    """Split dataset in train and test sets (stratified)."""

    filenames, labels = read_dataset_file(
        root_path,
        dataset_file,
        audio_dirname=audio_dirname,
        annotation_dirname=annotation_dirname)
    label_set = set.union(*labels)
    label_list = sorted(list(label_set))

    multilabel = False
    for file_label_set in labels:
        if len(file_label_set) > 1:
            multilabel = True

    if multilabel:

        # adapt data to iterative_train_test_split arguments
        filenames = np.expand_dims(np.array(filenames), axis=1)
        sparse_labels = lil_matrix((len(filenames), len(label_list)))
        for i, file_label_set in enumerate(labels):
            file_label_ind = [label_list.index(l) for l in file_label_set]
            sparse_labels[i, file_label_ind] = 1

        # multi-label stratified data split
        X_train, y_train, X_test, y_test = iterative_train_test_split(np.array(filenames), sparse_labels, test_size=test_size)

        # write dataset files
        for set_name, X in [('train', X_train), ('test', X_test)]:
            set_filename = dataset_file.replace('.csv', f'.{set_name}.csv')
            with open(set_filename, 'w') as set_file:
                set_file.write('#class subset: {}\n'.format(','.join([str(i) for i in sorted(list(label_set))])))
                for i in range(X.shape[0]):
                    set_file.write(f'{X[i,0]}\n')
                print(f'{set_filename} written')
        
    else:

        # adapt data to _train_test_split arguments
        labels = [list(l)[0] for l in labels]
        
        # multi-label stratified data split
        X_train, X_test, y_train, y_test = train_test_split(
            filenames,
            labels,
            test_size=test_size,
            stratify=labels)

        # write dataset files
        for set_name, X in [('train', X_train), ('test', X_test)]:
            set_filename = dataset_file.replace('.csv', f'.{set_name}.csv')
            with open(set_filename, 'w') as set_file:
                set_file.write('#class subset: {}\n'.format(','.join([str(i) for i in sorted(list(label_set))])))
                for filename in X:
                    set_file.write(f'{filename}\n')
                print(f'{set_filename} written')


def dataset_stat_per_file(
        root_path,
        dataset_file,
        audio_dirname='audio',
        annotation_dirname='annotations'):
    """Counts files and sum file durations per label in dataset"""

    d_num = defaultdict(int)
    d_dur = defaultdict(float)

    filenames, labels = read_dataset_file(
        root_path,
        dataset_file,
        audio_dirname=audio_dirname,
        annotation_dirname=annotation_dirname
    )
    for filename, label in tqdm(zip(filenames, labels)):
        for l in label:
            d_num[l] += 1
            audio = audioread.audio_open(
                os.path.join(
                    root_path,
                    filename
                ))
            d_dur[l] += audio.duration
        
    return d_num, d_dur


def dataset_stat_per_example(
        root_path,
        dataset_file,
        tfrecord_path,
        feature_shape,
        class_list,
        batch_size=32,
        audio_dirname='audio',
        annotation_dirname='annotations'):
    """Counts batches per label in dataset_file.
    
    Args:
        dataset_file: file containing a list of audio filenames, relative to root_path
        tfrecord_path: directory containing the tfrecords
        example shape: shape of the examples
        class_list: list of the classes used in the dataset (the label ids in the tfrecords are
            indices in this class_list)
        batch_size: batch size

    Returns:
        n_batches: number of batches
        n_examples_per_class: list of integers, such as n_examples_per_class[i] is the
            number of examples for class class_list[i]    
    """

    files, labels = read_dataset_file(
        root_path,
        dataset_file,
        audio_dirname=audio_dirname,
        annotation_dirname=annotation_dirname,
        replace_ext='.tf')
    
    files = [os.path.join(tfrecord_path, f) for f in files]

    dataset = tfrecords2tfdataset(
        files,
        feature_shape,
        class_list,
        training=False,
        batch_size=batch_size
    )
    it = dataset.make_one_shot_iterator()

    n_batches = 0
    n_examples_per_class = np.zeros((len(class_list),), dtype=np.int32)

    if tf.executing_eagerly():
        for _, one_hot, _, _  in it:
            n_examples_per_class += np.count_nonzero(one_hot, axis=0)
            n_batches += 1
    else:
        next_example = it.get_next()
        with tf.Session() as sess:
            try:
                while True:
                    _, one_hot, _, _ = sess.run(next_example)
                    n_examples_per_class += np.count_nonzero(one_hot, axis=0)
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass

    return n_batches, n_examples_per_class
