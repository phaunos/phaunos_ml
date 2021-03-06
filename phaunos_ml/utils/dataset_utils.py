import os
import time
from collections import Counter, defaultdict
import multiprocessing
from tqdm import tqdm
import audioread
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import tensorflow as tf
from tensorflow.python.framework import dtypes

from .audio_utils import audiofile2tfrecord
from .annotation_utils import read_annotation_file, ANN_EXT
from .tf_serialization_utils import serialized2data


"""
Utils for handling dataset files, being defined as a list of
audio file paths relative to a root path.
"""

def data2tfrecord(
        line,
        root_path,
        outdir_path,
        feature_extractor,
        audioroot_relpath,
        annroot_relpaths,
        label_sets,
        **kwargs
):
    """Target function for dataset2tfrecord multiprocessing"""

    audiofile_relpath = line.strip()
    annfile_relpaths = [audiofile_relpath.replace(audioroot_relpath, annroot_relpath) \
            .replace('.wav', ANN_EXT) for annroot_relpath in annroot_relpaths]

    audiofile2tfrecord(
        root_path,
        audiofile_relpath,
        outdir_path,
        feature_extractor,
        annfile_relpaths,
        label_sets,
        **kwargs
    )

def dataset2tfrecords(
        root_path,
        datasetfile_path,
        outdir_path,
        feature_extractor,
        audioroot_relpath,
        annroot_relpaths,
        label_sets,
        n_processes=None,
        **kwargs
):
    """ Compute fixed-size examples with features for all audio files in the dataset
    file and write to tfrecords.

    Args:
        root_path: See audio_utils.audiofile2tfrecord.
        datasetfile_path: file containing a list of audio file paths relative to root_path
        outdir_path: See audio_utils.audio2tfrecord.
        feature_extractor: See audio_utils.audio2tfrecord.
        audioroot_relpath: root path of the audio files, relative to root_path
        annroot_relpaths: list of root paths of the annotation files, relative to root_path
        label_sets: See audio_utils.audio2tfrecord.
        n_processes: number of processes to split the computation in.
                     If None, os.cpu_count() is used (multiprocessing.Pool's default)
            
    Returns:
        Write tfrecords in outdir_path.
    """

    if not n_processes:
        n_processes = os.cpu_count()

    lines = open(datasetfile_path, 'r').readlines()

    # Start processes
    pool = multiprocessing.Pool(n_processes)
    for line in lines:
        if line.startswith('#'):
            continue
        pool.apply_async(
            data2tfrecord,
            args=(
                line,
                root_path,
                outdir_path,
                feature_extractor,
                audioroot_relpath,
                annroot_relpaths,
                label_sets,
            ),
            kwds=kwargs
        )
    pool.close()
    pool.join()


def tfrecords2dataset(
        tfrecords,
        class_list,
        one_hot_label=True,
        batch_size = 8,
        shuffle_files=True,
        interleave_cycle_length=10,
        shuffle_size=1000,
        repeat=True
):
    """Generate a tf.data.Dataset from TFRecords.

    Args:
        tfrecords: list of TFRecords
        class_list: list of the class indices used in the dataset (used for one-hot encoding the label)
        one_hot_label (bool): whether to return the label as one-hot vector
            (multi-class classification) or binary value (binary classification)
        batch_size (int)
        shuffle_files (bool): shuffle tfrecord files before loading them
        interleave_cycle_length (int): see Tensorflow documentation:
            https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave
        shuffle_size (int): size of the buffer to be shuffled. 0 means no shuffle. A too big
            buffer might not fit in memory.
        repeat (bool): whether to repeat or not the dataset.
    """

    # Get list of files
    files = tf.convert_to_tensor(tfrecords, dtype=dtypes.string)
    files = tf.data.Dataset.from_tensor_slices(files)

    # Shuffle the files.
    # We can take a buffer of the size of the list, because it only contains strings
    # so the buffer will easily fits in memory.
    if shuffle_files:
        files = files.shuffle(len(tfrecords), reshuffle_each_iteration=True)

    # Read TFrecords
    if interleave_cycle_length < 2:
        dataset = tf.data.TFRecordDataset(files)
    else:
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=interleave_cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # deserialize to feature and one-hot encoded labels
    dataset = dataset.map(lambda x: serialized2data(
        x,
        class_list,
        one_hot_label=one_hot_label
    ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle, repeat and batch
    if shuffle_size:
        dataset = dataset.shuffle(shuffle_size)
    if repeat:
        dataset = dataset.repeat()
    if batch_size:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def create_subset(
        root_path,
        outdir_path,
        subset_relpath_list=['.'],
        audioroot_relpath='audio',
        annroot_relpath='annotations',
        label_set=None,
        max_num_files_per_label=None
):
    """Create a file with a list of audio_file.
    
    Args:
        root_path: root path of the audio and (optionally) annotation files.
        outdir_path: path of the output directory
        subset_relpath_list: list of paths, relative to root_path, containing audio and (optionally) annotation files
        audioroot_relpath: root path of the audio files, relative to every element in subset_relpath_list
        annroot_relpath: root path of the annotation files, relative to every element in subset_relpath_list
        label_set: if set, only files having at least one label from label_set are kept.
        max_num_files_per_label: set the maximum number of files per class. No maximum if set to None.

    Returns:
        Writes a dataset file of all audio files in subset_relpath_list having at least one label from label_set (if specified).
    """

    # create a folder for this subset
    subset_name = 'subset_{}'.format(str(int(time.time())))
    subsetfile_path = os.path.join(outdir_path, subset_name, f'{subset_name}.csv')
    os.makedirs(os.path.dirname(subsetfile_path), exist_ok=True)

    # Num files per class counter
    counter = Counter()

    with open(subsetfile_path, 'w') as out_file:
        if label_set:
            out_file.write('#class subset: {}\n'.format(','.join([str(i) for i in sorted(list(label_set))])))
        for subset_relpath in subset_relpath_list:
            audioroot_path = os.path.join(root_path, subset_relpath, audioroot_relpath)
            annroot_path = os.path.join(root_path, subset_relpath, annroot_relpath)
            for path, _, filenames in os.walk(audioroot_path):
                for filename in filenames:
                    add_file = True

                    # get file labels
                    ann_filename = os.path.join(
                        annroot_path,
                        os.path.relpath(path, audioroot_path),
                        filename.replace('.wav', ANN_EXT))
                    ann_set = read_annotation_file(ann_filename)
                    file_label_set = set.union(*map(lambda x:set(x.label_set), ann_set))

                    # get intersection
                    if label_set:
                        file_label_set = file_label_set.intersection(label_set)
                        if not file_label_set:
                            add_file = False
                        elif max_num_files_per_label:
                            # update counter
                            counter.update(file_label_set)
                            # if a class has more files than max_num_files_per_label, do not add
                            if np.any(np.array([counter[l] for l in file_label_set])>max_num_files_per_label):
                                add_file = False

                    # write file
                    if add_file:
                        audiofile_relpath = os.path.join(
                            os.path.relpath(path, root_path),
                            filename
                        )
                        out_file.write(f'{audiofile_relpath}\n')

    return subsetfile_path


def write_dataset_file(root_path, outfile_path, audioroot_relpath='audio'):
    """Write dataset file (list of audio file paths relative to root_path).

    Args:
        root_path: root path of the audio and (optionally) annotation files.
        outfile_path: output file
        audioroot_relpath: root path of the audio files, relative to root_path
    """
    
    with open(outfile_path, 'w') as f:
        for path, _, filenames in os.walk(os.path.join(root_path, audioroot_relpath)):
            for filename in filenames:
                if not filename.endswith('.wav'):
                    continue
                audiofile_relpath = os.path.relpath(
                    os.path.join(path, filename),
                    root_path
                )
                f.write(f'{audiofile_relpath}\n')


def read_dataset_file(
        root_path,
        datasetfile_path,
        audioroot_relpath='audio',
        annroot_relpath='annotations'):

    """Read dataset file (list of audio file paths relative to root_path).
    
    Args:
        root_path: root path of the audio and (optionally) annotation files.
        datasetfile_path: file containing a list of audio file paths relative to root_path
        audioroot_relpath: root path of the audio files, relative to root_path
        annroot_relpath: root path of the annotation files, relative to root_path
    """

    audiofile_relpaths = []
    labels = []

    for line in open(datasetfile_path, 'r'):
        if line.startswith('#') or not line.strip():
            continue
        audiofile_relpath = line.strip()

        # get annotation labels
        ann_filename = os.path.join(
            root_path,
            audiofile_relpath.replace(audioroot_relpath, annroot_relpath).replace('.wav', ANN_EXT))
        ann_set = read_annotation_file(ann_filename)
        labels.append(set.union(*map(lambda x:set(x.label_set), ann_set)))

        audiofile_relpaths.append(audiofile_relpath)

    return audiofile_relpaths, labels


def split_dataset(
        root_path,
        datasetfile_path,
        audioroot_relpath='audio',
        annroot_relpath='annotations',
        label_subset=set(),
        ratio=0.2,
        subset1_name='train',
        subset2_name='valid'
):
    """Split dataset in train and test sets (stratified).
    
    Args:
        root_path: root path of the audio and (optionally) annotation files.
        datasetfile_path: file containing a list of audio file paths relative to root_path
        audioroot_relpath: root path of the audio files, relative to root_path
        annroot_relpath: root path of the annotation files, relative to root_path
        label_subset: subset of labels to be kept.
        ratio: ratio of the number of files to be used for the 2nd subset.
        subset1_name: name appended to the input dataset file for the 1st subset
        subset2_name: name appended to the input dataset file for the 2nd subset
    """

    filenames, labels = read_dataset_file(
        root_path,
        datasetfile_path,
        audioroot_relpath=audioroot_relpath,
        annroot_relpath=annroot_relpath)
    label_set = label_subset if label_subset else set.union(*labels)
    label_list = sorted(list(label_set))

    if label_subset:
        # only keep labels in label_subset
        labels = [l.intersection(label_set) for l in labels]

    multilabel = False
    for file_label_set in labels:
        if len(file_label_set) > 1:
            multilabel = True
            break

    if multilabel:

        # adapt data to iterative_train_test_split arguments
        filenames = np.expand_dims(np.array(filenames), axis=1)
        sparse_labels = lil_matrix((len(filenames), len(label_list)))
        for i, file_label_set in enumerate(labels):
            file_label_ind = [label_list.index(l) for l in file_label_set]
            sparse_labels[i, file_label_ind] = 1

        # multi-label stratified data split
        X1, y1, X2, y2 = iterative_train_test_split(np.array(filenames), sparse_labels, test_size=ratio)

        # write dataset files
        for subset_name, X in [(subset1_name, X1), (subset2_name, X2)]:
            subsetfile_path = datasetfile_path.replace('.csv', f'.{subset_name}.csv')
            with open(subsetfile_path, 'w') as set_file:
                set_file.write('#class subset: {}\n'.format(','.join([str(i) for i in label_list])))
                for i in range(X.shape[0]):
                    set_file.write(f'{X[i,0]}\n')
                print(f'{subsetfile_path} written')
        
    else:

        # adapt data to _train_test_split arguments
        labels = [list(l)[0] for l in labels]
        
        # multi-label stratified data split
        X1, X2, y1, y2 = train_test_split(
            filenames,
            labels,
            test_size=ratio,
            stratify=labels)

        # write dataset files
        for subset_name, X in [(subset1_name, X1), (subset2_name, X2)]:
            subsetfile_path = datasetfile_path.replace('.csv', f'.{subset_name}.csv')
            with open(subsetfile_path, 'w') as set_file:
                set_file.write('#class subset: {}\n'.format(','.join([str(i) for i in sorted(list(label_set))])))
                for filename in X:
                    set_file.write(f'{filename}\n')
                print(f'{subsetfile_path} written')


def dataset_stat_per_file(
        root_path,
        datasetfile_path,
        audioroot_relpath='audio',
        annroot_relpath='annotations',
        get_duration=True):
    """Counts files and (optionally) sum file durations per label in dataset
    
    Args:
        root_path: root path of the audio and (optionally) annotation files.
        datasetfile_path: file containing a list of audio file paths relative to root_path
        audioroot_relpath: root path of the audio files, relative to root_path
        annroot_relpath: root path of the annotation files, relative to root_path
        get_duration: set to True to get total duration per label
    """

    d_num = defaultdict(int)
    d_dur = defaultdict(float)

    filenames, labels = read_dataset_file(
        root_path,
        datasetfile_path,
        audioroot_relpath=audioroot_relpath,
        annroot_relpath=annroot_relpath
    )
    for filename, label in tqdm(zip(filenames, labels)):
        if get_duration:
            audio = audioread.audio_open(
                os.path.join(
                    root_path,
                    filename
                ))
        for l in label:
            d_num[l] += 1
            if get_duration:
                d_dur[l] += audio.duration
        
    return d_num, d_dur


def dataset_stat_per_example(
        datasetfile_path,
        tfrecordroot_path,
        batch_size=32):
    """Counts batches per label in datasetfile_path.
    
    Args:
        datasetfile_path: file containing a list of audio file paths relative to root_path
        tfrecordroot_path: directory containing the tfrecords
        batch_size: batch size

    Returns:
        n_batches: number of batches
        n_examples_per_class: list of integers, such as n_examples_per_class[i] is the
            number of examples for class class_list[i]    
    """

    files = []
    class_list = None
    for line in open(datasetfile_path):
        if line.startswith('#'):
            # parse first line to get class list
            class_list = [int(i) for i in line.split(':')[1].strip().split(',')]
            continue
        f = os.path.join(tfrecordroot_path, line.strip().replace('.wav', '.tf'))
        if not os.path.isfile(f):
            print(f'File {f} not found.')
        else:
            files.append(f)

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda data: serialized2data(data, class_list))
    dataset = dataset.batch(batch_size, drop_remainder=True)

    n_batches = 0
    n_examples_per_class = np.zeros((len(class_list),), dtype=np.int32)

    for _, one_hot, _, _  in tqdm(dataset):
        n_examples_per_class += np.count_nonzero(one_hot, axis=0)
        n_batches += 1

    return n_batches, n_examples_per_class, class_list
