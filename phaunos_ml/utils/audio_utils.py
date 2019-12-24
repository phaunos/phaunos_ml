import os
import tensorflow as tf
import numpy as np
import librosa

from .tf_serialization_utils import serialize_data
from .annotation_utils import read_annotation_file, get_labels_in_range


"""
Utils to convert audio files to tfrecords.
"""


def seconds2hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def audiofile2tfrecord(
        root_path,
        audiofile_relpath,
        outdir_path,
        feature_extractor,
        annfile_relpath=None,
        activity_detector=None,
        min_activity_dur=None
):
    """ Compute fixed-size examples with features (and optionally labels)
    from an audio file and write to a tfrecord.

    Args:
        root_path: root path of the audio and (optionally) annotation files.
        audiofile_relpath: path, relative to root_path, to the audio file.
        outdir_path: path of the output directory.
        feature_extractor: see :func:`.feature_utils`.
        annfile_relpath: path, relative to root_path, to the annotation file, as described in :func:`.annotation_utils`. If set, labels are also written to the tfrecord.
        activity_detector: frame-based activity_detector, as described in nsb_aad.frame_based_detectors.
        min_activity_dur: minimum duration of activity to be found in an example.
    Returns:
        Writes one tfrecord in <outdir_path>/<audiofile_relpath> (changing the extension of the file
        from '.wav' to '.tf'.
    """

    # read audio
    y, sr = librosa.load(os.path.join(root_path, audiofile_relpath), sr=None)

    # read annotations
    if annfile_relpath:
        annotation_set = read_annotation_file(os.path.join(root_path, annfile_relpath))
    else:
        annotation_set =  None

    if activity_detector:
        # compute frame-based mask
        fb_mask = activity_detector.process(y)
        fb_mask_sr = activity_detector.frame_rate
    else:
        fb_mask = None
        fb_mask_sr = None


    audio2tfrecord(
        y,
        sr,
        outdir_path,
        audiofile_relpath.replace('.wav', '.tf'),
        feature_extractor,
        annotation_set,
        fb_mask=fb_mask,
        fb_mask_sr=fb_mask_sr,
        mask_min_dur=min_activity_dur)


def audio2tfrecord(
        audio,
        sr,
        outdir_path,
        tfrecord_relpath,
        feature_extractor,
        annotation_set=None,
        fb_mask=None,
        fb_mask_sr=None,
        mask_min_dur=None
):
    """ Compute fixed-size examples with features (and optionally labels)
    from audio data and write to a tfrecord.

    Args:
        audio: mono audio data (np array)
        sr: sample rate
        outdir_path: path of the output directory.
        tfrecord_relpath: relative path to which the tfrecord will be written in outdir_path.
        feature_extractor: see :func:`.feature_utils`.
        annotation_set: set of annotation objects.
        fb_mask: frame-based mask.
        fb_mask_sr: frame-based mask sample rate.
        mask_min_dur: minimum total duration of positive mask frames.
    Returns:
        Writes one tfrecord in <outdir_path>/<audiofile_relpath> (changing the extension of the file
        from '.wav' to '.tf'.
    """

    # compute features, segment-based mask and segment boundaries
    features, mask, times = feature_extractor.process(audio, sr, fb_mask, fb_mask_sr, mask_min_dur)

    # write tfrecord in either 'negative' or 'positive' subfolders
    # according to mask value
    out_filename_neg = os.path.join(outdir_path, 'negative', tfrecord_relpath)
    out_filename_pos = os.path.join(outdir_path, 'positive', tfrecord_relpath)
    os.makedirs(os.path.dirname(out_filename_neg), exist_ok=True)
    os.makedirs(os.path.dirname(out_filename_pos), exist_ok=True)
    with tf.io.TFRecordWriter(out_filename_neg) as writer_neg, \
            tf.io.TFRecordWriter(out_filename_pos) as writer_pos:

        for i in range(features.shape[0]):
            start_time, end_time = times[i] 
            # Last example's end time is set to original audio file duration
            # to avoid mislabeling.
            if i == features.shape[0] - 1:
                end_time = len(audio) / sr
            labels = get_labels_in_range(annotation_set, start_time, end_time) \
                if annotation_set else set()
            sdata = serialize_data(
                tfrecord_relpath,
                start_time,
                end_time,
                features[i],
                labels
            )
            if mask[i]:
                writer_pos.write(sdata)
            else:
                writer_neg.write(sdata)


def audio2data(
        audio,
        sr,
        feature_extractor,
        activity_detector,
        class_list,
        annotation_set=None,
        mask_min_dur=None
):
    """ Compute fixed-size examples with features (and optionally labels)
    from audio data.

    Args:
        audio: mono audio data (np array)
        sr: sample rate
        feature_extractor: see :func:`.feature_utils`.
        activity_detector: frame-based activity_detector, as described in nsb_aad.frame_based_detectors.
        class_list: list of classes used as the reference for one-hot label encoding.
        annotation_set: set of annotation objects.
        mask_min_dur: minimum total duration of positive mask frames.
    Returns:
        Lists of tuples for positive and negative examples.
    """

    # compute mask from activity_detection
    fb_mask = activity_detector.process(audio)
    fb_mask_sr = activity_detector.frame_rate

    # compute features, segment-based mask and segment boundaries
    features, mask, times = feature_extractor.process(audio, sr, fb_mask, fb_mask_sr, mask_min_dur)

    examples_pos = []
    examples_neg = []
    for i in range(features.shape[0]):

        start_time, end_time = times[i] 
        if i == features.shape[0] - 1:
            # Last example's end time is set to original audio file duration
            # to avoid mislabeling.
            end_time = start_time_offset + len(audio) / sr
        labels = get_labels_in_range(annotation_set, start_time, end_time) if annotation_set else set()

        # one-hot encode labels
        ind = np.where(np.in1d(
            sorted(class_list),
            list(labels)))[0]
        one_hot = np.zeros((len(class_list),))
        np.put(one_hot, ind, 1)

        if mask[i]:
            examples_pos.append((features[i], one_hot))
        else:
            examples_neg.append((features[i], one_hot))

    return examples_pos, examples_neg
