import os
import tensorflow as tf
import librosa

from .tf_utils import serialize_data
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
        audio_filename,
        out_dir,
        feature_extractor,
        annotation_filename=None,
        activity_detector=None,
        min_activity_dur=None
):
    """ Compute fixed-size examples with features (and optionally labels)
    from an audio file and write to a tfrecord.

    Args:
        root_path: root path of the audio files
        audio_filename: filename of the audio file, including path relative to root_path
        out_dir: path of the output directory
        feature_extractor: see :func:`.feature_utils`
        annotation_filename: annotation file, as described in :func:`.annotation_utils`
            If set, labels are also written to the tfrecord
    Returns:
        Writes one tfrecord in out_dir with the same basename as audio_file.
    """

    # read audio
    y, sr = librosa.load(os.path.join(root_path, audio_filename), sr=None)

    # read annotations
    if annotation_filename:
        annotation_set = read_annotation_file(os.path.join(root_path, annotation_filename))
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
        out_dir,
        audio_filename.replace('.wav', '.tf'),
        feature_extractor,
        annotation_set,
        fb_mask=fb_mask,
        fb_mask_sr=fb_mask_sr,
        mask_min_dur=min_activity_dur)


def audio2tfrecord(
        audio,
        sr,
        out_dir,
        filename,
        feature_extractor,
        annotation_set=None,
        start_time_offset=0,
        fb_mask=None,
        fb_mask_sr=None,
        mask_min_dur=None
):

    # compute features, segment-based mask and segment boundaries
    features, mask, times = feature_extractor.process(audio, sr, fb_mask, fb_mask_sr, mask_min_dur)

    # write tfrecord in either 'negative' or 'positive' subfolders
    # according to mask value
    out_filename_neg = os.path.join(out_dir, 'negative', filename)
    out_filename_pos = os.path.join(out_dir, 'positive', filename)
    os.makedirs(os.path.dirname(out_filename_neg), exist_ok=True)
    os.makedirs(os.path.dirname(out_filename_pos), exist_ok=True)
    with tf.io.TFRecordWriter(out_filename_neg) as writer_neg, \
            tf.io.TFRecordWriter(out_filename_pos) as writer_pos:

        for i in range(features.shape[0]):
            start_time, end_time = times[i] 
            # Last example's end time is set to original audio file duration
            # to avoid mislabeling.
            if i == features.shape[0] - 1:
                end_time = start_time_offset + len(audio) / sr
            labels = get_labels_in_range(annotation_set, start_time, end_time) if annotation_set else set()
            sdata = serialize_data(
                filename,
                start_time,
                end_time,
                features[i],
                labels
            )
            if mask[i]:
                writer_pos.write(sdata)
            else:
                writer_neg.write(sdata)

