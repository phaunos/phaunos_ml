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
        chunk_duration=None,
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
        chunk_duration: if set, create examples for chunks of the audio file
            (used for example in BirdCLEF 2019).
    Returns:
        Writes one tfrecord (or multiple if chunk_duration is set) in out_dir with the same
        basename as audio_file.
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


    if not chunk_duration:
        audio2tfrecord(
            y,
            sr,
            out_dir,
            audio_filename.replace('.wav', '.tf'),
            feature_extractor,
            annotation_set,
            fb_mask=fb_mask,
            fb_mask_sr=fb_mask_sr,
            mask_min_dur=min_activity_dur
        )
    else:
        chunk_ind = 0
        chunk_size = int(chunk_duration * sr)
        start_sample_ind = 0
        while start_sample_ind + chunk_size < len(y):
            start_time_offset = start_sample_ind / sr
            start_tuple = seconds2hms(int(start_time_offset))
            end_tuple = seconds2hms(int(start_time_offset + chunk_duration))
            start_str = f'{start_tuple[0]:02d}:{start_tuple[1]:02d}:{start_tuple[2]:02d}'
            end_str = f'{end_tuple[0]:02d}:{end_tuple[1]:02d}:{end_tuple[2]:02d}'
            out_filename = os.path.join(
                audio_filename.replace('.wav', ''),
                audio_filename.replace('.wav', f'_{start_str}-{end_str}.tf')
            )
            audio2tfrecord(
                y[start_sample_ind:start_sample_ind+chunk_size],
                sr,
                out_dir,
                out_filename,
                feature_extractor,
                annotation_set,
                start_time_offset=start_time_offset
            )
            chunk_ind += 1
            start_sample_ind = int(chunk_ind * chunk_duration * sr) # This is the only point of using chunks,
                                                                    # to avoid cumulative rounding errors of standard split.

        # last chunk
        min_last_chunk_duration = feature_extractor.actual_example_duration / 4 # arbitrary
        if (len(y) - start_sample_ind) / sr > min_last_chunk_duration:
            start_time_offset = start_sample_ind / sr
            out_filename = os.path.join(
                out_dir,
                audio_filename.replace('.wav', ''),
                audio_filename.replace('.wav', f'_{end_str}-end.tf')
            )
            audio2tfrecord(
                y[start_sample_ind:],
                sr,
                out_filename,
                annotation_set,
                feature_extractor,
                start_time_offset=start_time_offset
            )


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

