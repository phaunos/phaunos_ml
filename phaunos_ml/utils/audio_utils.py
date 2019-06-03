import os
import tensorflow as tf
import librosa

from .feature_utils import MelSpecExtractor
from .tf_utils import serialize_data
from .annotation_utils import read_annotation_file, get_labels_in_range


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
        chunk_duration=None
):
    
    # read audio
    y, sr = librosa.load(os.path.join(root_path, audio_filename), sr=None)

    # read annotations
    if annotation_filename:
        annotation_set = read_annotation_file(os.path.join(root_path, annotation_filename))
    else:
        annotation_set =  None

    if not chunk_duration:
        out_filename = os.path.join(
            out_dir,
            audio_filename.replace('.wav', '.tf')
        )
        audio2tfrecord(
            y,
            sr,
            out_filename,
            feature_extractor,
            annotation_set
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
                out_dir,
                audio_filename.replace('.wav', ''),
                audio_filename.replace('.wav', f'_{start_str}-{end_str}.tf')
            )
            audio2tfrecord(
                y[start_sample_ind:start_sample_ind+chunk_size],
                sr,
                out_filename,
                feature_extractor,
                annotation_set,
                start_time_offset=start_time_offset
            )
            chunk_ind += 1
            start_sample_ind = int(chunk_ind * chunk_duration * sr) # This is the only point of using chunks,
                                                                    # to avoid cumulative rounding errors of standard split.

        # last chunk
        if (len(y) - start_sample_ind) / sr > MIN_LAST_CHUNK_DURATION:
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
        out_filename,
        feature_extractor,
        annotation_set=None,
        start_time_offset=0
):

    # compute features
    features = feature_extractor.process(audio, sr)

    # write tfrecord
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(features.shape[0]):
        start_time = start_time_offset + feature_extractor.actual_example_hop_duration * i
        # Last example's end time is set to original audio file duration
        # to avoid mislabeling.
        if i == features.shape[0] - 1:
            end_time = start_time_offset + len(audio) / sr
        else:
            end_time = start_time + feature_extractor.actual_example_duration
        labels = get_labels_in_range(annotation_set, start_time, end_time) if annotation_set else set()
        sdata = serialize_data(
            os.path.basename(out_filename),
            start_time,
            end_time,
            features[i],
            labels
        )
        writer.write(sdata)
