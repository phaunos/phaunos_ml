import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import soundfile as sf

from phaunos_ml.utils.audio_utils import audio2tfrecord
from phaunos_ml.utils.annotation_utils import Annotation, write_annotation_file, PhaunosAnnotationError


def get_class_list(species_filename):
    class_codes = sorted([line.split(',')[0] for line in open(species_filename, 'r')][1:])
    assert len(class_codes) == 659, 'Wrong number of classes (must be 659)'
    return sorted(class_codes)


def timecode2seconds(timecode):
    time = datetime.strptime(timecode, '%H:%M:%S').time()
    return time.hour * 3600 + time.minute * 60 + time.second


def generate_train_ann_files(audio_path, metadata_path, ann_path, class_list):

    class_list = sorted(class_list)

    for root, _, filenames in os.walk(audio_path):
        for filename in filenames:
            rel_path = os.path.relpath(root, audio_path)

            metadata_filename = os.path.join(
                metadata_path,
                rel_path,
                filename.replace('wav', 'json')
            )
            ann_filename = os.path.join(
                ann_path,
                rel_path,
                filename.replace('wav', 'ann')
            )
            with open(metadata_filename, 'r') as json_file:
                json_data = json.load(json_file)
                class_ind = class_list.index(json_data['class code'])
                write_annotation_file([Annotation(label_set=set([class_ind]))], ann_filename)


def generate_val_ann_files(audio_path, metadata_path, ann_path, class_list):

    class_list = sorted(class_list)

    for root, _, filenames in os.walk(audio_path):
        for filename in filenames:
            ann_list = []
            rel_path = os.path.relpath(root, audio_path)

            metadata_filename = os.path.join(
                metadata_path,
                rel_path,
                filename.replace('wav', 'json')
            )
            with open(metadata_filename, 'r') as json_file:
                json_data = json.load(json_file)
                for data in json_data['ClassIds']:

                    try:
                        start_time_str, end_time_str = data['TimeCodes'].split('-')
                        start_time = timecode2seconds(start_time_str)
                        end_time = timecode2seconds(end_time_str)
                        class_ind = class_list.index(data['ClassId'])
                        ann_list.append(Annotation(start_time, end_time, set([class_ind])))
                    except PhaunosAnnotationError as e:
                        print(f'File {metadata_filename}: {e})')

            ann_filename = os.path.join(
                ann_path,
                rel_path,
                filename.replace('wav', 'ann')
            )
            write_annotation_file(ann_list, ann_filename)


def make_dataframe(audio_path, metadata_path):
    """Get {class_name: num_instance} dict."""

    data = []

    for root, _, filenames in os.walk(audio_path):
        for filename in filenames:
            if not filename.startswith('._') and filename.endswith('wav'):

                try:

                    label = root.split('/')[-1]
                    with sf.SoundFile(os.path.join(root, filename)) as audio_file:
                        duration = len(audio_file) / audio_file.samplerate

                    json_filename = os.path.join(metadata_path, label, filename.replace('wav', 'json'))
                    with open(json_filename, 'r') as json_file:
                        json_data = json.load(json_file)
                    
                    data.append([
                        label,
                        filename,
                        duration,
                        json_data['species'],
                        json_data['longitude'],
                        json_data['latitude'],
                        json_data['elevation'],
                    ])
                    
                except FileNotFoundError as e:
                    print(f"{e.filename} not found")
                                  
    return pd.DataFrame(data, columns=['label', 'filename', 'duration', 'species', 'longitude', 'latitude', 'elevation']) 
