import os

from phaunos_ml.utils.annotation_utils import Annotation, write_annotation_file


def get_class_list(sample_submission_filename):
    first_line = open(sample_submission_filename, 'r').readline()
    labels = first_line.strip().split(',')[1:]
    assert len(labels) == 80, 'Wrong number of classes (must be 80)'
    return sorted(labels)
 

def generate_ann_files(audio_path, label_filename, ann_path, class_list):

    class_list = sorted(class_list)

    label_dict = dict()
    lines = open(label_filename, 'r').readlines()[1:]
    for line in lines:
        cols = line.strip().replace('"', '').split(',')
        audio_filename = cols[0]
        label_ind = set([class_list.index(s) for s in cols[1:]])
        label_dict[audio_filename] = label_ind

    for root, _, filenames in os.walk(audio_path):
        for filename in filenames:
            rel_path = os.path.relpath(root, audio_path)

            ann_filename = os.path.join(
                ann_path,
                rel_path,
                filename.replace('wav', 'ann')
            )
            write_annotation_file([Annotation(label_set=label_dict[filename])], ann_filename)
