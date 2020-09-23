import os


"""
Utils for handling simple objects holding information on an audio file annotation, namely:
    - start_time, in seconds
    - end_time, in seconds (or -1 if the annotation is for the whole audio file)
    - label_set: label ids (set of int)

Annotation sets are written/read to/from csv files with lines in the following format:
    start_time,end_time,label_0#...#label_N
"""


ANN_EXT = '.ann'


class PhaunosAnnotationError(Exception):
    pass


class Annotation:

    def __init__(self, start_time=0, end_time=-1, label_set=frozenset()):
        if start_time < 0 or (end_time != -1 and end_time < start_time):
            raise PhaunosAnnotationError(
                "Wrong time parameters: start_time must be " +
                "greater than or equal to 0 and end_time must be greater than or equal to start_time " +
                f"(got {start_time} and {end_time})"
            )

        self._start_time = start_time
        self._end_time = end_time
        self._label_set = frozenset(label_set)
 
    @property
    def start_time(self):
        return self._start_time
    
    @property
    def end_time(self):
        return self._end_time

    @property
    def label_set(self):
        return self._label_set   


    def __eq__(self, other):
        return (self.start_time == other.start_time and 
                self.end_time == other.end_time and 
                self.label_set == other.label_set)

    def __lt__(self, other):
        return (self.start_time < other.start_time
                 and (other.end_time == -1 or self.end_time < other.end_time))

    def __hash__(self):
        return hash((self.start_time, self.end_time, self.label_set))

    def __repr__(self):
        return f'start_time: {self.start_time}, end_time: {self.end_time}, label_set: {self.label_set}'


def read_annotation_file(annotation_filename):
    """Return set of Annotation from csv file with lines in format
        start_time,end_time,label_0#...#label_N
    """
    annotation_set = set()
    for line in open(annotation_filename, 'r'):
        if line.startswith('#'):
            continue
        start_time_str, end_time_str, label_set_str = line.strip().split(',')
        annotation_set.add(Annotation(float(start_time_str), float(end_time_str), {int(i) for i in label_set_str.split('#') if i}))
    return annotation_set


def write_annotation_file(annotation_set, out_filename):
    """Write set of Annotation to csv file with lines in format
        start_time,end_time,label_0#...#label_N
    """
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    with open(out_filename, 'w') as out_file:
        for ann in sorted(list(annotation_set)):
            label_set_str = '#'.join(str(i) for i in ann.label_set)
            out_file.write(f'{ann.start_time:.3f},{ann.end_time:.3f},{label_set_str}\n')


def set_annotation_labels(from_annotation_set, to_annotation_set, overlap_ratio=0.5):
    """
    Map label from from_annotation_set to to_annotation_set such as a given annotation ann
    in to_annotation_set get all labels with overlap in from_annotation_set, only if this
    overlap is >= <ann duration> * overlap_ratio.
    Args:
        from_annotation_set: set of Annotation objects
        to_annotation_set: set of Annotation objects
        overlap_ratio (float in [0, 1]): min overlap ratio
    """

    to_annotation_set_new = set()

    for to_ann in to_annotation_set:
        to_annotation_set_new.add(Annotation(
            to_ann.start_time,
            to_ann.end_time,
            get_labels_in_range(
                from_annotation_set,
                to_ann.start_time,
                to_ann.end_time,
                overlap_ratio)
        ))
        
    return to_annotation_set_new


def map_annotation_set(ann_set, mapping):
    """Map annotation label set to new label set,
    according to mapping. Unmapped label are removed.

    Args:
        ann_set: annotation set
        mapping: dictionary with key = label_id and value = new_label_id
    """
    
    new_ann_set = set()

    for ann in ann_set:
            new_label_set = frozenset([mapping[l] for l in ann.label_set if l in mapping.keys()])
            if new_label_set:
                new_ann_set.add(Annotation(ann.start_time, ann.end_time, new_label_set))

    return new_ann_set


def get_labels_in_range(annotation_set, start_time, end_time, overlap_ratio=0.5):
    """"Get all labels from an annotation set in a time range.
    If the annotation is a timestamp, i.e. ann.start_time == ann.end_time,
    the overlap_ratio is not used. If the timestamped annotation is
    in the time range, get it.
    """
    label_set = set()
    for ann in annotation_set:
        if ann.start_time == ann.end_time:
            if ann.start_time > start_time and ann.start_time <= end_time:
                label_set.update(ann.label_set)
            continue
        overlap = _get_overlap(ann.start_time,
                               ann.end_time if ann.end_time != -1 else end_time,
                               start_time,
                               end_time)
        if overlap >= (end_time - start_time) * overlap_ratio:
            label_set.update(ann.label_set)
    return label_set


def _get_overlap(start1, end1, start2, end2):
    """Get overlap between the intervals [start1, end1] and [start2, end2]."""
    return max(0, min(end1, end2) - max(start1, start2))


def _set_end_time_when_missing(annotation_set, file_duration):
    # -1 end time means end of file
    # because it can mess up some computation, we replace it by file duration
    return set([Annotation(a.start_time, a.end_time if a.end_time > -1 else file_duration, a.label_set) for a in annotation_set])


def _make_subsets_of_overlapping_annotations(annotation_set):
    ann_subsets = []
    for ann in annotation_set:
        ind = _get_overlapping_annotation_subset(ann, ann_subsets)
        if ind < 0:
            ann_subsets.append([ann])
        else:
            ann_subsets[ind].append(ann)
    return ann_subsets


def _get_overlapping_annotation_subset(annotation, annotation_subsets):

    for i, ann_subset in enumerate(annotation_subsets):
        start_time = min([ann.start_time for ann in ann_subset])
        end_time = max([ann.end_time for ann in ann_subset])
        if _get_overlap(start_time, end_time, annotation.start_time, annotation.end_time) > 0:
            return i

    return -1
