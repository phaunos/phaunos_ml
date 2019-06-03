import os


ANN_EXT = '.ann'


class PhaunosAnnotationError(Exception):
    pass


class Annotation:

    def __init__(self, start_time=0, end_time=-1, label_set=frozenset()):
        if start_time < 0 or (end_time != -1 and end_time <= start_time):
            raise PhaunosAnnotationError(
                "Wrong time parameters: start_time must be " +
                "greater than or equal to 0 and end_time must be greater than start_time " +
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
        start_time_str, end_time_str, label_set_str = line.strip().split(',')
        annotation_set.add(Annotation(float(start_time_str), float(end_time_str), {int(i) for i in label_set_str.split('#') if i}))
    return annotation_set


def write_annotation_file(annotation_set, out_filename):

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

def get_labels_in_range(annotation_set, start_time, end_time, overlap_ratio=0.5):
    label_set = set()
    for ann in annotation_set:
        overlap = _get_overlap(ann.start_time,
                               ann.end_time if ann.end_time != -1 else end_time,
                               start_time,
                               end_time)
        if overlap >= (end_time - start_time) * overlap_ratio:
            label_set.update(ann.label_set)
    return label_set


def _get_overlap(start1, end1, start2, end2):
    """Get overlap between the intervals [start1, end1] and [start2, end2].
    Args:
        start1 (float)
        end1 (float)
        start2 (float)
        end2 (float)
    """
    return max(0, min(end1, end2) - max(start1, start2))
