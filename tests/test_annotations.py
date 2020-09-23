import pytest
import os
import filecmp

from phaunos_ml.utils.annotation_utils import (
    Annotation,
    read_annotation_file,
    write_annotation_file,
    set_annotation_labels,
    get_labels_in_range
)


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
ANNOTATION_FILE_1 = os.path.join(DATA_PATH, 'file_1.ann')
TMP_ANNOTATION_FILE_1 = ANNOTATION_FILE_1 + '.tmp'
ANNOTATION_FILE_2 = os.path.join(DATA_PATH, 'file_2.ann')
TMP_ANNOTATION_FILE_2 = ANNOTATION_FILE_2 + '.tmp'


class TestAnnotations:

    @pytest.fixture(scope="class")
    def annotation_set_1(self):
        annotation_set = set()
        annotation_set.add(Annotation(1, 3.2, set([5])))
        annotation_set.add(Annotation(2, 4.2, set([2])))
        annotation_set.add(Annotation(4, 5, set([3])))
        annotation_set.add(Annotation(6, 6.3, set([1,2])))
        return annotation_set


    @pytest.fixture(scope="class")
    def annotation_set_2(self):
        annotation_set = set()
        annotation_set.add(Annotation(2, 3))
        annotation_set.add(Annotation(4, 4.8))
        annotation_set.add(Annotation(6, 6.2))
        annotation_set.add(Annotation(6.5, 7))
        return annotation_set
    
    
    @pytest.fixture(scope="class")
    def annotation_set_1to2(self):
        annotation_set = set()
        annotation_set.add(Annotation(2, 3, set([2,5])))
        annotation_set.add(Annotation(4, 4.8, set([3])))
        annotation_set.add(Annotation(6, 6.2, set([1,2])))
        annotation_set.add(Annotation(6.5, 7))
        return annotation_set


    def test_write_annotation_file(self, annotation_set_1, annotation_set_2):

        write_annotation_file(annotation_set_1, TMP_ANNOTATION_FILE_1)
        assert filecmp.cmp(ANNOTATION_FILE_1, TMP_ANNOTATION_FILE_1)
        os.remove(TMP_ANNOTATION_FILE_1)
        
        write_annotation_file(annotation_set_2, TMP_ANNOTATION_FILE_2)
        assert filecmp.cmp(ANNOTATION_FILE_2, TMP_ANNOTATION_FILE_2)
        os.remove(TMP_ANNOTATION_FILE_2)


    def test_read_annotation_file(self, annotation_set_1, annotation_set_2):

        assert annotation_set_1 == read_annotation_file(ANNOTATION_FILE_1)
        assert annotation_set_2 == read_annotation_file(ANNOTATION_FILE_2)


    def test_set_annotation_file(self, annotation_set_1, annotation_set_2, annotation_set_1to2):

        assert set_annotation_labels(annotation_set_1, annotation_set_2) == annotation_set_1to2


    def test_timestamped_annotation(self):

        annotation_set = set()
        annotation_set.add(Annotation(2, 2, set([2,5])))
        annotation_set.add(Annotation(4.8, 4.8, set([3])))

        assert get_labels_in_range(annotation_set, 0, 2) == set([2, 5])
        assert not get_labels_in_range(annotation_set, 2, 4)
        assert get_labels_in_range(annotation_set, 4, 6) == set([3])
