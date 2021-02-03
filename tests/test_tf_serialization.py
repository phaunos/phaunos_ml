import pytest
import numpy as np
import tensorflow as tf

from phaunos_ml.utils.tf_serialization_utils import labelstr2onehot,\
    serialize_data,\
    serialized2example,\
    serialized2data


class TestTFSerialization:


    def test_labelstr2onehot(self):

        assert np.array_equal(
            labelstr2onehot('', [1,2,3,4,5]).numpy(),
            np.array([0,0,0,0,0]))

        assert np.array_equal(
            labelstr2onehot('2', [1,2,3,4,5]).numpy(),
            np.array([0,1,0,0,0]))

        assert np.array_equal(
            labelstr2onehot('6', [1,2,3,4,5]).numpy(),
            np.array([0,0,0,0,0]))

        assert np.array_equal(
            labelstr2onehot('1#2#5', [1,2,3,4,5]).numpy(),
            np.array([1,1,0,0,1]))
        
        assert np.array_equal(
            labelstr2onehot('1#2#5#6', [1,2,3,4,5]).numpy(),
            np.array([1,1,0,0,1]))
        
        assert np.array_equal(
            labelstr2onehot('6#7#8', [1,2,3,4,5]).numpy(),
            np.array([0,0,0,0,0]))

    def test_binarylabel(self):

        class_list = np.array([12, 34])
        class_list_wrong = np.array([12, 34, 56])

        serialized1 = serialize_data('abc.wav', 1, 2, np.random.rand(3,4,5), [12])
        serialized2 = serialize_data('def.wav', 3, 4, np.random.rand(3,4,5), [34])
        serialized3 = serialize_data('ghi.wav', 5, 5, np.random.rand(3,4,5), [1])

        data1 = serialized2data(serialized1, class_list, one_hot_label=False)
        assert data1[1] == 0

        data2 = serialized2data(serialized2, class_list, one_hot_label=False)
        assert data2[1] == 1

        with pytest.raises(tf.errors.InvalidArgumentError):
            data3 = serialized2data(serialized3, class_list, one_hot_label=False)
        
        with pytest.raises(tf.errors.InvalidArgumentError):
            data1 = serialized2data(serialized1, class_list_wrong, one_hot_label=False)
