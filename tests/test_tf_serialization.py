import pytest
import numpy as np
import tensorflow as tf

from phaunos_ml.utils.tf_serialization_utils import labelstr2onehot


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
