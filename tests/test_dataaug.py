import pytest
import numpy as np
import tensorflow as tf

from phaunos_ml.utils.dataug_utils import Mixup

tf.enable_eager_execution()


BATCH_SIZE = 2


class TestMixup:

    """Mix training and augmentation data and combine labels"""

    @pytest.fixture(scope="class")
    def train_channel_first(self):
        dataset = tf.data.Dataset.from_tensor_slices((
                tf.cast([[[[1,2]]],[[[3,4]]],[[[5,6]]],[[[7,8]]],[[[9,10]]]], tf.float32),
                tf.cast([0,0,0,1,1], tf.float32)))
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        return dataset

    @pytest.fixture(scope="class")
    def aug_channel_first(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast([[[[10,20]]],[[[30,40]]],[[[50,60]]]], tf.float32),
            tf.cast([0,1,0], tf.float32)))
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        return dataset

    @pytest.fixture(scope="class")
    def train_channel_last(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast([[[[1],[2]]],[[[3],[4]]],[[[5],[6]]],[[[7],[8]]],[[[9],[10]]]], tf.float32),
            tf.cast([0,0,0,1,1], tf.float32)))
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        return dataset

    @pytest.fixture(scope="class")
    def aug_channel_last(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast([[[[10],[20]]],[[[30],[40]]],[[[50],[60]]]], tf.float32),
            tf.cast([0,1,0], tf.float32)))
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        return dataset
    
    @pytest.fixture(scope="class")
    def mixup(self):
        return Mixup(w_min=0.5) # So that mixing coefficient is always 0.5

    def test_mixup_channel_first(self, mixup, train_channel_first, aug_channel_first):

        dataset = tf.data.Dataset.zip((train_channel_first, aug_channel_first))
        dataset = dataset.map(lambda dataset1, dataset2: mixup.process(
            dataset1[0], dataset1[1], dataset2[0], dataset2[1], BATCH_SIZE))

        data = []
        count = 0
        for d in dataset:
            data.append(d)
            if count == 3:
                break
            count += 1

        assert np.array_equal(data[0][0], [[[[5.5,11]]],[[[16.5, 22]]]])
        assert np.array_equal(data[0][1], [0,1])

        assert np.array_equal(data[1][0], [[[[27.5,33]]],[[[8.5, 14]]]])
        assert np.array_equal(data[1][1], [0,1])

        assert np.array_equal(data[2][0], [[[[19.5,25]]],[[[25.5, 31]]]])
        assert np.array_equal(data[2][1], [1,0])


    def test_mixup_channel_last(self, mixup, train_channel_last, aug_channel_last):

        dataset = tf.data.Dataset.zip((train_channel_last, aug_channel_last))
        dataset = dataset.map(lambda dataset1, dataset2: mixup.process(
            dataset1[0], dataset1[1], dataset2[0], dataset2[1], BATCH_SIZE))

        data = []
        count = 0
        for d in dataset:
            data.append(d)
            if count == 3:
                break
            count += 1

        assert np.array_equal(data[0][0], [[[[5.5],[11]]],[[[16.5], [22]]]])
        assert np.array_equal(data[0][1], [0,1])

        assert np.array_equal(data[1][0], [[[[27.5],[33]]],[[[8.5], [14]]]])
        assert np.array_equal(data[1][1], [0,1])

        assert np.array_equal(data[2][0], [[[[19.5],[25]]],[[[25.5], [31]]]])
        assert np.array_equal(data[2][1], [1,0])
