import pytest
import numpy as np
import tensorflow as tf

from phaunos_ml.utils.dataug_utils import Mixup


BATCH_SIZE = 2
N_TRAIN = 5
N_AUG = 3
C = 3


class TestMixup:

    """Mix training and augmentation data (both in format NCHW)
    and combine labels.
    """

    @pytest.fixture(scope="class")
    def train_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            np.random.rand(N_TRAIN,C,20,10),
            np.random.randint(0,2,[N_TRAIN,2])))
        dataset = dataset.repeat(count=N_AUG)
        dataset = dataset.batch(BATCH_SIZE)
        return dataset

    @pytest.fixture(scope="class")
    def aug_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            np.random.rand(N_AUG,C,20,10),
            np.random.randint(0,2,[N_AUG,2])))
        dataset = dataset.repeat(count=N_TRAIN)
        dataset = dataset.batch(BATCH_SIZE)
        return dataset
    
    @pytest.fixture(scope="class")
    def mixup(self):
        return Mixup(min_weight=0.5, max_weight=0.5) # So that mixing coefficient is always 0.5

    def test_mixup(self, mixup, train_dataset, aug_dataset):

        dataset = tf.data.Dataset.zip((train_dataset, aug_dataset))
        dataset = dataset.map(lambda dataset1, dataset2: mixup.process(
            dataset1[0], dataset1[1], dataset2[0], dataset2[1]))

        train_batches = [d for d in train_dataset]
        aug_batches = [d for d in aug_dataset]
        batches = [d for d in dataset]
        for i in range(len(batches)):
            assert np.array_equal(
                batches[i][0],
                (train_batches[i][0] + aug_batches[i][0]) * 0.5)
            assert np.array_equal(
                batches[i][1],
                np.logical_or(train_batches[i][1], aug_batches[i][1]))
