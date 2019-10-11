import tensorflow as tf
from tensorflow.contrib.image import sparse_image_warp


#########
# Mixup #
#########

"""Augmentation technique based on

Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization."
arXiv preprint arXiv:1710.09412 (2017).

Here we add options to
    - use Uniform distribution instead of the Beta distribution proposed
    in the paper,
    - and to combine one-hot labels using logical OR instead of weighting
"""

class Mixup:


    def __init__(self, w_min=0.2):
        """
        Args:
            w_min: minimum weight, in [0,1]
        """

        if w_min < 0 or w_min > 1:
            raise ValueError('w_min must be in [0,1]')
        self.dist = tf.distributions.Uniform(low=w_min, high=1-w_min)

    def process(self, batch1, label1, batch2, label2, batch_size):

        w = tf.cast(self.dist.sample(sample_shape=batch_size), batch1.dtype)

        # broadcast w to batch shape
        w = tf.expand_dims(w, -1)
        w = tf.expand_dims(w, -1)
        w = tf.expand_dims(w, -1)
        w = tf.broadcast_to(w, tf.shape(batch1))

        # weighted sum of the features
        batch = tf.multiply(w,batch1) + tf.multiply(1-w,batch2)

        # combined labels
        label = tf.logical_or(tf.cast(label1, tf.bool), tf.cast(label2, tf.bool))

        return batch, tf.cast(label, tf.int8)


###############
# SpecAugment #
###############

"""
Audio data augmentation based on time warping
and time and frequency masking.

Park, Daniel S., et al.
"Specaugment: A simple data augmentation method for automatic speech recognition."
arXiv preprint arXiv:1904.08779 (2019).
"""

def time_warp(data, w=80):
    """Pick a random point along the time axis between 
    w and n_time_bins-w and warp by a distance
    between [0,w] towards the left or the right.

    Args:
        data: batch of spectrogram. Shape [batch_size, n_freq_bins, n_time_bins, 1]
    """

    _, n_freq_bins, n_time_bins, _ = tf.shape(data)

    # pick a random point along the time axis in [w,n_time_bins-w]
    t = tf.random.uniform(
        shape=(),
        minval=w,
        maxval=n_time_bins-w,
        dtype=tf.int32)

    # pick a random translation vector in [-w,w] along the time axis
    tv = tf.cast(
        tf.random.uniform(shape=(), minval=-w, maxval=w, dtype=tf.int32),
        tf.float32)


    # set control points y-coordinates
    ctl_pt_freqs = tf.convert_to_tensor([
        0.0,
        tf.cast(n_freq_bins, tf.float32) / 2.0,
        tf.cast(n_freq_bins-1, tf.float32)])

    # set source control point x-coordinates
    ctl_pt_times_src = tf.convert_to_tensor([t, t, t], dtype=tf.float32)

    # set destination control points
    ctl_pt_times_dst = ctl_pt_times_src + tv
    ctl_pt_src = tf.expand_dims(tf.stack([ctl_pt_freqs, ctl_pt_times_src], axis=-1), 0)
    ctl_pt_dst = tf.expand_dims(tf.stack([ctl_pt_freqs, ctl_pt_times_dst], axis=-1), 0)

    return sparse_image_warp(data, ctl_pt_src, ctl_pt_dst, num_boundary_points=1)[0]
