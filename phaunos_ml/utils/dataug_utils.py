import tensorflow as tf


class Mixup:

    """Augmentation technique based on

    Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization."
    arXiv preprint arXiv:1710.09412 (2017).

    Here we add options to
        - use Uniform distribution instead of the Beta distribution proposed
        in the paper,
        - and to combine one-hot labels using logical OR instead of weighting
    """

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




        
