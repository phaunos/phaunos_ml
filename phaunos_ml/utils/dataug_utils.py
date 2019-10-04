import tensorflow as tf


class Mixup:

    def __init__(self, w_min=0.2):
        """
        Args:
            w_min: minimum weight
        """
        self.dist = tf.distributions.Uniform(low=w_min, high=1-w_min)

    def process(self, batch1, label1, batch2, label2, batch_size):

        w = tf.cast(self.dist.sample(sample_shape=batch_size), batch1.dtype)

        # broadcast w to batch shape
        w = tf.expand_dims(w, -1)
        w = tf.expand_dims(w, -1)
        w = tf.expand_dims(w, -1)
        w = tf.broadcast_to(w, tf.shape(batch1))
        print(w)
        print(batch1)
        print(batch2)

        # weighted sum of the features
        batch = tf.multiply(w,batch1) + tf.multiply(1-w,batch2)

        # combined labels
        label = tf.logical_or(tf.cast(label1, tf.bool), tf.cast(label2, tf.bool))

        return batch, label




        
