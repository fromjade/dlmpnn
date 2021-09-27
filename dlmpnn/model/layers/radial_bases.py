import tensorflow as tf
from tensorflow.keras import layers


class RBFLayer(layers.Layer):
    def __init__(self, num_radial, cutoff, name='rbf', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_radial = num_radial
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)

        def scale_init(shape, dtype):
            return tf.random.normal([shape], mean=3.5, stddev=0.2)
        self.rad_weight = self.add_weight(name="a1", shape=self.num_radial, dtype=tf.float32, initializer=scale_init, trainable=True)

    def legendre_rational(self, x, R, t):
         return ((2*t-1)/t) * ((x - 1) / (x + 1)) * R[-1] - ((t-1) / t) * R[-2]


    def call(self, inputs):

        Ri, Rj = inputs
        Dij = tf.sqrt(tf.reduce_sum(tf.math.square(Ri - Rj), -1))

        d_scaled = Dij * self.inv_cutoff
        d_scaled = tf.expand_dims(d_scaled, -1)

        x = d_scaled

        R1 = (x - 1) / (x + 1)
        R2 = (3 / 2) * ((x - 1) / (x + 1)) * R1 - (1 / 2) * 1

        R = [R1, R2]

        for i in range(3, self.num_radial + 1):
            R_i = self.legendre_rational(x, R, i)
            R.append(R_i)

        R = tf.concat(R, axis=-1)

        R = self.rad_weight * R

        return R

