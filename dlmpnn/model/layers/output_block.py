import tensorflow as tf
from tensorflow.keras import layers

from ..utils import GlorotOrthogonal


class OutputBlock(layers.Layer):
    def __init__(self, layer_index, emb_size, activation, **kwargs):
        super().__init__(name='OutputBlock', **kwargs)
        self.weight_go_init = GlorotOrthogonal()
        self.emb_size = emb_size
        self.weight_rbf_init = tf.initializers.GlorotNormal()
        output_init = GlorotOrthogonal()

        def scale_init(shape, dtype):
            return tf.exp(tf.random.normal([shape], mean=0.0, stddev=0.01))

        self.coef_rbf_a = self.add_weight(name="output"+str(layer_index)+".coef.rbf", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)
        self.coef_rbf_b = self.add_weight(name="output"+str(layer_index)+".coef.rbf", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)

        self.coef_x = self.add_weight(name="output"+str(layer_index)+".coef.x", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)
        self.coef_final = self.add_weight(name="output"+str(layer_index)+".coef.final", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)

        self.dense_rbf = layers.Dense(name="output"+str(layer_index)+".dense.rbf", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_rbf_init)
        self.dense_x = layers.Dense(name="output"+str(layer_index)+".dense.x", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_rbf_init)

        self.dense1 = layers.Dense(name="output"+str(layer_index)+".dense.1", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_rbf_init)
        self.dense2 = layers.Dense(name="output" + str(layer_index) + ".dense.2", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_rbf_init)
        self.dense3 = layers.Dense(name="output" + str(layer_index) + ".dense.3", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_rbf_init)
        self.dense_final = layers.Dense(name="output"+str(layer_index)+".dense_final", units=1, use_bias=False, kernel_initializer=output_init)

    def call(self, inputs):

        x, rbf, idnb_i, n_atoms = inputs

        rbf = self.coef_rbf_a * rbf

        x = self.dense_x(x)
        rbf = self.dense_rbf(rbf)

        x = self.coef_x * x
        rbf = self.coef_rbf_b * rbf

        conv_x = rbf * x

        x = tf.math.unsorted_segment_sum(conv_x, idnb_i, n_atoms)

        x = self.dense3(self.dense2(self.dense1(x)))
        x = self.dense_final(x) * self.coef_final

        return x
