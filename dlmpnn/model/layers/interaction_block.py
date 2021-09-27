import tensorflow as tf
from tensorflow.keras import layers
from ..utils import GlorotOrthogonal


class InteractionBlock(layers.Layer):
    def __init__(self, layer_index, emb_size,
                 activation=None, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.weight_go_init = GlorotOrthogonal()
        self.weight_rbf_init = tf.initializers.GlorotNormal()

        def scale_init(shape, dtype):
            return tf.exp(tf.random.normal([shape], mean=0.0, stddev=0.01))

        # Transformations of Bessel and spherical basis representations
        self.dense_ji = layers.Dense(name="inter"+str(layer_index)+".dense.ji", units=self.emb_size, activation=activation, use_bias=True,
                                     kernel_initializer=self.weight_rbf_init)
        self.dense_kj = layers.Dense(name="inter"+str(layer_index)+".dense.kj", units=self.emb_size, activation=activation, use_bias=True,
                                     kernel_initializer=self.weight_rbf_init)
        self.dense_rbf = layers.Dense(name="inter"+str(layer_index)+".rbf", units=emb_size, activation=activation, use_bias=True,
                                      kernel_initializer=self.weight_rbf_init)
        self.dense_cos = layers.Dense(name="inter"+str(layer_index)+".cos", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_go_init)

        # Dense transformations of input messages
        self.coef_rbf_a = self.add_weight(name="inter"+str(layer_index)+".a.coef.rbf", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)
        self.coef_cos_a = self.add_weight(name="inter"+str(layer_index)+".a.coef.cos", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)

        self.coef_rbf_b = self.add_weight(name="inter"+str(layer_index)+".b.coef.rbf", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)
        self.coef_cos_b = self.add_weight(name="inter"+str(layer_index)+".b.coef.cos", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)

        self.coef_x = self.add_weight(name="inter"+str(layer_index)+".coef.x", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)
        self.coef_final = self.add_weight(name="inter"+str(layer_index)+".coef.final", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)

        # Residual layers before skip connection

        self.dense1 = layers.Dense(name="inter"+str(layer_index)+".dense.1", units=emb_size, activation=activation)
        self.dense2 = layers.Dense(name="inter"+str(layer_index)+".dense.2", units=emb_size, activation=activation)
        self.dense3 = layers.Dense(name="inter"+str(layer_index)+".dense.3", units=emb_size, activation=activation)
        self.dense4 = layers.Dense(name="inter"+str(layer_index)+".dense.4", units=emb_size, activation=activation)
        self.dense5 = layers.Dense(name="inter"+str(layer_index)+".dense.5", units=emb_size, activation=activation)
        self.dense6 = layers.Dense(name="inter"+str(layer_index)+".dense.6", units=emb_size, activation=activation)
        self.dense7 = layers.Dense(name="inter"+str(layer_index)+".dense.7", units=emb_size, activation=activation)


    def call(self, inputs):

        x, rbf, id_expand_kj, id_reduce_ji, cos_ijk = inputs

        num_interactions = tf.shape(x)[0]

        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x)
        x_kj = tf.gather(x_kj, id_expand_kj)

        rbf = self.dense_rbf(self.coef_rbf_a * rbf) * self.coef_rbf_b
        cos_ijk = self.dense_cos(self.coef_cos_a * cos_ijk) * self.coef_cos_b

        x = self.coef_x * x

        filter = tf.gather(rbf, id_expand_kj) + cos_ijk
        conv_x = filter * x_kj
        conv_x = tf.math.unsorted_segment_sum(conv_x, id_reduce_ji, num_interactions)
        x2 = x_ji + conv_x

        x2 = self.dense2(self.dense1(x2)) + x2
        x = self.dense3(x2) + x
        x = self.dense5(self.dense4(x)) + x
        x = self.dense7(self.dense6(x)) + x

        x = x * self.coef_final

        return x
