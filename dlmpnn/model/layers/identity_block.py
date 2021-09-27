import tensorflow as tf
from tensorflow.keras import layers

from ..utils import GlorotOrthogonal


class IdentityBlock(layers.Layer):
    def __init__(self, layer_index, emb_size,
                 activation=None, output_init='zeros', name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight_go_init = GlorotOrthogonal()
        self.emb_size = emb_size
        self.weight_rbf_init = tf.initializers.GlorotNormal()
        if output_init == 'GlorotOrthogonal':
            output_init = GlorotOrthogonal()

        def scale_init(shape, dtype):
            return tf.exp(tf.random.normal([shape], mean=0.0, stddev=0.01))

        self.dense1 = layers.Dense(name="iden"+str(layer_index)+".dense1", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_go_init)
        self.dense2 = layers.Dense(name="iden"+str(layer_index)+".dense2", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_go_init)
        self.dense3 = layers.Dense(name="iden"+str(layer_index)+".dense3", units=emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_rbf_init)
        self.dense4 = layers.Dense(name="iden"+str(layer_index)+".dense4", units=emb_size, activation=activation, use_bias=True,
                                   kernel_initializer=self.weight_rbf_init)
        self.dense5 = layers.Dense(name="iden"+str(layer_index)+".dense5", units=1, use_bias=True, kernel_initializer=output_init) # use_bias=False

        self.coef_x = self.add_weight(name="iden"+str(layer_index)+".coef.x", shape=1, dtype=tf.float32, initializer=scale_init,
                                        trainable=True)


    def call(self, inputs):

        # Z: atom type / R: coordinates (x, y, z) /
        x_0 = inputs

        x_1 = self.dense1(x_0)
        x_2 = self.dense2(x_1)
        x_3 = self.dense3(x_2) + x_1
        x_4 = self.dense4(x_3) + x_2

        x = self.dense5(x_4) * self.coef_x

        return x




