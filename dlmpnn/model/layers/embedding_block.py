import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..utils import GlorotOrthogonal


class EmbeddingBlock(layers.Layer):
    def __init__(self, emb_size, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.weight_go_init = GlorotOrthogonal()
        self.weight_rbf_init = tf.initializers.GlorotNormal()

        # Atom embeddings: We go up to Pu (94). Use 95 dimensions because of 0-based indexing
        emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))
        self.embeddings = self.add_weight(name="embeddings", shape=(95, self.emb_size),
                                          dtype=tf.float32, initializer=emb_init, trainable=True)

        self.dense_rbf = layers.Dense(name="embed.rbf", units=self.emb_size,
                                           activation=activation, use_bias=True,
                                          kernel_initializer=self.weight_rbf_init)

        self.dense = layers.Dense(name="embed.dense", units=self.emb_size,
                                  activation=activation, use_bias=True,
                                  kernel_initializer=self.weight_rbf_init) # activation = activation

    def call(self, inputs):

        Z, rbf, idx_i, idx_j = inputs

        rbf = self.dense_rbf(rbf)

        Z_i = tf.gather(Z, idx_i)
        Z_j = tf.gather(Z, idx_j)

        x_i = tf.gather(self.embeddings, Z_i)
        x_j = tf.gather(self.embeddings, Z_j)

        x_0 = tf.gather(self.embeddings, Z)

        x = tf.concat([x_i, x_j, rbf], axis=-1)
        x = self.dense(x) # activation changed

        return x, x_0
