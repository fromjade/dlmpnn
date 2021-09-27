import tensorflow as tf
from tensorflow.keras import layers
from scipy.special import eval_legendre

class ABFLayer(layers.Layer):
    def __init__(self, name='Legendre', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, -1)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_legendre.html
        N = range(12)
        res = eval_legendre(N, inputs)
        return res