import tensorflow as tf

from .layers.embedding_block import EmbeddingBlock
from .layers.radial_bases import RBFLayer
from .layers.angle_bases import ABFLayer
from .layers.interaction_block import InteractionBlock
from .layers.output_block import OutputBlock
from .layers.identity_block import IdentityBlock


class DLMPNN(tf.keras.Model):

    def __init__(self, emb_size, num_blocks, num_radial, cutoff, activation='swish', name='DLMPNN', **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_blocks = num_blocks

        def scale_init(shape, dtype):
            return tf.exp(tf.random.normal([shape], mean=0.0, stddev=0.01))
        self.coef_mp = self.add_weight(name="final.coef.mp", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)
        self.coef_sg = self.add_weight(name="final.coef.sg", shape=1, dtype=tf.float32, initializer=scale_init, trainable=True)

        self.rbf_pair_layer = RBFLayer(num_radial, cutoff=cutoff)
        self.abf_triple_layer = ABFLayer()

        self.output_blocks = []
        self.int_blocks = []
        self.sbody_blocks = []
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)

        self.output_blocks.append(OutputBlock(0, emb_size, activation=activation, output_init='zeros'))
        self.sbody_blocks.append(IdentityBlock(0, emb_size, activation=activation))


        for i in range(num_blocks):
            self.int_blocks.append(InteractionBlock(i+1, emb_size, activation=activation))
            self.output_blocks.append(OutputBlock(i+1, emb_size, activation=activation, output_init='GlorotNormal'))
            self.sbody_blocks.append(IdentityBlock(i+1, emb_size, activation=activation))


    def call(self, inputs):
        Z, R                         = inputs['Z'], inputs['R']
        batch_seg                    = inputs['batch_seg']
        idx_i, idx_j                 = inputs['idx_i'], inputs['idx_j']
        idx_kj, idx_ji               = inputs['idx_kj'], inputs['idx_ji']
        cosine_ijk                   = inputs['cosine_ijk']

        n_atoms = tf.shape(Z)[0]

        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        rbf = self.rbf_pair_layer(Ri, Rj)
        abf = self.abf_triple_layer(cosine_ijk)

        x, x_0 = self.emb_block([Z, rbf, idx_i, idx_j])

        res_output = self.output_blocks[0]([x, rbf, idx_i, n_atoms])
        res_single = self.sbody_blocks[0](x_0)

        last_x = x

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([last_x, rbf, idx_kj, idx_ji, abf])
            o = self.output_blocks[i+1]([x, rbf, idx_i, n_atoms])
            res_output += o
            res_single = res_single + self.sbody_blocks[i+1](res_single)
            last_x = x

        res_output = tf.math.segment_sum(res_output, batch_seg)
        x_identity = tf.math.segment_sum(res_single, batch_seg)
        
        y_hat = self.coef_mp*res_output + self.coef_sg*x_identity

        return y_hat
