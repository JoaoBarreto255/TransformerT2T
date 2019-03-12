"""Based from implementation from Yu-Hsiang Huang.
This library contains
Contains methods:
    + get_non_pad_mask
    + get_attn_key_pad_mask
    + get_subsequent_mask

Contains Layers Class:
    + MakeSubsequentMask
    + MakeAttnKeyPadMask
    + MakeNonPadMask

Contains Initializer Class:
    + SinusoidInitializer
"""

import numpy as np
from keras import initializers as init
import keras.backend as K
from keras.layers import Layer


def get_non_pad_mask(seq, pad_const=0):
    return MakeNonPadMask(pad_const)(seq)


def get_attn_key_pad_mask(seq_k, seq_q, pad_const=0):
    return MakeAttnKeyPadMask(pad_const)([seq_k, seq_q])


def get_subsequent_mask(seq):
    return MakeSubsequentMask()(seq)


class MakeNonPadMask(Layer):

    def __init__(self, pad_const=0):
        super(MakeNonPadMask, self).__init__()
        self.pad_const = pad_const

    def compute_output_shape(self, input_shape):
        return input_shape + (1,)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert isinstance(input_shape, tuple)
        super(MakeNonPadMask, self).build()

    def call(self, inputs, **kwargs):
        mask = K.eval(K.not_equal(inputs, self.pad_const)) * 1.0
        mask = K.variable(mask, dtype=K.floatx())
        mask = K.expand_dims(mask)
        return mask


class MakeAttnKeyPadMask(Layer):

    def __init__(self, pad_const=0):
        super(MakeAttnKeyPadMask, self).__init__()
        self.pad_const = pad_const
        self.len_q = 0

    def compute_output_shape(self, input_shape):
        bc_sz, len_k = input_shape[0]
        _, len_q = input_shape[1]
        return bc_sz, len_q, len_k

    def build(self, input_shape):
        self.len_q = input_shape[-1][-1]
        super(MakeAttnKeyPadMask, self).build()

    def call(self, inputs, **kwargs):
        len_q = self.len_q
        padding_mask = K.expand_dims(K.equal(inputs[0], self.pad_const), 1)
        padding_mask = K.repeat_elements(padding_mask, len_q, 1)
        padding_mask = K.variable(K.eval(padding_mask) * 1.0, K.floatx())
        return padding_mask


class MakeSubsequentMask(Layer):

    def __init__(self):
        self.len_s = 0
        self.sz_b = 0
        super(MakeSubsequentMask, self).__init__()

    def build(self, input_shape):
        self.len_s, self.sz_b = input_shape
        super(MakeSubsequentMask, self).build()

    def compute_output_shape(self, input_shape):
        sz_b, len_s = input_shape
        return sz_b, len_s, len_s

    def call(self, inputs, **kwargs):
        sz_b, len_s = self.sz_b, self.len_s
        my_ones = np.triu(np.ones((len_s, len_s), dtype=np.uint8), k=1)
        result = K.variable(my_ones)
        return K.variable(K.repeat_elements(K.expand_dims(result, 0), sz_b, 0))


class SinusoidInitializer(init.Initializer):

    def __init__(self, n_position=None, d_hid=None, padding_index=None):
        self.n_position = n_position
        self.hidden_dim = d_hid,
        self.padding_index = padding_index

    def __call__(self, shape, **kwargs):
        self.n_position, self.hidden_dim = shape
        n_position, hidden_dim = shape
        padding_index = self.padding_index

        def cal_angle(pos, hid_idx):
            return pos / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        if padding_index is not None:
            sinusoid_table[padding_index] = 0

        return sinusoid_table

    def get_config(self):
        return {
            "number_of_position": self.n_position,
            "hidden_dimension": self.hidden_dim,
            "padding_index": self.padding_index
        }
