from keras.layers import Layer, multiply
from layers.SubLayers import PositionWiseFeedForward, MultiHeadAttention


class EncoderLayer(Layer):
    """Encoder Layer(based on implementation from Yu-Hsiang Huang):

    :param d_model: input dimension from model.
    :param d_inner: dimension from out of Encoder Layer.
    :param n_head: number of attention heads of queries, keys and values.
    :param d_k: dimension used for each queries and keys from head.

    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout_rate)
        self.pos_feedforward = PositionWiseFeedForward(d_model, d_inner, dropout_rate)

    def compute_output_shape(self, input_shape):
        return [input_shape, self.self_attention.compute_output_shape([input_shape]*3)[-1]]

    def call(self, inputs, non_pad_mask=None, self_attention_mask=None, **kwargs):
        enc_output, enc_self_attention = self.self_attention([inputs] * 3, mask=self_attention_mask)
        enc_output = multiply([enc_output, non_pad_mask])

        enc_output = self.pos_feedforward(enc_output)
        enc_output = multiply([enc_output, non_pad_mask])

        return [enc_output, enc_self_attention]


class DecoderLayer(Layer):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout_rate=0.1):
        self.self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout_rate)
        self.enc_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout_rate)
        self.pos_feedforward = PositionWiseFeedForward(d_model, d_inner, dropout_rate)
        super(DecoderLayer, self).__init__()

    def compute_output_shape(self, input_shape):
        dec_shape = input_shape[0]
        enc_shape = input_shape[-1]
        return [dec_shape, self.self_attention.compute_output_shape([dec_shape[0]] * 3),
                self.enc_attention.compute_output_shape([dec_shape, enc_shape, enc_shape])]

    def call(self, inputs, non_pad_mask=None, self_atten_mask=None, dec_enc_atten_mask=None, **kwargs):
        dec_input, enc_input = inputs

        dec_output, self_attention = self.self_attention([dec_input] * 3, mask=self_atten_mask)
        dec_output = multiply([dec_output, non_pad_mask])

        dec_output, dec_enc_attention = self.enc_attention([dec_output, enc_input, enc_input], mask=dec_enc_atten_mask)
        dec_output = multiply([dec_output, non_pad_mask])

        dec_output = self.pos_feedforward(dec_output)
        dec_output = multiply([dec_output, non_pad_mask])

        return [dec_output, self_attention, dec_enc_attention]

