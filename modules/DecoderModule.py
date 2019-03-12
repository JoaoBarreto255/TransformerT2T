"""
Decoder Module Class
"""
from layers import DecoderLayer, SinusoidInitializer, get_subsequent_mask, \
    get_non_pad_mask, get_attn_key_pad_mask

from keras.layers import Embedding, Layer, Dropout, add
import keras.backend as K


class DecoderModule(Layer):

    def __init__(self, size_vocab, max_sequence, embedding_size, n_layers, n_head, d_k,
                 d_v, d_model, d_inner, dropout_rate=0.1, return_self_attention=False):

        super(DecoderModule, self).__init__()
        n_position = max_sequence + 1
        self.n_position = n_position
        self.n_layers = n_layers
        self.size_of_vocabulary = size_vocab
        self.num_of_heads = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout_rate
        self.return_self_attention = return_self_attention
        self.emb_size = embedding_size

        self.embs_dropout = Dropout(dropout_rate)

        self.emb_layer = Embedding(size_vocab, embedding_size)

        self.pos_layer = Embedding(n_position, embedding_size,
                                   embeddings_initializer=SinusoidInitializer(padding_index=0),
                                   trainable=False)
        self.stack_decoder_layers = []
        for _ in range(n_layers):
            self.stack_decoder_layers.append(
                DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout_rate)
            )

    def compute_output_shape(self, input_shape):
        shape = self.emb_layer.compute_output_shape(input_shape[0])
        shape2 = input_shape[-1]
        shape, _, shape_attn = self.stack_decoder_layers[0].compute_output_shape([shape, shape2, shape2])
        if self.return_self_attention:
            return [shape] + [shape_attn for _ in range(self.n_layers)]
        return shape

    def __call__(self, inputs, return_attns=False, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 4
        tgt_seq = inputs[0]
        src_seq = inputs[2]

        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = K.greater(slf_attn_mask_keypad+slf_attn_mask_subseq, 0)
        slf_attn_mask = K.variable(K.eval(slf_attn_mask) * 1)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        return super(DecoderModule, self).__call__(inputs, non_pad_mask=non_pad_mask,
                                                   self_atten_mask=slf_attn_mask,
                                                   dec_enc_attn_mask=dec_enc_attn_mask,
                                                   return_attns=return_attns, **kwargs)

    def call(self, inputs, non_pad_mask=None, self_atten_mask=None,
             dec_enc_attn_mask=None, return_attns=False, **kwargs):
        seq_tgt, pos_tgt, _, enc_seq = inputs
        state = self.embs_dropout(add([self.emb_layer(seq_tgt), self.pos_layer(pos_tgt)]))
        attention_list = []
        for layer in self.stack_decoder_layers:
            state, _, attention = layer([state, enc_seq, enc_seq], non_pad_mask=non_pad_mask,
                                        self_atten_mask=self_atten_mask, dec_enc_attn_mask=dec_enc_attn_mask, **kwargs)
            attention_list.append(attention)
        if return_attns:
            return [state] + attention_list
        return state

    def get_inputs(self):
        return [self.emb_layer, self.pos_layer]

    def get_config(self):
        config = {
            "number_of_layers": self.n_layers,
            "max_vocabulary": self.size_of_vocabulary,
            "max_position": self.n_position,
            "embedding_size": self.emb_size,
            "number_of_heads(n_heads)": self.num_of_heads,
            "keys_dimension(d_k)": self.d_k,
            "values_dimension(d_v)": self.d_v,
            "model_dimension(d_model)": self.d_model,
            "inner_dimension(d_inner)": self.d_inner,
            "dropout_rate": self.dropout_rate
        }
        layer_conf = super(DecoderModule, self).get_config()

        return dict(list(config.items()) + list(layer_conf.items()))
