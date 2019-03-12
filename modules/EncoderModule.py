"""
Encoder Module Class.
"""

from layers import EncoderLayer, SinusoidInitializer, \
    get_non_pad_mask, get_attn_key_pad_mask

from keras.layers import Embedding, Layer, Dropout, add


class EncoderModule(Layer):

    def __init__(self, size_vocab, max_sequence, embedding_size, n_layers, n_head, d_k,
                 d_v, d_model, d_inner, dropout_rate=0.1, return_self_attention=False):

        super(EncoderModule, self).__init__()
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
        self.stack_encoder_layers = []
        for _ in range(n_layers):
            self.stack_encoder_layers.append(
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout_rate)
            )

    def call(self, inputs, return_attns=False, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 0
        words, positions = inputs
        state = add([self.emb_layer(words), self.pos_layer(positions)])
        state, attention = self.stack_encoder_layers[0](self.embs_dropout(state), **kwargs)
        attention_list = [attention]
        for layer in self.stack_encoder_layers[1:]:
            state, attention = layer(state, **kwargs)
            attention_list.append(attention)

        if self.return_self_attention or kwargs.get("return_attentions", False) \
           or kwargs.get("return_self_attentions", False) or return_attns:
            self.return_self_attention = True
            return [state] + attention_list
        return state

    def compute_output_shape(self, input_shape):
        shape = self.emb_layer.compute_output_shape(input_shape[0])
        shape, shape_attn = self.stack_encoder_layers[0].compute_output_shape([shape]*3)
        if self.return_self_attention:
            return [shape] + [shape_attn for _ in range(self.n_layers)]
        return shape

    def __call__(self, inputs, return_attentions=False, **kwargs):
        seq_words = inputs[0]
        slf_atten_mask = get_attn_key_pad_mask(seq_words, seq_words)
        non_pad_mask = get_non_pad_mask(seq_words)
        return super(EncoderModule, self).__call__(inputs, return_attns=return_attentions,
                                                   self_attention_mask=slf_atten_mask,
                                                   non_pad_mask=non_pad_mask, **kwargs)

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
        layer_conf = super(EncoderModule, self).get_config()

        return dict(list(config.items()) + list(layer_conf.items()))
