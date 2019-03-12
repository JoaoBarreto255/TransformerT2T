"""
Transform Module Class
"""
from DecoderModule import DecoderModule
from EncoderModule import EncoderModule
from keras.layers import Layer, Dense


class Transformer(Layer):

    def __init__(self, n_src_vocab, n_tgt_vocab, len_max_seq, embeding_size=512, d_model=512,
                 d_inner=2048, n_layers=6, n_head=8, d_k=None, d_v=None, dropout_rate=0.1,
                 return_attention=False, **kwargs):
        if d_model % n_head:
            print("Warnning: Your model dimension(d_model) should be divisible by number of heads(n_head)")
        assert d_model == embeding_size, "Your d_model must be the like embeding_size"
        self.n_src_vocab = n_src_vocab
        self.n_tgt_vocab = n_tgt_vocab
        self.len_max_seq = len_max_seq
        self.emb_size = embeding_size
        self.d_model = d_model
        self.d_inner = d_inner
        self.num_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k if d_k is not None else d_model//n_head
        self.d_v = d_v if d_v is not None else d_model//n_head
        self.dropout_rate = dropout_rate
        self.return_attention = return_attention
        super(Transformer, self).__init__(**kwargs)

        self.encoder = EncoderModule(n_src_vocab, len_max_seq, embeding_size, n_layers, n_head,
                                     self.d_k, self.d_v, self.d_model, self.d_inner, dropout_rate)

        self.decoder = DecoderModule(n_tgt_vocab, len_max_seq, embeding_size, n_layers, n_head,
                                     self.d_k, self.d_v, self.d_model, self.d_inner, dropout_rate,
                                     True)

        self.projection_layer = Dense(n_tgt_vocab, activation="softmax", use_bias=False)

    def compute_output_shape(self, input_shape):
        enc_out_shapes = self.encoder.compute_output_shape(input_shape[:2])
        dec_out_shapes = self.decoder.compute_output_shape([input_shape[2], enc_out_shapes[0]])
        dec_out_shapes, dec_atten_shapes = dec_out_shapes[0], dec_out_shapes[1:]

        if self.return_attention:
            return [dec_out_shapes]+dec_atten_shapes
        return dec_out_shapes

    def call(self, inputs, return_attention=True, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 4
        src_seq, src_pos, tgt_seq, tgt_pos = inputs
        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_outs = self.encoder([src_seq, src_pos])
        enc_out = enc_outs[0]

        dec_outs = self.decoder([tgt_seq, tgt_pos, src_seq, enc_out])
        dec_out = dec_outs[0]
        dec_attentions = dec_outs[1:]

        output = self.projection_layer(dec_out)
        if self.return_attention:
            return [output] + dec_attentions
        return output

    def get_config(self):
        return {
            "n_src_vocab": self.n_src_vocab,
            "n_tgt_vocab": self.n_tgt_vocab,
            "len_max_seq": self.len_max_seq,
            "embeding_size": self.emb_size,
            "d_model": self.d_model,
            "d_inner": self.d_inner,
            "n_layers": self.num_layers,
            "n_head": self.n_head,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "dropout_rate": self.dropout_rate,
            "return_attention": self.return_attention
        }
