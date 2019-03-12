from keras.models import Model
from keras.layers import Input

import keras.backend as K
import numpy as np
from modules import Transformer
from math import log2

__author__ = "João Luiz Costa Barreto"


class TransformerModel(Model):
    """class TransformModel(keras.model.Model): contains methods used in my Tranformer keras layer version without other
    hand implementation of model of keras.

    """
    def __init__(self, max_source_vocabulary, max_target_vocabulary, max_length_sequence=500, embedding_size=512,
                 d_model=512, d_inner=2048, num_of_layers=6, num_heads=8, d_k=None, d_v=None, dropout_rate=0.1,
                 return_attention=False):

        self.transformer = Transformer(max_source_vocabulary, max_target_vocabulary, max_length_sequence,
                                       embedding_size, d_model, d_inner, num_of_layers, num_heads, d_k, d_v,
                                       dropout_rate, return_attention)
        inputs = [Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,))]
        super(TransformerModel, self).__init__(inputs, outputs=[self.transformer])

    def get_attention_weights(self, inputs, batch_size=32):
        tmp_models = TransformerModel(return_attention=True, **self.get_config())
        tmp_models.set_weights(self.get_weights())

        return [K.eval(i) for i in tmp_models.predict(inputs, batch_size=batch_size)[1:]]

    def decode(self, inputs, beam_size=1, max_decode_length=1000, max_decode_length_rate=4, eos_tok_index=2):
        """Method decode: decode a sentence in source language to target sentece using beam search
        :param inputs: list with four elements, [source_sentence, source_position_of_word_in_sentence,
        target_sequence (a slice from target sentence, usually start_symbol), target_position]
        :param beam_size: size of beam, or number of hypotheses tested for each path
        :param max_decode_length: sentence length where the method should stop and resulting a no sense sentence.
        :param max_decode_length_rate: max number of times allowed where the target sentence length is bigger
        than source sentence length (similar intention from previous parameter)
        :param eos_tok_index: number index from word embedding of end-of-sentence token
        :return:
        """
        assert beam_size > 0
        assert max_decode_length > 0
        assert max_decode_length_rate > 0

        def get_k_top(x, k=beam_size):
            result = [np.argmax(x)[:, -1]]

            def set_zero(x, index):
                x[:, -1, index] *= 0
                return x

            aux = set_zero(inputs, result[-1])

            for _ in range(k-1):
                result.append(np.argmax(aux)[:, -1])
                aux = set_zero(aux, result[-1])

            return zip(result, np.take(x[:, -1], result, -1))

        def append_tgt_wd(xs, append_index):
            _, _, seq, pos = xs
            seq[0].append(append_index)
            pos[0].append(pos[0][-1] + 1)
            new_sent = inputs[:2]
            new_sent.append(seq)
            new_sent.append(pos)
            return new_sent

        cmp = lambda x, y: 0 if x == y else 1 if x > y else -1
        cmp2 = lambda x, y: cmp(x[0][-1][-1], y[0][-1][-1]) if not cmp(x[0][-1][-1], y[0][-1][-1]) \
                                                            else cmp(x[-1], y[-1])

        paths = [(inputs, log2(1))]

        while True:
            top = paths[0]
            tmp = self.predict(top[0])
            hip = get_k_top(tmp)

            if hip[0][0] == eos_tok_index:
                return top[0][-2], tmp, top[1]

            src_len, tgt_len = top[0][1][-1], top[0][-1][-1]
            if int(src_len * max_decode_length_rate) == tgt_len + 1 \
                    or tgt_len + 1 > max_decode_length:
                print("Warning: This sentence reach its limit size or proportion in relation to source size ")
                return top[0][-2], tmp, top[1]

            for h, hp in hip:
                new_in = append_tgt_wd(top[0], h)
                paths.append((new_in, log2(top[1]+hp)))

            del paths[0]

            paths.sort(cmp2)

    def get_config(self):
        trans = self.transformer.get_config()
        trans.pop("return_attention")
        dict1 = {"max_source_vocabulary": trans.pop("n_src_vocab"),
                 "max_target_vocabulary": trans.pop("n_tgt_vocab"),
                 "max_length_sequence": trans.pop("len_max_seq"),
                 "num_of_layers": trans.pop("n_layers"),
                 "num_heads": trans.pop("n_head")}

        for k, v in trans.items():
            dict1[k] = v
        return dict1
