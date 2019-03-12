from keras import backend as K
from keras import initializers as init
from keras.layers import Layer, Softmax, Dropout, dot, add, \
    Dense, Dot, Reshape, RepeatVector, Conv1D, Lambda
from keras.initializers import normal
from numba import jit

import numpy as np


class ScaledDotProductAttention(Layer):
    """ ScaledDotProductAttention
        Based on implementation from Yu-Hsiang Huang(2017) at arXiv:1706.03762v5.
        example:

        >>> import numpy as np
        >>> from layers import ScaledDotProductAttention
        >>>
        >>> attention = ScaledDotProductAttention(temperature=np.sqrt(64), dropout_rate=0.1)
        >>>
        >>> output, attention = attention([queries, keys, values])
    """

    def __init__(self, temperature, dropout_rate=0.1):
        """ :constructor __init__:
            :param temperature: value used to scale attention outs.
            :param dropout_rate: rate from outcomings from layer which will be dropped
            from others layers

            example:
            >>> import numpy as np
            >>> from layers import ScaledDotProductAttention
            >>>
            >>> attention = ScaledDotProductAttention(temperature=np.sqrt(64), dropout_rate=0.1)
            """
        super(ScaledDotProductAttention, self).__init__()
        self.scaled_dot = _ScaledDotProductAttentionAux(temperature)
        self.dropout = Dropout(dropout_rate)
        self.softmax = Softmax(axis=-1)

    def compute_output_shape(self, input_shape):
        batch_size, seq_size, _ = input_shape[0]
        _, seq_size2, _ = input_shape[1]
        tmp = self.scaled_dot.compute_output_shape(input_shape[:2])
        return [input_shape[-1], tmp]

    def call(self, inputs, mask=None, **kwargs):
        assert isinstance(inputs, list), "Error! Expect List of inputs."
        assert len(inputs) == 3, "Error! Expected three inputs."

        q, k, v = inputs

        attn = self.scaled_dot([q, v])

        if mask is not None:
            attn = add([attn, mask.T * -np.inf])

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        out = dot([attn, v], 1)

        return [out, attn]


class _ScaledDotProductAttentionAux(Dot):

    def __init__(self, temperature):
        self.temp = temperature
        super(_ScaledDotProductAttentionAux, self).__init__(-1)

    def compute_output_shape(self, input_shape):
        start, (length1, _) = input_shape[0][:-2], input_shape[0][-2:]
        length2, _ = input_shape[1][-2:]
        return start + (length1, length2)

    def call(self, inputs, **kwargs):
        q, k = inputs
        attn = super(_ScaledDotProductAttentionAux, self).call([q, k])
        return attn / self.temp


class LayerNorm(Layer):
    """
    Class LayerNorm:
    Normalize output from layer, based on article at arXiv:1607.06450v1
    from Jimmy Lei Ba, Jamie Ryan Kiros and Geoffrey E. Hinton

    ex.:

    >>> from layers import LayerNorm
    >>>
    >>> layer_norm = LayerNorm()
    >>>
    >>> layer_norm(inputs)
    """

    def __init__(self, size=None, eps=1e-16, alpha_init="ones", beta_init="zeros"):
        self.size = size
        self.eps = eps
        self.alpha_init = init.get(alpha_init)
        self.beta_init = init.get(beta_init)
        super(LayerNorm, self).__init__()
        self.alpha, self.beta = None, None

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        if self.size:
            size = (1, self.size)
        else:
            size = (1, input_shape[-1])

        self.alpha = self.add_weight("kernel_alpha", size, initializer=self.alpha_init)
        self.beta = self.add_weight("kernel_beta", size, initializer=self.beta_init)
        self.inshape = input_shape
        self.built = True

    def call(self, inputs, **kwargs):
        eps = self.eps

        def aux_func(x):
            mean = K.mean(x, axis=-1, keepdims=True)
            std = K.sqrt(K.var(inputs, axis=-1, keepdims=True) + eps)
            return (x - mean) / std

        output = Lambda(aux_func, self.inshape)(inputs)

        return output * self.alpha + self.beta


class _ReshapeAndPermute(Layer):

    def __init__(self, n_head, d_out, permute_axis=(2, 0, 1, 3), first_reshape=None, new_shape=None):
        assert isinstance(permute_axis, (tuple, list))
        self.n_head = n_head
        self.d_out = d_out
        self.axis = permute_axis
        self.in_shape = first_reshape
        self.new_shape = new_shape
        self.i_shape = None
        super(_ReshapeAndPermute, self).__init__()

    def compute_output_shape(self, input_shape):
        sz_b, length, _ = input_shape
        if self.new_shape:
            return self.new_shape
        return self.n_head * sz_b, length, self.d_out

    def build(self, input_shape):
        self.i_shape = input_shape
        super(_ReshapeAndPermute, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sz_b, length, d_model = self.i_shape
        in_shape = self.in_shape if self.in_shape else (sz_b, length, self.n_head, self.d_out)
        out = K.reshape(inputs, in_shape)
        out = K.permute_dimensions(out, self.axis)
        if self.new_shape:
            return K.reshape(out, self.new_shape)
        return K.reshape(out, (sz_b * self.n_head, length, self.d_out))


class MultiHeadAttention(Layer):

    def __init__(self, n_head, d_model, d_k=None, d_v=None, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k if d_k else int(d_model / n_head)
        self.d_v = d_v if d_v else int(d_model / n_head)

        d_k_w_init = normal(mean=0., stddev=np.sqrt(2.0 / (d_model + self.d_k)))
        d_v_w_init = normal(mean=0., stddev=np.sqrt(2.0 / (d_model + self.d_v)))

        self.w_qs = Dense(self.d_k * n_head, use_bias=False, kernel_initializer=d_k_w_init)
        self.w_ks = Dense(self.d_k * n_head, use_bias=False, kernel_initializer=d_k_w_init)
        self.w_vs = Dense(self.d_v * n_head, use_bias=False, kernel_initializer=d_v_w_init)

        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.layer_norm = LayerNorm(d_model)

        self.fc = Dense(self.d_model, use_bias=False, kernel_initializer='glorot_normal')
        self.dropout = Dropout(dropout_rate)
        self.shape_q, self.shape_k, self.shape_v = None, None, None

    def compute_output_shape(self, input_shape):
        sz_b, len_q, _ = self.shape_q
        _, len_k, _ = self.shape_k
        dims = _ReshapeAndPermute(self.n_head, self.d_v).compute_output_shape(input_shape[-1])
        dims = self.attention.compute_output_shape([dims, dims, dims])[-1]
        dims = _ReshapeAndPermute(self.n_head, self.d_v, permute_axis=(1, 0, 2, 3),
                                  first_reshape=(self.n_head, sz_b, len_q, len_k),
                                  new_shape=(sz_b, self.n_head, len_q, len_k)).compute_output_shape(dims)
        return [input_shape[0], dims]

    def build(self, input_shape):
        self.shape_q, self.shape_k, self.shape_v = input_shape
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q, k, v = inputs
        residual = q

        sz_b, len_q, _ = self.shape_q
        _, len_k, _ = self.shape_k
        _, len_v, _ = self.shape_v
        q = _ReshapeAndPermute(n_head, d_k)(self.w_qs(q))
        k = _ReshapeAndPermute(n_head, d_k)(self.w_ks(k))
        v = _ReshapeAndPermute(n_head, d_v)(self.w_vs(v))

        mask = Reshape((n_head, 1, 1))(RepeatVector(n_head)(mask)) if mask else None

        output, attention = self.attention([q, k, v], mask=mask)

        output = _ReshapeAndPermute(n_head, d_v, permute_axis=(1, 2, 0, 3),
                                    first_reshape=(n_head, sz_b, len_q, d_v),
                                    new_shape=(sz_b, len_q, n_head * d_v))(output)
        attention = _ReshapeAndPermute(n_head, d_v, permute_axis=(1, 0, 2, 3),
                                       first_reshape=(n_head, sz_b, len_q, len_k),
                                       new_shape=(sz_b, n_head, len_q, len_k))(attention)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output+residual)
        return [output, attention]


class PositionWiseFeedForward(Layer):

    def __init__(self, d_in, d_hidden, dropout_rate=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = Conv1D(d_hidden, 1)
        self.w2 = Conv1D(d_in, 1)
        self.layer_norm = LayerNorm(d_in)
        self.dropout = Dropout(dropout_rate)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        residual = inputs
        output = K.transpose(inputs)
        output = self.w2(K.relu(self.w1(output)))
        output = K.transpose(output)
        output = self.dropout(output)
        return self.layer_norm(add(output, residual))


jit
def test_module():
    from keras.models import Model
    from keras.layers import Input, Dense

    print("+=======================================+")
    print("|   Testing ScaledDotProductAttention   |")
    print("+=======================================+")

    size = [20, 10, 15]
    x = np.random.normal(size=size)

    inputs = Input(batch_shape=size)

    layer = ScaledDotProductAttention(15)([inputs, inputs, inputs])

    out = Dense(1)(layer[0])

    model = Model(inputs, [out])
    model.summary()
    model.compile("sgd", ["mse"], metrics=["accuracy"])

    print("\nmodel predict ...")
    print(model.predict(x))

    print("+=======================================+")
    print("|       Testing MultiHeadAttention      |")
    print("+=======================================+")

    layer = MultiHeadAttention(5, 15)([inputs, inputs, inputs])

    out = Dense(1)(layer[0])

    model = Model(inputs, [out])
    model.summary()
    model.compile("sgd", ["mse"], metrics=["accuracy"])

    print("\nmodel predict ...")
    print(model.predict(x))


if __name__ == "__main__":
    test_module()
