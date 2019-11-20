# from tensorflow.python.keras.engine.topology import Layer
from tensorflow.python.keras.layers import Layer, InputSpec

from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.regularizers import *
import tensorflow as tf
from tensorflow.python.keras import backend as K
# import tensorflow.python.keras.backend as K
# from tensorflow.python.keras import backend as K
# from tensorflow import  keras
# from tensorflow.python.keras import backend as K


class context2query_attention(Layer):

    def __init__(self, output_dim, c_maxlen, q_maxlen, dropout, **kwargs):
        self.output_dim = output_dim
        self.c_maxlen = c_maxlen
        self.q_maxlen = q_maxlen
        self.dropout = dropout
        super(context2query_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        ##[ context:(batch_size,time_step,dim),query:(batch_size,time_step,dim)]
        # input_shape: [(None, ?, 128), (None, ?, 128)]F
        init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
        self.W0 = self.add_weight(name='W0',
                                  shape=(input_shape[0][-1], 1),
                                  initializer=init,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[1][-1], 1),
                                  initializer=init,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, input_shape[0][-1]),
                                  initializer=init,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.bias = self.add_weight(name='linear_bias',
                                    shape=([1]),
                                    initializer='zero',
                                    regularizer=l2(3e-7),
                                    trainable=True)
        super(context2query_attention, self).build(input_shape)

    def mask_logits(self, inputs, mask, mask_value=-1e30):
        '''mask矩阵的值为1时就是原来的值，为零时对应padding部分的值'''
        mask = tf.cast(mask, tf.float32)
        #mask=1时就是input,mask=0时，inputs+mask_value是一个非常大的负数此时取softmax值几乎为0
        return inputs + mask_value * (1 - mask)

    def call(self, x, mask=None):
        '''
        shape=(batch_size,new_time_step,filters)
     x_cont=Tensor("layer_dropout_5/cond/Identity:0", shape=(None, None, 128), dtype=float32)
x_ques=Tensor("layer_dropout_11/cond/Identity:0", shape=(None, None, 128), dtype=float32)
c_mask=Tensor("batch_slice_4/Slice:0", shape=(None, None), dtype=bool)#
q_mask=Tensor("batch_slice_5/Slice:0", shape=(None, None), dtype=bool)
        '''
        x_cont, x_ques, c_mask, q_mask = x
        # get similarity matrix S
        ##K.dot(x_cont, self.W0)维度变化： [batch_size,time_step,dim] *[dim,1] =[batch_size,time_step,1]
        subres0 = K.tile(K.dot(x_cont, self.W0), [1, 1, self.q_maxlen])
        subres1 = K.tile(K.permute_dimensions(K.dot(x_ques, self.W1), pattern=(0, 2, 1)), [1, self.c_maxlen, 1])
        subres2 = K.batch_dot(x_cont * self.W2, K.permute_dimensions(x_ques, pattern=(0, 2, 1)))
        S = subres0 + subres1 + subres2
        S += self.bias
        q_mask = tf.expand_dims(q_mask, 1)
        #默认是对最后一维度，即axis=-1
        S_ = tf.nn.softmax(self.mask_logits(S, q_mask))
        c_mask = tf.expand_dims(c_mask, 2)
        S_T = K.permute_dimensions(tf.nn.softmax(self.mask_logits(S, c_mask), axis=1), (0, 2, 1))
        c2q = tf.matmul(S_, x_ques)
        q2c = tf.matmul(tf.matmul(S_, S_T), x_cont)
        result = K.concatenate([x_cont, c2q, x_cont * c2q, x_cont * q2c], axis=-1)

        return result

    def compute_output_shape(self, input_shape):
        ##输出张量的维度，与输入张量不一样时才使用，就是result的维度
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
