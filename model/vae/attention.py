import tensorflow as tf
import numpy as np
import keras
import math
from model.utils import make_triu


class SelfAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_embd, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = keras.layers.Dense(3*d_embd, use_bias=in_proj_bias)
        self.out_proj = keras.layers.Dense(d_embd, use_bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads
        
    def call(self, inputs, causal_mask=False):
        
        input_shape = inputs.shape
        b, s, d_embd = input_shape
        interm_shape = [b, s, self.n_heads, self.d_head]
        q, k, v = tf.split(self.in_proj(inputs), 3, axis=-1) # 3 tensors of shape (batch_size, Seq_Len, Dim)
        q = tf.transpose(tf.reshape(q, interm_shape), perm=[0, 2, 1, 3]) # (batch_size, n_heads, Seq_len, Dim/n_heads)
        k = tf.transpose(tf.reshape(k, interm_shape), perm=[0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, interm_shape), perm=[0, 2, 1, 3])
        
        weight = q @ tf.transpose(k, perm=[0, 1, 3, 2]) # (batch_size, n_heads, Seq_len, Seq_Len)
        
        if causal_mask:
            mask = make_triu(weight, dtype=tf.bool)
            weight = tf.where(mask, -np.inf, weight)
        weight /= math.sqrt(self.d_head)
        weight = tf.nn.softmax(weight, axis=- 1)
        
        output = weight @ v # (batch_size, n_heads, Seq_len, Dim/n_heads)
        output = tf.transpose(output, perm=[0, 2, 1, 3]) # (batch_size, Seq_len, n_heads, Dim/n_heads)
        output = tf.reshape(output, input_shape)
        output = self.out_proj(output)
        
        return output


if __name__ == "__main__":
    tensor = tf.random.normal(shape=(32, 32 * 32, 512))
    attention = SelfAttention(1, 512)
    output = attention(tensor, causal_mask=True)
    print()