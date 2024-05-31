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

class CrossAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = keras.layers.Dense(d_embed, use_bias=in_proj_bias)
        self.k_proj = keras.layers.Dense(d_embed, use_bias=in_proj_bias)
        self.v_proj = keras.layers.Dense(d_embed, use_bias=in_proj_bias)
        self.out_proj = keras.layers.Dense(d_embed, use_bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads
        
    def call(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
        
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interm_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = tf.transpose(tf.reshape(q, interm_shape), perm=[0, 2, 1, 3])
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = tf.transpose(tf.reshape(k, interm_shape), perm=[0, 2, 1, 3])
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = tf.transpose(tf.reshape(v, interm_shape), perm=[0, 2, 1, 3])
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ tf.transpose(k, perm=[0, 1, 3, 2])
        weight /= math.sqrt(self.d_head)
        weight = keras.activations.softmax(weight, -1)
        output = weight @ v
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = tf.reshape(output, input_shape)
        output = self.out_proj(output)
        return output
        
