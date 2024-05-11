import keras
import tensorflow as tf
from model.vae.attention import SelfAttention

class VAE_AttentionBlock(keras.layers.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.groupnorm = keras.layers.GroupNormalization(32)
        self.attention = SelfAttention(1, input_dim)
        
    def call(self, inputs):
        residue = inputs
        b, f, h, w = inputs.shape
        z = tf.reshape(inputs, [b, f, h * w]) # (batch_size, features, height * width)
        z = tf.transpose(z, perm=[0, 2, 1]) # (batch_size, height * width, features)
        z = self.attention(z) # (batch_size, height * width, features)
        z = tf.transpose(z, perm=[0, 2, 1]) # (batch_size, features, height * width)
        z = tf.reshape(z, [b, f, h, w]) # (batch_size, features, height, width)
        z += residue
        return z

class VAE_ResidualBlock(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.groupnorm_1 = keras.layers.GroupNormalization(32)
        self.conv_1 = keras.layers.Conv2D(output_dim, kernel_size=3, padding="same", data_format="channels_first")
        
        self.groupnorm_2 = keras.layers.GroupNormalization(32)
        self.conv_2 = keras.layers.Conv2D(output_dim, kernel_size=3, padding="same", data_format="channels_first")
        
        if input_dim == output_dim:
            self.residual_layer = keras.layers.Identity()
        else:
            self.residual_layer = keras.layers.Conv2D(output_dim, kernel_size=1, padding="same", data_format="channels_first")
    
    def call(self, inputs):
        residue = inputs
        z = self.groupnorm_1(inputs)
        z = keras.activations.silu(z)
        z = self.conv_1(z)
        z = self.groupnorm_2(z)
        z = keras.activations.silu(z)
        z = self.conv_2(z)
        z = z + self.residual_layer(residue)
        return z