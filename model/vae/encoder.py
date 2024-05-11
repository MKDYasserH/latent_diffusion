import tensorflow as tf
import keras
from model.vae.decoder import VAE_AttentionBlock, VAE_ResidualBlock
from model.utils import Swish

class VAE_Encoder(keras.Model):
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.layers_list = [
            keras.layers.Conv2D(128, kernel_size=3, padding="same", data_format="channels_first"), #(batch_size, 128, height, width)
            VAE_ResidualBlock(128,128), 
            VAE_ResidualBlock(128,128), #(batch_size, 128, height, width)
            keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="valid", data_format="channels_first"), #(batch_size, 128, height/2, width/2)
            VAE_ResidualBlock(128,256), #(batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256,256),
            keras.layers.Conv2D(256, kernel_size=3, strides=2, padding="valid", data_format="channels_first"), #(batch_size, 256, height/4, width/4)
            VAE_ResidualBlock(256,512), #(batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512,512),
            keras.layers.Conv2D(512, kernel_size=3, strides=2, padding="valid", data_format="channels_first"), #(batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            keras.layers.GroupNormalization(32),
            Swish(),
            keras.layers.Conv2D(self.latent_dim, kernel_size=3, padding="same", data_format="channels_first"), #(batch_size, 8, height/8, width/8)
            keras.layers.Conv2D(self.latent_dim, kernel_size=1, padding="valid", data_format="channels_first")  #(batch_size, 8, height/8, width/8)
        ]
        
    def call(self, inputs, noise):
        z = inputs
        for layer in self.layers_list:
            if getattr(layer, 'strides', None) == (2, 2):
                z = tf.pad(z, [[0,0],[0,0], [0,1], [0,1]])
            z = layer(z)
        mean, log_variance = tf.split(z, 2, axis=1)
        log_variance = tf.clip_by_value(log_variance, -30, 20)
        
        variance = tf.exp(log_variance)
        stdev = tf.sqrt(variance)
        z = mean + stdev * noise
        z *= 0.18215
        return z