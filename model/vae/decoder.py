import tensorflow as tf
import keras
from model.utils import Swish
from model.vae.layers import VAE_ResidualBlock, VAE_AttentionBlock

class VAE_Decoder(keras.Model):
    def __init__(self, latent_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.layers_list = [
            keras.layers.Conv2D(self.latent_dim, kernel_size=1, padding="valid", data_format="channels_first"),
            keras.layers.Conv2D(512, kernel_size=3, padding="same", data_format="channels_first"),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first"),
            keras.layers.Conv2D(512, kernel_size=3, padding="same", data_format="channels_first"),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first"),
            keras.layers.Conv2D(512, kernel_size=3, padding="same", data_format="channels_first"),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first"),
            keras.layers.Conv2D(256, kernel_size=3, padding="same", data_format="channels_first"),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            keras.layers.GroupNormalization(32),
            Swish(),
            keras.layers.Conv2D(3, kernel_size=3, padding="same", data_format="channels_first"),
        ]
    
    def call(self, inputs):
        inputs /= 0.18215
        z = inputs
        for layer in self.layers_list:
            z = layer(z)
        return z
