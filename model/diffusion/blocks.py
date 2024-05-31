import tensorflow as tf
import keras
from model.utils import Swish, CustomLayer
from model.vae.attention import SelfAttention, CrossAttention

class TimeEmbedding(keras.layers.Layer):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = keras.layers.Dense(4 * n_embd)
        self.linear_2 = keras.layers.Dense(4 * n_embd)
    
    def call(self, inputs):
        z = inputs
        z = self.linear_1(z)
        z = Swish()(z)
        z = self.linear_2(z)
        # (1, 1280)
        return z
    
class UNET_ResidualBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = keras.layers.GroupNormalization(32)
        self.conv_feature = keras.layers.Conv2D(out_channels, kernel_size=3, padding="same", data_format="channels_first")
        self.linear_time = keras.layers.Dense(out_channels)
        
        self.groupnorm_merged = keras.layers.GroupNormalization(32)
        self.conv_merged = keras.layers.Conv2D(out_channels, kernel_size=3, padding="same", data_format="channels_first")
        
        if in_channels == out_channels:
            self.residual_layer = keras.layers.Identity()
        else:
            self.residual_layer = keras.layers.Conv2D(out_channels, kernel_size=1, padding="valid", data_format="channels_first")
            
    def call(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = Swish()(feature)
        feature = self.conv_feature(feature)
        
        time = Swish()(time)
        time = self.linear_time(time)
        merged = feature + tf.expand_dims(tf.expand_dims(time, -1), -1)
        merged = self.groupnorm_merged(merged)
        merged = Swish()(merged)
        merged = self.conv_merged(merged)
        # (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)
    
class UNET_AttentionBlock(keras.layers.Layer):
    def __init__(self, n_head, n_embd, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = keras.layers.GroupNormalization(32, epsilon=1e-6)
        self.conv_input = keras.layers.Conv2D(channels, kernel_size=1, padding="valid", data_format="channels_first")
        
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = keras.layers.LayerNormalization()
        self.linear_geglu_1 = keras.layers.Dense(4 * channels * 2)
        self.linear_geglu_2 = keras.layers.Dense(channels)
        
        self.conv_output = keras.layers.Conv2D(channels, kernel_size=1, padding="valid", data_format="channels_first")
    
    def call(self, inputs, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)
        z = inputs
        residue_long = z
        z = self.groupnorm(z)
        z = self.conv_input(z)
        n, c, h, w = z.shape
        
        # (batch_size, features, height, width) -> # (batch_size, features, height * width)
        z = tf.reshape(z, [n, c, h*w])
        
        # (batch_size, features, height * width) -> # (batch_size, height * width, features)
        z = tf.transpose(z, perm=[0, 2, 1])
        
        residue_short = z
        z = self.layernorm_1(z)
        z = self.attention_1(z)
        z += residue_short
        
        residual_short = z
        
        # Normalisation + cross attention with skip connection
        z = self.layernorm_2(z)
        z = self.attention_2(z, context)
        z += residual_short
        residual_short = z
        # Normalisation + GeGlu and skip connection
        z = self.layernorm_3(z)
        z, gate = tf.split(self.linear_geglu_1(z), 2, -1)
        z = z * keras.activations.gelu(gate)
        z = self.linear_geglu_2(z)
        z += residue_short
        # (batch_size, height * width, features) -> (batch_size, features, height * width)
        z = tf.transpose(z, perm=[0, 2, 1])
        z = tf.reshape(z, [n, c, h, w])
        
        return self.conv_output(z) + residue_long
        
        

class UpSample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv = keras.layers.Conv2D(channels, kernel_size=3, padding="same", data_format="channels_first")
        self.upsample = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", data_format="channels_first"),
    
    def call(self, z):
        # (Batch_size, Features, Height*2, Width*2)
        z = self.upsample(z)
        return self.conv(z)
        

class SwitchSequential(keras.layers.Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def call(self, z, context, time):
        for layer in self.layers:
            if isinstance(layer, UNET_AttentionBlock):
                z = layer(z, context)
            elif isinstance(layer, UNET_ResidualBlock):
                z = layer(z, time)
            else:
                z = layer(z)
        return z

class UNET(keras.Model):
    def __init__(self):
        super().__init__()
        
        # ENCODER
        self.encoder = [
            # (Batch_size, 4, height/8, width/8)
            SwitchSequential([keras.layers.Conv2D(320, kernel_size=3, padding="same", data_format="channels_first")]),
            SwitchSequential([UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)]),
            SwitchSequential([UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)]),
            # (Batch_size, 32[0, height/16, width/16)
            SwitchSequential([keras.layers.Conv2D(320, kernel_size=3, strides=2, padding="same", data_format="channels_first")]),
            SwitchSequential([UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)]),
            SwitchSequential([UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)]),
            # (Batch_size, 64[0, height/32, width/32)
            SwitchSequential([keras.layers.Conv2D(640, kernel_size=3, strides=2, padding="same", data_format="channels_first")]),
            SwitchSequential([UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)]),
            SwitchSequential([UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)]),
            # (Batch_size, 12[80, height/64, width/64)
            SwitchSequential([keras.layers.Conv2D(1280, kernel_size=3, strides=2, padding="same", data_format="channels_first")]),
            SwitchSequential([UNET_ResidualBlock(1280, 1280)]),
            SwitchSequential([UNET_ResidualBlock(1280, 1280)])
        ]
        
        # BOTTLENECK
        self.bottleneck = SwitchSequential([UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160),UNET_ResidualBlock(1280, 1280)])
        
        
        # DECODER
        self.decoder = [
            # (Batch_size, 1280, Height/64, Width/64)
            SwitchSequential([UNET_ResidualBlock(2560, 1280)]),
            SwitchSequential([UNET_ResidualBlock(2560, 1280)]),
            SwitchSequential([UNET_ResidualBlock(2560, 1280), UpSample(1280)]),
            SwitchSequential([UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)]),
            SwitchSequential([UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)]),
            SwitchSequential([UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)]),
            SwitchSequential([UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)]),
            SwitchSequential([UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)]),
            SwitchSequential([UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)]),
            SwitchSequential([UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)]),
            SwitchSequential([UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)]),
            SwitchSequential([UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)])
        ]
    
    def call(self, inputs, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)
        z = inputs
        skip_connections = []
        for layer in self.encoder:
            z = layer(z, context, time)
            skip_connections.append(z)
        
        z = self.bottleneck(z, context, time)
        
        for layer in self.decoder:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            z = tf.concat([z, skip_connections.pop()], axis=1)
            z = layer(z, context, time)
        
        return z
        
        
class UNET_OutputLayer(keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = keras.layers.GroupNormalization(32)
        self.conv = keras.layers.Conv2D(out_channels, kernel_size=3, padding="same", data_format="channels_first")
    
    def call(self, inputs):
        z = inputs
        # (batch_size, 320, height/8, width/8)
        z = self.groupnorm(z)
        z = Swish()(z)
        z = self.conv(z)
        # (batch_size, 4, height/8, width/8)
        return z