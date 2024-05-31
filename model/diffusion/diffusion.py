import tensorflow as tf
import keras
from model.diffusion.blocks import TimeEmbedding, UNET, UNET_OutputLayer

class Diffusion(keras.Model):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
        
    def call(self, latent, context, time):
        # latent (batch_size, 4, height/8, Width/8)
        # context: (batch_size, Seq_Len, Dim)
        #time: (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (batch, 4, Height/8, Width/8) -> (Batch_Size, 320, Height/8, Width/8)
        output = self.unet(latent, context, time)        
        
        # (Batch_Size, 320, Height/8, Width/8) -> (batch_size, 4, Height/8, Width/8)
        output = self.final(output)
        
        # (batch_size, 4, Height/8, Width/8)
        return output
    
if __name__ == "__main__":
    latent = tf.random.normal(shape=(16, 4, 64, 64))
    context = tf.random.uniform(shape=(16, 77, 768), minval=0, maxval=10)
    time = tf.random.uniform(shape=(1, 320))
    
    diffusion = Diffusion()
    output = diffusion(latent, context, time)