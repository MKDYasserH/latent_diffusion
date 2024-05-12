import tensorflow as tf
import keras
from model.vae.attention import SelfAttention

class ClipEmbedding(keras.Model):
    def __init__(self, n_vocab, n_embd, n_tokens):
        super().__init__()
        self.token_embedding = keras.layers.Embedding(n_vocab, n_embd)
        self.position_embedding = tf.Variable(tf.zeros([n_tokens, n_embd]))
    
    def call(self, tokens):
        # (batch_size, Seq_len) -> (batch_size, Seq_len, dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x

class ClipLayer(keras.layers.Layer):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.linear_1 = keras.layers.Dense(4 * n_embd)
        self.linear_2 = keras.layers.Dense(n_embd)
        
    def call(self, z):
        #(batch_size, Seq_Len, dim)
        # Self attention
        residue = z
        z = self.layernorm_1(z)
        z = self.attention(z, causal_mask=True)
        z += residue
        
        # Feed Forward Layer
        residue = z
        z = self.layernorm_2(z)
        z = self.linear_1(z)
        z = z * tf.sigmoid(1.702 * z) # QuickGelu activation function
        z = self.linear_2(z)
        z += residue
        return z
        

class CLIP(keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = ClipEmbedding(49408, 768, 77)
        self.sub_layers = [ClipLayer(12,768) for i in range(12)]
        self.layernorm = keras.layers.LayerNormalization()
    
    def call(self, tokens):
        tokens = tf.cast(tokens, tf.int64)
        # (batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
        state = self.embedding(tokens)
        for sub_layer in self.sub_layers:
            state = sub_layer(state)
        
        # (batch_size, Seq_Len, Dim)
        output = self.layernorm(state)
        return output
        
if __name__ == "__main__":
    token = tf.random.uniform(shape=(32,77), minval=0, maxval=10, dtype=tf.int64)
    #clip_layer = ClipLayer(1, 16)
    clip_embd = ClipEmbedding(n_vocab=49408, n_embd=768, n_tokens=77)
    ouput = clip_embd(token)
    print()