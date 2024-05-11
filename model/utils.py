import tensorflow as tf
import keras


class Swish(keras.layers.Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def call(self, inputs):
        return keras.activations.silu(inputs)
    
def make_triu(x, dtype=tf.bool):
    ones_like = tf.ones_like(x)
    triu_matrix = tf.linalg.band_part(ones_like, 0, -1)
    result = triu_matrix - tf.linalg.diag(tf.linalg.diag_part(triu_matrix))
    return tf.cast(result, dtype=dtype)