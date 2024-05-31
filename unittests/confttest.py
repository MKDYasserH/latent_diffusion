import pytest
import tensorflow as tf

@pytest.fixture(scope='module')
def mock_encoder_data():
    tensor = tf.random.normal(shape=(32, 3, 256, 256))  # (batch_size, channels, height, width)
    return tensor

@pytest.fixture(scope='module')
def mock_decoder_data():
    tensor = tf.random.normal(shape=(32, 4, 32, 32)) # (batch_size, features, height, width)
    return tensor

@pytest.fixture(scope='module')
def mock_selfatt_data():
    tensor = tf.random.normal(shape=(32, 32 * 32, 512))
    return tensor

@pytest.fixture(scope='module')
def mock_clip_token_data():
    token = tf.random.uniform(shape=(32,77), minval=0, maxval=10, dtype=tf.int64)
    return token

@pytest.fixture(scope='module')
def mock_clip_layer_data():
    tensor = tf.random.uniform(shape=(32, 77, 768), minval=0, maxval=10, dtype=tf.float64)
    return tensor

@pytest.fixture(scope='module')
def mock_diffusion_time_data():
    tensor = tf.random.uniform(shape=(1, 320))
    return tensor