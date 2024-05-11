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