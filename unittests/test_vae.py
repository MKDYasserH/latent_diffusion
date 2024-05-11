import pytest
import tensorflow as tf
from unittests.confttest import mock_encoder_data, mock_decoder_data, mock_selfatt_data
from model.vae.encoder import VAE_Encoder
from model.vae.decoder import VAE_Decoder
from model.vae.attention import SelfAttention


def test_vae_encoder_output_shape(mock_encoder_data):
    # Test the output shape of the encoder with a sample input
    encoder = VAE_Encoder(latent_dim=8)
    noise = tf.random.normal(shape=(32, 4, 32, 32))  # Match latent dimensions

    output = encoder(mock_encoder_data, noise)

    # Expected output shape based on the model architecture
    expected_shape = (32, 4, 32, 32)
    assert output.shape == expected_shape

def test_vae_decoder_output_shape(mock_decoder_data):
    # Test the output shape of the decoder with a sample input
    decoder = VAE_Decoder(latent_dim=4)

    output = decoder(mock_decoder_data)

    # Expected output shape based on the model architecture
    expected_shape = (32, 3, 256, 256)
    assert output.shape == expected_shape

def test_selfattention_output_shape(mock_selfatt_data):
    # Test the output shape with a sample input
    layer = SelfAttention(n_heads=1, d_embd=512)

    output = layer(mock_selfatt_data)

    expected_shape = (32, 32 * 32, 512)  # Input shape should be preserved
    assert output.shape == expected_shape

def test_selfattention_causal_mask(mock_selfatt_data):
    layer = SelfAttention(n_heads=1, d_embd=512)

    output = layer(mock_selfatt_data, causal_mask=True)

    expected_shape = (32, 32 * 32, 512)  # Input shape should be preserved
    assert output.shape == expected_shape
