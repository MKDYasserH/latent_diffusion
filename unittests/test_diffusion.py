import tensorflow as tf
from model.diffusion.diffusion import Diffusion
from unittests.confttest import mock_clip_layer_data, mock_encoder_data, mock_diffusion_time_data

def test_diffusion_output_shape():
    # Instantiate the model
    diffusion = Diffusion()
    
    latent = mock_encoder_data
    context = mock_clip_layer_data
    time = mock_diffusion_time_data
    
    # Call the model
    output = diffusion.call(latent, context, time)
    
    # Expected output shape based on the model architecture
    expected_shape = (32, 4, 32, 32)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
