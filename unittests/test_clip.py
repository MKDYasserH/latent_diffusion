import tensorflow as tf
from unittests.confttest import mock_clip_token_data, mock_clip_layer_data
from model.clip import ClipEmbedding, ClipLayer, CLIP

def test_clip_embedding_output_shape(mock_clip_token_data):
    # Test the output shape of ClipEmbedding
    clip_embedding = ClipEmbedding(n_vocab=49408, n_embd=768, n_tokens=77)
    input_tokens = mock_clip_token_data

    output = clip_embedding(input_tokens)

    expected_shape = (32, 77, 768)  # (batch_size, seq_len, embd_dim)
    assert output.shape == expected_shape

def test_clip_layer_output_shape(mock_clip_layer_data):
    # Test the output shape of ClipLayer
    layer = ClipLayer(n_head=1, n_embd=768)
    input_data = mock_clip_layer_data

    output = layer(input_data)

    assert output.shape == input_data.shape  # Output shape should remain unchanged

def test_clip_model_output_shape(mock_clip_token_data):
    # Test the output shape of the CLIP model
    model = CLIP()
    input_tokens = mock_clip_token_data

    output = model(input_tokens)

    expected_shape = (32, 77, 768)  # (batch_size, seq_len, embd_dim)
    assert output.shape == expected_shape