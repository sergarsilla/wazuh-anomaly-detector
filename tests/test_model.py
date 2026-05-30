"""Tests for the autoencoder architecture."""

import torch

from src.model import LogAutoencoder


def test_reconstruction_preserves_shape() -> None:
    model = LogAutoencoder(input_dim=64, latent_dim=8)
    batch = torch.randn(10, 64)
    output = model(batch)
    assert output.shape == batch.shape


def test_latent_bottleneck_dimension() -> None:
    model = LogAutoencoder(input_dim=64, latent_dim=8)
    batch = torch.randn(5, 64)
    latent = model.encoder(batch)
    assert latent.shape == (5, 8)


def test_final_layer_is_unbounded() -> None:
    # A linear output layer should allow values outside [-1, 1] / [0, 1],
    # unlike a Tanh/Sigmoid head. We check the parameter count path indirectly
    # by confirming the last module is a Linear layer.
    model = LogAutoencoder()
    last_layer = list(model.decoder.children())[-1]
    assert isinstance(last_layer, torch.nn.Linear)
