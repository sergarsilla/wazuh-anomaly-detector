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


def test_output_layer_is_unbounded() -> None:
    # The spec requires a pure linear output (no Sigmoid/Tanh) so reconstructions
    # can span the real range of the scaled features. Check the architecture has
    # no bounding activation, and that outputs can actually exceed [-1, 1].
    model = LogAutoencoder()
    assert isinstance(list(model.decoder.children())[-1], torch.nn.Linear)
    assert not any(
        isinstance(m, (torch.nn.Sigmoid, torch.nn.Tanh)) for m in model.modules()
    )

    # Drive the net with large inputs; a bounded head could never produce these.
    torch.manual_seed(0)
    output = model(torch.randn(256, 64) * 50)
    assert output.abs().max().item() > 1.0
