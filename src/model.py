"""PyTorch autoencoder used for anomaly detection.

An autoencoder learns to compress an input into a small "latent" representation
and then reconstruct it. Trained only on *normal* traffic, it reconstructs normal
events with low error but struggles with unfamiliar (anomalous) ones, producing a
high reconstruction error. That error is the anomaly score.

Architecture (symmetric feedforward):
    encoder: 64 -> 32 -> 16 -> 8   (ReLU after every linear layer)
    decoder:  8 -> 16 -> 32 -> 64  (ReLU between layers; final layer is linear)

The final layer is intentionally left without an activation so the network can
reconstruct the full real-valued range of the scaled features.
"""

from __future__ import annotations

import torch
from torch import nn


class LogAutoencoder(nn.Module):
    """Symmetric feedforward autoencoder for log feature vectors."""

    def __init__(self, input_dim: int = 64, latent_dim: int = 8) -> None:
        super().__init__()
        hidden_one = input_dim // 2  # 32
        hidden_two = input_dim // 4  # 16

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_one),
            nn.ReLU(),
            nn.Linear(hidden_one, hidden_two),
            nn.ReLU(),
            nn.Linear(hidden_two, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_two),
            nn.ReLU(),
            nn.Linear(hidden_two, hidden_one),
            nn.ReLU(),
            nn.Linear(hidden_one, input_dim),  # Linear output (no activation).
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode ``x``, returning the reconstruction."""
        return self.decoder(self.encoder(x))
