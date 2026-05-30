"""Training pipeline for the anomaly-detection autoencoder.

Intended to run on the OCI training box. It learns to reconstruct *normal*
traffic only. The steps are:

1. Load a dataset of normal feature vectors (``normal_dataset.npy``).
2. Fit a StandardScaler (per-column mean/variance) and normalize the data.
3. Train the autoencoder with Adam + MSE loss for the configured epochs.
4. Persist the trained weights and write the scaler mean/variance back into the
   global config so inference applies the identical normalization.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Allow running as a standalone script (``python training/train.py``).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, update_config  # noqa: E402
from src.model import LogAutoencoder  # noqa: E402


def train_model(config_path: str, dataset_path: str) -> float:
    """Train the autoencoder on normal data and persist weights + scaler.

    Args:
        config_path: Path to the global JSON config.
        dataset_path: Path to a ``.npy`` array of shape (n_samples, input_dim)
            containing only normal traffic.

    Returns:
        The average reconstruction loss of the final epoch.
    """
    config = load_config(config_path)

    dataset = np.load(dataset_path).astype(np.float32)

    # Fit the scaler on normal data; its mean/var define the normalization used
    # everywhere downstream.
    scaler = StandardScaler()
    normalized = scaler.fit_transform(dataset).astype(np.float32)

    tensor_data = torch.from_numpy(normalized)
    # The autoencoder reconstructs its own input, so inputs and targets match.
    loader = DataLoader(
        TensorDataset(tensor_data, tensor_data),
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    model = LogAutoencoder(
        input_dim=int(config["input_dim"]),
        latent_dim=int(config["latent_dim"]),
    )
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = nn.MSELoss()

    model.train()
    epochs = int(config["epochs"])
    average_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            reconstruction = model(inputs)
            loss = criterion(reconstruction, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        average_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.6f}")

    # Persist weights and the scaler parameters into the config.
    os.makedirs(os.path.dirname(config["model_save_path"]) or ".", exist_ok=True)
    torch.save(model.state_dict(), config["model_save_path"])
    update_config(
        config_path,
        {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_var": scaler.var_.tolist(),
        },
    )
    print(f"Saved model to {config['model_save_path']} and updated scaler params.")
    return average_loss


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config/global_config.json"
    dataset_arg = sys.argv[2] if len(sys.argv) > 2 else "data/normal_dataset.npy"
    train_model(config_arg, dataset_arg)
