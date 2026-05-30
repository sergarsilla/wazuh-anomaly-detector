"""Dynamic anomaly-threshold calculation.

After training, this script measures the reconstruction error distribution on a
validation set of normal traffic and derives the alarm threshold:

    tau = mu + 3 * sigma

where ``mu`` and ``sigma`` are the mean and standard deviation of the per-sample
reconstruction errors. Anything above ``tau`` at inference time is flagged as an
anomaly. The threshold is written back into the global config.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, update_config  # noqa: E402
from src.features import standardize  # noqa: E402
from src.model import LogAutoencoder  # noqa: E402

# Number of standard deviations above the mean error used as the alarm threshold.
SIGMA_MULTIPLIER: float = 3.0


def calculate_anomaly_threshold(config_path: str, validation_dataset_path: str) -> float:
    """Compute and persist ``tau = mu + 3*sigma`` from validation errors.

    Args:
        config_path: Path to the global JSON config (must already contain the
            trained scaler mean/variance).
        validation_dataset_path: Path to a ``.npy`` array of normal validation
            vectors of shape (n_samples, input_dim).

    Returns:
        The computed threshold ``tau``.
    """
    config = load_config(config_path)

    dataset = np.load(validation_dataset_path).astype(np.float32)
    normalized = standardize(dataset, config["scaler_mean"], config["scaler_var"])

    model = LogAutoencoder(
        input_dim=int(config["input_dim"]),
        latent_dim=int(config["latent_dim"]),
    )
    model.load_state_dict(torch.load(config["model_save_path"], map_location="cpu"))
    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(normalized)
        reconstruction = model(inputs)
        # Per-sample mean squared error (MSE) across the feature dimension.
        per_sample_mse = torch.mean((inputs - reconstruction) ** 2, dim=1)

    errors = per_sample_mse.numpy()
    mu = float(np.mean(errors))
    sigma = float(np.std(errors))
    tau = mu + SIGMA_MULTIPLIER * sigma

    update_config(config_path, {"anomaly_threshold_tau": tau})
    print(f"mu={mu:.6f} sigma={sigma:.6f} -> tau={tau:.6f} (saved to config)")
    return tau


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config/global_config.json"
    validation_arg = (
        sys.argv[2] if len(sys.argv) > 2 else "data/validation_dataset.npy"
    )
    calculate_anomaly_threshold(config_arg, validation_arg)
