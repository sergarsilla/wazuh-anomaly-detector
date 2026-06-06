"""Dynamic anomaly-threshold calculation.

After training, this script measures the reconstruction error distribution on a
validation set of normal traffic and derives the alarm threshold as a high
percentile of that error:

    tau = percentile(errors, p)        (default p = 99.9)

Reconstruction errors are bounded below by zero and right-skewed, so the usual
``mu + 3*sigma`` (which assumes a Gaussian) badly under-estimates the tail and
floods the dashboard with false positives. A direct high percentile instead
means "the level only ~0.1% of *normal* traffic exceeds", which is exactly the
target false-positive rate. The percentile is configurable via ``tau_percentile``
and ``mu``/``sigma`` are still reported for reference. The threshold is written
back into the global config.
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

# Default percentile of the validation-error distribution used as the alarm
# threshold, when the config does not override ``tau_percentile``.
DEFAULT_TAU_PERCENTILE: float = 99.9


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
    percentile = float(config.get("tau_percentile", DEFAULT_TAU_PERCENTILE))
    tau = float(np.percentile(errors, percentile))

    update_config(config_path, {"anomaly_threshold_tau": tau})
    print(
        f"mu={mu:.6f} sigma={sigma:.6f} "
        f"p{percentile}={tau:.6f} -> tau={tau:.6f} (saved to config)"
    )
    return tau


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config/global_config.json"
    validation_arg = (
        sys.argv[2] if len(sys.argv) > 2 else "data/validation_dataset.npy"
    )
    calculate_anomaly_threshold(config_arg, validation_arg)
