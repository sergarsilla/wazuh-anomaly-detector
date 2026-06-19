"""Threshold calibration helper (read-only, advisory).

Runs a validation set of *normal* traffic through the trained autoencoder and
reports the reconstruction-error distribution together with several candidate
thresholds and the false-positive rate each one would produce on that normal
set:

    * Gaussian-style: mu+2*sigma, mu+3*sigma, mu+4*sigma
    * Percentiles:    p99, p99.9, p99.99

The false-positive rate is the fraction of the normal set whose error exceeds a
candidate threshold; on benign data that fraction *is* the false-positive rate
the detector would show in production. The script recommends the smallest
threshold that keeps the false-positive rate at or below a target (default
0.1%), matching the percentile rationale in ``training/evaluate.py``.

Unlike ``evaluate.py``, this script never writes anything back: it is a tuning
aid you run by hand to choose ``tau_percentile`` with evidence. Production
``tau`` is still computed and persisted by ``evaluate.py``.

Usage:
    python scripts/calibrate_threshold.py [config.json] [validation.npy] [target_fp]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.features import standardize  # noqa: E402
from src.model import LogAutoencoder  # noqa: E402

# Default acceptable share of normal traffic allowed to cross the threshold.
DEFAULT_TARGET_FP: float = 0.001  # 0.1%


def reconstruction_errors(config: dict, validation_dataset_path: str) -> np.ndarray:
    """Return the per-sample reconstruction MSE for a normal validation set."""
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
        per_sample_mse = torch.mean((inputs - reconstruction) ** 2, dim=1)
    return per_sample_mse.numpy()


def candidate_thresholds(errors: np.ndarray) -> "list[tuple[str, float]]":
    """Build the labelled (name, tau) candidates from the error distribution."""
    mu = float(np.mean(errors))
    sigma = float(np.std(errors))
    return [
        ("mu+2sigma", mu + 2.0 * sigma),
        ("mu+3sigma", mu + 3.0 * sigma),
        ("mu+4sigma", mu + 4.0 * sigma),
        ("p99", float(np.percentile(errors, 99.0))),
        ("p99.9", float(np.percentile(errors, 99.9))),
        ("p99.99", float(np.percentile(errors, 99.99))),
    ]


def false_positive_rate(errors: np.ndarray, tau: float) -> float:
    """Fraction of the (normal) set whose error exceeds ``tau``."""
    return float(np.mean(errors > tau))


def calibrate(
    config_path: str,
    validation_dataset_path: str,
    target_fp: float = DEFAULT_TARGET_FP,
) -> None:
    """Print the error distribution, the candidate table and a recommendation."""
    config = load_config(config_path)
    errors = reconstruction_errors(config, validation_dataset_path)

    print(f"Validation samples: {errors.size}")
    print(
        "Error distribution: "
        f"min={errors.min():.6f} mean={errors.mean():.6f} "
        f"median={np.median(errors):.6f} max={errors.max():.6f} "
        f"std={errors.std():.6f}"
    )
    print(f"\nTarget false-positive rate: {target_fp:.4%}\n")

    candidates = candidate_thresholds(errors)
    print(f"{'candidate':<12} {'tau':>12} {'false_positive_rate':>22}")
    print("-" * 48)
    for name, tau in candidates:
        print(f"{name:<12} {tau:>12.6f} {false_positive_rate(errors, tau):>21.4%}")

    # Recommend the lowest tau (most sensitive) that still meets the target, so
    # we maximise detection without exceeding the agreed false-positive budget.
    acceptable = [
        (name, tau)
        for name, tau in candidates
        if false_positive_rate(errors, tau) <= target_fp
    ]
    print()
    if acceptable:
        name, tau = min(acceptable, key=lambda item: item[1])
        print(f"Recommended: {name} (tau={tau:.6f}) -> FP <= {target_fp:.4%}")
    else:
        name, tau = max(candidates, key=lambda item: item[1])
        print(
            f"No candidate meets {target_fp:.4%}; strictest is {name} "
            f"(tau={tau:.6f}, FP={false_positive_rate(errors, tau):.4%}). "
            "Collect more normal traffic or raise the target."
        )


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config/global_config.json"
    validation_arg = sys.argv[2] if len(sys.argv) > 2 else "data/validation_dataset.npy"
    target_arg = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_TARGET_FP
    calibrate(config_arg, validation_arg, target_arg)
