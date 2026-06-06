"""Run the full training flow from a raw archives.json in a single command.

Chains the three training steps so the model can be (re)built with one call:

    1. export_dataset  - turn raw archives into normal/validation .npy vectors
    2. train_model     - train the autoencoder and persist weights + scaler
    3. calculate_anomaly_threshold - compute tau as a high percentile of error

Datasets are written to a temporary directory (they are intermediate artifacts),
while the model and tuned config land in their configured paths. Designed to run
inside the Docker "trainer" service, but works as a plain script too.

Env vars (used when run with no CLI args, e.g. inside Docker):
    CONFIG_PATH    - path to the global config (default: config/global_config.json)
    ARCHIVES_PATH  - path to the raw archives.json
                     (default: /var/ossec/logs/archives/archives.json)
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluate import calculate_anomaly_threshold  # noqa: E402
from training.export_dataset import export_dataset  # noqa: E402
from training.train import train_model  # noqa: E402


def run_training(
    config_path: str,
    archives_path: str,
    validation_ratio: float = 0.2,
) -> None:
    """Export, train and compute the threshold from a raw archives log."""
    with tempfile.TemporaryDirectory() as work_dir:
        print(f"[1/3] Exporting feature vectors from {archives_path} ...")
        total = export_dataset(archives_path, work_dir, validation_ratio=validation_ratio)
        if total == 0:
            raise SystemExit(
                "No usable events found in the archives log; nothing to train on. "
                "Let archives.json accumulate normal traffic first."
            )

        normal_path = os.path.join(work_dir, "normal_dataset.npy")
        validation_path = os.path.join(work_dir, "validation_dataset.npy")

        print("[2/3] Training the autoencoder ...")
        train_model(config_path, normal_path)

        if os.path.exists(validation_path):
            print("[3/3] Computing anomaly threshold (high percentile of error) ...")
            calculate_anomaly_threshold(config_path, validation_path)
        else:
            print(
                "[3/3] Not enough data for a validation split; "
                "threshold left unchanged. Capture more logs and re-run."
            )

    print("Training complete.")


if __name__ == "__main__":
    config = os.environ.get("CONFIG_PATH", "config/global_config.json")
    archives = os.environ.get(
        "ARCHIVES_PATH", "/var/ossec/logs/archives/archives.json"
    )
    run_training(config, archives)
