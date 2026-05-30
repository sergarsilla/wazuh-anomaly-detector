"""End-to-end training + threshold tests on synthetic data."""

import json
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.features import standardize
from src.model import LogAutoencoder
from training.evaluate import calculate_anomaly_threshold
from training.train import train_model

INPUT_DIM = 64


def _write_config(tmp_path: Path) -> str:
    config = {
        "wazuh_archives_path": "/tmp/archives.json",
        "wazuh_socket_path": "/tmp/queue",
        "model_save_path": str(tmp_path / "model.pt"),
        "input_dim": INPUT_DIM,
        "latent_dim": 8,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 30,
        "anomaly_threshold_tau": 0.0,
        "scaler_mean": [],
        "scaler_var": [],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return str(config_path)


def test_train_and_threshold_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    # Normal traffic: a tight cluster around a fixed center.
    center = rng.normal(0.0, 1.0, size=INPUT_DIM)
    normal = (center + rng.normal(0.0, 0.1, size=(400, INPUT_DIM))).astype(np.float32)
    validation = (
        center + rng.normal(0.0, 0.1, size=(150, INPUT_DIM))
    ).astype(np.float32)

    normal_path = tmp_path / "normal.npy"
    val_path = tmp_path / "val.npy"
    np.save(normal_path, normal)
    np.save(val_path, validation)

    config_path = _write_config(tmp_path)

    # --- Train ---
    final_loss = train_model(config_path, str(normal_path))
    assert isinstance(final_loss, float)
    assert (tmp_path / "model.pt").exists()

    config = load_config(config_path)
    assert len(config["scaler_mean"]) == INPUT_DIM
    assert len(config["scaler_var"]) == INPUT_DIM

    # --- Threshold ---
    tau = calculate_anomaly_threshold(config_path, str(val_path))
    assert tau > 0.0
    assert load_config(config_path)["anomaly_threshold_tau"] == tau

    # --- Detection sanity check ---
    # Anomalies drawn from a very different distribution should, on average,
    # produce a reconstruction error well above the threshold.
    config = load_config(config_path)
    model = LogAutoencoder(input_dim=INPUT_DIM, latent_dim=8)
    model.load_state_dict(torch.load(config["model_save_path"], map_location="cpu"))
    model.eval()

    anomalies = rng.normal(8.0, 2.0, size=(100, INPUT_DIM)).astype(np.float32)
    normalized = standardize(anomalies, config["scaler_mean"], config["scaler_var"])
    with torch.no_grad():
        inputs = torch.from_numpy(normalized)
        recon = model(inputs)
        anomaly_mse = torch.mean((inputs - recon) ** 2, dim=1).numpy()

    # The vast majority of anomalies should exceed tau.
    detection_rate = float(np.mean(anomaly_mse > tau))
    assert detection_rate > 0.9
