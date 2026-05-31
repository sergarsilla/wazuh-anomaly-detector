"""Real-time inference orchestrator.

Ties every component together into the production loop:

    tail archives.json -> sanitize -> vectorize -> standardize -> autoencoder
    -> reconstruction error (MSE) -> if MSE > tau, inject a Wazuh alert.

The loop is built to never die on a single malformed event: any per-event error
is swallowed and the loop moves on.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.config import load_config
from src.features import LogVectorizer, standardize
from src.ingester import tail_wazuh_archives
from src.injector import WazuhSocketInjector
from src.model import LogAutoencoder
from src.sanitizer import ISOSanitizer

# Wazuh rule id emitted with each anomaly (must match rules/local_rules.xml).
ANOMALY_RULE_ID: int = 100100

# Log a heartbeat every N processed lines so it is clear the detector is alive.
HEARTBEAT_EVERY: int = 1000

logger = logging.getLogger("anomaly_detector")


def load_inference_model(config: Dict[str, Any]) -> LogAutoencoder:
    """Instantiate the autoencoder, load trained weights, and set eval mode."""
    model = LogAutoencoder(
        input_dim=int(config["input_dim"]),
        latent_dim=int(config["latent_dim"]),
    )
    model.load_state_dict(torch.load(config["model_save_path"], map_location="cpu"))
    model.eval()
    return model


def reconstruction_error(
    model: LogAutoencoder,
    vector: np.ndarray,
    scaler_mean: List[float],
    scaler_var: List[float],
) -> float:
    """Standardize a single feature vector and return its reconstruction MSE."""
    normalized = standardize(vector, scaler_mean, scaler_var)
    with torch.no_grad():
        tensor = torch.from_numpy(normalized).unsqueeze(0)  # add batch dim
        reconstruction = model(tensor)
        return float(torch.mean((tensor - reconstruction) ** 2).item())


def process_line(
    raw_line: str,
    sanitizer: ISOSanitizer,
    vectorizer: LogVectorizer,
    model: LogAutoencoder,
    injector: WazuhSocketInjector,
    tau: float,
    scaler_mean: List[float],
    scaler_var: List[float],
) -> Optional[float]:
    """Process one raw log line; inject an alert if it scores above ``tau``.

    Returns the reconstruction error, or ``None`` if the line was dropped
    (corrupt, no telemetry, or a transient processing error).
    """
    event = sanitizer.process_event(raw_line)
    if event is None:
        return None

    try:
        vector = vectorizer.extract_vector(event)
        mse = reconstruction_error(model, vector, scaler_mean, scaler_var)
    except Exception:  # noqa: BLE001 - one bad event must not stop the pipeline
        return None

    if mse > tau:
        agent_id = str(event.get("agent", {}).get("id", "000"))
        process_name = str(event.get("data", {}).get("process_name") or "unknown")
        sent = injector.send_alert(agent_id, ANOMALY_RULE_ID, mse, process_name)
        logger.info(
            "anomaly: agent=%s process=%s score=%.4f tau=%.4f sent=%s",
            agent_id,
            process_name,
            mse,
            tau,
            sent,
        )

    return mse


def run_realtime_inference(config_path: str = "config/global_config.json") -> None:
    """Run the never-ending real-time anomaly-detection loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = load_config(config_path)

    sanitizer = ISOSanitizer()
    vectorizer = LogVectorizer()
    injector = WazuhSocketInjector(config["wazuh_socket_path"])
    model = load_inference_model(config)

    tau = float(config["anomaly_threshold_tau"])
    scaler_mean = config["scaler_mean"]
    scaler_var = config["scaler_var"]

    logger.info(
        "detector started: archives=%s tau=%.4f model=%s",
        config["wazuh_archives_path"],
        tau,
        config["model_save_path"],
    )

    processed = 0
    anomalies = 0
    for raw_line in tail_wazuh_archives(config["wazuh_archives_path"]):
        mse = process_line(
            raw_line,
            sanitizer,
            vectorizer,
            model,
            injector,
            tau,
            scaler_mean,
            scaler_var,
        )
        processed += 1
        if mse is not None and mse > tau:
            anomalies += 1
        if processed % HEARTBEAT_EVERY == 0:
            logger.info("heartbeat: processed=%d anomalies=%d", processed, anomalies)


if __name__ == "__main__":
    run_realtime_inference()
