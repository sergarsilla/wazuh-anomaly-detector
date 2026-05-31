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
        assert injector.send_alert("001", 100100, 0.873, "suspicious_proc") is True

        raw = server.recv(65536).decode("utf-8")
        assert raw.startswith(QUEUE_PREFIX)

        body = json.loads(raw[len(QUEUE_PREFIX):])
        alert = body["anomaly_detector"]
        assert alert["agent_id"] == "001"
        assert alert["rule_id"] == 100100
        assert alert["process_name"] == "suspicious_proc"
        assert alert["anomaly_score"] == 0.873
    finally:
        server.close()


def test_send_alert_returns_false_when_socket_missing(tmp_path: Path) -> None:
    injector = WazuhSocketInjector(str(tmp_path / "does_not_exist"))
    assert injector.send_alert("001", 100100, 0.5, "proc") is False
