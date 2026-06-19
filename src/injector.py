"""Alert injector that writes anomalies directly into Wazuh's queue socket.

When the pipeline detects an anomaly it sends a message to the Wazuh manager's
local UNIX datagram socket. The manager's analysisd then decodes it, matches it
against the custom rules in ``rules/local_rules.xml`` and surfaces it in the
dashboard like any other alert.

Wazuh's queue protocol expects messages of the form ``<queue>:<location>:<json>``.
We use queue ``1`` and location ``anomaly_detector``.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, List, Optional

# Wazuh queue-message prefix: queue id "1" and our custom location tag.
QUEUE_PREFIX: str = "1:anomaly_detector:"

# Cap the embedded command so a pathological line cannot blow past Wazuh's
# queue-message size limit.
MAX_COMMAND_LEN: int = 1024


class WazuhSocketInjector:
    """Send anomaly alerts to the Wazuh manager via its UNIX datagram socket."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path

    def send_alert(
        self,
        agent_id: str,
        rule_id: int,
        anomaly_score: float,
        process_name: str,
        *,
        command: str = "",
        user: str = "",
        timestamp: str = "",
        agent_name: str = "",
        severity: Optional[float] = None,
        top_features: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Send a single anomaly alert. Returns ``True`` on success.

        The optional enrichment fields (``command``, ``user``, ``timestamp``,
        ``agent_name``) give a downstream consumer — a human analyst or an LLM
        triage layer reading ``alerts.json`` — the context needed to judge the
        anomaly. The command is expected to be already sanitized by the caller
        (no raw PII) and is truncated to ``MAX_COMMAND_LEN``.

        ``severity`` is the score relative to the threshold (how many times over
        tau) and ``top_features`` the dimensions that drove the anomaly, both
        added to the payload only when provided so they explain *why* the event
        is anomalous, not just how large the score is.

        Socket failures (manager down, wrong path, permissions) are caught and
        reported as ``False`` so a transient problem never crashes the pipeline.
        """
        incident: Dict[str, Any] = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "rule_id": rule_id,
            "anomaly_score": round(float(anomaly_score), 6),
            "process_name": process_name,
            "user": user,
            "command": command[:MAX_COMMAND_LEN],
            "event_timestamp": timestamp,
        }
        if severity is not None:
            incident["severity"] = round(float(severity), 2)
        if top_features:
            incident["top_features"] = top_features
            # Flat, human-readable summary for the rule description / dashboard;
            # Wazuh references scalar fields far more cleanly than array elements.
            incident["explanation"] = "; ".join(
                str(feature.get("message", "")) for feature in top_features
            )
        payload = f"{QUEUE_PREFIX}{json.dumps({'anomaly_detector': incident})}"

        client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            client.sendto(payload.encode("utf-8"), self.socket_path)
            return True
        except OSError:
            return False
        finally:
            client.close()
