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

# Wazuh queue-message prefix: queue id "1" and our custom location tag.
QUEUE_PREFIX: str = "1:anomaly_detector:"


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
    ) -> bool:
        """Send a single anomaly alert. Returns ``True`` on success.

        Socket failures (manager down, wrong path, permissions) are caught and
        reported as ``False`` so a transient problem never crashes the pipeline.
        """
        incident = {
            "anomaly_detector": {
                "agent_id": agent_id,
                "rule_id": rule_id,
                "anomaly_score": round(float(anomaly_score), 6),
                "process_name": process_name,
            }
        }
        payload = f"{QUEUE_PREFIX}{json.dumps(incident)}"

        client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            client.sendto(payload.encode("utf-8"), self.socket_path)
            return True
        except OSError:
            return False
        finally:
            client.close()
