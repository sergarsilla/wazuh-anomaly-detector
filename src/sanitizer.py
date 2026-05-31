"""ISO 27001 compliant log sanitizer.

Before any event leaves this machine towards the feature vectorizer, every piece
of sensitive information (PII, secrets, public network addresses) is irreversibly
replaced with a constant marker. The original values never reach the model.

The masking is deliberately one-way: we substitute fixed placeholders, we do not
encrypt or tokenize, so the originals cannot be recovered downstream.
"""

from __future__ import annotations

import ipaddress
import json
import re
from typing import Any, Dict, Optional

# Constant markers used for irreversible substitution.
MASKED_CREDENTIAL: str = "[MASKED_CREDENTIAL]"
MASKED_IP: str = "10.0.0.0"
MASKED_EMAIL: str = "user@masked.local"


class ISOSanitizer:
    """Mask sensitive data in raw Wazuh events for ISO 27001 compliance."""

    def __init__(self) -> None:
        # PEM private key blocks (multi-line). Matched first so their inner
        # content is never picked up by the narrower patterns below.
        self._private_key_pattern = re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        )
        # JSON Web Tokens: three base64url segments separated by dots.
        self._jwt_pattern = re.compile(
            r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
        )
        # Explicit credentials passed as system parameters, e.g. ``--password=foo``
        # or ``token: abc123``. Only the value is masked; the key name is kept.
        self._credential_pattern = re.compile(
            r"(?i)\b(password|passwd|pwd|secret|token|api[-_]?key|apikey|"
            r"access[-_]?key|auth)\b\s*[:=]\s*[^\s\"',;]+"
        )
        # Standard email addresses.
        self._email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )
        # IPv4 candidates; validated and filtered to public addresses only.
        self._ipv4_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    def _mask_public_ip(self, match: re.Match[str]) -> str:
        """Replace a matched IPv4 with the marker only if it is publicly routable.

        Private/loopback/link-local addresses are behavioural signal and are kept;
        public addresses are PII and get masked.
        """
        candidate = match.group(0)
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            return candidate  # Not a valid IP (e.g. a version string): leave it.
        return MASKED_IP if address.is_global else candidate

    def sanitize_string(self, text: str) -> str:
        """Irreversibly replace every sensitive token found in ``text``.

        Patterns are applied from the broadest/most specific (private keys) to the
        narrowest (IPs) so that, for example, a key embedded in a parameter is not
        partially masked twice.
        """
        text = self._private_key_pattern.sub(MASKED_CREDENTIAL, text)
        text = self._jwt_pattern.sub(MASKED_CREDENTIAL, text)
        text = self._credential_pattern.sub(
            lambda m: f"{m.group(1)}={MASKED_CREDENTIAL}", text
        )
        text = self._email_pattern.sub(MASKED_EMAIL, text)
        text = self._ipv4_pattern.sub(self._mask_public_ip, text)
        return text

    def _sanitize_recursive(self, value: Any) -> Any:
        """Walk a nested structure, sanitizing every string leaf."""
        if isinstance(value, str):
            return self.sanitize_string(value)
        if isinstance(value, dict):
            return {key: self._sanitize_recursive(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._sanitize_recursive(item) for item in value]
        return value

    def process_event(self, raw_json: str) -> Optional[Dict[str, Any]]:
        """Deserialize a raw Wazuh event, sanitize its ``data`` block, and return it.

        Args:
            raw_json: A single raw JSON line from the Wazuh archives log.

        Returns:
            The parsed event with every string inside its ``data`` block
            sanitized, or ``None`` if the line is corrupt or carries no useful
            telemetry (no non-empty ``data`` block).
        """
        try:
            event = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            # Corrupt or non-JSON line: skip it without crashing the pipeline.
            return None

        if not isinstance(event, dict):
            return None

        data = event.get("data")
        if not isinstance(data, dict) or not data:
            return None

        # Skip Wazuh's own meta-events (our injected alerts and the
        # active-response logs they trigger); re-ingesting them would make the
        # detector flag its own activity in an endless loop.
        location = str(event.get("location", ""))
        if (
            location == "anomaly_detector"
            or location.startswith("/var/ossec/")
            or "anomaly_detector" in data
        ):
            return None

        event["data"] = self._sanitize_recursive(data)
        return event
