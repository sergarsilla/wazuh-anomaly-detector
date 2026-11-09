"""Feature extraction: turn a sanitized Wazuh event into a fixed 64-D vector.

The model only understands numbers, so every event must become a fixed-length
numeric vector. Schema v2 builds that vector from three groups:

* **Continuous behavioural features** — command shape (length, Shannon entropy,
  token count, longest token, non-alphanumeric ratio), the time of day encoded
  on a circle, and whether the session was attached to a terminal.
* **Indicator flags** — booleans for the command patterns that characterise an
  intrusion: piping a download into an interpreter, ``/dev/tcp`` sockets,
  decoding a payload into a shell, stopping the security tooling, touching
  credentials or persistence, escaping a container. An autoencoder trained on
  benign traffic reconstructs these as zero, so a set flag produces a large,
  targeted reconstruction error. Schema v1 had no such feature, which is why a
  40-character ``curl … | bash`` scored an order of magnitude below a long but
  harmless ``docker exec``.
* **Hashed categorical features** — high-cardinality strings (process name,
  acting user, target user, agent, working directory) mapped into a small fixed
  array with the *hashing trick*, which avoids an ever-growing vocabulary.

The output is always a ``float32`` array of length exactly ``INPUT_DIM`` (64),
zero-padded deterministically when the assembled features are shorter.

Changing the layout invalidates any previously trained model, because dimension
*i* no longer means what the scaler and weights were fitted on. Every layout
change must therefore bump ``FEATURE_SCHEMA_VERSION``; training records that
number in the config and inference refuses to start on a mismatch.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shlex
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

INPUT_DIM: int = 64
HASH_SIZE: int = 8

# Bump whenever the meaning or order of any dimension changes. Guards against
# scoring events with a model trained on a different layout.
FEATURE_SCHEMA_VERSION: int = 2

# Continuous behavioural features, in vector order.
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "command_length",
    "command_entropy",
    "command_token_count",
    "longest_token_length",
    "non_alphanumeric_ratio",
    "hour_sin",
    "hour_cos",
    "has_terminal",
)

# Value a set indicator flag takes in the vector. Flags are deliberately *not*
# standardized (see ``training.train``), so this weight is what decides how much
# one indicator is worth against the continuous features: a set flag contributes
# INDICATOR_WEIGHT**2 to the squared error, the same as a 6-sigma deviation on a
# continuous feature. Stating it as a constant is the point — z-scoring a binary
# feature instead makes its weight 1/std, so a flag seen once in the training
# sample would count 800x more than one never seen at all, which is backwards.
INDICATOR_WEIGHT: float = 6.0

# Indicator flags, in vector order. Each contributes a single dimension, 0.0 when
# the pattern is absent and INDICATOR_WEIGHT when it matches.
INDICATOR_FLAGS: tuple[str, ...] = (
    "pipe_to_interpreter",
    "remote_fetch",
    "raw_socket_redirect",
    "encoded_payload",
    "interactive_shell_spawn",
    "security_tooling_disabled",
    "log_tampering",
    "credential_access",
    "persistence_config",
    "privileged_account_config",
    "container_escape",
    "package_management",
)

# Categorical fields hashed into the vector. Each contributes HASH_SIZE elements.
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "process_name",
    "acting_user",
    "target_user",
    "agent_id",
    "working_directory",
)

# Candidate keys to look up each logical field inside a Wazuh ``data`` block.
# Wazuh's schema varies by decoder, so we try several common names.
_FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "acting_user": ("srcuser", "user", "euid", "uid"),
    "target_user": ("dstuser", "target_user"),
    "working_directory": ("pwd", "cwd"),
}

# Fields whose presence marks an event as carrying the process/command
# behavioural telemetry this detector is built to score. An event lacking all of
# them (e.g. Wazuh's own internal/periodic events) collapses to an all-zero
# vector with no behavioural signal, so scoring it only produces noise. Both
# training and inference skip such events to keep their distributions identical.
_TELEMETRY_FIELDS: tuple[str, ...] = (
    "command",
    "args",
    "process_name",
    "process",
    "exe",
    "comm",
)

# Regexes backing INDICATOR_FLAGS, in the same order. They are deliberately
# narrow: a flag that fires on routine administration would be learnt as normal
# and stop carrying signal. Where a lone token would be ambiguous (curl is
# ordinary; curl piped into a shell is not) the discriminating context is part
# of the pattern, or split across two flags the model can combine.
_INTERPRETERS = r"(?:ba|z|k|da)?sh|python[0-9.]*|perl|ruby|node|php"
_SECURITY_SERVICES = r"wazuh[\w-]*|suricata|fail2ban|auditd|falco|clamav|apparmor|selinux"

_FLAG_PATTERNS: Dict[str, re.Pattern[str]] = {
    "pipe_to_interpreter": re.compile(
        rf"\|\s*(?:sudo\s+)?(?:/\S*/)?(?:{_INTERPRETERS})\b", re.IGNORECASE
    ),
    "remote_fetch": re.compile(
        r"\b(?:curl|wget|fetch)\b.*?\b(?:https?|ftp)://|\b(?:curl|wget)\s+-", re.IGNORECASE
    ),
    "raw_socket_redirect": re.compile(r"/dev/(?:tcp|udp)/", re.IGNORECASE),
    "encoded_payload": re.compile(
        r"\bbase64\s+(?:-d|--decode)\b|\bxxd\s+-r\b|\bopenssl\s+enc\s+-d\b|"
        r"\bbase64\.b64decode\b",
        re.IGNORECASE,
    ),
    "interactive_shell_spawn": re.compile(
        rf"\b(?:{_INTERPRETERS})\s+-i\b|\bnc\b[^|]*\s-\w*e\w*\s|\bsocat\b.*\bexec\b|"
        r"\bpty\.spawn\b",
        re.IGNORECASE,
    ),
    "security_tooling_disabled": re.compile(
        rf"\b(?:systemctl|service)\s+(?:stop|disable|mask)\b.*(?:{_SECURITY_SERVICES})|"
        rf"\b(?:pkill|killall)\b.*(?:{_SECURITY_SERVICES})",
        re.IGNORECASE,
    ),
    "log_tampering": re.compile(
        r"\b(?:rm|shred|truncate)\b[^|]*(?:/var/log|\.log)\b|"
        r"\bjournalctl\s+--(?:vacuum|rotate)|>\s*/var/log/",
        re.IGNORECASE,
    ),
    "credential_access": re.compile(
        r"/etc/(?:shadow|gshadow)\b|\bid_(?:rsa|ed25519|ecdsa|dsa)\b|\.ssh/id_|"
        r"\.pgpass\b|\.aws/credentials\b|\.docker/config\.json\b",
        re.IGNORECASE,
    ),
    "persistence_config": re.compile(
        r"\bcrontab\b|/etc/cron|\bsystemctl\s+enable\b|/etc/systemd/system|"
        r"/etc/rc\.local\b|\.bash(?:rc|_profile)\b",
        re.IGNORECASE,
    ),
    "privileged_account_config": re.compile(
        r"\b(?:useradd|adduser|userdel)\b|\busermod\b[^|]*-a?G\b|"
        r"\bauthorized_keys\b|/etc/sudoers|\bvisudo\b",
        re.IGNORECASE,
    ),
    "container_escape": re.compile(
        r"\bdocker\s+run\b[^|]*(?:--privileged\b|--pid[= ]host\b|--net(?:work)?[= ]host\b|"
        r"(?:-v|--volume)\s*/:)|\bnsenter\b",
        re.IGNORECASE,
    ),
    "package_management": re.compile(
        r"\b(?:apt|apt-get|yum|dnf|pacman|apk|zypper|snap)\b\s+"
        r"(?:install|remove|purge|upgrade|update|-S)\b",
        re.IGNORECASE,
    ),
}


def _scalar_text(value: Any) -> str:
    """Return a scalar field as stripped text; structured values yield ``""``."""
    if isinstance(value, (dict, list)) or value is None:
        return ""
    return str(value).strip()


def has_process_telemetry(data: Dict[str, Any]) -> bool:
    """Return ``True`` if ``data`` carries usable process/command telemetry.

    Only scalar fields count. Syscollector's process inventory carries
    ``process`` as an object describing a running process, not a command that
    was executed, and must not be scored as one.
    """
    if not isinstance(data, dict):
        return False
    return any(_scalar_text(data.get(key)) for key in _TELEMETRY_FIELDS)


def read_command(data: Dict[str, Any]) -> str:
    """Return the command line carried by a Wazuh ``data`` block."""
    if not isinstance(data, dict):
        return ""
    return str(data.get("command") or data.get("args") or "")


def read_process_name(data: Dict[str, Any]) -> str:
    """Best-effort executable name for a Wazuh ``data`` block.

    Prefers an explicit field when the decoder provides one. The Wazuh sudo
    decoder does not, so it falls back to the basename of the command's first
    token: ``/usr/bin/docker exec …`` becomes ``docker``. Schema v1 only tried
    the explicit fields and therefore reported ``unknown`` on every event.
    """
    if not isinstance(data, dict):
        return "unknown"
    for alias in ("process_name", "process", "exe", "comm"):
        value = _scalar_text(data.get(alias))
        if value:
            return os.path.basename(value)

    command = read_command(data).strip()
    if not command:
        return "unknown"
    try:
        tokens = shlex.split(command)
    except ValueError:
        # Unbalanced quotes in a raw log line: fall back to whitespace split.
        tokens = command.split()
    return os.path.basename(tokens[0]) if tokens else "unknown"


def read_user(data: Dict[str, Any]) -> str:
    """Best-effort acting user from a Wazuh ``data`` block.

    Tries the same aliases the vectorizer uses (``srcuser`` first for sudo-style
    events); returns ``""`` when none resolve.
    """
    if isinstance(data, dict):
        for alias in _FIELD_ALIASES["acting_user"]:
            value = data.get(alias)
            if value is not None and str(value).strip():
                return str(value)
    return ""


def read_agent_id(event: Dict[str, Any]) -> str:
    """Return the agent id from the event envelope.

    The agent lives at the top level of a Wazuh alert, not inside ``data``.
    Schema v1 looked for it in ``data`` only, so every event hashed to the same
    empty value and the model could not tell one host from another.
    """
    agent = event.get("agent") if isinstance(event, dict) else None
    if isinstance(agent, dict):
        for key in ("id", "name"):
            value = agent.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _hour_of_day(event: Dict[str, Any]) -> "float | None":
    """Return the hour (0-24, fractional) of the event, or None if unavailable."""
    raw = str(event.get("timestamp", "")).strip() if isinstance(event, dict) else ""
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.hour + moment.minute / 60.0


class LogVectorizer:
    """Convert sanitized events into fixed-length numeric feature vectors."""

    def calculate_shannon_entropy(self, text: str) -> float:
        """Return the Shannon entropy (base 2) of the characters in ``text``.

        H(X) = -sum( P(x_i) * log2 P(x_i) ). High entropy suggests randomness or
        obfuscation (e.g. encoded payloads); plain commands score low.
        """
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        entropy = 0.0
        for count in counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        return entropy

    def apply_feature_hashing(self, category: str, value: str) -> List[float]:
        """Hash a categorical value into a fixed ``HASH_SIZE`` vector.

        The string ``f"{category}_{value}"`` is hashed deterministically (MD5, so
        results are stable across processes unlike Python's salted ``hash``). The
        hash modulo ``HASH_SIZE`` selects the bucket; a separate bit of the hash
        picks the sign, which helps cancel out collisions on average. MD5 is a
        bucket assignment here, never a security decision.
        """
        bucket = [0.0] * HASH_SIZE
        token = f"{category}_{value}".encode("utf-8")
        digest = int.from_bytes(
            hashlib.md5(token, usedforsecurity=False).digest(), "big"
        )
        index = digest % HASH_SIZE
        sign = 1.0 if (digest // HASH_SIZE) % 2 == 0 else -1.0
        bucket[index] += sign
        return bucket

    def _read_field(self, data: Dict[str, Any], field: str) -> str:
        """Read a logical field from ``data`` trying its known aliases."""
        for alias in _FIELD_ALIASES.get(field, (field,)):
            if alias in data and data[alias] is not None:
                return str(data[alias])
        return ""

    def _command_shape(self, command: str) -> List[float]:
        """Return the continuous descriptors of the command string itself."""
        tokens = command.split()
        longest = float(max((len(token) for token in tokens), default=0))
        non_alphanumeric = sum(1 for char in command if not char.isalnum())
        ratio = non_alphanumeric / len(command) if command else 0.0
        return [
            float(len(command)),
            self.calculate_shannon_entropy(command),
            float(len(tokens)),
            longest,
            ratio,
        ]

    def _time_of_day(self, event: Dict[str, Any]) -> List[float]:
        """Encode the hour on a circle so 23:00 and 00:00 sit next to each other.

        Both components are zero when the event carries no usable timestamp,
        which is a state distinct from any real hour. Training and inference
        share this code, so the model sees that state consistently.
        """
        hour = _hour_of_day(event)
        if hour is None:
            return [0.0, 0.0]
        angle = 2.0 * math.pi * hour / 24.0
        return [math.sin(angle), math.cos(angle)]

    def _indicator_flags(self, command: str) -> List[float]:
        """Return one dimension per entry of ``INDICATOR_FLAGS``.

        A matching pattern yields ``INDICATOR_WEIGHT`` rather than 1.0 so the
        indicator carries its intended weight without being standardized.
        """
        return [
            INDICATOR_WEIGHT if _FLAG_PATTERNS[name].search(command) else 0.0
            for name in INDICATOR_FLAGS
        ]

    def extract_vector(self, event: Dict[str, Any]) -> np.ndarray:
        """Build the fixed-length 64-D ``float32`` feature vector for an event.

        Layout: the continuous features, then one dimension per indicator flag,
        then ``HASH_SIZE`` values per categorical field, then deterministic zero
        padding up to ``INPUT_DIM``.
        """
        data = event.get("data", {}) if isinstance(event, dict) else {}
        if not isinstance(data, dict):
            data = {}

        command = read_command(data)

        vector: List[float] = self._command_shape(command)
        vector.extend(self._time_of_day(event))
        vector.append(1.0 if str(data.get("tty") or "").strip() else 0.0)
        vector.extend(self._indicator_flags(command))

        categorical_values = {
            "process_name": read_process_name(data),
            "acting_user": self._read_field(data, "acting_user"),
            "target_user": self._read_field(data, "target_user"),
            "agent_id": read_agent_id(event),
            "working_directory": self._read_field(data, "working_directory"),
        }
        for field in CATEGORICAL_FIELDS:
            vector.extend(self.apply_feature_hashing(field, categorical_values[field]))

        # Deterministic zero-padding (and a safety truncation) to exactly INPUT_DIM.
        if len(vector) < INPUT_DIM:
            vector.extend([0.0] * (INPUT_DIM - len(vector)))
        vector = vector[:INPUT_DIM]

        return np.asarray(vector, dtype=np.float32)


# Logical feature -> direction-aware human-readable templates. Continuous
# features have a meaningful direction (the standardized value's sign says
# whether the observed value is above or below the benign mean); indicator flags
# and hashed categorical features do not, so they get a single template.
_FEATURE_DESCRIPTIONS: Dict[str, tuple[str, ...]] = {
    "command_length": ("command length above normal", "command length below normal"),
    "command_entropy": ("command entropy above normal", "command entropy below normal"),
    "command_token_count": (
        "command token count above normal",
        "command token count below normal",
    ),
    "longest_token_length": (
        "unusually long single argument",
        "unusually short arguments",
    ),
    "non_alphanumeric_ratio": (
        "unusual proportion of punctuation and operators",
        "unusually plain command text",
    ),
    "hour_sin": ("unusual time of day", "unusual time of day"),
    "hour_cos": ("unusual time of day", "unusual time of day"),
    "has_terminal": (
        "interactive terminal where automation is expected",
        "no terminal where an interactive session is expected",
    ),
    "pipe_to_interpreter": ("output piped into a shell or interpreter",),
    "remote_fetch": ("downloads content from a remote host",),
    "raw_socket_redirect": ("opens a raw network socket via /dev/tcp",),
    "encoded_payload": ("decodes an encoded payload",),
    "interactive_shell_spawn": ("spawns an interactive or remote shell",),
    "security_tooling_disabled": ("stops or disables security tooling",),
    "log_tampering": ("removes or truncates logs",),
    "credential_access": ("reads credential material",),
    "persistence_config": ("touches a persistence mechanism",),
    "privileged_account_config": ("touches privileged account configuration",),
    "container_escape": ("runs a container with host-level access",),
    "package_management": ("installs or removes system packages",),
    "process_name": ("process name pattern unusual",),
    "acting_user": ("acting user pattern unusual",),
    "target_user": ("target user pattern unusual",),
    "agent_id": ("host pattern unusual",),
    "working_directory": ("working directory pattern unusual",),
}


def feature_layout() -> List[tuple[str, int, int]]:
    """Return ``(logical_name, start, end)`` spans over the feature vector.

    Continuous features and indicator flags occupy a single dimension each; a
    categorical field occupies ``HASH_SIZE`` contiguous dimensions. The trailing
    deterministic zero-padding is intentionally omitted: those dimensions always
    reconstruct to ~0 error and carry no behavioural meaning.
    """
    layout: List[tuple[str, int, int]] = []
    index = 0
    for name in CONTINUOUS_FEATURES + INDICATOR_FLAGS:
        layout.append((name, index, index + 1))
        index += 1
    for field in CATEGORICAL_FIELDS:
        layout.append((field, index, index + HASH_SIZE))
        index += HASH_SIZE
    return layout


def explain_anomaly(
    normalized: np.ndarray,
    per_dim_error: np.ndarray,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Translate a per-dimension reconstruction error into top contributors.

    Aggregates the squared error of each logical feature (summing the
    ``HASH_SIZE`` dimensions of a hashed categorical field into one score),
    ranks them, and returns the ``top_k`` with a natural-language message and
    their share of the total error. This turns "the score is high" into "*why*
    it is high", which both a human analyst and a downstream LLM triage layer
    can act on.

    ``normalized`` is the standardized feature vector (its sign gives the
    direction for continuous features); ``per_dim_error`` is ``(x - x̂)²`` per
    dimension.
    """
    total = float(per_dim_error.sum())
    contributions: List[Dict[str, Any]] = []
    for name, start, end in feature_layout():
        error = float(per_dim_error[start:end].sum())
        if error <= 0.0:
            continue
        templates = _FEATURE_DESCRIPTIONS[name]
        if len(templates) == 2:
            message = templates[0] if float(normalized[start]) >= 0.0 else templates[1]
        else:
            message = templates[0]
        contributions.append(
            {
                "feature": name,
                "message": message,
                "contribution_pct": round(100.0 * error / total, 1) if total > 0 else 0.0,
            }
        )

    contributions.sort(key=lambda item: item["contribution_pct"], reverse=True)
    return contributions[:top_k]


def standardize(
    vector: np.ndarray,
    mean: list[float] | np.ndarray,
    var: list[float] | np.ndarray,
) -> np.ndarray:
    """Apply standard (z-score) scaling using precomputed mean and variance.

    Shared by training/evaluation/inference so the exact same transformation is
    applied everywhere. Works on a single vector or a 2-D batch via broadcasting.
    Zero-variance columns are guarded to avoid division by zero.
    """
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.sqrt(np.asarray(var, dtype=np.float32))
    std_array[std_array == 0.0] = 1.0
    return ((vector - mean_array) / std_array).astype(np.float32)
