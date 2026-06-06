"""Feature extraction: turn a sanitized Wazuh event into a fixed 64-D vector.

The model only understands numbers, so every event must become a fixed-length
numeric vector. This module builds that vector from:

* **Continuous behavioural features** — command length, Shannon entropy (a proxy
  for obfuscation / randomness in the command) and the number of system calls.
* **Hashed categorical features** — high-cardinality strings such as the process
  name or user are mapped into a small fixed array using the *hashing trick*,
  which avoids maintaining an ever-growing vocabulary.

The output is always a ``float32`` array of length exactly ``INPUT_DIM`` (64),
zero-padded deterministically when the assembled features are shorter.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np

INPUT_DIM: int = 64
HASH_SIZE: int = 8

# Categorical fields hashed into the vector. Each contributes HASH_SIZE elements.
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "process_name",
    "parent_name",
    "user",
    "host_id",
)

# Candidate keys to look up each logical field inside a Wazuh ``data`` block.
# Wazuh's schema varies by decoder, so we try several common names.
_FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "process_name": ("process_name", "process", "exe", "comm"),
    "parent_name": ("parent_name", " parent", "pcomm", "ppid"),
    "user": ("user", "srcuser", "dstuser", "euid", "uid"),
    "host_id": ("host_id", "hostname"),
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


def has_process_telemetry(data: Dict[str, Any]) -> bool:
    """Return ``True`` if ``data`` carries usable process/command telemetry."""
    if not isinstance(data, dict):
        return False
    return any(
        data.get(key) is not None and str(data.get(key)).strip()
        for key in _TELEMETRY_FIELDS
    )


def read_process_name(data: Dict[str, Any]) -> str:
    """Best-effort human-readable process name from a Wazuh ``data`` block.

    Tries the same aliases the vectorizer uses; falls back to ``"unknown"`` when
    none resolve (e.g. an event identified only by its command line).
    """
    if isinstance(data, dict):
        for alias in _FIELD_ALIASES["process_name"]:
            value = data.get(alias)
            if value is not None and str(value).strip():
                return str(value)
    return "unknown"


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
        picks the sign, which helps cancel out collisions on average.
        """
        bucket = [0.0] * HASH_SIZE
        token = f"{category}_{value}".encode("utf-8")
        digest = int.from_bytes(hashlib.md5(token).digest(), "big")
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

    def _count_syscalls(self, data: Dict[str, Any], command: str) -> float:
        """Best-effort count of system calls in the event.

        Uses an explicit count/list field when present, otherwise falls back to
        the number of whitespace-separated tokens in the command.
        """
        for key in ("syscall_count", "syscalls", "syscall"):
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, list):
                return float(len(value))
        return float(len(command.split()))

    def extract_vector(self, event: Dict[str, Any]) -> np.ndarray:
        """Build the fixed-length 64-D ``float32`` feature vector for an event.

        Layout: 3 continuous features, then ``HASH_SIZE`` values per categorical
        field, then deterministic zero padding up to ``INPUT_DIM``.
        """
        data = event.get("data", {})
        if not isinstance(data, dict):
            data = {}

        command = str(data.get("command") or data.get("args") or "")

        continuous: List[float] = [
            float(len(command)),
            self.calculate_shannon_entropy(command),
            self._count_syscalls(data, command),
        ]

        vector: List[float] = list(continuous)
        for field in CATEGORICAL_FIELDS:
            value = self._read_field(data, field)
            vector.extend(self.apply_feature_hashing(field, value))

        # Deterministic zero-padding (and a safety truncation) to exactly INPUT_DIM.
        if len(vector) < INPUT_DIM:
            vector.extend([0.0] * (INPUT_DIM - len(vector)))
        vector = vector[:INPUT_DIM]

        return np.asarray(vector, dtype=np.float32)


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
