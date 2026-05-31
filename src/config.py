"""Helpers for reading and writing the global JSON configuration.

A single place to load and persist ``config/global_config.json`` so the training,
evaluation and inference code never duplicate JSON-handling logic.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the configuration dictionary from ``config_path``."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config_path: str, config: Dict[str, Any]) -> None:
    """Write ``config`` back to ``config_path`` as pretty-printed JSON."""
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def update_config(config_path: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``updates`` into the on-disk config and persist it. Returns the result."""
    config = load_config(config_path)
    config.update(updates)
    save_config(config_path, config)
    return config
