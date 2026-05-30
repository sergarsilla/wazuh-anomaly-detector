"""Tests for the tail-style archives ingester, including log rotation."""

import os
import threading
import time
from pathlib import Path
from typing import List

import pytest

import src.ingester as ingester_mod
from src.ingester import tail_wazuh_archives


def _start_consumer(file_path: str, sink: List[str]) -> threading.Thread:
    """Run the (infinite) generator in a daemon thread, collecting lines."""

    def consume() -> None:
        for line in tail_wazuh_archives(file_path):
            sink.append(line)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    return thread


def test_tails_new_lines_and_survives_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Speed up polling so the test runs quickly.
    monkeypatch.setattr(ingester_mod, "POLL_INTERVAL_SECONDS", 0.02)

    log_path = tmp_path / "archives.json"
    log_path.write_text("")  # File exists but is empty.

    collected: List[str] = []
    _start_consumer(str(log_path), collected)
    time.sleep(0.1)  # Let the reader open the file and seek to the end.

    # Append two lines to the live file.
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("line1\n")
        handle.write("line2\n")
        handle.flush()
    time.sleep(0.15)

    # Simulate logrotate: rename the current file away, then create a new one.
    old_inode = os.stat(log_path).st_ino
    os.rename(str(log_path), str(tmp_path / "archives.json.1"))
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("line3\n")
        handle.flush()
    new_inode = os.stat(log_path).st_ino

    # Sanity check: the rotation genuinely changed the inode.
    assert old_inode != new_inode
    time.sleep(0.2)

    assert collected == ["line1", "line2", "line3"]
