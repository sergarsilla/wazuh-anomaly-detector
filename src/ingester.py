"""Asynchronous ``tail -f`` reader for the Wazuh ``archives.json`` log.

The ingester continuously yields newly appended log lines. It is designed to
survive the two events that normally break naive ``tail`` implementations in a
production SIEM:

* The log file not existing yet (the reader waits without raising).
* Log rotation (``logrotate``), detected by watching the file's inode. When the
  inode on disk no longer matches the one we hold open, we transparently switch
  to the new file without losing events.

Nothing in this module should ever raise on transient I/O conditions: a missing
file, a rotated descriptor or a partially written line must never stop the
process.
"""

from __future__ import annotations

import os
import time
from typing import Generator, Optional, TextIO

# How long to wait (seconds) before polling again when no new data is available.
# Kept as a module-level constant so tests can shrink it via monkeypatching.
POLL_INTERVAL_SECONDS: float = 0.5


def _try_open(file_path: str) -> Optional[TextIO]:
    """Open ``file_path`` for reading, returning ``None`` if it is not there yet."""
    try:
        return open(file_path, "r", encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _has_rotated(file_path: str, current_inode: int) -> bool:
    """Return ``True`` if the file on disk has a different inode than the one held.

    A changed inode means ``logrotate`` (or any rename/recreate) has replaced the
    file, so the open descriptor now points at the old, rotated-away file.
    """
    try:
        return os.stat(file_path).st_ino != current_inode
    except OSError:
        # File momentarily absent (e.g. mid-rotation). Keep the current handle
        # and retry on the next iteration rather than treating it as a rotation.
        return False


def tail_wazuh_archives(file_path: str) -> Generator[str, None, None]:
    """Yield log lines from ``file_path`` as they are appended, forever.

    On first open the read pointer is positioned at the end of the file so only
    *new* events are emitted (existing history is skipped). After a rotation the
    freshly created file is read from its beginning, since it only contains new
    events.

    Args:
        file_path: Path to the Wazuh ``archives.json`` log.

    Yields:
        Each newly appended line, with the trailing newline stripped.
    """
    file_handle: Optional[TextIO] = None
    current_inode: Optional[int] = None
    is_first_open: bool = True

    try:
        while True:
            # (Re)open the file whenever we do not currently hold a handle.
            if file_handle is None:
                file_handle = _try_open(file_path)
                if file_handle is None:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                current_inode = os.fstat(file_handle.fileno()).st_ino
                if is_first_open:
                    # Skip pre-existing history: start tailing from the end.
                    file_handle.seek(0, os.SEEK_END)
                    is_first_open = False
                # On a post-rotation reopen we deliberately read from offset 0.

            position = file_handle.tell()
            line = file_handle.readline()

            if line and line.endswith("\n"):
                yield line.rstrip("\n")
                continue

            # Either EOF or a partially written line: rewind so we re-read it
            # once it is complete, then decide whether the file was rotated.
            file_handle.seek(position)

            if current_inode is not None and _has_rotated(file_path, current_inode):
                file_handle.close()
                file_handle = None
                continue

            time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        if file_handle is not None:
            file_handle.close()
