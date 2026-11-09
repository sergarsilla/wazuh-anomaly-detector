"""Build training/validation datasets from Wazuh archive logs.

Reads captured ``archives.json`` data, runs every event through the exact same
sanitizer + vectorizer used in production, and writes the resulting feature
vectors as ``.npy`` arrays ready for ``train.py`` / ``evaluate.py``.

Wazuh rotates ``archives.json`` daily into ``YYYY/MMM/*.json.gz`` (when
``logall_json`` is on), so the live file only ever holds the current day. To
learn a representative "normal", the input may be a single file, a glob, **or a
directory**: a directory is walked recursively and every ``*.json`` and
gzip-compressed ``*.json.gz`` archive is read (the rotated history), while the
``*.log``/``*.sum`` plain-text archives Wazuh also writes are ignored. The input
should contain *normal* traffic only, since the autoencoder learns what "normal"
looks like.

Usage:
    python training/export_dataset.py <archives_path> <output_dir> [--val-ratio 0.2]
    # <archives_path> may be archives.json, /var/ossec/logs/archives, or a glob
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import re
import sys
from typing import IO, Iterator, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import INPUT_DIM, LogVectorizer  # noqa: E402
from src.sanitizer import ISOSanitizer  # noqa: E402


def resolve_archive_files(archives_path: str) -> List[str]:
    """Expand a file, directory or glob into a sorted list of JSON archive files.

    A directory is searched recursively for the live ``*.json`` and the rotated,
    gzip-compressed ``*.json.gz`` archives (``*.log``/``*.sum`` are ignored), so
    training can use the full history under ``/var/ossec/logs/archives/YYYY/MMM/``
    rather than only the current day.

    Files of identical byte size are de-duplicated: Wazuh exposes the current day
    both as the live ``archives.json`` and as its not-yet-compressed dated copy,
    and counting it twice would over-weight a single day. Two distinct multi-MB
    logs sharing an exact size is implausible.
    """
    if os.path.isdir(archives_path):
        matches = glob.glob(os.path.join(archives_path, "**", "*.json"), recursive=True)
        matches += glob.glob(os.path.join(archives_path, "**", "*.json.gz"), recursive=True)
    elif any(ch in archives_path for ch in "*?["):
        matches = glob.glob(archives_path, recursive=True)
    else:
        matches = [archives_path]

    unique_by_size: "dict[int, str]" = {}
    for path in sorted(set(matches)):
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        unique_by_size.setdefault(size, path)
    return sorted(unique_by_size.values(), key=_chronological_key)


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_ROTATED_ARCHIVE = re.compile(r"(\d{4})/(" + "|".join(_MONTHS) + r")/ossec-archive-(\d{2})\.json")


def _chronological_key(path: str) -> Tuple[int, int, int, str]:
    """Order rotated archives by date; anything else (the live file) sorts last."""
    match = _ROTATED_ARCHIVE.search(path)
    if match is None:
        return (9999, 0, 0, path)
    year, month, day = match.groups()
    return (int(year), _MONTHS.index(month) + 1, int(day), path)


def _open_archive(path: str) -> IO[str]:
    """Open an archive file as text, decompressing transparently if gzipped."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_archive_lines(archives_path: str) -> Iterator[str]:
    """Yield non-empty, stripped lines across every resolved archive file.

    A rotated archive can be truncated — Wazuh was still writing when the day
    rolled over, or the disk filled — and decompression then raises partway
    through. Reading the whole history must not hinge on every file being
    intact, so a damaged archive yields whatever it produced before the fault
    and the walk moves on to the next one, with a note on stderr.
    """
    for file_path in resolve_archive_files(archives_path):
        try:
            with _open_archive(file_path) as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield line
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            print(f"Skipping unreadable archive {file_path}: {exc}", file=sys.stderr)


def build_vectors(archives_path: str, max_events: Optional[int] = None) -> np.ndarray:
    """Read archive logs and return an (n_events, INPUT_DIM) float32 array.

    ``archives_path`` may be a single file, a glob, or a directory (whose rotated
    ``*.json.gz`` history is read and decompressed too). Corrupt lines and events
    without useful telemetry are skipped silently, like the live pipeline does.
    """
    sanitizer = ISOSanitizer()
    vectorizer = LogVectorizer()
    vectors: List[np.ndarray] = []

    for line in iter_archive_lines(archives_path):
        event = sanitizer.process_event(line)
        if event is None:
            continue
        vectors.append(vectorizer.extract_vector(event))
        if max_events is not None and len(vectors) >= max_events:
            break

    if not vectors:
        return np.empty((0, INPUT_DIM), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def export_dataset(
    archives_path: str,
    output_dir: str,
    validation_ratio: float = 0.2,
    max_events: Optional[int] = None,
    seed: int = 42,
) -> int:
    """Export feature vectors from an archives log into train/val ``.npy`` files.

    Args:
        archives_path: Path to the raw Wazuh ``archives.json`` capture.
        output_dir: Directory where the ``.npy`` files are written.
        validation_ratio: Most recent fraction of events held out for validation
            (0 disables the split and writes only ``normal_dataset.npy``).
        max_events: Optional cap on the number of events to process.
        seed: RNG seed for shuffling the training set, for reproducible runs.

    Returns:
        The total number of feature vectors exported.
    """
    vectors = build_vectors(archives_path, max_events=max_events)
    total = int(vectors.shape[0])
    os.makedirs(output_dir, exist_ok=True)

    if total == 0:
        print("No usable events found; nothing was written.")
        return 0

    normal_path = os.path.join(output_dir, "normal_dataset.npy")

    if validation_ratio <= 0.0 or total < 2:
        np.save(normal_path, vectors)
        print(f"Exported {total} vectors -> {normal_path}")
        return total

    split = int(total * (1.0 - validation_ratio))
    split = max(1, min(split, total - 1))  # keep at least one row in each set
    # Vectors arrive in time order, so the validation set is the most recent
    # traffic: tau must hold on days the model never saw, as it will in
    # production. A random split calibrates it on days it was trained on.
    train_set, validation_set = vectors[:split].copy(), vectors[split:]
    np.random.default_rng(seed).shuffle(train_set)

    validation_path = os.path.join(output_dir, "validation_dataset.npy")
    np.save(normal_path, train_set)
    np.save(validation_path, validation_set)
    print(
        f"Exported {total} vectors: "
        f"{train_set.shape[0]} -> {normal_path}, "
        f"{validation_set.shape[0]} -> {validation_path}"
    )
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build .npy datasets from archives.json")
    parser.add_argument("archives_path", help="Path to the raw Wazuh archives.json")
    parser.add_argument("output_dir", help="Directory to write the .npy files into")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction held out for validation (0 to disable). Default: 0.2",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on number of events to process",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_dataset(
        args.archives_path,
        args.output_dir,
        validation_ratio=args.val_ratio,
        max_events=args.max_events,
    )
