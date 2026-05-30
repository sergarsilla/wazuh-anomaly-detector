"""Build training/validation datasets from a raw Wazuh archives log.

Reads a captured ``archives.json`` file, runs every event through the exact same
sanitizer + vectorizer used in production, and writes the resulting feature
vectors as ``.npy`` arrays ready for ``train.py`` / ``evaluate.py``.

Optionally splits the data into a training set and a validation set. The input
should contain *normal* traffic only, since the autoencoder learns what "normal"
looks like.

Usage:
    python training/export_dataset.py <archives.json> <output_dir> [--val-ratio 0.2]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import INPUT_DIM, LogVectorizer  # noqa: E402
from src.sanitizer import ISOSanitizer  # noqa: E402


def build_vectors(archives_path: str, max_events: Optional[int] = None) -> np.ndarray:
    """Read an archives log and return an (n_events, INPUT_DIM) float32 array.

    Corrupt lines and events without useful telemetry are skipped silently, the
    same way the live pipeline drops them.
    """
    sanitizer = ISOSanitizer()
    vectorizer = LogVectorizer()
    vectors: List[np.ndarray] = []

    with open(archives_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
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
        validation_ratio: Fraction of events held out for validation (0 disables
            the split and writes only ``normal_dataset.npy``).
        max_events: Optional cap on the number of events to process.
        seed: RNG seed for the shuffle, for reproducible splits.

    Returns:
        The total number of feature vectors exported.
    """
    vectors = build_vectors(archives_path, max_events=max_events)
    total = int(vectors.shape[0])
    os.makedirs(output_dir, exist_ok=True)

    if total == 0:
        print("No usable events found; nothing was written.")
        return 0

    # Shuffle so the train/val split is not biased by time ordering.
    rng = np.random.default_rng(seed)
    rng.shuffle(vectors)

    normal_path = os.path.join(output_dir, "normal_dataset.npy")

    if validation_ratio <= 0.0 or total < 2:
        np.save(normal_path, vectors)
        print(f"Exported {total} vectors -> {normal_path}")
        return total

    split = int(total * (1.0 - validation_ratio))
    split = max(1, min(split, total - 1))  # keep at least one row in each set
    train_set, validation_set = vectors[:split], vectors[split:]

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
