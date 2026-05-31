"""Tests for the dataset exporter."""

import json
from pathlib import Path

import numpy as np

from src.features import INPUT_DIM
from training.export_dataset import build_vectors, export_dataset


def _write_archives(path: Path, count: int) -> None:
    lines = []
    for i in range(count):
        lines.append(
            json.dumps(
                {
                    "agent": {"id": f"{i:03d}"},
                    "data": {"command": f"ls -la /tmp/{i}", "process_name": "ls"},
                }
            )
        )
    # Add noise the exporter must skip: a corrupt line and a telemetry-free one.
    lines.append("{ broken json")
    lines.append(json.dumps({"agent": {"id": "999"}}))
    path.write_text("\n".join(lines) + "\n")


def test_build_vectors_skips_unusable_lines(tmp_path: Path) -> None:
    archives = tmp_path / "archives.json"
    _write_archives(archives, count=10)
    vectors = build_vectors(str(archives))
    # 10 valid events; the corrupt and dataless lines are dropped.
    assert vectors.shape == (10, INPUT_DIM)
    assert vectors.dtype == np.float32


def test_export_dataset_splits_train_and_validation(tmp_path: Path) -> None:
    archives = tmp_path / "archives.json"
    _write_archives(archives, count=100)
    out_dir = tmp_path / "data"

    total = export_dataset(str(archives), str(out_dir), validation_ratio=0.2)
    assert total == 100

    train = np.load(out_dir / "normal_dataset.npy")
    validation = np.load(out_dir / "validation_dataset.npy")
    assert train.shape[1] == INPUT_DIM
    assert validation.shape[1] == INPUT_DIM
    assert train.shape[0] + validation.shape[0] == 100
    assert validation.shape[0] == 20


def test_export_dataset_no_split_when_ratio_zero(tmp_path: Path) -> None:
    archives = tmp_path / "archives.json"
    _write_archives(archives, count=15)
    out_dir = tmp_path / "data"

    export_dataset(str(archives), str(out_dir), validation_ratio=0.0)
    assert (out_dir / "normal_dataset.npy").exists()
    assert not (out_dir / "validation_dataset.npy").exists()
