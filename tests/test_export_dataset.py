"""Tests for the dataset exporter."""

import gzip
import json
import os
from pathlib import Path

import numpy as np

from src.features import INPUT_DIM
from training.export_dataset import (
    build_vectors,
    export_dataset,
    iter_archive_lines,
    resolve_archive_files,
)


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


def _write_gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_directory_resolves_json_and_gz_but_ignores_log_and_sum(tmp_path: Path) -> None:
    (tmp_path / "archives.json").write_text('{"a":1}\n')
    _write_gz(tmp_path / "2026" / "Jun" / "ossec-archive-19.json.gz", '{"b":2}\n')
    # Plain-text and checksum siblings Wazuh also writes must be ignored.
    (tmp_path / "2026" / "Jun" / "ossec-archive-19.log").write_text("ignored\n")
    (tmp_path / "2026" / "Jun" / "ossec-archive-19.log.sum").write_text("deadbeef\n")

    resolved = resolve_archive_files(str(tmp_path))
    assert any(p.endswith("archives.json") for p in resolved)
    assert any(p.endswith(".json.gz") for p in resolved)
    assert all(not p.endswith((".log", ".sum")) for p in resolved)


def test_same_size_current_day_is_deduplicated(tmp_path: Path) -> None:
    # The live file and its dated, not-yet-compressed copy are byte-identical.
    same = '{"event":"today"}\n'
    (tmp_path / "archives.json").write_text(same)
    dated = tmp_path / "2026" / "Jun" / "ossec-archive-20.json"
    dated.parent.mkdir(parents=True, exist_ok=True)
    dated.write_text(same)
    _write_gz(tmp_path / "2026" / "Jun" / "ossec-archive-19.json.gz", '{"event":"yesterday"}\n')

    resolved = resolve_archive_files(str(tmp_path))
    # One of the two identical copies is dropped; the distinct gz day stays.
    assert len(resolved) == 2


def test_iter_archive_lines_reads_through_gzip_and_skips_blanks(tmp_path: Path) -> None:
    (tmp_path / "archives.json").write_text('{"x":1}\n\n')
    _write_gz(tmp_path / "2026" / "Jun" / "ossec-archive-19.json.gz", '{"y":2}\n')

    lines = list(iter_archive_lines(str(tmp_path)))
    assert '{"x":1}' in lines
    assert '{"y":2}' in lines
    assert "" not in lines
