"""Tests for the feature vectorizer."""

import numpy as np

from src.features import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FEATURES,
    HASH_SIZE,
    INPUT_DIM,
    LogVectorizer,
    explain_anomaly,
    feature_layout,
)


def test_entropy_zero_for_uniform_string() -> None:
    vectorizer = LogVectorizer()
    assert vectorizer.calculate_shannon_entropy("aaaa") == 0.0


def test_entropy_one_bit_for_two_equal_symbols() -> None:
    vectorizer = LogVectorizer()
    assert vectorizer.calculate_shannon_entropy("ab") == 1.0


def test_entropy_empty_string() -> None:
    vectorizer = LogVectorizer()
    assert vectorizer.calculate_shannon_entropy("") == 0.0


def test_hashing_is_fixed_size_and_deterministic() -> None:
    vectorizer = LogVectorizer()
    first = vectorizer.apply_feature_hashing("process_name", "bash")
    second = vectorizer.apply_feature_hashing("process_name", "bash")
    assert len(first) == HASH_SIZE
    assert first == second  # Deterministic across calls.
    # Exactly one bucket is set, to +1 or -1.
    nonzero = [v for v in first if v != 0.0]
    assert len(nonzero) == 1
    assert nonzero[0] in (1.0, -1.0)


def test_extract_vector_shape_and_dtype() -> None:
    vectorizer = LogVectorizer()
    event = {
        "agent": {"id": "001"},
        "data": {
            "command": "rm -rf /",
            "process_name": "bash",
            "user": "root",
        },
    }
    vector = vectorizer.extract_vector(event)
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.shape == (INPUT_DIM,)


def test_extract_vector_handles_missing_data() -> None:
    vectorizer = LogVectorizer()
    vector = vectorizer.extract_vector({"data": {}})
    assert vector.shape == (INPUT_DIM,)
    assert vector.dtype == np.float32


def test_feature_layout_spans_continuous_then_categorical() -> None:
    layout = feature_layout()
    # Three single-dimension continuous features, then HASH_SIZE per categorical.
    expected = len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FIELDS)
    assert len(layout) == expected
    for name, start, end in layout[: len(CONTINUOUS_FEATURES)]:
        assert end - start == 1
    for name, start, end in layout[len(CONTINUOUS_FEATURES):]:
        assert end - start == HASH_SIZE
    # Spans are contiguous and stop before the zero padding.
    assert layout[0][1] == 0
    assert layout[-1][2] == len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FIELDS) * HASH_SIZE


def test_explain_anomaly_ranks_top_contributors() -> None:
    per_dim = np.zeros(INPUT_DIM, dtype=np.float32)
    normalized = np.zeros(INPUT_DIM, dtype=np.float32)
    # command_entropy (index 1) dominates, observed above the benign mean.
    per_dim[1] = 9.0
    normalized[1] = 2.5
    # process_name hash bucket (index 3) is a weaker contributor.
    per_dim[3] = 1.0

    top = explain_anomaly(normalized, per_dim, top_k=3)

    assert top[0]["feature"] == "command_entropy"
    assert top[0]["message"] == "command entropy above normal"
    assert top[0]["contribution_pct"] == 90.0
    assert top[1]["feature"] == "process_name"
    assert top[1]["message"] == "process name pattern unusual"


def test_explain_anomaly_direction_below_normal() -> None:
    per_dim = np.zeros(INPUT_DIM, dtype=np.float32)
    normalized = np.zeros(INPUT_DIM, dtype=np.float32)
    per_dim[0] = 4.0  # command_length
    normalized[0] = -1.7  # observed below the benign mean

    top = explain_anomaly(normalized, per_dim, top_k=1)
    assert top[0]["message"] == "command length below normal"


def test_explain_anomaly_empty_when_no_error() -> None:
    zeros = np.zeros(INPUT_DIM, dtype=np.float32)
    assert explain_anomaly(zeros, zeros) == []
