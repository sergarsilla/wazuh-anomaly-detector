"""Tests for the feature vectorizer."""

import numpy as np

from src.features import HASH_SIZE, INPUT_DIM, LogVectorizer


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
