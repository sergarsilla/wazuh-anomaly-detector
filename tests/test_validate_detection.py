"""Tests for the acceptance-check corpus loader."""

from __future__ import annotations

import json

from scripts import validate_detection


def _write_corpus(tmp_path, payload) -> str:
    path = tmp_path / "validation_corpus.local.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_defaults_apply_without_a_local_corpus(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(validate_detection, "LOCAL_CORPUS_PATH", str(tmp_path / "absent.json"))
    benign, malicious, identity = validate_detection.load_corpora()
    assert benign == validate_detection.DEFAULT_BENIGN_COMMANDS
    assert malicious == validate_detection.DEFAULT_MALICIOUS_COMMANDS
    assert identity == validate_detection.DEFAULT_IDENTITY


def test_identity_override_replaces_only_named_fields(tmp_path, monkeypatch) -> None:
    """A partial identity keeps the placeholder for the fields it omits."""
    path = _write_corpus(
        tmp_path, {"identity": {"agent_id": "000", "srcuser": "operator"}}
    )
    monkeypatch.setattr(validate_detection, "LOCAL_CORPUS_PATH", path)
    _, _, identity = validate_detection.load_corpora()
    assert identity["agent_id"] == "000"
    assert identity["srcuser"] == "operator"
    assert identity["pwd"] == validate_detection.DEFAULT_IDENTITY["pwd"]


def test_unknown_identity_keys_are_ignored(tmp_path, monkeypatch) -> None:
    """The scoring event has a fixed shape; stray keys must not reach it."""
    path = _write_corpus(tmp_path, {"identity": {"agent_id": "000", "nonsense": "x"}})
    monkeypatch.setattr(validate_detection, "LOCAL_CORPUS_PATH", path)
    _, _, identity = validate_detection.load_corpora()
    assert set(identity) == set(validate_detection.DEFAULT_IDENTITY)


def test_identity_reaches_the_scored_event(tmp_path, monkeypatch) -> None:
    """The hashed categorical fields must carry the configured identity.

    Guards the defect this override exists for: scoring every command as an
    agent and user absent from the training data adds a constant reconstruction
    error to the whole corpus.
    """
    path = _write_corpus(
        tmp_path,
        {"identity": {"agent_id": "017", "agent_name": "worker", "pwd": "/srv"}},
    )
    monkeypatch.setattr(validate_detection, "LOCAL_CORPUS_PATH", path)
    _, _, identity = validate_detection.load_corpora()

    captured = {}

    class _Vectorizer:
        def extract_vector(self, event):
            captured.update(event)
            return validate_detection.LogVectorizer().extract_vector(event)

    class _Model:
        def __call__(self, tensor):
            return tensor

    dim = validate_detection.LogVectorizer().extract_vector({"data": {"command": "x"}}).shape[0]
    validate_detection._score(
        _Model(),
        _Vectorizer(),
        "/usr/bin/docker ps",
        [0.0] * dim,
        [1.0] * dim,
        identity,
    )
    assert captured["agent"] == {"id": "017", "name": "worker"}
    assert captured["data"]["pwd"] == "/srv"
    assert captured["data"]["srcuser"] == validate_detection.DEFAULT_IDENTITY["srcuser"]
