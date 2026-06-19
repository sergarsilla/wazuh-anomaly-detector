"""Tests for the archive-profiling summary helper."""

from __future__ import annotations

from scripts.profile_archives import summarize


def test_summarize_counts_coverage_decoders_and_fields() -> None:
    events = [
        (True, {"decoder": {"name": "sshd"}, "data": {"srcip": "203.0.113.5", "user": "root"}}),
        (True, {"decoder": {"name": "sshd"}, "data": {"srcip": "203.0.113.6"}}),
        (True, {"decoder": {"name": "audit"}}),            # no data block
        (False, {}),                                        # unparseable line
    ]
    report = summarize(events, top=10)

    assert report["total"] == 3
    assert report["unparsed"] == 1
    assert report["with_data"] == 2
    assert report["coverage"] == 2 / 3
    assert dict(report["top_decoders"])["sshd"] == 2
    # srcip appears in both data blocks, user in one.
    fields = dict(report["top_data_fields"])
    assert fields["srcip"] == 2
    assert fields["user"] == 1
    # An example is kept for a decoder that had a data block, not for one without.
    assert "sshd" in report["examples"]
    assert "audit" not in report["examples"]


def test_summarize_handles_no_events() -> None:
    report = summarize([], top=5)
    assert report == {
        "total": 0,
        "unparsed": 0,
        "with_data": 0,
        "coverage": 0.0,
        "top_decoders": [],
        "top_data_fields": [],
        "examples": {},
    }
