"""Profile a Wazuh ``archives.json`` log to see what telemetry is available.

The detector only vectorises events that carry a useful ``data`` block, so a
large share of the archive may be ignored. This read-only script summarises the
log to guide which extra event types are worth capturing (PLAN.md WS2):

    * total events parsed (and unparseable lines skipped)
    * how many carry a non-empty ``data`` block, and the coverage percentage
    * the most common ``decoder.name`` values
    * the most frequent fields seen inside ``data``
    * one example ``data`` block per top decoder

It never writes anything. Run it against a copy or the live archive:

    python scripts/profile_archives.py [archives.json] [--top N] [--limit N]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, Tuple


def iter_events(path: str, limit: int | None = None) -> Iterator[Tuple[bool, Dict[str, Any]]]:
    """Yield ``(ok, event)`` per line; ``ok`` is False for unparseable lines."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for count, line in enumerate(handle):
            if limit is not None and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield True, json.loads(line)
            except json.JSONDecodeError:
                yield False, {}


def summarize(events: Iterable[Tuple[bool, Dict[str, Any]]], top: int = 15) -> Dict[str, Any]:
    """Aggregate event/decoder/field statistics from parsed archive events."""
    total = 0
    unparsed = 0
    with_data = 0
    decoders: Counter[str] = Counter()
    data_fields: Counter[str] = Counter()
    examples: Dict[str, Dict[str, Any]] = {}

    for ok, event in events:
        if not ok:
            unparsed += 1
            continue
        total += 1
        decoder = str(((event.get("decoder") or {}).get("name")) or "(none)")
        decoders[decoder] += 1

        data = event.get("data")
        if isinstance(data, dict) and data:
            with_data += 1
            data_fields.update(data.keys())
            examples.setdefault(decoder, data)

    coverage = (with_data / total) if total else 0.0
    return {
        "total": total,
        "unparsed": unparsed,
        "with_data": with_data,
        "coverage": coverage,
        "top_decoders": decoders.most_common(top),
        "top_data_fields": data_fields.most_common(top),
        "examples": {name: examples[name] for name, _ in decoders.most_common(top) if name in examples},
    }


def print_report(report: Dict[str, Any]) -> None:
    """Render the summary as a readable text report."""
    print(f"Parsed events:      {report['total']}")
    print(f"Unparseable lines:  {report['unparsed']}")
    print(
        f"With data block:    {report['with_data']} "
        f"({report['coverage']:.1%} of parsed events)"
    )

    print("\nTop decoders (decoder.name):")
    for name, count in report["top_decoders"]:
        print(f"  {count:>8}  {name}")

    print("\nMost frequent fields inside data:")
    for field, count in report["top_data_fields"]:
        print(f"  {count:>8}  {field}")

    print("\nExample data block per top decoder:")
    for name, sample in report["examples"].items():
        print(f"  [{name}] {json.dumps(sample, ensure_ascii=False)[:200]}")


def main(argv: list[str]) -> None:
    path = "/var/ossec/logs/archives/archives.json"
    top = 15
    limit: int | None = None

    positional: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--top" and i + 1 < len(argv):
            top = int(argv[i + 1]); i += 2; continue
        if arg == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1]); i += 2; continue
        positional.append(arg); i += 1
    if positional:
        path = positional[0]

    print(f"Profiling {path}" + (f" (first {limit} lines)" if limit else "") + "\n")
    print_report(summarize(iter_events(path, limit), top=top))


if __name__ == "__main__":
    main(sys.argv[1:])
