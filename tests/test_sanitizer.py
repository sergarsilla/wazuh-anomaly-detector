"""Tests for the ISO 27001 sanitizer."""

import json

from src.sanitizer import (
    MASKED_CREDENTIAL,
    MASKED_EMAIL,
    MASKED_IP,
    ISOSanitizer,
)


def test_masks_public_ip_but_keeps_private() -> None:
    sanitizer = ISOSanitizer()
    result = sanitizer.sanitize_string("connect 8.8.8.8 from 192.168.1.10")
    assert MASKED_IP in result
    assert "8.8.8.8" not in result
    # Private address is behavioural signal and must be preserved.
    assert "192.168.1.10" in result


def test_masks_email() -> None:
    sanitizer = ISOSanitizer()
    result = sanitizer.sanitize_string("login from alice@example.com")
    assert "alice@example.com" not in result
    assert MASKED_EMAIL in result


def test_masks_jwt() -> None:
    sanitizer = ISOSanitizer()
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.s5hQ-abcDEF_123"
    result = sanitizer.sanitize_string(f"Authorization: Bearer {jwt}")
    assert jwt not in result
    assert MASKED_CREDENTIAL in result


def test_masks_credential_parameter() -> None:
    sanitizer = ISOSanitizer()
    result = sanitizer.sanitize_string("mysql -u root --password=SuperSecret123")
    assert "SuperSecret123" not in result
    assert MASKED_CREDENTIAL in result


def test_masks_private_key_block() -> None:
    sanitizer = ISOSanitizer()
    key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA1234567890\n"
        "-----END RSA PRIVATE KEY-----"
    )
    result = sanitizer.sanitize_string(f"key dump: {key}")
    assert "MIIEpAIBAAKCAQEA1234567890" not in result
    assert MASKED_CREDENTIAL in result


def test_process_event_sanitizes_data_block() -> None:
    sanitizer = ISOSanitizer()
    raw = json.dumps(
        {
            "agent": {"id": "001"},
            "data": {"command": "curl https://x --password=secret 8.8.8.8"},
        }
    )
    event = sanitizer.process_event(raw)
    assert event is not None
    command = event["data"]["command"]
    assert "secret" not in command
    assert "8.8.8.8" not in command
    # Identifying metadata outside the data block is left untouched.
    assert event["agent"]["id"] == "001"


def test_process_event_returns_none_on_corrupt_json() -> None:
    sanitizer = ISOSanitizer()
    assert sanitizer.process_event("{not valid json") is None


def test_process_event_returns_none_without_data_block() -> None:
    sanitizer = ISOSanitizer()
    assert sanitizer.process_event(json.dumps({"agent": {"id": "001"}})) is None
