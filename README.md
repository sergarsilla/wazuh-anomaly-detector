# 🛡️ Wazuh Anomaly Detector

Real-time, unsupervised anomaly detection for the [Wazuh](https://wazuh.com/) SIEM,
powered by a PyTorch autoencoder. It tails Wazuh's raw event log, masks sensitive
data locally (ISO 27001), turns each event into a numeric feature vector, and flags
behaviour that deviates from the learned "normal" — injecting alerts straight back
into the Wazuh dashboard.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CPU-ee4c2c">
  <img alt="Tests" src="https://img.shields.io/badge/tests-48%20passing-3fb950">
  <img alt="Docker" src="https://img.shields.io/badge/docker-ready-2496ed">
</p>

---

## Table of Contents

- [How it works](#-how-it-works)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech stack](#-tech-stack)
- [Project structure](#-project-structure)
- [Quick start (Docker)](#-quick-start-docker)
- [Local development](#-local-development)
- [Configuration](#-configuration)
- [Monitoring](#-monitoring)
- [Testing](#-testing)
- [Privacy & ISO 27001](#-privacy--iso-27001)
- [License](#-license)

---

## 🧠 How it works

The system runs in **two phases**:

| Phase | When | What it does |
|-------|------|--------------|
| **Training** | One-off (re-run periodically) | Learns what *normal* traffic looks like and produces a trained model + a dynamic threshold. |
| **Detection** | Continuously (24/7) | Scores live events; anything above the threshold is reported to Wazuh. |

The core idea: an autoencoder is trained **only on normal traffic**. It learns to
reconstruct normal events with low error, but struggles to reconstruct anything it
has never seen (an attack), producing a high **reconstruction error**. When that
error exceeds the dynamic threshold `τ` (a high percentile of the normal-traffic
error, p99.9 by default), the event is flagged as an anomaly.

Training reads the **full rotated history** Wazuh keeps under
`/var/ossec/logs/archives/` (the gzipped daily archives), not just the current
day, so the model learns a representative "normal" and over-flags far less.

---

## ✨ Features

- **Unsupervised** — no labelled attack data required; learns from normal traffic.
- **Explainable alerts** — each alert carries a calibrated `severity` (how many times
  over `τ`) plus the top per-feature contributors in plain language, so a human (and a
  downstream LLM triage layer) sees *why* an event is anomalous, not just *how much*.
- **Privacy-first (ISO 27001)** — public IPs, emails, JWTs, API keys, credentials and
  private keys are irreversibly masked *before* any analysis.
- **Resilient ingestion** — `tail -f` style reader that survives missing files,
  partial lines and log rotation (inode tracking).
- **History-aware training** — trains on the gzipped rotated archive history, not just
  the current day.
- **CPU-only** — lightweight inference, no GPU needed.
- **Native Wazuh integration** — alerts are injected into the manager's queue socket
  and surfaced via a custom rule in the dashboard.
- **Fully Dockerized** — one image, two commands: train and detect.

---

## 🏗 Architecture

```
TRAINING (one-off)
  archives.json ─► sanitize ─► vectorize (64-D) ─► train autoencoder ─► evaluate (τ)
                                                                  │
                                                                  ▼
                                                          model.pt + config

DETECTION (24/7)
  archives.json ─► sanitize ─► vectorize ─► autoencoder ─► MSE > τ ? ─► inject alert
       (tail -f)     (PII)       (64-D)     (reconstruct)               (Wazuh socket)
                                                                              │
                                                                              ▼
                                                                       Wazuh dashboard
```

---

## 🧰 Tech stack

- **Python 3.11+**
- **PyTorch** (CPU build) — the autoencoder
- **NumPy / scikit-learn / pandas** — numeric processing & scaling
- **Docker / Docker Compose** — packaging and deployment

---

## 📁 Project structure

```
wazuh-anomaly-detector/
├── config/
│   └── global_config.json      # Paths, hyperparameters, threshold & scaler params
├── src/
│   ├── ingester.py             # tail -f reader (rotation-safe)
│   ├── sanitizer.py            # ISO 27001 PII/secret masking
│   ├── features.py             # Shannon entropy + hashing trick → 64-D vector
│   ├── model.py                # PyTorch autoencoder (64→8→64)
│   ├── injector.py             # Sends alerts to the Wazuh UNIX socket
│   ├── pipeline.py             # Real-time inference loop
│   └── config.py               # Config load/save helpers
├── training/
│   ├── export_dataset.py       # Build .npy datasets from the archive history (gz-aware)
│   ├── train.py                # Train the autoencoder + fit scaler
│   ├── evaluate.py             # Compute τ (high percentile of error)
│   └── run_training.py         # One-shot: export → train → evaluate
├── scripts/
│   ├── calibrate_threshold.py  # Compare τ candidates + false-positive rates (advisory)
│   └── profile_archives.py     # Profile archives.json: event/decoder/field coverage
├── rules/
│   └── local_rules.xml         # Custom Wazuh rule (id 100100)
├── models/                     # Trained model.pt lands here (generated)
├── tests/                      # 48 pytest cases
├── .github/workflows/ci.yml    # CI: install CPU torch + run pytest
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Quick start (Docker)

Run this **on the Wazuh manager host**. The detector reads the local archives log and
writes to the local Wazuh socket via bind mounts.

> ⚠️ **Order matters.** The detector needs a trained `model.pt`, which is created in the
> training step. Train *before* starting the detector.

```bash
# 1. Install Docker (if needed) and build the image
sudo apt install -y docker.io docker-compose-v2
git clone <YOUR_REPO_URL> wazuh-anomaly-detector
cd wazuh-anomaly-detector
# The live config (with the trained scaler/tau) is not tracked; create it once.
cp config/global_config.example.json config/global_config.json
sudo docker compose build

# 2. Let Wazuh accumulate NORMAL traffic. Training uses the whole archive
#    directory by default (the rotated *.json.gz history), so a few days/weeks
#    of retained archives is plenty — check there is data:
sudo du -sh /var/ossec/logs/archives

# 3. Train the model (one-off: export → train → evaluate).
#    --build keeps the trainer image in sync with the current code; it is
#    required when re-training after a code change. Cap memory on a large
#    history with MAX_EVENTS, or point ARCHIVES_PATH at a single file/glob.
sudo docker compose run --rm --build trainer
# e.g. cap to 1M events:  sudo docker compose run --rm -e MAX_EVENTS=1000000 trainer

# 4. Install the Wazuh rule and restart the manager (one-off)
sudo bash -c 'cat rules/local_rules.xml >> /var/ossec/etc/rules/local_rules.xml'
sudo systemctl restart wazuh-manager

# 5. Start the detector (runs 24/7, restarts on reboot)
sudo docker compose up -d detector
```

> 💡 Wazuh only writes to `archives.json` when `<logall_json>yes</logall_json>` is set
> in `/var/ossec/etc/ossec.conf` (then restart the manager). This logs **every** event,
> so watch disk usage (`df -h`) and consider enabling it only during capture windows.

---

## 💻 Local development

For working on the code (tests, experimentation) without Docker:

```bash
# Create a virtual environment (uv recommended)
uv venv --python 3.13 .venv

# Install CPU-only torch (avoids pulling multi-GB CUDA deps) + the rest
VIRTUAL_ENV=.venv uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
VIRTUAL_ENV=.venv uv pip install "numpy>=1.26" "scikit-learn>=1.4" "pandas>=2.2" "pytest>=8.0"

# Create the (untracked) live config from the template
cp config/global_config.example.json config/global_config.json

# Run the training steps manually
.venv/bin/python training/export_dataset.py /path/to/archives.json ./data
.venv/bin/python training/train.py config/global_config.json data/normal_dataset.npy
.venv/bin/python training/evaluate.py config/global_config.json data/validation_dataset.npy

# Run the real-time pipeline
.venv/bin/python -m src.pipeline
```

---

## ⚙️ Configuration

All runtime settings live in [`config/global_config.json`](config/global_config.json):

| Key | Description |
|-----|-------------|
| `wazuh_archives_path` | Path to Wazuh's `archives.json` |
| `wazuh_socket_path` | Path to the Wazuh manager queue socket |
| `model_save_path` | Where the trained weights are stored |
| `input_dim` / `latent_dim` | Vector size (64) and bottleneck size (8) |
| `learning_rate` / `batch_size` / `epochs` | Training hyperparameters |
| `anomaly_threshold_tau` | Dynamic alarm threshold (written by `evaluate.py`) |
| `tau_percentile` | Percentile of normal-traffic error used as the threshold (default 99.9) |
| `alert_cooldown_seconds` | Per-signature throttle between repeat alerts, in seconds (default 1800) |
| `scaler_mean` / `scaler_var` | Normalization params (written by `train.py`) |

> `anomaly_threshold_tau`, `scaler_mean` and `scaler_var` start empty/zero and are
> filled automatically during training — you don't edit them by hand.

---

## 📊 Monitoring

| To check… | Command |
|-----------|---------|
| Is the detector running? | `sudo docker compose ps` |
| Live detector logs | `sudo docker compose logs -f detector` |
| CPU / RAM usage | `sudo docker stats wazuh-anomaly-detector` |
| Are alerts reaching Wazuh? | `sudo tail -f /var/ossec/logs/alerts/alerts.json \| grep anomaly_detector` |
| Stop everything | `sudo docker compose down` |
| Retrain later | repeat step 3, then `sudo docker compose restart detector` |

---

## 🧪 Testing

```bash
.venv/bin/python -m pytest tests/
```

The suite (48 tests) covers PII masking, log-rotation handling, feature extraction,
per-feature anomaly explanations, the autoencoder, the end-to-end training/threshold
pipeline, socket injection, the gz-aware history exporter and the threshold-calibration
and archive-profiling helpers.

---

## 🔒 Privacy & ISO 27001

Sensitive data never leaves the local sanitizer. `src/sanitizer.py` irreversibly masks
public IPv4 addresses, emails, JWTs, API keys, explicit credentials and private key
blocks before events are vectorized. Private/internal IPs are preserved as behavioural
signal. Note that the raw `archives.json` on disk still contains PII and falls under
your log-protection and retention policies.

---

## 📄 License

Licensed under the Apache License 2.0 — see [`LICENSE`](LICENSE).
