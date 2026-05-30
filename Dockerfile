# Anomaly-detection inference engine for Wazuh.
# CPU-only image: all inference runs on CPU, so we install the CPU build of torch
# to keep the image small (no multi-GB CUDA layers).
FROM python:3.13-slim

# Avoid interactive prompts and keep Python output unbuffered for live logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (better layer caching). torch comes from the CPU index.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
 && pip install --no-cache-dir "numpy>=1.26" "scikit-learn>=1.4" "pandas>=2.2"

# Application code. config/ and models/ are mounted as volumes at runtime so the
# trained model and tuned thresholds can change without rebuilding the image.
COPY src/ ./src/
COPY training/ ./training/

# Run the real-time inference loop.
CMD ["python", "-m", "src.pipeline"]
