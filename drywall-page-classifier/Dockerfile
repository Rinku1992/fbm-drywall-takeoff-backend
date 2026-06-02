# Dockerfile — drywall-page-classifier
#
# CUDA 12.4 runtime image. Model weights are BAKED INTO THE IMAGE for fastest
# cold-start. Update model = rebuild image.
#
# Build context must contain a valid trained checkpoint at:
#     checkpoints/best.pth

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ─── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# pip with --break-system-packages because the base image's Python is system-managed
RUN python -m pip install --upgrade pip setuptools wheel --break-system-packages

# ─── PyTorch (CUDA 12.4 wheels) ───────────────────────────────
# Dedicated layer so it caches across rebuilds (the wheels are huge).
RUN pip install --no-cache-dir --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 torchvision==0.20.1

# ─── App dependencies ─────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

WORKDIR /app

# ─── Application code ─────────────────────────────────────────
COPY app /app/app

# ─── Model checkpoint (baked into image) ──────────────────────
# Must be present in build context.
COPY checkpoints/best.pth /app/checkpoints/best.pth
RUN test -f /app/checkpoints/best.pth \
        || (echo "FATAL: checkpoints/best.pth not found in build context" && exit 1)

# ─── Service config + service-account keys ────────────────────
# config/gcp.yaml is read at startup by app/config.py:_load_gcp_credentials.
# The two SA JSON files MUST be staged at the build-context root by the
# operator before each `docker build` (same convention as drywall-takeoff-3d).
COPY config /app/config
COPY prj-fbm-drywall-dev-*.json /app/
RUN test -f /app/config/gcp.yaml \
        || (echo "FATAL: config/gcp.yaml not found in build context" && exit 1) \
    && test -n "$(ls /app/prj-fbm-drywall-dev-*.json 2>/dev/null)" \
        || (echo "FATAL: no prj-fbm-drywall-dev-*.json files staged in build context" && exit 1)

# Cloud Run injects PORT
ENV PORT=8080

# workers=1 — one process per container, one GPU per container, concurrency
# managed by Cloud Run via containerConcurrency in service.yaml.
CMD exec uvicorn app.main:app \
        --host 0.0.0.0 \
        --port ${PORT} \
        --workers 1 \
        --log-level info
