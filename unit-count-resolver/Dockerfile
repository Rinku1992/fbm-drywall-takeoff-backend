# Dockerfile — unit-count-resolver (master plan D5a)
#
# Structural pattern follows xtimator-page-classifier/Dockerfile, minus everything
# GPU/model related: this service has no checkpoint and needs no CUDA. Base image
# and system deps follow xtimator-3d/Dockerfile, which is the closest match for
# the libraries the lifted helpers use.
#
# The build context MUST contain config/gcp.yaml (Cloud SQL + Vertex service
# account key paths). It is NOT committed to the repo — the operator stages it
# before `docker build`, the same convention xtimator-page-classifier uses.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# ─── Application code ─────────────────────────────────────────
COPY app /app/app

# ─── Service config (staged by the operator, not in git) ──────
COPY config /app/config
RUN test -f /app/config/gcp.yaml \
        || (echo "FATAL: config/gcp.yaml not found in build context" && exit 1)

EXPOSE 8080

# workers=1 — the resolve job runs in this process's thread pool via
# BackgroundTasks, so a single worker keeps the job and its logs together.
CMD exec uvicorn app.main:app \
        --host 0.0.0.0 \
        --port ${PORT} \
        --workers 1 \
        --log-level info
