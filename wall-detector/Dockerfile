FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    SD_MODEL_PATH=/models/stable-diffusion-v1-4 \
    CONTROLNET_PATH=/models/Finetuned_Iter_17/controlnet \
    LORA_PATH=/models/Finetuned_Iter_17/unet

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Bake model weights into the image (CI-CD push).
WORKDIR /models

RUN mkdir -p /models/stable-diffusion-v1-4
COPY models/stable-diffusion-v1-4 /models/stable-diffusion-v1-4
RUN mkdir -p /models/Finetuned_Iter_17
COPY models/Finetuned_Iter_17 /models/Finetuned_Iter_17

WORKDIR /app

COPY . /app
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]
