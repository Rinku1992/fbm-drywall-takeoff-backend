# scripts/config.py
"""
Configuration for the drywall-page-classifier Cloud Run service.

Constants here match the training config (`config_v2.py`) in the
floorplan_classifier repo, so model weights load correctly.
"""
import os
from pathlib import Path

import yaml


# ─── Model ────────────────────────────────────────────────────
# Trained checkpoint baked into the Docker image at this path.
# (See Dockerfile — COPY checkpoints/best.pth /app/checkpoints/best.pth)
MODEL_CHECKPOINT_PATH = Path(
    os.environ.get("MODEL_CHECKPOINT_PATH", "/app/checkpoints/best.pth")
)

# Must match training (`config_v2.IMG_SIZE`)
IMG_SIZE = 640
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TILE_FRACTION = 0.60  # 5-view tile-max corner crop size

# Class ordering — MUST match training (alphabetical ImageFolder order)
MULTICLASS_CLASS_NAMES = [
    "electrical_plan",
    "elevation_plan",
    "floor_plan",
    "foundation_plan",
    "non_floor_plan",
    "roof_plan",
]
NUM_CLASSES = len(MULTICLASS_CLASS_NAMES)
FLOOR_PLAN_IDX = MULTICLASS_CLASS_NAMES.index("floor_plan")


# ─── Inference behaviour ──────────────────────────────────────
# Floor plan confidence threshold — proven on test set: 0.85 caught all 5
# false positives without demoting any true floor plans.
# Lower → more aggressive (catches borderline FPs but risks demoting real ones)
# Higher → less aggressive (preserves recall but more FPs)
# Set to 0.0 to disable the threshold entirely.
FP_THRESHOLD = float(os.environ.get("FP_THRESHOLD", "0.85"))


# Map our internal class names (snake_case) to the labels the rest of the
# pipeline already uses (UPPER_SNAKE, matches the existing planType Literal
# in drywall-takeoff-3d/prompts.py).
SNAKE_TO_LLM_LABEL = {
    "electrical_plan":  "ELECTRICAL_PLAN",
    "elevation_plan":   "ELEVATION_PLAN",
    "floor_plan":       "FLOOR_PLAN",
    "foundation_plan":  "FOUNDATION_PLAN",
    "non_floor_plan":   "NOT_ARCHITECTURAL_PLAN",   # important: maps to "NOT_ARCHITECTURAL"
    "roof_plan":        "ROOF_PLAN",
}


# ─── Service limits ──────────────────────────────────────────
# Cloud Run default request body limit is 32MB. Reject requests
# clearly larger than this to fail fast (Cloud Run will return 413
# itself, but we add our own check for clearer errors).
MAX_IMAGES_PER_REQUEST = int(os.environ.get("MAX_IMAGES_PER_REQUEST", "10"))


# ─── GCS configuration ────────────────────────────────────────
# Bucket where drywall-takeoff-3d uploads PDFs. Path pattern:
#     gs://{GCS_BUCKET_NAME}/{project_id_lower}/{plan_id_lower}/floor_plan.PDF
GCS_BUCKET_NAME = "drywall-takeoff-artifacts-dev"


# ─── Service account bootstrap ────────────────────────────────
# Loaded at module-import time so google.cloud.storage.Client() picks up
# the right credentials from GOOGLE_APPLICATION_CREDENTIALS.
def _load_gcp_credentials():
    """
    Read gcp.yaml from /app/config/gcp.yaml (the same convention as
    drywall-takeoff-3d) and export GOOGLE_APPLICATION_CREDENTIALS so
    google-cloud-storage uses the drywall service account.
    """
    gcp_yaml_path = Path("/app/config/gcp.yaml")
    if not gcp_yaml_path.exists():
        # Fallback for local dev — running from drywall-page-classifier/
        gcp_yaml_path = Path("config/gcp.yaml")
    if gcp_yaml_path.exists():
        with open(gcp_yaml_path, "r") as f:
            creds = yaml.safe_load(f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds["service_drywall_account_key"]


_load_gcp_credentials()


# ─── Preprocessing configuration ──────────────────────────────
# Folder where the service renders PNG pages during a request.
# Cloud Run has 32 GiB of ephemeral /tmp; a 100-page PDF at DPI 300 is well
# under that. /tmp is reset between instances, so no cleanup is required.
PREPROCESSING_OUTPUT_DIR = Path(os.environ.get("PREPROCESSING_OUTPUT_DIR", "/tmp/classifier_pages"))

# Max parallel page-render workers. Match drywall-takeoff-3d which uses 10.
PREPROCESSING_MAX_WORKERS = int(os.environ.get("PREPROCESSING_MAX_WORKERS", "10"))
