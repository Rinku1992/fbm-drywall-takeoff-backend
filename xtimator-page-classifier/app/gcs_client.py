"""
GCS client for downloading floor plan PDFs.

The classifier service downloads PDFs from the same canonical path that
drywall-takeoff-3d uses for upload:
    gs://drywall-takeoff-artifacts-dev/{project_id_lower}/{plan_id_lower}/floor_plan.PDF

Service account credentials are read from the file path in the env var
GOOGLE_APPLICATION_CREDENTIALS (set by config.py at startup from gcp.yaml).
"""
import logging
from pathlib import Path

from google.cloud import storage

from .config import GCS_BUCKET_NAME

logger = logging.getLogger("classifier.gcs")


def download_floorplan_pdf(
    project_id: str,
    plan_id: str,
    organization_slug: str,
    destination_path: Path = Path("/tmp/floor_plan.PDF"),
) -> Path:
    """
    Download the floor-plan PDF for a given project/plan from GCS.

    Path pattern matches drywall-takeoff-3d/helper.py:download_floorplan:
        gs://{bucket}/{project_id_lower}/{plan_id_lower}/floor_plan.PDF

    Returns the local Path the PDF was written to. Raises FileNotFoundError
    if the object doesn't exist in GCS.
    """
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    gcs_url = f"gs://{GCS_BUCKET_NAME}/{blob_path}"

    logger.info(f"GCS: downloading {gcs_url} → {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(
            f"PDF not found in GCS: {gcs_url}"
        )

    blob.download_to_filename(str(destination_path))
    size_mb = destination_path.stat().st_size / 1e6
    logger.info(f"GCS: downloaded {size_mb:.2f} MB")
    return destination_path
