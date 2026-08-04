"""
GCS fetch for the unit-count-resolver service.

Path pattern is the SAME one xtimator-3d uploads to and xtimator-page-classifier
reads from:

    gs://{bucket}/{organization_slug}/{project_id_lower}/{plan_id_lower}/floor_plan.PDF

NOTE (reported in MF_REBUILD_REPORT.md): the build prompt described this path as
`gs://.../{project}/{plan}/floor_plan.PDF`, but the real layout has an
`organization_slug` segment in FRONT — see xtimator-3d/helper.py:1271-1273
(download_floorplan) and xtimator-page-classifier/app/gcs_client.py:38. The
classifier service is handed the slug by its caller; this service is not, because
its request body is {project_id, plan_id, user_id} per the spec. It therefore
derives the slug itself from user_id via the same query xtimator-3d uses
(shared.py:load_organization_slug), so the specified request body still works.
"""
import logging
from pathlib import Path

from google.cloud.storage import Client as CloudStorageClient

logger = logging.getLogger("unit_count_resolver.gcs")


def download_floorplan_pdf(credentials, organization_slug, project_id, plan_id, destination_path):
    """Download the plan's floor_plan.PDF from GCS to `destination_path`.

    Raises FileNotFoundError if the object does not exist. Returns the Path.
    """
    bucket_name = credentials["CloudStorage"]["bucket_name"]
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    gcs_url = f"gs://{bucket_name}/{blob_path}"

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    client = CloudStorageClient()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"PDF not found in GCS: {gcs_url}")

    logger.info(f"[UNIT_COUNTS] GCS download {gcs_url} -> {destination_path}")
    blob.download_to_filename(str(destination_path))
    return destination_path
