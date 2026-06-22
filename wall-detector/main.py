import os
import sys
from pathlib import Path
import logging
from ruamel.yaml import YAML
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from google.cloud.storage import Client as CloudStorageClient

from wall_detector import WallDetector


def upload_segmented_walls(segmented_path, plan_id, project_id, credentials, page_number, organization_slug):
    client = CloudStorageClient()
    blob_object_name = segmented_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{page_number}/{blob_object_name}"
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(segmented_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"


def respond_with_JSON_payload(credentials, image: Image, project_id, plan_id, user_id, page_number, organization_slug):
    destination_path = Path("/tmp/wall_detected.png")
    destination_path = destination_path.parent.joinpath(project_id).joinpath(plan_id).joinpath(user_id).joinpath(str(page_number)).joinpath(destination_path.name)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination_path)

    gcs_bucket_URL = upload_segmented_walls(destination_path, plan_id, project_id, credentials, str(page_number).zfill(4), organization_slug)
    return JSONResponse(
        content=dict(gcs_bucket_URL=gcs_bucket_URL),
        status_code=200,
        media_type="application/json",
    )


def enable_logging_on_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s"}',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )


def load_gcp_credentials() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("config/gcp.yaml", 'r') as f:
        credentials = yaml.load(f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials["service_account_key"]

    return credentials


def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("hyperparameters.yaml", 'r') as f:
        hyperparameters = yaml.load(f)

    return hyperparameters


app = FastAPI(title="Wall Detector (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
wall_detector = WallDetector()

@app.post("/detect_wall")
async def detect_wall(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    organization_slug = parameters.get("organization_slug") or body.get("organization_slug")
    page_number = parameters.get("page_number") or body.get("page_number")
    mask = parameters.get("mask") or body.get("mask")
    logging.info("SYSTEM: Received a Wall Detection Request")

    hyperparameters = load_hyperparameters()
    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{str(page_number).zfill(4)}/{CREDENTIALS["CloudStorage"]["blob_name"]}"
    blob = bucket.blob(blob_path)
    destination_path = Path("/tmp/floor_plan.png")
    destination_path = destination_path.parent.joinpath(project_id).joinpath(plan_id).joinpath(user_id).joinpath(str(page_number)).joinpath(destination_path.name)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_path)

    mask_offset = None
    if mask:
        mask_offset = dict(horizontal=mask.get("horizontal", 0), vertical=mask.get("vertical", 0))
    image = wall_detector.detect(destination_path, hyperparameters, mask_offset=mask_offset)

    logging.info("SYSTEM: Wall Detection Completed")
    return respond_with_JSON_payload(CREDENTIALS, image, project_id, plan_id, user_id, page_number, organization_slug)
