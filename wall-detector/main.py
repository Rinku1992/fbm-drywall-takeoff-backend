import os
import sys
import logging
from ruamel.yaml import YAML
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from google.cloud.storage import Client as CloudStorageClient

from wall_detector import WallDetector


def respond_with_image_payload(image: Image):
    image_path = "/tmp/wall_detected.png"
    image.save(image_path)

    return FileResponse(
        image_path,
        media_type="image/png",
        filename="wall_detected.png"
    )


def enable_logging_on_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s"}',
        stream=sys.stdout
    )


def load_gcp_credentials() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("gcp.yaml", 'r') as f:
        credentials = yaml.load(f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials["service_account_key"]

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
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Wall Detection Request")

    hyperparameters = load_hyperparameters()
    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{str(index).zfill(2)}/{CREDENTIALS["CloudStorage"]["blob_name"]}"
    blob = bucket.blob(blob_path)
    image_path = "/tmp/floor_plan.png"
    blob.download_to_filename(image_path)

    wall_detector = WallDetector()
    image = wall_detector.detect(image_path, hyperparameters)

    logging.info("SYSTEM: Wall Detection Completed")
    return respond_with_image_payload(image)