import os
import sys
import logging
from ruamel.yaml import YAML
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from google.cloud.storage import Client as CloudStorageClient
from google.cloud import secretmanager

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
    with open("config/gcp.yaml", 'r') as f:
        credentials = yaml.load(f)

    return credentials


def download_secrets(credentials):
    def _download_secret(secret_manager_client, secret_key, secret_url):
        response = secret_manager_client.access_secret_version(request={"name": secret_url})
        secret_data = json.loads(response.payload.data)
        with open(credentials[secret_key], 'w') as f:
            json.dump(secret_data, f)
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    secret_manager_client = secretmanager.SecretManagerServiceClient()
    executor = ThreadPoolExecutor(max_workers=5)
    download_secret_futures = list()
    for secret_key, secret_url in credentials["SecretManager"].items():
        download_secret_futures.append(executor.submit(_download_secret, secret_manager_client, secret_key, secret_url))
    [future.result() for future in download_secret_futures]

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials["service_compute_account_key"]


def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("config/hyperparameters.yaml", 'r') as f:
        hyperparameters = yaml.load(f)

    return hyperparameters


app = FastAPI(title="Wall Detector (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
download_secrets(CREDENTIALS)

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
