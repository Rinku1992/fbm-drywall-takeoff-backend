import os
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import logging
from ruamel.yaml import YAML
from PIL import Image
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from google.cloud.storage import Client as CloudStorageClient
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)

from wall_detector import WallDetector


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)


def _verify_model_weights_or_die():
    """Fail fast if required model files are not available at startup."""
    sd_root = os.getenv("SD_MODEL_PATH", "/models/stable-diffusion-v1-4")
    controlnet_root = os.getenv("CONTROLNET_PATH", "/models/Finetuned_Iter_17/controlnet")
    lora_root = os.getenv("LORA_PATH", "/models/Finetuned_Iter_17/unet")

    required = [
        f"{sd_root}/model_index.json",
        f"{sd_root}/unet/diffusion_pytorch_model.safetensors",
        f"{sd_root}/vae/diffusion_pytorch_model.safetensors",
        f"{sd_root}/text_encoder/model.safetensors",
        f"{controlnet_root}/config.json",
        f"{controlnet_root}/diffusion_pytorch_model.safetensors",
        f"{lora_root}/adapter_model.safetensors",
        f"{lora_root}/adapter_config.json",
    ]

    missing = [p for p in required if not Path(p).exists()]
    if missing:
        msg = (
            "Model files are missing at startup. Check that model weights "
            "are baked into the Docker image. Missing files:\n  - "
            + "\n  - ".join(missing)
        )
        logging.error(msg)
        raise RuntimeError(msg)

    logging.info("SYSTEM: Model file validation passed")


def upload_segmented_walls(segmented_path, plan_id, project_id, credentials, page_number, organization_slug):
    client = CloudStorageClient()
    blob_object_name = segmented_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{page_number}/{blob_object_name}"
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(segmented_path)
    return f"gs://{credentials['CloudStorage']['bucket_name']}/{blob_path}"


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
        force=True,
    )


def load_gcp_credentials() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("config/gcp.yaml", "r") as f:
        credentials = yaml.load(f)

    return credentials


def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("config/hyperparameters.yaml", "r") as f:
        hyperparameters = yaml.load(f)

    return hyperparameters


enable_logging_on_stdout()


wall_detector = None
_model_lock = threading.Lock()
_inference_lock = asyncio.Lock()
_model_ready = False


def get_wall_detector():
    global wall_detector

    if wall_detector is not None:
        return wall_detector

    with _model_lock:
        if wall_detector is None:
            logging.info("SYSTEM: Loading Wall Detector model...")

            _verify_model_weights_or_die()

            wall_detector = WallDetector(
                ckpt_path=os.getenv("CONTROLNET_PATH", "/models/Finetuned_Iter_17/controlnet"),
                lora_path=os.getenv("LORA_PATH", "/models/Finetuned_Iter_17/unet"),
                stable_diffusion_ckpt=os.getenv("SD_MODEL_PATH", "/models/stable-diffusion-v1-4"),
            )

            logging.info("SYSTEM: Wall Detector model loaded successfully")

    return wall_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    def _preload():
        global _model_ready
        try:
            logging.info("SYSTEM: Starting model preload in background...")
            get_wall_detector()
            _model_ready = True
            logging.info("SYSTEM: Model ready to serve requests")
        except Exception as e:
            logging.exception(f"SYSTEM: Model preload failed: {e}")
    threading.Thread(target=_preload, daemon=True).start()
    yield


CREDENTIALS = load_gcp_credentials()

app = FastAPI(title="Wall Detector (Cloud Run)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    if _model_ready:
        return JSONResponse(content={"status": "ready"}, status_code=200)
    return JSONResponse(content={"status": "loading"}, status_code=503)


@app.post("/detect_wall")
async def detect_wall(request: Request):
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
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{str(page_number).zfill(4)}/{CREDENTIALS['CloudStorage']['blob_name']}"
    blob = bucket.blob(blob_path)
    destination_path = Path("/tmp/floor_plan.png")
    destination_path = destination_path.parent.joinpath(project_id).joinpath(plan_id).joinpath(user_id).joinpath(str(page_number)).joinpath(destination_path.name)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_path)

    mask_offset = None
    if mask:
        mask_offset = dict(horizontal=mask.get("horizontal", 0), vertical=mask.get("vertical", 0))

    async with _inference_lock:
        image = get_wall_detector().detect(destination_path, hyperparameters, mask_offset=mask_offset)

    logging.info("SYSTEM: Wall Detection Completed")
    return respond_with_JSON_payload(CREDENTIALS, image, project_id, plan_id, user_id, page_number, organization_slug)
