import os
import sys
import logging
from datetime import timedelta, datetime, date, time
from decimal import Decimal
from base64 import b64encode
from ruamel.yaml import YAML
from pathlib import Path
import json
from collections import defaultdict
import requests
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_core import ValidationError

from google.cloud.storage import Client as CloudStorageClient
from google.cloud import bigquery

import pandas as pd
from preprocessing import preprocess
from modeller_2d import FloorPlan2D
from extrapolate_3d import Extrapolate3D


def respond_with_UI_payload(payload):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=200,
        media_type="application/json",
    )


def upload_floorplan(plan_path, user_id, plan_id, project_id, credentials, index=None):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if index:
        blob_path = f"{project_id}.{plan_id}.{user_id}/{plan_path.name}.{index}"
    else:
        blob_path = f"{project_id}.{plan_id}.{user_id}/{plan_path.name}"
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(plan_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"


def download_floorplan(user_id, plan_id, project_id, credentials, destination_path="/tmp/floor_plan.PDF"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id}.{plan_id}.{user_id}/floor_plan.PDF"
    blob = bucket.blob(blob_path)

    blob.download_to_filename(destination_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"


def insert_model_2d(
    model_2d,
    page_number,
    plan_id,
    user_id,
    project_id,
    GCS_URL_floorplan_page,
    GCS_URL_target_drywalls_page,
    credentials
    ):
    GBQ_query = """
    MERGE `drywall_takeoff.models` t
    USING (
        SELECT
            @plan_id AS plan_id,
            @project_id AS project_id,
            @user_id AS user_id,
            @page_number AS page_number,
            @model_2d AS model_2d,
            @source AS source,
            @target_drywalls AS target_drywalls,
    ) s
    ON t.project_id = s.project_id AND t.plan_id = s.plan_id AND t.user_id = s.user_id AND t.page_number = s.page_number
    WHEN MATCHED THEN
    UPDATE SET
        model_2d = s.model_2d,
        updated_at = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
    INSERT (
        plan_id,
        project_id,
        user_id,
        page_number,
        model_2d,
        model_3d,
        takeoff,
        source,
        target_drywalls,
        created_at,
        updated_at
    )
    VALUES (
        s.plan_id,
        s.project_id,
        s.user_id,
        s.page_number,
        s.model_2d,
        JSON '{}',
        JSON '{}',
        s.source,
        s.target_drywalls,
        CURRENT_TIMESTAMP(),
        CURRENT_TIMESTAMP()
    );
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("model_2d", "JSON", model_2d),
            bigquery.ScalarQueryParameter("source", "STRING", GCS_URL_floorplan_page),
            bigquery.ScalarQueryParameter("target_drywalls", "STRING", GCS_URL_target_drywalls_page)
        ]
    )

    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    query_output = bigquery_client.query(GBQ_query, job_config=job_config).result()
    return query_output


def insert_model_3d(
    model_3d,
    page_number,
    plan_id,
    user_id,
    project_id,
    credentials
    ):
    GBQ_query = """
    UPDATE `drywall_takeoff.models`
    SET
        model_3d = @model_3d,
        updated_at = CURRENT_TIMESTAMP()
    WHERE
        project_id = @project_id
        AND plan_id = @plan_id
        AND user_id = @user_id
        AND page_number = @page_number
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("model_3d", "JSON", model_3d)
        ]
    )

    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    query_output = bigquery_client.query(GBQ_query, job_config=job_config).result()
    return query_output


def insert_takeoff(
    takeoff,
    page_number,
    plan_id,
    user_id,
    project_id,
    credentials
    ):
    GBQ_query = """
    UPDATE `drywall_takeoff.models` t
    SET
        takeoff = @takeoff,
        updated_at = CURRENT_TIMESTAMP()
    WHERE
        project_id = @project_id
        AND plan_id = @plan_id
        AND user_id = @user_id
        AND page_number = @page_number
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("takeoff", "JSON", takeoff)
        ]
    )

    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    query_output = bigquery_client.query(GBQ_query, job_config=job_config).result()
    return query_output


def insert_plan(
    project_id,
    user_id,
    credentials,
    payload_plan=None,
    plan_id=None,
    GCS_URL_floorplan=None,
    n_pages=None
    ):
    GBQ_query = """
    MERGE `drywall_takeoff.plans` t
    USING (
        SELECT
            @plan_id AS plan_id,
            @project_id AS project_id,
            @user_id AS user_id,
            @plan_name AS plan_name,
            @plan_type AS plan_type,
            @file_type AS file_type,
            @pages AS pages,
            @source AS source
    ) s
    ON t.project_id = s.project_id AND t.plan_id = s.plan_id AND t.user_id = s.user_id
    WHEN MATCHED THEN
    UPDATE SET
        pages = s.pages,
        source = s.source,
        updated_at = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
    INSERT (
        plan_id,
        project_id,
        user_id,
        plan_name,
        plan_type,
        file_type,
        pages,
        source,
        created_at,
        updated_at
    )
    VALUES (
        s.plan_id,
        s.project_id,
        s.user_id,
        s.plan_name,
        s.plan_type,
        s.file_type,
        s.pages,
        s.source,
        CURRENT_TIMESTAMP(),
        CURRENT_TIMESTAMP()
    );
    """
    if not plan_id:
        plan_id = payload_plan.plan_id
    plan_name, plan_type, file_type = '', '', ''
    if payload_plan:
        plan_name, plan_type, file_type = payload_plan.plan_name, payload_plan.plan_type, payload_plan.file_type
    if not n_pages:
        n_pages = 0
    if not GCS_URL_floorplan:
        GCS_URL_floorplan = ''
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("plan_name", "STRING", plan_name),
            bigquery.ScalarQueryParameter("plan_type", "STRING", plan_type),
            bigquery.ScalarQueryParameter("file_type", "STRING", file_type),
            bigquery.ScalarQueryParameter("pages", "INT64", n_pages),
            bigquery.ScalarQueryParameter("source", "STRING", GCS_URL_floorplan)
        ]
    )

    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    query_output = bigquery_client.query(GBQ_query, job_config=job_config).result()
    return query_output


def insert_project(payload_project, credentials):
    GBQ_query = """
    MERGE `drywall_takeoff.projects` t
    USING (
        SELECT
            @project_id AS project_id,
            @project_name AS project_name,
            @project_location AS project_location,
            @FBM_branch AS FBM_branch,
            @project_type AS project_type,
            @project_area AS project_area,
            @contractor_name AS contractor_name
    ) s
    ON t.project_id = s.project_id
    WHEN NOT MATCHED THEN
    INSERT (
        project_id,
        project_name,
        project_location,
        FBM_branch,
        project_type,
        project_area,
        contractor_name,
        created_at
    )
    VALUES (
        s.project_id,
        s.project_name,
        s.project_location,
        s.FBM_branch,
        s.project_type,
        s.project_area,
        s.contractor_name,
        CURRENT_TIMESTAMP()
    );
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_id", "STRING", payload_project.project_id),
            bigquery.ScalarQueryParameter("project_name", "STRING", payload_project.project_name),
            bigquery.ScalarQueryParameter("project_location", "STRING", payload_project.project_location),
            bigquery.ScalarQueryParameter("FBM_branch", "STRING", payload_project.FBM_branch),
            bigquery.ScalarQueryParameter("project_type", "STRING", payload_project.project_type),
            bigquery.ScalarQueryParameter("project_area", "STRING", payload_project.project_area),
            bigquery.ScalarQueryParameter("contractor_name", "STRING", payload_project.contractor_name)
        ]
    )

    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    query_output = bigquery_client.query(GBQ_query, job_config=job_config).result()
    GBQ_query = f"SELECT created_at FROM `{credentials["GBQServer"]["table_name_projects"]}` WHERE project_id = '{payload_project.project_id}'"
    query_output = bigquery_client.query(GBQ_query).result()
    created_at = list(query_output)[0].created_at.isoformat()
    return created_at


def floorplan_to_walls(credentials, index):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["wall_detector"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token

    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{credentials["CloudRun"]["APIs"]["wall_detector"]}/detect_wall",
        headers=headers,
        json=dict(index=index)
    )

    image_path  = Path("/tmp/floor_plan_wall_segmented.png")
    with open(image_path, "wb") as f:
        f.write(response.content)
    return image_path


def load_UI_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.map(lambda x: float(x) if isinstance(x, Decimal) else x)
    df = df.map(lambda x: x.isoformat() if isinstance(x, date) else x)
    df = df.map(lambda x: x.date().isoformat() if isinstance(x, datetime) else x)
    df = df.map(lambda x: x.isoformat() if isinstance(x, time) else x)
    df = df.map(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    df = df.map(lambda x: b64encode(x).decode("utf-8") if isinstance(x, bytes) else x)

    return df


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


app = FastAPI(title="Drywall Takeoff (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PayloadProject(BaseModel):
    project_id: str
    project_name: str
    project_location: str
    project_area: str
    project_type: str
    contractor_name: str
    FBM_branch: str

class PayloadPlan(BaseModel):
    plan_id: str
    plan_name: str
    plan_type: str
    file_type: str


@app.post("/generate_project")
async def generate_project(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    try:
        payload_project = PayloadProject(**parameters)
    except ValidationError:
        payload_project = PayloadProject(**body)
    created_at = insert_project(payload_project, CREDENTIALS)
    logging.info(f"SYSTEM: New Project {payload_project.project_name} generated successfully")
    return respond_with_UI_payload(
        dict(
            project_id=payload_project.project_id,
            project_name=payload_project.project_name,
            created_at=created_at,
        )
    )


@app.get("/load_projects")
async def load_projects():
    enable_logging_on_stdout()
    GBQ_query = f"SELECT * FROM `{CREDENTIALS["GBQServer"]["table_name_projects"]}`"
    bigquery_client = bigquery.Client.from_service_account_json(CREDENTIALS["GBQServer"]["service_account_key"])
    query_output = bigquery_client.query(GBQ_query).to_dataframe()
    dataframe = load_UI_dataframe(query_output)
    projects = dataframe.to_dict(orient="records")

    logging.info(f"SYSTEM: Project Metadata retrieved successfully")
    return respond_with_UI_payload(dict(projects=projects))


@app.post("/load_project")
async def load_project(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")

    GBQ_query = f"SELECT projects.project_id, models.page_number, models.model_2d, models.model_3d, models.created_at, models.updated_at, models.takeoff, models.source, plans.user_id, plans.plan_id, plans.plan_name, plans.plan_type, plans.file_type FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` AS models JOIN `{CREDENTIALS["GBQServer"]["table_name_plans"]}` AS plans ON models.project_id = plans.project_id AND models.plan_id = plans.plan_id JOIN `{CREDENTIALS["GBQServer"]["table_name_projects"]}` AS projects ON plans.project_id = projects.project_id WHERE projects.project_id = '{project_id}';"
    bigquery_client = bigquery.Client.from_service_account_json(CREDENTIALS["GBQServer"]["service_account_key"])
    query_output = bigquery_client.query(GBQ_query).to_dataframe()
    dataframe = load_UI_dataframe(query_output)
    project_data = dataframe.to_dict(orient="records")

    logging.info(f"SYSTEM: Project Data retrieved successfully")
    return respond_with_UI_payload(dict(project_data=project_data))


@app.post("/generate_floorplan_upload_signed_URL")
async def generate_floorplan_upload_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    payload_plan = parameters.get("plan") or body.get("plan")
    user_id = parameters.get("user_id") or body.get("user_id")
    payload_plan = PayloadPlan(**payload_plan)
    logging.info("SYSTEM: Received Signed Floorplan download URL generation Request")

    insert_plan(
        project_id,
        user_id,
        CREDENTIALS,
        payload_plan=payload_plan
    )

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id}.{payload_plan.plan_id}.{user_id}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="PUT",
        content_type="application/octet-stream",
    )

    return url


@app.post("/floorplan_to_2d")
async def floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")

    hyperparameters = load_hyperparameters()
    pdf_path = Path("/tmp/floor_plan.PDF")
    GCS_URL_floorplan = download_floorplan(user_id, plan_id, project_id, CREDENTIALS, destination_path=pdf_path)
    logging.info("SYSTEM: Floorplan Downloaded")

    floor_plan_paths_preprocessed = preprocess(pdf_path)
    insert_plan(
        project_id,
        user_id,
        CREDENTIALS,
        plan_id=plan_id,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
    )
    logging.info("SYSTEM: Floorplan Preprocessing Completed")

    floor_plan_modeller_2d = FloorPlan2D(hyperparameters)
    walls_2d_all = list()
    for index, floor_plan_path in enumerate(floor_plan_paths_preprocessed):
        floorplan_page_source = upload_floorplan(floor_plan_path, user_id, plan_id, project_id, CREDENTIALS)
        logging.info(f"SYSTEM: Preprocessed Floorplan Image uploaded to GCS from PAGE: {index}")
        wall_segmented_path = floorplan_to_walls(CREDENTIALS, index)
        upload_floorplan(wall_segmented_path, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
        logging.info(f"SYSTEM: Wall Detection Completed from PAGE: {index}")

        walls_2d, walls_2d_path = floor_plan_modeller_2d.model(image_path=wall_segmented_path)
        model_2d_path = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_path)
        upload_floorplan(model_2d_path, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
        model_2d_path_overlay_enabled = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_path, overlay_enabled=True)
        target_drywalls_page_source = upload_floorplan(model_2d_path_overlay_enabled, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
        logging.info(f"SYSTEM: A 2D Model of the Floorplan from PAGE: {index} Generated Successfully")
        insert_model_2d(walls_2d, index, plan_id, user_id, project_id, floorplan_page_source, target_drywalls_page_source, CREDENTIALS)
        walls_2d_all.append(walls_2d)

    return respond_with_UI_payload(walls_2d_all)


@app.post("/update_floorplan_to_2d")
async def update_floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    index = parameters.get("index") or body.get("index")
    logging.info("SYSTEM: Received a Floorplan 2D Model Update Request")

    insert_model_2d(walls_2d_JSON, index, plan_id, user_id, project_id, None, CREDENTIALS)
    logging.info("SYSTEM: Floorplan 2D Model Updated Successfully")


@app.post("/floorplan_to_3d")
async def floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    index = parameters.get("index") or body.get("index")
    logging.info("SYSTEM: Received a Floorplan 3D Model Generation Request")

    model_2d_path = "/tmp/walls_2d.json"
    with open(model_2d_path, 'w') as f:
        json.dump(walls_2d_JSON, f)
    hyperparameters = load_hyperparameters()
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    walls_3d, walls_3d_path = floor_plan_modeller_3d.extrapolate(model_2d_path=model_2d_path)
    model_3d_path = floor_plan_modeller_3d.save_plot_3d(walls_3d_path)
    upload_floorplan(model_3d_path, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
    insert_model_3d(walls_3d, index, plan_id, user_id, project_id, CREDENTIALS)
    logging.info("SYSTEM: A 3D Model of the Floorplan Generated Successfully")

    return respond_with_UI_payload(walls_3d)


@app.post("/update_floorplan_to_3d")
async def update_floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d_JSON = parameters.get("walls_3d") or body.get("walls_3d")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    index = parameters.get("index") or body.get("index")
    logging.info("SYSTEM: Received a Floorplan 3D Model Update Request")

    insert_model_3d(walls_3d_JSON, index, plan_id, user_id, project_id, CREDENTIALS)
    logging.info("SYSTEM: Floorplan 3D Model Updated Successfully")


@app.post("/generate_drywall_overlaid_floorplan_download_signed_URL")
async def generate_drywall_overlaid_floorplan_download_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    index = parameters.get("index") or body.get("index")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    logging.info("SYSTEM: Received Signed Floorplan download URL generation Request")

    GBQ_query = f"SELECT target_drywalls FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE project_id = '{project_id}' AND plan_id = '{plan_id}' AND user_id = '{user_id}' AND page_number = {index};"
    bigquery_client = bigquery.Client.from_service_account_json(CREDENTIALS["GBQServer"]["service_account_key"])
    query_output = bigquery_client.query(GBQ_query).result()
    drywall_overlaid_floorplan_source_path = list(query_output)[0].target_drywalls
    _, _, _, blob_path = drywall_overlaid_floorplan_source_path.split('/', 3)

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="GET",
        content_type="application/octet-stream",
    )

    return url


@app.post("/compute_takeoff")
async def compute_takeoff(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d_JSON = parameters.get("walls_3d") or body.get("walls_3d")
    index = parameters.get("index") or body.get("index")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    logging.info("SYSTEM: Received a Drywall Takeoff computation Request")

    drywall_takeoff = dict(total=0, per_drywall=defaultdict(lambda: 0))
    for wall in walls_3d_JSON:
        surface_area = wall["height"] * wall["length"]
        drywall_count = 0
        for drywall in wall["surfaces_drywall"]:
            if drywall["enabled"]:
                drywall_takeoff["per_drywall"][drywall["type"]] += surface_area
                drywall_count += 1
        drywall_takeoff["total"] += drywall_count * surface_area

    drywall_takeoff["total"] = round(drywall_takeoff["total"], 2)
    for key in drywall_takeoff["per_drywall"]:
        drywall_takeoff["per_drywall"][key] = round(drywall_takeoff["per_drywall"][key], 2)

    insert_takeoff(drywall_takeoff, index, plan_id, user_id, project_id, CREDENTIALS)
    logging.info("SYSTEM: Drywall Takeoff Computed Successfully for the provided Floorplan")
    return respond_with_UI_payload(drywall_takeoff)
