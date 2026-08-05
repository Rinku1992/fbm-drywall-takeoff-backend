import os
import sys
import re
import logging
import colorsys
from datetime import datetime, timezone, timedelta
from ruamel.yaml import YAML
from pathlib import Path
import json
from time import time as from_unix_epoch
from time import sleep
from collections import defaultdict
from functools import partial
import requests
from requests.adapters import HTTPAdapter
from contextlib import asynccontextmanager
from geopy import Nominatim
from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from pydantic_core import ValidationError
from concurrent.futures import ThreadPoolExecutor
import traceback
import asyncio
import tempfile
import shutil

from google.cloud.storage import Client as CloudStorageClient
from google.cloud import secretmanager

import pandas as pd
import numpy as np
import math
import cv2
from pdf2image.pdf2image import pdfinfo_from_path

from extrapolate_3d import Extrapolate3D
from floor_plan import FloorPlan
from email_notification import trigger_support
from otp_management import (
    generate_otp,
    trigger_otp_email,
    is_authenticated,
    load_secret_json,
    create_external_login_jwt,
    normalize_email,
    is_valid_email,
    PayloadRequestExternalOtp,
    PayloadVerifyExternalOtp,
    OTP_EXPIRATION_MINUTES,
    MAX_OTP_ATTEMPTS,
)
from helper import (
    load_pg_pool,
    close_pg_pool,
    pg_run,
    load_vertex_ai_client,
    sha256,
    upload_floorplan,
    insert_model_2d,
    insert_layout_2d,
    is_duplicate,
    delete_plan,
    load_floorplan_to_structured_2d_ID_token,
    load_visual_grounding,
    load_templates,
    query_drywall,
    load_subscriber_client,
    query_subscriber_messages,
    map_floorplan_to_multipage_elevation,
    load_elevation_map,
    floorplan_to_pages,
    page_to_svg,
    insert_page,
    load_drywall_weights,
    download_floorplan,
    enforce_early_stopping,
    load_organization_slug,
    create_session,
    is_session_active,
    lock_user,
    unlock_user,
)
from policy import AccessControlService
from prompts import VISUAL_GROUNDING_DETECTOR, SLOPED_CEILING_CHOICES


def respond_with_UI_payload(payload, status_code=200, disable_caching=False):
    if disable_caching:
        return JSONResponse(
            content=json.loads(json.dumps(payload)),
            status_code=status_code,
            media_type="application/json",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json"
    )

def load_parameter_first_not_none(*values):
    return next(
        (v for v in values if v is not None),
        None
    )

async def insert_model_2d_revision(
    model_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    pg_pool,
    credentials,
    page_section_number=None,
    ):
    if not page_section_number:
        page_section_number = 'I'
    if not model_2d.get("metadata", None):
        query = f"SELECT model_2d->'metadata' AS metadata FROM {credentials["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s"
        query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, int(page_number), page_section_number,), fetch=True))
        metadata = query_output[0]["metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        model_2d["metadata"] = metadata
    query = f"SELECT MAX(revision_number) AS revision_number FROM {credentials["CloudSQL"]["table_name_model_revisions_2d"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s"
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, int(page_number), page_section_number,), fetch=True))
    
    if query_output and query_output[0]["revision_number"] is not None:
        revision_number = query_output[0]["revision_number"] + 1
    else:
        revision_number = 1

    query = f"""
        INSERT INTO {credentials["CloudSQL"]["table_name_model_revisions_2d"]} (
            plan_id,
            project_id,
            user_id,
            page_number,
            page_section_number,
            scale,
            model,
            created_at,
            revision_number
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            CURRENT_TIMESTAMP,
            %s
        );
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(plan_id, project_id, user_id, int(page_number), page_section_number, scale, json.dumps(model_2d), int(revision_number),)))


async def insert_model_3d_revision(
    model_3d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    pg_pool,
    credentials
    ):
    query = f"SELECT MAX(revision_number) AS revision_number FROM {credentials["CloudSQL"]["table_name_model_revisions_3d"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s"
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, int(page_number),), fetch=True))
    
    if query_output and query_output[0]["revision_number"] is not None:
        revision_number = query_output[0]["revision_number"] + 1
    else:
        revision_number = 1

    query = f"""
        INSERT INTO {credentials["CloudSQL"]["table_name_model_revisions_3d"]} (
            plan_id,
            project_id,
            user_id,
            page_number,
            scale,
            model,
            takeoff,
            created_at,
            revision_number
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s::jsonb,
            '{{}}'::jsonb,
            CURRENT_TIMESTAMP,
            %s
        );
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(plan_id, project_id, user_id, int(page_number), scale, json.dumps(model_3d), int(revision_number),)))


async def insert_model_3d(
    model_3d,
    scale,
    page_number,
    page_section_number,
    plan_id,
    project_id,
    pg_pool,
    credentials
    ):
    query = f"""
        UPDATE {credentials["CloudSQL"]["table_name_models"]} as t
        SET
            model_3d = %s,
            scale = COALESCE(NULLIF(%s, ''), t.scale),
            updated_at = CURRENT_TIMESTAMP
        WHERE
            LOWER(project_id) = LOWER(%s)
            AND LOWER(plan_id) = LOWER(%s)
            AND page_number = %s
            AND page_section_number = %s
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(
        json.dumps(model_3d),
        scale,
        project_id,
        plan_id,
        page_number,
        page_section_number,
    )))


async def delete_floorplan(project_id, plan_id, user_id, pg_pool, credentials):
    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_pages"]}
        WHERE
            LOWER(project_id) = LOWER(%s)
            AND LOWER(plan_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_plans"]}
        WHERE
            LOWER(project_id) = LOWER(%s)
            AND LOWER(plan_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_models"]}
        WHERE
            LOWER(project_id) = LOWER(%s)
            AND LOWER(plan_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_model_revisions_2d"]}
        WHERE
            LOWER(project_id) = LOWER(%s)
            AND LOWER(plan_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_model_revisions_3d"]}
        WHERE
            LOWER(project_id) = LOWER(%s)
            AND LOWER(plan_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id,)))

    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
    prefix = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)


async def delete_project(project_id, user_id, pg_pool, credentials):
    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_pages"]}
        WHERE
            LOWER(project_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_plans"]}
        WHERE
            LOWER(project_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_projects"]}
        WHERE
            LOWER(project_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_models"]}
        WHERE
            LOWER(project_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_model_revisions_2d"]}
        WHERE
            LOWER(project_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id,)))

    query = f"""
        DELETE FROM {credentials["CloudSQL"]["table_name_model_revisions_3d"]}
        WHERE
            LOWER(project_id) = LOWER(%s);
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id,)))

    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
    prefix = f"{organization_slug}/{project_id.lower()}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)


async def insert_takeoff(
    takeoff,
    waste_factor_average,
    drywall_negate_opening_area_threshold,
    page_number,
    page_section_number,
    plan_id,
    user_id,
    project_id,
    revision_number,
    pg_pool,
    credentials
):
    query = f"""
        UPDATE {credentials["CloudSQL"]["table_name_models"]} t
        SET
            takeoff = %s,
            waste_average = %s,
            drywall_negate_opening_area_threshold = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE
            LOWER(project_id) = LOWER(%s)
            AND LOWER(plan_id) = LOWER(%s)
            AND page_number = %s
            AND page_section_number = %s
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(
        takeoff,
        waste_factor_average,
        drywall_negate_opening_area_threshold,
        project_id,
        plan_id,
        page_number,
        page_section_number,
    )))

    if revision_number:
        query = f"""
            UPDATE {credentials["CloudSQL"]["table_name_model_revisions_3d"]} t
            SET
                takeoff = %s,
                user_id = %s
            WHERE
                LOWER(project_id) = LOWER(%s)
                AND LOWER(plan_id) = LOWER(%s)
                AND page_number = %s
                AND page_section_number = %s
                AND revision_number = %s
        """
        await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(
            takeoff,
            user_id,
            project_id,
            plan_id,
            page_number,
            page_section_number,
            revision_number,
        )))


async def insert_plan(
    project_id,
    user_id,
    status,
    pg_pool,
    credentials,
    payload_plan=None,
    plan_id=None,
    size_in_bytes=None,
    GCS_URL_floorplan=None,
    n_pages=None
    ):
    query = f"""
        INSERT INTO {credentials["CloudSQL"]["table_name_plans"]} (
            plan_id,
            project_id,
            user_id,
            status,
            plan_name,
            plan_type,
            file_type,
            pages,
            size_in_bytes,
            source,
            sha256,
            created_at,
            updated_at
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (project_id, plan_id) DO UPDATE SET
            pages = EXCLUDED.pages,
            source = EXCLUDED.source,
            sha256 = EXCLUDED.sha256,
            status = EXCLUDED.status,
            size_in_bytes = EXCLUDED.size_in_bytes,
            updated_at = CURRENT_TIMESTAMP
    """
    sha_256 = ''
    if plan_id:
        pdf_path = Path("/tmp/floor_plan.PDF")
        await download_floorplan(plan_id, project_id, user_id, credentials, pg_pool, destination_path=pdf_path)
        sha_256 = sha256(pdf_path)
    if not plan_id:
        plan_id = payload_plan.plan_id
    plan_name, plan_type, file_type = '', '', ''
    if payload_plan:
        plan_name, plan_type, file_type = payload_plan.plan_name, payload_plan.plan_type, payload_plan.file_type
    if not n_pages:
        n_pages = 0
    if not GCS_URL_floorplan:
        GCS_URL_floorplan = ''
    if not size_in_bytes:
        size_in_bytes = 0
    await run_in_threadpool(partial(
        pg_run,
        credentials,
        pg_pool,
        query,
        params=(
            plan_id,
            project_id,
            user_id,
            status,
            plan_name,
            plan_type,
            file_type,
            n_pages,
            size_in_bytes,
            GCS_URL_floorplan,
            sha_256,
        ),
    ))


async def insert_project(payload_project, pg_pool, credentials):
    query = f"""
        INSERT INTO {credentials["CloudSQL"]["table_name_projects"]} (
            project_id,
            project_name,
            project_location,
            project_location_pincode,
            "FBM_branch",
            project_type,
            project_area,
            contractor_name,
            created_at,
            created_by
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            CURRENT_TIMESTAMP,
            %s
        )
        ON CONFLICT (project_id) DO NOTHING
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(
        payload_project.project_id,
        payload_project.project_name,
        payload_project.project_location,
        payload_project.project_location_pincode,
        payload_project.FBM_branch,
        payload_project.project_type,
        payload_project.project_area,
        payload_project.contractor_name,
        payload_project.created_by
    )))
    query = f"SELECT created_at FROM {credentials["CloudSQL"]["table_name_projects"]} WHERE project_id = %s"
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(payload_project.project_id,), fetch=True))
    created_at = query_output[0]["created_at"].isoformat()
    return created_at


def floorplan_to_structured_2d(
    credentials,
    session,
    id_token,
    project_id,
    plan_id,
    user_id,
    page_number,
    mask_factor,
    bounding_box_offsets,
    elevation_pages,
    architectural_scale,
    session_uuid,
    model,
    predict,
):
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }
    response = session.post(
        f"{credentials["CloudRun"]["APIs"]["floorplan_to_structured_2d"]}/floorplan_to_structured_2d",
        headers=headers,
        json=dict(
            project_id=project_id,
            plan_id=plan_id,
            user_id=user_id,
            page_number=page_number,
            mask_factor=mask_factor,
            bounding_box_offsets=bounding_box_offsets,
            elevation_pages=elevation_pages,
            architectural_scale=architectural_scale,
            session_uuid=session_uuid,
            model=model,
            predict=predict,
        ),
    )
    return response.raise_for_status()


async def floorplan_to_preview_pages(
    credentials,
    project_id,
    plan_id,
    user_id,
    n_pages,
    pdf_path,
    pg_pool,
    maximum_dpi,
    minimum_dpi
):
    preview_pages = list()
    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    floor_plan_processed_paths, pages = await floorplan_to_pages(
        credentials,
        pg_pool,
        project_id,
        plan_id,
        user_id,
        pdf_path,
        n_pages,
        maximum_dpi,
        minimum_dpi
    )
    for floor_plan_processed_path, page in zip(floor_plan_processed_paths, pages["pages"]):
        metadata_page = dict(page_number=page["page_number"])
        metadata_page["plan_type"] = page["plan_type"]
        metadata_page["is_floorplan"] = page["plan_type"].upper().find("FLOOR") != -1
        metadata_page["status"] = "NOT STARTED"
        svg_path=Path(f"/tmp/{project_id}/{plan_id}/{user_id}/scaled_floor_plan_{str(page["page_number"]).zfill(4)}.svg")
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        floorplan_svg = page_to_svg(floor_plan_path=floor_plan_processed_path, svg_path=svg_path)
        floorplan_svg_source = await upload_floorplan(floorplan_svg, plan_id, project_id, user_id, credentials, pg_pool, index=str(page["page_number"]).zfill(4))
        _, _, _, blob_path = floorplan_svg_source.split('/', 3)
        blob = bucket.blob(blob_path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=credentials["CloudStorage"]["expiration_in_minutes"]),
            method="GET",
        )
        metadata_page["signed_url_GCS"] = url
        floor_plan_processed_image = cv2.imread(floor_plan_processed_path)
        floor_plan_processed_image = cv2.resize(floor_plan_processed_image, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        floor_plan_processed_path_thumbnail = floor_plan_processed_path.parent.joinpath(floor_plan_processed_path.name.replace("floor_plan", "floor_plan_thumbnail"))
        cv2.imwrite(floor_plan_processed_path_thumbnail, floor_plan_processed_image)
        svg_path_thumbnail=Path(f"/tmp/{project_id}/{plan_id}/{user_id}/scaled_floor_plan_thumbnail_{str(page["page_number"]).zfill(4)}.svg")
        floorplan_svg_thumbnail = page_to_svg(floor_plan_path=floor_plan_processed_path_thumbnail, svg_path=svg_path_thumbnail)
        floorplan_svg_source_thumbnail = await upload_floorplan(floorplan_svg_thumbnail, plan_id, project_id, user_id, credentials, pg_pool, index=str(page["page_number"]).zfill(4))
        _, _, _, blob_path = floorplan_svg_source_thumbnail.split('/', 3)
        blob = bucket.blob(blob_path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=credentials["CloudStorage"]["expiration_in_minutes"]),
            method="GET",
        )
        metadata_page["signed_url_thumbnail_GCS"] = url
        query = f"UPDATE {credentials["CloudSQL"]["table_name_pages"]} SET source = %s, thumbnail = %s WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
        await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(floorplan_svg_source, floorplan_svg_source_thumbnail, project_id, plan_id, page["page_number"],)))
        preview_pages.append(metadata_page)
        logging.info(f"SYSTEM: Preview Generated for {page["page_number"]+1}/{n_pages} pages")
    return preview_pages


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
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials["service_drywall_account_key"]

    return credentials


def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("config/hyperparameters.yaml", 'r') as f:
        hyperparameters = yaml.load(f)

    return hyperparameters


CREDENTIALS = load_gcp_credentials()
pg_pool = dict()
DRYWALL_TEMPLATES = None
access_control = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DRYWALL_TEMPLATES
    global pg_pool
    global access_control

    for attempt in range(10):
        try:
            engine = load_pg_pool(CREDENTIALS)
            pg_pool["engine"] = engine
            DRYWALL_TEMPLATES = await load_templates(
                pg_pool,
                CREDENTIALS
            )
            access_control = AccessControlService(CREDENTIALS, pg_pool)

            break
        except Exception as e:
            logging.exception(e)

            if attempt == 9:
                raise

            await asyncio.sleep(min(2 ** attempt, 30))

    yield

    if pg_pool and pg_pool["engine"]:
        close_pg_pool()

app = FastAPI(title="Drywall Takeoff (Cloud Run)", lifespan=lifespan)
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
    project_location_pincode: str="92780"
    project_area: str
    project_type: str
    contractor_name: str
    FBM_branch: str
    created_by: str

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
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=payload_project.created_by)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {payload_project.created_by} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    geolocator = Nominatim(user_agent="xtimator_app")
    payload_project.FBM_branch = f"{payload_project.project_location_pincode}, {payload_project.project_location}"
    if geolocator.geocode(f"{payload_project.project_location_pincode}, {payload_project.project_location}", language="en"):
        payload_project.FBM_branch = ','.join(geolocator.geocode(f"{payload_project.project_location_pincode}, {payload_project.project_location}", language="en").address.rsplit(',', 2)[1:]).strip()
    created_at = await insert_project(payload_project, pg_pool, CREDENTIALS)
    logging.info(f"SYSTEM: New Project {payload_project.project_name} generated successfully")
    return respond_with_UI_payload(
        dict(
            project_id=payload_project.project_id,
            project_name=payload_project.project_name,
            created_at=created_at,
        )
    )


@app.post("/load_projects")
async def load_projects(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    user_id = parameters.get("user_id") or body.get("user_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    is_admin = await access_control.is_admin(user_id)
    if bool(is_admin):
        if is_admin.super:
            where_clause = "TRUE"
            params = ()

        elif is_admin.local:
            peers = await access_control.load_organization_users(user_id)
            where_clause = "LOWER(p.created_by) = ANY(%s)"
            params = (peers,)
            peers_partner = await access_control.load_regional_users_partner_organizations(user_id)
            region_names = await access_control.load_user_region_names(user_id)
            where_clause_partner = "LOWER(p.created_by) = ANY(%s) AND LOWER(p.\"FBM_branch\") = ANY(%s)"
            params_partner = (peers_partner, region_names,)

    else:
        peers = await access_control.load_regional_users(user_id)
        region_names = await access_control.load_user_region_names(user_id)
        where_clause = "LOWER(p.created_by) = ANY(%s) AND LOWER(p.\"FBM_branch\") = ANY(%s)"
        params = (peers, region_names,)

    query = f"""
        SELECT
            p.*,
            CASE
                WHEN COALESCE(pc.plan_count, 0) = 0
                    THEN 'NOT STARTED'
                ELSE 'ACTIVE'
            END AS status

        FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} p

        LEFT JOIN (
            SELECT
                LOWER(project_id) AS project_id,
                COUNT(*) AS plan_count
            FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]}
            GROUP BY LOWER(project_id)
        ) pc
            ON LOWER(p.project_id) = pc.project_id

        WHERE {where_clause} AND p.is_published = TRUE

        ORDER BY p.created_at DESC NULLS LAST;
    """
    projects = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=params, fetch=True))
    if is_admin.local:
        query = f"""
            SELECT
                p.*,
                CASE
                    WHEN COALESCE(pc.plan_count, 0) = 0
                        THEN 'NOT STARTED'
                    ELSE 'ACTIVE'
                END AS status

            FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} p

            LEFT JOIN (
                SELECT
                    LOWER(project_id) AS project_id,
                    COUNT(*) AS plan_count
                FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]}
                GROUP BY LOWER(project_id)
            ) pc
                ON LOWER(p.project_id) = pc.project_id

            WHERE {where_clause_partner} AND p.is_published = TRUE

            ORDER BY p.created_at DESC NULLS LAST;
        """
        projects_partner = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=params_partner, fetch=True))
        projects += projects_partner

    logging.info("SYSTEM: Project Metadata retrieved successfully")
    return respond_with_UI_payload(
        jsonable_encoder({
            "projects": projects
        })
    )


@app.post("/publish_project")
async def publish_project(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"""
        UPDATE
        FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} p

        SET is_published = TRUE
        WHERE LOWER(project_id) = LOWER(%s);
    """
    projects = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id,)))

    logging.info(f"SYSTEM: Project: {project_id} Published successfully")


@app.post("/retract_project")
async def retract_project(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"""
        UPDATE
        FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} p

        SET is_published = FALSE
        WHERE LOWER(project_id) = LOWER(%s);
    """
    projects = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id,)))

    logging.info(f"SYSTEM: Project: {project_id} Retracted successfully")


@app.post("/load_project_plans")
async def load_project_plans(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    is_admin = await access_control.is_admin(user_id)
    if bool(is_admin):
        if is_admin.super:
            where_clause = "TRUE"
            params = (project_id,)

        elif is_admin.local:
            peers = await access_control.load_organization_users(user_id)
            where_clause = "LOWER(pl.user_id) = ANY(%s)"
            params = (peers, project_id, peers,)
            peers_partner = await access_control.load_regional_users_partner_organizations(user_id)
            region_names = await access_control.load_user_region_names(user_id)
            where_clause_partner = "LOWER(pl.user_id) = ANY(%s) AND LOWER(pr.\"FBM_branch\") = ANY(%s)"
            params_partner = (peers_partner, region_names, project_id, peers_partner, region_names,)

    else:
        peers = await access_control.load_regional_users(user_id)
        region_names = await access_control.load_user_region_names(user_id)
        where_clause = "LOWER(pl.user_id) = ANY(%s) AND LOWER(pr.\"FBM_branch\") = ANY(%s)"
        params = (peers, region_names, project_id, peers, region_names,)

    query = f"""
        WITH page_stats AS (
            SELECT
                LOWER(plan_id) AS plan_id,
                COUNT(*) FILTER (
                    WHERE UPPER(status) IN ('COMPLETED', 'SCALE_NOT_DETECTED')
                ) AS pages_completed,
                COUNT(*) FILTER (
                    WHERE UPPER(status) NOT IN ('COMPLETED', 'NOT STARTED', 'SCALE_NOT_DETECTED')
                ) AS pages_in_progress
            FROM {CREDENTIALS["CloudSQL"]["table_name_pages"]}
            GROUP BY LOWER(plan_id)
        )

        SELECT
            pr.*,

            (
                SELECT COALESCE(
                    jsonb_agg(
                        to_jsonb(pl)
                        || jsonb_build_object(
                            'pages_completed', COALESCE(ps.pages_completed, 0),
                            'pages_in_progress', COALESCE(ps.pages_in_progress, 0)
                        )
                        ORDER BY pl.created_at
                    ),
                    '[]'::jsonb
                )
                FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} pl
                LEFT JOIN page_stats ps
                    ON LOWER(pl.plan_id) = ps.plan_id
                WHERE
                    LOWER(pl.project_id) = LOWER(pr.project_id)
                    AND {where_clause}
            ) AS project_plans

        FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr

        WHERE
            LOWER(pr.project_id) = LOWER(%s)
            AND (
                EXISTS (
                    SELECT 1
                    FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} pl
                    WHERE
                        LOWER(pl.project_id) = LOWER(pr.project_id)
                        AND {where_clause} AND pl.is_published = TRUE
                )
            )
    """
    rows = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=params, fetch=True))
    rows_partner = list()
    if is_admin.local:
        query = f"""
            WITH page_stats AS (
                SELECT
                    LOWER(plan_id) AS plan_id,
                    COUNT(*) FILTER (
                        WHERE UPPER(status) IN ('COMPLETED', 'SCALE_NOT_DETECTED')
                    ) AS pages_completed,
                    COUNT(*) FILTER (
                        WHERE UPPER(status) NOT IN ('COMPLETED', 'NOT STARTED', 'SCALE_NOT_DETECTED')
                    ) AS pages_in_progress
                FROM {CREDENTIALS["CloudSQL"]["table_name_pages"]}
                GROUP BY LOWER(plan_id)
            )

            SELECT
                pr.*,

                (
                    SELECT COALESCE(
                        jsonb_agg(
                            to_jsonb(pl)
                            || jsonb_build_object(
                                'pages_completed', COALESCE(ps.pages_completed, 0),
                                'pages_in_progress', COALESCE(ps.pages_in_progress, 0)
                            )
                            ORDER BY pl.created_at
                        ),
                        '[]'::jsonb
                    )
                    FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} pl
                    LEFT JOIN page_stats ps
                        ON LOWER(pl.plan_id) = ps.plan_id
                    WHERE
                        LOWER(pl.project_id) = LOWER(pr.project_id)
                        AND {where_clause_partner}
                ) AS project_plans

            FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr

            WHERE
                LOWER(pr.project_id) = LOWER(%s)
                AND (
                    EXISTS (
                        SELECT 1
                        FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} pl
                        WHERE
                            LOWER(pl.project_id) = LOWER(pr.project_id)
                            AND {where_clause_partner} AND pl.is_published = TRUE
                    )
                )
        """
        rows_partner = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=params_partner, fetch=True))

    if not rows and not rows_partner:
        return respond_with_UI_payload(dict(project_metadata=dict(), project_plans=list()))

    plans = dict()
    if rows:
        row = rows[0]
        project_metadata = dict(row)
        project_plans = project_metadata.pop("project_plans", list())
        for project_plan in project_plans:
            plans[project_plan["plan_id"].lower()] = project_plan
    if is_admin.local and rows_partner:
        row_partner = rows_partner[0]
        project_metadata = dict(row_partner)
        partner_project_plans = project_metadata.pop("project_plans", list())
        for partner_project_plan in partner_project_plans:
            plans.setdefault(partner_project_plan["plan_id"].lower(), partner_project_plan)

    project_plans = list(plans.values())

    logging.info("SYSTEM: Project Plans Data retrieved successfully")
    return respond_with_UI_payload(
        jsonable_encoder({
            "project_metadata": project_metadata,
            "project_plans": project_plans
        })
    )


@app.post("/publish_plan")
async def publish_plan(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"""
        UPDATE
        FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} p

        SET is_published = TRUE
        WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);
    """
    projects = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,)))

    logging.info(f"SYSTEM: Plan: {plan_id} Published successfully")


@app.post("/retract_plan")
async def retract_plan(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"""
        UPDATE
        FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} p

        SET is_published = FALSE
        WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);
    """
    projects = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,)))

    logging.info(f"SYSTEM: Plan: {plan_id} Published successfully")


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
    logging.info("SYSTEM: Received Signed Floorplan upload URL generation Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    await insert_plan(
        project_id,
        user_id,
        "NOT STARTED",
        pg_pool,
        CREDENTIALS,
        payload_plan=payload_plan,
    )

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    organization_slug = await load_organization_slug(CREDENTIALS, pg_pool, user_id)
    blob_path = f"{organization_slug}/{project_id.lower()}/{payload_plan.plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="PUT",
        content_type="application/octet-stream",
    )

    return url


@app.post("/generate_floorplan_download_signed_URL")
async def generate_floorplan_download_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    logging.info("SYSTEM: Received Signed Floorplan download URL generation Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    organization_slug = await load_organization_slug(CREDENTIALS, pg_pool, user_id)
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="GET",
        content_type="application/octet-stream",
    )

    return url


@app.post("/load_plan_pages")
async def load_plan_pages(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    is_admin = await access_control.is_admin(user_id)
    if bool(is_admin):
        if is_admin.super:
            where_clause = "TRUE"
            params = (project_id, plan_id,)

        elif is_admin.local:
            peers = await access_control.load_organization_users(user_id)
            where_clause = "LOWER(pl.user_id) = ANY(%s)"
            params = (project_id, plan_id, peers,)

    else:
        peers = await access_control.load_regional_users(user_id)
        region_names = await access_control.load_user_region_names(user_id)
        where_clause = "LOWER(pl.user_id) = ANY(%s) AND LOWER(pr.\"FBM_branch\") = ANY(%s)"
        params = (project_id, plan_id, peers, region_names,)

    query = f"""
        SELECT pl.*
        FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} pl
        JOIN {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr
            ON LOWER(pl.project_id) = LOWER(pr.project_id)
        WHERE
            LOWER(pl.project_id) = LOWER(%s)
            AND LOWER(pl.plan_id) = LOWER(%s)
            AND {where_clause}
    """
    rows = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=params, fetch=True))

    if not rows:
        return respond_with_UI_payload(dict(plan_metadata=dict(), plan_pages=list()))

    row = rows[0]
    plan_metadata = dict(row)

    query = f"""
        SELECT
            *
        FROM {CREDENTIALS["CloudSQL"]["table_name_pages"]}
        WHERE
            LOWER(project_id) = LOWER(%s)
            AND LOWER(plan_id) = LOWER(%s)
    """
    rows = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,), fetch=True))

    plan_pages = list()
    client = CloudStorageClient()
    for row in rows:
        plan_page = dict(row)
        bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
        url = ''
        if plan_page["source"]:
            _, _, _, blob_path = plan_page["source"].split('/', 3)
            blob = bucket.blob(blob_path)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
                method="GET",
            )
        plan_page["signed_url_GCS"] = url
        url = ''
        if plan_page["thumbnail"]:
            _, _, _, blob_path = plan_page["thumbnail"].split('/', 3)
            blob = bucket.blob(blob_path)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
                method="GET",
            )
        plan_page["signed_url_thumbnail_GCS"] = url
        plan_pages.append(plan_page)
    return respond_with_UI_payload(jsonable_encoder({
        "plan_metadata": plan_metadata,
        "plan_pages": plan_pages
    }))


@app.post("/floorplan_to_preview")
async def floorplan_to_preview(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    logging.info("SYSTEM: Received a Floorplan Preview Generation Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    pdf_path = Path("/tmp/floor_plan.PDF")
    await download_floorplan(plan_id, project_id, user_id, CREDENTIALS, pg_pool, destination_path=pdf_path)
    plan_duplicate = await is_duplicate(pg_pool, CREDENTIALS, access_control, pdf_path, project_id, user_id)
    if plan_duplicate:
        await delete_plan(CREDENTIALS, pg_pool, plan_id, project_id)
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))

    n_pages = pdfinfo_from_path(pdf_path)["Pages"]
    await insert_plan(
        project_id,
        user_id,
        "GENERATING PREVIEW",
        pg_pool,
        CREDENTIALS,
        plan_id=plan_id,
        n_pages=n_pages
    )
    logging.info("SYSTEM: Floorplan Downloaded for preview generation")

    hyperparameters = load_hyperparameters()
    payload_preview = await floorplan_to_preview_pages(
        CREDENTIALS,
        project_id,
        plan_id,
        user_id,
        n_pages,
        pdf_path,
        pg_pool,
        hyperparameters["modelling"]["scale_adoption"]["dpi"]["maximum"],
        hyperparameters["modelling"]["scale_adoption"]["dpi"]["minimum"]
    )

    logging.info("SYSTEM: Preview generated Successfully")
    return respond_with_UI_payload(payload_preview)


@app.post("/floorplan_to_2d")
async def floorplan_to_2d(request: Request):
    def get_param(name, default=True):
        value = parameters.get(name)
        if value in (None, ''):
            value = body.get(name)
        if value in (None, ''):
            value = default
        return value

    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    pages_metadata = parameters.get("pages_metadata") or body.get("pages_metadata")
    model = get_param("model")
    predict = get_param("predict")
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    for index, page_metadata in enumerate(pages_metadata):
        session_uuid = await create_session(CREDENTIALS, pg_pool, project_id, plan_id, user_id, page_metadata["page_number"])
        page_metadata["session_uuid"] = session_uuid

    pdf_path = Path("/tmp/floor_plan.PDF")
    query = f"SELECT user_id FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s)"
    user_id_owner = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,), fetch=True))
    if user_id_owner:
        GCS_URL_floorplan = await download_floorplan(plan_id, project_id, user_id_owner[0]["user_id"], CREDENTIALS, pg_pool, destination_path=pdf_path)
        user_id_owner = user_id_owner[0]["user_id"]
    else:
        GCS_URL_floorplan = await download_floorplan(plan_id, project_id, user_id, CREDENTIALS, pg_pool, destination_path=pdf_path)
        user_id_owner = user_id
    logging.info("SYSTEM: Floorplan Downloaded")

    #client = CloudStorageClient()
    #bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    #blob_path = f"tmp/{user_id.lower()}/{project_id.lower()}/{plan_id.lower()}/floorplan_structured_2d.json"
    #blob = bucket.blob(blob_path)
    #if blob.exists():
    #    blob.delete()

    size_in_bytes = Path(pdf_path).stat().st_size
    n_pages = pdfinfo_from_path(pdf_path)["Pages"]
    await insert_plan(
        project_id,
        user_id_owner,
        "IN PROGRESS",
        pg_pool,
        CREDENTIALS,
        plan_id=plan_id,
        size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=n_pages,
    )
    for index, page_metadata in enumerate(pages_metadata):
        session_is_active = await is_session_active(CREDENTIALS, pg_pool, page_metadata["session_uuid"], project_id, plan_id, user_id, page_metadata["page_number"])
        if session_is_active:
            await insert_page(
                plan_id,
                user_id,
                project_id,
                page_metadata["page_number"],
                False,
                "IN PROGRESS",
                pg_pool,
                CREDENTIALS,
            )
    hyperparameters = load_hyperparameters()
    if hyperparameters["modelling"]["enable_early_stopping"]:
        pages_metadata = await enforce_early_stopping(CREDENTIALS, pg_pool, project_id, plan_id, user_id, pdf_path, pages_metadata)
    ip_address = request.headers.get("X-Client-IP", (request.client.host if request.client else None))
    vertex_ai_client, vertex_ai_generation_config, is_cached = load_vertex_ai_client(
        CREDENTIALS,
        ip_address,
        prompts=[VISUAL_GROUNDING_DETECTOR]
    )
    elevation_map = await map_floorplan_to_multipage_elevation(
        CREDENTIALS,
        pg_pool,
        project_id,
        plan_id,
        ip_address,
        pdf_path
    )
    pages_metadata = await load_visual_grounding(
        CREDENTIALS,
        pg_pool,
        project_id,
        plan_id,
        user_id_owner,
        ip_address,
        pages_metadata,
        vertex_ai_client=vertex_ai_client,
        vertex_ai_generation_config=vertex_ai_generation_config,
        is_cached=is_cached
    )

    walls_2d_all = dict(pages=list())
    status = "COMPLETED"
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=len(pages_metadata),
        pool_maxsize=len(pages_metadata),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    subscriber_client = load_subscriber_client(CREDENTIALS)
    query = f"UPDATE {CREDENTIALS["CloudSQL"]["table_name_plans"]} SET multipage_elevation_map = %s WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
    await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(json.dumps(elevation_map), project_id, plan_id)))
    try:
        with ThreadPoolExecutor(max_workers=20) as executor:
            for index, page_metadata in enumerate(pages_metadata):
                page_number = page_metadata["page_number"]
                query = f"UPDATE {CREDENTIALS["CloudSQL"]["table_name_pages"]} SET mask_factor = %s, bounding_box_offsets = %s WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
                await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(json.dumps(page_metadata["mask_factor"]), json.dumps(page_metadata["bounding_box_offsets"]), project_id, plan_id, page_number)))
                if index != 0 and index % 25 == 0:
                    sleep(120)
                id_token = load_floorplan_to_structured_2d_ID_token(CREDENTIALS)
                elevation_pages = load_elevation_map(elevation_map, page_number)
                if not page_metadata.get("architectural_scale"):
                    query = f"SELECT scale FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s"
                    architectural_scales = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, page_number,), fetch=True))
                    if architectural_scales:
                        for architectural_scale in architectural_scales:
                            if architectural_scale["scale"]:
                                page_metadata["architectural_scale"] = architectural_scale["scale"]
                executor.submit(
                    floorplan_to_structured_2d,
                    CREDENTIALS,
                    session,
                    id_token,
                    project_id,
                    plan_id,
                    user_id,
                    page_number,
                    page_metadata["mask_factor"],
                    page_metadata["bounding_box_offsets"],
                    elevation_pages,
                    page_metadata.get("architectural_scale"),
                    page_metadata["session_uuid"],
                    model,
                    predict
                )
            query_payloads = [dict(project_id=project_id, plan_id=plan_id, page_number=page_metadata["page_number"]) for page_metadata in pages_metadata]
            timeout = from_unix_epoch() + 7200
            all_pages_extracted = False
            sleep_time = 1
            while from_unix_epoch() < timeout:
                notifications_arrived_all, acknowledged_queries = query_subscriber_messages(CREDENTIALS, subscriber_client, query_payloads)
                for acknowledged_query in acknowledged_queries:
                    query_payloads.remove(acknowledged_query)
                if notifications_arrived_all:
                    all_pages_extracted = True
                    break
                sleep(sleep_time)
            if not all_pages_extracted:
                raise AssertionError(f"Extraction has failed for PAGE(s): {[query_payload["page_number"] for query_payload in query_payloads]}")
            for page_metadata in pages_metadata:
                page_number = page_metadata["page_number"]
                query = f"SELECT page_section_number, model_2d, scale FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
                query_output_sections = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, page_number,), fetch=True))
                for query_output in query_output_sections:
                    walls_2d = json.loads(query_output["model_2d"]) if isinstance(query_output["model_2d"], str) else query_output["model_2d"]
                    if walls_2d["walls_2d"] and walls_2d["polygons"]:
                        page = dict(
                            plan_id=plan_id,
                            page_number=page_number,
                            page_section_number=query_output["page_section_number"],
                            scale=query_output["scale"],
                            walls_2d=walls_2d["walls_2d"],
                            polygons=walls_2d["polygons"],
                            **walls_2d["metadata"]
                        )
                        walls_2d_all["pages"].append(page)
    except Exception as e:
        stacktrace = traceback.format_exc()
        logging.error(f"SYSTEM: Floorplan extraction failed with error: {e}; stacktrace: {stacktrace}")
        status = "FAILED"
    await insert_plan(
        project_id,
        user_id_owner,
        status,
        pg_pool,
        CREDENTIALS,
        plan_id=plan_id,
        size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=n_pages,
    )

    #with open("/tmp/floorplan_structured_2d.json", 'w') as f:
    #    json.dump(walls_2d_all, f, indent=4)
    #blob.upload_from_filename("/tmp/floorplan_structured_2d.json")
    return respond_with_UI_payload(walls_2d_all)


@app.post("/undo_floorplan_to_2d")
async def undo_floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 2D Model Cancellation Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    query = (
        f"SELECT session_id FROM {CREDENTIALS["CloudSQL"]["table_name_sessions"]} "
        f"WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND LOWER(user_id) = LOWER(%s) AND page_number = %s AND status = 'ACTIVE' "
        f"ORDER BY created_at DESC LIMIT 1;"
    )
    query_output = await run_in_threadpool(partial(
        pg_run,
        CREDENTIALS,
        pg_pool,
        query,
        params=(project_id, plan_id, user_id, page_number,),
        fetch=True
    ))
    if not query_output:
        return
    session_id = query_output[0]["session_id"]
    query = (
        f"UPDATE {CREDENTIALS["CloudSQL"]["table_name_sessions"]} SET status = 'ABORTED' "
        f"WHERE LOWER(session_id) = LOWER(%s);"
    )
    await run_in_threadpool(partial(
        pg_run,
        CREDENTIALS,
        pg_pool,
        query,
        params=(session_id,),
    ))
    query = (
        f"SELECT status FROM {CREDENTIALS["CloudSQL"]["table_name_sessions"]} "
        f"WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
    )
    query_output = await run_in_threadpool(partial(
        pg_run,
        CREDENTIALS,
        pg_pool,
        query,
        params=(project_id, plan_id, page_number,),
        fetch=True
    ))
    status = "NOT STARTED"
    if query_output:
        statuses_all = [row["status"] for row in query_output]
        if "COMPLETED" in statuses_all:
            status = "COMPLETED"
    await insert_page(
        plan_id,
        user_id,
        project_id,
        page_number,
        False,
        status,
        pg_pool,
        CREDENTIALS,
    )


@app.post("/load_2d_revision")
async def load_2d_revision(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    revision_number = parameters.get("revision_number") or body.get("revision_number")
    logging.info(f"SYSTEM: Received Floorplan 2D Model (Revision: {revision_number}) Load Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT model FROM {CREDENTIALS["CloudSQL"]["table_name_model_revisions_2d"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s AND revision_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, page_number, page_section_number, revision_number,), fetch=True))
    walls_2d_JSON, polygons_JSON = list(), list()
    if query_output and query_output[0]["model"] is not None:
        walls_2d = json.loads(query_output[0]["model"]) if isinstance(query_output[0]["model"], str) else query_output[0]["model"]
        walls_2d_JSON, polygons_JSON = walls_2d["walls_2d"], walls_2d["polygons"]
    return respond_with_UI_payload(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), disable_caching=True)


@app.post("/load_available_revision_numbers_2d")
async def load_available_revision_numbers_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    logging.info(f"SYSTEM: Received Available Revisions Load Request for 2D Model")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT revision_number FROM {CREDENTIALS["CloudSQL"]["table_name_model_revisions_2d"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, page_number, page_section_number,), fetch=True))
    revision_numbers = list()
    if query_output:
        for revision in query_output:
            if revision["revision_number"] is not None:
                revision_numbers.append(revision["revision_number"])

    return respond_with_UI_payload(revision_numbers)


@app.post("/load_2d_all")
async def load_2d_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    page_number = parameters.get("page_number", '') or body.get("page_number", '')
    load_lazy = parameters.get("load_lazy", "true") or body.get("load_lazy", "true")
    logging.info("SYSTEM: Received All Floorplan 2D Models Load Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    is_admin = await access_control.is_admin(user_id)
    if bool(is_admin):
        if is_admin.super:
            where_clause = "TRUE"

        elif is_admin.local:
            peers = await access_control.load_organization_users(user_id)
            where_clause = "LOWER(m.user_id) = ANY(%s)"

    else:
        peers = await access_control.load_regional_users(user_id)
        region_names = await access_control.load_user_region_names(user_id)
        where_clause = "LOWER(m.user_id) = ANY(%s) AND LOWER(pr.\"FBM_branch\") = ANY(%s)"

    if load_lazy == "false":
        status = "IN PROGRESS"
        query = f"""
            SELECT pl.pages
            FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} pl
            FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr
            WHERE
                LOWER(pl.project_id) = LOWER(%s)
                AND LOWER(pl.plan_id) = LOWER(%s)
                AND LOWER(pl.user_id) = ANY(%s)
                AND LOWER(pr.\"FBM_branch\") = ANY(%s)
        """
        query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, peers, region_names), fetch=True))
        if not query_output:
            return respond_with_UI_payload(dict(error="Floor Plan already exists"))
        n_pages = query_output[0]["pages"]
        timeout = from_unix_epoch() + (n_pages * 900)
        while from_unix_epoch() < timeout:
            query = f"SELECT status FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
            try:
                query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,), fetch=True))
                status = query_output[0]["status"]
                if status == "COMPLETED":
                    break
            except IndexError:
                return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
            sleep(5)
        if status != "COMPLETED":
            return respond_with_UI_payload(dict(error=f"Floor Plan extraction not completed within {(n_pages * 900)/60} minutes"), status_code=500)

    walls_2d_all = dict(pages=list())
    if page_number != '':
        query = f"""
            SELECT
                m.page_number,
                m.page_section_number,
                m.scale,
                m.model_2d
            FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} m
            JOIN {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr
                ON m.project_id = pr.project_id
            WHERE
                LOWER(m.project_id) = LOWER(%s)
                AND LOWER(m.plan_id) = LOWER(%s)
                AND m.page_number = %s
                AND {where_clause}
            ORDER BY m.page_number
        """
        if bool(is_admin) and is_admin.super:
            params = (project_id, plan_id, int(page_number),)
        elif bool(is_admin) and is_admin.local:
            params = (project_id, plan_id, int(page_number), peers,)
        else:
            params = (project_id, plan_id, int(page_number), peers, region_names,)
    else:
        query = f"""
            SELECT
                m.page_number,
                m.page_section_number,
                m.scale,
                m.model_2d
            FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} m
            JOIN {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr
                ON m.project_id = pr.project_id
            WHERE
                LOWER(m.project_id) = LOWER(%s)
                AND LOWER(m.plan_id) = LOWER(%s)
                AND {where_clause}
            ORDER BY
                m.page_number,
                m.page_section_number
        """
        if bool(is_admin) and is_admin.super:
            params = (project_id, plan_id,)
        elif bool(is_admin) and is_admin.local:
            params = (project_id, plan_id, peers,)
        else:
            params = (project_id, plan_id, peers, region_names,)
    rows = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=params, fetch=True))

    page_to_model_2d_minimal = dict()
    rows = sorted(rows, key=lambda row: f"{row["page_number"]}-{row["page_section_number"]}")
    for row in rows:
        if not row["model_2d"]:
            continue

        walls_2d = json.loads(row["model_2d"]) if isinstance(row["model_2d"], str) else row["model_2d"]
        page_to_model_2d_minimal[row["page_number"]] = dict(
            page_section_number=row["page_section_number"],
            scale=row["scale"],
            walls_2d=walls_2d["walls_2d"],
            polygons=walls_2d["polygons"],
            metadata=walls_2d["metadata"]
        )
        if walls_2d["walls_2d"] and walls_2d["polygons"]:
            page = {
                "plan_id": plan_id,
                "page_number": row["page_number"],
                "page_section_number": row["page_section_number"],
                "scale": row["scale"],
                "walls_2d": walls_2d.get("walls_2d", list()),
                "polygons": walls_2d.get("polygons", list()),
                **walls_2d.get("metadata", dict()),
            }
            walls_2d_all["pages"].append(page)
    for page_number in list(set(page_to_model_2d_minimal.keys()) - set([page["page_number"] for page in walls_2d_all["pages"]])):
        page = {
            "plan_id": plan_id,
            "page_number": page_number,
            "page_section_number": page_to_model_2d_minimal[page_number]["page_section_number"],
            "scale": page_to_model_2d_minimal[page_number]["scale"],
            "walls_2d": page_to_model_2d_minimal[page_number]["walls_2d"],
            "polygons": page_to_model_2d_minimal[page_number]["polygons"],
            **page_to_model_2d_minimal[page_number]["metadata"],
        }
        walls_2d_all["pages"].append(page)

    return respond_with_UI_payload(walls_2d_all)


@app.post("/load_layout_2d_all")
async def load_layout_2d_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    page_number = parameters.get("page_number", '') or body.get("page_number", '')
    load_lazy = parameters.get("load_lazy", "true") or body.get("load_lazy", "true")
    logging.info("SYSTEM: Received All Floorplan 2D Layouts Load Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Xtimator application")
        return respond_with_UI_payload(is_user_not_authenticated)
    is_admin = await access_control.is_admin(user_id)
    if bool(is_admin):
        if is_admin.super:
            where_clause = "TRUE"

        elif is_admin.local:
            peers = await access_control.load_organization_users(user_id)
            where_clause = "LOWER(m.user_id) = ANY(%s)"

    else:
        peers = await access_control.load_regional_users(user_id)
        region_names = await access_control.load_user_region_names(user_id)
        where_clause = "LOWER(m.user_id) = ANY(%s) AND LOWER(pr.\"FBM_branch\") = ANY(%s)"

    if load_lazy == "false":
        status = "IN PROGRESS"
        query = f"""
            SELECT pl.pages
            FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} pl
            FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr
            WHERE
                LOWER(pl.project_id) = LOWER(%s)
                AND LOWER(pl.plan_id) = LOWER(%s)
                AND LOWER(pl.user_id) = ANY(%s)
                AND LOWER(pr.\"FBM_branch\") = ANY(%s)
        """
        query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, peers, region_names), fetch=True))
        if not query_output:
            return respond_with_UI_payload(dict(error="Floor Plan already exists"))
        n_pages = query_output[0]["pages"]
        timeout = from_unix_epoch() + (n_pages * 900)
        while from_unix_epoch() < timeout:
            query = f"SELECT status FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
            try:
                query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,), fetch=True))
                status = query_output[0]["status"]
                if status == "COMPLETED":
                    break
            except IndexError:
                return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
            sleep(5)
        if status != "COMPLETED":
            return respond_with_UI_payload(dict(error=f"Floor Plan extraction not completed within {(n_pages * 900)/60} minutes"), status_code=500)

    walls_2d_all = dict(pages=list())
    if page_number != '':
        query = f"""
            SELECT
                m.page_number,
                m.page_section_number,
                m.scale,
                m.layout_2d
            FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} m
            JOIN {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr
                ON m.project_id = pr.project_id
            WHERE
                LOWER(m.project_id) = LOWER(%s)
                AND LOWER(m.plan_id) = LOWER(%s)
                AND m.page_number = %s
                AND {where_clause}
            ORDER BY m.page_number
        """
        if bool(is_admin) and is_admin.super:
            params = (project_id, plan_id, int(page_number),)
        else:
            params = (project_id, plan_id, int(page_number), peers, region_names,)
    else:
        query = f"""
            SELECT
                m.page_number,
                m.page_section_number,
                m.scale,
                m.layout_2d
            FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} m
            JOIN {CREDENTIALS["CloudSQL"]["table_name_projects"]} pr
                ON m.project_id = pr.project_id
            WHERE
                LOWER(m.project_id) = LOWER(%s)
                AND LOWER(m.plan_id) = LOWER(%s)
                AND {where_clause}
            ORDER BY
                m.page_number,
                m.page_section_number
        """
        if bool(is_admin) and is_admin.super:
            params = (project_id, plan_id,)
        else:
            params = (project_id, plan_id, peers, region_names,)
    rows = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=params, fetch=True))

    page_to_layout_2d_minimal = dict()
    rows = sorted(rows, key=lambda row: f"{row["page_number"]}-{row["page_section_number"]}")
    for row in rows:
        if not row["layout_2d"]:
            continue

        walls_2d = json.loads(row["layout_2d"]) if isinstance(row["layout_2d"], str) else row["layout_2d"]
        page_to_layout_2d_minimal[row["page_number"]] = dict(
            page_section_number=row["page_section_number"],
            scale=row["scale"],
            walls_2d=walls_2d["walls_2d"],
            polygons=walls_2d["polygons"],
            metadata=walls_2d["metadata"]
        )
        if walls_2d["walls_2d"] and walls_2d["polygons"]:
            page = {
                "plan_id": plan_id,
                "page_number": row["page_number"],
                "page_section_number": row["page_section_number"],
                "scale": row["scale"],
                "walls_2d": walls_2d.get("walls_2d", list()),
                "polygons": walls_2d.get("polygons", list()),
                **walls_2d.get("metadata", dict()),
            }
            walls_2d_all["pages"].append(page)
    for page_number in list(set(page_to_layout_2d_minimal.keys()) - set([page["page_number"] for page in walls_2d_all["pages"]])):
        page = {
            "plan_id": plan_id,
            "page_number": page_number,
            "page_section_number": page_to_layout_2d_minimal[page_number]["page_section_number"],
            "scale": page_to_layout_2d_minimal[page_number]["scale"],
            "walls_2d": page_to_layout_2d_minimal[page_number]["walls_2d"],
            "polygons": page_to_layout_2d_minimal[page_number]["polygons"],
            **page_to_layout_2d_minimal[page_number]["metadata"],
        }
        walls_2d_all["pages"].append(page)

    return respond_with_UI_payload(walls_2d_all)


@app.post("/load_2d_timestamp")
async def load_2d_timestamp(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number", '') or body.get("page_number", '')

    query = f"SELECT created_at FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s LIMIT 1;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, page_number,), fetch=True))
    created_at = query_output[0]["created_at"]
    return respond_with_UI_payload(dict(timestamp=created_at.isoformat()))


@app.post("/update_floorplan_to_2d")
async def update_floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    scale = parameters.get("scale") or body.get("scale")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    index = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    logging.info("SYSTEM: Received a Floorplan 2D Model Update Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    user_scope = await access_control.load_scope(user_id, project_id, plan_id, index)
    if not user_scope.update:
         return respond_with_UI_payload(dict(
             email=user_id,
             permission="denied",
             message=(
                "You don't have permission to update this item. "
                "Please contact your administrator if you need access."
            ),
         ))

    hyperparameters = load_hyperparameters()

    query = f"SELECT model_2d->'walls_2d' AS walls_2d, model_2d->'polygons' AS polygons FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
    walls_2d = query_output[0]["walls_2d"]
    walls_2d_JSON_outdated = json.loads(walls_2d) if isinstance(walls_2d, str) else walls_2d
    polygons = query_output[0]["polygons"]
    polygons_JSON_outdated = json.loads(polygons) if isinstance(polygons, str) else polygons

    for wall in walls_2d_JSON[:]:
        for drywall in wall["polygons_drywall"][:]:
            if drywall["type"].upper() == "DISABLED":
                drywall["color"] = [255, 0, 0]
                continue
            if drywall["type_stacked"]:
                color_stacked = list()
                for drywall_type in drywall["type_stacked"]:
                    drywall_template = query_drywall(drywall_type, DRYWALL_TEMPLATES)
                    if not drywall_template:
                        continue
                    color_stacked.append(drywall_template["color_code"][::-1])
                drywall["color_stacked"] = color_stacked
            else:
                drywall_template = query_drywall(drywall["type"], DRYWALL_TEMPLATES)
                if not drywall_template:
                    continue
                drywall["color"] = drywall_template["color_code"][::-1]
    for polygon in polygons_JSON[:]:
        if polygon["polygon_drywall"]["type"].upper() == "DISABLED":
            polygon["polygon_drywall"]["color"] = [137, 137, 137]
            continue
        drywall_template = query_drywall(polygon["polygon_drywall"]["type"], DRYWALL_TEMPLATES)
        if not drywall_template:
            continue
        polygon["polygon_drywall"]["color"] = drywall_template["color_code"][::-1]

    plan = FloorPlan(hyperparameters)
    wall_lines = [[[wall_2d["wall_line"][0]['x'], wall_2d["wall_line"][0]['y'], wall_2d["wall_line"][1]['x'], wall_2d["wall_line"][1]['y']]] for wall_2d in walls_2d_JSON]
    wall_lines = plan.normalize(wall_lines)
    wall_line_ids = [wall_2d["id"] for wall_2d in walls_2d_JSON]

    query = f"SELECT model_2d->'metadata' AS metadata FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
    metadata = query_output[0]["metadata"]
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    height, width = metadata["height_in_pixels"], metadata["width_in_pixels"],
    DPI = metadata["DPI"]
    scale_x = width / 1920
    scale_y = height / 1080
    resolution_scale = (scale_x, scale_y,)

    polygons_JSON = plan.reshape_polygons(polygons_JSON, walls_2d_JSON, architectural_scale=scale, resolution_scale=resolution_scale, DPI=DPI)
    #polygons_JSON_sharded = list()
    #polygon_idx = 0
    #for polygon in polygons_JSON:
    #    perimeter_lines_contour = plan.load_perimeter(polygon["vertices"], wall_lines, scale=resolution_scale)
    #    polygons_sharded = plan.shard_polygon(polygon["vertices"], perimeter_lines_contour, scale, resolution_scale)
    #    for (polygon_area_shoelace, area, vertices) in polygons_sharded:
    #        polygon["id"] = polygon_idx
    #        polygon["area"] = area
    #        polygon["polygon_area_shoelace"] = polygon_area_shoelace
    #        polygon["vertices"] = vertices
    #        polygons_JSON_sharded.append(polygon)
    #        polygon_idx += 1
    #polygons_JSON = polygons_JSON_sharded
    for polygon in polygons_JSON[:]:
        perimeter_lines_contour = plan.load_perimeter(polygon["vertices"], wall_lines, scale=resolution_scale)
        perimeter_wall_line_ids = [wall_line_ids[wall_lines.index(perimeter_line_contour)] for perimeter_line_contour in perimeter_lines_contour]
        polygon_ids_drywall_interior = list()
        for perimeter_wall_line_id in perimeter_wall_line_ids:
            for wall_2d in walls_2d_JSON[:]:
                if wall_2d["id"] == perimeter_wall_line_id:
                    drywall_indices = plan.direction_polygon_interior(polygon["vertices"], wall_2d)
                    for drywall_index in drywall_indices:
                        polygon_ids_drywall_interior.append(f"{perimeter_wall_line_id}.{drywall_index}")
                        if drywall_index == 'a':
                            wall_2d["polygons_drywall"][0]["room_name"] = polygon["room_name"]
                        else:
                            wall_2d["polygons_drywall"][1]["room_name"] = polygon["room_name"]
                    break
        polygon["polygon_ids_drywall_interior"] = polygon_ids_drywall_interior

    walls_2d_JSON, polygons_JSON = plan.detect_and_recover_from_polygon_drywall_anomaly(walls_2d_JSON, polygons_JSON, walls_2d_JSON_outdated, polygons_JSON_outdated)
    await insert_model_2d(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, None, None, pg_pool, CREDENTIALS, page_section_number=page_section_number)
    await insert_model_2d_revision(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS, page_section_number=page_section_number)
    logging.info("SYSTEM: Floorplan 2D Model Updated Successfully")
    return respond_with_UI_payload(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), disable_caching=True)


@app.post("/update_floorplan_to_layout_2d")
async def update_floorplan_to_layout_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    scale = parameters.get("scale") or body.get("scale")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    index = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    logging.info("SYSTEM: Received a Floorplan 2D Model Update Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    user_scope = await access_control.load_scope(user_id, project_id, plan_id, index)
    if not user_scope.update:
         return respond_with_UI_payload(dict(
             email=user_id,
             permission="denied",
             message=(
                "You don't have permission to update this item. "
                "Please contact your administrator if you need access."
            ),
         ))

    hyperparameters = load_hyperparameters()

    query = f"SELECT layout_2d->'walls_2d' AS walls_2d, layout_2d->'polygons' AS polygons FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
    walls_2d = query_output[0]["walls_2d"]
    walls_2d_JSON_outdated = json.loads(walls_2d) if isinstance(walls_2d, str) else walls_2d
    polygons = query_output[0]["polygons"]
    polygons_JSON_outdated = json.loads(polygons) if isinstance(polygons, str) else polygons

    plan = FloorPlan(hyperparameters)
    wall_lines = [[[wall_2d["wall_line"][0]['x'], wall_2d["wall_line"][0]['y'], wall_2d["wall_line"][1]['x'], wall_2d["wall_line"][1]['y']]] for wall_2d in walls_2d_JSON]
    wall_lines = plan.normalize(wall_lines)
    wall_line_ids = [wall_2d["id"] for wall_2d in walls_2d_JSON]

    query = f"SELECT layout_2d->'metadata' AS metadata FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
    metadata = query_output[0]["metadata"]
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    height, width = metadata["height_in_pixels"], metadata["width_in_pixels"]
    DPI = metadata["DPI"]
    scale_x = width / 1920
    scale_y = height / 1080
    resolution_scale = (scale_x, scale_y,)

    polygons_JSON = plan.reshape_polygons(polygons_JSON, walls_2d_JSON, architectural_scale=scale, resolution_scale=resolution_scale, DPI=DPI)
    for polygon in polygons_JSON[:]:
        perimeter_lines_contour = plan.load_perimeter(polygon["vertices"], wall_lines, scale=resolution_scale)
        perimeter_wall_line_ids = [wall_line_ids[wall_lines.index(perimeter_line_contour)] for perimeter_line_contour in perimeter_lines_contour]
        polygon_ids_drywall_interior = list()
        for perimeter_wall_line_id in perimeter_wall_line_ids:
            for wall_2d in walls_2d_JSON[:]:
                if wall_2d["id"] == perimeter_wall_line_id:
                    drywall_indices = plan.direction_polygon_interior(polygon["vertices"], wall_2d)
                    for drywall_index in drywall_indices:
                        polygon_ids_drywall_interior.append(f"{perimeter_wall_line_id}.{drywall_index}")
                        if drywall_index == 'a':
                            wall_2d["polygons_drywall"][0]["room_name"] = polygon["room_name"]
                        else:
                            wall_2d["polygons_drywall"][1]["room_name"] = polygon["room_name"]
                    break
        polygon["polygon_ids_drywall_interior"] = polygon_ids_drywall_interior

    walls_2d_JSON, polygons_JSON = plan.detect_and_recover_from_polygon_drywall_anomaly(walls_2d_JSON, polygons_JSON, walls_2d_JSON_outdated, polygons_JSON_outdated)
    await insert_layout_2d(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, None, None, pg_pool, CREDENTIALS, page_section_number=page_section_number)
    logging.info("SYSTEM: Floorplan 2D Layout Updated Successfully")
    return respond_with_UI_payload(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), disable_caching=True)


@app.post("/update_scale")
async def update_scale(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    scale = parameters.get("scale") or body.get("scale")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    logging.info("SYSTEM: Received a Scale Update Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    user_scope = await access_control.load_scope(user_id, project_id, plan_id, page_number)
    if not user_scope.update:
         return respond_with_UI_payload(dict(
             email=user_id,
             permission="denied",
             message=(
                "You don't have permission to update this item. "
                "Please contact your administrator if you need access."
            ),
         ))
    query = f"UPDATE {CREDENTIALS["CloudSQL"]["table_name_models"]} SET scale = %s WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(scale, project_id, plan_id, page_number, page_section_number,)))
    logging.info("SYSTEM: Scale Updated Successfully")

    if walls_2d_JSON and polygons_JSON:
        hyperparameters = load_hyperparameters()
        plan = FloorPlan(hyperparameters)
        walls_2d_JSON, polygons_JSON = plan.scale_walls_2d_and_polygons(walls_2d_JSON, polygons_JSON, scale)
        logging.info(f"SYSTEM: Walls 2D and Polygons computed with scale: {scale}")
        await insert_model_2d(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, page_number, plan_id, user_id, project_id, None, None, pg_pool, CREDENTIALS, page_section_number=page_section_number)
        await insert_model_2d_revision(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, page_number, plan_id, user_id, project_id, pg_pool, CREDENTIALS, page_section_number=page_section_number)
        return respond_with_UI_payload(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), disable_caching=True)


@app.post("/load_scale")
async def load_scale(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    logging.info("SYSTEM: Received a Scale Update Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT scale FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, page_number, page_section_number), fetch=True))
    return respond_with_UI_payload(dict(scale=query_output[0]["scale"]))


@app.post("/floorplan_to_3d")
async def floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    scale = parameters.get("scale") or body.get("scale")
    index = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    logging.info("SYSTEM: Received a Floorplan 3D Model Generation Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    model_2d_path = "/tmp/walls_2d.json"
    with open(model_2d_path, 'w') as f:
        json.dump(walls_2d_JSON, f)
    polygons_path = "/tmp/polygons.json"
    with open(polygons_path, 'w') as f:
        json.dump(polygons_JSON, f)

    query = f"SELECT model_2d->'metadata' AS metadata FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index,), fetch=True))
    metadata = query_output[0]["metadata"]
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    if not walls_2d_JSON or not polygons_JSON:
        return respond_with_UI_payload(dict(walls_3d=list(), polygons=list(), metadata=metadata))

    hyperparameters = load_hyperparameters()
    if not scale:
        query = f"SELECT scale FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
        query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
        scale = query_output[0]["scale"]
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    walls_3d, polygons_3d, walls_3d_path, polygons_3d_path = floor_plan_modeller_3d.extrapolate(scale, model_2d_path=model_2d_path, polygons_path=polygons_path)
    #gltf_paths = floor_plan_modeller_3d.gltf(model_2d_path=model_2d_path, polygons_path=polygons_path)
    model_3d_path = floor_plan_modeller_3d.save_plot_3d(walls_3d_path, polygons_3d_path)
    model_3d_path_sectioned = model_3d_path.parent.joinpath(f"{model_3d_path.stem}_sectioned_{page_section_number.replace('/', '_')}").with_suffix(".png")
    model_3d_path.rename(model_3d_path_sectioned)
    await upload_floorplan(model_3d_path_sectioned, plan_id, project_id, user_id, CREDENTIALS, pg_pool, index=str(index).zfill(4))
    #for gltf_path in gltf_paths:
    #    upload_floorplan(gltf_path, plan_id, project_id, user_id, CREDENTIALS, pg_pool, index=str(index).zfill(4), directory="gltf")
    await insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, page_section_number, plan_id, project_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: A 3D Model of the Floorplan Generated Successfully")

    return respond_with_UI_payload(dict(walls_3d=walls_3d, polygons=polygons_3d, metadata=metadata))


@app.post("/load_3d_revision")
async def load_3d_revision(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    revision_number = parameters.get("revision_number") or body.get("revision_number")
    logging.info(f"SYSTEM: Received Floorplan 3D Model (Revision: {revision_number}) Load Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT model FROM {CREDENTIALS["CloudSQL"]["table_name_model_revisions_3d"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND revision_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, int(page_number), int(revision_number),), fetch=True))
    walls_3d_JSON = dict()
    if query_output and query_output[0]["model"] is not None:
        walls_3d_JSON = json.loads(query_output[0]["model"])

    return respond_with_UI_payload(walls_3d_JSON)


@app.post("/load_available_revision_numbers_3d")
async def load_available_revision_numbers_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info(f"SYSTEM: Received Available Revisions Load Request for 3D Model")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT revision_number FROM {CREDENTIALS["CloudSQL"]["table_name_model_revisions_3d"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, int(page_number),)))
    revision_numbers = list()
    if query_output:
        for revision in query_output:
            if revision["revision_number"] is not None:
                revision_numbers.append(revision["revision_number"])

    return respond_with_UI_payload(revision_numbers)


@app.post("/update_floorplan_to_3d")
async def update_floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d = parameters.get("walls_3d") or body.get("walls_3d")
    polygons_3d = parameters.get("polygons") or body.get("polygons")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    scale = parameters.get("scale") or body.get("scale")
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 3D Model Update Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    await insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, project_id, pg_pool, CREDENTIALS)
    await insert_model_3d_revision(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Floorplan 3D Model Updated Successfully")


@app.post("/generate_drywall_overlaid_floorplan_download_signed_URL")
async def generate_drywall_overlaid_floorplan_download_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    index = parameters.get("page_number") or body.get("page_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    load_lazy = parameters.get("load_lazy", "true") or body.get("load_lazy", "true")
    logging.info("SYSTEM: Received Signed Floorplan download URL generation Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    if load_lazy == "false":
        status = "IN PROGRESS"
        query = f"SELECT pages FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
        query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,), fetch=True))
        if not query_output:
            return respond_with_UI_payload(dict(error="Floor Plan already exists"))
        n_pages = query_output[0]["pages"]
        timeout = from_unix_epoch() + (n_pages * 900)
        while from_unix_epoch() < timeout:
            query = f"SELECT status FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
            try:
                query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,), fetch=True))
                status = query_output[0]["status"]
                if status == "COMPLETED":
                    break
            except IndexError:
                return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
            sleep(5)
        if status != "COMPLETED":
            return respond_with_UI_payload(dict(error=f"Floor Plan extraction not completed within {(n_pages * 900)/60} minutes"), status_code=500)

    query = f"SELECT target_drywalls FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, int(index),), fetch=True))
    drywall_overlaid_floorplan_source_path = query_output[0]["target_drywalls"]
    _, _, _, blob_path = drywall_overlaid_floorplan_source_path.split('/', 3)

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="GET",
    )

    return url


@app.post("/remove_floorplan")
async def remove_floorplan(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received a Plan Deletion Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT * FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND LOWER(user_id) = LOWER(%s);"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, user_id,), fetch=True))
    if not query_output:
        return respond_with_UI_payload(dict(status="FAILED", message="Plan: {} cannot be deleted".format(plan_id)))
    await delete_floorplan(project_id, plan_id, user_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Plan Deleted Successfully")
    return respond_with_UI_payload(dict(status="SUCCESS", message=f"Plan: {plan_id} Deleted Successfully"))


@app.post("/remove_project")
async def remove_project(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    logging.info("SYSTEM: Received a Project Deletion Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT * FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(created_by) = LOWER(%s);"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, user_id,), fetch=True))
    if not query_output:
        return respond_with_UI_payload(dict(status="FAILED", message="Project: {} cannot be deleted".format(project_id)))
    await delete_project(project_id, user_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Project Deleted Successfully")
    return respond_with_UI_payload(dict(status="SUCCESS", message=f"Project: {project_id} Deleted Successfully"))


@app.post("/load_waste_average")
async def load_waste_average(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d", list()) or body.get("walls_2d", list())
    polygons_JSON = parameters.get("polygons", list()) or body.get("polygons", list())
    page_number = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT waste_average FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, page_number, page_section_number,), fetch=True))
    waste_average = query_output[0]["waste_average"]
    if waste_average is not None:
        return respond_with_UI_payload(dict(waste_average_in_percentage=waste_average))

    _, waste_average, _ = load_drywall_weights(
        walls_2d_JSON,
        polygons_JSON,
        compute_waste_average_standard=True,
        drywall_templates=DRYWALL_TEMPLATES
    )
    return respond_with_UI_payload(dict(waste_average_in_percentage=waste_average))


@app.post("/load_drywall_negate_opening_area_threshold")
async def load_drywall_negate_opening_area_threshold(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    page_number = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT drywall_negate_opening_area_threshold FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, page_number, page_section_number,), fetch=True))
    drywall_negate_opening_area_threshold = query_output[0]["drywall_negate_opening_area_threshold"]
    return respond_with_UI_payload(dict(drywall_negate_opening_area_threshold=drywall_negate_opening_area_threshold))


@app.post("/compute_takeoff")
async def compute_takeoff(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d", list()) or body.get("walls_2d", list())
    polygons_JSON = parameters.get("polygons", list()) or body.get("polygons", list())
    waste_factor_average = load_parameter_first_not_none(parameters.get("waste_factor_average"), body.get("waste_factor_average"))
    drywall_negate_opening_area_threshold = load_parameter_first_not_none(parameters.get("drywall_negate_opening_area_threshold"), body.get("drywall_negate_opening_area_threshold"))
    index = parameters.get("page_number") or body.get("page_number")
    page_section_number = parameters.get("page_section_number") or body.get("page_section_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    revision_number = parameters.get("revision_number", '') or body.get("revision_number", '')
    load_preview = load_parameter_first_not_none(parameters.get("load_preview"), body.get("load_preview"))
    logging.info("SYSTEM: Received a Drywall Takeoff computation Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)
    user_scope = await access_control.load_scope(user_id, project_id, plan_id, index)
    if load_preview == False and not user_scope.update:
         return respond_with_UI_payload(dict(
             email=user_id,
             permission="denied",
             message=(
                "You don't have permission to update this item. "
                "Please contact your administrator if you need access."
            ),
         ))
    query = f"SELECT scale FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
    scale = query_output[0]["scale"]

    if not walls_2d_JSON:
        if revision_number:
            query = f"SELECT model->'walls_2d' AS walls_2d FROM {CREDENTIALS["CloudSQL"]["table_name_model_revisions_2d"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s AND revision_number = %s;"
            query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number, revision_number,), fetch=True))
        else:
            query = f"SELECT model_2d->'walls_2d' AS walls_2d FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
            query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
        walls_2d_JSON = query_output[0].get("walls_2d")
    
        if walls_2d_JSON is None:
            walls_2d_JSON = list()

    if not polygons_JSON:
        if revision_number:
            query = f"SELECT model->'polygons' FROM {CREDENTIALS["CloudSQL"]["table_name_model_revisions_2d"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s AND revision_number = %s;"
            query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number, revision_number,), fetch=True))
        else:
            query = f"SELECT model_2d->'polygons' FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
            query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
        polygons_JSON = query_output[0].get("polygons")

        if polygons_JSON is None:
            polygons_JSON = list()

    hyperparameters = load_hyperparameters()
    plan = FloorPlan(hyperparameters)
    walls_2d_JSON, polygons_JSON = plan.scale_walls_2d_and_polygons(walls_2d_JSON, polygons_JSON, scale)
    drywall_takeoff = dict(
        total=dict(roof=0, wall=0),
        per_drywall=dict(
            roof=defaultdict(lambda: defaultdict(lambda: 0)),
            wall=defaultdict(lambda: defaultdict(lambda: 0))
        )
    )
    drywall_weights, waste_standard, _ = load_drywall_weights(
        walls_2d_JSON,
        polygons_JSON,
        compute_waste_average_standard=True,
        drywall_templates=DRYWALL_TEMPLATES
    )
    if waste_factor_average is None and load_preview is None:
        query = f"SELECT waste_average FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
        query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))

        waste_factor_average = query_output[0]["waste_average"]
    if waste_factor_average is not None:
        waste_factor_average_delta = float(waste_factor_average) - waste_standard
    else:
        waste_factor_average_delta = 0
    normalization_variance_aware = sum(w**2 for w in drywall_weights.values())
    if drywall_negate_opening_area_threshold is None and load_preview is None:
        query = f"SELECT drywall_negate_opening_area_threshold FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
        query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id, index, page_section_number,), fetch=True))
        drywall_negate_opening_area_threshold = query_output[0]["drywall_negate_opening_area_threshold"]
    for wall in walls_2d_JSON:
        drywall_negate_area = 0
        if wall["openings"]:
            for opening in wall["openings"]:
                drywall_negate_area += opening["count"] * opening["length"] * opening["height"]
            if drywall_negate_opening_area_threshold is not None and drywall_negate_area < drywall_negate_opening_area_threshold:
                drywall_negate_area = 0
            if drywall_negate_opening_area_threshold is None:
                drywall_negate_area = 0
        for drywall in wall["polygons_drywall"]:
            surface_area_original = (drywall["height"] * wall["length"])
            surface_area = surface_area_original - drywall_negate_area
            if drywall["type"].upper() == "DISABLED":
                continue
            if drywall["type_stacked"]:
                stack_length = len(drywall["type_stacked"])
                drywall_negate_area_stacked = drywall_negate_area / stack_length
                for drywall_type, drywall_height in zip(drywall["type_stacked"], drywall["height_stacked"]):
                    drywall_template = query_drywall(drywall_type, DRYWALL_TEMPLATES)
                    if not drywall_template:
                        continue
                    waste_factor_delta = waste_factor_average_delta * (drywall_weights[drywall_type] / normalization_variance_aware)
                    waste_factor = max(0, float(drywall_template["waste"]) + waste_factor_delta) / 100
                    net_sqft = wall["length"] * drywall_height
                    drywall_area = max(1, drywall["layers"]) * (net_sqft - drywall_negate_area_stacked)
                    total_sqft = drywall_area * (1 + waste_factor)
                    drywall_takeoff["total"]["wall"] += total_sqft
                    sheet_size = drywall_template["sheet_size"]
                    sheet_area_sqft = float(sheet_size.split('x')[0]) * float(sheet_size.split('x')[1])
                    sheets_required_total = math.ceil(total_sqft / sheet_area_sqft)
                    sheets_required_no_waste = math.ceil(net_sqft / sheet_area_sqft)
                    drywall_takeoff["per_drywall"]["wall"][drywall_type] = dict(
                        total_sqft=round(drywall_takeoff["per_drywall"]["wall"][drywall_type]["total_sqft"]+total_sqft, 2),
                        net_sqft=round(drywall_takeoff["per_drywall"]["wall"][drywall_type]["net_sqft"]+net_sqft, 2),
                        waste_percentage=round(waste_factor*100, 2),
                        drywall_negation_area_in_SQFT=round(drywall_takeoff["per_drywall"]["wall"][drywall_type]["drywall_negation_area_in_SQFT"]+drywall_negate_area, 2),
                        sheet_size=sheet_size,
                        sheets_required_total=drywall_takeoff["per_drywall"]["wall"][drywall_type]["sheets_required_total"]+sheets_required_total,
                        sheets_required_no_waste=drywall_takeoff["per_drywall"]["wall"][drywall_type]["sheets_required_no_waste"]+sheets_required_no_waste
                    )
            else:
                drywall_template = query_drywall(drywall["type"], DRYWALL_TEMPLATES)
                if not drywall_template:
                    continue
                waste_factor_delta = waste_factor_average_delta * (drywall_weights[drywall["type"]] / normalization_variance_aware)
                waste_factor = max(0, float(drywall_template["waste"]) + waste_factor_delta) / 100
                net_sqft = surface_area_original
                drywall_area = max(1, drywall["layers"]) * surface_area
                total_sqft = drywall_area * (1 + waste_factor)
                drywall_takeoff["total"]["wall"] += total_sqft
                sheet_size = drywall_template["sheet_size"]
                sheet_area_sqft = float(sheet_size.split('x')[0]) * float(sheet_size.split('x')[1])
                sheets_required_total = math.ceil(total_sqft / sheet_area_sqft)
                sheets_required_no_waste = math.ceil(net_sqft / sheet_area_sqft)
                drywall_takeoff["per_drywall"]["wall"][drywall["type"]] = dict(
                    total_sqft=round(drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["total_sqft"]+total_sqft, 2),
                    net_sqft=round(drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["net_sqft"]+net_sqft, 2),
                    waste_percentage=round(waste_factor*100, 2),
                    drywall_negation_area_in_SQFT=round(drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["drywall_negation_area_in_SQFT"]+drywall_negate_area, 2),
                    sheet_size=sheet_size,
                    sheets_required_total=drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["sheets_required_total"]+sheets_required_total,
                    sheets_required_no_waste=drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["sheets_required_no_waste"]+sheets_required_no_waste
                )
    for polygon in polygons_JSON:
        if polygon["polygon_drywall"]["type"].upper() == "DISABLED":
            polygon["polygon_drywall"]["enabled"] = False
            continue
        if polygon["polygon_drywall"]["type_stacked"]:
            if not isinstance(polygon["polygon_drywall"]["layers"], list):
                polygon["polygon_drywall"]["layers"] = [polygon["polygon_drywall"]["layers"]]
            for drywall_type, layers in zip(polygon["polygon_drywall"]["type_stacked"], polygon["polygon_drywall"]["layers"]):
                drywall_template = query_drywall(drywall_type, DRYWALL_TEMPLATES)
                if not drywall_template:
                    continue
                surface_area_flat, surface_area_sloped = polygon["area"], polygon["area"]
                if polygon["type"] in SLOPED_CEILING_CHOICES:
                    surface_area_sloped = plan.compute_sloped_area_polygon(
                        polygon["area"],
                        polygon["pitch"],
                    )
                waste_factor_delta = waste_factor_average_delta * (drywall_weights[drywall_type] / normalization_variance_aware)
                waste_factor = max(0, float(drywall_template["waste"]) + waste_factor_delta) / 100
                net_sqft = surface_area_flat
                drywall_area = max(1, layers) * surface_area_sloped
                total_sqft = drywall_area * (1 + waste_factor)
                sheet_size = drywall_template["sheet_size"]
                sheet_area_sqft = float(sheet_size.split('x')[0]) * float(sheet_size.split('x')[1])
                sheets_required_total = math.ceil(total_sqft / sheet_area_sqft)
                sheets_required_no_waste = math.ceil(net_sqft / sheet_area_sqft)
                drywall_takeoff["per_drywall"]["roof"][drywall_type] = dict(
                    total_sqft=round(drywall_takeoff["per_drywall"]["roof"][drywall_type]["total_sqft"]+total_sqft, 2),
                    net_sqft=round(drywall_takeoff["per_drywall"]["roof"][drywall_type]["net_sqft"]+net_sqft, 2),
                    waste_percentage=round(waste_factor*100, 2),
                    sheet_size=sheet_size,
                    sheets_required_total=drywall_takeoff["per_drywall"]["roof"][drywall_type]["sheets_required_total"]+sheets_required_total,
                    sheets_required_no_waste=drywall_takeoff["per_drywall"]["roof"][drywall_type]["sheets_required_no_waste"]+sheets_required_no_waste
                )
                drywall_takeoff["total"]["roof"] += total_sqft
        else:
            drywall_template = query_drywall(polygon["polygon_drywall"]["type"], DRYWALL_TEMPLATES)
            if not drywall_template:
                continue
            surface_area_flat, surface_area_sloped = polygon["area"], polygon["area"]
            if polygon["type"] in SLOPED_CEILING_CHOICES:
                surface_area_sloped = plan.compute_sloped_area_polygon(
                    polygon["area"],
                    polygon["pitch"],
                )
            waste_factor_delta = waste_factor_average_delta * (drywall_weights[polygon["polygon_drywall"]["type"]] / normalization_variance_aware)
            waste_factor = max(0, float(drywall_template["waste"]) + waste_factor_delta) / 100
            net_sqft = surface_area_flat
            drywall_area = max(1, polygon["polygon_drywall"]["layers"]) * surface_area_sloped
            total_sqft = drywall_area * (1 + waste_factor)
            sheet_size = drywall_template["sheet_size"]
            sheet_area_sqft = float(sheet_size.split('x')[0]) * float(sheet_size.split('x')[1])
            sheets_required_total = math.ceil(total_sqft / sheet_area_sqft)
            sheets_required_no_waste = math.ceil(net_sqft / sheet_area_sqft)
            drywall_takeoff["per_drywall"]["roof"][polygon["polygon_drywall"]["type"]] = dict(
                total_sqft=round(drywall_takeoff["per_drywall"]["roof"][polygon["polygon_drywall"]["type"]]["total_sqft"]+total_sqft, 2),
                net_sqft=round(drywall_takeoff["per_drywall"]["roof"][polygon["polygon_drywall"]["type"]]["net_sqft"]+net_sqft, 2),
                waste_percentage=round(waste_factor*100, 2),
                sheet_size=sheet_size,
                sheets_required_total=drywall_takeoff["per_drywall"]["roof"][polygon["polygon_drywall"]["type"]]["sheets_required_total"]+sheets_required_total,
                sheets_required_no_waste=drywall_takeoff["per_drywall"]["roof"][polygon["polygon_drywall"]["type"]]["sheets_required_no_waste"]+sheets_required_no_waste
            )
            drywall_takeoff["total"]["roof"] += total_sqft

    drywall_takeoff["total"]["wall"] = round(drywall_takeoff["total"]["wall"], 2)
    drywall_takeoff["total"]["roof"] = round(drywall_takeoff["total"]["roof"], 2)

    if load_preview == False or load_preview is None:
        waste_factor_average = waste_factor_average if waste_factor_average is not None else waste_standard
        await insert_takeoff(
            drywall_takeoff,
            waste_factor_average,
            drywall_negate_opening_area_threshold,
            index,
            page_section_number,
            plan_id,
            user_id,
            project_id,
            revision_number,
            pg_pool,
            CREDENTIALS
        )
        logging.info("SYSTEM: Drywall Takeoff computation saved")
    logging.info("SYSTEM: Drywall Takeoff Computed Successfully for the provided Floorplan")
    return respond_with_UI_payload(drywall_takeoff)


@app.post("/summarize_takeoff_all")
async def summarize_takeoff_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    logging.info("SYSTEM: Received Total Drywall Takeoff summarization Request")
    is_user_not_authenticated = await is_authenticated(CREDENTIALS, pg_pool, request, user_id=user_id)
    if is_user_not_authenticated:
        logging.warning(f"SYSTEM: User: {user_id} is not authorized to access Drywall application")
        return respond_with_UI_payload(is_user_not_authenticated)

    query = f"SELECT page_number, page_section_number, scale, waste_average, drywall_negate_opening_area_threshold, takeoff, model_2d FROM {CREDENTIALS["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
    rows = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id), fetch=True))
    drywall_takeoff_all = list()
    for row in rows:
        drywall_takeoff = dict(row)
        if not drywall_takeoff["model_2d"]:
            continue

        walls_2d = json.loads(drywall_takeoff["model_2d"]) if isinstance(drywall_takeoff["model_2d"], str) else drywall_takeoff["model_2d"]
        if walls_2d["walls_2d"] and walls_2d["polygons"]:
            drywall_takeoff.pop("model_2d")
            drywall_takeoff_all.append(drywall_takeoff)

    return respond_with_UI_payload(jsonable_encoder({"drywall_takeoff_all": drywall_takeoff_all}))


@app.get("/insert_templates")
async def insert_templates():
    def parse_fire_rating(description: str):
        if "TYPE C" in description:
            return "Type C"
        if "TYPE X" in description:
            return "Type X"
        return None

    def parse_lightweight(description: str):
        return "LITE" in description.upper()

    def parse_wide_stretch(description: str):
        return "WIDE-STRETCH" in description.upper()

    def parse_thickness(description: str):
        match = re.search(r'(\d+\/\d+)"', description)
        if match:
            fraction = match.group(1)
            numerator, denominator = fraction.split("/")
            return float(numerator) / float(denominator)
        return None

    def generate_random_colors(n, seed=0):
        rng = np.random.default_rng(seed)
        total_colors = 256**3
        excluded_index = 255 * 256 * 256 + 0 * 256 + 0

        indices = rng.choice(
            total_colors - 1,
            size=n,
            replace=False
        )

        indices = np.where(indices >= excluded_index, indices + 1, indices)
        colors = list()
        for index in indices:
            r = index // (256 * 256)
            g = (index // 256) % 256
            b = index % 256
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hue_degrees = h * 360
            if (hue_degrees < 20 or hue_degrees > 340):
                r = max(0, r - 100)
                g = min(255, g + 50)
                b = min(255, b + 100)
            if v < 40:
                r = min(255, r + 100)
                g = min(255, g + 100)
                b = min(255, b + 100)
            colors.append((int(r), int(g), int(b)))

        return [dict(r=int(color[0]), g=int(color[1]), b=int(color[2])) for color in colors]

    dataframe = pd.read_excel("Drywall_P_Code_20260122.xlsx")
    rows_to_insert = list()
    product_color_codes = generate_random_colors(dataframe.size)

    SKU_IDs = list()
    for (_, row), product_color_code in zip(dataframe.iterrows(), product_color_codes):
        if pd.isna(row["user10"]) or pd.isna(row["user11"]) or pd.isna(row["PRODUCT_CAT_CODE"]) or pd.isna(row["PRODUCT_CAT_DESC"]) or not isinstance(row["PRODUCT_CAT_CODE"], int):
            continue
        if row["user10"] in SKU_IDs:
            continue
        SKU_IDs.append(row["user10"])
        sku_description = str(row["user11"]).upper()

        parsed_row = (
            row["user10"],
            row["user11"],
            int(row["PRODUCT_CAT_CODE"]),
            row["PRODUCT_CAT_DESC"],
            parse_thickness(sku_description),
            parse_fire_rating(sku_description),
            parse_lightweight(sku_description),
            parse_wide_stretch(sku_description),
            product_color_code,
            row["waste (%)"],
            row["sheet Size (in ft. x ft.)"],
        )

        rows_to_insert.append(parsed_row)

    query = f"""
        INSERT INTO {CREDENTIALS["CloudSQL"]["table_name_sku"]} (
            "sku_id",
            "sku_description",
            "product_cat_code",
            "product_cat_description",
            "thickness_inches",
            "fire_rating",
            "is_lightweight",
            "is_wide_stretch",
            "color_code",
            "waste",
            "sheet_size"
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
    """
    await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=rows_to_insert, execute_many=True))
    logging.info("SYSTEM: Templates successfully inserted")


@app.post("/request_otp")
async def request_otp(request: PayloadRequestExternalOtp):
    enable_logging_on_stdout()
    user_email = normalize_email(request.user_id)
    logging.info(f"SYSTEM: Received External OTP Request for {user_email}")

    if not is_valid_email(user_email):
        return respond_with_UI_payload(
            dict(
                user_type="EXTERNAL",
                email=user_email,
                token="INVALID"
            )
        )

    query = f"""
        SELECT
            is_external, is_locked
        FROM {CREDENTIALS["CloudSQL"]["table_name_users"]}
        WHERE LOWER(user_email) = LOWER(%s)
        LIMIT 1;
    """
    user_access_control = await run_in_threadpool(
        partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email,), fetch=True)
    )

    if not user_access_control:
        return respond_with_UI_payload(
            dict(
                user_type="EXTERNAL",
                email=user_email,
                token="INVALID"
            )
        )
    if not user_access_control[0]["is_external"]:
        return respond_with_UI_payload(
            dict(
                user_type="INTERNAL",
                email=user_email,
                token="INVALID"
            )
        )
    if user_access_control[0]["is_locked"]:
        return respond_with_UI_payload(
            dict(
                user_type="EXTERNAL",
                email=user_email,
                token="ACCOUNT LOCKED"
            )
        )

    otp = generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRATION_MINUTES)

    query = f"""
        INSERT INTO {CREDENTIALS["CloudSQL"]["table_name_otp"]} (
            email,
            otp_code,
            created_at,
            expires_at,
            is_verified,
            attempts
        )
        VALUES (
            %s,
            %s,
            CURRENT_TIMESTAMP,
            %s,
            FALSE,
            0
        )
        ON CONFLICT (email) DO UPDATE SET
            otp_code = EXCLUDED.otp_code,
            created_at = CURRENT_TIMESTAMP,
            expires_at = EXCLUDED.expires_at,
            is_verified = FALSE,
            attempts = 0;
    """

    await run_in_threadpool(
        partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email, otp, expires_at,))
    )
    
    sender = CREDENTIALS["Email"]["sender_email"]
    recipient = user_email
    
    try:
        logging.info(f"SYSTEM: Sending OTP email from {sender} to {recipient}")
        trigger_otp_email(CREDENTIALS, sender, recipient, otp)
        logging.info(f"SYSTEM: OTP email sent successfully using sender {sender}")
    
    except Exception as e:
        logging.exception(f"SYSTEM: OTP email failed using sender {sender}")
        return respond_with_UI_payload(
            dict(
                success=False,
                message="Failed to send OTP email",
                details=[f"SEND_FAILED_FROM_{sender}: {repr(e)}"],
                error_code="EMAIL_SEND_FAILED"
            ),
            status_code=500
        )
    
    return respond_with_UI_payload(
        dict(
            success=True,
            message="OTP sent to email",
            email=user_email,
            expires_in_minutes=OTP_EXPIRATION_MINUTES,
        )
    )


@app.post("/verify_otp")
async def verify_otp(request: PayloadVerifyExternalOtp):
    enable_logging_on_stdout()
    user_email = normalize_email(request.user_id)
    otp = str(request.otp or "").strip()
    logging.info(f"SYSTEM: Received External OTP Verification Request for {user_email}")

    if not is_valid_email(user_email):
        return respond_with_UI_payload(
            dict(
                user_type="EXTERNAL",
                email=user_email,
                token="INVALID")
            )

    query = f"""
        SELECT
            is_external
        FROM {CREDENTIALS["CloudSQL"]["table_name_users"]}
        WHERE LOWER(user_email) = LOWER(%s)
        LIMIT 1;
    """
    is_external = await run_in_threadpool(
        partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email,), fetch=True)
    )

    if not is_external:
        return respond_with_UI_payload(
            dict(
                user_type="EXTERNAL",
                email=user_email,
                token="INVALID"
            )
        )

    if len(otp) != 6 or not otp.isdigit():
        return respond_with_UI_payload(
            dict(
                success=False,
                message="Invalid OTP format",
                error_code="INVALID_OTP_FORMAT"
            ),
            status_code=400
        )

    query = f"""
        SELECT
            otp_code,
            expires_at,
            is_verified,
            attempts
        FROM {CREDENTIALS["CloudSQL"]["table_name_otp"]}
        WHERE LOWER(email) = LOWER(%s)
        LIMIT 1;
    """

    otp_rows = await run_in_threadpool(
        partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email,), fetch=True)
    )

    if not otp_rows:
        return respond_with_UI_payload(
            dict(
                success=False,
                message="Invalid OTP or email",
                error_code="OTP_NOT_FOUND"
            ),
            status_code=400
        )

    otp_record = otp_rows[0]

    if otp_record["is_verified"]:
        return respond_with_UI_payload(
            dict(
                success=False,
                message="OTP already used",
                error_code="OTP_ALREADY_USED"
            ),
            status_code=400
        )

    if datetime.now(timezone.utc) > otp_record["expires_at"]:
        query = f"""
            DELETE FROM {CREDENTIALS["CloudSQL"]["table_name_otp"]}
            WHERE LOWER(email) = LOWER(%s);
        """

        await run_in_threadpool(
            partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email,))
        )

        return respond_with_UI_payload(
            dict(
                success=False,
                message="OTP has expired. Please request a new one",
                error_code="OTP_EXPIRED"
            ),
            status_code=400
        )

    if int(otp_record["attempts"]) >= MAX_OTP_ATTEMPTS:
        query = f"""
            DELETE FROM {CREDENTIALS["CloudSQL"]["table_name_otp"]}
            WHERE LOWER(email) = LOWER(%s);
        """

        await run_in_threadpool(
            partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email,))
        )
        await lock_user(CREDENTIALS, pg_pool, user_email)

        return respond_with_UI_payload(
            dict(
                success=False,
                message="Max attempts exceeded. User Account Locked",
                error_code="MAX_ATTEMPTS_EXCEEDED"
            ),
            status_code=429
        )

    if otp_record["otp_code"] != otp:
        query = f"""
            UPDATE {CREDENTIALS["CloudSQL"]["table_name_otp"]}
            SET attempts = attempts + 1
            WHERE LOWER(email) = LOWER(%s);
        """

        await run_in_threadpool(
            partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email,))
        )

        remaining_attempts = MAX_OTP_ATTEMPTS - int(otp_record["attempts"]) - 1

        return respond_with_UI_payload(
            dict(
                success=False,
                message=f"Invalid OTP. {remaining_attempts} attempts remaining",
                error_code="INVALID_OTP"
            ),
            status_code=400
        )

    query = f"""
        SELECT
            user_email
        FROM {CREDENTIALS["CloudSQL"]["table_name_users"]}
        WHERE LOWER(user_email) = LOWER(%s)
        LIMIT 1;
    """

    users = await run_in_threadpool(
        partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email,), fetch=True)
    )

    if not users:
        return respond_with_UI_payload(
            dict(
                success=False,
                message="User not found",
                error_code="USER_NOT_FOUND"
            ),
            status_code=404
        )

    user_id = users[0]["user_email"]

    query = f"""
        UPDATE {CREDENTIALS["CloudSQL"]["table_name_otp"]}
        SET is_verified = TRUE
        WHERE LOWER(email) = LOWER(%s);
    """

    await run_in_threadpool(
        partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_email,))
    )

    token = create_external_login_jwt(CREDENTIALS, user_id)
    jwt_config = load_secret_json(CREDENTIALS["JWT"]["secret_path"])

    return respond_with_UI_payload(
        dict(
            success=True,
            message="OTP verified successfully",
            token=token,
            tokenType="Bearer",
            expiresIn=jwt_config.get("expiration_hours", 24) * 60 * 60,
            user=dict(
                user_id=user_id,
                email=user_id
            )
        )
    )


@app.post("/authenticate_internal_user")
async def authenticate_internal_user(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    user_id = parameters.get("user_id") or body.get("user_id")
    is_authenticated = parameters.get("is_authenticated") or body.get("is_authenticated")

    query = f"""
        SELECT
            is_external, is_locked
        FROM {CREDENTIALS["CloudSQL"]["table_name_users"]}
        WHERE LOWER(user_email) = LOWER(%s)
        LIMIT 1;
    """
    user_access_control = await run_in_threadpool(
        partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_id,), fetch=True)
    )

    if not user_access_control:
        return dict(user_type="INTERNAL", email=user_id, token="UNREGISTERED")

    user_type = "EXTERNAL" if user_access_control[0]["is_external"] else "INTERNAL"
    if user_access_control[0]["is_locked"]:
        return respond_with_UI_payload(
            dict(user_type=user_type, email=user_id, token="ACCOUNT LOCKED")
        )
    if user_access_control[0]["is_external"]:
        return respond_with_UI_payload(
            dict(user_type=user_type, email=user_id, token="EXTERNAL USER")
        )
    if is_authenticated:
        query = f"""
            DELETE
            FROM {CREDENTIALS["CloudSQL"]["table_name_internal_users_authentication_throttled"]}
            WHERE LOWER(user_email) = LOWER(%s);
        """
        await run_in_threadpool(
            partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_id,))
        )
        return respond_with_UI_payload(
            dict(user_type=user_type, email=user_id, token="SUCCESS")
        )
    if not is_authenticated:
        query = f"""
            SELECT
                attempts
            FROM {CREDENTIALS["CloudSQL"]["table_name_internal_users_authentication_throttled"]}
            WHERE LOWER(user_email) = LOWER(%s)
            LIMIT 1;
        """
        user_access_control = await run_in_threadpool(
            partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_id,), fetch=True)
        )
        failed_attempts = 0
        if user_access_control and user_access_control[0]["attempts"]:
            failed_attempts = user_access_control[0]["attempts"]
            if failed_attempts >= 5:
                await lock_user(CREDENTIALS, pg_pool, user_id)
                return respond_with_UI_payload(
                    dict(user_type=user_type, email=user_id, token="ACCOUNT LOCKED")
                )
        query = f"""
            INSERT INTO {CREDENTIALS["CloudSQL"]["table_name_internal_users_authentication_throttled"]} AS t (
            user_email,
            attempted_at,
            attempts
            )
            VALUES (
                %s,
                CURRENT_TIMESTAMP,
                %s
            )
            ON CONFLICT (user_email) DO UPDATE SET
                attempts = t.attempts + 1,
                attempted_at = CURRENT_TIMESTAMP
        """
        await run_in_threadpool(
            partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_id, 1,))
        )
        return respond_with_UI_payload(
            dict(user_type=user_type, email=user_id, token=f"AUTHENTICATION FAILED ({failed_attempts + 1}/5)")
        )


@app.post("/unlock_user_account")
async def unlock_user_account(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    user_id = parameters.get("user_id") or body.get("user_id")

    query = f"""
        DELETE
        FROM {CREDENTIALS["CloudSQL"]["table_name_internal_users_authentication_throttled"]}
        WHERE LOWER(user_email) = LOWER(%s);
    """
    await run_in_threadpool(
        partial(pg_run, CREDENTIALS, pg_pool, query, params=(user_id,))
    )
    await unlock_user(CREDENTIALS, pg_pool, user_id)
    logging.info(f"SYSTEM: User Account: {user_id} is Unlocked")


@app.post("/email_support_center")
async def email_support_center(
    request: Request,
    user_id: str = Form(None),
    subject: str = Form(None),
    content: str = Form(None),
    attachment: UploadFile | None = File(None),
):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    user_id = parameters.get("user_id") or body.get("user_id") or user_id
    subject = parameters.get("subject") or body.get("subject") or subject
    content = parameters.get("content") or body.get("content") or content

    attachment_path = None
    if attachment is not None:
        suffix = Path(attachment.filename).suffix
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:
            shutil.copyfileobj(attachment.file, tmp)
            attachment_path = tmp.name

    trigger_support(
        CREDENTIALS,
        user_id,
        CREDENTIALS["Email"]["support_center_id"],
        subject,
        content,
        attachment_path=attachment_path,
    )
    logging.info("SYSTEM: Support email forwarded")


from typing import Optional
from helper import phoenix_call
from vertexai.generative_models import Part, Content
from fastapi import File, Form, UploadFile, File
@app.post("/chat_gemini")
async def chat_gemini(prompt: str=Form(...),  image_png_bytes: Optional[UploadFile]=File(None)):
    enable_logging_on_stdout()

    logging.info("SYSTEM: Received a Chat Request")
    query = Content(role="user", parts=[Part.from_text(prompt)])
    if image_png_bytes is not None:
        bytes_canvas = await image_png_bytes.read()
        query = Content(role="user", parts=[Part.from_text(prompt), Part.from_data(data=bytes_canvas, mime_type="image/png")])
    vertex_ai_client, vertex_ai_generation_config, is_cached = load_vertex_ai_client(
        CREDENTIALS,
        None,
        prompts=[prompt]
    )
    if is_cached:
        response = phoenix_call(
            lambda feedback_prompt, temperature: vertex_ai_client.generate_content(
                contents=[feedback_prompt, query] if feedback_prompt else [query],
                generation_config={**vertex_ai_generation_config, "temperature": temperature},
            ),
            max_retry=CREDENTIALS["VertexAI"]["llm"]["max_retry"],
        )
    else:
        response = phoenix_call(
            lambda feedback_prompt, temperature: vertex_ai_client(prompt).generate_content(
                contents=[feedback_prompt, query] if feedback_prompt else [query],
                generation_config={**vertex_ai_generation_config, "temperature": temperature},
            ),
            max_retry=CREDENTIALS["VertexAI"]["llm"]["max_retry"],
        )
    return respond_with_UI_payload(dict(response=response))


from prompts import RESIDENTIAL_MULTI_FAMILY_SCHEMA
from pypdf import PdfReader, PdfWriter
from io import BytesIO
@app.post("/floorplan_to_2d_multi_family")
async def floorplan_to_2d_multi_family(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")

    vertex_ai_client, vertex_ai_generation_config, is_cached = load_vertex_ai_client(
        CREDENTIALS,
        None,
        prompts=[RESIDENTIAL_MULTI_FAMILY_SCHEMA]
    )

    pdf_path = Path("/tmp/floor_plan.PDF")
    GCS_URL_floorplan = await download_floorplan(plan_id, project_id, user_id, CREDENTIALS, pg_pool, destination_path=pdf_path)
    with open(pdf_path, "rb") as f:
        bytes_pdf = f.read()
    reader = PdfReader(BytesIO(bytes_pdf))
    pages = list()

    for index, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)

        buffer = BytesIO()
        writer.write(buffer)

        pages.append({
            "page_number": index,
            "bytes": buffer.getvalue()
        })

    parts = list()
    for page in pages:
        parts.append(Part.from_text(f"PAGE: {page["page_number"]}"))
        parts.append(Part.from_data(page["bytes"], mime_type="application/pdf"))
    query = Content(role="user", parts=parts)

    if is_cached:
        response = phoenix_call(
            lambda feedback_prompt, temperature: vertex_ai_client.generate_content(
                contents=[feedback_prompt, query] if feedback_prompt else [query],
                generation_config={**vertex_ai_generation_config, "temperature": temperature},
            ),
            max_retry=CREDENTIALS["VertexAI"]["llm"]["max_retry"],
        )
    else:
        response = phoenix_call(
            lambda feedback_prompt, temperature: vertex_ai_client(RESIDENTIAL_MULTI_FAMILY_SCHEMA).generate_content(
                contents=[feedback_prompt, query] if feedback_prompt else [query],
                generation_config={**vertex_ai_generation_config, "temperature": temperature},
            ),
            max_retry=CREDENTIALS["VertexAI"]["llm"]["max_retry"],
        )
    print(response)
    return respond_with_UI_payload(dict(response=response))
