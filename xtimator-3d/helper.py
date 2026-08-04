import asyncio
import json
from json.decoder import JSONDecodeError
import logging
import hashlib
import uuid
import requests
from pathlib import Path
import datetime
import base64
from time import sleep
from pypdf import PdfReader, PdfWriter
from io import BytesIO
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import subprocess
from collections import defaultdict
from functools import partial
from fastapi.concurrency import run_in_threadpool
import firebase_admin
from firebase_admin import auth as auth_firebase, credentials as credentials_firebase

import math
import re
import random
random.seed(0)
import cv2
from PIL import Image

import geoip2.database as geoip2_database
from google.cloud.storage import Client as CloudStorageClient
import google_crc32c
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from google.oauth2 import service_account
from google.cloud.pubsub_v1 import SubscriberClient
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
)
from google.auth.transport.requests import Request
from google.cloud.sql.connector import Connector, IPTypes
from google.auth.exceptions import TransportError
from sqlalchemy import create_engine
from sqlalchemy.exc import (
    OperationalError,
    InterfaceError,
    TimeoutError,
    DBAPIError
)
from pg8000.dbapi import (
    InterfaceError as InterfaceErrorPG8000,
    DatabaseError as DatabaseErrorPG8000,
)
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Content, Part
from vertexai.caching import CachedContent

from prompts import (
    FLOORPLAN_TO_MULTIPAGE_ELEVATION_MAPPER,
    FloorplanToMultipageElevationMapperResponse,
    ARCHITECTURAL_DRAWING_CLASSIFIER,
    ArchitecturalDrawingClassifierResponse,
    VISUAL_GROUNDING_DETECTOR,
    VisualGroundingDetectorResponse,
    FEEDBACK_GENERATOR,
    UNIT_MATCH_RESOLVER,
    UnitMatchResponse
)
from preprocessing import preprocess
from email_notification import trigger
from vector_pdf import is_vector, extract_scale


_pg_engine = None
_connector = None
_db_credentials = None

def load_pg_pool(credentials):
    global _pg_engine, _connector, _db_credentials

    if _pg_engine is not None:
        return _pg_engine

    if _connector is None:
        _connector = Connector(refresh_strategy="LAZY")

    pg = credentials["CloudSQL"]

    instance_connection_name = pg["connection_name"]
    db_name = pg["database_name"]
    sa_key = pg["service_account_key"]

    with open(sa_key, "r") as f:
        sa_payload = json.load(f)

    user = sa_payload["client_email"]

    _db_credentials = service_account.Credentials.from_service_account_file(
        sa_key,
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform"
        ]
    )

    request = Request()

    def get_conn():

        nonlocal request

        if (
            _db_credentials.expired
            or _db_credentials.token is None
        ):
            _db_credentials.refresh(request)

        conn = _connector.connect(
            instance_connection_name,
            pg["driver"],
            user=user,
            password=_db_credentials.token,
            db=db_name,
            enable_iam_auth=True,
            ip_type=IPTypes.PRIVATE,
        )

        conn.autocommit = True
        return conn

    _pg_engine = create_engine(
        "postgresql+pg8000://",
        creator=get_conn,
        pool_size=pg.get("min_pool_size", 3),
        max_overflow=max(
            pg.get("max_pool_size", 10)
            - pg.get("min_pool_size", 3),
            0
        ),
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        pool_use_lifo=True,
        future=True,
    )

    return _pg_engine

def close_pg_pool():
    global _pg_engine, _connector, _db_credentials

    if _pg_engine is not None:
        try:
            _pg_engine.dispose()
        except Exception as e:
            logging.warning(f"SYSTEM: {e}")
        _pg_engine = None

    if _connector is not None:
        try:
            _connector.close()
        except Exception as e:
            logging.warning(f"SYSTEM: {e}")
        _connector = None

    _db_credentials = None

def pg_run(
    credentials,
    pg_pool,
    query,
    params=None,
    fetch=False,
    max_retries=5,
    initial_backoff=1.0,
    max_backoff=30.0,
    execute_many=False,
):
    if params is None:
        params = ()

    for attempt in range(max_retries):

        conn = None
        cursor = None

        try:

            conn = pg_pool["engine"].raw_connection()
            cursor = conn.cursor()

            if execute_many:
                cursor.executemany(query, params)
            else:
                cursor.execute(query, params)

            result = None

            if fetch:
                rows = cursor.fetchall()
                columns = [c[0] for c in cursor.description]
                result = [
                    dict(zip(columns, row))
                    for row in rows
                ]

            conn.commit()

            return result

        except (
            DBAPIError,
            DatabaseErrorPG8000,
            OperationalError,
            InterfaceError,
            InterfaceErrorPG8000,
            TimeoutError,
            TransportError,
        ) as e:

            if conn is not None:
                try:
                    conn.invalidate()
                except Exception:
                    pass

            message = str(e).lower()

            retryable = any(x in message for x in (
                "deadlock",
                "serialization",
                "lock not available",
                "too many connections",
                "connection reset",
                "server closed",
                "network error",
                "timeout",
                "broken pipe",
                "ssl syscall",
                "terminating connection",
            ))

            if retryable and attempt + 1 < max_retries:

                delay = min(
                    initial_backoff * (2 ** attempt)
                    + random.uniform(0, 1),
                    max_backoff,
                )

                logging.warning(
                    "SYSTEM: PostgreSQL retry "
                    f"{attempt+1}/{max_retries} "
                    f"after {delay:.2f}s "
                    f"({type(e).__name__}: {e})"
                )

                sleep(delay)
                continue

            if isinstance(e, TransportError) and attempt + 1 < max_retries:
                close_pg_pool()
                delay = min(
                    initial_backoff * (2 ** attempt)
                    + random.uniform(0, 1),
                    max_backoff,
                )
                sleep(delay)
                engine = load_pg_pool(credentials)
                pg_pool["engine"] = engine
                continue

            raise

        except Exception:

            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass

            raise

        finally:

            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass

            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    raise RuntimeError(
        f"SYSTEM: PostgreSQL query failed after {max_retries} retries."
    )

def sha256(path, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

async def upload_floorplan(
    plan_path,
    plan_id,
    project_id,
    user_id,
    credentials,
    pg_pool,
    index=None,
    directory=None, 
    max_retries=5,
):
    def crc32c_base64(filename):
        checksum = google_crc32c.Checksum()
        with open(filename, "rb") as f:
            while chunk := f.read(1024 * 1024):
                checksum.update(chunk)
        return base64.b64encode(checksum.digest()).decode("utf-8")

    organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
    client = CloudStorageClient()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if directory:
        if index:
            blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{index}/{directory}/{blob_object_name}"
        else:
            blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{directory}/{blob_object_name}"
    else:
        if index:
            blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{index}/{blob_object_name}"
        else:
            blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{blob_object_name}"

    local_size =  Path(plan_path).stat().st_size
    local_crc = crc32c_base64(plan_path)
    for attempt in range(max_retries):
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(plan_path, checksum="crc32c")
        blob.reload()

        remote_size = int(blob.size)
        remote_crc = blob.crc32c

        if (
            remote_size == local_size
            and remote_crc == local_crc
        ):
            return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"
        try:
            blob.delete()
        except Exception:
            pass

        await asyncio.sleep(min(2 ** attempt, 30))
    raise RuntimeError(
        f"Upload verification failed after {max_retries} attempts."
    )

async def insert_model_2d(
    model_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    GCS_URL_floorplan_page,
    GCS_URL_target_drywalls_page,
    pg_pool,
    credentials,
    page_section_number=None,
    page_sections=None,
    ):
    if not page_section_number:
        page_section_number = 'I'
    if not page_sections:
        query = f"SELECT page_sections FROM {credentials["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
        query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, int(page_number), page_section_number,), fetch=True))
        page_sections = query_output[0]["page_sections"]
    if not model_2d.get("metadata", None):
        query = f"SELECT model_2d->'metadata' AS metadata FROM {credentials["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s"
        query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, int(page_number), page_section_number,), fetch=True))
        metadata = query_output[0]["metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        model_2d["metadata"] = metadata
    query = f"""
        INSERT INTO {credentials["CloudSQL"]["table_name_models"]} AS t (
            plan_id,
            project_id,
            user_id,
            page_number,
            page_sections,
            page_section_number,
            scale,
            model_2d,
            model_3d,
            takeoff,
            source,
            target_drywalls,
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
            %s::jsonb,
            '{{}}'::jsonb,
            '{{}}'::jsonb,
            %s,
            %s,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (project_id, plan_id, page_number, page_section_number) DO UPDATE SET
            model_2d = EXCLUDED.model_2d,
            scale = COALESCE(NULLIF(EXCLUDED.scale, ''), t.scale),
            updated_at = CURRENT_TIMESTAMP
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(
        plan_id,
        project_id,
        user_id,
        int(page_number),
        int(page_sections),
        page_section_number,
        scale,
        json.dumps(model_2d),
        GCS_URL_floorplan_page,
        GCS_URL_target_drywalls_page
    )))

async def insert_layout_2d(
    layout_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    GCS_URL_floorplan_page,
    GCS_URL_target_drywalls_page,
    pg_pool,
    credentials,
    page_section_number=None,
    page_sections=None,
    ):
    if not page_section_number:
        page_section_number = 'I'
    if not page_sections:
        query = f"SELECT page_sections FROM {credentials["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s;"
        query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, int(page_number), page_section_number,), fetch=True))
        page_sections = query_output[0]["page_sections"]
    if not layout_2d.get("metadata", None):
        query = f"SELECT layout_2d->'metadata' AS metadata FROM {credentials["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s AND page_section_number = %s"
        query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, int(page_number), page_section_number,), fetch=True))
        metadata = query_output[0]["metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        layout_2d["metadata"] = metadata
    query = f"""
        INSERT INTO {credentials["CloudSQL"]["table_name_models"]} AS t (
            plan_id,
            project_id,
            user_id,
            page_number,
            page_sections,
            page_section_number,
            scale,
            layout_2d,
            model_3d,
            takeoff,
            source,
            target_drywalls,
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
            %s::jsonb,
            '{{}}'::jsonb,
            '{{}}'::jsonb,
            %s,
            %s,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (project_id, plan_id, page_number, page_section_number) DO UPDATE SET
            layout_2d = EXCLUDED.layout_2d,
            scale = COALESCE(NULLIF(EXCLUDED.scale, ''), t.scale),
            updated_at = CURRENT_TIMESTAMP
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(
        plan_id,
        project_id,
        user_id,
        int(page_number),
        int(page_sections),
        page_section_number,
        scale,
        json.dumps(layout_2d),
        GCS_URL_floorplan_page,
        GCS_URL_target_drywalls_page
    )))

async def is_duplicate(pg_pool, credentials, access_control, pdf_path, project_id, user_id):
    sha_256 = sha256(pdf_path)
    organization_slug_target = await load_organization_slug(credentials, pg_pool, user_id)
    is_admin = await access_control.is_admin(user_id)
    target_user_is_global_admin = is_admin.super
    query = f"SELECT plan_id, sha256, status, user_id FROM {credentials["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s)"
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id,), fetch=True))
    for plan_target in list(query_output):
        organization_slug_reference = await load_organization_slug(credentials, pg_pool, plan_target["user_id"])
        if organization_slug_reference != organization_slug_target and not target_user_is_global_admin:
            continue
        if plan_target["sha256"] == sha_256:
            if plan_target["status"] == "FAILED":
                await delete_plan(credentials, pg_pool, plan_target["plan_id"], project_id)
                return False
            return plan_target["plan_id"]
    return False

async def delete_plan(credentials, pg_pool, plan_id, project_id):
    query = f"DELETE FROM {credentials["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id,)))
    return query_output

def load_floorplan_to_structured_2d_ID_token(credentials):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["floorplan_to_structured_2d"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token
    return id_token

def load_floorplan_to_preview_ID_token(credentials):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["floorplan_to_preview"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token
    return id_token

def load_unit_count_resolver_ID_token(credentials, audience):
    """ID token for the unit-count-resolver service (D5a).

    Same idiom as the two loaders above; the only difference is that the audience
    is passed in rather than read from credentials["CloudRun"]["APIs"], because
    the resolver URL comes from the UNIT_COUNT_RESOLVER_URL env var (the service
    does not exist in gcp.yaml until it is first deployed).
    """
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=audience
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token
    return id_token

async def load_templates(pg_pool, credentials):
    drywall_skus_primary = ["D12L", "D12LW", "D12MM", "D58F", "D12GMTB", "DCB12"]
    drywall_sku_order = {sku: index for index, sku in enumerate(drywall_skus_primary)}
    query = f"SELECT * FROM {credentials["CloudSQL"]["table_name_sku"]}"
    product_templates = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, fetch=True))

    logging.info("SYSTEM: Product Templates retrieved successfully")
    product_templates_target = list()
    cached_templates_sku = list()
    for product_template in product_templates:
        product_template = dict(product_template)
        if product_template["sku_id"] in cached_templates_sku:
            continue
        cached_templates_sku.append(product_template["sku_id"])
        product_template["sku_variant"] = f"{product_template["sku_id"]} - {product_template["sku_description"]}"
        product_template["color_code"] = [product_template["color_code"]['b'], product_template["color_code"]['g'], product_template["color_code"]['r']]
        product_templates_target.append(product_template)

    product_templates_target_primary = list(filter(lambda template: template["sku_id"] in drywall_skus_primary, product_templates_target))
    product_templates_target_primary.sort(key=lambda template: drywall_sku_order.get(template["sku_id"], len(product_templates_target)))
    [product_templates_target.remove(template_primary) for template_primary in product_templates_target_primary]
    product_templates_target = product_templates_target_primary + product_templates_target
    return product_templates_target

def query_drywall(query_sku_variant, drywall_templates):
    for drywall_template in drywall_templates:
        if drywall_template["sku_variant"] == query_sku_variant:
            return drywall_template

def load_subscriber_client(credentials):
    credentials_SA = service_account.Credentials.from_service_account_file(credentials["PubSub"]["service_account_key"])
    subscriber = SubscriberClient(credentials=credentials_SA)
    return subscriber

def query_subscriber_messages(credentials, subscriber_client, queries):
    credentials_SA = service_account.Credentials.from_service_account_file(credentials["PubSub"]["service_account_key"])
    subscription_path = subscriber_client.subscription_path(credentials_SA.project_id, credentials["PubSub"]["subscription_name"])
    try:
        response = subscriber_client.pull(
            request=dict(subscription=subscription_path, max_messages=credentials["PubSub"]["max_messages"]),
            timeout=credentials["PubSub"]["timeout"]
        )
    except DeadlineExceeded:
        return False, list()

    acknowledged_queries = list()
    for received_message in response.received_messages:
        try:
            message = json.loads(received_message.message.data.decode("utf-8"))
            for query in queries:
                if query == message:
                    subscriber_client.acknowledge(
                        request=dict(subscription=subscription_path, ack_ids=[received_message.ack_id])
                    )
                    acknowledged_queries.append(query)
                    if len(acknowledged_queries) == len(queries):
                        return True, acknowledged_queries
        except JSONDecodeError:
            continue
    return False, acknowledged_queries

def load_vertex_ai_client(credentials, ip_address, prompts=None, default_region="us-central1", max_retry=5, base_delay=1.0):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    region = load_nearest_region(
        ip_address,
        credentials["geolite_database"],
        credentials["VertexAI"]["llm"]["available_regions"],
        default_region=default_region
    )
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = lambda system_instruction: GenerativeModel(
        credentials["VertexAI"]["llm"]["model_name"],
        system_instruction=system_instruction
    )
    is_cached = False
    if prompts and GenerativeModel(credentials["VertexAI"]["llm"]["model_name"]).count_tokens(prompts).total_tokens >= 1024:
        is_cached = True
        n_iterations = 0
        while n_iterations < max_retry:
            try:
                cached_content = CachedContent.create(
                    model_name=credentials["VertexAI"]["llm"]["model_name"],
                    contents=prompts,
                    ttl=datetime.timedelta(hours=5),
                    display_name="drywall_predictor_cache"
                )
                break
            except (ResourceExhausted, InternalServerError) as e:
                n_iterations += 1
                if n_iterations >= max_retry:
                    raise e
                sleep_time = base_delay * (2 ** (n_iterations - 1)) + random.uniform(0, 0.5)
                sleep(sleep_time)
                logging.warning(f"SYSTEM: Vertex AI Gemini: {e}: RETRYING ...")
        vertex_ai_client = GenerativeModel.from_cached_content(cached_content)
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config, is_cached

def load_nearest_region(ip_address, geolite_database, available_regions, default_region="us-central1"):
    def _compute_haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2):
        R = 6371
        d_latitude = math.radians(latitude_2-latitude_1)
        d_longitude = math.radians(longitude_2-longitude_1)
        a = math.sin(d_latitude/2)**2 + math.cos(math.radians(latitude_1)) * math.cos(math.radians(latitude_2)) * math.sin(d_longitude/2)**2
        return 2*R*math.asin(math.sqrt(a))

    if not ip_address or "," not in ip_address:
        return default_region
    if ip_address and "," in ip_address:
        ip_address = ip_address.split(",")[0].strip()
    geoip2_reader = geoip2_database.Reader(geolite_database)
    try:
        response = geoip2_reader.city(ip_address)
        (response.location.latitude, response.location.longitude, response.country.iso_code)
        nearest_region = None
        minimum_distance = float("inf")
        for region, (latitude, longitude) in available_regions.items():
            distance = _compute_haversine_distance(response.location.latitude, response.location.longitude, latitude, longitude)
            if distance < minimum_distance:
                minimum_distance = distance
                nearest_region = region
        return nearest_region
    except Exception:
        return default_region

def phoenix_call(generate_content_lambda, max_retry=5, base_delay=1.0, pydantic_model=None, verify_field_counts=None):
    n_iterations = 0
    temperature = 0
    exceptions = list()
    feedback_prompt = ''
    while n_iterations < max_retry:
        try:
            response = generate_content_lambda(feedback_prompt, temperature)
            if pydantic_model:
                json_response = json.loads(response.text.strip("`json").replace("{{", '{').replace("}}", '}'))
                if verify_field_counts:
                    for field, count in verify_field_counts.items():
                        if len(json_response[field]) != count:
                            raise ValueError(f"Predicted {field} count: {len(json_response[field])} does not match with the expected number: {count}")
                response_json_pydantic = pydantic_model(**json_response)
                return response_json_pydantic, json_response
            return response.text
        except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded) as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            sleep_time = base_delay * (2 ** (n_iterations - 1)) + random.uniform(0, 0.5)
            sleep(sleep_time)
            logging.warning(f"SYSTEM: Vertex AI Gemini: {e}: RETRYING ...")
        except Exception as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            exceptions.append(e)
            system_feedback = [Part.from_text(FEEDBACK_GENERATOR.format(max_retry=max_retry, exceptions=exceptions))]
            feedback_prompt = Content(role="model", parts=system_feedback)
            temperature = min(0.25 * (n_iterations + 1) / max_retry, 0.25)
            logging.warning(f"SYSTEM: Vertex AI Gemini: Response Generation/Parsing failed with ERROR: {e}: RETRYING ...")
            logging.warning(f"SYSTEM: RETRYING with TEMPERATURE: {temperature}")

async def map_floorplan_to_multipage_elevation(credentials, pg_pool, project_id, plan_id, client_ip_address, pdf_path):
    query = f"SELECT multipage_elevation_map FROM {credentials["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id,), fetch=True))
    elevation_map = json.loads(query_output[0]["multipage_elevation_map"]) if isinstance(query_output[0]["multipage_elevation_map"], str) else query_output[0]["multipage_elevation_map"]
    if elevation_map:
        return elevation_map

    vertex_ai_client, vertex_ai_generation_config, is_cached = load_vertex_ai_client(
        credentials,
        client_ip_address,
        prompts=[FLOORPLAN_TO_MULTIPAGE_ELEVATION_MAPPER]
    )
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

    try:
        if is_cached:
            _, elevation_map = phoenix_call(
                lambda feedback_prompt, temperature: vertex_ai_client.generate_content(
                    contents=[feedback_prompt, query] if feedback_prompt else [query],
                    generation_config={**vertex_ai_generation_config, "temperature": temperature},
                ),
                max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                pydantic_model=FloorplanToMultipageElevationMapperResponse,
            )
        else:
            _, elevation_map = phoenix_call(
                lambda feedback_prompt, temperature: vertex_ai_client(FLOORPLAN_TO_MULTIPAGE_ELEVATION_MAPPER).generate_content(
                    contents=[feedback_prompt, query] if feedback_prompt else [query],
                    generation_config={**vertex_ai_generation_config, "temperature": temperature},
                ),
                max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                pydantic_model=FloorplanToMultipageElevationMapperResponse,
            )
    except Exception as e:
        logging.warning(f"SYSTEM: Floor Plan to Multi-page Elevation mapping has failed: {e}")
        elevation_map = dict()

    return elevation_map

def load_elevation_map(elevation_map, page_number):
    page_numbers = list()
    for group in elevation_map["floorplan_groups"]:
        if group["floorplan_page"] == page_number:
            page_numbers = [elevation_page["page_number"] for elevation_page in group["elevation_pages"]]
            return page_numbers
    return page_numbers

def classify_plan(
    credentials,
    client_ip_address,
    plan_paths,
    page_batch,
    vertex_ai_client=None,
    vertex_ai_generation_config=None,
    is_cached=None
):
    if not vertex_ai_client:
        vertex_ai_client, vertex_ai_generation_config, is_cached = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[ARCHITECTURAL_DRAWING_CLASSIFIER]
        )
    query_parts = list()
    for page_number, plan_path in zip(page_batch, plan_paths):
        plan_BGR = cv2.imread(plan_path)
        _, canvas_buffer_array = cv2.imencode(".png", plan_BGR)
        bytes_canvas = canvas_buffer_array.tobytes()
        query_parts.append(Part.from_text(f"PAGE: {page_number}"))
        query_parts.append(Part.from_data(data=bytes_canvas, mime_type="image/png"))
    query = Content(role="user", parts=query_parts)
    try:
        if is_cached:
            _, plan_types = phoenix_call(
                lambda feedback_prompt, temperature: vertex_ai_client.generate_content(
                    contents=[feedback_prompt, query] if feedback_prompt else [query],
                    generation_config={**vertex_ai_generation_config, "temperature": temperature},
                ),
                max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                pydantic_model=ArchitecturalDrawingClassifierResponse,
                verify_field_counts=dict(pages=len(plan_paths))
            )
        else:
            _, plan_types = phoenix_call(
                lambda feedback_prompt, temperature: vertex_ai_client(ARCHITECTURAL_DRAWING_CLASSIFIER).generate_content(
                    contents=[feedback_prompt, query] if feedback_prompt else [query],
                    generation_config={**vertex_ai_generation_config, "temperature": temperature},
                ),
                max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                pydantic_model=ArchitecturalDrawingClassifierResponse,
                verify_field_counts=dict(pages=len(plan_paths))
            )
    except Exception as e:
        logging.warning(f"SYSTEM: Plan Classification has failed: {e}")
        pages = [dict(
            page_number=page_number,
            plan_type=["FLOOR_PLAN"],
            mask_factor=dict(horizontal=0.0, vertical=0.0),
            bounding_box_offsets=[dict(offset_top_left=[0.0, 0.0], offset_bottom_right=[1.0, 1.0], title='', plan_type="FLOOR_PLAN")]
        ) for page_number in range(len(plan_paths))]
        plan_types = dict(pages=pages)

    return plan_types

def detect_bounding_boxes(
    credentials,
    client_ip_address,
    plan_paths,
    page_batch,
    vertex_ai_client=None,
    vertex_ai_generation_config=None,
    is_cached=None
):
    if not vertex_ai_client:
        vertex_ai_client, vertex_ai_generation_config, is_cached = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[VISUAL_GROUNDING_DETECTOR]
        )
    query_parts = list()
    for page_number, plan_path in zip(page_batch, plan_paths):
        plan_BGR = cv2.imread(plan_path)
        _, canvas_buffer_array = cv2.imencode(".png", plan_BGR)
        bytes_canvas = canvas_buffer_array.tobytes()
        query_parts.append(Part.from_text(f"PAGE: {page_number}"))
        query_parts.append(Part.from_data(data=bytes_canvas, mime_type="image/png"))
    query = Content(role="user", parts=query_parts)
    try:
        if is_cached:
            _, bounding_boxes = phoenix_call(
                lambda feedback_prompt, temperature: vertex_ai_client.generate_content(
                    contents=[feedback_prompt, query] if feedback_prompt else [query],
                    generation_config={**vertex_ai_generation_config, "temperature": temperature},
                ),
                max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                pydantic_model=VisualGroundingDetectorResponse,
                verify_field_counts=dict(pages=len(plan_paths))
            )
        else:
            _, bounding_boxes = phoenix_call(
                lambda feedback_prompt, temperature: vertex_ai_client(VISUAL_GROUNDING_DETECTOR).generate_content(
                    contents=[feedback_prompt, query] if feedback_prompt else [query],
                    generation_config={**vertex_ai_generation_config, "temperature": temperature},
                ),
                max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                pydantic_model=VisualGroundingDetectorResponse,
                verify_field_counts=dict(pages=len(plan_paths))
            )
    except Exception as e:
        logging.warning(f"SYSTEM: Bounding Box detection has failed: {e}")
        pages = [dict(
            page_number=page_number,
            mask_factor=dict(horizontal=0.0, vertical=0.0),
            bounding_box_offsets=[dict(offset_top_left=[0.0, 0.0], offset_bottom_right=[1.0, 1.0], title='', plan_type="FLOOR_PLAN")]
        ) for page_number in page_batch]
        bounding_boxes = dict(pages=pages)

    return bounding_boxes

def plan_to_preview(
    credentials,
    project_id,
    plan_id,
    user_id,
    organization_slug,
):
    id_token = load_floorplan_to_preview_ID_token(credentials)
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"{credentials["CloudRun"]["APIs"]["floorplan_to_preview"]}/classify_pages",
        headers=headers,
        json=dict(
            project_id=project_id,
            plan_id=plan_id,
            user_id=user_id,
            organization_slug=organization_slug,
        ),
    )
    response.raise_for_status()
    plan_types = response.json()
    return plan_types

async def floorplan_to_pages(credentials, pg_pool, project_id, plan_id, user_id, pdf_path, n_pages, maximum_dpi, minimum_dpi, batch_size=10):
    organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
    plan_types = plan_to_preview(credentials, project_id, plan_id, user_id, organization_slug)
    pages_to_insert = list()
    for page in plan_types["pages"]:
        pages_to_insert.append({
            "plan_id": plan_id,
            "project_id": project_id,
            "user_id": user_id,
            "page_number": page["page_number"],
            "mask_factor": json.dumps(dict()),
            "bounding_box_offsets": json.dumps(dict()),
            "source": '',
            "thumbnail": '',
            "plan_type": page["plan_type"],
            "extracted": False,
            "status": "NOT STARTED",
            "is_floorplan": "FLOOR" in page["plan_type"].upper(),
        })
    await insert_pages_batch(
        pages_to_insert,
        pg_pool,
        credentials,
    )
    page_batches = [list(range(batch_index * batch_size, batch_index * batch_size + batch_size)) for batch_index in range(n_pages // batch_size)]
    if n_pages % batch_size:
        page_batches += [list(range(n_pages - (n_pages % batch_size), n_pages))]
    floor_plan_paths_preprocessed = list()
    for page_batch in page_batches:
        futures = list()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for page_number in page_batch:
                future = executor.submit(
                    preprocess,
                    pdf_path,
                    page_number,
                    maximum_dpi,
                    minimum_dpi,
                )
                futures.append(future)
        for future in futures:
            floor_plan_paths_preprocessed.append(future.result())
    for page_number, floor_plan_path_preprocessed in enumerate(floor_plan_paths_preprocessed):
        await upload_floorplan(floor_plan_path_preprocessed, plan_id, project_id, user_id, credentials, pg_pool, index=str(page_number).zfill(4))
    return floor_plan_paths_preprocessed, plan_types

def page_to_svg(
    floor_plan_path="/tmp/floor_plan.png",
    pdf_path="/tmp/scaled_floor_plan.pdf",
    svg_path="/tmp/scaled_floor_plan.svg",
):
    canvas = Image.open(floor_plan_path)
    if canvas.mode != "RGB":
        canvas = canvas.convert("RGB")

    canvas.save(pdf_path, save_all=True)

    subprocess.run(
        ["pdftocairo", "-svg", pdf_path, svg_path],
        check=True
    )
    tree = ET.parse(svg_path)
    root = tree.getroot()
    root.set("width", "100%")
    root.set("height", "100%")
    if not root.get("preserveAspectRatio"):
        root.set("preserveAspectRatio", "xMidYMid meet")
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)

    return Path(svg_path)

async def insert_page(
    plan_id,
    user_id,
    project_id,
    page_number,
    extracted,
    status,
    pg_pool,
    credentials,
    plan_type=dict(),
    GCS_URL_page=None,
    GCS_URL_page_thumbnail=None,
    mask_factor=dict(),
    bounding_box_offsets=dict(),
    is_floorplan=None,
):
    query = f"""
        INSERT INTO {credentials["CloudSQL"]["table_name_pages"]} (
            plan_id,
            project_id,
            user_id,
            page_number,
            mask_factor,
            bounding_box_offsets,
            source,
            thumbnail,
            plan_type,
            extracted,
            status,
            is_floorplan,
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
            %s,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (project_id, plan_id, page_number) DO UPDATE SET
            extracted = EXCLUDED.extracted,
            updated_at = CURRENT_TIMESTAMP,
            status = EXCLUDED.status
    """
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(
        plan_id,
        project_id,
        user_id,
        int(page_number),
        json.dumps(mask_factor),
        json.dumps(bounding_box_offsets),
        GCS_URL_page,
        GCS_URL_page_thumbnail,
        json.dumps(plan_type),
        extracted,
        status,
        is_floorplan
    )))

async def insert_pages_batch(
    pages,
    pg_pool,
    credentials,
):
    if not pages:
        return None

    query = f"""
        INSERT INTO {credentials["CloudSQL"]["table_name_pages"]} (
            plan_id,
            project_id,
            user_id,
            page_number,
            mask_factor,
            bounding_box_offsets,
            source,
            thumbnail,
            plan_type,
            extracted,
            status,
            is_floorplan,
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
            %s,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (project_id, plan_id, page_number) DO UPDATE SET
            extracted = EXCLUDED.extracted,
            updated_at = CURRENT_TIMESTAMP,
            status = EXCLUDED.status,
            plan_type = EXCLUDED.plan_type,
            source = EXCLUDED.source,
            thumbnail = EXCLUDED.thumbnail,
            mask_factor = EXCLUDED.mask_factor,
            bounding_box_offsets = EXCLUDED.bounding_box_offsets,
            is_floorplan = EXCLUDED.is_floorplan
    """

    page_rows = list()
    for page in pages:
        page_rows.append(
            (
                page["plan_id"],
                page["project_id"],
                page["user_id"],
                int(page["page_number"]),
                json.dumps(page.get("mask_factor", dict())),
                json.dumps(page.get("bounding_box_offsets", dict())),
                page.get("source", ''),
                page.get("thumbnail", ''),
                json.dumps(page.get("plan_type", list())),
                page["extracted"],
                page["status"],
                page["is_floorplan"],
            )
        )

    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=page_rows, execute_many=True))

def load_drywall_weights(walls_2d_JSON, polygons_JSON, compute_waste_average_standard=False, drywall_templates=None):
    weights_drywall = defaultdict(lambda: 0)
    drywall_count = 0
    waste_factor_total = 0
    for wall in walls_2d_JSON:
        for drywall in wall["polygons_drywall"]:
            if not drywall["enabled"]:
                continue
            if drywall["type_stacked"]:
                for drywall_type in drywall["type_stacked"]:
                    drywall_template = query_drywall(drywall_type, drywall_templates)
                    if not drywall_template:
                        continue
                    if compute_waste_average_standard:
                        waste_factor_total += float(drywall_template["waste"])
                    drywall_count += 1
                    weights_drywall[drywall_type] += 1
            else:
                drywall_template = query_drywall(drywall["type"], drywall_templates)
                if not drywall_template:
                    continue
                if compute_waste_average_standard:
                    waste_factor_total += float(drywall_template["waste"])
                drywall_count += 1
                weights_drywall[drywall["type"]] += 1
    for polygon in polygons_JSON:
        if not polygon["polygon_drywall"]["enabled"] or polygon["polygon_drywall"]["type"] == "DISABLED":
            continue
        drywall_template = query_drywall(polygon["polygon_drywall"]["type"], drywall_templates)
        if not drywall_template:
            continue
        if compute_waste_average_standard:
            waste_factor_total += float(drywall_template["waste"])
        drywall_count += 1
        weights_drywall[polygon["polygon_drywall"]["type"]] += 1
    for drywall_type in weights_drywall.keys():
        weights_drywall[drywall_type] /= drywall_count

    if compute_waste_average_standard:
        waste_average = 0
        if drywall_count != 0:
            waste_average = waste_factor_total / drywall_count
        return weights_drywall, waste_average, drywall_count
    return weights_drywall, drywall_count

async def load_visual_grounding(
    credentials,
    pg_pool,
    project_id,
    plan_id,
    user_id,
    ip_address,
    pages_metadata,
    batch_size=10,
    vertex_ai_client=None,
    vertex_ai_generation_config=None,
    is_cached=None
):
    pages_metadata_filtered = list()
    for page_metadata in pages_metadata[:]:
        query = f"SELECT mask_factor, bounding_box_offsets FROM {credentials["CloudSQL"]["table_name_pages"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
        query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, int(page_metadata["page_number"]),), fetch=True))
        mask_factor, bounding_box_offsets = query_output[0]["mask_factor"], query_output[0]["bounding_box_offsets"]
        mask_factor = json.loads(mask_factor) if isinstance(mask_factor, str) else mask_factor
        bounding_box_offsets = json.loads(bounding_box_offsets) if isinstance(bounding_box_offsets, str) else bounding_box_offsets
        if mask_factor and bounding_box_offsets:
            page_metadata["mask_factor"] = mask_factor
            page_metadata["bounding_box_offsets"] = bounding_box_offsets
            continue
        pages_metadata_filtered.append(page_metadata)
    n_pages = len(pages_metadata_filtered)
    page_batches = [list(map(lambda page_metadata: page_metadata["page_number"], pages_metadata_filtered[batch_index * batch_size: batch_index * batch_size + batch_size])) for batch_index in range(n_pages // batch_size)]
    if n_pages % batch_size != 0:
        page_batches += [list(map(lambda page_metadata: page_metadata["page_number"], pages_metadata_filtered[n_pages - (n_pages % batch_size): n_pages]))]

    plan_paths = dict()
    for page_metadata in pages_metadata_filtered:
        index = str(page_metadata["page_number"]).zfill(4)
        destination_path = f"/tmp/floor_plan_{index}.png"
        blob_name = "floor_plan.png"
        await download_floorplan(plan_id, project_id, user_id, credentials, pg_pool, index=index, blob_name=blob_name, destination_path=destination_path, wait_until_exists=True)
        plan_paths[page_metadata["page_number"]] = destination_path
    for page_batch in page_batches:
        bounding_boxes = detect_bounding_boxes(
            credentials,
            ip_address,
            [plan_paths[page_number] for page_number in page_batch],
            page_batch,
            vertex_ai_client=vertex_ai_client,
            vertex_ai_generation_config=vertex_ai_generation_config,
            is_cached=is_cached
        )
        for bounding_box in bounding_boxes["pages"]:
            for page_metadata in pages_metadata[:]:
                if bounding_box["page_number"] == page_metadata["page_number"]:
                    page_metadata["mask_factor"] = bounding_box["mask_factor"]
                    page_metadata["bounding_box_offsets"] = bounding_box["bounding_box_offsets"]
    return pages_metadata

async def download_floorplan(
    plan_id,
    project_id,
    user_id,
    credentials,
    pg_pool,
    index=None,
    blob_name="floor_plan.PDF",
    destination_path="/tmp/floor_plan.PDF",
    max_retries=5,
    wait_until_exists=False,
):
    def crc32c_base64(filename):
        checksum = google_crc32c.Checksum()
        with open(filename, "rb") as f:
            while chunk := f.read(1024 * 1024):
                checksum.update(chunk)
        return base64.b64encode(checksum.digest()).decode("utf-8")

    organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if index:
        blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{index}/{blob_name}"
    else:
        blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{blob_name}"

    blob = bucket.blob(blob_path)
    if wait_until_exists:
        while not blob.exists():
            sleep(1)
    blob.reload()

    expected_size = int(blob.size)
    expected_crc = blob.crc32c

    for attempt in range(max_retries):
        try:
            blob.download_to_filename(destination_path)
            actual_size = Path(destination_path).stat().st_size

            if actual_size != expected_size:
                raise RuntimeError(
                    f"Downloaded size mismatch "
                    f"({actual_size} != {expected_size})"
                )

            actual_crc = crc32c_base64(destination_path)

            if actual_crc != expected_crc:
                raise RuntimeError(
                    f"CRC32C mismatch "
                    f"({actual_crc} != {expected_crc})"
                )

            return (
                f"gs://"
                f"{credentials["CloudStorage"]["bucket_name"]}"
                f"/{blob_path}"
            )

        except Exception as e:

            logging.warning(
                "Download verification failed "
                f"(attempt {attempt+1}/{max_retries}): {e}"
            )

            try:
                Path(destination_path).unlink(missing_ok=True)
            except FileNotFoundError:
                pass

            if attempt == max_retries - 1:
                raise

            await asyncio.sleep(min(2 ** attempt, 30))

async def trigger_email_notification(
    credentials,
    pg_pool,
    status,
    project_id,
    plan_id,
    user_id,
    page_number,
    notify_group=False,
):
    message = f"Plan: {plan_id} | Page Number: {page_number} | Extraction: {status}"
    if notify_group:
        query = f"""
            WITH current_user_cte AS (
                SELECT %s AS user_id
            ),

            current_user_groups AS (
                SELECT DISTINCT group_id
                FROM {credentials["CloudSQL"]["table_name_users"]} u
                CROSS JOIN unnest(COALESCE(u.group_ids, ARRAY[]::text[])) AS group_id
                JOIN current_user_cte cu
                    ON LOWER(u.user_id) = LOWER(cu.user_id)
            ),

            matching_users AS (
                SELECT DISTINCT
                    g.user_id
                FROM {credentials["CloudSQL"]["table_name_groups"]} g
                JOIN current_user_groups cug
                    ON g.group_id = cug.group_id
            ),

            fallback_user AS (
                SELECT cu.user_id
                FROM current_user_cte cu
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM current_user_groups
                )
            ),

            final_users AS (
                SELECT user_id
                FROM matching_users

                UNION

                SELECT user_id
                FROM fallback_user
            )

            SELECT LOWER(user_id) AS user_id
            FROM final_users
        """
        query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(user_id,), fetch=True))
        user_ids_group = [row["user_id"] for row in query_output]
        for user_id_group in user_ids_group:
            trigger(
                credentials,
                credentials["Email"]["sender_email"],
                user_id_group,
                user_id_group,
                plan_id,
                project_id,
                page_number,
                "FBM Xtimator Team",
                message=message,
            )
    else:
        trigger(
            credentials,
            credentials["Email"]["sender_email"],
            user_id,
            user_id,
            plan_id,
            project_id,
            page_number,
            "FBM Xtimator Team",
            message=message,
        )

async def enforce_early_stopping(credentials, pg_pool, project_id, plan_id, user_id, pdf_path, pages_metadata):
    pages_metadata_unleashed = list()
    for page_metadata in pages_metadata:
        page_number = page_metadata["page_number"]
        logging.info(f"SYSTEM: Vector scale check STARTED for PAGE {page_number}")
        scale_value, scale_source = None, None
        if page_metadata.get("architectural_scale"):
            scale_value, scale_source = page_metadata["architectural_scale"], "frontend"
            pages_metadata_unleashed.append(page_metadata)
        else:
            query = f"SELECT scale FROM {credentials["CloudSQL"]["table_name_models"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s"
            architectural_scales = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, page_number,), fetch=True))
            if architectural_scales:
                for architectural_scale in architectural_scales:
                    if architectural_scale["scale"]:
                        page_metadata["architectural_scale"] = architectural_scale["scale"]
                        scale_value, scale_source = architectural_scale["scale"], "db"
            if is_vector(pdf_path, project_id, plan_id, page_number):
                query = (
                    f"SELECT vector_scale FROM {credentials["CloudSQL"]["table_name_pages"]} "
                    f"WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
                )
                query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, page_number,), fetch=True))
                if query_output and query_output[0]["vector_scale"]:
                    pages_metadata_unleashed.append(page_metadata)
                    continue
                vector_scale = extract_scale(pdf_path, page_number, project_id, plan_id)
                if vector_scale:
                    scale_value, scale_source = vector_scale, "vector"
                    pages_metadata_unleashed.append(page_metadata)
                else:
                    await insert_page(
                        plan_id,
                        user_id,
                        project_id,
                        page_number,
                        False,
                        "SCALE_NOT_DETECTED",
                        pg_pool,
                        credentials,
                    )
                    logging.info(f"SYSTEM: Scale NOT detected (vector, whole-page, pre-grounding) - pages.status SCALE_NOT_DETECTED for PAGE: {page_number}")
                    await trigger_email_notification(
                        credentials,
                        pg_pool,
                        "SCALE NOT DETECTED",
                        project_id,
                        plan_id,
                        user_id,
                        page_number=page_number,
                    )
            else:
                pages_metadata_unleashed.append(page_metadata)
            logging.info(f"SYSTEM: Scale check result PAGE {page_number}: {scale_value or 'none'} (source: {scale_source or 'none'})")
    return pages_metadata_unleashed

async def load_organization_slug(credentials, pg_pool, user_id):
    query = f"""SELECT COALESCE(o.organization_slug, 
            NULLIF(split_part(u.user_email,'@',2),''), 
            u.user_email) AS org_or_domain
        FROM {credentials["CloudSQL"]["table_name_users"]} u
        LEFT JOIN {credentials["CloudSQL"]["table_name_organizations"]} o ON TEXT(u.organization_id) = TEXT(o.organization_id)
        WHERE LOWER(u.user_email) = LOWER(%s);
    """
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(user_id,), fetch=True))
    if query_output and query_output[0]["org_or_domain"]:
        return query_output[0]["org_or_domain"]
    return user_id.split('@')[1]

def is_firebase_authenticated(credentials, request, user_id=None):
    authorization_header = request.headers.get("Authorization", None)
    if not authorization_header:
        return False, user_id
    id_token = authorization_header.split()[1]

    credential = credentials_firebase.Certificate(credentials["service_firebase_account_key"])
    drywall_app = firebase_admin.initialize_app(credential)

    try:
        decoded_token = auth_firebase.verify_id_token(id_token)
        user_id = decoded_token["email"]
        logging.info(f"SYSTEM: Authentication successfully verified for user: {user_id}")
        firebase_admin.delete_app(drywall_app)
        return True, user_id

    except auth_firebase.InvalidIdTokenError:
        logging.info("SYSTEM: Authentication verification failed due to Invalid or Expired ID token")
        firebase_admin.delete_app(drywall_app)
        return False, user_id

    except ValueError as e:
        logging.info(f"SYSTEM: Authentication verification failed due to Invalid ID token: {e}")
        firebase_admin.delete_app(drywall_app)
        return False, user_id

async def create_session(credentials, pg_pool, project_id, plan_id, user_id, page_number):
    session_uuid = uuid.uuid4().hex
    query = (
        f"INSERT INTO {credentials["CloudSQL"]["table_name_sessions"]} (session_id, user_id, project_id, plan_id, page_number, created_at, status) "
        f"VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s);"
    )
    await run_in_threadpool(partial(
        pg_run,
        credentials,
        pg_pool,
        query,
        params=(session_uuid, user_id, project_id, plan_id, page_number, "ACTIVE",),
    ))
    return session_uuid

async def is_session_active(credentials, pg_pool, session_id, project_id, plan_id, user_id, page_number):
    query = (
        f"SELECT status FROM {credentials["CloudSQL"]["table_name_sessions"]} "
        f"WHERE LOWER(session_id) = LOWER(%s) AND LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND LOWER(user_id) = LOWER(%s) AND page_number = %s;"
    )
    query_output = await run_in_threadpool(partial(
        pg_run,
        credentials,
        pg_pool,
        query,
        params=(session_id, project_id, plan_id, user_id, page_number,),
        fetch=True
    ))
    status = query_output[0]["status"]
    return status == "ACTIVE"

async def lock_user(
    credentials,
    pg_pool,
    user_id
):
    query = f"UPDATE {credentials["CloudSQL"]["table_name_users"]} SET is_locked = TRUE WHERE LOWER(user_email) = LOWER(%s);"
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(user_id,)))

async def unlock_user(
    credentials,
    pg_pool,
    user_id
):
    query = f"UPDATE {credentials["CloudSQL"]["table_name_users"]} SET is_locked = FALSE WHERE LOWER(user_email) = LOWER(%s);"
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(user_id,)))


# ═════════ Multi-family Matcher + Multiplier (D12, D13, D14) ═══════════════
# LIFTED from feat/multifamily xtimator-3d/helper.py:2150-2455 (matcher) and
# 2343-2455 (multiplier). Behaviour is UNCHANGED except for one addition,
# marked [UNIT_MULTIPLY LOG] in summarize_unit_counts: an explicit per-section
# log line (section title -> matched type -> count applied).
#
# NOT lifted here: the resolver itself (vector ranking, thumbnail triage,
# extraction, D10 precedence, persistence). That moved to the standalone
# unit-count-resolver Cloud Run service (master plan D5a).
#
# SCALING SEMANTICS (read from the code, reported in MF_REBUILD_REPORT.md,
# NOT changed): the multiply is applied PER SECTION -- each section's takeoff
# is multiplied by that section's own resolved count inside
# _accumulate_scaled_takeoff -- but the only SCALED artifact produced is the
# aggregate project_total. The per-section rows in section_rollup carry the
# matched type/count/method but no scaled takeoff, and the caller's existing
# drywall_takeoff_all rows are never modified. So: scaled per-section
# internally, surfaced only as a project total.

_UNIT_MATCH_FILLER = {
    "floor", "plan", "plans", "type", "types", "room", "rooms", "unit", "units",
    "enlarged", "dimensioned", "level", "building", "bldg", "sheet", "the", "and",
}


_UNIT_MATCH_ORDINAL_RE = re.compile(r"\b\d+\s*(?:st|nd|rd|th)\b")


_UNIT_MATCH_ABBREV = [
    (re.compile(r"\b(\d+)\s*br\b"), r"\1 bedroom"),
    (re.compile(r"\b(\d+)\s*ba\b"), r"\1 bath"),
    (re.compile(r"\bbr\b"), "bedroom"),
    (re.compile(r"\bba\b"), "bath"),
]


_UNIT_MATCH_AREA_TOLERANCE = 0.05


def _normalize_unit_label(label):
    """Normalize a unit-type label or section title to a token list for matching:
    lowercase, drop floor ordinals, expand BR/BA, strip punctuation and filler
    words (D14). Deterministic and auditable."""
    text = (label or "").lower()
    text = _UNIT_MATCH_ORDINAL_RE.sub(" ", text)
    for pattern, replacement in _UNIT_MATCH_ABBREV:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [token for token in text.split() if token and token not in _UNIT_MATCH_FILLER]


def _match_section_by_rules(section_tokens, normalized_types):
    """Rule match: a unit type matches when ALL its (normalized) tokens appear in
    the section's tokens. Prefer the most specific (largest) match; a tie between
    equally specific types is AMBIGUOUS (deferred, not forced). `normalized_types`
    is a list of (unit_type, token_list). Returns (unit_type|None, method)."""
    section_set = set(section_tokens)
    candidates = []
    for unit_type, type_tokens in normalized_types:
        type_set = set(type_tokens)
        if type_set and type_set <= section_set:
            candidates.append((unit_type, len(type_set)))
    if not candidates:
        return None, "none"
    candidates.sort(key=lambda c: -c[1])
    if len(candidates) > 1 and candidates[0][1] == candidates[1][1]:
        return None, "ambiguous"
    return candidates[0][0], "rule"


def _match_section_by_area(section_area, types_with_area, tolerance=_UNIT_MATCH_AREA_TOLERANCE):
    """Area fallback (D14): pick the unit type whose area is within `tolerance` of
    the section's area and closest. Ambiguous (two equally close within tolerance)
    or no section area -> None. `types_with_area` is a list of (unit_type, area)."""
    if section_area is None:
        return None
    scored = []
    for unit_type, area in types_with_area:
        if area is None or area <= 0:
            continue
        relative_diff = abs(area - section_area) / area
        if relative_diff <= tolerance:
            scored.append((unit_type, relative_diff))
    if not scored:
        return None
    scored.sort(key=lambda s: s[1])
    if len(scored) > 1 and abs(scored[0][1] - scored[1][1]) < 1e-9:
        return None   # equally close -> ambiguous, do not force
    return scored[0][0]


def _match_leftovers_llm(credentials, client_ip_address, leftovers, unit_types, context):
    """ONE batched LLM call (D14) for the sections that rules + area could not
    resolve. Returns {section_key: matched_unit_type_or_None}. Never raises; on any
    failure returns {} (callers treat missing as no-match)."""
    valid_types = {unit_type["unit_type"] for unit_type in unit_types}
    request_payload = {
        "unit_types": [{"unit_type": u["unit_type"], "area": u.get("area")} for u in unit_types],
        "sections": [
            {"index": index, "title": section["title"], "area": section.get("area")}
            for index, section in enumerate(leftovers)
        ],
    }
    try:
        vertex_ai_client, generation_config, is_cached = load_vertex_ai_client(
            credentials, client_ip_address, prompts=[UNIT_MATCH_RESOLVER]
        )
        query = Content(role="user", parts=[Part.from_text(json.dumps(request_payload))])
        if is_cached:
            response, _ = phoenix_call(
                lambda feedback_prompt, temperature: vertex_ai_client.generate_content(
                    contents=[feedback_prompt, query] if feedback_prompt else [query],
                    generation_config={**generation_config, "temperature": temperature},
                ),
                max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                pydantic_model=UnitMatchResponse,
                verify_field_counts=dict(matches=len(leftovers)),
            )
        else:
            response, _ = phoenix_call(
                lambda feedback_prompt, temperature: vertex_ai_client(UNIT_MATCH_RESOLVER).generate_content(
                    contents=[feedback_prompt, query] if feedback_prompt else [query],
                    generation_config={**generation_config, "temperature": temperature},
                ),
                max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                pydantic_model=UnitMatchResponse,
                verify_field_counts=dict(matches=len(leftovers)),
            )
    except Exception as e:
        logging.warning(f"[UNIT_MULTIPLY] [{context}] batched match LLM failed: {e}; leftovers -> no match")
        return {}

    resolved = {}
    for match in response.matches:
        if 0 <= match.index < len(leftovers):
            # Only accept a label the resolver actually offered (never invent a type).
            matched = match.matched_unit_type if match.matched_unit_type in valid_types else None
            resolved[leftovers[match.index]["key"]] = matched
    return resolved


def match_sections_to_unit_types(credentials, client_ip_address, sections, unit_types, project_id, plan_id):
    """Map each takeoff section to a resolved unit type (item 7 / D14). Does NOT
    multiply — produces the section->type mapping only.

    Args:
        sections: list of {"key": <hashable>, "title": <page_section_number/title>,
                  "area": <sqft or None>}.
        unit_types: the resolved plans.unit_counts["unit_counts"] list
                    ({"unit_type", "count", "area"}); empty when nothing resolved.

    Returns {section_key: {"matched_unit_type": <str or None>, "method":
    "rule"|"area"|"llm"|"none"}}. Honest "no match" (None) whenever nothing
    resolves — never a forced/wrong match.
    """
    context = f"project={project_id} plan={plan_id}"
    result = {}

    # No resolved unit types -> every section is a no-match (defaults x1 in item 8).
    if not unit_types:
        for section in sections:
            result[section["key"]] = {"matched_unit_type": None, "method": "none"}
            logging.info(
                f"[UNIT_MULTIPLY] [{context}] section \"{section['title']}\" → no match "
                f"(no resolved unit types), will default ×1"
            )
        return result

    normalized_types = [(u["unit_type"], _normalize_unit_label(u["unit_type"])) for u in unit_types]
    types_with_area = [(u["unit_type"], u.get("area")) for u in unit_types]

    leftovers = []
    for section in sections:
        section_tokens = _normalize_unit_label(section["title"])
        matched, method = _match_section_by_rules(section_tokens, normalized_types)
        if matched is None:
            area_match = _match_section_by_area(section.get("area"), types_with_area)
            if area_match is not None:
                matched, method = area_match, "area"
        if matched is not None:
            result[section["key"]] = {"matched_unit_type": matched, "method": method}
            logging.info(
                f"[UNIT_MULTIPLY] [{context}] section \"{section['title']}\" → matched {matched} ({method})"
            )
        else:
            leftovers.append(section)

    # Stage 3: ONE batched LLM call for everything rules + area left unresolved.
    if leftovers:
        logging.info(f"[UNIT_MULTIPLY] [{context}] {len(leftovers)} section(s) unresolved by rules/area → batched LLM")
        llm_matches = _match_leftovers_llm(credentials, client_ip_address, leftovers, unit_types, context)
        for section in leftovers:
            matched = llm_matches.get(section["key"])
            if matched is not None:
                result[section["key"]] = {"matched_unit_type": matched, "method": "llm"}
                logging.info(
                    f"[UNIT_MULTIPLY] [{context}] section \"{section['title']}\" → matched {matched} (llm)"
                )
            else:
                result[section["key"]] = {"matched_unit_type": None, "method": "none"}
                logging.info(
                    f"[UNIT_MULTIPLY] [{context}] section \"{section['title']}\" → no match, will default ×1"
                )

    return result


def _num(value):
    """Numeric value or 0 (booleans and non-numbers excluded)."""
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, (int, float)) else 0


def _accumulate_scaled_takeoff(accumulator, takeoff, count):
    """Add one section's takeoff (scaled by `count`) into the project accumulator.
    Mirrors the compute_takeoff shape: total.{roof,wall} + per_drywall.{roof,wall}
    .<type>.<numeric fields>. Non-numeric fields are ignored."""
    total = takeoff.get("total") or {}
    for surface in ("roof", "wall"):
        accumulator["total"][surface] += _num(total.get(surface, 0)) * count
    per_drywall = takeoff.get("per_drywall") or {}
    for surface in ("roof", "wall"):
        for drywall_type, fields in (per_drywall.get(surface) or {}).items():
            bucket = accumulator["per_drywall"][surface].setdefault(drywall_type, {})
            for field, value in (fields or {}).items():
                numeric = _num(value)
                if numeric or field in bucket:
                    bucket[field] = bucket.get(field, 0) + numeric * count


def _round_takeoff(accumulator):
    """Round the accumulated project total for display (matches compute_takeoff)."""
    for surface in ("roof", "wall"):
        accumulator["total"][surface] = round(accumulator["total"][surface], 2)
        for fields in accumulator["per_drywall"][surface].values():
            for field in fields:
                fields[field] = round(fields[field], 2)


def summarize_unit_counts(credentials, client_ip_address, drywall_takeoff_all, unit_counts_payload, project_id, plan_id):
    """Build the additive multi-family rollup for summarize_takeoff_all (item 8).

    Matches each collected section (from drywall_takeoff_all) to a resolved unit
    type, multiplies its takeoff by the resolved count (unmatched → ×1), and sums
    into a project total. Returns a summary dict to attach to the response; the
    caller's existing payload is not modified. Section area is passed to the matcher
    ONLY if already present on the row (no new extraction) — otherwise absent.
    """
    context = f"project={project_id} plan={plan_id}"
    unit_types = (unit_counts_payload or {}).get("unit_counts") or []
    counts_by_type = {u["unit_type"]: u.get("count", 1) for u in unit_types}
    logging.info(
        f"[UNIT_MULTIPLY] [{context}] multiplier START — sections={len(drywall_takeoff_all)} "
        f"resolved_source={(unit_counts_payload or {}).get('source', 'none_found')} unit_types={len(unit_types)}"
    )

    sections = [
        {
            "key": (row.get("page_number"), row.get("page_section_number")),
            "title": str(row.get("page_section_number")),
            "area": row.get("area"),   # present only if the row already carries it
        }
        for row in drywall_takeoff_all
    ]
    matches = match_sections_to_unit_types(credentials, client_ip_address, sections, unit_types, project_id, plan_id)

    project_total = {"total": {"roof": 0.0, "wall": 0.0}, "per_drywall": {"roof": {}, "wall": {}}}
    section_rollup = []
    applied = False
    for row in drywall_takeoff_all:
        key = (row.get("page_number"), row.get("page_section_number"))
        match = matches.get(key, {"matched_unit_type": None, "method": "none"})
        matched_type = match["matched_unit_type"]
        count = counts_by_type.get(matched_type, 1) if matched_type is not None else 1
        if count != 1:
            applied = True
        takeoff = row.get("takeoff")
        takeoff = json.loads(takeoff) if isinstance(takeoff, str) else (takeoff or {})
        _accumulate_scaled_takeoff(project_total, takeoff, count)
        section_rollup.append({
            "page_number": row.get("page_number"),
            "page_section_number": row.get("page_section_number"),
            "matched_unit_type": matched_type,
            "count": count,
            "method": match["method"],
        })
        # [UNIT_MULTIPLY LOG] — item 12. The lifted line logged only
        # `section "X" → <type> × <count>`. This states all three required facts
        # explicitly — section title, matched type, and the count actually
        # APPLIED to that section — plus the match method, so a ×1 caused by "no
        # match" is distinguishable in logs from a ×1 that is the type's real
        # resolved count. Behaviour is unchanged; this is logging only.
        logging.info(
            f"[UNIT_MULTIPLY] [{context}] section \"{row.get('page_section_number')}\" "
            f"(page={row.get('page_number')}) → matched_type={matched_type or 'NO MATCH'} "
            f"(method={match['method']}) → count_applied=×{count}"
            f"{'' if matched_type is not None else ' [D13 default]'}"
        )

    _round_takeoff(project_total)
    provenance = (unit_counts_payload or {}).get("provenance") or {}
    summary = {
        "applied": applied,   # False => every section ×1 (single-family / no counts)
        "resolver": {
            "source": (unit_counts_payload or {}).get("source", "none_found"),
            "total_units": (unit_counts_payload or {}).get("total_units"),
            "source_form": provenance.get("source_form"),
            "source_pages": provenance.get("source_pages"),
            "detection_path": provenance.get("detection_path"),
            "disagreement": provenance.get("disagreement"),
        },
        "sections": section_rollup,
        "project_total": project_total,
    }
    logging.info(
        f"[UNIT_MULTIPLY] [{context}] ROLLUP applied={applied} sections={len(section_rollup)} "
        f"project_total wall={project_total['total']['wall']} roof={project_total['total']['roof']}"
    )
    return summary


