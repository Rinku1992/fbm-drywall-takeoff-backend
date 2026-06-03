import logging
import json
import sys
import os
import requests
from pathlib import Path
from ruamel.yaml import YAML
from time import sleep
import datetime

from random import uniform
from PIL import Image
import numpy as np
import math
import random
random.seed(0)
from functools import partial
from fastapi.concurrency import run_in_threadpool

import geoip2.database as geoip2_database
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud.storage import Client as CloudStorageClient
from google.cloud import bigquery
from fastapi.encoders import jsonable_encoder
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from google.api_core.exceptions import (
    BadRequest,
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
    TooManyRequests
)
from google.auth.transport.requests import Request
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy.pool import QueuePool
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
from vertexai.generative_models import Content, Part
from vertexai.caching import CachedContent
from google.oauth2 import service_account
from google.cloud.pubsub_v1 import PublisherClient

from transcriber import Transcriber
from prompts import FEEDBACK_GENERATOR
from email_notification import trigger


def load_vertex_ai_client(credentials, ip_address, prompts=None, default_region="us-central1"):
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
        cached_content = CachedContent.create(
            model_name=credentials["VertexAI"]["llm"]["model_name"],
            contents=prompts,
            ttl=datetime.timedelta(minutes=60),
            display_name="drywall_predictor_cache"
        )
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

def transcribe(credentials, hyperparameters, floor_plan_path):
    transcriber = Transcriber(credentials, hyperparameters)
    return transcriber.transcribe(floor_plan_path, [0, 1, -1, -2])

def upload_floorplan(plan_path, plan_id, project_id, credentials, index=None, directory=None):
    client = CloudStorageClient()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if directory:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{directory}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{directory}/{blob_object_name}"
    else:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{blob_object_name}"
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(plan_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"

def enable_logging_on_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s"}',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

def load_gcp_credentials() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("gcp.yaml", 'r') as f:
        credentials = yaml.load(f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials["service_drywall_account_key"]

    return credentials

def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("hyperparameters.yaml", 'r') as f:
        hyperparameters = yaml.load(f)

    return hyperparameters

def download_floorplan(user_id, plan_id, project_id, credentials, index=None, destination_path="/tmp/floor_plan_wall_processed.png"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if index:
        blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/floor_plan.png"
        blob = bucket.blob(blob_path)

        destination_path = Path(destination_path)
        destination_path = destination_path.parent.joinpath(project_id).joinpath(plan_id).joinpath(user_id).joinpath(destination_path.name)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(destination_path)
        return destination_path

    destination_path="/tmp/floor_plan.PDF"
    destination_path = Path(destination_path)
    destination_path = destination_path.parent.joinpath(project_id).joinpath(plan_id).joinpath(user_id).joinpath(destination_path.name)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)

    blob.download_to_filename(destination_path)
    return destination_path

def download_segmented_walls(plan_id, project_id, index, credentials, destination_path="/tmp/floor_plan_wall_segmented.png"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/wall_detected.png"
    blob = bucket.blob(blob_path)

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_path)
    return destination_path

_pg_pool = None
_connector = None

def load_pg_pool(credentials):
    global _pg_pool, _connector

    if _pg_pool is not None:
        return _pg_pool

    if _connector is None:
        _connector = Connector()

    pg = credentials["CloudSQL"]
    instance_connection_name = pg["connection_name"]
    db_name = pg["database_name"]
    sa_key = pg["service_account_key"]
    with open(sa_key, "r") as f:
        sa_payload = json.load(f)
    user = sa_payload["client_email"]

    def get_conn():
        creds = service_account.Credentials.from_service_account_file(
            sa_key,
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform"
            ]
        )
        creds.refresh(Request())

        conn = _connector.connect(
            instance_connection_name,
            pg["driver"],
            user=user,
            password=creds.token,
            db=db_name,
            enable_iam_auth=True,
            ip_type=IPTypes.PRIVATE
        )

        conn.autocommit = True
        return conn

    _pg_pool = QueuePool(
        creator=get_conn,
        pool_size=pg.get("min_pool_size", 3),
        max_overflow=max(
            pg.get("max_pool_size", 10)
            - pg.get("min_pool_size", 3),
            0
        ),
        timeout=30,
        recycle=3000
    )

    return _pg_pool

def close_pg_pool():
    global _pg_pool, _connector

    if _pg_pool:
        _pg_pool.dispose()
        _pg_pool = None

    if _connector:
        _connector.close()
        _connector = None

def pg_run(
    connection_pool,
    query,
    params=None,
    fetch=False,
    max_retries=5,
    initial_backoff=1.0,
    max_backoff=30.0,
    execute_many=False,
):
    if params is None:
        params = tuple()

    for attempt in range(max_retries):
        conn = None
        cursor = None

        try:
            conn = connection_pool.connect()
            cursor = conn.cursor()
            if execute_many:
                cursor.executemany(query, params)
            else:
                cursor.execute(query, params)
            result = None

            if fetch:
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                result = [
                    dict(zip(columns, row))
                    for row in rows
                ]
            conn.commit()

            return result

        except (DBAPIError, DatabaseErrorPG8000) as e:
            error_message = str(e).lower()
            retryable_db_terms = [
                "deadlock detected",
                "serialization failure",
                "could not serialize access",
                "lock not available",
                "too many connections",
            ]

            retryable_network_terms = [
                "connection reset",
                "connection aborted",
                "server closed the connection",
                "could not connect",
                "timeout expired",
                "broken pipe",
                "ssl syscall error",
                "terminating connection",
                "network is unreachable",
                "network error",
            ]
            should_retry = (
                any(t in error_message for t in retryable_db_terms)
                or any(t in error_message for t in retryable_network_terms)
            )
            if conn:
                try:
                    conn.invalidate()
                except Exception:
                    pass

            if should_retry:
                sleep_time = min(
                    initial_backoff * (2 ** attempt)
                    + random.uniform(0, 1),
                    max_backoff
                )
                logging.warning(
                    f"SYSTEM: PostgreSQL failure "
                    f"attempt={attempt + 1}/{max_retries}. "
                    f"Retrying in {sleep_time:.2f}s. "
                    f"Error={type(e).__name__}: {e}"
                )
                sleep(sleep_time)
                continue

            raise

        except (
            OperationalError,
            InterfaceError,
            TimeoutError,
            InterfaceErrorPG8000
        ) as e:
            if conn:
                try:
                    conn.invalidate()
                except Exception:
                    pass
            sleep_time = min(
                initial_backoff * (2 ** attempt)
                + random.uniform(0, 1),
                max_backoff
            )
            logging.warning(
                f"SYSTEM: PostgreSQL connection failure "
                f"({type(e).__name__}) "
                f"attempt={attempt + 1}/{max_retries}. "
                f"Retrying in {sleep_time:.2f}s"
            )

            sleep(sleep_time)
            continue

        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass

            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    raise RuntimeError(
        f"SYSTEM: PostgreSQL query failed after "
        f"{max_retries} retries."
    )

def load_bigquery_client(credentials):
    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    return bigquery_client

def bigquery_run(
    credentials,
    bigquery_client,
    GBQ_query,
    job_config=dict(),
    max_retries=5,
    initial_backoff=1.0,
    max_backoff=30.0
):
    query_job_config = bigquery.QueryJobConfig(
        destination_encryption_configuration=bigquery.EncryptionConfiguration(
            kms_key_name=credentials["GBQServer"]["KMS_key"]
        ),
        **job_config
    )

    for attempt in range(max_retries):
        try:
            query_output = bigquery_client.query(
                GBQ_query,
                job_config=query_job_config
            )

            return query_output

        except (
            ResourceExhausted,
            ServiceUnavailable,
            DeadlineExceeded,
            InternalServerError,
            TooManyRequests
        ) as e:

            sleep_time = min(
                initial_backoff * (2 ** attempt) + random.uniform(0, 1),
                max_backoff
            )

            logging.warning(
                f"SYSTEM: Transient BigQuery error "
                f"({type(e).__name__}) "
                f"attempt={attempt + 1}/{max_retries}. "
                f"Retrying in {sleep_time:.2f}s"
            )

            sleep(sleep_time)

        except BadRequest as e:
            error_message = str(e)

            if "Could not serialize access to table" in error_message:

                sleep_time = min(
                    initial_backoff * (2 ** attempt) + random.uniform(0, 1),
                    max_backoff
                )

                logging.warning(
                    f"SYSTEM: BigQuery concurrent update conflict "
                    f"attempt={attempt + 1}/{max_retries}. "
                    f"Retrying in {sleep_time:.2f}s"
                )

                sleep(sleep_time)
                continue

            raise

        except Exception as e:
            error_message = str(e).lower()

            retryable_terms = [
                "connection reset",
                "connection aborted",
                "timed out",
                "temporarily unavailable",
                "network is unreachable",
                "broken pipe"
            ]

            if any(term in error_message for term in retryable_terms):
                sleep_time = min(
                    initial_backoff * (2 ** attempt) + random.uniform(0, 1),
                    max_backoff
                )

                logging.warning(
                    f"SYSTEM: Network-related BigQuery failure "
                    f"attempt={attempt + 1}/{max_retries}. "
                    f"Retrying in {sleep_time:.2f}s"
                )

                sleep(sleep_time)
                continue

            raise

    raise RuntimeError(
        f"SYSTEM: BigQuery query failed after "
        f"{max_retries} retries."
    )

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
    await run_in_threadpool(partial(pg_run, pg_pool, query, params=(
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

async def insert_model_2d(
    model_2d,
    scale,
    page_number,
    page_sections,
    page_section_number,
    plan_id,
    user_id,
    project_id,
    target_drywalls,
    pg_pool,
    credentials
    ):
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
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (project_id, plan_id, page_number, page_section_number) DO UPDATE SET
            model_2d = EXCLUDED.model_2d,
            scale = COALESCE(NULLIF(EXCLUDED.scale, ''), t.scale),
            user_id = EXCLUDED.user_id,
            updated_at = CURRENT_TIMESTAMP
    """
    await run_in_threadpool(partial(pg_run, pg_pool, query, params=(
        plan_id,
        project_id,
        user_id,
        int(page_number),
        int(page_sections),
        page_section_number,
        scale,
        json.dumps(model_2d),
        target_drywalls,
    )))

def insert_model_2d_batch(rows, bigquery_client, credentials):
    GBQ_query = """
    MERGE `drywall_takeoff.models` t
    USING UNNEST(@rows) s
    ON LOWER(t.project_id) = LOWER(s.project_id)
       AND LOWER(t.plan_id) = LOWER(s.plan_id)
       AND t.page_number = s.page_number
       AND t.page_section_number = s.page_section_number

    WHEN MATCHED THEN
    UPDATE SET
        model_2d = SAFE.PARSE_JSON(s.model_2d),
        scale = COALESCE(NULLIF(s.scale, ''), t.scale),
        user_id = s.user_id,
        updated_at = CURRENT_TIMESTAMP()

    WHEN NOT MATCHED THEN
    INSERT (
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
        target_drywalls,
        created_at,
        updated_at
    )
    VALUES (
        s.plan_id,
        s.project_id,
        s.user_id,
        s.page_number,
        s.page_sections,
        s.page_section_number,
        s.scale,
        SAFE.PARSE_JSON(s.model_2d),
        JSON '{}',
        JSON '{}',
        s.target_drywalls,
        CURRENT_TIMESTAMP(),
        CURRENT_TIMESTAMP()
    )
    """

    job_config = dict(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "rows",
                "STRUCT<\
                    plan_id STRING,\
                    project_id STRING,\
                    user_id STRING,\
                    page_number INT64,\
                    page_sections INT64,\
                    page_section_number STRING,\
                    scale STRING,\
                    model_2d JSON,\
                    target_drywalls STRING\
                >",
                rows
            )
        ],
    )

    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output

async def load_templates(pg_pool, credentials):
    query = f"SELECT * FROM {credentials["CloudSQL"]["table_name_sku"]}"
    product_templates = await run_in_threadpool(partial(pg_run, pg_pool, query, fetch=True))

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
    return jsonable_encoder(product_templates_target)

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
        except (ResourceExhausted, DeadlineExceeded) as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            sleep_time = base_delay * (2 ** (n_iterations - 1)) + uniform(0, 0.5)
            sleep(sleep_time)
            logging.warning(f"SYSTEM: Vertex AI Gemini: {e}: RETRYING ...")
        except ServiceUnavailable as e:
            logging.warning(f"SYSTEM: Vertex AI Gemini: {e}")
            raise e
        except Exception as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            exceptions.append(e)
            system_feedback = [Part.from_text(FEEDBACK_GENERATOR.format(max_retry=max_retry, exceptions=exceptions))]
            feedback_prompt = Content(role="model", parts=system_feedback)
            temperature = min(0.5 * (n_iterations + 1) / max_retry, 0.5)
            logging.warning(f"SYSTEM: Vertex AI Gemini: Response Generation/Parsing failed with ERROR: {e}: RETRYING ...")
            logging.warning(f"SYSTEM: RETRYING with TEMPERATURE: {temperature}")

def load_section_from_page(wall_segmented_path, floor_plan_path, bounding_box_offset, section_name):
    offset_top_left_X, offset_top_left_Y = bounding_box_offset["offset_top_left"]
    offset_bottom_right_X, offset_bottom_right_Y = bounding_box_offset["offset_bottom_right"]
    offset_top_left_X = max(offset_top_left_X - 0.05, 0)
    offset_top_left_Y = max(offset_top_left_Y - 0.05, 0)
    offset_bottom_right_X = min(offset_bottom_right_X + 0.05, 1)
    offset_bottom_right_Y = min(offset_bottom_right_Y + 0.05, 1)
    canvas = Image.open(wall_segmented_path)
    canvas = canvas.convert("RGB")
    width_in_pixels, height_in_pixels = canvas.size
    canvas_original = Image.open(floor_plan_path)
    canvas_original = canvas_original.convert("RGB")
    width_in_pixels_original, height_in_pixels_original = canvas_original.size

    canvas = canvas.resize((width_in_pixels_original, height_in_pixels_original), Image.Resampling.NEAREST)
    image = np.array(canvas).copy()
    LEFT = round(offset_top_left_X * width_in_pixels_original)
    TOP = round(offset_top_left_Y * height_in_pixels_original)
    BOTTOM = round(offset_bottom_right_Y * height_in_pixels_original)
    RIGHT = round(offset_bottom_right_X * width_in_pixels_original)
    image[:TOP, :] = 255
    image[:, :LEFT] = 255
    image[:, RIGHT:] = 255
    image[BOTTOM:, :] = 255
    canvas = Image.fromarray(image)
    canvas = canvas.resize((width_in_pixels, height_in_pixels), Image.Resampling.NEAREST)
    wall_segmented_path_sectioned = wall_segmented_path.parent.joinpath(f"{wall_segmented_path.stem}_sectioned_{section_name.replace('/', '_')}").with_suffix(".png")
    canvas.save(wall_segmented_path_sectioned, format="png")

    return str(wall_segmented_path_sectioned)

def apply_pixel_margin_to_bounding_box(bounding_box_offset, margin_offset=0):
    offset_top_left_X, offset_top_left_Y = bounding_box_offset["offset_top_left"]
    offset_bottom_right_X, offset_bottom_right_Y = bounding_box_offset["offset_bottom_right"]
    offset_top_left_X = max(offset_top_left_X - margin_offset, 0)
    offset_top_left_Y = max(offset_top_left_Y - margin_offset, 0)
    offset_bottom_right_X = min(offset_bottom_right_X + margin_offset, 1)
    offset_bottom_right_Y = min(offset_bottom_right_Y + margin_offset, 1)

    return (offset_top_left_X, offset_top_left_Y), (offset_bottom_right_X, offset_bottom_right_Y)

def polygon_to_structured_2d(credentials, query_json):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["polygon_to_structured_2d"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token

    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{credentials["CloudRun"]["APIs"]["polygon_to_structured_2d"]}/polygon_to_structured_2d",
        headers=headers,
        json=query_json
    )
    return response.status_code, response.content

def load_publisher_client(credentials):
     credentials_SA = service_account.Credentials.from_service_account_file(credentials["PubSub"]["service_account_key"])
     publisher = PublisherClient(credentials=credentials_SA)
     publisher_client = lambda payload: publisher.publish(credentials["PubSub"]["topic_name"], json.dumps(payload).encode("utf-8"))

     return publisher_client

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
    query = f"SELECT group_id FROM {credentials["CloudSQL"]["table_name_users"]}, unnest(COALESCE(group_ids, ARRAY[]::text[])) AS group_id WHERE LOWER(user_id) = LOWER(%s)"
    query_output = await run_in_threadpool(partial(pg_run, pg_pool, query, params=(user_id,), fetch=True))
    group_ids = [row["group_id"] for row in query_output]
    group_id = " | ".join(group_ids)
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
        query_output = await run_in_threadpool(partial(pg_run, pg_pool, query, params=(user_id,), fetch=True))
        user_ids_group = [row["user_id"] for row in query_output]
        for user_id_group in user_ids_group:
            trigger(
                credentials,
                user_id,
                user_id_group,
                user_id_group,
                plan_id,
                project_id,
                page_number,
                group_id,
                message=message,
            )
    else:
        trigger(
            credentials,
            user_id,
            user_id,
            user_id,
            plan_id,
            project_id,
            page_number,
            group_id,
            message=message,
        )
