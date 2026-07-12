import asyncio
import logging
import json
import sys
import os
import requests
from pathlib import Path
from ruamel.yaml import YAML
from time import sleep
import base64
from copy import deepcopy
from json.decoder import JSONDecodeError

from PIL import Image
import numpy as np
import math
import random
random.seed(0)
from functools import partial
from fastapi.concurrency import run_in_threadpool

import geoip2.database as geoip2_database
from google.cloud.storage import Client as CloudStorageClient
import google_crc32c
from fastapi.encoders import jsonable_encoder
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from google.api_core.exceptions import DeadlineExceeded
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
from google.oauth2 import service_account
from google.cloud.pubsub_v1 import SubscriberClient
from google.cloud.pubsub_v1 import PublisherClient

from transcriber import Transcriber
from vector_pdf import is_vector, extract_scales_from_sections_of_a_page
from email_notification import trigger


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

async def upload_floorplan(
    plan_path,
    plan_id,
    project_id,
    user_id,
    credentials,
    pg_pool,
    index=None,
    directory=None,
    max_retries=5
):
    def crc32c_base64(filename):
        checksum = google_crc32c.Checksum()
        with open(filename, "rb") as f:
            while chunk := f.read(1024 * 1024):
                checksum.update(chunk)
        return base64.b64encode(checksum.digest()).decode("utf-8")

    client = CloudStorageClient()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
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

async def download_floorplan(
    user_id,
    plan_id,
    project_id,
    credentials,
    pg_pool,
    index=None,
    destination_path="/tmp/floor_plan_wall_processed.png",
    max_retries=5
):
    def crc32c_base64(filename):
        checksum = google_crc32c.Checksum()
        with open(filename, "rb") as f:
            while chunk := f.read(1024 * 1024):
                checksum.update(chunk)
        return base64.b64encode(checksum.digest()).decode("utf-8")

    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
    if index:
        blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{index}/floor_plan.png"
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
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
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

            return destination_path

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

async def download_segmented_walls(plan_id, project_id, user_id, index, credentials, pg_pool, destination_path="/tmp/floor_plan_wall_segmented.png"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
    blob_path = f"{organization_slug}/{project_id.lower()}/{plan_id.lower()}/{index}/wall_detected.png"
    blob = bucket.blob(blob_path)

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_path)
    return destination_path

_pg_engine = None
_connector = None
_db_credentials = None

def load_pg_pool(credentials):
    global _pg_engine, _connector, _db_credentials

    if _pg_engine is not None:
        return _pg_engine

    if _connector is None:
        _connector = Connector()

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
        _pg_engine.dispose()
        _pg_engine = None

    if _connector is not None:
        _connector.close()
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
        target_drywalls,
    )))

async def load_templates(pg_pool, credentials):
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
    return jsonable_encoder(product_templates_target)

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
    queries_unacknowledged = deepcopy(queries)
    for received_message in response.received_messages:
        try:
            message = json.loads(received_message.message.data.decode("utf-8"))
            for query in queries_unacknowledged:
                if query["session_uuid"] == message.get("session_uuid") and query["page_section_number"] == message.get("page_section_number"):
                    subscriber_client.acknowledge(
                        request=dict(subscription=subscription_path, ack_ids=[received_message.ack_id])
                    )
                    acknowledged_queries.append(message)
                    queries_unacknowledged.remove(query)
                    if len(acknowledged_queries) == len(queries):
                        return True, acknowledged_queries
                    break
        except JSONDecodeError:
            continue
    return False, acknowledged_queries

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
                user_id,
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
            user_id,
            user_id,
            user_id,
            plan_id,
            project_id,
            page_number,
            "FBM Xtimator Team",
            message=message,
        )

async def load_metadata_from_vector_pdf(
    credentials,
    pg_pool,
    pdf_path,
    project_id,
    plan_id,
    page_number,
    bounding_box_offsets
):
    vector_scales = [None for _ in bounding_box_offsets]
    vector_ceiling_heights = [None for _ in bounding_box_offsets]
    query = (
        f"SELECT is_vector, vector_scale, vector_ceiling_height FROM {credentials["CloudSQL"]["table_name_pages"]} "
        f"WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
    )
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(project_id, plan_id, page_number,), fetch=True))
    if query_output and query_output[0]["is_vector"] is not None:
        is_vector_pdf = query_output[0]["is_vector"]
        if is_vector_pdf:
            vector_scales = query_output[0]["vector_scale"]
            vector_scales = json.loads(vector_scales) if isinstance(vector_scales, str) else vector_scales
            vector_ceiling_heights = query_output[0]["vector_ceiling_height"]
            vector_ceiling_heights = json.loads(vector_ceiling_heights) if isinstance(vector_ceiling_heights, str) else vector_ceiling_heights
        return is_vector_pdf, vector_scales, vector_ceiling_heights

    is_vector_pdf = is_vector(pdf_path, project_id, plan_id, page_number)
    query = (
        f"UPDATE {credentials["CloudSQL"]["table_name_pages"]} "
        f"SET is_vector = %s "
        f"WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s;"
    )
    await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(is_vector_pdf, project_id, plan_id, page_number,)))

    if is_vector_pdf:
        metadata = extract_scales_from_sections_of_a_page(
            pdf_path, page_number, bounding_box_offsets, project_id, plan_id, apply_pixel_margin_to_bounding_box
        )
        vector_scales = [data["scale"] for data in metadata]
        query = (
            f"UPDATE {credentials["CloudSQL"]["table_name_pages"]} "
            f"SET vector_scale = %s "
            f"WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) "
            f"AND page_number = %s;"
        )
        await run_in_threadpool(partial(
            pg_run, credentials, pg_pool, query,
            params=(json.dumps(vector_scales), project_id, plan_id, int(page_number),)
        ))
        vector_ceiling_heights = [data["ceiling_height"] for data in metadata]
        query = (
            f"UPDATE {credentials["CloudSQL"]["table_name_pages"]} "
            f"SET vector_ceiling_height = %s "
            f"WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) "
            f"AND page_number = %s;"
        )
        await run_in_threadpool(partial(
            pg_run, credentials, pg_pool, query,
            params=(json.dumps(vector_ceiling_heights), project_id, plan_id, int(page_number),)
        ))
    return is_vector_pdf, vector_scales, vector_ceiling_heights

async def load_organization_slug(credentials, pg_pool, user_id):
    query = f"""SELECT COALESCE(o.organization_slug, 
            NULLIF(split_part(u.user_id,'@',2),''), 
            u.user_id) AS org_or_domain
        FROM {credentials["CloudSQL"]["table_name_users"]} u
        LEFT JOIN {credentials["CloudSQL"]["table_name_organizations"]} o ON TEXT(u.organization_id) = TEXT(o.organization_id)
        WHERE LOWER(u.user_id) = LOWER(%s);
    """
    query_output = await run_in_threadpool(partial(pg_run, credentials, pg_pool, query, params=(user_id,), fetch=True))
    if query_output and query_output[0]["org_or_domain"]:
        return query_output[0]["org_or_domain"]
    return user_id.split('@')[1]

async def update_status(credentials, pg_pool, status, project_id, plan_id, user_id, page_number):
    await insert_page(
        plan_id,
        user_id,
        project_id,
        page_number,
        False,
        status,
        pg_pool,
        credentials,
    )

async def return_futures_early_section_to_structured_2d(futures):
    tasks = futures
    results = list()

    while tasks:
        done, tasks = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                is_success, model_2d, scale, page_section_number = task.result()

                if not is_success:
                    for task in tasks:
                        task.cancel()

                    await asyncio.gather(*tasks, return_exceptions=True)
                    return [[False, dict(), None, None]] * len(futures)
                results.append([is_success, model_2d, scale, page_section_number])

            except Exception:
                for task in tasks:
                    task.cancel()

                await asyncio.gather(*tasks, return_exceptions=True)
                return [[False, dict(), None, None]] * len(futures)
    return results

async def is_session_active(credentials, pg_pool, session_id, project_id, plan_id, user_id, page_number, completed_as_active=False):
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
    if completed_as_active:
        return status == "ACTIVE" or status == "COMPLETED"
    return status == "ACTIVE"

async def terminate_session(credentials, pg_pool, session_id):
    query = (
        f"UPDATE {credentials["CloudSQL"]["table_name_sessions"]} SET status = 'COMPLETED' "
        f"WHERE LOWER(session_id) = LOWER(%s);"
    )
    await run_in_threadpool(partial(
        pg_run,
        credentials,
        pg_pool,
        query,
        params=(session_id,),
    ))
