import json
from json.decoder import JSONDecodeError
import logging
import hashlib
import requests
from pathlib import Path
import datetime
from time import sleep
from pypdf import PdfReader, PdfWriter
from io import BytesIO
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import subprocess
from collections import defaultdict

import math
import random
random.seed(0)
import cv2
from PIL import Image

import geoip2.database as geoip2_database
from google.cloud import bigquery
from google.cloud.storage import Client as CloudStorageClient
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from google.oauth2 import service_account
from google.cloud.pubsub_v1 import SubscriberClient
from google.api_core.exceptions import (
    BadRequest,
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
    TooManyRequests
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
    FEEDBACK_GENERATOR
)
from preprocessing import preprocess


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

def sha256(path, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

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

def insert_model_2d(
    model_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    GCS_URL_floorplan_page,
    GCS_URL_target_drywalls_page,
    bigquery_client,
    credentials,
    page_section_number=None,
    page_sections=None,
    ):
    if not page_section_number:
        page_section_number = 'I'
    if not page_sections:
        GBQ_query = f"SELECT page_sections FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number} AND page_section_number = '{page_section_number}';"
        query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
        page_sections = list(query_output)[0].page_sections
    if not model_2d.get("metadata", None):
        GBQ_query = f"SELECT model_2d.metadata FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number} AND page_section_number = '{page_section_number}';"
        query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
        metadata = list(query_output)[0].metadata
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        model_2d["metadata"] = metadata
    GBQ_query = """
    MERGE `drywall_takeoff.models` t
    USING (
        SELECT
            @plan_id AS plan_id,
            @project_id AS project_id,
            @user_id AS user_id,
            @page_number AS page_number,
            @page_sections AS page_sections,
            @page_section_number AS page_section_number,
            @model_2d AS model_2d,
            @source AS source,
            @target_drywalls AS target_drywalls,
            @scale AS scale,
    ) s
    ON LOWER(t.project_id) = LOWER(s.project_id) AND LOWER(t.plan_id) = LOWER(s.plan_id) AND t.page_number = s.page_number AND t.page_section_number = s.page_section_number
    WHEN MATCHED THEN
    UPDATE SET
        model_2d = s.model_2d,
        scale = COALESCE(NULLIF(s.scale, ''), t.scale),
        user_id = @user_id,
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
        s.page_sections,
        s.page_section_number,
        s.scale,
        s.model_2d,
        JSON '{}',
        JSON '{}',
        s.source,
        s.target_drywalls,
        CURRENT_TIMESTAMP(),
        CURRENT_TIMESTAMP()
    );
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("page_sections", "INT64", page_sections),
            bigquery.ScalarQueryParameter("page_section_number", "STRING", page_section_number),
            bigquery.ScalarQueryParameter("scale", "STRING", scale),
            bigquery.ScalarQueryParameter("model_2d", "JSON", model_2d),
            bigquery.ScalarQueryParameter("source", "STRING", GCS_URL_floorplan_page),
            bigquery.ScalarQueryParameter("target_drywalls", "STRING", GCS_URL_target_drywalls_page)
        ]
    )

    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output

def is_duplicate(bigquery_client, credentials, pdf_path, project_id):
    sha_256 = sha256(pdf_path)
    GBQ_query = f"SELECT plan_id, sha256, status FROM `drywall_takeoff.plans` WHERE LOWER(project_id) = LOWER('{project_id}');"
    query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
    for plan_target in list(query_output):
        if plan_target.sha256 == sha_256:
            if plan_target.status == "FAILED":
                delete_plan(credentials, bigquery_client, plan_target.plan_id, project_id)
                return False
            return plan_target.plan_id
    return False

def delete_plan(credentials, bigquery_client, plan_id, project_id):
    GBQ_query = f"DELETE FROM `drywall_takeoff.plans` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
    query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
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

def load_templates(bigquery_client, credentials):
    GBQ_query = f"SELECT * FROM `{credentials["GBQServer"]["table_name_sku"]}`"
    product_templates = list(bigquery_run(credentials, bigquery_client, GBQ_query).result())

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
            ttl=datetime.timedelta(hours=5),
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
            logging.warning(f"SYSTEM: {e}: RETRYING ...")
        except Exception as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            exceptions.append(e)
            system_feedback = [Part.from_text(FEEDBACK_GENERATOR.format(max_retry=max_retry, exceptions=exceptions))]
            feedback_prompt = Content(role="model", parts=system_feedback)
            temperature = min(0.5 * (n_iterations + 1) / max_retry, 0.5)
            logging.warning(f"SYSTEM: Response Generation/Parsing failed with ERROR: {e}")
            logging.warning(f"SYSTEM: RETRYING with TEMPERATURE: {temperature}")

def map_floorplan_to_multipage_elevation(credentials, client_ip_address, pdf_path):
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
            user_id=user_id
        ),
    )
    response.raise_for_status()
    plan_types = response.json()
    return plan_types

def floorplan_to_pages(credentials, bigquery_client, project_id, plan_id, user_id, pdf_path, n_pages, batch_size=10):
    plan_types = plan_to_preview(credentials, project_id, plan_id, user_id)
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
    insert_pages_batch(
        pages_to_insert,
        bigquery_client,
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
                    page_number
                )
                futures.append(future)
        for future in futures:
            floor_plan_paths_preprocessed.append(future.result())
    for page_number, floor_plan_path_preprocessed in enumerate(floor_plan_paths_preprocessed):
        upload_floorplan(floor_plan_path_preprocessed, plan_id, project_id, credentials, index=str(page_number).zfill(4))
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

def insert_page(
    plan_id,
    user_id,
    project_id,
    page_number,
    extracted,
    status,
    bigquery_client,
    credentials,
    plan_type=dict(),
    GCS_URL_page=None,
    GCS_URL_page_thumbnail=None,
    mask_factor=dict(),
    bounding_box_offsets=dict(),
    is_floorplan=None,
):
    GBQ_query = """
    MERGE `drywall_takeoff.pages` t
    USING (
        SELECT
            @plan_id AS plan_id,
            @project_id AS project_id,
            @user_id AS user_id,
            @page_number AS page_number,
            @mask_factor AS mask_factor,
            @bounding_box_offsets AS bounding_box_offsets,
            @source AS source,
            @thumbnail AS thumbnail,
            @plan_type AS plan_type,
            @extracted AS extracted,
            @status AS status,
            @is_floorplan AS is_floorplan
    ) s
    ON LOWER(t.project_id) = LOWER(s.project_id) AND LOWER(t.plan_id) = LOWER(s.plan_id) AND t.page_number = s.page_number
    WHEN MATCHED THEN
    UPDATE SET
        extracted = s.extracted,
        updated_at = CURRENT_TIMESTAMP(),
        status = s.status
    WHEN NOT MATCHED THEN
    INSERT (
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
        s.plan_id,
        s.project_id,
        s.user_id,
        s.page_number,
        s.mask_factor,
        s.bounding_box_offsets,
        s.source,
        s.thumbnail,
        s.plan_type,
        s.extracted,
        s.status,
        s.is_floorplan,
        CURRENT_TIMESTAMP(),
        CURRENT_TIMESTAMP()
    );
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("mask_factor", "JSON", mask_factor),
            bigquery.ScalarQueryParameter("bounding_box_offsets", "JSON", bounding_box_offsets),
            bigquery.ScalarQueryParameter("source", "STRING", GCS_URL_page),
            bigquery.ScalarQueryParameter("thumbnail", "STRING", GCS_URL_page_thumbnail),
            bigquery.ScalarQueryParameter("plan_type", "JSON", plan_type),
            bigquery.ScalarQueryParameter("extracted", "BOOL", extracted),
            bigquery.ScalarQueryParameter("status", "STRING", status),
            bigquery.ScalarQueryParameter("is_floorplan", "BOOL", is_floorplan)
        ]
    )

    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output

def insert_pages_batch(
    pages,
    bigquery_client,
    credentials,
):
    if not pages:
        return None

    GBQ_query = """
    MERGE `drywall_takeoff.pages` t
    USING (
        SELECT *
        FROM UNNEST(@pages)
    ) s
    ON LOWER(t.project_id) = LOWER(s.project_id)
    AND LOWER(t.plan_id) = LOWER(s.plan_id)
    AND t.page_number = s.page_number

    WHEN MATCHED THEN
    UPDATE SET
        extracted = s.extracted,
        updated_at = CURRENT_TIMESTAMP(),
        status = s.status,
        plan_type = s.plan_type,
        source = s.source,
        thumbnail = s.thumbnail,
        mask_factor = s.mask_factor,
        bounding_box_offsets = s.bounding_box_offsets,
        is_floorplan = s.is_floorplan

    WHEN NOT MATCHED THEN
    INSERT (
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
        s.plan_id,
        s.project_id,
        s.user_id,
        s.page_number,
        s.mask_factor,
        s.bounding_box_offsets,
        s.source,
        s.thumbnail,
        s.plan_type,
        s.extracted,
        s.status,
        s.is_floorplan,
        CURRENT_TIMESTAMP(),
        CURRENT_TIMESTAMP()
    )
    """

    page_struct_type = bigquery.StructQueryParameterType(
        bigquery.ScalarQueryParameterType("STRING", name="plan_id"),
        bigquery.ScalarQueryParameterType("STRING", name="project_id"),
        bigquery.ScalarQueryParameterType("STRING", name="user_id"),
        bigquery.ScalarQueryParameterType("INT64", name="page_number"),
        bigquery.ScalarQueryParameterType("JSON", name="mask_factor"),
        bigquery.ScalarQueryParameterType("JSON", name="bounding_box_offsets"),
        bigquery.ScalarQueryParameterType("STRING", name="source"),
        bigquery.ScalarQueryParameterType("STRING", name="thumbnail"),
        bigquery.ScalarQueryParameterType("JSON", name="plan_type"),
        bigquery.ScalarQueryParameterType("BOOL", name="extracted"),
        bigquery.ScalarQueryParameterType("STRING", name="status"),
        bigquery.ScalarQueryParameterType("BOOL", name="is_floorplan"),
    )

    page_rows = list()
    for page in pages:
        page_rows.append(
            bigquery.StructQueryParameter(
                None,
                bigquery.ScalarQueryParameter(
                    "plan_id", "STRING", page["plan_id"]
                ),
                bigquery.ScalarQueryParameter(
                    "project_id", "STRING", page["project_id"]
                ),
                bigquery.ScalarQueryParameter(
                    "user_id", "STRING", page["user_id"]
                ),
                bigquery.ScalarQueryParameter(
                    "page_number", "INT64", page["page_number"]
                ),
                bigquery.ScalarQueryParameter(
                    "mask_factor", "JSON", page.get("mask_factor", dict())
                ),
                bigquery.ScalarQueryParameter(
                    "bounding_box_offsets",
                    "JSON",
                    page.get("bounding_box_offsets", dict())
                ),
                bigquery.ScalarQueryParameter(
                    "source", "STRING", page.get("source", '')
                ),
                bigquery.ScalarQueryParameter(
                    "thumbnail", "STRING", page.get("thumbnail", '')
                ),
                bigquery.ScalarQueryParameter(
                    "plan_type", "JSON", page.get("plan_type", dict())
                ),
                bigquery.ScalarQueryParameter(
                    "extracted", "BOOL", page["extracted"]
                ),
                bigquery.ScalarQueryParameter(
                    "status", "STRING", page["status"]
                ),
                bigquery.ScalarQueryParameter(
                    "is_floorplan", "BOOL", page["is_floorplan"]
                ),
            )
        )
    job_config = dict(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "pages",
                page_struct_type,
                page_rows
            )
        ]
    )

    return bigquery_run(
        credentials,
        bigquery_client,
        GBQ_query,
        job_config=job_config
    ).result()

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

def load_visual_grounding(
    credentials,
    project_id,
    plan_id,
    ip_address,
    pages_metadata,
    batch_size=10,
    vertex_ai_client=None,
    vertex_ai_generation_config=None,
    is_cached=None
):
    n_pages = len(pages_metadata)
    page_batches = [list(map(lambda page_metadata: page_metadata["page_number"], pages_metadata[batch_index * batch_size: batch_index * batch_size + batch_size])) for batch_index in range(n_pages // batch_size)]
    page_batches += [list(map(lambda page_metadata: page_metadata["page_number"], pages_metadata[n_pages - (n_pages % batch_size): n_pages]))]

    plan_paths = dict()
    for page_metadata in pages_metadata:
        index = str(page_metadata["page_number"]).zfill(4)
        destination_path = f"/tmp/floor_plan_{index}.png"
        blob_name = "floor_plan.png"
        download_floorplan(plan_id, project_id, credentials, index=index, blob_name=blob_name, destination_path=destination_path)
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
            for page_metadata in pages_metadata:
                if bounding_box["page_number"] == page_metadata["page_number"]:
                    page_metadata["mask_factor"] = bounding_box["mask_factor"]
                    page_metadata["bounding_box_offsets"] = bounding_box["bounding_box_offsets"]
    return pages_metadata

def download_floorplan(plan_id, project_id, credentials, index=None, blob_name="floor_plan.PDF", destination_path="/tmp/floor_plan.PDF"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if index:
        blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{blob_name}"
    else:
        blob_path = f"{project_id.lower()}/{plan_id.lower()}/{blob_name}"
    blob = bucket.blob(blob_path)

    blob.download_to_filename(destination_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"
