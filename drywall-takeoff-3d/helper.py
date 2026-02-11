import logging
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud.storage import Client as CloudStorageClient

from transcriber import Transcriber


def load_vertex_ai_client(credentials, region="us-central1"):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = GenerativeModel(credentials["VertexAI"]["llm"]["model_name"])
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config

def bigquery_run(credentials, GBQ_query, job_config=dict()):
    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    job_config = bigquery.QueryJobConfig(
        destination_encryption_configuration=bigquery.EncryptionConfiguration(
            kms_key_name=credentials["GBQServer"]["KMS_key"]
        ),
        **job_config
    )
    query_output = bigquery_client.query(GBQ_query, job_config=job_config)
    return query_output

def transcribe(credentials, hyperparameters, floor_plan_path):
    transcriber = Transcriber(credentials, hyperparameters)
    return transcriber.transcribe(floor_plan_path, [0, 1, -1, -2])

def sha256(path, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def upload_floorplan(plan_path, user_id, plan_id, project_id, credentials, index=None, directory=None):
    client = CloudStorageClient()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if directory:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{index}/{directory}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{directory}/{blob_object_name}"
    else:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{index}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{blob_object_name}"
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
    credentials
    ):
    if not model_2d.get("metadata", None):
        GBQ_query = f"SELECT model_2d FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
        query_output = bigquery_run(credentials, GBQ_query).result()
        walls_2d_raw = list(query_output)[0].model_2d
        walls_2d = json.loads(walls_2d_raw) if isinstance(walls_2d_raw, str) else walls_2d_raw
        model_2d["metadata"] = walls_2d["metadata"]
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
            @scale AS scale,
    ) s
    ON LOWER(t.project_id) = LOWER(s.project_id) AND LOWER(t.plan_id) = LOWER(s.plan_id) AND t.page_number = s.page_number
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
            bigquery.ScalarQueryParameter("scale", "STRING", scale),
            bigquery.ScalarQueryParameter("model_2d", "JSON", model_2d),
            bigquery.ScalarQueryParameter("source", "STRING", GCS_URL_floorplan_page),
            bigquery.ScalarQueryParameter("target_drywalls", "STRING", GCS_URL_target_drywalls_page)
        ]
    )

    query_output = bigquery_run(credentials, GBQ_query, job_config=job_config).result()
    return query_output

def extract_floorplan_from_page(
    credentials,
    hyperparameters,
    user_id,
    project_id,
    plan_id,
    floor_plan_preprocessed_bytes,
    page_number,
    floor_plan_modeller_2d,
    floorplan_to_walls_worker,
    transcribe_worker
    ):
    with open(f"/tmp/floor_plan_{str(page_number).zfill(2)}.png", "wb") as f:
        floor_plan_preprocessed_path = f.write(floor_plan_preprocessed_bytes)
    logging.info(f"SYSTEM: Processing Page: {page_number}")
    floorplan_page_source = upload_floorplan(floor_plan_preprocessed_path, user_id, plan_id, project_id, credentials, index=str(page_number).zfill(2))
    logging.info(f"SYSTEM: Preprocessed Floorplan Image uploaded to GCS from PAGE: {page_number}")
    futures = dict()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures["floorplan_to_walls"] = executor.submit(
            floorplan_to_walls_worker,
            credentials,
            project_id,
            plan_id,
            user_id,
            page_number,
            output_path=f"/tmp/floor_plan_wall_segmented_{str(page_number).zfill(2)}.png"
        )
        futures["transcriber"] = executor.submit(
            transcribe_worker,
            credentials,
            hyperparameters,
            floor_plan_preprocessed_path,
        )
        wall_segmented_path = futures["floorplan_to_walls"].result()
        upload_floorplan(wall_segmented_path, user_id, plan_id, project_id, credentials, index=str(page_number).zfill(2))
        logging.info(f"SYSTEM: Wall Detection Completed from PAGE: {page_number}")

        transcription_block_with_centroids, transcription_headers_and_footers = futures["transcriber"].result()
        logging.info(f"SYSTEM: Transcription Completed from PAGE: {page_number}")

    walls_2d, polygons, walls_2d_path, external_contour = floor_plan_modeller_2d.model(
        image_path=wall_segmented_path,
        model_2d_path=f"/tmp/walls_2d_{str(page_number).zfill(2)}.json",
        floor_plan_path=floor_plan_preprocessed_path,
        transcription_block_with_centroids=transcription_block_with_centroids,
        transcription_headers_and_footers=transcription_headers_and_footers
    )
    floor_plan_modeller_2d.load_drywall_choices(walls_2d, polygons)
    floor_plan_modeller_2d.load_ceiling_choices(polygons)
    if not polygons:
        return
    #if verbose.upper() == "TRUE":
    model_2d_path = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_preprocessed_path)
    upload_floorplan(model_2d_path, user_id, plan_id, project_id, credentials, index=str(page_number).zfill(2))
    #model_2d_path_overlay_enabled = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_path, overlay_enabled=True)
    #upload_floorplan(model_2d_path_overlay_enabled, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
    floorplan_baseline, floorplan_page_statistics = floor_plan_modeller_2d.scale_to(floor_plan_path=floor_plan_preprocessed_path)
    floorplan_baseline_page_source = upload_floorplan(floorplan_baseline, user_id, plan_id, project_id, credentials, index=str(page_number).zfill(2))
    logging.info(f"SYSTEM: A 2D Model of the Floorplan from PAGE: {page_number} Generated Successfully")
    metadata = dict(
        size_in_bytes=floorplan_page_statistics["size"],
        height_in_pixels=floorplan_page_statistics["height"],
        width_in_pixels=floorplan_page_statistics["width"],
        origin=["LEFT", "TOP"],
        offset=(0, 0),
        contour_root_vertices=external_contour
    )
    insert_model_2d(
        dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata),
        floor_plan_modeller_2d.scale,
        page_number,
        plan_id,
        user_id,
        project_id,
        floorplan_page_source,
        floorplan_baseline_page_source,
        credentials
    )
    page = dict(
        plan_id=plan_id,
        page_number=page_number,
        size_in_bytes=floorplan_page_statistics["size"],
        height_in_pixels=floorplan_page_statistics["height"],
        width_in_pixels=floorplan_page_statistics["width"],
        origin=["LEFT", "TOP"],
        offset=(0, 0),
        contour_root_vertices=external_contour,
        scale=floor_plan_modeller_2d.scale,
        walls_2d=walls_2d,
        polygons=polygons,
    )
    return page
