import logging
from pathlib import Path
import json
from functools import partial
from time import sleep
from geopy.geocoders import Nominatim
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import asyncio

import random
random.seed(0)

from google.cloud import secretmanager

from preprocessing import preprocess
from modeller_2d import FloorPlan2D
from helper import (
    enable_logging_on_stdout,
    load_gcp_credentials,
    load_hyperparameters,
    upload_floorplan,
    download_floorplan,
    download_segmented_walls,
    insert_model_2d,
    load_pg_pool,
    close_pg_pool,
    load_templates,
    load_section_from_page,
    apply_pixel_margin_to_bounding_box,
    load_publisher_client,
    update_status,
    pg_run,
    is_session_active,
)
from prompts import CEILING_CHOICES, WALL_CHOICES


def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json",
    )


async def load_segmented_walls(credentials, pg_pool, project_id, plan_id, user_id, page_number, output_path=None, max_retry=5):
    with open(output_path, "wb") as f:
        f.write(b'')
    for _ in range(max_retry):
        output_path = await download_segmented_walls(plan_id, project_id, user_id, str(page_number).zfill(4), credentials, pg_pool, destination_path=output_path)
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            break
        sleep(20)

    return Path(output_path)


async def floorplan_to_structured_2d_sectioned(
    credentials,
    pg_pool,
    session_uuid,
    floor_plan_modeller_2d,
    project_id,
    plan_id,
    user_id,
    page_number,
    page_sections,
    page_section_number,
    wall_segmented_path,
    floor_plan_processed_path,
    bounding_box_offset,
    transcription_block_with_centroids,
    floorplan_page_statistics,
    floorplan_baseline_page_source,
    elevation_processed_paths,
    model,
    predict,
    architectural_scale,
    standard_ceiling_height,
    DPI_in_use,
    allow_none_scale=False,
    trust_scale=True,
):
    floor_plan_modeller_2d.reload(page_section_number)
    wall_segmented_sectioned_path = load_section_from_page(
        wall_segmented_path,
        floor_plan_processed_path,
        bounding_box_offset,
        page_section_number
    )
    bounding_box_offset_marginalized = apply_pixel_margin_to_bounding_box(bounding_box_offset)
    if model and predict:
        walls_2d, polygons, _, external_contour = floor_plan_modeller_2d.model_and_predict(
            bounding_box_offset_marginalized,
            image_path=wall_segmented_sectioned_path,
            elevation_paths=elevation_processed_paths,
            model_2d_path=f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(page_number).zfill(4)}_{str(page_section_number).replace('/', '_')}.json",
            floor_plan_path=floor_plan_processed_path,
            transcription_block_with_centroids=transcription_block_with_centroids,
            architectural_scale=architectural_scale,
            standard_ceiling_height=standard_ceiling_height,
            allow_none_scale=allow_none_scale,
            trust_scale=trust_scale,
        )
        walls_2d_layout, polygons_layout = floor_plan_modeller_2d.model_to_sketch(walls_2d, polygons)
        if walls_2d and polygons:
            floor_plan_modeller_2d.load_drywall_choices(walls_2d, polygons)
            floor_plan_modeller_2d.load_ceiling_choices(polygons)
            floor_plan_modeller_2d.load_wall_choices(walls_2d)
            floor_plan_modeller_2d.load_drywall_choices(walls_2d_layout, polygons_layout)
            floor_plan_modeller_2d.load_ceiling_choices(polygons_layout)
            floor_plan_modeller_2d.load_wall_choices(walls_2d_layout)
    elif model and not predict:
        walls_2d_layout, polygons_layout, _, external_contour = floor_plan_modeller_2d.model(
            bounding_box_offset_marginalized,
            image_path=wall_segmented_sectioned_path,
            elevation_paths=elevation_processed_paths,
            model_2d_path=f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(page_number).zfill(4)}_{str(page_section_number).replace('/', '_')}.json",
            floor_plan_path=floor_plan_processed_path,
            transcription_block_with_centroids=transcription_block_with_centroids,
            architectural_scale=architectural_scale,
            standard_ceiling_height=standard_ceiling_height,
            allow_none_scale=allow_none_scale,
            trust_scale=trust_scale,
        )
        if walls_2d_layout and polygons_layout:
            floor_plan_modeller_2d.load_drywall_choices(walls_2d_layout, polygons_layout)
            floor_plan_modeller_2d.load_ceiling_choices(polygons_layout)
            floor_plan_modeller_2d.load_wall_choices(walls_2d_layout)
        #model_2d_path = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_processed_path)
        #model_2d_path_sectioned = model_2d_path.parent.joinpath(f"{model_2d_path.stem}_sectioned_{page_section_number}").with_suffix(".png")
        #model_2d_path.rename(model_2d_path_sectioned)
        #await upload_floorplan(model_2d_path_sectioned, plan_id, project_id, user_id, CREDENTIALS, pg_pool, index=str(page_number).zfill(4))
        #model_2d_path_overlay_enabled = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_processed_path, overlay_enabled=True)
        #await upload_floorplan(model_2d_path_overlay_enabled, plan_id, project_id, user_id, CREDENTIALS, pg_pool, index=str(page_number).zfill(4))

    metadata = dict(
        size_in_bytes=floorplan_page_statistics["size"],
        height_in_pixels=floorplan_page_statistics["height_in_pixels"],
        width_in_pixels=floorplan_page_statistics["width_in_pixels"],
        height_in_points=floorplan_page_statistics["height_in_points"],
        width_in_points=floorplan_page_statistics["width_in_points"],
        origin=["LEFT", "TOP"],
        offset=(0, 0),
        contour_root_vertices=external_contour,
        scales_architectural=floor_plan_modeller_2d.scales_architectural,
        drywall_choices_color_codes=floor_plan_modeller_2d.drywall_choices_color_codes,
        wall_choices=WALL_CHOICES,
        ceiling_choices=CEILING_CHOICES,
        DPI=DPI_in_use,
    )
    session_is_active = await is_session_active(credentials, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
    if session_is_active:
        if model and predict:
            await insert_model_2d(
                floor_plan_modeller_2d.normalize_scale(floor_plan_modeller_2d.scale),
                page_number,
                page_sections,
                page_section_number,
                plan_id,
                user_id,
                project_id,
                floorplan_baseline_page_source,
                pg_pool,
                credentials,
                model_2d=dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata),
                layout_2d=dict(walls_2d=walls_2d_layout, polygons=polygons_layout, metadata=metadata),
                insert_model=model,
                insert_layout=predict,
            )
        elif model and not predict:
            await insert_model_2d(
                floor_plan_modeller_2d.normalize_scale(floor_plan_modeller_2d.scale),
                page_number,
                page_sections,
                page_section_number,
                plan_id,
                user_id,
                project_id,
                floorplan_baseline_page_source,
                pg_pool,
                credentials,
                model_2d=dict(walls_2d=list(), polygons=list(), metadata=metadata),
                layout_2d=dict(walls_2d=walls_2d_layout, polygons=polygons_layout, metadata=metadata),
                insert_model=model,
                insert_layout=predict,
            )
    if floor_plan_modeller_2d.is_scale_detected:
        logging.info(f"SYSTEM: A 2D Model of the Floorplan from PAGE: {page_number} and SECTION: {page_section_number} Generated Successfully")
    else:
        logging.warning(f"SYSTEM: Architectural Scale not detected for PAGE: {page_number} and SECTION: {page_section_number}. Waiting for Architectural Scale input from the user")
    return floor_plan_modeller_2d.is_scale_detected, dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata), floor_plan_modeller_2d.normalize_scale(floor_plan_modeller_2d.scale)


async def floorplan_to_page(credentials, pg_pool, project_id, plan_id, user_id, pdf_path, page_number, maximum_dpi, minimum_dpi):
    floor_plan_path_preprocessed, dpi_in_use = preprocess(pdf_path, page_number, maximum_dpi=maximum_dpi, minimum_dpi=minimum_dpi)
    await upload_floorplan(floor_plan_path_preprocessed, plan_id, project_id, user_id, credentials, pg_pool, index=str(page_number).zfill(4))
    return floor_plan_path_preprocessed, dpi_in_use


def load_elevation_pages(pdf_path, elevation_page_numbers):
    elevation_paths_preprocessed = list()
    for elevation_page_number in elevation_page_numbers:
        elevation_path_preprocessed, _ = preprocess(
            pdf_path,
            elevation_page_number,
            image_path=f"/tmp/elevation_plan_{str(elevation_page_number).zfill(4)}.png"
        )
        elevation_paths_preprocessed.append(elevation_path_preprocessed)
    return elevation_paths_preprocessed


CREDENTIALS = load_gcp_credentials()
pg_pool = dict()
DRYWALL_TEMPLATES = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DRYWALL_TEMPLATES
    global pg_pool

    for attempt in range(10):
        try:
            engine = load_pg_pool(CREDENTIALS)
            pg_pool["engine"] = engine
            DRYWALL_TEMPLATES = await load_templates(
                pg_pool,
                CREDENTIALS
            )

            break
        except Exception as e:
            logging.exception(e)

            if attempt == 9:
                raise

            await asyncio.sleep(min(2 ** attempt, 30))

    yield

    if pg_pool and pg_pool["engine"]:
        close_pg_pool()

app = FastAPI(title="Floorplan-to-Structured-2D (Cloud Run)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/floorplan_section_to_structured_2d")
async def floorplan_section_to_structured_2d(request: Request):
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
    page_number = parameters.get("page_number") or body.get("page_number")
    transcription_block_with_centroids = parameters.get("transcription_block_with_centroids") or body.get("transcription_block_with_centroids")
    is_vector = parameters.get("is_vector") or body.get("is_vector")
    vector_standard_ceiling_height = parameters.get("vector_standard_ceiling_height") or body.get("vector_standard_ceiling_height")
    bounding_box_offset = parameters.get("bounding_box_offset") or body.get("bounding_box_offset")
    number_of_sections = parameters.get("number_of_sections") or body.get("number_of_sections")
    elevation_pages = parameters.get("elevation_pages") or body.get("elevation_pages")
    model = get_param("model")
    predict = get_param("predict")
    architectural_scale = parameters.get("architectural_scale") or body.get("architectural_scale")
    session_uuid = parameters.get("session_uuid") or body.get("session_uuid")
    page_number = int(page_number)
    logging.info("SYSTEM: Received a Floorplan 2D Sectioned Model Generation Request")

    query = f"SELECT user_id FROM {CREDENTIALS["CloudSQL"]["table_name_plans"]} WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s)"
    user_id_owner = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id, plan_id,), fetch=True))
    if user_id_owner:
        user_id_owner = user_id_owner[0]["user_id"]
    else:
        user_id_owner = user_id
    pdf_path = await download_floorplan(user_id_owner, plan_id, project_id, CREDENTIALS, pg_pool)
    logging.info("SYSTEM: Floorplan Downloaded for extraction")

    hyperparameters = load_hyperparameters()
    publish_handler = load_publisher_client(CREDENTIALS)
    ip_address = request.headers.get("X-Client-IP", (request.client.host if request.client else None))
    try:
        floor_plan_processed_path, dpi_in_use = await floorplan_to_page(
            CREDENTIALS,
            pg_pool,
            project_id,
            plan_id,
            user_id,
            pdf_path,
            page_number,
            hyperparameters["modelling"]["scale_adoption"]["dpi"]["maximum"],
            hyperparameters["modelling"]["scale_adoption"]["dpi"]["minimum"]
        )
        hyperparameters["modelling"]["scale_adoption"]["dpi"]["in_use"] = dpi_in_use
        elevation_processed_paths = load_elevation_pages(pdf_path, elevation_pages)
    except Exception as e:
        future = publish_handler(
            dict(
                session_uuid=session_uuid,
                project_id=project_id,
                plan_id=plan_id,
                page_number=page_number,
                page_section_number=bounding_box_offset["title"],
                is_scale_detected="NA"
            )
        )
        future.result()
        logging.warning(f"SYSTEM: Floorplan extraction has failed for Page Number: {page_number} with Error: {e}")
        return respond_with_UI_payload(dict(status="FAILED", message=f"NO Floor Plan layout observed"))

    floorplan_baseline_page_source = None
    svg_path=f"/tmp/{project_id}/{plan_id}/{user_id}/scaled_floor_plan_{str(page_number).zfill(4)}.svg"
    Path(svg_path).parent.mkdir(parents=True, exist_ok=True)
    floorplan_baseline, floorplan_page_statistics = FloorPlan2D.scale_to(floor_plan_path=floor_plan_processed_path, svg_path=svg_path)
    floorplan_baseline_page_source = await upload_floorplan(floorplan_baseline, plan_id, project_id, user_id, CREDENTIALS, pg_pool, index=str(page_number).zfill(4))
    wall_segmented_path = await load_segmented_walls(
        CREDENTIALS,
        pg_pool,
        project_id,
        plan_id,
        user_id_owner,
        page_number,
        output_path=f"/tmp/{project_id}/{plan_id}/{user_id}/floor_plan_wall_segmented_{str(page_number).zfill(4)}.png"
    )
    query = f"SELECT project_location, project_location_pincode FROM {CREDENTIALS["CloudSQL"]["table_name_projects"]} WHERE LOWER(project_id) = LOWER(%s)"
    query_output = await run_in_threadpool(partial(pg_run, CREDENTIALS, pg_pool, query, params=(project_id,), fetch=True))
    project_location, project_location_pincode = query_output[0]["project_location"], query_output[0]["project_location_pincode"]
    geolocator = Nominatim(user_agent="xtimator_app")
    project_address = f"{project_location_pincode}, {project_location}"
    if geolocator.geocode(f"{project_location_pincode}, {project_location}", language="en"):
        project_state, project_country = geolocator.geocode(f"{project_location_pincode}, {project_location}", language="en").address.rsplit(',', 2)[1:]
        project_address = f"{project_state} {project_location_pincode}, {project_country}"
    vertex_ai_clients = FloorPlan2D.load_vertex_ai_clients(CREDENTIALS, ip_address, DRYWALL_TEMPLATES, project_address)
    logging.info(f"SYSTEM: Extracting structured model from SECTION: {bounding_box_offset["title"]} / OFFSET: {bounding_box_offset} in PAGE: {page_number}")
    await update_status(CREDENTIALS, pg_pool, f"DETECTING GEOMETRY IN SECTION: `{bounding_box_offset["title"]}`", project_id, plan_id, user_id, page_number)
    floor_plan_modeller_2d = FloorPlan2D(CREDENTIALS, hyperparameters, DRYWALL_TEMPLATES, project_address)
    floor_plan_modeller_2d.from_vertex_ai_clients(*vertex_ai_clients)
    is_scale_detected, _, _ = await floorplan_to_structured_2d_sectioned(
        CREDENTIALS,
        pg_pool,
        session_uuid,
        floor_plan_modeller_2d,
        project_id,
        plan_id,
        user_id,
        page_number,
        number_of_sections,
        bounding_box_offset["title"],
        wall_segmented_path,
        floor_plan_processed_path,
        bounding_box_offset,
        transcription_block_with_centroids,
        floorplan_page_statistics,
        floorplan_baseline_page_source,
        elevation_processed_paths,
        model,
        predict,
        architectural_scale,
        vector_standard_ceiling_height,
        dpi_in_use,
        allow_none_scale=hyperparameters["modelling"]["enable_early_stopping"] and is_vector,
        trust_scale=is_vector,
    )
    future = publish_handler(
        dict(
            session_uuid=session_uuid,
            project_id=project_id,
            plan_id=plan_id,
            page_number=page_number,
            page_section_number=bounding_box_offset["title"],
            is_scale_detected=is_scale_detected,
        )
    )
    future.result()
