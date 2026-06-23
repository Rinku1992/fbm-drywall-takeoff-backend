import logging
from pathlib import Path
import json
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ReadTimeout, ChunkedEncodingError
from time import sleep
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import asyncio

import random
random.seed(0)

import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from google.cloud import secretmanager

from preprocessing import preprocess
from modeller_2d import FloorPlan2D
from helper import (
    enable_logging_on_stdout,
    load_gcp_credentials,
    load_hyperparameters,
    transcribe,
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
    insert_page,
    trigger_email_notification,
    load_metadata_from_vector_pdf,
    load_organization_slug,
)
from prompts import CEILING_CHOICES, WALL_CHOICES


def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json",
    )


async def floorplan_to_walls(credentials, pg_pool, project_id, plan_id, user_id, page_number, mask, output_path=None, max_retry=5):
    def load_headers_with_id_token():
        auth_req = google.auth.transport.requests.Request()
        service_account_credentials = IDTokenCredentials.from_service_account_file(
            credentials["service_drywall_account_key"],
            target_audience=credentials["CloudRun"]["APIs"]["wall_detector"]
        )
        service_account_credentials.refresh(auth_req)
        id_token = service_account_credentials.token

        headers = {
            "Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json"
        }
        return headers

    with open(output_path, "wb") as f:
        f.write(b'')
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max_retry+1,
        pool_maxsize=max_retry+1,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    for index in range(max_retry + 1):
        try:
            headers = load_headers_with_id_token()
            organization_slug = await load_organization_slug(credentials, pg_pool, user_id)
            response = session.post(
                f"{credentials["CloudRun"]["APIs"]["wall_detector"]}/detect_wall",
                headers=headers,
                json=dict(
                    project_id=project_id,
                    plan_id=plan_id,
                    user_id=user_id,
                    organization_slug=organization_slug,
                    page_number=page_number,
                    mask=mask,
                ),
                timeout=900
            )
            if response.status_code == 200:
                for _ in range(max_retry):
                    output_path = await download_segmented_walls(plan_id, project_id, user_id, str(page_number).zfill(4), credentials, pg_pool, destination_path=output_path)
                    if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                        break
                    sleep(20)
                break
        except (ConnectionError, ReadTimeout, ChunkedEncodingError) as e:
            if index < max_retry:
                logging.warning(f"SYSTEM: Wall Segmentation failed with error: {e}")
                logging.warning(f"SYSTEM: RETRYING({index + 1}) ...")
                sleep(min(60, 2 ** index))

    return Path(output_path)


async def page_to_structured_2d(
    credentials,
    pg_pool,
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
    predict_drywall,
    architectural_scale,
    standard_ceiling_height,
    allow_none_scale=False,
):
    floor_plan_modeller_2d.reload()
    wall_segmented_sectioned_path = load_section_from_page(
        wall_segmented_path,
        floor_plan_processed_path,
        bounding_box_offset,
        page_section_number
    )
    bounding_box_offset_marginalized = apply_pixel_margin_to_bounding_box(bounding_box_offset)
    if predict_drywall:
        walls_2d, polygons, _, external_contour = floor_plan_modeller_2d.model_and_predict(
            bounding_box_offset_marginalized,
            image_path=wall_segmented_sectioned_path,
            elevation_paths=elevation_processed_paths,
            model_2d_path=f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(page_number).zfill(4)}_{str(page_section_number).replace('/', '_')}.json",
            floor_plan_path=floor_plan_processed_path,
            transcription_block_with_centroids=transcription_block_with_centroids,
            architectural_scale=architectural_scale,
            standard_ceiling_height=standard_ceiling_height,
            allow_none_scale=allow_none_scale
        )
    else:
        walls_2d, polygons, _, external_contour = floor_plan_modeller_2d.model(
            bounding_box_offset_marginalized,
            image_path=wall_segmented_sectioned_path,
            elevation_paths=elevation_processed_paths,
            model_2d_path=f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(page_number).zfill(4)}_{str(page_section_number).replace('/', '_')}.json",
            floor_plan_path=floor_plan_processed_path,
            transcription_block_with_centroids=transcription_block_with_centroids,
            architectural_scale=architectural_scale,
            standard_ceiling_height=standard_ceiling_height,
            allow_none_scale=allow_none_scale
        )
    if walls_2d and polygons:
        floor_plan_modeller_2d.load_drywall_choices(walls_2d, polygons)
        floor_plan_modeller_2d.load_ceiling_choices(polygons)
        floor_plan_modeller_2d.load_wall_choices(walls_2d)
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
        ceiling_choices=CEILING_CHOICES
    )
    await insert_model_2d(
        dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata),
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
    )
    if floor_plan_modeller_2d.is_scale_detected:
        logging.info(f"SYSTEM: A 2D Model of the Floorplan from PAGE: {page_number} and SECTION: {page_section_number} Generated Successfully")
    else:
        logging.warning(f"SYSTEM: Architectural Scale not detected for PAGE: {page_number} and SECTION: {page_section_number}. Waiting for Architectural Scale input from the user")
    return floor_plan_modeller_2d.is_scale_detected


async def floorplan_to_page(credentials, pg_pool, project_id, plan_id, user_id, pdf_path, page_number, dpi):
    floor_plan_path_preprocessed = preprocess(pdf_path, page_number, dpi=dpi)
    await upload_floorplan(floor_plan_path_preprocessed, plan_id, project_id, user_id, credentials, pg_pool, index=str(page_number).zfill(4))
    return floor_plan_path_preprocessed


def load_elevation_pages(pdf_path, elevation_page_numbers):
    elevation_paths_preprocessed = list()
    for elevation_page_number in elevation_page_numbers:
        elevation_path_preprocessed = preprocess(
            pdf_path,
            elevation_page_number,
            image_path=f"/tmp/elevation_plan_{str(elevation_page_number).zfill(4)}.png"
        )
        elevation_paths_preprocessed.append(elevation_path_preprocessed)
    return elevation_paths_preprocessed


CREDENTIALS = load_gcp_credentials()
pg_pool = None
DRYWALL_TEMPLATES = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DRYWALL_TEMPLATES
    global pg_pool

    pg_pool = load_pg_pool(CREDENTIALS)

    DRYWALL_TEMPLATES = await load_templates(
        pg_pool,
        CREDENTIALS
    )

    yield

    if pg_pool:
        close_pg_pool()

app = FastAPI(title="Floorplan-to-Structured-2D (Cloud Run)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/floorplan_to_structured_2d")
async def floorplan_to_structured_2d(request: Request):
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
    mask_factor = parameters.get("mask_factor") or body.get("mask_factor")
    bounding_box_offsets = parameters.get("bounding_box_offsets") or body.get("bounding_box_offsets")
    elevation_pages = parameters.get("elevation_pages") or body.get("elevation_pages")
    predict_drywall = parameters.get("predict_drywall") or body.get("predict_drywall") or "true"
    architectural_scale = parameters.get("architectural_scale") or body.get("architectural_scale")
    page_number = int(page_number)
    predict_drywall = predict_drywall.upper() == "TRUE"
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")

    pdf_path = await download_floorplan(user_id, plan_id, project_id, CREDENTIALS, pg_pool)
    logging.info("SYSTEM: Floorplan Downloaded for extraction")

    hyperparameters = load_hyperparameters()
    publish_handler = load_publisher_client(CREDENTIALS)
    ip_address = request.headers.get("X-Client-IP", (request.client.host if request.client else None))
    try:
        floor_plan_processed_path = await floorplan_to_page(
            CREDENTIALS,
            pg_pool,
            project_id,
            plan_id,
            user_id,
            pdf_path,
            page_number,
            hyperparameters["modelling"]["scale_adoption"]["dpi"]
        )
        elevation_processed_paths = load_elevation_pages(pdf_path, elevation_pages)
    except Exception as e:
        future = publish_handler(dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
        future.result()
        logging.warning(f"SYSTEM: Floorplan extraction has failed for Page Number: {page_number} with Error: {e}")
        await insert_page(
            plan_id,
            user_id,
            project_id,
            page_number,
            True,
            "FAILED",
            pg_pool,
            CREDENTIALS,
        )
        await trigger_email_notification(
            CREDENTIALS,
            pg_pool,
            "FAILED",
            project_id,
            plan_id,
            user_id,
            page_number=page_number
        )
        return respond_with_UI_payload(dict(status="FAILED", message=f"NO Floor Plan layout observed"))
    logging.info(f"SYSTEM: Floorplan Preprocessing Completed: Page Number: {page_number}")

    floorplan_baseline_page_source = None
    svg_path=f"/tmp/{project_id}/{plan_id}/{user_id}/scaled_floor_plan_{str(page_number).zfill(4)}.svg"
    floorplan_baseline, floorplan_page_statistics = FloorPlan2D.scale_to(floor_plan_path=floor_plan_processed_path, svg_path=svg_path)
    floorplan_baseline_page_source = await upload_floorplan(floorplan_baseline, plan_id, project_id, user_id, CREDENTIALS, pg_pool, index=str(page_number).zfill(4))
    if not bounding_box_offsets:
        metadata = dict(
            size_in_bytes=floorplan_page_statistics["size"],
            height_in_pixels=floorplan_page_statistics["height_in_pixels"],
            width_in_pixels=floorplan_page_statistics["width_in_pixels"],
            height_in_points=floorplan_page_statistics["height_in_points"],
            width_in_points=floorplan_page_statistics["width_in_points"],
            origin=["LEFT", "TOP"],
            offset=(0, 0),
            contour_root_vertices=list(),
            scales_architectural=FloorPlan2D.scales_architectural,
            drywall_choices_color_codes=list(),
            wall_choices=WALL_CHOICES,
            ceiling_choices=CEILING_CHOICES
        )
        await insert_model_2d(
            dict(walls_2d=list(), polygons=list(), metadata=metadata),
            "0.25``:1`0``",
            page_number,
            0,
            "NA",
            plan_id,
            user_id,
            project_id,
            floorplan_baseline_page_source,
            pg_pool,
            CREDENTIALS,
        )
        future = publish_handler(dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
        future.result()
        logging.warning(f"SYSTEM: NO valid Floorplan layout observed: Page Number: {page_number}")
        await insert_page(
            plan_id,
            user_id,
            project_id,
            page_number,
            True,
            "COMPLETED",
            pg_pool,
            CREDENTIALS,
        )
        await trigger_email_notification(
            CREDENTIALS,
            pg_pool,
            "COMPLETED",
            project_id,
            plan_id,
            user_id,
            page_number=page_number
        )
        return respond_with_UI_payload(dict(status="SUCCESS", message="NO Floor Plan layout observed"))

    is_vector, scales, standard_ceiling_heights = await load_metadata_from_vector_pdf(
        CREDENTIALS,
        pg_pool,
        pdf_path,
        project_id,
        plan_id,
        page_number,
        bounding_box_offsets,
    )
    if not architectural_scale and is_vector:
        architectural_scale = scales

    futures = dict()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures["floorplan_to_walls"] = floorplan_to_walls(
            CREDENTIALS,
            pg_pool,
            project_id,
            plan_id,
            user_id,
            page_number,
            mask_factor,
            output_path=f"/tmp/{project_id}/{plan_id}/{user_id}/floor_plan_wall_segmented_{str(page_number).zfill(4)}.png"
        )
        futures["transcriber"] = executor.submit(
            transcribe,
            CREDENTIALS,
            hyperparameters,
            floor_plan_processed_path,
        )
    wall_segmented_path = await asyncio.gather(futures["floorplan_to_walls"])
    logging.info(f"SYSTEM: Wall Detection Completed from PAGE: {page_number}")

    transcription_block_with_centroids, _ = futures["transcriber"].result()
    logging.info(f"SYSTEM: Transcription Completed from PAGE: {page_number}")

    if FloorPlan2D.is_none(wall_segmented_path):
        for bounding_box_offset in bounding_box_offsets:
            metadata = dict(
                size_in_bytes=floorplan_page_statistics["size"],
                height_in_pixels=floorplan_page_statistics["height_in_pixels"],
                width_in_pixels=floorplan_page_statistics["width_in_pixels"],
                height_in_points=floorplan_page_statistics["height_in_points"],
                width_in_points=floorplan_page_statistics["width_in_points"],
                origin=["LEFT", "TOP"],
                offset=(0, 0),
                contour_root_vertices=list(),
                scales_architectural=FloorPlan2D.scales_architectural,
                drywall_choices_color_codes=list(),
                wall_choices=WALL_CHOICES,
                ceiling_choices=CEILING_CHOICES
            )
            await insert_model_2d(
                dict(walls_2d=list(), polygons=list(), metadata=metadata),
                "0.25``:1`0``",
                page_number,
                len(bounding_box_offsets),
                bounding_box_offset["title"],
                plan_id,
                user_id,
                project_id,
                floorplan_baseline_page_source,
                pg_pool,
                CREDENTIALS,
            )
        future = publish_handler(dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
        future.result()
        logging.warning(f"SYSTEM: Floorplan Segmentation FAILED: Page Number: {page_number}")
        await insert_page(
            plan_id,
            user_id,
            project_id,
            page_number,
            True,
            "COMPLETED",
            pg_pool,
            CREDENTIALS,
        )
        await trigger_email_notification(
            CREDENTIALS,
            pg_pool,
            "FAILED",
            project_id,
            plan_id,
            user_id,
            page_number=page_number
        )
        return respond_with_UI_payload(dict(status="SUCCESS", message="NO Floor Plan layout observed"))
    if not FloorPlan2D.is_none(wall_segmented_path):
        futures = list()
        vertex_ai_clients = FloorPlan2D.load_vertex_ai_clients(CREDENTIALS, ip_address, DRYWALL_TEMPLATES)
        scale_detected = True
        architectural_scales = architectural_scale
        architectural_scales = architectural_scales if isinstance(architectural_scales, list) else [architectural_scales for _ in bounding_box_offsets]
        for bounding_box_offset, architectural_scale, standard_ceiling_height in zip(bounding_box_offsets, architectural_scales, standard_ceiling_heights):
            logging.info(f"SYSTEM: Extracting structured model from SECTION: {bounding_box_offset["title"]} / OFFSET: {bounding_box_offset} in PAGE: {page_number}")
            floor_plan_modeller_2d = FloorPlan2D(CREDENTIALS, hyperparameters, DRYWALL_TEMPLATES)
            floor_plan_modeller_2d.from_vertex_ai_clients(*vertex_ai_clients)
            futures.append(
                page_to_structured_2d(
                    CREDENTIALS,
                    pg_pool,
                    floor_plan_modeller_2d,
                    project_id,
                    plan_id,
                    user_id,
                    page_number,
                    len(bounding_box_offsets),
                    bounding_box_offset["title"],
                    wall_segmented_path,
                    floor_plan_processed_path,
                    bounding_box_offset,
                    transcription_block_with_centroids,
                    floorplan_page_statistics,
                    floorplan_baseline_page_source,
                    elevation_processed_paths,
                    predict_drywall,
                    architectural_scale,
                    standard_ceiling_height,
                    allow_none_scale=hyperparameters["modelling"]["enable_early_stopping"] and is_vector,
                )
            )
        results = await asyncio.gather(*futures, return_exceptions=False)
        for is_scale_detected in results:
            scale_detected = (scale_detected and is_scale_detected)
        future = publish_handler(dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
        future.result()
    if scale_detected:
        await insert_page(
            plan_id,
            user_id,
            project_id,
            page_number,
            True,
            "COMPLETED",
            pg_pool,
            CREDENTIALS,
        )
        await trigger_email_notification(
            CREDENTIALS,
            pg_pool,
            "COMPLETED",
            project_id,
            plan_id,
            user_id,
            page_number=page_number
        )
    else:
        await insert_page(
            plan_id,
            user_id,
            project_id,
            page_number,
            True,
            "SCALE_NOT_DETECTED",
            pg_pool,
            CREDENTIALS,
        )
        await trigger_email_notification(
            CREDENTIALS,
            pg_pool,
            "SCALE NOT DETECTED",
            project_id,
            plan_id,
            user_id,
            page_number=page_number
        )
    return respond_with_UI_payload(dict(status="SUCCESS", message="Floor Plan extraction completed"))
