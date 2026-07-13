import logging
from pathlib import Path
import json
import requests
from functools import partial
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ReadTimeout, ChunkedEncodingError
from time import sleep
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor
import asyncio

import random
random.seed(0)

import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from google.cloud import secretmanager

from preprocessing import preprocess
from floor_plan import FloorPlan
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
    load_section_from_page,
    apply_pixel_margin_to_bounding_box,
    load_publisher_client,
    insert_page,
    trigger_email_notification,
    load_metadata_from_vector_pdf,
    load_organization_slug,
    update_status,
    pg_run,
    load_subscriber_client,
    query_subscriber_messages,
    is_session_active,
    terminate_session,
)
from layout_parameters import CEILING_CHOICES, WALL_CHOICES


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
            credentials["service_compute_account_key"],
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


def floorplan_section_to_structured_2d(credentials, query_json):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["floorplan_section_to_structured_2d"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token

    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{credentials["CloudRun"]["APIs"]["floorplan_section_to_structured_2d"]}/floorplan_section_to_structured_2d",
        headers=headers,
        json=query_json
    )
    return response.status_code, response.content


def section_to_structured_2d(
    floor_plan_modeller_2d,
    project_id,
    plan_id,
    user_id,
    page_number,
    page_section_number,
    wall_segmented_path,
    floor_plan_processed_path,
    bounding_box_offset,
    transcription_block_with_centroids,
    floorplan_page_statistics,
    elevation_processed_paths,
    predict_drywall,
    architectural_scale,
    standard_ceiling_height,
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
            allow_none_scale=allow_none_scale,
            trust_scale=trust_scale,
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
            allow_none_scale=allow_none_scale,
            trust_scale=trust_scale,
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
    if floor_plan_modeller_2d.is_scale_detected:
        logging.info(f"SYSTEM: A 2D Model of the Floorplan from PAGE: {page_number} and SECTION: {page_section_number} Generated Successfully")
    else:
        logging.warning(f"SYSTEM: Architectural Scale not detected for PAGE: {page_number} and SECTION: {page_section_number}. Waiting for Architectural Scale input from the user")
    return floor_plan_modeller_2d.is_scale_detected, dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata), floor_plan_modeller_2d.normalize_scale(floor_plan_modeller_2d.scale), page_section_number


async def floorplan_to_page(credentials, pg_pool, project_id, plan_id, user_id, pdf_path, page_number, dpi):
    floor_plan_path_preprocessed = preprocess(pdf_path, page_number, dpi=dpi)
    await upload_floorplan(floor_plan_path_preprocessed, plan_id, project_id, user_id, credentials, pg_pool, index=str(page_number).zfill(4))
    return floor_plan_path_preprocessed


CREDENTIALS = load_gcp_credentials()
pg_pool = dict()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pg_pool

    for attempt in range(10):
        try:
            engine = load_pg_pool(CREDENTIALS)
            pg_pool["engine"] = engine

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
    session_uuid = parameters.get("session_uuid") or body.get("session_uuid")
    page_number = int(page_number)
    predict_drywall = predict_drywall.upper() == "TRUE"
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")

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
    subscriber_client = load_subscriber_client(CREDENTIALS)
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
    except Exception as e:
        future = publish_handler(dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
        future.result()
        logging.warning(f"SYSTEM: Floorplan extraction has failed for Page Number: {page_number} with Error: {e}")
        session_is_active =  await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
        if not session_is_active:
            return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
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
        await terminate_session(CREDENTIALS, pg_pool, session_uuid)
        return respond_with_UI_payload(dict(status="FAILED", message=f"NO Floor Plan layout observed"))
    logging.info(f"SYSTEM: Floorplan Preprocessing Completed: Page Number: {page_number}")

    floorplan_baseline_page_source = None
    svg_path=f"/tmp/{project_id}/{plan_id}/{user_id}/scaled_floor_plan_{str(page_number).zfill(4)}.svg"
    Path(svg_path).parent.mkdir(parents=True, exist_ok=True)
    floorplan_baseline, floorplan_page_statistics = FloorPlan.scale_to(floor_plan_path=floor_plan_processed_path, svg_path=svg_path)
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
            scales_architectural=FloorPlan.scales_architectural,
            drywall_choices_color_codes=list(),
            wall_choices=WALL_CHOICES,
            ceiling_choices=CEILING_CHOICES
        )
        session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
        if not session_is_active:
            return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
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
        session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
        if not session_is_active:
            return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
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
        await terminate_session(CREDENTIALS, pg_pool, session_uuid)
        return respond_with_UI_payload(dict(status="SUCCESS", message="NO Floor Plan layout observed"))

    session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
    if not session_is_active:
        return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
    await update_status(CREDENTIALS, pg_pool, "READING PDF TEXT", project_id, plan_id, user_id, page_number)
    is_vector, scales, standard_ceiling_heights = await load_metadata_from_vector_pdf(
        CREDENTIALS,
        pg_pool,
        pdf_path,
        project_id,
        plan_id,
        page_number,
        bounding_box_offsets,
    )
    if is_vector:
        session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
        if not session_is_active:
            return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
        await update_status(CREDENTIALS, pg_pool, "SCALE DETECTED", project_id, plan_id, user_id, page_number)
        if architectural_scale:
            architectural_scales_updated = list()
            for scale in scales:
                if scale:
                    architectural_scales_updated.append(scale)
                else:
                    architectural_scales_updated.append(architectural_scale)
            architectural_scale = architectural_scales_updated
        else:
            architectural_scale = scales

    futures = dict()
    session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
    if not session_is_active:
        return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
    await update_status(CREDENTIALS, pg_pool, "DETECTING WALLS AND TRANSCRIPTIONS", project_id, plan_id, user_id, page_number)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures["floorplan_to_walls"] = floorplan_to_walls(
            CREDENTIALS,
            pg_pool,
            project_id,
            plan_id,
            user_id_owner,
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
    wall_segmented_path, = await asyncio.gather(futures["floorplan_to_walls"])
    session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
    if not session_is_active:
        return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
    await update_status(CREDENTIALS, pg_pool, "WALLS DETECTED", project_id, plan_id, user_id, page_number)
    logging.info(f"SYSTEM: Wall Detection Completed from PAGE: {page_number}")

    transcription_block_with_centroids, _ = futures["transcriber"].result()
    session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
    if not session_is_active:
        return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
    await update_status(CREDENTIALS, pg_pool, "TRANSCRIPTIONS DETECTED", project_id, plan_id, user_id, page_number)
    logging.info(f"SYSTEM: Transcription Completed from PAGE: {page_number}")

    if FloorPlan.is_none(wall_segmented_path):
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
                scales_architectural=FloorPlan.scales_architectural,
                drywall_choices_color_codes=list(),
                wall_choices=WALL_CHOICES,
                ceiling_choices=CEILING_CHOICES
            )
            session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
            if not session_is_active:
                return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
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
        session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
        if not session_is_active:
            return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
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
        await terminate_session(CREDENTIALS, pg_pool, session_uuid)
        return respond_with_UI_payload(dict(status="SUCCESS", message="NO Floor Plan layout observed"))
    if not FloorPlan.is_none(wall_segmented_path):
        architectural_scales = (
            architectural_scale
            if isinstance(architectural_scale, list)
            else [architectural_scale] * len(bounding_box_offsets)
        )
        executor = ThreadPoolExecutor(max_workers=2)
        query_payloads = list()
        for bounding_box_offset, architectural_scale, standard_ceiling_height in zip(bounding_box_offsets, architectural_scales, standard_ceiling_heights):
            query_json = dict(
                project_id=project_id,
                user_id=user_id,
                plan_id=plan_id,
                page_number=page_number,
                transcription_block_with_centroids=transcription_block_with_centroids,
                is_vector=is_vector,
                vector_standard_ceiling_height=standard_ceiling_height,
                bounding_box_offset=bounding_box_offset,
                number_of_sections=len(bounding_box_offsets),
                elevation_pages=elevation_pages,
                predict_drywall=predict_drywall,
                architectural_scale=architectural_scale,
                session_uuid=session_uuid,
            )
            session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number)
            if not session_is_active:
                return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
            executor.submit(
                floorplan_section_to_structured_2d,
                CREDENTIALS,
                query_json,
            )
            query_payloads.append(
                dict(
                    project_id=project_id,
                    plan_id=plan_id,
                    page_number=page_number,
                    page_section_number=bounding_box_offset["title"],
                    session_uuid=session_uuid
                )
            )
        all_sections_extracted = False
        sleep_time = 1
        while not all_sections_extracted:
            session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number, completed_as_active=True)
            if not session_is_active:
                return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
            notifications_arrived_all, acknowledged_queries = query_subscriber_messages(CREDENTIALS, subscriber_client, query_payloads)
            for acknowledged_query in acknowledged_queries:
                if not acknowledged_query["is_scale_detected"]:
                    future = publish_handler(dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
                    future.result()
                    session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number, completed_as_active=True)
                    if not session_is_active:
                        return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
                    await insert_page(
                        plan_id,
                        user_id,
                        project_id,
                        page_number,
                        True,
                        "SCALE NOT DETECTED",
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
                    await terminate_session(CREDENTIALS, pg_pool, session_uuid)
                    return respond_with_UI_payload(dict(status="SUCCESS", message="Floor Plan extraction completed"))

                if acknowledged_query["is_scale_detected"] == "NA":
                    future = publish_handler(dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
                    future.result()
                    logging.warning(f"SYSTEM: Floorplan extraction has failed for Page Number: {page_number} with Error: {e}")
                    session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number, completed_as_active=True)
                    if not session_is_active:
                        return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
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
                    await terminate_session(CREDENTIALS, pg_pool, session_uuid)
                    return respond_with_UI_payload(dict(status="FAILED", message=f"NO Floor Plan layout observed"))

                for query_payload in query_payloads:
                    if query_payload["page_section_number"] == acknowledged_query["page_section_number"]:
                        query_payloads.remove(query_payload)
            if notifications_arrived_all:
                all_sections_extracted = True
                break
            sleep(sleep_time)
        future = publish_handler(dict(project_id=project_id, plan_id=plan_id, page_number=page_number))
        future.result()
        session_is_active = await is_session_active(CREDENTIALS, pg_pool, session_uuid, project_id, plan_id, user_id, page_number, completed_as_active=True)
        if not session_is_active:
            return respond_with_UI_payload(dict(status="ABORTED", message=f"Session Aborted"))
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
    await terminate_session(CREDENTIALS, pg_pool, session_uuid)
    return respond_with_UI_payload(dict(status="SUCCESS", message="Floor Plan extraction completed"))
