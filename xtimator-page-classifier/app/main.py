"""
Drywall Page Classifier — Cloud Run service entry point.

Replaces the classification responsibility of the Vertex AI Gemini call in
drywall-takeoff-3d/helper.py:classify_plan. The LLM still produces
mask_factor and bounding_box_offsets (via a modified prompt — see the
drywall-takeoff-3d patch), and our service produces plan_type. Both run
in parallel from the caller's perspective.

Endpoint:
    POST /classify_pages
    Content-Type: multipart/form-data

    Form fields:
        metadata:  JSON string -- {"page_numbers": [0, 1, 2, ...]}
        images:    one or more PNG files (binary)

    The number of images MUST equal len(page_numbers), and they MUST be
    sent in the same order as the page_numbers entries.

    Returns:
        JSON matching ClassifyResponse — list of PagePrediction, one per
        input image, in the same order.
"""


import asyncio
import functools
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pdf2image.pdf2image import pdfinfo_from_path

from .schemas import ClassifyResponse, ClassifyPagesRequest, RequestMetadata  # noqa: F401
from .inference import PageClassifier
from .config import (
    MODEL_CHECKPOINT_PATH,
    MAX_IMAGES_PER_REQUEST,  # noqa: F401
    PREPROCESSING_OUTPUT_DIR,
    PREPROCESSING_MAX_WORKERS,
)
from .gcs_client import download_floorplan_pdf
from .preprocessing import preprocess


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("classifier")

# Loaded once at startup
classifier: PageClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model + warmup. Shutdown: nothing (Cloud Run kills us)."""
    global classifier
    logger.info("=" * 60)
    logger.info("Starting drywall-page-classifier service")
    logger.info("=" * 60)

    t0 = time.time()
    logger.info(f"Loading model from {MODEL_CHECKPOINT_PATH}...")
    classifier = PageClassifier(MODEL_CHECKPOINT_PATH)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    classifier.warmup()
    logger.info(f"Warmup forward pass completed in {time.time() - t0:.1f}s")

    logger.info("Service ready to accept requests")
    yield
    logger.info("Service shutting down")


app = FastAPI(
    title="Drywall Page Classifier",
    version="1.0.0",
    description="ConvNeXt-Small architectural page classifier "
                "(in-house replacement for Gemini classification).",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"service": "drywall-page-classifier", "version": "1.0.0"}


@app.get("/healthz")
def healthz():
    """
    Cloud Run startup + liveness probe.
    Returns 200 only when the model is loaded.
    """
    if classifier is None or not classifier.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet",
        )
    return {"status": "ready"}


@app.post("/classify_pages", response_model=ClassifyResponse)
async def classify_pages(request: ClassifyPagesRequest):
    """
    Classify every page of a floor-plan PDF stored in GCS.

    The PDF is fetched from:
        gs://drywall-takeoff-artifacts-dev/{project_id_lower}/{plan_id_lower}/floor_plan.PDF

    All pages are rendered (DPI 150, matching training distribution) in
    parallel and classified. Response shape is identical to the previous
    multipart endpoint.

    Log phases (grep-friendly prefixes):
        [REQUEST]     — request lifecycle
        [GCS]         — PDF download
        [PNG]         — PDF→PNG conversion (per-page + phase markers)
        [CLASSIFIER]  — model inference (per-page + phase markers)
        [CLEANUP]     — /tmp cleanup
    """
    if classifier is None or not classifier.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    request_id = f"project={request.project_id} plan={request.plan_id} user={request.user_id}"
    logger.info(f"[REQUEST] [{request_id}] /classify_pages received")
    t_start = time.time()

    # All blocking work (GCS download, PDF parsing, page rendering, file reads)
    # is offloaded to this loop's default thread pool via run_in_executor so the
    # event loop stays free to answer Cloud Run's /healthz liveness probe. Without
    # this, a long render (e.g. a 300-page PDF) blocks the loop past the probe's
    # 3-strike deadline and Cloud Run kills the container mid-request.
    loop = asyncio.get_running_loop()

    # ─── Download PDF from GCS ────────────────────────────────
    pdf_path = Path(f"/tmp/{request.plan_id}/floor_plan.PDF")
    try:
        logger.info(f"[GCS] [{request_id}] download started")
        t0 = time.time()
        await loop.run_in_executor(
            None,
            functools.partial(
                download_floorplan_pdf,
                project_id=request.project_id,
                plan_id=request.plan_id,
                organization_slug=request.organization_slug,
                destination_path=pdf_path,
            ),
        )
        gcs_elapsed = time.time() - t0
        pdf_size_mb = pdf_path.stat().st_size / 1e6
        logger.info(
            f"[GCS] [{request_id}] download finished — {pdf_size_mb:.2f} MB "
            f"in {gcs_elapsed:.2f}s"
        )
    except FileNotFoundError as e:
        logger.warning(f"[GCS] [{request_id}] PDF not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"[GCS] [{request_id}] download failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GCS download failed: {e}",
        )

    # ─── Count pages ──────────────────────────────────────────
    try:
        pdf_info = await loop.run_in_executor(None, pdfinfo_from_path, str(pdf_path))
        n_pages = pdf_info["Pages"]
        logger.info(f"[REQUEST] [{request_id}] PDF has {n_pages} pages")
    except Exception as e:
        logger.exception(f"[REQUEST] [{request_id}] failed to read PDF metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read PDF metadata: {e}",
        )

    if n_pages == 0:
        logger.warning(f"[REQUEST] [{request_id}] PDF has 0 pages, returning empty response")
        return ClassifyResponse(
            pages=[],
            inference_time_seconds=0.0,
            total_time_seconds=round(time.time() - t_start, 3),
        )

    # ─── Render all pages in parallel + load PNGs into memory ─────────
    # Use a request-specific output dir so concurrent calls don't collide.
    render_dir = PREPROCESSING_OUTPUT_DIR / request.plan_id
    render_dir.mkdir(parents=True, exist_ok=True)
    base_image_path = render_dir / "page.png"

    def render_and_read() -> tuple[list[bytes], list[int], float]:
        """Blocking: render every page to PNG (parallel) then read the bytes.

        Runs entirely inside a worker thread (via run_in_executor) so the
        event loop is never parked on the .result() calls below — that is
        what keeps /healthz answerable during a long render. Must not touch
        the event loop or any asyncio objects.
        """
        t_render = time.time()
        page_image_paths: list[Path] = [None] * n_pages

        def render_one(idx: int) -> tuple[int, Path, float]:
            t_page = time.time()
            out = preprocess(str(pdf_path), idx, image_path=str(base_image_path))
            return idx, out, time.time() - t_page

        with ThreadPoolExecutor(max_workers=PREPROCESSING_MAX_WORKERS) as executor:
            futures = [executor.submit(render_one, idx) for idx in range(n_pages)]
            for fut in futures:
                idx, out_path, page_elapsed = fut.result()  # propagates exceptions
                page_image_paths[idx] = out_path
                logger.info(
                    f"[PNG] [{request_id}] page={idx} rendered in "
                    f"{page_elapsed * 1000:.0f}ms ({out_path.stat().st_size / 1e6:.2f} MB)"
                )

        elapsed = time.time() - t_render

        images: list[bytes] = []
        nums: list[int] = []
        for idx in range(n_pages):
            with open(page_image_paths[idx], "rb") as f:
                images.append(f.read())
            nums.append(idx)
        return images, nums, elapsed

    logger.info(
        f"[PNG] [{request_id}] conversion started — {n_pages} pages "
        f"(DPI=150, max_workers={PREPROCESSING_MAX_WORKERS})"
    )
    try:
        image_bytes_list, page_numbers, preprocess_time = await loop.run_in_executor(
            None, render_and_read
        )
        logger.info(
            f"[PNG] [{request_id}] conversion finished — {n_pages} pages in "
            f"{preprocess_time:.2f}s (avg {preprocess_time / n_pages * 1000:.0f}ms/page)"
        )
    except Exception as e:
        logger.exception(f"[PNG] [{request_id}] conversion/read failed: {e}")
        # Clean up partial render dir
        shutil.rmtree(render_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Page rendering failed: {e}",
        )

    # ─── Classify (run blocking GPU inference in a thread so the event loop
    # stays free to serve /healthz liveness probes) ──────────
    logger.info(f"[CLASSIFIER] [{request_id}] inference started — {n_pages} pages")
    try:
        t0 = time.time()
        predictions = await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            classifier.classify_images,
            image_bytes_list,
            page_numbers,
        )
        inference_time = time.time() - t0

        # Per-page log lines for full visibility
        for pred in predictions:
            demoted_tag = " [DEMOTED]" if pred.threshold_demoted else ""
            logger.info(
                f"[CLASSIFIER] [{request_id}] page={pred.page_number} "
                f"plan_type={pred.plan_type} conf={pred.confidence:.3f}"
                f"{demoted_tag}"
            )

        logger.info(
            f"[CLASSIFIER] [{request_id}] inference finished — {n_pages} pages "
            f"in {inference_time:.2f}s (avg {inference_time / n_pages * 1000:.0f}ms/page)"
        )
    except Exception as e:
        logger.exception(f"[CLASSIFIER] [{request_id}] inference failed: {e}")
        shutil.rmtree(render_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e}",
        )

    # ─── Cleanup ──────────────────────────────────────────────
    try:
        shutil.rmtree(render_dir, ignore_errors=True)
        try:
            pdf_path.unlink()
        except FileNotFoundError:
            pass
        logger.info(f"[CLEANUP] [{request_id}] /tmp artifacts removed")
    except Exception as e:
        # Cleanup is best-effort; log but don't fail the request
        logger.warning(f"[CLEANUP] [{request_id}] failed: {e}")

    total_time = time.time() - t_start
    logger.info(
        f"[REQUEST] [{request_id}] complete in {total_time:.2f}s "
        f"(gcs={gcs_elapsed:.2f}s, png={preprocess_time:.2f}s, "
        f"classify={inference_time:.2f}s)"
    )

    return ClassifyResponse(
        pages=predictions,
        inference_time_seconds=round(inference_time, 3),
        total_time_seconds=round(total_time, 3),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
