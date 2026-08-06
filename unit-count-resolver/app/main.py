"""
Unit-Count Resolver — Cloud Run service entry point (master plan D5a).

Replaces the detached in-process background task that ran inside xtimator-3d on
feat/multifamily. That placement was tested and failed: the task died with its
instance, and it shared Vertex quota with the sectioning call during the
automatic burst (master plan section 9a). This service has its own memory, timeout
and lifecycle.

Endpoint:
    POST /resolve_unit_counts
    Content-Type: application/json
    Body: {"project_id": "...", "plan_id": "...", "user_id": "..."}

    Returns 202 IMMEDIATELY. The resolve job then runs in the background:
    fetch the PDF from GCS -> detect candidate pages -> extract counts ->
    apply D10 precedence -> write plans.unit_counts.

Security: internal-only ingress + ID-token auth, enforced at the Cloud Run
platform layer by the deploy flags `--ingress internal --no-allow-unauthenticated`
(see .github/workflows/unit_count_resolver.yml). This matches wall-detector and
xtimator-page-classifier, neither of which verifies the token in-app; callers mint
one with IDTokenCredentials(target_audience=<service URL>) — see
xtimator-3d/helper.py:549-557.

Log phases (grep-friendly, D19):
    [UNIT_COUNTS] — every resolver phase, with timings
"""
import asyncio
import logging
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from .config import RESOLVER_WORK_DIR, load_gcp_credentials
from .gcs_client import download_floorplan_pdf
from .resolver import resolve_unit_counts
from .schemas import ResolveUnitCountsAccepted, ResolveUnitCountsRequest
from .shared import close_pg_pool, load_organization_slug, load_pg_pool


logging.basicConfig(
    level=logging.INFO,
    format='{"severity": "%(levelname)s", "message": "%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

CREDENTIALS = load_gcp_credentials()
pg_pool = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build the Cloud SQL pool (same retry shape as xtimator-3d)."""
    logging.info("[UNIT_COUNTS] unit-count-resolver starting")
    for attempt in range(10):
        try:
            pg_pool["engine"] = load_pg_pool(CREDENTIALS)
            break
        except Exception as e:
            logging.exception(e)
            if attempt == 9:
                raise
            await asyncio.sleep(min(2 ** attempt, 30))
    logging.info("[UNIT_COUNTS] unit-count-resolver ready")
    yield
    if pg_pool and pg_pool.get("engine"):
        close_pg_pool()


app = FastAPI(
    title="Unit-Count Resolver (Cloud Run)",
    version="1.0.0",
    description="Multi-family unit-count resolution: detect -> extract -> persist plans.unit_counts.",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"service": "unit-count-resolver", "version": "1.0.0"}


@app.get("/healthz")
def healthz():
    """Cloud Run startup + liveness probe. Ready once the DB pool exists."""
    if not pg_pool.get("engine"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database pool not ready",
        )
    return {"status": "ready"}


def _run_resolve_job(project_id, plan_id, user_id):
    """The whole background job. Sync/blocking — BackgroundTasks runs a plain
    `def` in a worker thread, which is exactly what resolve_unit_counts wants.

    Catches EVERYTHING: a resolver failure must never escape as an unhandled
    background exception.
    """
    context = f"project={project_id} plan={plan_id}"
    t0 = perf_counter()
    work_dir = Path(RESOLVER_WORK_DIR) / project_id / plan_id
    pdf_path = work_dir / "floor_plan.PDF"
    try:
        organization_slug = load_organization_slug(CREDENTIALS, pg_pool, user_id)
        download_floorplan_pdf(
            CREDENTIALS,
            organization_slug=organization_slug,
            project_id=project_id,
            plan_id=plan_id,
            destination_path=pdf_path,
        )
        logging.info(
            f"[UNIT_COUNTS] [{context}] PDF fetched ({pdf_path.stat().st_size / 1e6:.2f} MB); resolving"
        )
        # client_ip_address=None -> load_nearest_region falls back to its default
        # region. This service has no end-user IP to geolocate (it is called
        # service-to-service), so the default is the correct behaviour, not a gap.
        resolve_unit_counts(
            CREDENTIALS,
            pg_pool,
            None,
            str(pdf_path),
            project_id,
            plan_id,
            # Only used when UNIT_COUNT_DEBUG is on, to place debug artifacts in
            # the plan's own GCS folder.
            organization_slug=organization_slug,
        )
        logging.info(f"[UNIT_COUNTS] [{context}] job COMPLETED in {perf_counter() - t0:.3f}s")
    except FileNotFoundError as e:
        logging.warning(f"[UNIT_COUNTS] [{context}] job ABORTED — PDF not in GCS: {e}")
    except Exception as e:
        logging.warning(f"[UNIT_COUNTS] [{context}] job FAILED (contained): {e}", exc_info=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post(
    "/resolve_unit_counts",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ResolveUnitCountsAccepted,
)
async def resolve_unit_counts_endpoint(
    request: ResolveUnitCountsRequest,
    background_tasks: BackgroundTasks,
):
    """Accept the job and return 202 immediately; resolve in the background.

    The caller (xtimator-3d, end of /floorplan_to_preview) is fire-and-forget with
    a 5s timeout and never depends on the outcome, so nothing here may block.
    """
    logging.info(
        f"[UNIT_COUNTS] [project={request.project_id} plan={request.plan_id} "
        f"user={request.user_id}] /resolve_unit_counts accepted"
    )
    background_tasks.add_task(
        _run_resolve_job, request.project_id, request.plan_id, request.user_id
    )
    return ResolveUnitCountsAccepted(
        project_id=request.project_id,
        plan_id=request.plan_id,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
