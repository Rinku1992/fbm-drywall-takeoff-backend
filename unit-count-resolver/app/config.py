"""
Configuration for the unit-count-resolver Cloud Run service (D5a).

Follows the same convention as xtimator-page-classifier/app/config.py and
xtimator-3d/main.py:load_gcp_credentials — credentials come from a gcp.yaml that
the operator stages into the build context (it is NOT committed; it carries the
Cloud SQL / Vertex service-account key paths).
"""
import os
from pathlib import Path

from ruamel.yaml import YAML


def load_gcp_credentials() -> dict:
    """Read config/gcp.yaml and export GOOGLE_APPLICATION_CREDENTIALS.

    Mirrors xtimator-3d/main.py:load_gcp_credentials (main.py:652-658) so the
    same gcp.yaml works for this service unchanged.
    """
    yaml = YAML(typ="safe", pure=True)
    gcp_yaml_path = Path("/app/config/gcp.yaml")
    if not gcp_yaml_path.exists():
        # Local dev — running from unit-count-resolver/
        gcp_yaml_path = Path("config/gcp.yaml")
    with open(gcp_yaml_path, "r") as f:
        credentials = yaml.load(f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials["service_drywall_account_key"]
    return credentials


# ─── Resolver tunables ────────────────────────────────────────
# The resolver renders at its OWN DPI (master plan Phase 1 item 1): the repo's
# shared render DPI is hardcoded for the modelling path and is not appropriate here.

# Detection (scanned path) thumbnail DPI, and the one-shot higher-DPI retry.
UNIT_COUNT_THUMBNAIL_DPI = int(os.environ.get("UNIT_COUNT_THUMBNAIL_DPI", "72"))
UNIT_COUNT_THUMBNAIL_RETRY_DPI = int(os.environ.get("UNIT_COUNT_THUMBNAIL_RETRY_DPI", "144"))
# Thumbnails per detection call (D7).
UNIT_COUNT_TRIAGE_BATCH_SIZE = int(os.environ.get("UNIT_COUNT_TRIAGE_BATCH_SIZE", "25"))

# D5c — triage batches are SERIALIZED (one at a time). This is the pause between
# consecutive batches, so the resolver's Gemini footprint stays small enough not
# to matter if it ever overlaps the sectioning call's quota window.
UNIT_COUNT_TRIAGE_BACKOFF_SECONDS = float(os.environ.get("UNIT_COUNT_TRIAGE_BACKOFF_SECONDS", "2.0"))

# Extraction (high-res) render DPI and how many candidate pages get sent (D7).
UNIT_COUNT_EXTRACTION_DPI = int(os.environ.get("UNIT_COUNT_EXTRACTION_DPI", "200"))
UNIT_COUNT_MAX_CANDIDATES = int(os.environ.get("UNIT_COUNT_MAX_CANDIDATES", "3"))

# BUG FIX 3 — render cap. Aurora's 113M-pixel page renders broke extraction and
# trip PIL's ~89M-pixel DecompressionBombError. Cap the LONGEST SIDE of any
# extraction render; the zoom is reduced to fit, never increased.
UNIT_COUNT_MAX_RENDER_LONGEST_SIDE = int(os.environ.get("UNIT_COUNT_MAX_RENDER_LONGEST_SIDE", "4000"))
# Secondary guard on total pixels, kept below PIL's default 89,478,485 limit.
UNIT_COUNT_MAX_RENDER_PIXELS = int(os.environ.get("UNIT_COUNT_MAX_RENDER_PIXELS", "80000000"))

# BUG FIX 1 — fabrication guard. Extraction gets at most this many attempts, at a
# FIXED temperature (no escalation). Master plan section 9 / D5c.
UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS = int(os.environ.get("UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS", "2"))
UNIT_COUNT_EXTRACTION_TEMPERATURE = float(os.environ.get("UNIT_COUNT_EXTRACTION_TEMPERATURE", "0"))

# Where the resolver stages the PDF it pulls from GCS.
RESOLVER_WORK_DIR = Path(os.environ.get("RESOLVER_WORK_DIR", "/tmp/unit_count_resolver"))
