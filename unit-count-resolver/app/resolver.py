"""
Unit-Count Resolver core (D5a) — detect -> extract -> resolve -> persist.

LIFTED from `xtimator-3d/helper.py` @ feat/multifamily (f1d96eb7). Provenance of
each piece is recorded per-function below and in MF_REBUILD_REPORT.md.

Lifted UNCHANGED:
    _score_unit_count_page_text, rank_unit_count_pages_vector   (vector text-scan ranking)
    _unify_candidate_pages, _apply_precedence                   (D10 precedence)

Lifted WITH FIXES:
    _triage_pass, triage_unit_count_pages_scanned  -- D5c serialized + backoff (item 10)
    _extract_unit_counts                           -- bounded attempts/fixed temp (item 7)
                                                      + render cap (item 9)
    _persist_unit_counts, resolve_unit_counts      -- extraction health, per-type
                                                      logging, sanity check, and the
                                                      extraction_failed state (items 7, 8)

DELIBERATELY NOT LIFTED (the old execution model that D5a replaces):
    launch_unit_count_resolver, _run_unit_count_resolver, _snapshot_pdf,
    _UNIT_COUNT_RESOLVER_TASKS  -- the detached in-process task machinery. This
    service IS the replacement; see master plan section 7.

Everything here is sync/blocking by design: main.py runs it in a worker thread.
"""
import json
import logging
import re
from pathlib import Path
from time import perf_counter, sleep

import fitz

from .config import (
    UNIT_COUNT_THUMBNAIL_DPI,
    UNIT_COUNT_THUMBNAIL_RETRY_DPI,
    UNIT_COUNT_TRIAGE_BATCH_SIZE,
    UNIT_COUNT_TRIAGE_BACKOFF_SECONDS,
    UNIT_COUNT_EXTRACTION_DPI,
    UNIT_COUNT_MAX_CANDIDATES,
    UNIT_COUNT_MAX_RENDER_LONGEST_SIDE,
    UNIT_COUNT_MAX_RENDER_PIXELS,
    UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS,
    UNIT_COUNT_EXTRACTION_TEMPERATURE,
)
from .prompts import (
    UNIT_COUNT_DETECTOR,
    UnitCountDetectionResponse,
    UNIT_COUNT_EXTRACTOR,
    UnitCountResponse,
)
from .shared import (
    Content,
    Part,
    load_vertex_ai_client,
    phoenix_call,
    pg_run,
)


# ─────────────────── Vector text-scan ranking (lifted UNCHANGED) ─────────────
# From feat/multifamily xtimator-3d/helper.py:1458-1608. Reads each page's text
# layer via fitz.get_text() DIRECTLY (NOT vector_pdf.py, which only parses
# scale/ceiling) and RANKS pages by unit-count likelihood. Zero Gemini calls (D5c).

_UNIT_COUNT_MIN_PAGE_TEXT_CHARS = 100

# Unit-type-like tokens (D8 — one signal among several): 1BR / 2 BR, STUDIO,
# 1 BED / 2 BEDROOM(S), 2BA, UNIT TYPE, TYPE A / TYPE 2 / TYPE A1, UNIT.
_UNIT_TYPE_RE = re.compile(
    r"\b\d\s?BR\b"
    r"|\bSTUDIO\b"
    r"|\b\d\s?BED(?:ROOM)?S?\b"
    r"|\b\d\s?BA\b"
    r"|\bUNIT\s?TYPE\b"
    r"|\bTYPE\s?[A-Z0-9]{1,3}\b"
    r"|\bUNIT\b",
    re.IGNORECASE,
)
# Schedule / count vocabulary (D8 — another signal, not the only one).
_UNIT_COUNT_KEYWORD_RE = re.compile(
    r"\b(?:TOTAL|COUNT|SCHEDULE|MIX|TABULATION|DWELLING|UNITS?|QUANTITY|QTY|NO\.?\s?OF\s?UNITS)\b",
    re.IGNORECASE,
)
# A small standalone integer (1..999): count cells / number columns, not the
# long dimension strings ("29'-0\"") that dominate ordinary plan sheets.
_SMALL_INT_RE = re.compile(r"(?<![\d.])\d{1,3}(?![\d.])")

def _score_unit_count_page_text(text):
    """Score one page's text layer for unit-count likelihood (D8, multi-signal).

    Pure/deterministic; performs NO extraction. Returns (score, signals) where
    signals is a dict of the raw counts that fed the score:
      - unit_type_hits: unit-type-like tokens (1BR, STUDIO, TYPE A, UNIT, ...)
      - keyword_hits:   schedule/count vocabulary (schedule, mix, total, ...)
      - number_rows:    lines ending in a small integer (number-column / table-row proxy)
    Each signal is capped before weighting so no single one (e.g. a stray
    keyword) can dominate the ranking.
    """
    if not text:
        return 0.0, {"unit_type_hits": 0, "keyword_hits": 0, "number_rows": 0}

    unit_type_hits = len(_UNIT_TYPE_RE.findall(text))
    keyword_hits = len(_UNIT_COUNT_KEYWORD_RE.findall(text))
    number_rows = sum(
        1 for line in text.splitlines()
        if line.strip() and _SMALL_INT_RE.search(line.strip().split()[-1])
    )

    score = (
        2.0 * min(unit_type_hits, 12) / 12.0
        + 1.5 * min(keyword_hits, 8) / 8.0
        + 1.0 * min(number_rows, 15) / 15.0
    )
    return round(score, 4), {
        "unit_type_hits": unit_type_hits,
        "keyword_hits": keyword_hits,
        "number_rows": number_rows,
    }

def rank_unit_count_pages_vector(pdf_path, project_id, plan_id, top_k_log=5):
    """Rank pages likely to hold unit counts via the vector text layer (item 3).

    Reads each page's text with fitz.get_text("text") DIRECTLY and scores it with
    the multi-signal heuristic above. Page numbers are 0-indexed (fitz/repo
    convention). Extraction is NOT performed here.

    Returns:
        {
          "applicable": bool,     # False => scanned / no text layer; the caller
                                  #          must hand off to the thumbnail path.
          "reason": str,          # "vector" | "scanned_no_text_layer" |
                                  #  "open_failed" | "no_pages"
          "ranked_pages": [       # best-first; empty when not applicable (or when
                                  #  applicable but no page shows any signal).
              {"page_number": int, "score": float, "signals": {...}}, ...
          ],
        }

    Never raises: degrades to applicable=False on any failure so the resolver can
    fall back to the thumbnail path. Logs under [UNIT_COUNTS] with timing and the
    top-ranked pages.
    """
    context = f"project={project_id} plan={plan_id}"
    t0 = perf_counter()
    logging.info(f"[UNIT_COUNTS] [{context}] vector text-scan ranking — started")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.warning(
            f"[UNIT_COUNTS] [{context}] PDF open failed: {e}; applicable=False "
            f"(hand off to thumbnail path) — done in {perf_counter() - t0:.3f}s"
        )
        return {"applicable": False, "reason": "open_failed", "ranked_pages": []}

    try:
        n_pages = doc.page_count
        if n_pages == 0:
            logging.info(
                f"[UNIT_COUNTS] [{context}] pages=0; applicable=False — "
                f"done in {perf_counter() - t0:.3f}s"
            )
            return {"applicable": False, "reason": "no_pages", "ranked_pages": []}

        scored = []
        pages_with_text = 0
        for page_index in range(n_pages):
            try:
                text = doc.load_page(page_index).get_text("text") or ""
            except Exception as e:
                logging.warning(
                    f"[UNIT_COUNTS] [{context} page={page_index}] text read failed: {e}; scored 0"
                )
                text = ""
            if len(text.strip()) >= _UNIT_COUNT_MIN_PAGE_TEXT_CHARS:
                pages_with_text += 1
            score, signals = _score_unit_count_page_text(text)
            if score > 0:
                scored.append({"page_number": page_index, "score": score, "signals": signals})

        # No usable text layer anywhere => scanned; defer to the thumbnail path.
        if pages_with_text == 0:
            logging.info(
                f"[UNIT_COUNTS] [{context}] pages={n_pages} pages_with_text=0 => vector scan "
                f"NOT applicable (scanned); hand off to thumbnail path — "
                f"done in {perf_counter() - t0:.3f}s"
            )
            return {"applicable": False, "reason": "scanned_no_text_layer", "ranked_pages": []}

        ranked = sorted(scored, key=lambda r: (-r["score"], r["page_number"]))

        top_preview = ", ".join(
            f"p{r['page_number']}={r['score']}"
            f"(ut={r['signals']['unit_type_hits']},"
            f"kw={r['signals']['keyword_hits']},"
            f"nr={r['signals']['number_rows']})"
            for r in ranked[:top_k_log]
        ) or "none"
        logging.info(
            f"[UNIT_COUNTS] [{context}] pages={n_pages} pages_with_text={pages_with_text} "
            f"candidates={len(ranked)} top[{top_preview}] — done in {perf_counter() - t0:.3f}s"
        )
        return {"applicable": True, "reason": "vector", "ranked_pages": ranked}
    finally:
        doc.close()



# ─────────────── Scanned-path thumbnail triage (D5c, item 10) ────────────────
# From feat/multifamily xtimator-3d/helper.py:1611-1815.
#
# ITEM 10 / D5c — SERIALIZED TRIAGE. Finding reported in MF_REBUILD_REPORT.md:
# the lifted loop was ALREADY sequential (a plain `for batch_index in
# range(n_batches)` with a blocking phoenix_call inside — helper.py:1641). The
# MF_CURRENT_STATUS line "thumbnail triage batches fired freely" does not match
# that code; there was never a ThreadPoolExecutor or gather here. What was
# genuinely missing is the BACKOFF between consecutive calls, so a many-page
# scanned set still issued back-to-back Gemini calls as fast as they returned.
# Added below: an explicit sleep between batches, an explicit no-concurrency
# contract in the docstring, and logging of the serialization so it is auditable.


def _triage_pass(doc, dpi, batch_size, vertex_ai_client, generation_config, is_cached, credentials, context):
    """One full detection pass over ALL pages of an open fitz doc at `dpi`.

    Renders each page to a PNG thumbnail (own DPI via fitz matrix), batches
    `batch_size` per phoenix_call, and returns a flat list of hit dicts
    {page_number, form, confidence, evidence} (0-indexed). Per-batch progress is
    logged under [UNIT_COUNTS]. A failed render or a failed batch is skipped
    (logged), never fatal.

    D5c CONTRACT: batches run STRICTLY ONE AT A TIME — this loop is sequential
    and each phoenix_call blocks until it returns. Do NOT parallelise it. A
    UNIT_COUNT_TRIAGE_BACKOFF_SECONDS pause separates consecutive calls so the
    resolver's Gemini footprint stays small enough not to matter if it overlaps
    the sectioning call's quota window (master plan section 9a).
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    n_pages = doc.page_count
    n_batches = (n_pages + batch_size - 1) // batch_size
    hits = list()

    logging.info(
        f"[UNIT_COUNTS] [{context}] triage pass dpi={dpi} — {n_batches} batch(es), "
        f"SERIALIZED (one at a time, {UNIT_COUNT_TRIAGE_BACKOFF_SECONDS}s backoff between calls)"
    )

    for batch_index in range(n_batches):
        # D5c: pause BEFORE every call after the first. Placed here (not after the
        # call) so a `continue` on a skipped/failed batch cannot bypass it.
        if batch_index > 0 and UNIT_COUNT_TRIAGE_BACKOFF_SECONDS > 0:
            sleep(UNIT_COUNT_TRIAGE_BACKOFF_SECONDS)

        start = batch_index * batch_size
        end = min(start + batch_size, n_pages)
        query_parts = list()
        rendered = list()
        for page_index in range(start, end):
            try:
                png_bytes = doc.load_page(page_index).get_pixmap(matrix=matrix).tobytes("png")
            except Exception as e:
                logging.warning(f"[UNIT_COUNTS] [{context} page={page_index}] thumbnail render failed: {e}; skipped")
                continue
            query_parts.append(Part.from_text(f"PAGE: {page_index}"))
            query_parts.append(Part.from_data(data=png_bytes, mime_type="image/png"))
            rendered.append(page_index)

        if not rendered:
            logging.warning(
                f"[UNIT_COUNTS] [{context}] triage batch {batch_index + 1}/{n_batches} "
                f"— pages {start}–{end - 1} (dpi={dpi}) — no pages rendered; skipped"
            )
            continue

        query = Content(role="user", parts=query_parts)
        try:
            if is_cached:
                detection, _ = phoenix_call(
                    lambda feedback_prompt, temperature: vertex_ai_client.generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**generation_config, "temperature": temperature},
                    ),
                    max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=UnitCountDetectionResponse,
                    verify_field_counts=dict(pages=len(rendered)),
                )
            else:
                detection, _ = phoenix_call(
                    lambda feedback_prompt, temperature: vertex_ai_client(UNIT_COUNT_DETECTOR).generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**generation_config, "temperature": temperature},
                    ),
                    max_retry=credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=UnitCountDetectionResponse,
                    verify_field_counts=dict(pages=len(rendered)),
                )
        except Exception as e:
            logging.warning(
                f"[UNIT_COUNTS] [{context}] triage batch {batch_index + 1}/{n_batches} "
                f"— pages {start}–{end - 1} (dpi={dpi}) — detection failed: {e}; batch skipped"
            )
            continue

        batch_hits = [page for page in detection.pages if page.has_unit_counts]
        for page in batch_hits:
            hits.append({
                "page_number": page.page_number,
                "form": page.form,
                "confidence": page.confidence,
                "evidence": page.evidence,
            })
        hit_str = ", ".join(f"p{page.page_number}({page.form})" for page in batch_hits) or "none"
        logging.info(
            f"[UNIT_COUNTS] [{context}] triage batch {batch_index + 1}/{n_batches} "
            f"— pages {start}–{end - 1} (dpi={dpi}) — hit on {hit_str}"
        )

    return hits


def triage_unit_count_pages_scanned(
    credentials,
    client_ip_address,
    pdf_path,
    project_id,
    plan_id,
    low_dpi=UNIT_COUNT_THUMBNAIL_DPI,
    retry_dpi=UNIT_COUNT_THUMBNAIL_RETRY_DPI,
    batch_size=UNIT_COUNT_TRIAGE_BATCH_SIZE,
):
    """Scanned-PDF detection path: find candidate pages via vision triage.

    Lifted from feat/multifamily helper.py:1709-1815; body unchanged except that
    the two `_triage_pass` calls now inherit the D5c serialization/backoff above,
    and the defaults come from config.py instead of module constants.

    Returns:
        {
          "found": bool, "reason": str, "hits": [...],
          "dpi_used": int, "retried": bool,
        }

    Never raises: degrades to found=False on any failure. Page numbers 0-indexed.
    """
    context = f"project={project_id} plan={plan_id}"
    t0 = perf_counter()
    logging.info(
        f"[UNIT_COUNTS] [{context}] scanned-path thumbnail triage — started "
        f"(batch={batch_size}, dpi={low_dpi})"
    )

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.warning(
            f"[UNIT_COUNTS] [{context}] PDF open failed: {e}; found=False — "
            f"done in {perf_counter() - t0:.3f}s"
        )
        return {"found": False, "reason": "open_failed", "hits": [], "dpi_used": low_dpi, "retried": False}

    try:
        n_pages = doc.page_count
        if n_pages == 0:
            logging.info(f"[UNIT_COUNTS] [{context}] pages=0; found=False — done in {perf_counter() - t0:.3f}s")
            return {"found": False, "reason": "no_pages", "hits": [], "dpi_used": low_dpi, "retried": False}

        try:
            vertex_ai_client, generation_config, is_cached = load_vertex_ai_client(
                credentials, client_ip_address, prompts=[UNIT_COUNT_DETECTOR]
            )
        except Exception as e:
            logging.warning(
                f"[UNIT_COUNTS] [{context}] Vertex client init failed: {e}; found=False — "
                f"done in {perf_counter() - t0:.3f}s"
            )
            return {"found": False, "reason": "client_init_failed", "hits": [], "dpi_used": low_dpi, "retried": False}

        # Pass 1 — low-res triage over all pages.
        hits = _triage_pass(doc, low_dpi, batch_size, vertex_ai_client, generation_config, is_cached, credentials, context)
        dpi_used = low_dpi
        retried = False

        # Safeguard: zero hits at low DPI can be a resolution miss (tiny/faint
        # tables). Retry ONCE at a higher thumbnail DPI before declaring none-found.
        if not hits:
            logging.info(
                f"[UNIT_COUNTS] [{context}] pass 1 (dpi={low_dpi}) found 0 hits; "
                f"retrying once at dpi={retry_dpi}"
            )
            retried = True
            hits = _triage_pass(doc, retry_dpi, batch_size, vertex_ai_client, generation_config, is_cached, credentials, context)
            dpi_used = retry_dpi

        # De-duplicate by page (keep highest confidence) and rank best-first.
        by_page = dict()
        for hit in hits:
            existing = by_page.get(hit["page_number"])
            if existing is None or hit["confidence"] > existing["confidence"]:
                by_page[hit["page_number"]] = hit
        ranked = sorted(by_page.values(), key=lambda h: (-h["confidence"], h["page_number"]))

        if not ranked:
            logging.info(
                f"[UNIT_COUNTS] [{context}] NONE-FOUND after {'retry' if retried else 'pass 1'} "
                f"(dpi={dpi_used}) — done in {perf_counter() - t0:.3f}s"
            )
            return {"found": False, "reason": "none_found", "hits": [], "dpi_used": dpi_used, "retried": retried}

        top_preview = ", ".join(
            f"p{h['page_number']}({h['form']},{round(h['confidence'], 2)})" for h in ranked[:5]
        )
        logging.info(
            f"[UNIT_COUNTS] [{context}] triage found {len(ranked)} candidate page(s) at dpi={dpi_used} "
            f"top[{top_preview}] — done in {perf_counter() - t0:.3f}s"
        )
        return {"found": True, "reason": "hits", "hits": ranked, "dpi_used": dpi_used, "retried": retried}
    finally:
        doc.close()


# ─────────── Candidate unification + D10 precedence (lifted UNCHANGED) ───────
# From feat/multifamily xtimator-3d/helper.py:1830-1843 and 1914-1947.
#
# NOTE on the stale inline comment inside _apply_precedence: its `else` branch is
# annotated "Defensive: UnitCountResponse validation forbids empty-and-no-total."
# That statement was true on feat/multifamily and is NO LONGER true here — bug
# fix 2 makes an empty-and-no-total response valid. The branch is nonetheless
# still unreachable, because resolve_unit_counts returns on
# `extraction.is_none_found` BEFORE calling this function. The code is left
# byte-identical to the lift on purpose; this note is the correction.

def _unify_candidate_pages(vector_result, scanned_result, max_candidates):
    """Collapse the vector (item 3) OR scanned (item 4) detection result into one
    ranked list of 0-indexed candidate page numbers (best-first), capped at
    `max_candidates`. Only one detection path runs per document; this normalises
    whichever result we have into a single list plus the path that produced it."""
    if vector_result is not None and vector_result.get("applicable"):
        pages = [p["page_number"] for p in vector_result["ranked_pages"]]
        path = "vector"
    elif scanned_result is not None:
        pages = [h["page_number"] for h in scanned_result["hits"]]
        path = "scanned"
    else:
        pages, path = [], "none"
    return pages[:max_candidates], path

def _apply_precedence(extraction, context):
    """Apply the D10 resolved-count precedence: table > prose > none.

    Produces exactly ONE resolved count per unit type. When a table exists its
    per-type counts win, EVEN IF a prose total also exists and disagrees; the
    disagreement is logged (never passed downstream, never retried). Returns
    (resolved, disagreement) where resolved is
    {"source", "unit_counts", "total_units"} and disagreement is a note or None.
    """
    per_type = [
        {"unit_type": t.unit_type, "count": t.count, "area": t.area}
        for t in extraction.per_type_counts
    ]
    total_units = extraction.total_units
    source_form = extraction.source_form
    disagreement = None

    if per_type:
        # Table (or the table half of "mixed") wins the resolved value.
        table_sum = sum(t["count"] for t in per_type)
        if source_form == "mixed" and total_units is not None and total_units != table_sum:
            disagreement = (
                f"counts from table (sum={table_sum}); prose total={total_units} disagreed — table used"
            )
            logging.info(f"[UNIT_COUNTS] [{context}] DISAGREEMENT: {disagreement}")
        resolved = {"source": "table", "unit_counts": per_type, "total_units": table_sum}
    elif total_units is not None:
        # No table anywhere; prose total is the resolved value (per-type unresolved).
        resolved = {"source": "prose", "unit_counts": [], "total_units": total_units}
    else:
        # Defensive: UnitCountResponse validation forbids empty-and-no-total.
        resolved = {"source": "none_found", "unit_counts": [], "total_units": None}

    return resolved, disagreement

# ─────────── High-res extraction + persistence (items 7, 8, 9) ───────────────
# From feat/multifamily xtimator-3d/helper.py:1818-2081, with the fabrication
# guard (item 7), the none-found handling (item 8) and the render cap (item 9).


def _capped_matrix(page, dpi, context, page_index):
    """BUG FIX 3 (item 9) — build a fitz.Matrix for `dpi`, REDUCED if the
    resulting raster would be too large.

    BEFORE (feat/multifamily helper.py:1859-1865): the extraction render was
        zoom = high_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        ... get_pixmap(matrix=matrix)
    with no size bound at all. An architectural E-size sheet at 200 DPI is
    ~113M pixels — that is Aurora. It exceeds PIL's default
    DecompressionBombError threshold (~89.5M pixels) and produced renders large
    enough to break the extraction call outright.

    AFTER: the zoom is scaled DOWN (never up) so that
        longest side  <= UNIT_COUNT_MAX_RENDER_LONGEST_SIDE (default 4000 px)
        total pixels  <= UNIT_COUNT_MAX_RENDER_PIXELS        (default 80M, under PIL's ~89.5M)
    Both bounds are applied; the tighter one wins. Capping is logged so a
    downscaled read is visible when a result is reviewed.
    """
    zoom = dpi / 72.0
    width_pt, height_pt = page.rect.width, page.rect.height
    if width_pt <= 0 or height_pt <= 0:
        return fitz.Matrix(zoom, zoom), False

    width_px, height_px = width_pt * zoom, height_pt * zoom
    capped = False

    longest = max(width_px, height_px)
    if longest > UNIT_COUNT_MAX_RENDER_LONGEST_SIDE:
        zoom *= UNIT_COUNT_MAX_RENDER_LONGEST_SIDE / longest
        width_px, height_px = width_pt * zoom, height_pt * zoom
        capped = True

    total_px = width_px * height_px
    if total_px > UNIT_COUNT_MAX_RENDER_PIXELS:
        zoom *= (UNIT_COUNT_MAX_RENDER_PIXELS / total_px) ** 0.5
        width_px, height_px = width_pt * zoom, height_pt * zoom
        capped = True

    if capped:
        logging.info(
            f"[UNIT_COUNTS] [{context} page={page_index}] render CAPPED — "
            f"requested dpi={dpi} ({width_pt * (dpi / 72.0):.0f}x{height_pt * (dpi / 72.0):.0f}px) "
            f"-> effective dpi={zoom * 72.0:.1f} ({width_px:.0f}x{height_px:.0f}px, "
            f"{width_px * height_px / 1e6:.1f}MP)"
        )
    return fitz.Matrix(zoom, zoom), capped


def _extract_unit_counts(credentials, client_ip_address, pdf_path, candidate_pages, high_dpi, context):
    """Render the candidate pages at HIGH DPI (capped) and run UNIT_COUNT_EXTRACTOR.

    BUG FIX 1 (item 7) — BOUNDED, FIXED-TEMPERATURE extraction.

    BEFORE (feat/multifamily helper.py:1886-1907): ONE phoenix_call with
        max_retry=credentials["VertexAI"]["llm"]["max_retry"]        # = 5 (gcp.yaml:43)
    and phoenix_call escalates temperature on every retry — on that branch to a
    0.5 ceiling (helper.py:651). So a hard page got up to 5 attempts at rising
    temperature, and the caller could not tell a first-call answer from one that
    emerged out of a 5-deep spiral. That is the documented Aurora fabrication
    path (master plan section 9).

    AFTER: this function owns the attempt loop. Each attempt is a phoenix_call
    with max_retry=1 (exactly one shot, raises on failure), and the lambda
    IGNORES the temperature phoenix_call offers, pinning
    UNIT_COUNT_EXTRACTION_TEMPERATURE instead. Attempts stop at
    UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS (default 2). No escalation exists anywhere
    on this path.

    Returns (extraction_or_None, attempts_used). extraction is None when every
    attempt failed — the caller then stores extraction_failed, NEVER a guess.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.warning(f"[UNIT_COUNTS] [{context}] extraction PDF open failed: {e}; extraction=None")
        return None, 0
    try:
        query_parts = list()
        rendered = list()
        for page_index in candidate_pages:
            try:
                page = doc.load_page(page_index)
                matrix, _ = _capped_matrix(page, high_dpi, context, page_index)
                png_bytes = page.get_pixmap(matrix=matrix).tobytes("png")
            except Exception as e:
                logging.warning(f"[UNIT_COUNTS] [{context} page={page_index}] high-res render failed: {e}; skipped")
                continue
            query_parts.append(Part.from_text(f"PAGE: {page_index}"))
            query_parts.append(Part.from_data(data=png_bytes, mime_type="image/png"))
            rendered.append(page_index)

        if not rendered:
            logging.warning(f"[UNIT_COUNTS] [{context}] extraction rendered 0 pages; extraction=None")
            return None, 0

        try:
            vertex_ai_client, generation_config, is_cached = load_vertex_ai_client(
                credentials, client_ip_address, prompts=[UNIT_COUNT_EXTRACTOR]
            )
        except Exception as e:
            logging.warning(f"[UNIT_COUNTS] [{context}] Vertex client init failed: {e}; extraction=None")
            return None, 0

        query = Content(role="user", parts=query_parts)
        fixed_temperature = UNIT_COUNT_EXTRACTION_TEMPERATURE
        attempts = 0
        last_error = None

        while attempts < UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS:
            attempts += 1
            try:
                if is_cached:
                    extraction, _ = phoenix_call(
                        # `temperature` from phoenix_call is deliberately unused:
                        # fixed temperature, no escalation (item 7).
                        lambda feedback_prompt, temperature: vertex_ai_client.generate_content(
                            contents=[feedback_prompt, query] if feedback_prompt else [query],
                            generation_config={**generation_config, "temperature": fixed_temperature},
                        ),
                        max_retry=1,
                        pydantic_model=UnitCountResponse,
                    )
                else:
                    extraction, _ = phoenix_call(
                        lambda feedback_prompt, temperature: vertex_ai_client(UNIT_COUNT_EXTRACTOR).generate_content(
                            contents=[feedback_prompt, query] if feedback_prompt else [query],
                            generation_config={**generation_config, "temperature": fixed_temperature},
                        ),
                        max_retry=1,
                        pydantic_model=UnitCountResponse,
                    )
                logging.info(
                    f"[UNIT_COUNTS] [{context}] extraction attempt {attempts}/"
                    f"{UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS} OK (temperature={fixed_temperature}, fixed)"
                )
                return extraction, attempts
            except Exception as e:
                last_error = e
                logging.warning(
                    f"[UNIT_COUNTS] [{context}] extraction attempt {attempts}/"
                    f"{UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS} failed (temperature={fixed_temperature}, fixed): {e}"
                )

        logging.warning(
            f"[UNIT_COUNTS] [{context}] extraction EXHAUSTED after {attempts} attempt(s); "
            f"last error: {last_error}"
        )
        return None, attempts
    finally:
        doc.close()


def _page_text_layer(pdf_path, page_numbers):
    """Concatenated text layer of `page_numbers`, or None when there is none.

    Cheap fitz read used only by the sanity check. Returns None for a scanned
    page set (no usable text), which the caller treats as "cannot verify".
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None
    try:
        chunks = []
        for page_index in page_numbers:
            try:
                chunks.append(doc.load_page(page_index).get_text("text") or "")
            except Exception:
                continue
        text = "\n".join(chunks)
        return text if len(text.strip()) >= _UNIT_COUNT_MIN_PAGE_TEXT_CHARS else None
    finally:
        doc.close()


def _looks_fabricated(extraction, pdf_path, source_pages, context):
    """Sanity check for the Aurora failure shape (item 7).

    Rejects an accepted answer when BOTH hold:
      1. every per-type count is the SAME round number (>= 2 types, value a
         multiple of 5) — Aurora returned STUDIO/1BED/2BED/3BED = 10/10/10/10
         against a true 8/18/8/11; and
      2. NONE of the extracted unit-type labels appears in the page's text layer.

    Condition 2 is checked ONLY for vector PDFs. If the pages have no usable text
    layer (scanned — the majority of real MF sets, master plan section 12) the check
    CANNOT be performed and the answer is accepted; a scanned set has no cheap
    ground truth to test against and failing closed there would break every
    legitimate scanned extraction.

    Returns (is_suspect, reason).
    """
    counts = [t.count for t in extraction.per_type_counts]
    if len(counts) < 2:
        return False, None
    if len(set(counts)) != 1:
        return False, None
    value = counts[0]
    if value % 5 != 0:
        return False, None

    pages = source_pages or []
    text = _page_text_layer(pdf_path, pages)
    if text is None:
        logging.info(
            f"[UNIT_COUNTS] [{context}] sanity check: uniform round counts "
            f"({len(counts)} types all ={value}) but pages {pages} have no text layer "
            f"(scanned) — cannot verify, ACCEPTED"
        )
        return False, None

    haystack = re.sub(r"[^a-z0-9]+", " ", text.lower())
    for unit_type in extraction.per_type_counts:
        label = re.sub(r"[^a-z0-9]+", " ", (unit_type.unit_type or "").lower()).strip()
        if label and label in haystack:
            return False, None

    reason = (
        f"all {len(counts)} extracted types share the identical round count {value}, "
        f"and none of the extracted type labels "
        f"({', '.join(t.unit_type for t in extraction.per_type_counts)}) appears in the "
        f"text layer of page(s) {pages}"
    )
    return True, reason


def _persist_unit_counts(credentials, pg_pool, project_id, plan_id, payload, context):
    """Write the resolved unit-counts payload to plans.unit_counts (JSONB).

    Lifted from feat/multifamily helper.py:1950-1969, unchanged except that the
    docstring no longer claims a detached task and the log line reports attempts.
    """
    t_persist = perf_counter()
    table_name_plans = credentials["CloudSQL"]["table_name_plans"]
    query = (
        f"UPDATE {table_name_plans} SET unit_counts = %s "
        f"WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s);"
    )
    try:
        pg_run(credentials, pg_pool, query, params=(json.dumps(payload), project_id, plan_id))
        logging.info(
            f"[UNIT_COUNTS] [{context}] persisted unit_counts (source={payload['source']}, "
            f"attempts={payload.get('provenance', {}).get('attempts')}) "
            f"in {perf_counter() - t_persist:.3f}s"
        )
    except Exception as e:
        logging.warning(f"[UNIT_COUNTS] [{context}] persist FAILED: {e}")


def _provenance(source_form, source_pages, candidate_pages, detection_path, extraction_dpi,
                disagreement, attempts, flagged_suspect, suspect_reason=None):
    """Provenance block stored inside plans.unit_counts.

    ADDED vs feat/multifamily (item 7): `attempts` and `flagged_suspect` — the
    extraction-health signal. An answer from the first call and one that only
    appeared on the last permitted attempt must not look identical in the DB.
    """
    return {
        "source_form": source_form,
        "source_pages": source_pages,
        "candidate_pages": candidate_pages,
        "detection_path": detection_path,
        "extraction_dpi": extraction_dpi,
        "disagreement": disagreement,
        "attempts": attempts,
        "flagged_suspect": flagged_suspect,
        "suspect_reason": suspect_reason,
    }


def _none_found_payload(detection_path, candidate_pages, extraction_dpi, attempts=0):
    """Empty map written when no counts are found anywhere (sections -> x1)."""
    return {
        "source": "none_found",
        "unit_counts": [],
        "total_units": None,
        "provenance": _provenance(
            "none_found", [], candidate_pages, detection_path, extraction_dpi,
            None, attempts, False,
        ),
    }


def _extraction_failed_payload(detection_path, candidate_pages, extraction_dpi, attempts,
                               suspect_reason=None):
    """BUG FIX 1 (item 7) — the honest failure state.

    BEFORE: a failed extraction fell through to `_none_found_payload`, which is
    indistinguishable from "the document genuinely has no counts". Worse, the
    retry spiral that preceded it could instead return a fabricated answer stored
    with full apparent confidence (source_form=table, disagreement=null).

    AFTER: extraction failure (or a sanity-check rejection) is stored as its own
    state, `source="extraction_failed"`, carrying the attempt count. Downstream
    behaviour is the SAME as none_found — unit_counts is empty, so every section
    defaults x1 (D13) — but the DB and the logs now say WHY. A made-up answer is
    never written.
    """
    return {
        "source": "extraction_failed",
        "unit_counts": [],
        "total_units": None,
        "provenance": _provenance(
            None, [], candidate_pages, detection_path, extraction_dpi,
            None, attempts, bool(suspect_reason), suspect_reason,
        ),
    }


def resolve_unit_counts(
    credentials,
    pg_pool,
    client_ip_address,
    pdf_path,
    project_id,
    plan_id,
    high_dpi=UNIT_COUNT_EXTRACTION_DPI,
    max_candidates=UNIT_COUNT_MAX_CANDIDATES,
):
    """Resolver core: detect -> extract -> sanity-check -> resolve -> persist.

    Lifted from feat/multifamily helper.py:1989-2081. Structure is unchanged;
    what is new is the extraction-health plumbing (attempts), the
    extraction_failed state, the sanity check, and the full per-type log line
    (D18 / master plan section 9).

    Sync/blocking by design — main.py runs it in a worker thread. Never raises.
    """
    context = f"project={project_id} plan={plan_id}"
    t0 = perf_counter()
    logging.info(f"[UNIT_COUNTS] [{context}] resolver STARTED")

    # 1. Detection: vector first; fall back to scanned triage.
    vector_result = rank_unit_count_pages_vector(pdf_path, project_id, plan_id)
    scanned_result = None
    if not vector_result["applicable"]:
        scanned_result = triage_unit_count_pages_scanned(
            credentials, client_ip_address, pdf_path, project_id, plan_id
        )
    candidate_pages, detection_path = _unify_candidate_pages(vector_result, scanned_result, max_candidates)
    logging.info(
        f"[UNIT_COUNTS] [{context}] detection path={detection_path} candidates={candidate_pages}"
    )

    # 2. Nothing found anywhere -> empty map, source none_found (loud).
    if not candidate_pages:
        logging.warning(
            f"[UNIT_COUNTS] [{context}] NONE-FOUND (path={detection_path}); writing empty map "
            f"source=none_found — unmatched sections default x1"
        )
        payload = _none_found_payload(detection_path, candidate_pages, None)
        _persist_unit_counts(credentials, pg_pool, project_id, plan_id, payload, context)
        logging.info(f"[UNIT_COUNTS] [{context}] resolver DONE (none_found) in {perf_counter() - t0:.3f}s")
        return payload

    # 3. Bounded, fixed-temperature extraction on the top candidates.
    logging.info(
        f"[UNIT_COUNTS] [{context}] extraction phase — path={detection_path} "
        f"candidates={candidate_pages} dpi={high_dpi} "
        f"max_attempts={UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS} "
        f"temperature={UNIT_COUNT_EXTRACTION_TEMPERATURE} (fixed)"
    )
    t_extract = perf_counter()
    extraction, attempts = _extract_unit_counts(
        credentials, client_ip_address, pdf_path, candidate_pages, high_dpi, context
    )

    # 3a. Every attempt failed -> extraction_failed. NEVER a made-up answer.
    if extraction is None:
        logging.warning(
            f"[UNIT_COUNTS] [{context}] extraction FAILED after {attempts} attempt(s); "
            f"writing source=extraction_failed — sections default x1 (no answer invented)"
        )
        payload = _extraction_failed_payload(detection_path, candidate_pages, high_dpi, attempts)
        _persist_unit_counts(credentials, pg_pool, project_id, plan_id, payload, context)
        logging.info(f"[UNIT_COUNTS] [{context}] resolver DONE (extraction_failed) in {perf_counter() - t0:.3f}s")
        return payload

    # 3b. BUG FIX 2 (item 8) — a legitimately empty read is a VALID result.
    # It reached us as a normal validated response (not an exception), cost one
    # call, and is recorded as none_found rather than burning retries.
    if extraction.is_none_found:
        logging.info(
            f"[UNIT_COUNTS] [{context}] extraction returned a legitimate EMPTY result "
            f"(no counts on pages {candidate_pages}) after {attempts} attempt(s); "
            f"source=none_found — sections default x1"
        )
        payload = _none_found_payload(detection_path, candidate_pages, high_dpi, attempts)
        _persist_unit_counts(credentials, pg_pool, project_id, plan_id, payload, context)
        logging.info(f"[UNIT_COUNTS] [{context}] resolver DONE (none_found) in {perf_counter() - t0:.3f}s")
        return payload

    logging.info(
        f"[UNIT_COUNTS] [{context}] extraction OK in {perf_counter() - t_extract:.3f}s — "
        f"source_form={extraction.source_form} types={len(extraction.per_type_counts)} "
        f"total={extraction.total_units} pages={extraction.source_pages} attempts={attempts}"
    )

    # 3c. Sanity check (item 7): reject the Aurora fabrication shape.
    is_suspect, suspect_reason = _looks_fabricated(extraction, pdf_path, extraction.source_pages, context)
    if is_suspect:
        logging.warning(
            f"[UNIT_COUNTS] [{context}] SANITY CHECK REJECTED the extracted answer — {suspect_reason}. "
            f"Rejected counts were: "
            f"{ {t.unit_type: t.count for t in extraction.per_type_counts} }. "
            f"Writing source=extraction_failed — sections default x1 (no answer invented)"
        )
        payload = _extraction_failed_payload(
            detection_path, candidate_pages, high_dpi, attempts, suspect_reason=suspect_reason
        )
        _persist_unit_counts(credentials, pg_pool, project_id, plan_id, payload, context)
        logging.info(f"[UNIT_COUNTS] [{context}] resolver DONE (extraction_failed/suspect) in {perf_counter() - t0:.3f}s")
        return payload

    # 4. D10 precedence -> one resolved count per type.
    resolved, disagreement = _apply_precedence(extraction, context)
    payload = {
        **resolved,
        "provenance": _provenance(
            extraction.source_form, extraction.source_pages, candidate_pages,
            detection_path, high_dpi, disagreement, attempts, False,
        ),
    }

    # 5. D18 / section 9 — log the FULL per-type breakdown, not just an aggregate.
    # The Aurora fabrication went undetected partly because only "types=4 total=40"
    # was ever logged.
    breakdown = {entry["unit_type"]: entry["count"] for entry in resolved["unit_counts"]}
    logging.info(
        f"[UNIT_COUNTS] [{context}] resolved: {breakdown} "
        f"total={resolved['total_units']} attempts={attempts}"
    )

    # 6. Persist.
    _persist_unit_counts(credentials, pg_pool, project_id, plan_id, payload, context)
    logging.info(
        f"[UNIT_COUNTS] [{context}] resolver DONE source={resolved['source']} "
        f"types={len(resolved['unit_counts'])} total={resolved['total_units']} "
        f"disagreement={'yes' if disagreement else 'no'} attempts={attempts} "
        f"in {perf_counter() - t0:.3f}s"
    )
    return payload
