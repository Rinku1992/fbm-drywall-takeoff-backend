"""
vector_extract.py — vector-PDF detection + per-page scale extraction (single file).

Contains everything for the vector scale path:
  • the robust architectural-scale parser (canonical output, e.g. 0.25``:1`0``)
  • logging/timing helpers ([VECTOR] / [VECTOR_SCALE] prefixes, Cloud-Run friendly)
  • is_vector detection and parallel per-page scale extraction

Public API (imported by the orchestrator):
    detect_is_vector(pdf_path, project_id, plan_id) -> bool
    extract_scales_for_pages(pdf_path, page_numbers, project_id, plan_id,
                             max_workers=8) -> dict[int, str | None]

All page numbers are 0-indexed (matches the rest of the repo and fitz). The
canonical scale format is assumed to match what `normalize_scale` expects; Phase
2.5 verifies this before anything is wired in.
"""

import re
import logging
import time
from fractions import Fraction
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF

logger = logging.getLogger("vector_scale")

# ─────────────────────────── scale parser ────────────────────────────────────
# Matches the STRUCTURE  <ratio>" = <feet>'-<inches>"  (not a fixed string list).
# Handles fractional / whole / mixed ratios; optional 'Scale:' label; tight or
# loose spacing; missing dash; FEET-ONLY (`1/4" = 1'` -> inches 0); curly quotes
# and prime marks. Emits the codebase canonical form, e.g. 1/4" = 1'-0" -> 0.25``:1`0``

_INCH = r'["“”″]'
_FOOT = r"['‘’′]"

SCALE_RE = re.compile(
    rf"""
    (?P<left>
        \d+
        (?:\s*/\s*\d+)?
        (?:\s+\d+\s*/\s*\d+)?
    )
    \s* {_INCH}? \s* = \s*
    (?P<feet>\d+) \s* {_FOOT}
    (?:
        [^\S\n]* -? [^\S\n]*
        (?P<inches>\d+)
        [^\S\n]* {_INCH}?
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_LABEL_RE = re.compile(r"scale", re.IGNORECASE)


def _paper_decimal(left: str) -> float:
    """'1/4' -> 0.25, '3/16' -> 0.1875, '1 1/2' -> 1.5, '1' -> 1.0."""
    left = re.sub(r"\s*/\s*", "/", left.strip())
    left = re.sub(r"\s+", " ", left)
    if " " in left:                      # mixed number, e.g. '1 1/2'
        whole, frac = left.split(" ", 1)
        return float(Fraction(whole)) + float(Fraction(frac))
    return float(Fraction(left))


def to_canonical(left: str, feet: str, inches) -> str:
    """Build the codebase canonical scale string, e.g. 0.25``:1`0``"""
    paper_str = ("%g" % _paper_decimal(left))   # 0.25, 0.1875, 1, 1.5 (no trailing zeros)
    inches = inches if inches is not None else "0"
    return f"{paper_str}``:{feet}`{inches}``"


def find_scales(text: str):
    """Return list of dicts: {canonical, raw, has_label, span}, best-first."""
    if not text:
        return []
    results = []
    for m in SCALE_RE.finditer(text):
        canonical = to_canonical(m.group("left"), m.group("feet"), m.group("inches"))
        look_back = text[max(0, m.start() - 20): m.start()]
        results.append({
            "canonical": canonical,
            "raw": m.group(0).strip(),
            "has_label": bool(_LABEL_RE.search(look_back)),
            "span": m.span(),
        })
    seen, deduped = set(), []
    for r in sorted(results, key=lambda r: (not r["has_label"], r["span"][0])):
        if r["canonical"] not in seen:
            seen.add(r["canonical"])
            deduped.append(r)
    return deduped


def best_scale(text: str):
    """Return the single most likely canonical scale string, or None."""
    found = find_scales(text)
    return found[0]["canonical"] if found else None


# ─────────────────────────── ceiling parser ──────────────────────────────────
# Match only the word CEILING next to a height (e.g. 9' CEILING).
_CEILING_KEYWORD = r"CEILING"
CEILING_RE = re.compile(
    rf"""
    (?P<feet>\d+)\s*{_FOOT}                       # feet, e.g. 9'
    (?:\s*-?\s*(?P<inches>\d+)\s*{_INCH})?         # optional -I"
    \s*
    {_CEILING_KEYWORD}
    """,
    re.VERBOSE | re.IGNORECASE,
)
# Non-numeric ceiling markers => not a simple single height => defer to LLM.
# Only "VAULTED/SLOPED CEILING" (adjacent) or "OPEN TO BELOW" defer — a bare
# "SLOPE" elsewhere (e.g. "SLAB ON GRADE SLOPE") must NOT trigger deferral.
_VAULT_RE = re.compile(r"(?:VAULT(?:ED)?|SLOPED?)\s+CEILING|OPEN\s+TO\s+BELOW", re.IGNORECASE)


def _ceiling_feet(feet, inches):
    return round(int(feet) + (int(inches) / 12.0 if inches else 0.0), 3)


def best_ceiling(text: str):
    """
    Return the page's ceiling height in feet (float), or None if not confidently
    a single value. Rule: most-frequent wins; tie -> max. If a vaulted/sloped/
    open-to-below marker is present, return None (defer to the LLM).
    """
    if not text:
        return None
    if _VAULT_RE.search(text):
        return None
    values = [_ceiling_feet(m.group("feet"), m.group("inches"))
              for m in CEILING_RE.finditer(text)]
    if not values:
        return None
    counts = Counter(values)
    top = max(counts.values())
    tied = [v for v, c in counts.items() if c == top]
    return max(tied)   # most-frequent; tie -> max


# ─────────────────────── logging / timing helpers ────────────────────────────
def ctx(project_id, plan_id, page_number=None):
    """Consistent context string for log correlation."""
    base = f"project={project_id} plan={plan_id}"
    return base if page_number is None else f"{base} page={page_number}"


@contextmanager
def timed(prefix: str, context: str, message: str):
    """
    Logs start + elapsed wall-clock time. The 'done in Xs' line is what to grep
    in Cloud Run for timing, e.g.:
        [VECTOR_SCALE] [project=.. plan=.. page=1] extract — done in 0.031s
    """
    t0 = time.perf_counter()
    logger.info(f"{prefix} [{context}] {message} — started")
    try:
        yield
    except Exception:
        logger.exception(f"{prefix} [{context}] {message} — FAILED after "
                         f"{time.perf_counter() - t0:.3f}s")
        raise
    else:
        logger.info(f"{prefix} [{context}] {message} — done in "
                    f"{time.perf_counter() - t0:.3f}s")


# ─────────────────────── vector detection (fast, text-first) ─────────────────
# "Vector" = the PDF has an extractable text layer. That is the cheap, reliable
# signal: get_text() returns plenty of text for vector PDFs and ~nothing for
# scanned ones, in milliseconds. We sample the first few pages (not only page 0,
# since a cover / site / survey page can be image-only in an otherwise vector
# set) and short-circuit as soon as one page clears the text threshold.
VECTOR_TEXT_MIN_CHARS = 100   # a real drawing's text layer easily exceeds this
VECTOR_SAMPLE_PAGES = 3       # max pages to sample before deciding


def is_vector(pdf_path, project_id, plan_id) -> bool:
    """True if the PDF has an extractable text layer (fast, text-first)."""
    context = ctx(project_id, plan_id)
    with timed("[VECTOR]", context, "is_vector detection"):
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.warning(f"[VECTOR] [{context}] open failed: {e}; "
                           f"defaulting is_vector=False")
            return False
        try:
            n = doc.page_count
            if n == 0:
                logger.info(f"[VECTOR] [{context}] pages=0 => is_vector=False")
                return False
            sample = min(VECTOR_SAMPLE_PAGES, n)
            max_text = 0
            for i in range(sample):
                text_len = len((doc.load_page(i).get_text("text") or "").strip())
                max_text = max(max_text, text_len)
                if text_len >= VECTOR_TEXT_MIN_CHARS:
                    logger.info(f"[VECTOR] [{context}] pages={n} sampled={i + 1} "
                                f"text_chars={text_len} => is_vector=True")
                    return True   # short-circuit; no need to check more pages
            logger.info(f"[VECTOR] [{context}] pages={n} sampled={sample} "
                        f"max_text_chars={max_text} => is_vector=False")
            return False
        finally:
            doc.close()


def _extract_one(pdf_path, page_number, project_id, plan_id):
    """Open the PDF, read one page's text, parse scale + ceiling.
    Returns (page, {"scale": ..., "ceiling_height": ...})."""
    context = ctx(project_id, plan_id, page_number)
    try:
        with timed("[VECTOR_SCALE]", context, "extract"):
            doc = fitz.open(pdf_path)
            try:
                text = doc.load_page(page_number).get_text("text") or ""
            finally:
                doc.close()
            scale = best_scale(text)
            ceiling = best_ceiling(text)
            logger.info(f"[VECTOR_SCALE] [{context}] scale={scale!r} "
                        f"ceiling_height={ceiling!r}")
            return page_number, {"scale": scale, "ceiling_height": ceiling}
    except Exception as e:
        logger.warning(f"[VECTOR_SCALE] [{context}] extraction failed: {e}; "
                       f"page will fall back to existing flow")
        return page_number, {"scale": None, "ceiling_height": None}


def extract_scales_for_pages(pdf_path, page_numbers, project_id, plan_id, max_workers=8):
    """
    Extract scale + ceiling for each (0-indexed) page in parallel.
    Returns {page_number: {"scale": str|None, "ceiling_height": float|None}}.
    Never raises.
    """
    context = ctx(project_id, plan_id)
    results = {}
    if not page_numbers:
        logger.info(f"[VECTOR_SCALE] [{context}] no floor-plan pages to process")
        return results
    with timed("[VECTOR_SCALE]", context,
               f"scale extraction for floor_plan pages {sorted(page_numbers)}"):
        workers = min(max_workers, len(page_numbers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_extract_one, pdf_path, pn, project_id, plan_id)
                       for pn in page_numbers]
            for fut in as_completed(futures):
                page_number, info = fut.result()
                results[page_number] = info
    found_scale = sum(1 for v in results.values() if v["scale"])
    found_ceiling = sum(1 for v in results.values() if v["ceiling_height"] is not None)
    logger.info(f"[VECTOR_SCALE] [{context}] summary: scale {found_scale}/"
                f"{len(page_numbers)}, ceiling {found_ceiling}/{len(page_numbers)}")
    return results


# ─────────────────────────── self-test ───────────────────────────────────────
if __name__ == "__main__":
    SAMPLES = [
        ('GROUND FLOOR PLAN\nSCALE:  1/4" = 1\'\nA1', "0.25``:1`0``"),
        ('SCALE: 1/8"=1\'0"', "0.125``:1`0``"),
        ('3/16" = 1\'-0"', "0.1875``:1`0``"),
        ('1 1/2" = 1\'-0"', "1.5``:1`0``"),
        ('29\'-0" dimension only', None),   # must NOT match (no '=')
    ]
    ok = 0
    for txt, exp in SAMPLES:
        got = best_scale(txt)
        ok += (got == exp)
        print(f"[{'PASS' if got==exp else 'FAIL'}] {txt!r:40s} -> {got!r} (exp {exp!r})")
    print(f"{ok}/{len(SAMPLES)} passed")

    from collections import Counter as _C  # ensure import present
    assert best_ceiling("LIVING 10' CEILING KITCHEN 10' CEILING BED 9' CEILING") == 10.0
    assert best_ceiling("BED 9' CEILING BED 9' CEILING LIVING 10' CEILING") == 9.0
    assert best_ceiling("A 9' CEILING B 10' CEILING") == 10.0           # 1-1 tie -> max
    # bare "SLOPE" (slab note) must NOT defer; real ceilings still win
    assert best_ceiling("4\" CONCRETE SLAB ON GRADE SLOPE ... 9' CEILING 9' CEILING") == 9.0
    assert best_ceiling("GREAT ROOM VAULTED CEILING 9' CEILING") is None  # vaulted ceiling -> defer
    assert best_ceiling("just 29'-0\" dimensions") is None
    print("ceiling checks passed")
