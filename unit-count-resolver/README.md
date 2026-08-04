# unit-count-resolver

Multi-family Unit-Count Resolver — a standalone Cloud Run service (master plan **D5a**).

Given a plan set already uploaded to GCS, it works out how many units of each type
the project contains and writes the answer to `plans.unit_counts`. Downstream,
`summarize_takeoff_all` in `xtimator-3d` matches each measured section to a unit
type and multiplies (D12); anything unmatched stays ×1 (D13), so single-family
projects are completely unaffected.

## Endpoint

```
POST /resolve_unit_counts
{"project_id": "...", "plan_id": "...", "user_id": "..."}
```

Returns **202 immediately**, then resolves in the background. Callers are
fire-and-forget and never depend on the outcome.

`GET /healthz` reports ready once the Cloud SQL pool is up.

## What it does

1. **Detect** candidate pages.
   - *Vector PDFs*: rank pages by a text-layer heuristic read straight from
     `fitz.get_text()`. **Zero Gemini calls.**
   - *Scanned PDFs*: render low-DPI thumbnails and run `UNIT_COUNT_DETECTOR`
     over them in batches. Batches are **strictly serialized** with a backoff
     between calls (D5c) — do not parallelise them.
2. **Extract** from the top 1–3 candidates at high DPI with
   `UNIT_COUNT_EXTRACTOR`: **max 2 attempts at a fixed temperature**, renders
   **pixel-capped**.
3. **Resolve** via D10 precedence (table > prose > none) to exactly one count per
   unit type, then **persist** with provenance.

Every run logs under `[UNIT_COUNTS]`, including the full per-type breakdown
(`resolved: {...} total=N attempts=N`) — an aggregate-only log is what let the
Aurora fabrication go unnoticed (master plan §9).

## Result states

`plans.unit_counts.source` is one of:

| source | meaning | multiplier effect |
|---|---|---|
| `table` | per-type counts read from a schedule/matrix | scales matched sections |
| `prose` | narrative total only, no per-type breakdown | no per-type scaling |
| `none_found` | no counts in the document, or extraction legitimately reported none | every section ×1 |
| `extraction_failed` | every attempt failed, or the answer was rejected by the sanity check | every section ×1 |

`extraction_failed` is deliberately distinct from `none_found`: a **failure is
never dressed up as an answer**, and never as a made-up count. `provenance`
carries `attempts` and `flagged_suspect` so a first-call answer is
distinguishable from a last-chance one.

## Deploy

Via `.github/workflows/unit_count_resolver.yml` (manual dispatch). The first
`gcloud run deploy` creates the service.

**Two things that are not optional:**

- **`config/gcp.yaml` must be staged into the build context before `docker build`.**
  It is not in git (it carries service-account key paths and the Azure secret) —
  same convention as `xtimator-page-classifier`. The Dockerfile fails the build if
  it is missing.
- **`--no-cpu-throttling` must stay in the deploy flags.** The service returns 202
  and does its work in a background task; with Cloud Run's default CPU throttling
  that work would freeze the instant the response is sent.

The caller side needs `UNIT_COUNT_RESOLVER_URL` set on the **drywall-takeoff-3d**
service, pointing at this service's URL. Use `--update-env-vars`, never
`--set-env-vars` (the latter wipes the whole env set).

## Configuration

All tunables are env vars with working defaults — see `app/config.py`. The ones
that matter: `UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS` (2),
`UNIT_COUNT_EXTRACTION_TEMPERATURE` (0),
`UNIT_COUNT_MAX_RENDER_LONGEST_SIDE` (4000),
`UNIT_COUNT_TRIAGE_BACKOFF_SECONDS` (2.0).

`MF_RESOLVER_ENABLED=false` on **drywall-takeoff-3d** disables the trigger
without touching this service.
