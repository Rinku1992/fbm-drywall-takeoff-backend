# Multi-Family Drywall Estimation — MASTER PLAN
**Purpose of this file:** the locked, end-to-end plan for extending Xtimator's single-family drywall pipeline to multi-family (MF) residential projects. This file changes only when a decision is formally revisited. Day-to-day progress lives in `MF_CURRENT_STATUS.md`.

**Last updated:** 2026-08-03

---

## 1. Objective

Given an MF residential plan set (PDF), produce:
1. The **total** drywall / gypsum board quantity for the project.
2. A **breakdown** at Building → Floor → Unit level.

## 2. Scope of v1 (locked with the team — read this first)

v1 is **multiplication only**. We deliver the **same 2D model, 3D model, and takeoff** the product produces today, **correctly scaled to the real number of units**, with an aggregated total.

v1 explicitly does **NOT** include:
- A navigable 3D building the user clicks through.
- Per-floor / per-unit drill-down in the UI.
- Cross-unit drywall design transfer.
- A stored building hierarchy schema (buildings → floors → units) as a product feature.

The richer hierarchy vision (navigable building, per-level rollups, cross-unit editing) was discussed with the team and **confirmed out of scope** for now — the team agreed "just the multiplication." That vision remains a possible future direction, not a v1 commitment. (Note: an experimental `RESIDENTIAL_MULTI_FAMILY_SCHEMA` hierarchy-inference prompt exists in `prompts.py` from that exploration; it is **not** part of the v1 path and can be repurposed or removed as the pipeline needs.)

## 3. The core logic (one formula, every project)

Architects draw each unique unit type once and repeat it. Therefore:

```
Project drywall = Σ over unique unit types:
                  (drywall of ONE unit of that type)
                  × (units of that type per floor)
                  × (identical floors)
                  × (identical buildings)
```

Repetition composes at three levels (unit × floor × building). All MF projects share this logic; they differ only in **where the counts are written** and **how hard they are to read**. There are no separate "categories" needing separate logic.

Two jobs follow:
1. **Measure each unique unit type once** — the existing pipeline already does this (an enlarged unit plan looks like a single-family plan).
2. **Get the multipliers right** — read how many of each unit type exist, and multiply. This is the core new thing.

## 4. Agreed end-to-end flow (team-approved, plain language)

**Create project (project type chosen here).** The user creates a project and selects **multi-family or single-family**. This selector **already exists on the frontend** — no new frontend work. Single-family runs the current pipeline untouched; multi-family switches on the counting step below.

**Upload & page classification — UNCHANGED.** User uploads the plan set; pages are classified and shown exactly as today (same ConvNeXt classifier, same tiles, same selection).

**Measuring each unit — UNCHANGED.** User selects the unit-plan pages; each runs the existing steps: sections → bounding boxes → scale → wall detection → 2D extraction.

**Counting the units — NEW.** The backend reads the counts from the document; the user never types them. Runs off the request path so the page screen is not slowed. Pass 1: cheap detection over page thumbnails ("does this page have counts?" — table, cover summary, or prose). Pass 2: high-quality read of only the pages that passed, extracting each unit type, its count, and its area. Reads from the page image, never from parsed text (parsing shreds table layout).

**Storing counts — NEW.** Saved against the plan: each type, count, area, and where the number came from. Computed once, reused. If a table and a written total disagree, the table wins and the disagreement is logged for a human.

**Multiplying at takeoff — NEW, sits on the existing step.** At the roll-up (`summarize_takeoff_all`), each measured section is matched to a unit type — by name first, by area when names don't line up. Matched: drywall × count. Unmatched: × 1 (the safety line — single-family and non-MF plans behave exactly as today). The per-section takeoff calculation itself is unchanged; the multiply sits on top at roll-up.

**Traceability.** Every run reports how it got its counts — table, prose, or none-found (×1). No silent failures.

## 5. Current pipeline (unchanged backbone)

Step 1 PDF upload → Step 2 render pages to PNG → Step 3 ConvNeXt page classification (6 buckets, output goes to frontend) → Step 4 Gemini bounding boxes/sections per page (each box has a title = `page_section_number`) → per section: Step 5 scale+ceiling (vector text first, LLM fallback) → Step 6 wall detector + Vision transcriber → Step 7 Gemini wall mapping → Step 8 `/compute_takeoff` per section → `/summarize_takeoff_all` project roll-up.

Everything downstream is keyed by `(page_number, page_section_number)`. **Sections are already the unit mechanism.**

## 6. Locked decisions (with rationale)

| # | Decision | Rationale |
|---|---|---|
| D1 | Measure units ONLY from enlarged unit plans; never wall-measure overall floor plans (v1) | Avoids double counting; avoids light/thin interior lineweights the wall detector may miss; detector stays on drawings it is good at |
| D2 | ConvNeXt classifier, its 6 buckets, and frontend contract are UNTOUCHED | Output goes to frontend; retraining needs labeled MF data we don't have; user selects pages in the UI as today |
| D3 | No NEW frontend work in Phase 1 | The MF/SF project-type selector already exists on the frontend (see D4a); count entry stays backend-side. "No frontend changes" means no new build, not "no frontend involvement" |
| D4 | Unit counts are resolved by the BACKEND (Unit-Count Resolver), not entered by the user | Documents almost always state the counts; keeps count entry off the UI |
| D4a | Project type (MF vs SF) is selected by the USER at project creation, using the EXISTING frontend selector | User knows the type; explicit selection is more reliable than backend inference; requires no new frontend work; lets SF projects bypass MF code entirely |
| D5 | **SUPERSEDED by D5a/D5b/D5c — see §7.** Original detached-background-task placement tested and failed. Replacement DECIDED: separate `unit-count-resolver` Cloud Run service, triggered after classification+bounding-boxes persist, with a bounded/serialized LLM footprint | Detached task died with its instance; in-process placement shared Vertex quota with the sectioning call during the automatic burst (§9a) |
| D6 | Resolver is self-contained: input = the PDF only; no dependency on classifier output | Avoids service coupling and timing dependencies (still holds under any D5 replacement) |
| D7 | Two-prompt funnel: Prompt 1 = DETECTION only (low-res thumbnails, batched, yes/no per page), Prompt 2 = EXTRACTION only (high-res render of hit pages, schema-validated) | Detection needs only thumbnails (cheap, scales with document); extraction needs high res but only for 1–3 pages (scales with answer, not document). One combined prompt would force high-res on all pages — slow, costly, less reliable |
| D8 | Detection looks for unit-count information in ANY form: tables, prose/project descriptions, legends — not tables only | Real files prove it: Humboldt's count is a prose sentence; Aurora/Xanthia state totals in text alongside tables |
| D9 | Extraction reads from page IMAGES via vision, never from parsed text | Text extraction destroys table structure (proved on Grace Commons — OCR garbled Name/Count columns); vision reads the 2D layout intact |
| D10 | Extraction schema admits partial truth (per-type counts, total_units, source_form (table/prose/mixed), source_pages). **Resolved-count precedence: table > prose > none** — if a table exists, its per-type counts win even when a prose total also exists and disagrees; if there is no table but prose counts exist, use prose; if neither, no count and the section defaults ×1 (D13). The multiplier always receives exactly ONE resolved count per unit type, never two. When a table and prose both exist and disagree, the table wins the resolved value but the disagreement is LOGGED under `[UNIT_COUNTS]`; the two source values live ONLY in logs, never in the resolved output the multiplier consumes. `source_form` records which source won. | Prose-only documents give totals without per-type breakdown; store the honest partial rather than nothing. Downstream must multiply against a single unambiguous count per type, so precedence is fixed rather than reconciled; a table-vs-prose discrepancy is a real document signal surfaced to a human via logs — never passed to the multiplier and never retried away by `phoenix_call` |
| D11 | Counts stored in new column `plans.unit_counts JSONB DEFAULT '{}'` with provenance | Plan-level data; mirrors existing `multipage_elevation_map` precedent; provenance makes results diagnosable |
| D12 | Multiplication happens in `summarize_takeoff_all`: normalize each section title → match to unit type → takeoff × count → sum | It is already the "sum across sections" function; extraction is never gated by counts |
| D13 | Unmatched sections default to ×1 | Single-family and non-MF plans behave EXACTLY as today — the non-regression guarantee |
| D14 | Name matching: rule-based normalization first (strip filler words, expand BR/BA abbreviations), one batched LLM call for leftovers, AREA (sq ft) as disambiguator | Rules are fast/auditable/deterministic; LLM only where rules fail; area survives naming mismatches ("1 Bed Type Room" vs "1BR" vs codes) |
| D15 | Understanding only counts and labels; it NEVER decides what gets extracted | Wrong counts then cost only a re-multiplication (instant), never re-extraction; enables end-of-run correction |
| D16 | Count confirmation happens at END of run in the summary, not mid-run | User preference; safe because of D15 |
| D17 | Common areas (corridors, stairs, elevators, lobbies) in v1 = labeled configurable ALLOWANCE % (0% townhome-style, ~10–15% corridor-style), always shown as a separate line | Auto-carving corridors from dense/scanned floor plans is the hardest CV problem; user-drawn outlines are out of product scope; unit-side faces of corridor walls are already counted in unit plans so the miss is bounded |
| D18 | Every project reports which resolver path fired (provenance in summary + `[UNIT_COUNTS]` logs) — **must include the resolved per-type counts, not just an aggregate** | No silent failures; "no table found → ×1" is declared, not hidden. The Aurora fabrication went undetected partly because per-type counts were never logged (see §9) |
| D19 | Logging prefixes: `[UNIT_COUNTS]` (resolver phases with timings) and `[UNIT_MULTIPLY]` (per-section match + multiply), matching the repo's existing grep-friendly idiom | Requirement: extraction steps must be clearly traceable in logs |

## 7. Resolver execution model — DECIDED (replaces D5)

**D5 (detached in-process background task) was tested and proved unreliable — it is not coming back.** The replacement, decided 2026-08-04:

**D5a — Separate Cloud Run service** (`unit-count-resolver/`, top-level folder mirroring the existing per-service pattern like `xtimator-page-classifier/`). FastAPI, `POST /resolve_unit_counts {project_id, plan_id, user_id}`, fetches the PDF from GCS itself, returns 202 immediately, does the work, writes `plans.unit_counts`. Internal-only ingress + ID-token auth (same pattern as wall-detector). Own memory/timeout/lifecycle — solves the durability/contention problems that killed the detached task. Created by its own deploy workflow on first `gcloud run deploy` (no manual console setup expected).

**D5b — Sequenced trigger (the LLM-contention fix, part 1).** The backend fires the trigger **after page classification + bounding-box results are persisted for the plan** (the point where pages become ready for the frontend) — NOT at upload start. Rationale: bounding boxes run automatically during upload; the old resolver launched at upload start and overlapped exactly that burst — the likely page-20 mechanism (see §9a). Everything after the burst is user-triggered, so resolver overlap with the automatic phase is eliminated by construction. Fire-and-forget POST (short timeout, exceptions logged never raised — the upload flow must never depend on it). Trigger stays backend-driven (preserves D4).

**D5c — Rate-limited LLM footprint (the LLM-contention fix, part 2).** A separate service does NOT isolate Vertex quota (quota is per project+region+model, shared across services). So the resolver must be too small to matter when overlap happens: extraction = ONE Gemini call, max 2 attempts, fixed temperature (no escalation); scanned-path thumbnail triage batches SERIALIZED (one at a time, small backoff). Vector path's page ranking is a text scan — zero Gemini calls. Optional escape hatch if contention is ever proven to persist: point the resolver's Vertex calls at a different region (per-region quota = genuinely separate pool, one config value).

**Note:** D5a alone still loses work if the instance dies mid-run. Acceptable for now (re-trigger is cheap and the run is short once D5c bounds it); Cloud Tasks/Pub/Sub in front, or a Cloud Run Job, is the hardening step if it proves flaky in practice.

## 8. Accounting rules (correctness core)

- **Faces, not walls:** drywall attaches to a face; a shared wall = two faces. Each unit's enlarged plan shows only its own inside faces, so multiplying by count yields every face exactly once. No double counting by construction.
- **One drawing = one job:** enlarged plans are for measuring; overall floor plans are for counting. Never both.
- **C1 — positional variation:** the same unit type differs by position (middle unit = party walls both sides; end unit = exterior wall one side). Same face area, different assembly/layers. v1 may ship the drawn-condition simplification (documented); the positional side-wall template adjustment is a fast-follow.

## 9a. The page-20 mechanism — silent full-page fallback (discovered in code, 2026-08-04)

`xtimator-3d/helper.py` `detect_bounding_boxes` (the Gemini sectioning call, `VISUAL_GROUNDING_DETECTOR` prompt) and `classify_plan` both have a **silent failure fallback**: on ANY exception — including `phoenix_call` exhausting its retries — the `except` block returns **one full-page bounding box** (`offset_top_left=[0,0]`, `offset_bottom_right=[1,1]`, single box, empty title) instead of erroring. Consequence: a failed sectioning call on a multi-plan page degrades, without any error surfacing, to ONE section covering the whole page → one dropdown in the frontend, and the 2D model runs on the whole page as one section (walls look like garbage). This matches the Aurora page-20 symptom exactly (3 dropdowns on main, 1 on the feature branch, while the resolver was burning escalating Gemini retries in the same window — likely starving the sectioning call's quota). The fallback DOES log `SYSTEM: Bounding Box detection has failed` — verifiable in Cloud logs for any suspect run. **Scope:** this fallback is core-pipeline code — document it, verify it, do NOT modify it as part of MF work. The MF-side mitigation is D5b + D5c (don't overlap the burst; keep the resolver's LLM footprint tiny).

## 9. Correctness watch — count fabrication (learned on Aurora)

On the Aurora test the extraction pass, after repeated failures, returned a **fabricated** count (a tidy STUDIO/1BED/2BED/3BED = 10/10/10/10 = 40) in place of the true Vista types (8/18/8/11 = 45). It was stored with full apparent confidence (`source_form=table`, `disagreement=null`). This is challenge **C5** (wrong counts are the most expensive error) occurring live. Guardrails the design must enforce:
- **Log the resolved per-type counts** (D18), so a wrong answer is visible in logs, not only in the DB.
- **Record extraction health** — retry count / whether the answer came only after escalating-temperature retries — and flag a result as low-confidence when it did. An answer from the first call and one from a 4-retry death spiral must not look identical.
- The multiplier's ×1 safety net (D13) protects the *total* from a bad name match, but does **not** protect against a fabricated count that happens to match; extraction correctness is the real defense.

## 10. Phases

### Phase 1 — Aurora-class (table/prose counts + enlarged plans exist) ← CURRENT
Covers any project whose counts are stated somewhere in the document and whose unit types have enlarged/dedicated plans (Aurora, Grace Commons, Xanthia, Humboldt, Stok-with-tags…).
Build:
1. **Unit-Count Resolver** in `xtimator-3d`: text-layer structural scan (vector — via `fitz.get_text()` directly, **not** `vector_pdf.py`, which only parses scale/ceiling) / thumbnail vision triage (scanned) → high-res extraction call → validated `{type, count, area}` + total + provenance → `plans.unit_counts`. Detection and extraction are both **Gemini vision via `phoenix_call`**; the Cloud Vision **OCR** transcriber (`floorplan-to-structured-2d/transcriber.py`, a different service) is **not** reused. Rendering uses the resolver's **own DPI parameter**. **Execution model is OPEN (§7).**
2. **Matcher**: section-title normalization + count lookup (D14).
3. **Multiplier** in `summarize_takeoff_all` (D12, D13) + provenance in response (D18).
4. Logging (D19) + per-type count logging + extraction-health signal (§9). Migration: one new column (D11).
Touch list: `xtimator-3d/{helper.py, main.py, prompts.py}` + one DB column, plus whatever the §7 execution-model choice adds. Frontend, classifier, extraction services, `/compute_takeoff`: untouched.

### Phase 2 — Counts from drawings (no table anywhere)
For sets like 1643 Boulder: count unit TAGS on overall floor plans via vision (per-floor tag list → counts per type), floor multipliers from titles ("Levels 2–7" ⇒ ×6). Plugs into the resolver as the next rung; no redesign.

### Phase 3 — Richer accuracy
Positional side-wall adjustment (C1) as takeoff-template logic; fire-rated party-wall + shaft-wall template rows; measured common areas replacing the allowance (corridor/core carved as their own sections from overall plans); per-building rollups from site-plan composition.

### Phase 4 — Automation & hardening (optional/backlog)
Floor-Plan Subtyper (auto-tag OVERALL vs ENLARGED once user-confirmed labels accumulate); ConvNeXt 6→7 class retrain option; mapping validation UI.

## 11. Known challenges catalog

C1 positional side-wall variation (middle vs end) · C2 shared-wall double counting (solved by faces rule) · C3 common areas (allowance v1 → measured v2) · C4 scanned sets — all reads via OCR/vision, lower certainty · C5 wrong counts are the most expensive error (mitigate: per-type logging, extraction-health flag, provenance, end-of-run correction that only re-multiplies) · C6 mirrored units = same quantity, group with base type · C7 label mismatch between enlarged-plan names and floor-plan tags (normalize → area fallback → flag).

## 12. Reference PDF catalog (assessed)

| File | Kind | Text layer | Unit counts source | Difficulty |
|---|---|---|---|---|
| 108 Ocean View | Single-family baseline | Vector | N/A | Baseline |
| Aurora Westlake | Townhomes, 45 units, 4 types, dedicated sheets | Vector | Cover table + prose | Easy |
| Stok on Pearl | Corridor studios, enlarged plans + tagged floor plan | Scan | Tags (no table) | Medium |
| 2727 29th St BLDG A | Apartments, name-mismatched labels | Scan (rasterized) | No table found; tags + area matching | Hard |
| 1643 Boulder (IFC) | 3-story mixed-use, 8–9 types | Scan | NO table — Phase 2 case | Med-Hard |
| Grace Commons | Multi-level, UFAS/ANSI variants | Scan | "Unit Type Schedule" ON PLAN SHEET A102 | Easy-Med |
| Eastwind | Commercial office reno — NOT MF | Vector | N/A — route to single-pass | N/A |
| 3700 Xanthia | 8 bldgs × 4 stories, 573 units, 2 phases | Scan | Full per-bldg/level matrix p2 | Medium |
| Humboldt (1700) | 149-unit podium | Scan | Mix table + prose total p2 | Medium |
| 2099 Chestnut | 8-story typical-floor tower ("Level 2-7" plan) | Scan | Table unconfirmed | Medium |

Lessons encoded above: counts live anywhere (cover, plan sheets, prose); scans dominate; naming varies; Eastwind-type files must be routed out of MF handling.
