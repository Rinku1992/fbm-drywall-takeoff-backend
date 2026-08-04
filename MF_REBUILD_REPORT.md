# MF Rebuild Report — `feat/mf-resolver-service`

**Branch:** `feat/mf-resolver-service`, cut from `origin/main` @ **`87e6696e`**
(`87e6696ebdb0de1bfb54099dd400b88fb39fbfc5`).
**Code source for the lift:** `feat/multifamily` @ `f1d96eb7` (never merged; files
extracted individually with `git show`).
**Not done, per instructions:** not pushed, not deployed, workflow not run.

Every claim below cites `file:line`. Line numbers for `feat/multifamily` refer to
that branch's files; line numbers without a branch qualifier refer to this branch
as committed.

---

## 1. What was lifted, and from where

### 1a. Into the new service `unit-count-resolver/`

| Item | Source (`feat/multifamily`) | Destination | Lift fidelity |
|---|---|---|---|
| Vector text-scan ranking | `xtimator-3d/helper.py:1490` `_score_unit_count_page_text`, `:1523` `rank_unit_count_pages_vector`, plus constants `:1466-1487` | `app/resolver.py:89`, `:121` | **Verbatim** (AST source extraction, not retyped) |
| Scanned thumbnail triage | `helper.py:1626` `_triage_pass`, `:1709` `triage_unit_count_pages_scanned` | `app/resolver.py:232`, `:331` | Lifted **+ D5c backoff** (§4.4) |
| High-res extraction | `helper.py:1846` `_extract_unit_counts` | `app/resolver.py:538` | Lifted **+ bounded attempts / fixed temp** (§4.1) and **+ render cap** (§4.3) |
| D10 precedence | `helper.py:1914` `_apply_precedence`, `:1830` `_unify_candidate_pages` | `app/resolver.py:449`, `:434` | **Verbatim** |
| Persistence + provenance | `helper.py:1950` `_persist_unit_counts`, `:1972` `_none_found_payload` | `app/resolver.py:716`, `:760` | Lifted **+ extraction-health fields** (§4.1) |
| Orchestration | `helper.py:1989` `resolve_unit_counts` | `app/resolver.py:799` | Structure unchanged; new failure/none-found branches |
| Detection prompt + models | `xtimator-3d/prompts.py:545` `UNIT_COUNT_DETECTOR`, `:591-618` models | `app/prompts.py` | **Verbatim** |
| Extraction prompt | `prompts.py:620` `UNIT_COUNT_EXTRACTOR` | `app/prompts.py` | Lifted **+ NOTHING FOUND rule** (§4.2) |
| `UnitTypeCount` | `prompts.py:666` | `app/prompts.py` | **Verbatim** |
| `UnitCountResponse` | `prompts.py:678` | `app/prompts.py` | **Rewritten** — bug fix 2 (§4.2) |

Shared helpers copied into `app/shared.py` from **`origin/main`** (not the feature
branch — see §6.1), byte-for-byte via AST extraction:
`load_pg_pool` (`helper.py:79`), `close_pg_pool` (`:149`), `pg_run` (`:168`),
`load_vertex_ai_client` (`:630`), `load_nearest_region` (`:668`),
`phoenix_call` (`:695`).

One helper was **adapted, not copied**: `load_organization_slug`
(`origin/main helper.py:1464`) is `async` and awaits `run_in_threadpool(pg_run)`.
`app/shared.py:load_organization_slug` is the plain sync equivalent — same query,
same fallback — because the resolver's whole job runs synchronously inside a
worker thread. Documented at `unit-count-resolver/app/shared.py:16-22`.

### 1b. Into `xtimator-3d` (matcher + multiplier, item 12)

| Item | Source | Destination | Fidelity |
|---|---|---|---|
| Matcher (rules → area → batched LLM) | `feat/multifamily helper.py:2159-2340` (`_normalize_unit_label`, `_match_section_by_rules`, `_match_section_by_area`, `_match_leftovers_llm`, `match_sections_to_unit_types`) | `xtimator-3d/helper.py:1712` and above | **Verbatim** |
| Multiplier / rollup | `helper.py:2351-2455` (`_num`, `_accumulate_scaled_takeoff`, `_round_takeoff`, `summarize_unit_counts`) | `xtimator-3d/helper.py:1785`, `:1811` | Verbatim **except one log line** (§5) |
| Matcher prompt + models | `prompts.py:712` `UNIT_MATCH_RESOLVER`, `:745` `UnitMatch`, `:749` `UnitMatchResponse` | `xtimator-3d/prompts.py` (appended) | **Verbatim** |
| `summarize_takeoff_all` wiring | `feat/multifamily main.py:2380-2398` | `xtimator-3d/main.py:2904-2924` | **Verbatim** (comment updated to mention `extraction_failed`) |

### 1c. Deliberately NOT lifted

`launch_unit_count_resolver` (`feat/multifamily helper.py:2129`),
`_run_unit_count_resolver` (`:2105`), `_snapshot_pdf` (`:2097`) and the
`_UNIT_COUNT_RESOLVER_TASKS` strong-reference set (`:2094`) — the detached
in-process task machinery. The service **is** the replacement (D5a).

### 1d. Migration

`migrations/2026-07_add_plans_unit_counts.sql` lifted **unchanged** and verified
byte-identical against `feat/multifamily` (`diff` returned empty). Note this file
does **not** exist on `origin/main` — the `migrations/` directory was untracked
and empty in the working tree, so the file had to be lifted, not merely reused.

---

## 2. Project-type column finding (item 11)

**The column exists — no DB change needed.** It is:

```
projects.project_type
```

Evidence:
- `xtimator-3d/main.py:505` — `project_type` in the `INSERT` column list of `insert_project`.
- `xtimator-3d/main.py:531` — the bound parameter `payload_project.project_type`.
- `xtimator-3d/main.py:796` — `PayloadProject.project_type: str`.

**Caveat you should confirm before trusting the trigger.** The column is
**free-text `str`**, populated by the existing frontend selector (D4a). Nothing in
the repo constrains, enumerates, or validates its values — `grep -rn project_type`
across all `.py`/`.sql`/`.md` returns only those three lines. The actual literal
the frontend writes for multi-family could not be verified (no DB access during
the build).

The predicate is therefore tolerant (`xtimator-3d/main.py:582`
`_is_multifamily_project_type`): it normalises punctuation and case, then accepts
`mf`, `multi family`, `multifamily`, or anything containing both `multi` and
`family`. It matches `MULTI_FAMILY`, `Multi-Family`, `multi family`,
`MULTIFAMILY`, `MF`; it rejects `SINGLE_FAMILY`, `Commercial`, `None`, `''`.

**Action for you:** run `SELECT DISTINCT project_type FROM projects;` and confirm
the MF value is matched. If it is something unexpected (a code, an integer, an
enum id), the predicate needs one line changed.

---

## 3. §9a verification — silent full-page fallback (NOT modified)

Verified present, unchanged, and **not touched** by this branch:

| Function | `except` block | Fallback written | Log signature |
|---|---|---|---|
| `classify_plan` | `xtimator-3d/helper.py:818` (def), except at `:861-870` | `offset_top_left=[0.0,0.0]`, `offset_bottom_right=[1.0,1.0]`, single box, `title=''` (`helper.py:868`) | `SYSTEM: Plan Classification has failed` — `helper.py:863` |
| `detect_bounding_boxes` | `xtimator-3d/helper.py:874` (def), except at `:917-926` | same full-page box (`helper.py:923`) | `SYSTEM: Bounding Box detection has failed` — `helper.py:919` |

Both catch bare `Exception`, which includes `phoenix_call` exhausting its retries,
and return one whole-page section instead of raising. This is the mechanism behind
the Aurora page-20 symptom (3 dropdowns → 1). **Confirmed core-pipeline scope;
changed nothing.** `git diff origin/main -- xtimator-3d/helper.py` contains no hunk
in either function.

**Incidental finding:** `classify_plan` is now **dead code** — it is defined at
`helper.py:818` and called from nowhere (`grep -rn "classify_plan" xtimator-3d/*.py`
returns only the definition). Classification is done by the ConvNeXt service via
`plan_to_preview` (`helper.py:929`). So of the two §9a fallbacks, only
`detect_bounding_boxes`' is reachable today. Not acted on — core-pipeline scope.

---

## 4. Bug fixes — before / after

### 4.1 Fabrication guard (item 7)

**Before.** `feat/multifamily helper.py:1886-1907`: one `phoenix_call` with
`max_retry=credentials["VertexAI"]["llm"]["max_retry"]` = **5**
(`xtimator-3d/config/gcp.yaml:43`). `phoenix_call` escalates temperature on every
retry — on that branch to a **0.5** ceiling
(`feat/multifamily helper.py:651`; `origin/main helper.py:726` caps at 0.25).
So a hard page got up to 5 shots at rising temperature, and the stored result
carried no trace of how it was obtained.

**After** (`unit-count-resolver/app/resolver.py:530` `_extract_unit_counts`):

- The **attempt loop is owned by the resolver**, not by `phoenix_call`. Each
  attempt is `phoenix_call(..., max_retry=1)` — exactly one shot, raises on failure.
- The generate lambda **ignores** the `temperature` argument `phoenix_call` offers
  and pins `UNIT_COUNT_EXTRACTION_TEMPERATURE` (default `0`). No escalation path
  exists on this code path.
- Attempts stop at `UNIT_COUNT_EXTRACTION_MAX_ATTEMPTS` (default **2**).
- Returns `(extraction, attempts)` so the count is recorded, not inferred.

**Failure state.** `resolver.py:773` `_extraction_failed_payload` writes
`source="extraction_failed"` with `provenance.attempts=N`. Previously a failed
extraction fell through to `_none_found_payload`, making "no counts in the
document" and "we could not read it" indistinguishable. Downstream effect is the
same (empty `unit_counts` → every section ×1, D13), but the DB and logs now say
which happened. **No answer is ever invented.**

**Sanity check.** `resolver.py:665` `_looks_fabricated` rejects an accepted answer
when **both** hold: (a) ≥2 types all share the *identical* count and that value is
a multiple of 5 — the Aurora shape, `10/10/10/10` against a true `8/18/8/11`; and
(b) **none** of the extracted type labels appears in the pages' text layer
(cheap `fitz.get_text()` read, `resolver.py:642`). Condition (b) is only checkable
on vector PDFs; when the pages have no text layer the check **accepts and logs
that it could not verify** (`resolver.py:695-698`) — failing closed there would
break every legitimate scanned extraction, and scanned sets dominate the corpus
(master plan §12). Rejection is logged with the rejected counts and stored as
`extraction_failed` with `flagged_suspect=true` and `suspect_reason`.

**Per-type logging (D18).** `resolver.py:917-920`:

```
[UNIT_COUNTS] [project=… plan=…] resolved: {'VISTA A': 8, 'VISTA B': 18, …} total=45 attempts=1
```

The previous aggregate-only line (`types=4 total=40`) is why the fabrication was
invisible.

**Health in provenance.** `resolver.py:739` `_provenance` adds `attempts`,
`flagged_suspect`, `suspect_reason` to the stored JSON. A first-call answer and a
last-chance one no longer look identical.

### 4.2 None-found accepted as valid (item 8)

**Before** (`feat/multifamily prompts.py:678-702`): `source_form` was a
non-optional `Literal["table","prose","mixed"]`; `source_pages` had a validator
requiring non-empty; and a model validator raised when
`not per_type_counts and total_units is None`. A legitimately empty read was
**inexpressible** — every field forbade it. The `ValidationError` surfaced inside
`phoenix_call`, which treats any exception as malformed output and re-asks, at
escalating temperature, up to 5 times. That is the retry spiral that produced the
fabrication.

**After** (`unit-count-resolver/app/prompts.py` `UnitCountResponse`):
`source_form` is `Optional`, `source_pages` defaults to `[]`, counts may be empty
— **but only as a coherent whole**. The validator splits the two cases the old
code conflated:

- **Legitimately empty → accept, no retry:** counts `[]` **and** total `None`
  **and** `source_form is None` **and** `source_pages == []`. Exposed as
  `is_none_found`; the resolver stores `none_found` (`resolver.py:877`).
- **Malformed → raise, retry as before:** nothing found yet `source_form` or
  `source_pages` asserted (a half-answer), or counts/total present with no
  provenance, or duplicate labels.

The prompt was widened to match, otherwise the model could never emit the empty
answer: a **NOTHING FOUND** rule was added to `UNIT_COUNT_EXTRACTOR`, stating that
an empty result is a correct and expected outcome, that it is always preferable to
guessing, and that partially-empty mixtures are not allowed.

**Verified behaviourally** (10/10 checks, run against the committed module):

| Case | Expected | Result |
|---|---|---|
| fully-empty none-found | accept, `is_none_found=True` | PASS |
| nothing found but `source_form="table"` | reject | PASS |
| nothing found but `source_pages=[3]` | reject | PASS |
| counts present, `source_form=None` | reject | PASS |
| counts present, `source_pages=[]` | reject | PASS |
| real 4-type table answer | accept, `is_none_found=False` | PASS |
| prose-only partial truth | accept | PASS |
| duplicate `unit_type` | reject | PASS |

### 4.3 Render cap (item 9)

**Before** (`feat/multifamily helper.py:1859-1865`): `zoom = high_dpi / 72.0`,
`fitz.Matrix(zoom, zoom)`, `get_pixmap(...)` — **no size bound at all**.

**After** (`unit-count-resolver/app/resolver.py:489` `_capped_matrix`): the zoom is
scaled **down only** so the longest side ≤ `UNIT_COUNT_MAX_RENDER_LONGEST_SIDE`
(default **4000 px**) and total pixels ≤ `UNIT_COUNT_MAX_RENDER_PIXELS` (default
**80 MP**, under PIL's ~89.5 MP `DecompressionBombError` threshold). Both bounds
apply; the tighter wins. Capping is logged with requested vs effective DPI.

**Verified numerically** (4/4 checks):

| Sheet @ DPI | Uncapped | Capped | PIL bomb? |
|---|---|---|---|
| ARCH E1 42×30in @ 200 | 8400×6000 = 50.4 MP | 4000×2857 = 11.4 MP | ok |
| ARCH E 48×36in @ 200 | 9600×7200 = 69.1 MP | 4000×3000 = 12.0 MP | ok |
| **ARCH E1 42×30in @ 300** | **12600×9000 = 113.4 MP** | 4000×2857 = 11.4 MP | **TRIPS** |
| 72×48in @ 200 | 14400×9600 = 138.2 MP | 4000×2667 = 10.7 MP | TRIPS |
| Letter 8.5×11in @ 200 | 1700×2200 = 3.7 MP | unchanged | ok |

The third row reproduces the reported **113M-pixel Aurora render to within 0.4%**,
which suggests Aurora was an ARCH E1 sheet rendered at 300 DPI. Offered as a
plausible reconstruction, not an established fact — I could not inspect the file.

### 4.4 Serialized triage (item 10, D5c)

**Discrepancy — please read.** `MF_CURRENT_STATUS.md:27` says the first
implementation's "thumbnail triage batches fired freely". **The code does not
support that.** `feat/multifamily helper.py:1641` is a plain sequential
`for batch_index in range(n_batches):` with a blocking `phoenix_call` inside —
there is no `ThreadPoolExecutor`, no `asyncio.gather`, no concurrency of any kind.
The batches were **already serialized**.

What was genuinely missing is the **backoff**: batches fired back-to-back as fast
as they returned. So this item is a smaller change than the checklist implies.

**After** (`unit-count-resolver/app/resolver.py:232` `_triage_pass`): a
`UNIT_COUNT_TRIAGE_BACKOFF_SECONDS` (default **2.0 s**) pause before every call
after the first — placed *before* the call, not after, so a `continue` on a
skipped or failed batch cannot bypass it (`resolver.py:251-254`). The
no-concurrency requirement is now an explicit documented contract rather than an
accident of the loop shape, and the serialization is logged per pass so it is
auditable.

---

## 5. Item 12 — is scaling per-section or project-total only?

**Read from the code; behaviour NOT changed.**

**Answer: the multiply is applied *per section*, but the only *scaled output* is
the aggregate project total. No scaled per-section figure is ever surfaced.**

Evidence in `xtimator-3d/helper.py:1811` `summarize_unit_counts`:

1. Each row gets its own count: `count = counts_by_type.get(matched_type, 1) if matched_type is not None else 1` (`helper.py:1845`).
2. That row's takeoff is scaled by **its own** count and accumulated:
   `_accumulate_scaled_takeoff(project_total, takeoff, count)` (`helper.py:1850`),
   which multiplies each numeric field by `count` (`helper.py:1785-1799`).
3. The per-section record appended to `section_rollup` (`helper.py:1851-1857`)
   carries `matched_unit_type`, `count`, `method` — **but no scaled takeoff**.
4. `drywall_takeoff_all` — the caller's existing payload — is **never mutated**.
   `main.py:2920` attaches the summary under a separate `unit_count_summary` key.

So a consumer reading per-section numbers sees today's unscaled values; only
`unit_count_summary.project_total` reflects the multiplication. Mathematically the
total is correct either way (Σ takeoffᵢ × countᵢ). Flagging it because a UI showing
per-section rows next to a scaled project total would look inconsistent, and
`MF_CURRENT_STATUS.md:36` notes the multiplier has never been validated on a real
match. **No change made — that is your call.**

**Logging added** (`helper.py:1858-1869`), the one intentional deviation from a
verbatim lift:

```
[UNIT_MULTIPLY] [project=… plan=…] section "UNIT A - 1ST FLOOR PLAN" (page=19)
    → matched_type=VISTA A (method=rule) → count_applied=×8
```

Unmatched sections log `matched_type=NO MATCH … count_applied=×1 [D13 default]`,
so a ×1 from "no match" is distinguishable from a ×1 that is the type's real count.

---

## 6. D5b rationale revised

Bounding boxes are NOT part of the automatic upload burst — they fire in
`/floorplan_to_2d`, which is user-triggered per page. So there is no
"after classification + bounding boxes" moment to sequence behind, and the
resolver-vs-sectioning collision window cannot be eliminated by trigger timing.
The protection against the page-20 failure is therefore the bounded LLM footprint
(items 7, 9, 10: max-2-attempt fixed-temp extraction, render cap, serialized
triage) — not the trigger point. Trigger point is end of `/floorplan_to_preview`,
same position the old resolver used; what changed vs the old design is the
footprint, the separate service, and the bug fixes.

### Supporting evidence

| Phase | Where it persists | Trigger type |
|---|---|---|
| Page classification | `/floorplan_to_preview` → `floorplan_to_pages` → `insert_pages_batch` (`helper.py:955`, `:974`) | automatic on upload |
| Bounding boxes | `/floorplan_to_2d` (`main.py:1525`) | **user-triggered**, after page selection |

`floorplan_to_pages` explicitly writes `bounding_box_offsets` as an **empty dict**
at preview time (`helper.py:966`); the real values are computed by
`load_visual_grounding` (`helper.py:1214`) — called from exactly one place,
`main.py:1496` inside `/floorplan_to_2d` — and persisted at `main.py:1525`,
immediately before the per-page `floorplan_to_structured_2d` fan-out.

Firing at that second point would have put the resolver at the **start of the
heaviest Vertex phase**, which is the opposite of D5b's intent.

---

## 7. The trigger patch

Two additions to `xtimator-3d/main.py`, plus the call site.

**Call site** — end of `/floorplan_to_preview`, immediately before the 200:

```diff
     logging.info("SYSTEM: Preview generated Successfully")
+
+    # D5b — Unit-Count Resolver trigger. Fires here, at the end of the preview
+    # path, because this is the point where page classification is persisted and
+    # the pages are ready for the frontend. Fire-and-forget: MF projects only,
+    # 5s timeout, all exceptions contained (see trigger_unit_count_resolver).
+    await trigger_unit_count_resolver(project_id, plan_id, user_id)
+
     return respond_with_UI_payload(payload_preview)
```

`main.py:1398`, inside `floorplan_to_preview` (`main.py:1344`).

**`_is_multifamily_project_type`** (`main.py:582`) — the tolerant predicate in §2.

**`trigger_unit_count_resolver`** (`main.py:603`) — guard order and containment:

1. `MF_RESOLVER_ENABLED` — flag **kept**; only `false`/`0`/`no` (case-insensitive)
   disable. Unset behaves as enabled, exactly as before.
2. `UNIT_COUNT_RESOLVER_URL` unset → log and return (never a crash).
3. `SELECT project_type FROM projects WHERE LOWER(project_id)=LOWER(%s)` → not
   multi-family → log and return. Single-family path is untouched.
4. Mint an ID token for the resolver audience
   (`helper.py:572` `load_unit_count_resolver_ID_token`, same idiom as
   `load_floorplan_to_structured_2d_ID_token` at `helper.py:552`).
5. `httpx.AsyncClient(timeout=5)` → `POST {url}/resolve_unit_counts` with
   `{project_id, plan_id, user_id}`.

The whole body is wrapped in `try/except Exception` that logs and swallows
(`main.py:651-653`) — deliberately broad, because **the upload flow must never
depend on the resolver**. Nothing is raised; nothing is awaited beyond the 5 s
timeout.

`httpx` added to `xtimator-3d/requirements.txt` — it was not previously a
dependency (the repo uses `requests`), and the spec calls for `httpx`.

---

## 8. The new service

```
unit-count-resolver/
├── Dockerfile              # page-classifier shape, minus GPU/checkpoint
├── requirements.txt        # subset of xtimator-3d's
├── README.md
├── config/.gitignore       # blocks gcp.yaml / *.json from ever being committed
└── app/{main,config,gcs_client,schemas,prompts,resolver,shared}.py
```

`POST /resolve_unit_counts {project_id, plan_id, user_id}` → **202 immediately**
(`app/main.py:143-146`), work runs in a `BackgroundTask` (`app/main.py:161`).
`GET /healthz` gates on the DB pool.

Ingress/auth follow wall-detector and page-classifier: enforced at the platform
layer by `--ingress internal --no-allow-unauthenticated`, with no in-app token
verification (neither sibling does any either; `wall-detector/main.py` has no auth
code at all). Deploy flags in `.github/workflows/unit_count_resolver.yml`.

Self-containment respected: no imports across service folders.

---

## 9. Things that did NOT match the instructions

Listed because you asked for them, not smoothed over.

1. **GCS path has an extra segment.** The spec says
   `gs://.../{project}/{plan}/floor_plan.PDF`. The real path is
   `gs://{bucket}/{organization_slug}/{project}/{plan}/floor_plan.PDF` —
   `xtimator-3d/helper.py:1291-1293` and
   `xtimator-page-classifier/app/gcs_client.py:38`. The classifier is *handed* the
   slug by its caller; this service is not, since the spec fixes the body to
   `{project_id, plan_id, user_id}`. **Resolved without changing the contract:**
   the service derives the slug from `user_id` using the same query
   (`app/shared.py:load_organization_slug`, called at `app/main.py:112`).

2. **"After classification + bounding boxes persist" is not a real point in the
   code.** Full evidence in §6. Resolved by your decision to fire at the end of
   `/floorplan_to_preview`.

3. **Triage was already serialized.** `MF_CURRENT_STATUS.md:27` overstates this;
   only the backoff was missing. Details in §4.4.

4. **`classify_plan` is dead code.** §3. Not acted on.

5. **The migration file is not on `origin/main`.** Item 6 says "reuse the existing
   migration file"; it exists only on `feat/multifamily`, and `migrations/` was an
   untracked empty directory. Lifted byte-identical and now tracked.

6. **`Optional` import.** `feat/multifamily` had widened
   `xtimator-3d/prompts.py:1` to `from typing import List, Set, Literal, Optional`.
   `origin/main` has not. The lifted `UnitMatch` needs it, so the same one-word
   change was reapplied — otherwise `NameError` at import.

7. **`MF_MASTER_PLAN.md` / `MF_CURRENT_STATUS.md` are untracked.** They exist on
   `feat/multifamily` and as untracked working-tree files, but are **not** on
   `origin/main`, so they are not on this branch either. I read them from the
   working tree. **Not committed** — tell me if you want them tracked here.

8. **`--no-cpu-throttling` is required and was added.** Not in the spec, but the
   service returns 202 then works in the background; under Cloud Run's default CPU
   throttling that work would freeze the moment the response is sent. Flagged in
   the workflow with a comment explaining why it must not be removed.

9. **Caller-side env var still to be set.** `UNIT_COUNT_RESOLVER_URL` must be added
   to the **drywall-takeoff-3d** service. Use `--update-env-vars`, never
   `--set-env-vars` (`MF_CURRENT_STATUS.md:61`). Until it is set, the trigger logs
   a warning and does nothing — safe by construction.

10. **`config/gcp.yaml` is not committed** (deliberate). The Dockerfile fails the
    build if it is not staged, matching page-classifier. Given this repo already
    has unrotated credentials in its history, a new committed secrets file would
    make that worse.

---

## 10. Verification performed

| Check | Result |
|---|---|
| `python -m compileall unit-count-resolver` | **OK** |
| `python -m compileall xtimator-3d` | **OK** |
| Undefined-name scan (AST) over all 11 touched files | **OK** — no unresolved names |
| Bug fix 2 behavioural suite (10 cases) | **10/10 PASS** |
| Bug fix 3 render-cap suite (4 cases) | **4/4 PASS** |
| Migration byte-identical to `feat/multifamily` | **confirmed** |
| §9a fallbacks unmodified | confirmed — no diff hunk in either function |

**Not verified — cannot be, without deploy access:** the service has never been
run or built; no Vertex, GCS, or Cloud SQL call has been exercised; the trigger has
never fired; the matcher has still never been validated on a real match
(`MF_CURRENT_STATUS.md:36`); and the `project_type` literal is unconfirmed (§2).
The test suites above cover pure logic only.

---

## 11. Suggested next steps

1. Confirm the `project_type` MF literal (§2) — the one input that could silently
   disable the whole feature.
2. Apply `migrations/2026-07_add_plans_unit_counts.sql` to Cloud SQL.
3. Deploy the resolver; set `UNIT_COUNT_RESOLVER_URL` on drywall-takeoff-3d with
   `--update-env-vars`.
4. Aurora regression: expect **3 dropdowns on page 20**, counts `8/18/8/11 = 45`,
   and a `[UNIT_COUNTS] resolved: {...} attempts=N` line. If the counts are wrong
   again, `flagged_suspect` and `attempts` in `plans.unit_counts.provenance` now
   tell you whether extraction struggled.
5. Ocean View single-family regression: `unit_count_summary.applied` must be
   `false` and the totals identical to today.
6. Grace Commons for the scanned path and the serialized triage backoff.
7. Decide the §5 question: should per-section rows expose scaled figures, or stay
   unscaled with only the project total scaled?

---

## 12. Addendum (2026-08-04) — project-type gate bypassed, trigger fires for ALL projects

**Interim decision by Ravikant.** §2 of this report flagged the `projects.project_type`
literal as unverified. It has now been checked with `SELECT DISTINCT project_type
FROM projects`, and the finding invalidates the gate as designed:

> **The column contains only `COMMERCIAL` and `RESIDENTIAL`.** The frontend MF/SF
> selector that **D4a** assumes already exists **has not been built yet**.

Neither value can express multi-family, so `_is_multifamily_project_type` rejects
100% of real rows today (verified: both return `False`). Left as-is, the trigger
would never fire for any project and the whole resolver path would be dead code
in production.

**Decision:** run the resolver for **every** project type until the frontend
selector ships. Single-family / commercial safety is guaranteed by the **D13 ×1
default** — a project with no resolvable unit counts stores `none_found` (or
`extraction_failed`), `unit_counts` stays empty, every section multiplies ×1, and
the takeoff equals today's exactly.

### What changed

`xtimator-3d/main.py:trigger_unit_count_resolver` — new env var
**`MF_PROJECT_TYPE_GATE`**, default **`"false"`** (gate OFF):

| `MF_PROJECT_TYPE_GATE` | Behaviour | Log line |
|---|---|---|
| unset / `false` / `0` / `no` **(default today)** | DB query skipped; fires for **all** projects | `trigger: project_type gate OFF — firing for all projects` |
| `true` / `1` / `yes` | Original D4a check restored | `trigger: project_type gate ON — project_type=… is/is not multi-family` |

The `SELECT project_type` query and `_is_multifamily_project_type` are **kept, not
deleted** — the predicate is simply not called while the gate is off. When the
frontend lands, flipping the env var to `true` restores the D4a behaviour with
**zero code change**; the only follow-up is confirming the real MF literal matches
the spellings the predicate accepts. A `TODO` at
`_is_multifamily_project_type` records this, including today's actual DB values.

`MF_RESOLVER_ENABLED` is unaffected and still the master off-switch: it is
evaluated **before** the project-type gate, so `MF_RESOLVER_ENABLED=false` disables
the trigger outright regardless of `MF_PROJECT_TYPE_GATE`.

Gate parsing verified across 10 env-var spellings (unset, `''`, `false`, `FALSE`,
`no`, `true`, `TRUE`, `1`, `yes`, `' True '`) — all resolve as intended, and the
predicate still returns `False` for both `COMMERCIAL` and `RESIDENTIAL`.

### Two consequences worth tracking

1. **The resolver now runs on every upload, not a multi-family subset.** That is
   the intent, but it multiplies the resolver's Vertex footprint by the full
   project volume rather than the MF slice. D5c keeps each run small (vector path
   = zero Gemini calls; scanned path = serialized batches; extraction = max 2
   calls), so this should stay modest — but it is worth watching quota and cost
   once deployed, since it was not the load profile D5a was sized against.

2. **D13 protects unmatched sections, not mismatched ones.** The ×1 default makes
   a *no-counts* project safe by construction. It does not protect a
   non-residential project whose drawings happen to contain a schedule the
   extractor reads as unit counts **and** whose section titles or areas then match
   a resolved type. That requires two independent coincidences and the matcher is
   built to return "no match" rather than force one (D14), so the risk is low —
   but it is no longer zero the way a project-type gate would have made it. The
   `[UNIT_MULTIPLY]` per-section log lines (§5) are how you would spot it: a
   commercial project showing `count_applied=×N` for N > 1 is the signal.
