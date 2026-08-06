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

---

## 13. Addendum (2026-08-05) — first live run: two fixes

First live Aurora run (`PLAN_1785862120698`). The resolver machinery behaved
exactly as designed — **1 attempt, temperature 0, render cap fired, 23s, honest
provenance, and page 20 kept its 3 dropdowns** (the D5c bounded-footprint bet
held; §9a regression did not reproduce). Two real defects surfaced.

### 13.1 Extraction chose the wrong table

**Symptom.** Ranking was correct — page 2 (the real unit schedule) was candidate
#1. The **extractor** then read the **accessibility compliance table** on pages
11–12 instead:

| Returned (wrong) | Truth (page 2) |
|---|---|
| Accessible Dwelling Units = 10 | VISTA A = 8 |
| Type A Dwelling Units = 10 | VISTA B = 18 |
| Type B Dwelling Units = 100 | VISTA C = 8 |
| areas all `null`, total 110 | VISTA D = 11, with areas, total 45 |

Those row labels are **ANSI A117.1 / IBC compliance categories**, not unit types
— the same apartment is counted under several of them, so 120 is not a unit
count at all. Nothing in the prompt distinguished a unit schedule from any other
table of labels-and-numbers.

**Fix** (`unit-count-resolver/app/prompts.py`, `UNIT_COUNT_EXTRACTOR`) — three
new sections, no code change:

1. **"WHAT YOU ARE LOOKING FOR — THE UNIT TYPE SCHEDULE"** — defines the target
   positively: rows keyed by distinct unit-type names, a count column, and
   *usually a per-unit AREA column, called out as the strongest signal*.
2. **"TABLES YOU MUST NOT READ"** — names the compliance vocabulary explicitly
   (`Accessible/Adaptable Dwelling Units`, `Type A/B/C Dwelling Units`, ANSI
   A117.1, UFAS, ADA, mobility/hearing-visual), lists the tells (no area column,
   round regulatory numbers, sits near code-analysis notes), and carries the
   **Aurora 10/10/100 shape as an explicit negative example**.
3. **"CHOOSING BETWEEN CANDIDATE PAGES"** — the tie-break: prefer named types
   *with* areas; when one candidate has areas and another does not, take the one
   with areas; reject a compliance table **even when it is the only table
   present** (return NOTHING FOUND instead); never merge rows across tables.

An `INTERNAL CONSISTENCY` accuracy rule was also added, telling the model that
rows failing to sum to a printed total means it has misread or mixed tables.

**Ranking code was NOT touched** — it did its job.

### 13.2 Internal inconsistency was invisible

**Symptom.** Extraction reported `total=110` while its own per-type rows summed
to **120**, and provenance still said `disagreement=no`. That flag is set in
`_apply_precedence`, which compares **table vs prose** and only when
`source_form == "mixed"` — an answer contradicting *itself* was never checked.

**Fix** (`unit-count-resolver/app/resolver.py`, `_check_internal_consistency`) —
independent of `source_form`. Whenever per-type counts **and** an explicit total
are both present and disagree:

- a loud `[UNIT_COUNTS] INTERNAL INCONSISTENCY:` warning with the sum, the total,
  the signed difference, and the full per-type breakdown;
- `flagged_suspect = true` and
  `suspect_reason = "internal_sum_mismatch(sum=X, total=Y)"` in provenance;
- the reason is folded into the existing `disagreement` field, so one provenance
  field now surfaces *every* "these numbers disagree" signal;
- the run-complete line reports `flagged_suspect=yes|no`.

**The answer is flagged, NOT rejected** — as instructed. A mismatch means the
read is untrustworthy, but the numbers are still evidence worth persisting.

### 13.3 Where the multiplier actually needs to hook in

**Finding: the "View Estimates" screen calls `/compute_takeoff`
(`xtimator-3d/main.py:2666`) — NOT `/summarize_takeoff_all`
(`main.py:2908`), which is the only place `summarize_unit_counts` is wired.**
That fully explains the zero `[UNIT_MULTIPLY]` lines.

Evidence:
- Those two are the **only** endpoints in the repo that produce takeoff figures
  (`grep` over `main.py` routes; no other service defines one).
- `/compute_takeoff` returns exactly the shape the screen shows —
  `total.wall`, `total.roof` (ceiling), and `per_drywall.<surface>.<SKU type>`
  with `total_sqft`, `net_sqft`, `sheets_required_total`,
  `sheets_required_no_waste` (`main.py:2731`, `main.py:2884-2907`).
- It carries a `load_preview` flag (`main.py:2684`) that suppresses the DB write
  (`main.py:2888`) — i.e. an explicit read-only "just show me" mode, which is
  what a view screen calls.
- `/summarize_takeoff_all` is the only reader of the stored `takeoff` column
  (`main.py:2925`) and the only caller of `summarize_unit_counts`.

**Important consequence:** `/compute_takeoff` is **per-section** — it takes
`page_number` + `page_section_number` and returns ONE section. So the estimates
screen is showing a *section* total, not a project total. The build request said
to match "the sections in the response"; there is only ever one, so the preview
matches that single section and scales it.

**Temporary preview added** (`main.py`, `MF_SCALED_ESTIMATES_PREVIEW`, default
`"false"`):

- Off (default): **no-op** — the response is byte-identical to before.
- On: reads `plans.unit_counts`, runs the **real matcher**
  (`match_sections_to_unit_types`) against that one section, scales
  `total.{roof,wall}` and every numeric `per_drywall` field by the matched count,
  and logs the **same** `[UNIT_MULTIPLY]` line `summarize_takeoff_all` emits
  (plus a `[SCALED-ESTIMATES-PREVIEW — TEMPORARY]` marker).
- Scaling runs **after** `insert_takeoff`, so what is **persisted stays
  unscaled** — only the returned view changes.
- Wrapped in `try/except` returning the untouched takeoff on any failure.
- `sheets_required_*` stay integral; non-numeric fields (e.g. SKU labels) are
  untouched; the response shape never changes.

Marked `TEMPORARY FOR TESTING` in a delimited block with removal instructions.
`/summarize_takeoff_all` and `summarize_unit_counts` were **not modified**.

This is a stopgap for observing the multiply, **not** an answer to §5. The real
question — should per-section rows show scaled figures, or stay unscaled with
only a project total scaled? — is now sharper, because the screen users actually
look at is per-section and has no project-total view wired to it at all.

### 13.4 Verification

34/34 behavioural checks pass: 9 on the prompt's new content (including that the
NOTHING FOUND rule survived), 6 on the consistency checker (Aurora 120-vs-110
flagged with the exact reason string; the true 8/18/8/11=45 **not** flagged;
no-total and prose-only cases correctly skipped), and 19 on the preview (10
env-var spellings, count=1 no-op, float/int scaling, sheet integrality,
non-numeric fields untouched, shape preserved). `compileall` clean on both folders.

---

## 14. Addendum (2026-08-05) — ranking bug: why the schedule page was never sent

The Aurora unit schedule is at **fitz index 1 (viewer page 2)**, but all three
live runs sent candidates `[2, 11, 12]`. Index 1 was never given to the extractor,
so §13.1's prompt fix could not possibly have helped on this document — it was
choosing among pages that did not contain the schedule.

### 14.1 There is no indexing bug

Traced hop by hop; the same 0-based integer is carried end to end with no
arithmetic:

| Hop | file:line |
|---|---|
| fitz iteration | `resolver.py:207` — `for page_index in range(n_pages)` |
| text read | `resolver.py:209` — `doc.load_page(page_index)` |
| scorer records | `resolver.py:221` — `{"page_number": page_index, ...}` |
| log prints | `resolver.py:241` — `f"p{r['page_number']}"` |
| candidates built | `resolver.py:488` — `[p["page_number"] for p in ...]` |
| render passes back | `resolver.py:617-619` — `doc.load_page(page_index)` |

So the logged `p2` is **fitz index 2 = viewer page 3**, not viewer page 2. The
candidates `[2, 11, 12]` are viewer pages 3, 12, 13. Index 1 lost on score.

### 14.2 Root cause — cap saturation collapsed the ranking into a tie

Cap saturation across the 112-page set:

| signal | cap | pages at/over cap |
|---|---|---|
| `number_rows` | 15 | **104 / 112** |
| `keyword_hits` | 8 | 23 / 112 |
| `unit_type_hits` | 12 | 13 / 112 |
| **all three** | — | **12 / 112 → score exactly 4.5** |

Twelve pages tied at the 4.5 maximum. The sort is `(-score, page_number)`
(`resolver.py:238`), so the tie broke on **lowest page index**, and the top three
were simply the three lowest-numbered saturated pages. Ranking had degenerated
into *"the first three pages that max out every counter."*

**This also answers "can 140 and 16 number-runs really tie?" — yes.** Both exceed
the cap of 15, so both clamp to 1.0.

The schedule page scored **3.8333 (rank 14)**: `ut=8, kw=21, nr=214`. Its `kw` and
`nr` were far above their caps and contributed nothing extra, while `ut=8` fell
short of the cap of 12.

### 14.3 Why `ut` was only 8 — the regex prefers compliance tables

`_UNIT_TYPE_RE` matches `1BR`, `STUDIO`, `n BED`, `n BA`, `UNIT TYPE`,
`TYPE <A-Z0-9>`, `UNIT`. It does **not** match `VISTA` — which appears **64 times**
on index 1. Its 8 matches were
`['TYPES','TYPE\nGFA','TYPE\nGFA','UNIT','UNIT','UNIT','UNIT','TYPE OF']`.

Indexes 11–14 are ICC A117.1 accessibility sheets, dense with `TYPE A` / `TYPE B` /
`TYPE C`, scoring `ut=27`. **The unit-type regex systematically favours
accessibility compliance tables over a real schedule whose types carry marketing
names.** The extractor's wrong pick in §13.1 was the downstream symptom of this.

Ground truth on index 1, read straight from the text layer:
```
TYPE  GFA (SF)  NO. UNITS  SUB-TOTAL (SF)
VISTA I    1,872  X   8  = 14,976
VISTA II   1,671  X  18  = 30,078
VISTA III  1,791  X   8  = 14,328
VISTA IV   1,260  X  11  = 13,860
SUB-TOTAL                45
```

### 14.4 What was rejected, with measurements

Both intuitive fixes were tested against the real PDF and **both fail**:

| Option | top-3 | schedule rank | verdict |
|---|---|---|---|
| Uncap the signals | `[20, 23, 26]` | **7** | does not fix |
| Widen candidates 3→5, no scoring change | `[2, 11, 12, 13, 14]` | **14** | does not fix |
| Uncap **and** widen to 5 | `[20, 23, 26, 55, 53]` | **7** | does not fix |

Uncapping just replaces one bad proxy with another: pages 20/23/26 are dense
schedule-ish sheets with hundreds of number-rows and no unit counts.

### 14.5 The fix — a schedule-header signal

`_UNIT_SCHEDULE_HEADER_RE` (`resolver.py:89`) matches header-row vocabulary that a
genuine unit mix/schedule has and a compliance table does not: `NO. UNITS`,
`UNIT MIX`, `UNIT SCHEDULE`, `UNIT TABULATION`, `GFA`, `SUB-TOTAL`, `TOTAL UNITS`.
Weighted **3.0** (capped at 6 hits) — the largest weight in the scorer, chosen to
break the 4.5 tie. Existing caps and weights are unchanged.

`UNIT_COUNT_MAX_CANDIDATES` 3 → **5** (`config.py`) as cheap insurance, on top of
the scoring change rather than instead of it.

Per-page features now log `sch=` alongside `ut/kw/nr` (`resolver.py:246`), and the
log comment records that `p<N>` is the 0-indexed fitz page — so the next wrong
pick is diagnosable from Cloud logs without a local rerun, which this
investigation required.

**Measured result on Aurora:**

| rank | fitz | viewer | score | ut | kw | nr | sch | |
|---|---|---|---|---|---|---|---|---|
| 1 | **1** | **2** | **6.8333** | 8 | 21 | 214 | **9** | **the unit schedule** |
| 2 | 2 | 3 | 5.0 | 12 | 34 | 140 | 1 | |
| 3 | 11 | 12 | 5.0 | 27 | 30 | 16 | 1 | A117.1 sheet |
| 4 | 12 | 13 | 5.0 | 27 | 30 | 16 | 1 | A117.1 sheet |
| 5 | 13 | 14 | 5.0 | 27 | 30 | 16 | 1 | A117.1 sheet |

Index 1 goes **14th → 1st**, the top score is now **unique** (12-way tie broken),
margin **1.8333**. Candidates become `[1, 2, 11, 12, 13]` (viewer 2, 3, 12, 13, 14).

### 14.6 Deliberately NOT done, and what is still unproven

- **No compliance penalty.** A negative weight on `A117.1`/`ICC`/`TYPE A/B/C
  DWELLING` also worked in testing (it pushed the A117.1 sheets out of the top 5
  entirely), but the extractor prompt already rejects compliance tables
  downstream. Penalising at rank as well would double the backfire risk for a
  real schedule that shares a sheet with accessibility notes, for no extra
  coverage. Decision by Ravikant.
- **Consequence of that choice:** three A117.1 sheets are still in the candidate
  list at ranks 3–5. Correctness now depends on the extractor prompt doing its
  job — the two fixes are coupled, not independent.
- **Tuned on ONE document.** `GFA` and `SUB-TOTAL` may be this architect's house
  style. Aurora is the only PDF available locally (the repo contains no other),
  so this is n=1. **It must be checked against Grace Commons, Xanthia and
  Humboldt** — all scanned, so they exercise the triage path rather than this
  scorer, but their schedules will still test the header vocabulary once
  extracted.
- **The `_UNIT_TYPE_RE` blind spot is not fixed.** It still cannot see
  marketing-name unit types like `VISTA`. The schedule-header signal routes
  around that rather than solving it; a set whose schedule lacks all of the
  header phrases would still rank poorly.

---

## 15. Addendum (2026-08-06) — extraction fix: stop seeding the model

The model returned generic bedroom-count names (STUDIO / 1 BED / 2 BED / 3 BED)
**twice** on a document whose types are named VISTA I–IV. The prompt was supplying
those names itself.

### 15.1 What was found in the prompt

Three places handed the model example type names, the worst sitting directly on
the field it fills in:

1. **The `unit_type` field description** — the closest thing to a template for the
   fabricated answer:
   > `` `unit_type` ``: the unit-type label EXACTLY as printed (e.g. **"1BR-A", "Type 2", "2 Bed / 2 Bath", "Studio S1"**). Do NOT normalize, expand, or rename it.

2. **The "WHAT YOU ARE LOOKING FOR" section**:
   > rows keyed by DISTINCT UNIT-TYPE NAMES — marketing or plan names such as **"VISTA", "1BR-A", "Type 2", "Studio S1", "PLAN A"**

3. **The §13.1 negative example**, which supplied concrete *numbers*:
   > `Accessible Dwelling Units .... 10` / `Type A Dwelling Units ........ 10` / `Type B Dwelling Units ....... 100`

`"Studio S1"` and `"2 Bed / 2 Bath"` are one normalisation step away from
`STUDIO` and `2 BED`. The instruction said "as printed" while the examples
demonstrated a naming convention — and the model followed the examples.

### 15.2 The scrub

- **All example names removed.** Every field now carries a placeholder —
  `<TYPE-NAME-AS-PRINTED>`, `<COUNT-AS-PRINTED>`, `<AREA-AS-PRINTED>`,
  `<TOTAL-AS-PRINTED>`, `<PAGE-NUMBER>` — in both the field descriptions and the
  output template, with a closing note that these are placeholders, not values.
- The prompt now **states that it withholds examples on purpose**: *"This prompt
  deliberately gives you NO example names: any name you return that you cannot
  point to on the page is a fabrication."*
- **Negative example de-numbered** to `<N>` — the shape is still taught, the
  figures are gone.
- **New overriding accuracy rule**: every name and number returned must be
  visibly present on the supplied images; *"A partial answer of two names you can
  actually see beats four you cannot."*
- **New self-check**: if several returned names are generic bedroom-count
  categories rather than labels actually read, the model has defaulted to a
  convention — discard and return only what is visible, or the empty answer.
- **D8 restored.** The section was renamed *UNIT TYPE SCHEDULE* → *UNIT TYPE
  INFORMATION*, and all four valid forms are now listed explicitly: a full table,
  a **partial/half-drawn table**, a **legend/keyed note**, or a **prose
  sentence**. The previous wording said "table" throughout, which contradicted D8
  and told the model to ignore three of the four forms real documents use.
  `source_form` guidance now maps partial tables and legends onto `"table"`.
- **Empty-beats-guess strengthened**: *"AN EMPTY ANSWER IS ALWAYS BETTER THAN A
  GUESS. It is recorded as a valid result and costs nothing; an invented one
  silently corrupts a construction estimate."*

### 15.3 `_looks_fabricated` widened

Condition 1 previously required the shared count to be a **multiple of 5** (the
first Aurora fabrication was 10/10/10/10). That was an unnecessary escape hatch —
a fabricated 7/7/7/7 is exactly as wrong and sailed straight through. The clause
is removed: **any** identical count across ≥2 types, combined with labels that
appear nowhere in the text layer, now flags. Real unit mixes are essentially never
perfectly uniform, so the roundness test only narrowed coverage.

Vector-only gating is unchanged: a scanned set has no cheap ground truth, and
failing closed there would break every legitimate scanned extraction.

### 15.4 `UNIT_COUNT_DEBUG` (default off)

The last two diagnoses had to *infer* what the model did. This makes a run leave
evidence:

- **Raw model response logged before parsing** (1500 chars, configurable).
  Implemented by wrapping the `generate_content` call inside the lambda
  `phoenix_call` invokes — the only point where the unparsed text still exists.
  A response that fails Pydantic validation is otherwise completely invisible;
  all the caller ever sees is the exception.
- **Candidate PNGs uploaded** to
  `gs://{bucket}/{org}/{project}/{plan}/unit_count_debug/extraction_page_NNNN.png`.
  These are the *capped, downscaled* renders actually sent — not what you get by
  opening the PDF yourself — so this is the only way to check whether a schedule
  was legible at the DPI used.

`phoenix_call` was **not** modified (it is a verbatim lift from `origin/main`).
When the flag is off, `_log_raw_response` is not called at all and the lambda is
byte-identical in behaviour to before. Both the logger and the uploader are
best-effort: the logger cannot raise (verified against a response whose `.text`
throws) and returns the response object unchanged; the uploader returns `None` on
any failure. `organization_slug` is threaded through `resolve_unit_counts` purely
so debug artifacts land in the plan's own folder, and the PNG upload additionally
requires it to be present.

### 15.5 Verification and one pre-existing weakness found

50/50 new checks pass (27 prompt-scrub, 9 fabrication-check, 14 debug-mode), plus
all three earlier suites and the ranking verification — ranking output on Aurora
is unchanged, as required.

**Pre-existing weakness, NOT fixed (out of scope, flagged for later):**
`_looks_fabricated` matches labels against the text layer by **substring**:

```python
if label and label in haystack:   # label "a" matches "plan", "labels", "matching"
```

A short unit-type label — `"A"`, `"B"`, `"C"`, which are entirely plausible real
plan names — will almost always be found inside ordinary words, silently
disabling the whole check for such documents. This was caught because a test
using single-letter labels failed; the test was wrong, but the weakness is real
and predates this change. A word-boundary match would fix it. **Not changed here
— this session was scoped to a minimal extraction fix.**

---

## 16. Addendum (2026-08-06) — attachment audit: the images WERE attached

Run 2 returned `STUDIO/1BED/2BED/3BED = 10/40/40/10, total=100` at temperature 0
from a page whose debug PNG legibly shows `VISTA I–IV = 8/18/8/11, SUB-TOTAL 45`.
The stated suspicion was that the page images never reached the request.

### 16.1 Answering it plainly: the parts are attached, and correct

**No attachment bug. The images are in the request.** Traced hop by hop:

| Hop | file:line | Result |
|---|---|---|
| pixmap render | `resolver.py:713` — `page.get_pixmap(matrix=matrix).tobytes("png")` | real PNG bytes |
| text part | `resolver.py:717` — `Part.from_text(f"PAGE: {page_index}")` | one per page |
| image part | `resolver.py:718` — `Part.from_data(data=png_bytes, mime_type="image/png")` | one per page, **mime correct** |
| Content assembly | `resolver.py:775` — `Content(role="user", parts=query_parts)` | interleaved text/image |
| the call | `resolver.py:800/821` — `contents=[query]` | present in **all four** lambda branches |

Measured payload for Aurora's five candidates: **8.25 MB** (2.59 / 1.60 / 1.35 /
1.35 / 1.35 MB) — non-empty, and comfortably inside Vertex's ~20 MB inline
ceiling, so the 3→5 candidate widening did not overflow the request either.

**The `UNIT_COUNT_DEBUG` wrapper did not alter anything.** Diffing
`_extract_unit_counts` between `3601ca8a` (Aug 4, the run that read real content)
and `08e6efd4` shows the render→bytes→Part→Content path is **byte-identical**.
The wrapper's only effect is `_log_raw_response(...) if UNIT_COUNT_DEBUG else ...`
around the same `generate_content(contents=[query], ...)` call — same parts, same
order, nothing dropped. Ruled out.

### 16.2 What DID change: the prompt-delivery path flipped silently

`load_vertex_ai_client` (`shared.py:16`) contains a hidden switch:

```python
if prompts and GenerativeModel(...).count_tokens(prompts).total_tokens >= 1024:
    is_cached = True
    cached_content = CachedContent.create(contents=prompts, ...)   # prompt as a USER TURN
    vertex_ai_client = GenerativeModel.from_cached_content(cached_content)
```

Below 1024 tokens the prompt is a **system instruction**. Above it, the prompt
becomes **cached content** and the model is rebuilt with **no system instruction
at all** — so the request degenerates into two consecutive `user` turns:

1. the entire extractor prompt, *including a complete JSON output template*
2. the page images

The extractor prompt crossed that threshold during the 2026-08-05/06 fixes:

| revision | chars | words | vs 1024-token threshold | outcome |
|---|---|---|---|---|
| `3601ca8a` Aug 4 | 4,033 | 583 | below (~760 tok at 1.3 tok/word) | **read real page content** |
| `7bcac550` Aug 5 | 7,279 | 1,092 | above | — |
| `08e6efd4` Aug 6 | 9,878 | 1,495 | **certainly above** | **fabricated** |

**Falsification check passed:** `UNIT_COUNT_DETECTOR` is 419 words, never crossed
the threshold, stayed on the system-instruction path — and detection has kept
working correctly throughout. The only prompt that crossed is the only one
producing fabrications.

That is a plausible mechanism for the exact symptom: with the instruction demoted
from system role to ordinary conversation text — and that text containing a full
JSON schema — a model has materially less pressure to ground its answer in the
images of the following turn.

**Stated honestly: this is the strongest available hypothesis, not a proof.**
Confirming it requires a live Vertex call, which cannot be made here. What *is*
proven is that the delivery path changed between the working and failing runs,
and that the images were attached in both.

### 16.3 The fix

`prompts=None` at `resolver.py:786`, pinning extraction to the system-instruction
path — exactly the delivery run 1 used. `shared.py` is **not** modified (verbatim
lift from `origin/main`), and the triage call is left alone (its prompt is far
from the threshold, and it works).

Nothing is lost by skipping the cache: `load_vertex_ai_client` only ever *creates*
a `CachedContent` (with a fixed `display_name`, never looked up or reused), so on
this path the cached branch added a per-call cache-creation round trip and reused
nothing. The comment at the call site records why the pin is load-bearing, so it
is not "tidied" back later.

### 16.4 Request-composition logging (item 2)

The saved PNGs prove pages **rendered**; they say nothing about what was **sent**.
`_log_request_composition` (`resolver.py:625`) now logs, per attempt: total part
count, image-part count, each part's type/mime/byte-length (**never content**),
the summed inline bytes, and **which delivery path is active** — the last being
invisible at the call site and the thing that changed here. Zero image parts
triggers an explicit warning that any returned counts are invented. Gated on
`UNIT_COUNT_DEBUG`; `_describe_part` is fully defensive and degrades to
`undescribable(...)` rather than raising.

### 16.5 Name grounding (item 3)

`_check_names_grounded` (`resolver.py:906`) asks the question no existing guard
asked: **do the extracted names appear in the document at all?**

The live failure slipped through both existing guards — verified, not assumed:
`_looks_fabricated` requires every count to be *identical* and 10/40/40/10 is not
(confirmed by test: it returns `False`), and `_check_internal_consistency` only
compares rows against a stated total.

If **none** of the extracted names appears in the text layer of **any** candidate
page, the result is flagged `flagged_suspect=true`,
`suspect_reason="names_not_in_text_layer"`. Vector-only, matching the existing
guard's scope. **Flags, never rejects.** Either guard alone is sufficient, and
when both fire `suspect_reason` carries both.

Matching uses **word boundaries**, not the substring test `_looks_fabricated`
uses — that weakness was recorded in §15.5 and is deliberately not repeated here
(a substring test lets a one-character name like `"A"` match inside "plan").

Verified against the real Aurora text layer: the live failure
`STUDIO/1BED/2BED/3BED` is **flagged**, the true `VISTA I–IV = 8/18/8/11` is
**not**, and partial grounding (one real name among unknowns) is accepted, since
a legitimate label can be image-only.

### 16.6 Verification

34/34 new checks pass, plus all four earlier suites and the ranking verification
(Aurora ranking output unchanged). `compileall` clean.

**Not fixed here — the §15.5 substring weakness in `_looks_fabricated` remains.**
The new check uses word boundaries; the older one still does not.
