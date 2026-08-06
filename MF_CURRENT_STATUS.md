# MF Drywall — CURRENT STATUS
**Purpose of this file:** the living snapshot. Read `MF_MASTER_PLAN.md` first for the locked plan and decisions; this file only says where we are right now. **Update this file at every milestone** (keep it short — replace, don't append essays).

**Last updated:** 2026-08-05

---

## Where we are

**REBUILD CODE-COMPLETE, UNTESTED — Phase 1, second implementation.**
Branch `feat/mf-resolver-service` off `origin/main` @ `87e6696e`, **pushed to origin**, not deployed. One live Aurora run has happened (see below); nothing else has been exercised. Full evidence in **`MF_REBUILD_REPORT.md`** (lift provenance per item, bug-fix before/after, findings).

The first implementation (branch `feat/multifamily`) worked end to end but had three defects (fabricated counts, none-found validation bug, and it ran in-process sharing Vertex quota with the sectioning call — the likely page-20 mechanism, master plan §9a). Scope and architecture are settled (master plan §2 = multiplication only; §7 = D5a/D5b/D5c).

**Approach: LIFT, don't rewrite.** MF code lifted file-by-file from `feat/multifamily` (via AST source extraction — verbatim where unchanged), with the three bug fixes applied during the lift. `feat/multifamily` is abandoned but remains the code source — do NOT delete it until the rebuild is deployed and validated.

**⚠ D5b RATIONALE REVISED during the build.** Bounding boxes are NOT part of the automatic upload burst — they fire in `/floorplan_to_2d`, which is user-triggered per page (`main.py:1496`, persisted `main.py:1525`). So there is no "after classification + bounding boxes" moment to sequence behind, and the resolver-vs-sectioning collision window **cannot be eliminated by trigger timing**. Protection against the page-20 failure is therefore the **bounded LLM footprint** (2-attempt fixed-temp extraction, render cap, serialized triage) — not the trigger point. Trigger sits at the end of `/floorplan_to_preview`, the same position the old resolver used; what changed is the footprint, the separate service, and the bug fixes. **Master plan §7 still carries the superseded "eliminated by construction" wording — not yet amended.**

**⚠ PROJECT-TYPE GATE BYPASSED (interim).** `projects.project_type` holds only `COMMERCIAL` / `RESIDENTIAL` — the D4a frontend MF/SF selector was never built, so no gate on that column can work. The trigger therefore **fires for every project**, behind `MF_PROJECT_TYPE_GATE` (default `false` = off). The predicate and query are kept; flip the var to `true` when the selector ships. SF/commercial correctness rests on the **D13 ×1 default**. Report §12.

## First live run — Aurora, `PLAN_1785862120698` (2026-08-05)

**PASSED — the machinery and the D5c bet.**
- **Page 20 shows 3 dropdowns with the resolver enabled.** The page-20 regression did NOT reproduce; the bounded LLM footprint held (master plan §9a, D5c).
- Resolver ran clean: **1 attempt, temperature 0, render cap fired, 23s job, honest provenance.** No hang, no OOM.

**FAILED — extraction read the wrong table.**
**Root cause turned out to be RANKING, not the extractor** (see report §14): the schedule page (fitz index 1 / viewer page 2) was **never sent** — candidates were fitz `[2, 11, 12]` = viewer pages 3, 12, 13. The logged `p2` is the 0-indexed fitz page, i.e. viewer page 3, which is what made ranking look correct. It returned the **accessibility compliance table** from pages 11–12 (Accessible / Type A / Type B Dwelling Units = 10/10/100, all areas null) instead of the real unit schedule on page 2 (4 Vista types, 8/18/8/11 = 45, with areas). Those are ANSI A117.1 categories, so the same apartment is counted under several of them — the number does not describe how many units exist.
Also: extraction logged `total=110` while its own rows sum to **120**, and `disagreement=no` — that flag only ever compared table-vs-prose, never the answer against itself. **Both fixed** (prompt targets the unit type schedule + excludes compliance tables; new internal-sum check flags the mismatch) — **and then the deeper ranking cause was found and fixed too (§14)**. The earlier note that "ranking worked" was wrong: it was an artefact of reading the 0-indexed `p2` as viewer page 2.

**BLOCKED — multiplier still never exercised.**
Zero `[UNIT_MULTIPLY]` lines on "View Estimates". **Cause found: that screen calls `/compute_takeoff` (`main.py:2666`), not `/summarize_takeoff_all` (`main.py:2908`) where the multiplier is wired.** `/compute_takeoff` is per-section and returns one section's totals. Interim: a TEMPORARY scaled-estimates preview behind `MF_SCALED_ESTIMATES_PREVIEW` (default off) so the multiply can be seen in the existing screen — to be removed once the report §5 display decision lands.

**Constraint:** cannot deploy/test right now — deploy + test happen when access returns.

## What the rebuild changes vs the first implementation

| First implementation (feat/multifamily) | Rebuild (feat/mf-resolver-service) |
|---|---|
| Detached in-process task inside xtimator-3d | Separate `unit-count-resolver/` Cloud Run service (D5a) |
| Launched at upload start (believed to overlap the automatic bounding-box burst) | Triggered at end of `/floorplan_to_preview` (**D5b revised — see report §6**; bounding boxes are user-triggered in `/floorplan_to_2d`, so there was no burst to sequence behind and the position is effectively unchanged) |
| Extraction: up to 4 retries at escalating temperature 0.2→0.5 → fabricated counts | Max 2 attempts, fixed temp, sanity checks, store `extraction_failed` instead of inventing (§9) |
| Aggregate-only logging (`types=4 total=40`) — fabrication invisible | Full per-type breakdown + attempts + suspect flag logged and persisted |
| None-found = Pydantic error → wasted retries | None-found = valid result → `none_found` → ×1 |
| Uncapped high-res renders (113M-pixel pages broke extraction) | Pixel cap on the extraction render (4000px longest side / 80MP) |
| ~~Thumbnail triage batches fired freely~~ — **WRONG, corrected during the lift:** batches were ALREADY sequential (`feat/multifamily helper.py:1641` is a plain `for` loop with a blocking `phoenix_call`; no executor, no gather ever existed) | Only the **backoff between calls** was missing. Added, plus no-concurrency made an explicit documented contract (D5c) |

Unchanged from the first implementation (lifted as-is): vector text-scan ranking, D10 precedence, `plans.unit_counts` persistence + provenance, matcher (rule normalization → area fallback → batched LLM), multiplier in `summarize_takeoff_all` with ×1 default, the existing migration file.

## Carried-over findings (evidence from the first implementation — still true)

- **Fabrication (C5 live):** Aurora stored STUDIO/1BED/2BED/3BED = 10/10/10/10 = 40; ground truth is 4 Vista types = 8/18/8/11 = 45. Produced by the retry spiral; stored with full apparent confidence. The rebuild's §9 guardrails exist because of this.
- **Page-20 regression:** 3 enlarged plans → 1 dropdown on the feature branch (3 on main). Mechanism identified in code (master plan §9a): sectioning call fails → silent full-page fallback → one section + garbage-looking walls. Resolver quota pressure during the automatic burst is the likely trigger. D5b/D5c are the mitigation. The `SYSTEM: Bounding Box detection has failed` warning line is the log signature to check on any suspect run.
- **The earlier Aurora "hang" never reproduced** on the merged build (ran clean in 859s, no OOM). Treat as intermittent/unexplained, not fixed.
- **Multiplier never validated on a real match** — Aurora's matcher correctly refused the fabricated names (`applied=False`, ×1). Whether scaling applies per-section vs project-total must be confirmed from the code during the lift (item 11 of the build prompt) and exercised in testing.
- **1800s timeout ceiling is thin** for large PDFs (observed 504 at exactly 1800s on `/floorplan_to_2d`; Aurora preview 1190s). Pre-existing, core-pipeline scope — reinforces keeping the resolver off the request path.

## Build checklist (rebuild)

- [x] Branch `feat/mf-resolver-service` off current main (`87e6696e`)
- [x] Lift MF code from `feat/multifamily` (helper fns, prompts, models, migration — migration byte-identical; it did NOT exist on main)
- [x] New service `unit-count-resolver/` — FastAPI, GCS fetch, 202-then-work, internal ingress + ID-token auth, Dockerfile + deploy workflow (page-classifier shape, GPU/checkpoint steps dropped)
- [x] Bug fix 1 — fabrication guard: 2 attempts max, fixed temp, sanity check, `extraction_failed` state, per-type logging, health in provenance
- [x] Bug fix 2 — none-found accepted as valid (`none_found` → ×1, no retry burn) — 10/10 behavioural checks pass
- [x] Bug fix 3 — pixel cap on extraction renders — 4/4 checks pass
- [x] D5c — backoff added; batches were already serial (see table above)
- [x] Trigger in xtimator-3d: end of `/floorplan_to_preview` (**NOT** "after bounding boxes" — see D5b revision above), MF projects only, `MF_RESOLVER_ENABLED` guard kept, fire-and-forget httpx POST timeout=5
- [x] Matcher + multiplier lifted; per-section `[UNIT_MULTIPLY]` logging added; scaling confirmed = **per-section multiply, project-total-only output** (per-section rows stay unscaled — behaviour unchanged, decision pending)
- [x] `MF_REBUILD_REPORT.md` written (lift provenance, fix diffs, findings) — review before push
- [x] `python -m compileall` clean on both touched service folders (re-verified against the committed tree)
- [x] **`projects.project_type` checked** — contains only `COMMERCIAL` / `RESIDENTIAL`; the D4a frontend MF/SF selector **does not exist yet**, so no gate on this column can work. **Interim decision (Ravikant): trigger fires for ALL projects**, behind `MF_PROJECT_TYPE_GATE` (default `false` = off). Predicate + query kept, not deleted; set the var to `true` when the selector ships. SF/commercial safety = D13 ×1 default. See report §12
- [ ] Apply `migrations/2026-07_add_plans_unit_counts.sql` to Cloud SQL
- [ ] Push, deploy (when access returns), first `gcloud run deploy` creates the service
- [ ] Set `UNIT_COUNT_RESOLVER_URL` on drywall-takeoff-3d (`--update-env-vars`, never `--set-env-vars`). Leave `MF_PROJECT_TYPE_GATE` unset for now
- [ ] **When the frontend MF/SF selector ships:** set `MF_PROJECT_TYPE_GATE=true` and confirm the real MF literal matches `_is_multifamily_project_type`
- [ ] **Watch after deploy:** resolver now runs on EVERY upload, not just MF — check Vertex quota/cost against the load profile D5a was sized for
- [x] Test: Aurora — **3 dropdowns on page 20 CONFIRMED** with the resolver enabled (2026-08-05); D5c footprint held, no regression
- [ ] Test: Aurora — **correct Vista counts 8/18/8/11=45** — FAILED first run (read the accessibility table, 120). Re-run after this session's prompt + consistency fixes
- [ ] **Re-run Aurora 2–3× for stability** — one clean run is not evidence the extractor reliably picks the right table
- [ ] Test: matched-case multiplication actually scales; Ocean View SF regression identical to today (needs `MF_SCALED_ESTIMATES_PREVIEW=true`, or the real §5 display decision)
- [ ] Test: Grace Commons (scanned path, serialized triage)

## Environment / infra notes (unchanged)

- **Cloud Run:** `drywall-takeoff-3d-fbm`, `us-central1`, project `prj-fbm-drywall-dev`. 8 CPU / 32 GiB, `cpu-throttling: false`, `minScale: 1`, `containerConcurrency: 1`, `timeoutSeconds: 1800`. New resolver service gets its own (smaller) config via its workflow.
- **Local gcloud from Brillio Git Bash does NOT work** (corporate TLS inspection). Use Cloud Shell or the Console UI. Do not set `MSYS_NO_PATHCONV=1`.
- **Reading logs:** app stdout does NOT show OOM/container death; pull the system stream (`varlog/system`) and request logs (`run.googleapis.com%2Frequests`) too.
- **`--update-env-vars`, never `--set-env-vars`** (the latter wipes the service's whole env set).

## Deferred / lower-priority open items

- **Decide:** should per-section rows expose SCALED figures, or stay unscaled with only `project_total` scaled? Current behaviour is the latter (lifted unchanged, not a new choice). Totals are correct either way; a UI showing both side by side would look inconsistent.
- **Amend master plan §7** — its D5b text still says overlap is "eliminated by construction"; that is now known false (see D5b revision above). Left alone because the master plan changes only on a formal decision revisit.
- Durability hardening for the resolver service (queue in front, or Cloud Run Job) if fire-and-forget proves flaky.
- Different-region Vertex escape hatch (D5c) — only if contention persists after D5b/D5c.
- Prose-only documents: partial store, "per-type unresolved."
- C1 positional adjustment (Phase 3). 2099 Chestnut table unconfirmed.
- Experimental `RESIDENTIAL_MULTI_FAMILY_SCHEMA` prompt in `prompts.py`: not on the v1 path; repurpose or remove.
- Core-pipeline (NOT MF scope): silent full-page fallback in `detect_bounding_boxes`/`classify_plan` (§9a) deserves a loud degraded-state flag; inline page processing + blocking classify call + 1800s ceiling fragility for large PDFs.

## Update log (one line per milestone)

- 2026-07-17 → 2026-07-23 — First implementation: built, merged with main, deployed, tested (full history in prior status revisions).
- 2026-08-03 — Aurora results: ran clean (no hang) but counts FABRICATED (40 vs true 45); scope locked with team = multiplication only; detached task confirmed failed.
- 2026-08-04 — **Architecture decided (D5a/D5b/D5c):** separate `unit-count-resolver` service, triggered after classification+bounding-boxes persist, bounded/serialized LLM footprint. **Page-20 mechanism found in code (§9a):** silent full-page fallback in `detect_bounding_boxes` on sectioning-call failure. **Rebuild started:** fresh branch `feat/mf-resolver-service` off current main, lifting tested code from `feat/multifamily` with the three bug fixes applied during the lift. Build prompt issued to Claude Code.
- 2026-08-04 — **Rebuild CODE-COMPLETE** (3 commits on `feat/mf-resolver-service`, not pushed/deployed). Service + all three bug fixes + trigger + matcher/multiplier + workflow + `MF_REBUILD_REPORT.md` done; compileall clean. **Three build-time corrections:** (1) D5b re-based — bounding boxes are user-triggered in `/floorplan_to_2d`, not part of the upload burst, so timing can't remove the collision window; the bounded footprint is the real protection. (2) Triage was already serial — only backoff was missing. (3) GCS path carries an `organization_slug` segment (resolver derives it from `user_id`, request body unchanged). **Blocking unknown:** the `projects.project_type` MF literal is unverified.
- 2026-08-06 — **Run 7: delivery fix WORKED, but extraction mis-targeted (report §17).** The model returned R-1/R-2/R-3/R-MH — **real labels off page 1's vicinity-map zoning legend** — so it is reading the page now; it targeted the wrong thing and invented counts (10/10/10/20) and one identical area (1570 ×4) around them. Fixes: (1) prompt now excludes **zoning/land-use district codes** the way it already excluded accessibility tables; (2) three **INPUT→OUTPUT worked examples** with synthetic PLAN-AA/PLAN-BB labels and explicit "procedure only, never copy" framing — pairs, not bare outputs, because output-only examples are what caused the STUDIO/1BED regurgitation; (3) **vector pages now send the text layer alongside the image** (6000 chars/page, image=layout, text=exact values — refines D9 rather than reversing it; scanned unchanged); (4) new **identical_areas** suspect flag. Verified statically on the real PDF: request = 15 parts / 5 images / 8.25 MB, and page 1's sent text yields the true rows 8/18/8/11 with differing areas. **Key finding:** name-grounding CANNOT catch run-7 (R-x are genuinely in the text layer) — only identical_areas does; and had the areas differed, no guard would fire and only the prompt would defend. 60/60 new checks pass; ranking and delivery path untouched.
- 2026-08-06 — **Attachment audit: images WERE attached; the prompt-delivery path had silently flipped (report §16).** Traced render→bytes→Part→Content→generate_content: mime `image/png` correct, bytes non-empty (8.25 MB across 5 candidates, inside Vertex's ~20 MB ceiling), and the path is **byte-identical** to the Aug 4 run that worked — the UNIT_COUNT_DEBUG wrapper changed nothing. **Real cause found:** `load_vertex_ai_client` (shared.py:16) switches delivery at 1024 tokens — below it the prompt is a `system_instruction`, above it it becomes `CachedContent` with **no system instruction at all**, leaving two consecutive user turns (prompt+JSON template, then images). My Aug 5/6 prompt growth (583 → 1,495 words) crossed that threshold; the detector prompt (419 words) never did and detection kept working. **Fixed** by pinning `prompts=None` at the extraction call — the exact delivery run 1 used; `shared.py` untouched, cache reused nothing anyway. Added request-composition logging (part count/mime/bytes + which delivery path) and a name-grounding check (`names_not_in_text_layer`, word-boundary, vector-only, flags never rejects) — verified on the real PDF: the live 10/40/40/10 failure is flagged, true 8/18/8/11 is not. **Caveat:** the delivery flip is the strongest hypothesis, not proven — it needs a live run to confirm.
- 2026-08-06 — **Extraction fix: the prompt was seeding the fabrication (report §15).** The model returned STUDIO/1 BED/2 BED/3 BED twice on a VISTA document because the prompt's own examples supplied those names — worst of all on the `unit_type` field itself (`e.g. "Studio S1", "2 Bed / 2 Bath"`). All example names and numbers replaced with `<TYPE-NAME-AS-PRINTED>`-style placeholders; prompt now states it withholds examples deliberately, adds a visible-only rule and a self-check for generic bedroom-count names. **D8 restored**: partial/half-drawn tables, legends and prose are now named as valid forms (the prompt had said "table" throughout). `_looks_fabricated` widened to ANY identical count (the multiple-of-5 clause let 7/7/7/7 through). New `UNIT_COUNT_DEBUG` (default off) logs the raw model response before parsing and saves the candidate PNGs to GCS, so the next diagnosis reads evidence instead of inferring. Ranking untouched. 50/50 new checks pass. **Found but NOT fixed:** `_looks_fabricated` matches labels by substring, so a one-letter type name like "A" always matches and silently disables the check — pre-existing, needs a word-boundary match.
- 2026-08-05 — **Ranking bug found and fixed (report §14).** The schedule page was never a candidate: `number_rows` hit its cap on 104/112 pages and 12 pages tied at the 4.5 maximum, so the sort's lowest-index tie-break picked `[2, 11, 12]` while the real schedule sat at rank 14. `_UNIT_TYPE_RE` cannot match the marketing name `VISTA` (64 occurrences on that page) but matches `TYPE A/B/C` all over the ICC A117.1 sheets, so the scorer actively preferred compliance tables. Fix: new schedule-header signal (`NO. UNITS`/`GFA`/`SUB-TOTAL`/…) at weight 3.0, candidates 3→5, `sch=` added to per-page logs. Measured on Aurora: index 1 goes 14th → **1st**, tie broken, margin 1.8333. Uncapping alone and widening alone were both tested and both FAIL. No compliance penalty (Ravikant) — so the A117.1 sheets remain candidates 3–5 and correctness now depends on the extractor prompt too; the two fixes are coupled. Tuned on n=1 document.
- 2026-08-05 — **First live Aurora run.** PASSED: 3 dropdowns on page 20 with the resolver on (D5c held), 1 attempt / temp 0 / render cap fired / 23s / honest provenance, no hang or OOM. FAILED: extractor picked the ANSI A117.1 accessibility table (10/10/100, no areas) over the real Vista schedule (8/18/8/11=45), and its own `total=110` vs row-sum 120 went unflagged. BLOCKED: zero `[UNIT_MULTIPLY]` — **"View Estimates" calls `/compute_takeoff`, not `/summarize_takeoff_all`**. Fixes this session: extractor prompt now targets the unit type schedule and rejects compliance tables (Aurora shape as the negative example); new internal-sum check flags `internal_sum_mismatch` without rejecting; TEMPORARY `MF_SCALED_ESTIMATES_PREVIEW` preview on `/compute_takeoff` (default off) so the multiply can finally be observed. Ranking untouched — it was correct.
- 2026-08-04 — **`project_type` resolved, and it invalidated the gate:** DB holds only `COMMERCIAL`/`RESIDENTIAL`; the D4a frontend selector was never built. **Interim decision (Ravikant): resolver fires for ALL projects**, gate preserved behind `MF_PROJECT_TYPE_GATE` (default off) for zero-code-change re-enable. SF safety = D13 ×1. Master plan + status docs committed to the branch (4 commits total). Report §12 records the decision and two consequences to watch (resolver now runs on every upload; D13 protects unmatched sections, not a false match).
