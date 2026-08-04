-- Adds plans.unit_counts (multi-family Unit-Count Resolver output; mirrors the
-- existing multipage_elevation_map JSONB precedent). Provenance is stored inside
-- the JSON payload. Idempotent and safe to re-run.
--
-- This repo has no automated migration runner; apply manually to Cloud SQL, the
-- same way prior schema was applied (see README plans DDL).
ALTER TABLE plans ADD COLUMN IF NOT EXISTS unit_counts JSONB DEFAULT '{}'::jsonb;
