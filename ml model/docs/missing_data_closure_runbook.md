# Missing Data Closure Runbook

## Scripts Added

- `scripts/run_missing_data_audit.py`
  - Generates:
    - `data/audit/missing_games_by_endpoint.csv`
    - `data/audit/missing_games_by_season.csv`
    - `data/audit/feature_store_integrity_report.json`
    - `data/audit/reingest_manifest_plays.csv`
    - `data/audit/reingest_manifest_subs.csv`
    - `data/audit/reingest_manifest_lineups.csv`
    - `data/audit/model_readiness_gate.json`
    - `data/audit/source_void_games.csv`
    - `data/audit/model_readiness_dual_source.json`
    - `data/audit/retry_policy_cache.json`

- `scripts/repair_college_feature_store.py`
  - Deterministically collapses duplicate `(season, athlete_id, split_id)` rows.
  - Backfills `team_pace` and `conference` from team-season references.
  - Writes a report to `data/audit/feature_store_repair_report.json`.

- `scripts/run_missing_data_closure.py`
  - Orchestrates staged backfill:
    0. Rebuilds manual scrape bridges (`build-bridges`) so in-progress NCAA.org files are accounted for
    1. Postseason manifest fetch for missing seasons
    2. Plays pass (`plays` + participants via plays ingest)
    3. Substitutions pass (`subs`)
    4. Lineups pass (`lineups`)
    5. Rebuild derived/final artifacts
    6. Final readiness audit
  - Enforces retry suppression from `retry_policy_cache.json`.
  - Writes endpoint attempt logs to `data/audit/ingest_attempts.csv`.
  - Dry-run by default, use `--execute` for live run.

## CLI Extension Added

- `python -m cbd_pbp.cli resume-ingest-endpoints`
  - New endpoint-scoped ingest command to enforce staged backfill order.
  - Supports `--only-game-ids-file` and `--skip-game-ids-file` for quota-safe targeting.
  - Examples:
    - `python -m cbd_pbp.cli resume-ingest-endpoints --season 2018 --season-type regular --endpoints plays --out data/warehouse.duckdb`
    - `python -m cbd_pbp.cli resume-ingest-endpoints --season 2018 --season-type regular --endpoints subs --out data/warehouse.duckdb`
    - `python -m cbd_pbp.cli resume-ingest-endpoints --season 2018 --season-type regular --endpoints lineups --out data/warehouse.duckdb`

## Suggested Execution

1. Preview commands:
   - `python scripts/run_missing_data_closure.py`
2. Execute full run:
   - `python scripts/run_missing_data_closure.py --execute`
3. Force recheck source-void IDs only when intentionally revalidating provider state:
   - `python scripts/run_missing_data_closure.py --execute --force-recheck`
4. Audit-only refresh (no ingest calls):
   - `python scripts/run_missing_data_closure.py --audit-only --execute`
5. Validate gates:
   - API execution gate: `data/audit/model_readiness_gate.json`
   - Dual-source availability gate: `data/audit/model_readiness_dual_source.json`
6. Validate critical docs:
   - `python scripts/check_doc_health.py`
7. Inspect manual scrape accounting:
   - `python scripts/report_manual_scrape_state.py`

## No-Waste Rules

1. Do not repeatedly retry IDs classified as `provider_empty` unless `--force-recheck` is set.
2. Respect cooldown for retryable failures from `retry_policy_cache.json`.
3. Prefer manual-source coverage for availability accounting when API is source-empty.
4. Use `--skip-bridges` only if you intentionally want to keep previous bridge state.
