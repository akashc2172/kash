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

- `scripts/repair_college_feature_store.py`
  - Deterministically collapses duplicate `(season, athlete_id, split_id)` rows.
  - Backfills `team_pace` and `conference` from team-season references.
  - Writes a report to `data/audit/feature_store_repair_report.json`.

- `scripts/run_missing_data_closure.py`
  - Orchestrates staged backfill:
    1. Postseason manifest fetch for missing seasons
    2. Plays pass (`plays` + participants via plays ingest)
    3. Substitutions pass (`subs`)
    4. Lineups pass (`lineups`)
    5. Rebuild derived/final artifacts
    6. Final readiness audit
  - Dry-run by default, use `--execute` for live run.

## CLI Extension Added

- `python -m cbd_pbp.cli resume-ingest-endpoints`
  - New endpoint-scoped ingest command to enforce staged backfill order.
  - Examples:
    - `python -m cbd_pbp.cli resume-ingest-endpoints --season 2018 --season-type regular --endpoints plays --out data/warehouse.duckdb`
    - `python -m cbd_pbp.cli resume-ingest-endpoints --season 2018 --season-type regular --endpoints subs --out data/warehouse.duckdb`
    - `python -m cbd_pbp.cli resume-ingest-endpoints --season 2018 --season-type regular --endpoints lineups --out data/warehouse.duckdb`

## Suggested Execution

1. Preview commands:
   - `python scripts/run_missing_data_closure.py`
2. Execute full run:
   - `python scripts/run_missing_data_closure.py --execute`
3. Validate final gate:
   - Inspect `data/audit/model_readiness_gate.json`
