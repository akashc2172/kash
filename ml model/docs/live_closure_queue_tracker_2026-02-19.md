# Live Closure Queue Tracker (2026-02-19)

Source: `data/audit/reingest_manifest_*.csv` + active closure session `51875`.

## Queue Order (from `run_missing_data_closure.py`)
1. `plays` by `(season, season_type)` ascending
2. `subs` by `(season, season_type)` ascending
3. `lineups` by `(season, season_type)` ascending

## Manifest Queue Sizes

### Plays
- 2023 postseason: 1
- 2023 regular: 88
- 2024 regular: 68
- 2025 regular: 141

### Subs
- 2023 postseason: 113
- 2023 regular: 3000
- 2024 regular: 118
- 2025 regular: 8

### Lineups
- 2023 postseason: 113
- 2023 regular: 3000
- 2024 postseason: 120
- 2024 regular: 181
- 2025 postseason: 36
- 2025 regular: 145

## Live Status Snapshot
- Observed complete earlier in logs:
  - `subs 2023 postseason (113)`
  - `subs 2023 regular (3000)` (advanced past 2000 then queue reset)
- Current active queue (inferred from ordering + progress reset):
  - `lineups 2023 regular`
  - Last observed: `1053 / 3000` (~35%)
- Pending after current queue:
  - `lineups 2024 postseason (120)`
  - `lineups 2024 regular (181)`
  - `lineups 2025 postseason (36)`
  - `lineups 2025 regular (145)`

## After Endpoint Queues Finish
1. `build-derived --season 2025 --season-type regular`
2. `scripts/repair_college_feature_store.py --in-place`
3. `college_scripts/build_prospect_career_store_v2.py`
4. `nba_scripts/build_unified_training_table.py`
5. Final `scripts/run_missing_data_audit.py` refresh
