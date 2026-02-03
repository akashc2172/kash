# CollegeBasketballData PBP → Feature Store Pipeline (v1)

This repo ingests play-by-play + lineups + subs + season stats from https://api.collegebasketballdata.com
and materializes:

- `dim_*` tables (games/teams/players/venues/conferences)
- `fact_*` raw tables (plays, lineups, substitutions, lines, ratings, recruiting, draft)
- derived fact tables:
  - `fact_player_game`, `fact_team_game`
  - windowed long tables: `fact_player_window`, `fact_team_window`
- **Enhanced RAPM** (7 variants including leverage-weighted, O/D split, rubber-band adjusted)
- optional: xPts + over/under expectation + passing value-over-exp

## New Features (Jan 2025)

### Enhanced RAPM Variants
The `calculate_historical_rapm.py` script now computes 7 RAPM variants:
- `rapm_standard`: Possession-weighted (original)
- `rapm_leverage_weighted`: Weights stints by leverage index (clutch signal)
- `rapm_high_leverage`: Only high/very_high leverage stints
- `rapm_non_garbage`: Excludes garbage time
- `o_rapm` / `d_rapm`: Offensive/Defensive split
- `rapm_rubber_adj`: Rubber-band effect correction

### Win Probability & Leverage Model
- `compute_win_probability()`: Time-weighted logistic model
- `compute_leverage_index()`: Expected WP swing from possession outcomes (pbpstats methodology)
- Leverage buckets: `garbage`, `low`, `medium`, `high`, `very_high`

### New Feature Blocks
See `college_scripts/compute_enhanced_features.py`:
- **Athleticism**: `dunk_rate`, `putback_rate`, `transition_freq`, `rim_pressure_index`
- **Defense Activity**: `deflection_proxy`, `contest_proxy`
- **Decision Discipline**: `pressure_handle_proxy`, `clutch_shooting_delta`
- **Shot Creation**: `self_creation_rate`, `self_creation_eff`
- **Context**: `leverage_poss_share`

## Quickstart

1) Create a `.env` (or export env vars):

```
CBD_API_KEY=your_key_here
CBD_BASE_URL=https://api.collegebasketballdata.com
```

2) Install (recommended in a venv):

```
pip install -r requirements.txt
```

3) Run a season ingest + build:

```
python -m cbd_pbp.cli ingest-season --season 2025 --season-type regular --out data/warehouse.duckdb
python -m cbd_pbp.cli build-derived --season 2025 --season-type regular --out data/warehouse.duckdb
python -m cbd_pbp.cli build-windows --season 2025 --season-type regular --out data/warehouse.duckdb --windows season_to_date,rolling10,vs_top_quartile_opponents
```

4) Export a wide training matrix (optional):

```
python -m cbd_pbp.cli export-wide --season 2025 --season-type regular --out data/warehouse.duckdb --window-ids season_to_date,rolling10 --dest data/player_asof_wide.parquet
```

## Notes

- Plays include `assistAthleteId` and `shotInfo.range/location` so assisted/unassisted splits are exact.
- Windowed tables are LONG: key = (athleteId, season, teamId, asOfGameId, window_id).
  Export wide on demand.
- PlayType taxonomy is configurable in `cbd_pbp/config/playtype_rules.yaml`.

## Docs Index

- `docs/INDEX.md`
- `docs/WORKSPACE_STATUS.md`
