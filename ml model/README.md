# CollegeBasketballData PBP → Feature Store Pipeline (v1)

This repo ingests play-by-play + lineups + subs + season stats from https://api.collegebasketballdata.com
and materializes:

- `dim_*` tables (games/teams/players/venues/conferences)
- `fact_*` raw tables (plays, lineups, substitutions, lines, ratings, recruiting, draft)
- derived fact tables:
  - `fact_player_game`, `fact_team_game`
  - windowed long tables: `fact_player_window`, `fact_team_window`
- optional: RAPM (ridge APM) from lineup stints
- optional: xPts + over/under expectation + passing value-over-exp

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
