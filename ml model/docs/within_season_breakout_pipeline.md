# Within-Season Breakout Pipeline (Windows + Timing)

**Date**: 2026-02-03
**Status**: Implementation plan + DAG

## Goal

Capture *within-season* improvement patterns (e.g., “only got good in March”) in a way that:

- Works across eras and missing coverage
- Does **not** hardcode “late breakout = bad”
- Can interact with learned archetypes (prototype-dependent effects)
- Avoids fake zeros: missing remains `NaN` with explicit masks

This complements *career-stage breakout* (year-to-year timing) already in the career store.

## Data Inputs

Primary source (modern coverage):
- DuckDB `data/warehouse.duckdb`
  - `fact_player_game` (player-game box-ish totals + on/off ratings)
  - `dim_games` (season + startDate)

Key assumption:
- Within-season windows are best-effort when player-game data exists; otherwise features are `NaN`.

## Outputs

1. Per athlete-season within-season windows table:
- `data/college_feature_store/within_season_windows_v1.parquet`
- Grain: one row per `(athlete_id, season)`

2. Career store augmented:
- `data/college_feature_store/prospect_career_long_v1.parquet` (adds per-season window features)
- `data/college_feature_store/prospect_career_v1.parquet` (final_* snapshots include final season’s within-season features)

3. Trajectory stub unchanged in concept, but it can now include the new sequence features.

## Features (v1)

We compute two kinds of signals:

### A) Window aggregates (last N games)
Computed when `games_played >= N` (else `NaN`), for `N=5,10`:

- `ws_minutes_last5`, `ws_minutes_last10`
- `ws_fga_last5`, `ws_fga_last10`
- `ws_pts_last5`, `ws_pts_last10`
- `ws_pps_last5`, `ws_pps_last10` (PPS = pts/fga; NaN if denom 0)
- `ws_on_net_rating_last5_mean`, `ws_on_net_rating_last10_mean`

And comparisons:

- `ws_delta_pps_last5_minus_prev5`, `ws_delta_pps_last10_minus_prev10`\n+- `ws_delta_minutes_last5_minus_prev5`, `ws_delta_minutes_last10_minus_prev10`

Masks:

- `has_ws_last5`, `has_ws_last10`

### B) Within-season breakout timing (continuous)
A continuous timing signal in `[0,1]`:

- `ws_breakout_timing_minutes`
- `ws_breakout_timing_volume` (FGA)
- `ws_breakout_timing_eff` (PPS proxy)

Definition (v1):

- Sort games by `startDate`
- Compute rolling-3 metric (minutes, FGA, PPS)
- Take the index of the maximum rolling-5 value
- Normalize to `[0,1]` by `(idx / (games_played-1))`

Masks:

- `has_ws_breakout_timing_*` (1 if enough games)

## Missingness Rules (Critical)

- If a player-season is missing required underlying rows (no player-game data), output is **`NaN`** and mask=0.
- If a value is truly 0 (e.g., player played but recorded 0 FGA), output is 0.
- Never impute missing to 0.

## How The Model Uses This (Archetype-Dependent)

We want “late within-season breakout” to matter differently by archetype.

Two safe patterns:

1) Concatenate into decoder:
- `y_hat = Decoder([z, archetype_probs, ws_features])`

2) Mixture-of-Experts head:
- Experts correspond to learned archetypes
- Gating is `archetype_probs`
- Each expert can learn different sensitivities to within-season breakout

## DAG (End-to-End)

```mermaid
flowchart TD
  A[DuckDB: fact_player_game] --> B[Join dim_games (season,startDate)]
  B --> C[Sort by player-season-date]
  C --> D[Window Aggregates: last10, prev10]
  C --> E[Rolling5 max timing -> ws_breakout_timing_*]
  D --> F[within_season_windows_v1.parquet]
  E --> F

  F --> G[Merge into prospect_career_long_v1 (per season)]
  G --> H[Derive final_* snapshots + slopes + deltas]
  H --> I[prospect_career_v1.parquet]

  I --> J[build_unified_training_table.py]
  J --> K[Latent encoder inputs]
  K --> L[Archetype-conditioned head(s)]
```

## Validation / QA

- Row count sanity: `within_season_windows_v1` should have <= unique `(athlete_id, season)` present in `fact_player_game`.
- Masks sanity: `has_ws_last10` implies `ws_minutes_last10` not null.
- NaN safety: seasons without player-game coverage must have `has_ws_* = 0` and window fields `NaN`.

## Increasing Coverage (How To Get More Seasons/Games)

Within-season features require player-game coverage (`fact_player_game`). If you only see 2025 or only a handful of games per player, that’s expected when only a partial season was ingested/built.

To expand coverage (examples):

1. Ingest additional seasons into the warehouse (requires CBD API key):
   - `python -m cbd_pbp.cli ingest-season --season 2024 --season-type regular --out data/warehouse.duckdb`
   - Repeat for seasons you want (e.g., 2019–2025)

2. Rebuild derived tables (produces/refreshes player-game facts):
   - `python -m cbd_pbp.cli build-derived --season 2024 --season-type regular --out data/warehouse.duckdb`

3. Re-run within-season windows builder:
   - `python college_scripts/build_within_season_windows_v1.py`

After that, rebuild the career store so the new windows features are joined:
   - `python college_scripts/build_prospect_career_store_v2.py`
