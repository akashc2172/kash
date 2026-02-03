# Career Feature Spec (Prospect Career Store)

**Date**: 2026-02-03
**Source**: `college_scripts/build_prospect_career_store_v2.py`

This spec defines the intended schema for career features derived from `college_features_v1.parquet` (ALL__ALL split only). The career store is emitted in two formats:

- **Wide**: one row per athlete (`prospect_career_v1.parquet`)
- **Long**: one row per athlete-season (`prospect_career_long_v1.parquet`)

## Inputs (Required Columns)

These columns must exist in `college_features_v1.parquet` for a field to be emitted:

- Core totals: `minutes_total`, `fga_total`, `ast_total`, `tov_total`, `stl_total`, `blk_total`
- Shooting rates: `rim_fg_pct`, `three_fg_pct`, `ft_pct`
- Spatial (Tier 2): `avg_shot_dist`, `corner_3_rate`, `corner_3_pct`, `xy_coverage`, `deep_3_rate`, `rim_purity`, `shot_dist_var`
- Context: `team_pace`
- Optional: `heightInches`

Derived columns (always attempted):

- `points_derived`
- `trueShootingPct`
- `poss_total`
- `usage`
- `games_played` (if available from `warehouse.duckdb`)
- `minutes_per_game` (if `games_played` available)
- `poss_per_game` (if `games_played` available)
- Within-season windows (if `within_season_windows_v1.parquet` available)

## Output A: Wide Career Store

**File**: `data/college_feature_store/prospect_career_v1.parquet`

**Primary keys**:
- `athlete_id`

**Core columns**:
- `season`: final college season
- `teamId`: team in final season
- `career_years`: number of college seasons with data

**Career shape columns**:
- `slope_*`: linear slope between first and last season for each available metric
- `delta_*`: final season YoY delta (last season minus prior season)
- `career_wt_*`: recency-weighted career average for each metric

**Final season snapshot**:
- `final_*`: final season value for each available metric

**Other**:
- `delta_height` (if `heightInches` exists)
- `final_poss_per_game` (if `final_games_played` available)
- Breakout timing (continuous, non-hardcoded):
  - `breakout_rank_volume`, `breakout_rank_usage`, `breakout_rank_eff`
  - `breakout_timing_volume`, `breakout_timing_usage`, `breakout_timing_eff`
  - `breakout_timing_avg`

**Notes**:
- All `*_` columns are emitted only when the base metric exists in the feature store.
- Spatial Tier 2 features preserve `NaN` gating when coverage thresholds are not met.
- Missingness is preserved: if a breakout dimension cannot be computed (e.g., usage requires minutes/games), it remains `NaN`.

## Within-Season Windows (v1)

If `data/college_feature_store/within_season_windows_v1.parquet` is present, these columns are joined into the career store.

- `games_played_pg` (games present in `fact_player_game` for that athlete-season)
- Window masks: `has_ws_last5`, `has_ws_last10`, `has_ws_prev5`, `has_ws_prev10`
- Window aggregates: `ws_minutes_last5`, `ws_minutes_last10`, `ws_fga_last5`, `ws_fga_last10`, `ws_pts_last5`, `ws_pts_last10`, `ws_pps_last5`, `ws_pps_last10`, `ws_on_net_rating_last5_mean`, `ws_on_net_rating_last10_mean`
- Window deltas: `ws_delta_pps_last5_minus_prev5`, `ws_delta_pps_last10_minus_prev10`, `ws_delta_minutes_last5_minus_prev5`, `ws_delta_minutes_last10_minus_prev10`
- Breakout timing + masks: `has_ws_breakout_timing_minutes`, `ws_breakout_timing_minutes`, `has_ws_breakout_timing_volume`, `ws_breakout_timing_volume`, `has_ws_breakout_timing_eff`, `ws_breakout_timing_eff`

All of these are expected to be `NaN` when missing or when there are insufficient games.

## Output B: Long Career Store

**File**: `data/college_feature_store/prospect_career_long_v1.parquet`

**Grain**: one row per `(athlete_id, season)` for split `ALL__ALL`.

**Columns (expected)**:

- Identifiers: `athlete_id`, `season`, `teamId`
- Career context: `season_rank`, `career_years`
- Volume: `games_played`, `minutes_total`, `minutes_per_game`, `poss_total`, `poss_per_game`
- Rates: `trueShootingPct`, `usage`, `rim_fg_pct`, `three_fg_pct`, `ft_pct`
- Box totals: `fga_total`, `ast_total`, `tov_total`, `stl_total`, `blk_total`
- Spatial: `avg_shot_dist`, `corner_3_rate`, `corner_3_pct`, `xy_coverage`, `deep_3_rate`, `rim_purity`, `shot_dist_var`

## Rationale

- **Wide store** supports classic “final season + trajectory” models.
- **Long store** supports sequence/temporal models and allows reliability weighting by games/minutes.

## Future Additions (Planned)

- `age_at_season`, `class_year`, transfer indicators
- season-level normalization (ASTz, usage z-scores)
- confidence/uncertainty weights by exposure (minutes, games)
