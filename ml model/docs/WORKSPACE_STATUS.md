# Workspace Status (ML Model)

**Date**: 2026-02-03
**Scope**: Working summary for data ingestion, feature store, and model training in `ml model/`.

## Current State

- Training pipeline scripts are in place and ready to run once data files exist.
- Historical scraping is in progress to backfill 2010-2014, 2016, 2018-2019.
- Historical lineup reconstruction has been validated for 2015 and 2017.
- Career store builder outputs both wide and long formats.
- Trajectory stub builder is wired into the training pipeline (optional step).

## Key Documents

- `ml model/PROJECT_MAP.md`
- `ml model/docs/next_steps_plan.md`
- `ml model/docs/ml_model_master_plan.md`
- `ml model/docs/model_architecture_dag.md`
- `ml model/docs/latent_input_plan.md`
- `ml model/docs/career_feature_spec.md`

## Immediate Next Steps

1. Finish historical scraping for missing seasons, then clean:
   - `college_scripts/scrapers/scrape_ncaa_master.py`
   - `college_scripts/utils/clean_historical_pbp_v2.py`

2. Backfill minutes and turnovers from historical PBP:
   - `college_scripts/derive_minutes_from_historical_pbp.py`

3. Confirm required data files are present (or copied) before training:
   - `data/college_feature_store/college_features_v1.parquet`
   - `data/college_feature_store/prospect_career_v1.parquet`
   - `data/warehouse_v2/dim_player_nba_college_crosswalk.parquet`
   - `data/warehouse_v2/fact_player_peak_rapm.parquet`
   - `data/warehouse_v2/fact_player_year1_epm.parquet`

4. Run pipeline checks and training:
   - `python3 nba_scripts/run_training_pipeline.py --check`
   - `python3 nba_scripts/run_training_pipeline.py --all`

## Career Data Formatting (Current)

The career store builder produces two artifacts:

- Wide career summary: `data/college_feature_store/prospect_career_v1.parquet`
  - Final-season snapshot, career slopes, deltas, and recency-weighted averages.

- Long career table: `data/college_feature_store/prospect_career_long_v1.parquet`
  - One row per athlete-season (ALL__ALL only) with per-season metrics and
    optional `games_played`, `minutes_per_game`, and `poss_per_game`.

- Trajectory stub: `data/training/trajectory_stub_v1.parquet`
  - One row per athlete with list-valued season sequences (for future sequence models).

## Within-Season Breakout (Current)

- Builder: `college_scripts/build_within_season_windows_v1.py`
- Output: `data/college_feature_store/within_season_windows_v1.parquet`
- Joined into career store outputs when present.

This split supports both trajectory modeling (long) and final-season modeling (wide).

## Notes

- Do not change `clean_historical_pbp_v2.py` output schema for `onFloor`.
- Keep Tier 2 spatial features as `NaN` when coverage thresholds are unmet.
- Avoid leakage by excluding post-draft NBA stats from features.
