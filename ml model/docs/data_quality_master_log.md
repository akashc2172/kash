# Data Quality Master Log

Last updated: `2026-02-19 22:03:00`

## Purpose
- Track every full pipeline validation and hardening run.
- Enforce strict visibility of data integrity before model decisions.
- Keep unresolved issues explicit until fixed.
- Enforce no-repeat policy from `mistake_prevention_retrospective_2026-02-19.md`.

## Mistake-Prevention Anchors (Must Reference in Strict Runs)
- Pre-Stage0 guard review: `mistake_prevention_retrospective_2026-02-19.md`
- Pre-Stage6 guard re-check: `mistake_prevention_retrospective_2026-02-19.md`

## Latest Artifacts
- Hardening audit: `/Users/akashc/my-trankcopy/ml model/data/audit/hardening_run_20260219_055643/final_release_audit.json`  
- Readiness report: `/Users/akashc/my-trankcopy/ml model/data/audit/hardening_run_20260219_055643/final_readiness_report.md`  
- Granular audit: `/Users/akashc/my-trankcopy/ml model/data/audit/granular_pipeline_audit_20260218_215554/summary.json`  
- Pretrain gate: `/Users/akashc/my-trankcopy/ml model/data/audit/nba_pretrain_gate_20260218_215649.json`

## Latest Hardening Stage Results
- `stage0_snapshot`: **PASS** (critical_failure=False)
- `stage1_college_validation`: **FAIL (non-critical only)** (critical_failure=False)
  - `cardinality` (critical=False): college feature store still contains duplicate athlete/season/split rows upstream
- `stage2_nba_target_hardening`: **PASS** (critical_failure=False)
- `stage3_crosswalk_validation`: **PASS** (critical_failure=False)
- `stage4_dag_contract`: **PASS** (critical_failure=False)
- `stage4_unified_rebuild`: **PASS** (critical_failure=False)
- `stage5_gate_checks`: **PASS** (critical_failure=False)
- `stage6_training`: **PASS** (critical_failure=False)
- `stage7_inference`: **PASS** (critical_failure=False)

## Latest Strict-Gate Verdict
- `overall_passed`: **False** (contains non-critical stage1 failure)
- `critical_passed`: **True** (all critical gates passed; run allowed to complete)
- Strict policy behavior validated: critical failures hard-stop; non-critical issues logged only.

## Latest Granular Coverage Snapshot
- `training_rows`: `1065`
- `training_cols`: `423`
- `input_columns_checked`: `102`
- `target_columns_checked`: `5`
- `all_numeric_columns_checked`: `375`
- `approx_checks_inputs_targets`: `1712`
- `approx_checks_all_numeric`: `6000`
- `approx_scalar_checks_run`: `7712`
- `dead_input_columns_count`: `9` (all known masked/deferred branch fields)
- `low_cov_input_columns_count`: `32`
- `duplicate_nba_id_crosswalk`: `0`
- `duplicate_athlete_id_crosswalk`: `0`
- `star_sanity_fail_count`: `4`

## Current Hard-Fail Policy
- Stop on duplicate target keys, missing contract columns, or gate failures.
- Non-critical failures remain listed here until resolved.

## Open Issues (Latest Run)
### Dead Inputs
- `age_at_season` (no external age source yet; currently proxy-derived context)
- `has_age_at_season` (tracks missing external age source)
- `final_has_ws_last10`
- `final_ws_minutes_last10`
- `final_ws_pps_last10`
- `final_ws_delta_pps_last5_minus_prev5`
- `final_has_ws_breakout_timing_eff`
- `final_ws_breakout_timing_eff`
- `has_within_window_data`

### Low-Coverage Inputs
- `career_wt_three_fg_pct`
- `college_assisted_share_mid`
- `college_assisted_share_three`
- `college_avg_shot_dist`
- `college_corner_3_pct`
- `college_corner_3_rate`
- `college_deep_3_rate`
- `college_dev_p10`
- `college_dev_p50`
- `college_dev_p90`
- `college_dev_quality_weight`
- `college_recruiting_rank`
- `college_recruiting_rating`
- `college_recruiting_stars`
- `college_rim_purity`
- `college_shot_dist_var`
- `college_three_fg_pct`
- `college_three_fg_pct_team_resid`
- `college_three_fg_pct_z`
- `delta_three_fg_pct`
- `final_three_fg_pct`
- `has_transfer_context`
- `slope_three_fg_pct`
- `transfer_conf_delta_mean`
- `transfer_event_count`
- `transfer_max_shock`
- `transfer_mean_shock`
- `transfer_pace_delta_mean`
- `transfer_role_delta_mean`

## Notes
- Low coverage can be expected for optional branches (transfer/dev/spatial) depending on source availability.
- Ranking quality failures are tracked separately from wiring integrity; both must pass before declaring readiness.
- Mistake-prevention references were enforced pre-Stage0 and pre-Stage6 in strict runner.

## 2026-02-18 Patch Addendum (Impact Stack RAPM Split Wiring)

### What was fixed
- `college_scripts/build_college_impact_stack_v1.py` now augments from historical scrape RAPM and maps to `athlete_id` using season + normalized name bridge from `stg_shots` + `dim_games`.
- Historical RAPM file selection now prefers `data/historical_rapm_results_enhanced.csv` (contains `o_rapm`, `d_rapm`) and falls back to `data/historical_rapm_results_lambda1000.csv`.
- Canonical O/D RAPM fields are now populated with real lineup-derived values where available:
  - `rIPM_off_std`
  - `rIPM_def_std`
  - plus aligned non-garbage / lev_wt / rubber / recency aliases.

### Current measured coverage after rebuild
- `college_impact_stack_v1.parquet`:
  - rows: `28,260`
  - source mix: `historical_scrape=18,417`, `api_only=9,843`
  - `rIPM_off_std` non-null: `65.17%`
  - `rIPM_def_std` non-null: `65.17%`
  - O/D seasons present from historical split file: `2012, 2013, 2014, 2015, 2017`
- `training/unified_training_table.parquet` (post-2011 cohort):
  - rows: `1,065`
  - `college_rapm_standard` non-null: `20.09%`
  - `college_o_rapm` non-null: `20.00%`
  - `college_d_rapm` non-null: `13.71%`
  - no duplicate `nba_id`.

### Validation status
- Smoke retrain completed (`train_latent_model.py`, 30 epochs).
- Inference completed with full train-serve feature parity (`tier1=49, tier2=9, career=47, within=6`).
- Regression tests:
  - `tests/test_wiring_edge_cases.py`: pass
  - `tests/test_dev_rate_label_math.py`: pass

### Remaining issue
- Predicted distributions remain over-shrunk in model outputs (`pred_peak_rapm_raw` low variance), so rankings are still not acceptable even after wiring fixes.
- This is now a model-behavior issue (not missing-column wiring) and requires a dedicated calibration/loss/target-variance pass.

## 2026-02-18 Model-Behavior Fix Addendum (Ranking Compression)

### Root cause addressed
- Primary RAPM head had collapsed output spread in inference (`pred_peak_rapm_raw` near-constant), causing poor ranking separation.

### Code changes
- `/Users/akashc/my-trankcopy/ml model/models/prospect_model.py`
  - RAPM primary loss now uses masked MSE directly (disabled heteroscedastic NLL for RAPM mean fitting).
  - Added RAPM variance-matching regularizer:
    - `losses['rapm_var'] = (std(pred_rapm) - std(target_rapm))^2`
    - weighted by new `lambda_rapm_var`.
- `/Users/akashc/my-trankcopy/ml model/nba_scripts/train_latent_model.py`
  - Added CLI arg `--lambda-rapm-var` (default `0.20`).
  - Wired through to `ProspectLoss`.
- `/Users/akashc/my-trankcopy/ml model/nba_scripts/nba_prospect_inference.py`
  - Replaced narrow 6-feature rank combiner with richer meta feature matrix including:
    - model outputs (`pred_peak_rapm`, `pred_year1_epm`, `pred_dev_rate`, survival prob),
    - college usage/efficiency/playmaking/defense/rebounding,
    - recruiting + team strength + leverage + impact + career context.
  - Keeps ridge fallback; attempts `HistGradientBoostingRegressor` when available and selected by validation RMSE.

### Measured outcome (latest run)
- New model run: `latent_model_20260218_223756`
- New inference: `prospect_predictions_20260218_223809.parquet`
- Output spread improved:
  - `pred_peak_rapm_raw std`: `~0.7006` (previous collapsed run was ~`0.004`)
  - `pred_peak_rapm std`: `~0.7147`
- NBA-linked season rank sanity (illustrative):
  - Zion Williamson (2019): rank `2`
  - Ja Morant (2019): rank `6`
  - Paolo Banchero (2022): rank `2`
  - Anthony Edwards (2020): rank `77`
  - Cade Cunningham (2021): rank `68`

### Artifact updates
- `prospect_predictions_20260218_ensemble_best.parquet` now points to the latest improved run (`223809`).
- Season-tab Excel export with names:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/prospect_rankings_by_season_20260218_223809.xlsx`

## 2026-02-19 Incremental Quality Log

### Critical fixes applied
1. Historical cleaner rewritten to stream per-season parts and materialize combined parquet without all-row in-memory accumulation.
2. Unified-table merge fixed to avoid null->zero overwrite of `ast/stl/blk/tov/games` when derived stats are missing.
3. Added historical exposure backfill mapping (season + normalized name bridge) into unified assembly.

### Coverage audit artifacts
- `/Users/akashc/my-trankcopy/ml model/data/audit/api_manual_coverage_by_season_20260219.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/granular_pipeline_audit_20260219_085953/summary.json`

### Coverage finding (needs action)
- Combined API/manual game coverage still has `7,888` uncovered expected games (2011-2025).
- Largest uncovered cohorts include 2011, 2012, 2013, 2021.
- This directly impacts exposure features and ranking reliability for affected players/seasons.
