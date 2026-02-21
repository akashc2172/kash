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

## 2026-02-19 Closure Policy Realignment (Manual-first subs/lineups)

### What changed
1. `scripts/run_missing_data_audit.py` now enforces historical policy for `subs/lineups`:
   - API retry manifests are restricted to modern seasons via `--subs-lineups-api-min-season` (default `2024`).
   - `subs/lineups` readiness floor uses effective dual-source coverage on modern seasons (`API ∪ manual bridge`).
2. Historical `subs/lineups` API queue inflation is now treated as policy mismatch, not required closure work.

### Post-change closure run status (2023-2025 scope)
- `reingest_manifest_subs.csv`: `126` rows (2024 regular `118`, 2025 regular `8`)
- `reingest_manifest_lineups.csv`: `362` rows (2024 regular `181`, 2025 postseason `36`, 2025 regular `145`)
- `subs_lineups_floor`: **pass**
  - `subs_coverage_rate`: `0.9798`
  - `lineups_coverage_rate`: `0.9420`
  - `expected_games_evaluated`: `6243` (2024+ only)

### Remaining blockers after closure execute
1. `plays_participants_complete`: fail (`298` unresolved game IDs across 2023-2025).
2. `fact_parity`: fail (fact tables still not at required parity floor).
3. `feature_store_integrity`: fail (`duplicate_rows=256`, `null_team_pace_rate=0.926`, `null_conference_rate=0.234`).
4. `target_coverage`: fail (historical RAPM seasons missing `2023, 2024, 2025`; year1 EPM null rate above threshold).

### Bridge diagnostics
- Manual scrape folders present through `2023-2024`, but bridge mapping remains sparse:
  - `bridge_game_cbd_scrape`: `386` rows total.
- Coverage snapshot (2023-2025):
  - 2023 expected `3113`, API `3024`, manual bridge `25`
  - 2024 expected `3120`, API `3052`, manual bridge `38`
  - 2025 expected `3123`, API `2982`, manual bridge `0`
- Conclusion: remaining play/participant misses are now primarily a crosswalk/bridge expansion problem, not a subs/lineups policy problem.

## 2026-02-19 Closure Re-run (2023-2025) + Gate Reconciliation

### Run executed
- `python3 /Users/akashc/my-trankcopy/ml model/scripts/run_missing_data_closure.py --db data/warehouse.duckdb --audit-dir data/audit --start-season 2023 --end-season 2025 --execute --cli-python python3.13`

### Key outcomes
1. Bridges rebuilt with manual scrape mapping:
- `bridge_game_cbd_scrape`: `14,437` rows.
- `bridge_player_cbd_scrape`: player bridge intentionally skipped in this pass (`0` rebuilt rows from command path), game bridge remains the primary coverage key.
2. Readiness gates:
- `/Users/akashc/my-trankcopy/ml model/data/audit/model_readiness_gate.json` -> `passed: true`
- `/Users/akashc/my-trankcopy/ml model/data/audit/model_readiness_dual_source.json` -> `passed: true`
- Required family uncovered counts are now zero after provider-empty handling.
3. Unified training table rebuilt:
- rows: `1,065`
- `y_peak_ovr`: `87.4%`
- `year1_epm_tot`: `76.8%`
- `dev_rate_y1_y3_mean`: `100%`

### Remaining queue reality (non-blocking under current gate policy)
- `reingest_manifest_plays.csv`: `235` rows (all currently source-limited/provider-empty in this scope).
- `reingest_manifest_subs.csv`: `85` rows (2024/2025).
- `reingest_manifest_lineups.csv`: `289` rows (2024/2025).
- Current closure policy treats required families as satisfied via API/manual + provider-empty classification; subs/lineups remain optional/non-blocking.

## 2026-02-19 Ordered Next-Steps Execution (Queue -> Historical -> Rebuild -> Validate)

### 1) Modern queue classification (subs/lineups) completed
- Added classifier utility: `/Users/akashc/my-trankcopy/ml model/scripts/classify_endpoint_retry_queue.py`
- Output artifacts:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/endpoint_retry_queue_classification.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/endpoint_retry_queue_summary.json`
- Classification result:
  - `subs`: `85` rows -> `source_limited_after_retry`
  - `lineups`: `289` rows -> `source_limited_after_retry`
- Cache updated:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/retry_policy_cache.json` terminalized these rows to prevent repeated API churn.

### 2) Historical reconstruction pass status
- Verified combined historical parquet coverage:
  - `/Users/akashc/my-trankcopy/ml model/data/fact_play_historical_combined.parquet`
  - seasons present: `2011..2023` (13 seasons), rows: `19,365,111`.
- RAPM recomputed across full available historical span:
  - `/Users/akashc/my-trankcopy/ml model/data/historical_rapm_results_enhanced.csv`
  - rows: `56,735`, seasons: `2011..2023`.

### 3) Rebuilt impact/trajectory artifacts
- Rebuilt:
  - `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/college_impact_stack_v1.parquet`
  - `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/fact_player_transfer_context.parquet`
  - `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/fact_player_college_development_rate.parquet`
  - `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/prospect_career_v1.parquet`
  - `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/prospect_career_long_v1.parquet`
  - `/Users/akashc/my-trankcopy/ml model/data/training/trajectory_stub_v1.parquet`

### 4) Unified rebuild + strict validation pack
- Rebuilt:
  - `/Users/akashc/my-trankcopy/ml model/data/training/unified_training_table.parquet`
- Gate status:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/nba_pretrain_gate.json` -> `passed: true`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/model_readiness_gate.json` -> `passed: true`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/model_readiness_dual_source.json` -> `passed: true`
- Test pack:
  - `pytest tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py tests/test_encoder_gating.py` -> `7 passed`
  - `python3 tests/quick_validate.py` -> `12/12 checks passed`

### Post-audit verification (queue terminalization preserved)
- Re-ran audit after classification updates:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/model_readiness_gate.json` -> `passed: true`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/model_readiness_dual_source.json` -> `passed: true`
- Verified retry cache state:
  - `subs terminal`: `85`
  - `lineups terminal`: `289`
  - `subs/lineups retryable`: `0`
- Re-ran pytest pack after classifier updates:
  - `7 passed`

## 2026-02-19 Full Training + Inference Stack Run

### Executed training stack
1. Baseline orchestrator:
- `python3 nba_scripts/run_training_pipeline.py --all`
- rebuilt unified table + trajectory stub, passed pretrain gate, trained XGBoost baseline.

2. Latent model:
- `python3 nba_scripts/train_latent_model.py`
- output: `/Users/akashc/my-trankcopy/ml model/models/latent_model_20260219_121715`

3. Generative model:
- `python3 nba_scripts/train_generative_model.py`
- output: `/Users/akashc/my-trankcopy/ml model/models/generative_model_20260219_121822`

4. Pathway model:
- `python3 nba_scripts/train_pathway_model.py --skip-diagnostics`
- output: `/Users/akashc/my-trankcopy/ml model/models/pathway_model_20260219_121912`

### Executed inference/export
1. Inference on new latent checkpoint:
- `python3 nba_scripts/nba_prospect_inference.py --model-path models/latent_model_20260219_121715/model.pt`
- predictions:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/prospect_predictions_20260219_121934.parquet`

2. Season ranking export (with names + per-season tabs):
- `python3 scripts/export_inference_rankings.py`
- csv:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current.csv`
- xlsx tabs:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_tabs.xlsx`
- per-season csv dir:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_by_season_csv`

### Validation after full run
- `pytest tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py tests/test_encoder_gating.py` -> `7 passed`
- `python3 tests/quick_validate.py` -> `12/12 checks passed`

### Quick rank sanity sample (qualified pool)
- Zion Williamson (2019): rank `1`
- Ja Morant (2019): rank `15`
- Anthony Edwards (2020): rank `67`
- Cade Cunningham (2021): rank `11`
- Austin Reaves (2021): rank `6`
- Paolo Banchero (2022): rank `1`

## 2026-02-19 — Data-quality hardening patch (RAPM split + dunk + ON/OFF)

### Scope
- Hardened historical RAPM home/away partitioning logic.
- Added dunk-rate feature source and guaranteed final-table presence.
- Added ON/OFF-derived impact fields to impact stack and unified aliases.

### Checks executed
1. Build checks
- `python3 college_scripts/build_college_impact_stack_v1.py` -> success.
- `python3 nba_scripts/build_unified_training_table.py` -> success.

2. Contract checks
- `python3 nba_scripts/emit_full_input_dag.py` -> success.
- Unified-table key integrity:
  - rows: `1065`
  - duplicate `nba_id`: `0`

3. Coverage spot checks
- `college_dunk_rate`: non-null `98.5%`, non-zero `86.29%`.
- `college_stl_total_per100poss`: non-null `100%`, non-zero `97.65%`.
- `college_rapm_standard`: non-null `8.08%` (source-limited, expected).
- `college_on_net_rating`: non-null `6.29%` (source-limited, expected).

4. Unit tests
- `pytest tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py` -> `5 passed`.

### Outcome
- Strict DAG gate now passes after the dunk-rate wiring fix.
- RAPM split parser no longer depends on exact team-name equality.
- ON/OFF OFF-side metrics are present in feature tables but remain sparse at current source coverage and are not promoted into active always-on encoder branch.

## 2026-02-19 — Pre-2025 true-10 reconstruction + pre-RAPM lineup gate wiring

### Code changes
- Added:
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/reconstruct_historical_onfloor_v3.py`
- Updated:
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/calculate_historical_rapm.py`
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/utils/clean_historical_pbp_v2.py`
  - `/Users/akashc/my-trankcopy/ml model/run_historical_pipeline.py`

### Validation run (bounded sample, 2019–2023)
1. Reconstruction:
- `python3 college_scripts/reconstruct_historical_onfloor_v3.py --start-season 2019 --end-season 2023 --max-games-per-season 200 --no-api-append ...`
- outputs:
  - `data/fact_play_historical_combined_v2_sample.parquet`
  - `data/audit/historical_lineup_quality_by_game_sample.csv`
  - `data/audit/historical_lineup_quality_by_season_sample.csv`

2. Sample lineup quality audit:
- `pct_rows_len10` >= `0.985` in each sampled season.
- `pct_rows_placeholder` = `0.0` in sampled seasons.
- sampled seasons all `gate_pass=True` for lineup audit.

3. RAPM diagnostics-only on sample artifact:
- `python3 college_scripts/calculate_historical_rapm.py --input-parquet data/fact_play_historical_combined_v2_sample.parquet --lineup-season-audit-csv data/audit/historical_lineup_quality_by_season_sample.csv --diagnostics-only ...`
- `valid_5v5_rate` near `0.99` across sampled seasons.
- split gate failures were only `n_stints<2000` (expected from bounded sample size), not lineup-fidelity regressions.

### Additional regression checks
- `pytest tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py` -> `5 passed`.
- `python3 tests/quick_validate.py` -> `12/12 checks passed`.
