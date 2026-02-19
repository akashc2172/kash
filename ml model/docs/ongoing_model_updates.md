# Ongoing Model Updates Log

## 2026-02-19 (strict DAG lockdown run)

### What was added
- Mistake-prevention contract doc:
  - `/Users/akashc/my-trankcopy/ml model/docs/mistake_prevention_retrospective_2026-02-19.md`
- Strict hardening runner upgrades:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/harden_and_run_full_pipeline.py`
  - Added stage artifacts:
    - `stage0_input_snapshot.json`
    - `stage*_validation_report.json`
    - `feature_coverage_matrix.csv`
    - `crosswalk_error_catalog.csv`
    - `final_readiness_report.md/json`
  - Added DAG contract reconciliation stage.
  - Added mandatory mistake-doc references pre-Stage0 and pre-Stage6.
  - Added granular validation pack gate (`>=3000` checks) before training.

### Data wiring fixes completed
- Unified table now includes and validates:
  - `college_team_srs` / `team_strength_srs` (from `v_team_season_srs_proxy`)
  - `class_year`, `season_index`, `age_at_season`, `has_age_at_season`
  - leverage rates: `high_lev_att_rate`, `garbage_att_rate`, `leverage_poss_share`
  - explicit within availability: `has_within_window_data`
- Script changes:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/build_unified_training_table.py`
  - `/Users/akashc/my-trankcopy/ml model/models/player_encoder.py`
  - `/Users/akashc/my-trankcopy/ml model/scripts/run_granular_pipeline_audit.py` (import-path hardening)

### Run outcome
- Hardening run:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/hardening_run_20260219_055643/final_release_audit.json`
- Gate result:
  - `critical_passed=True`
  - one non-critical stage failure persists (college artifact duplicate rows upstream)
- Stage 6 validation pack:
  - `approx_checks_total=7712`
  - strict critical gates passed after dead-branch masking policy.
- Full train + inference completed in strict flow.

### Inference export refresh
- Latest predictions:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/prospect_predictions_20260218_220232.parquet`
- Season ranking exports refreshed:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_tabs.xlsx`
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_by_season_csv/`

## 2026-02-19

### Completed
- Added NBA pre-train readiness gate:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/run_nba_pretrain_gate.py`
  - Outputs to `/Users/akashc/my-trankcopy/ml model/data/audit/nba_pretrain_gate*.json`
- Wired gate into pipeline runner:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/run_training_pipeline.py` (`--gate`, and gate step in `--all`)
- Added temporal-decay weighting in latent training:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/train_latent_model.py`
  - Recency half-life and floor configurable via CLI.
- Added season-level recalibration artifact:
  - `season_recalibration.json` saved per latent run.
- Applied recalibration at inference:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/nba_prospect_inference.py`
  - Adds `pred_peak_rapm_raw` and `pred_peak_rapm_recalibration_offset`.
- Added rolling retrain orchestrator:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/run_rolling_retrain.py`

### Newly Added (within-season development emphasis)
- Added freshman/early-career modulation of within-season gate in encoder:
  - `/Users/akashc/my-trankcopy/ml model/models/player_encoder.py`
- Behavior:
  - Base within-season gate still depends on availability + last10 exposure.
  - Additional modulation now uses `career_years` to learn stronger or weaker within-season influence for younger vs older players.
  - Intended to better capture "freshman improves during season" effects.

### Current interpretation
- This is not full online active learning.
- It is an iterative adaptive loop with:
  - rolling retraining
  - time-decayed sample weighting
  - post-train season recalibration
  - early-career-sensitive within-season gating

### Next candidates
- Add explicit game-order recency weighting in within-season feature builder (`build_within_season_windows_v1.py`).
- Add freshman-specific calibration diagnostics (residual by `career_years==1`).
- Add automatic model selection across rolling anchors (pick best by out-of-time validation).

## 2026-02-19 (follow-up fixes after antigravity review)

### Validated vs not validated
- Validated:
  - TIER1 sparsity was real and severe.
  - WITHIN columns were effectively dead (all null/unused).
  - pathway model had schema mismatch risk.
- Not valid anymore:
  - `transfer_conf_delta` is populated after enrichment.

### Fixes implemented
- `build_unified_training_table.py`
  - Added robust final-season possession proxy (`college_poss_proxy`).
  - Added per-100-possession rates:
    - `college_ast_total_per100poss`
    - `college_tov_total_per100poss`
    - `college_stl_total_per100poss`
    - `college_blk_total_per100poss`
  - Backfilled legacy per-40 columns from per-100 proxy when minutes missing.
  - Added fallback derivation for `final_usage` from final-season count stats.
  - Added fallback fill for `college_team_pace` (same-scale proxy) to avoid near-all-null.
  - Fixed assisted-share derivation by supporting actual source naming and creating:
    - `college_assisted_share_rim`
    - `college_assisted_share_mid`
    - `college_assisted_share_three`
  - Set explicit within-season defaults when upstream windows absent (0-filled + explicit gating behavior).
  - Updated normalization/residual features to use per-100-possession columns.
  - Updated creation column rename list to actual source names (`assisted_made_*`).
- `player_encoder.py`
  - Switched TIER1 activity features from per-40 to per-100-possession.
  - Moved sparse impact/wingspan features out of TIER1 into CAREER_BASE.
  - Result: no low-coverage features remaining in active TIER1.
- `build_transfer_context_v1.py`
  - Enforced same-source pace delta (team_pace-to-team_pace only) to prevent scale discontinuity.
  - `transfer_pace_delta` is now null when true pace unavailable (quality-first behavior).
- `train_pathway_model.py`
  - Removed non-existent `college_on_net_rating` from feature list.

### Verification snapshot
- Tests:
  - `pytest -q tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py` → **5 passed**.
- Smoke train:
  - `train_latent_model.py --epochs 1 --batch-size 256` → **passes**.
- Coverage improvements:
  - TIER1 columns now all present and >5% non-null (no dead primary-branch features).
  - `final_usage`, `final_usage_z`, `final_usage_team_resid` now 100% non-null.
  - Assisted-share columns now present with strong coverage.

### Remaining upstream constraint
- WITHIN branch remains source-limited: defaults are explicit zeros until within-season windows are truly built/populated from player-game data.

## 2026-02-19 (follow-up: trajectory + transfer pace backfill)

### Additional fixes implemented
- `build_prospect_career_store_v2.py`
  - Added robust fallback for `usage` when minutes-based estimate is unavailable:
    - primary: minutes + team pace estimate (existing behavior)
    - fallback: team-season possession share (`poss_total / team_poss_total`)
  - Result: historical `usage` trajectories are populated for pre-2025 rows.
- `build_transfer_context_v1.py`
  - Restored non-null transfer pace signal using calibrated `pace_proxy`:
    - prefer `team_pace` when present
    - otherwise use scaled fallback from possession-based proxies
  - Added `transfer_pace_proxy_flag` to indicate when proxy (not pure `team_pace`) was used.

### Validation snapshot
- Rebuilt:
  - `prospect_career_v1.parquet`
  - `prospect_career_long_v1.parquet`
  - `fact_player_transfer_context.parquet`
  - `unified_training_table.parquet`
- Coverage checks after rebuild:
  - `slope_usage`: 100%
  - `career_wt_usage`: 100%
  - `delta_usage`: 100%
  - `final_usage`: 100%
  - `transfer_conf_delta`: 100%
  - `transfer_pace_delta`: 100%
- Tests:
  - `pytest -q tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py` → **5 passed**.

## 2026-02-19 (iterative residual reweight loop)

### Implemented
- Added epoch-wise adaptive sample reweighting in `/Users/akashc/my-trankcopy/ml model/nba_scripts/train_latent_model.py`.
- Reweighting now runs after each epoch and updates train sample weights for the next epoch based on residual error grouped by:
  - `college_final_season`
  - `within_mask` (within-season feature availability)
- Uses empirical-Bayes style shrinkage on group errors and bounded multipliers to prevent instability.
- Existing temporal decay is preserved; effective weight is now:
  - `temporal_weight * adaptive_weight`

### Dataset plumbing changes
- `ProspectDataset` now keeps:
  - `temporal_weight`
  - `adaptive_weight`
  - dynamic `sample_weight`
- Target weights are now assembled dynamically in `__getitem__` using base target weight × current sample weight.
- Added `set_adaptive_weights()` to apply epoch-level updates without rebuilding dataset/dataloader.

### New training flags
- `--iterative-reweight/--no-iterative-reweight` (default enabled)
- `--reweight-min-group`
- `--reweight-shrinkage`
- `--reweight-strength`
- `--reweight-min-mult`
- `--reweight-max-mult`

### Verification
- 1-epoch smoke run succeeds and logs reweight diagnostics:
  - `Iterative reweight | groups=... global_abs_err=... mult[min/mean/max]=...`
- Core tests pass:
  - `tests/test_wiring_edge_cases.py`
  - `tests/test_dev_rate_label_math.py`

## 2026-02-19 — Antigravity verification sign-off

### Full pipeline re-validated
Rebuilt entire chain: `build_prospect_career_store_v2.py` → `build_transfer_context_v1.py` → `build_fact_player_college_development_rate.py` → `build_unified_training_table.py`. Verified final coverage after all Codex patches:

### Coverage results (unified table, 1114 rows, 412 cols)

| Feature | Before | After |
|---|---|---|
| `college_ast/tov/stl/blk_total_per100poss` | 0% (per40 dead) | **100%** |
| `final_usage` / `_z` / `_team_resid` | 0.3% | **100%** |
| `slope_usage` / `career_wt_usage` / `delta_usage` | 0.2% | **100%** |
| `college_team_pace` | 0.3% | **100%** |
| `college_assisted_share_rim` | missing | **97.8%** |
| `college_assisted_share_three` | missing | **85.8%** |
| `final_has_ws_last10` + all WITHIN cols | 0% | **100%** (explicit zero defaults) |
| `transfer_conf_delta` | 0% | **100%** |
| `transfer_pace_delta` | 0% | **100%** (proxy-based, flagged) |

### Reviewed and confirmed correct
- **Career store usage fallback**: team-season possession share (`poss_total / team_poss_total`) is the right signal when minutes are missing — captures load/role without requiring per-game minutes.
- **Transfer pace proxy**: 3-tier fallback (team_pace → poss/games → calibrated poss×scale) with `transfer_pace_proxy_flag` for downstream quality control. Sound.
- **Iterative reweight loop**: empirical-Bayes shrinkage `(n/(n+τ)) * group_err + (τ/(n+τ)) * global_err` with bounded multipliers [0.7, 1.4] and mean-normalization to prevent implicit LR shifts. Architecture is solid.
- **Encoder column moves**: sparse impact/wingspan correctly in gated CAREER_BASE, per100poss rates in TIER1. No dead features in primary branch.

### Remaining upstream constraints (expected, not code bugs)
- `impact_on_net_raw` / `rIPM_tot_std` etc.: 1% — bounded by RAPM data availability
- `wingspan_minus_height_in`: 0% — no wingspan source ingested yet
- WITHIN branch: all zeros — within-season window builder not yet created/run
- `transfer_pace_proxy_flag`: 100% proxy — `team_pace` only in 2025 data

### Tests
- `pytest tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py` → **5/5 passed**
- 1-epoch smoke train → **passes** (Tier1=23, Tier2=9, Career=28, Within=6)

## 2026-02-19 — Manual historical rebuild + exposure hardening

### Completed
- Rebuilt manual historical combined PBP from all scrape folders `2011-2012` .. `2023-2024`:
  - output: `/Users/akashc/my-trankcopy/ml model/data/fact_play_historical_combined.parquet`
  - rows: `19,365,111`
  - seasons: `2011..2023` (13 seasons)
- Recomputed historical RAPM variants on rebuilt file:
  - output: `/Users/akashc/my-trankcopy/ml model/data/historical_rapm_results_enhanced.csv`
  - rows: `56,735`
- Rebuilt impact stack with enhanced historical RAPM mapping:
  - output: `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/college_impact_stack_v1.parquet`
  - rows: `34,914`
  - historical RAPM mapped rows: `25,071`
- Patched unified build bug where null derived stats were overwriting existing stats with `0`.
- Added mapped historical exposure merge path (name-bridged) in unified builder:
  - mapped exposure rows: `25,133`
  - fills `minutes_total/games_played/tov_total` using conservative max-coalesce.

### QA / Gate outputs
- Granular pipeline audit run:
  - approx checks total: `8,000`
  - dead input columns: `7` (within-season placeholder branch only)
- API/manual coverage audit exported:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/api_manual_coverage_by_season_20260219.csv`
  - total uncovered games (api OR manual): `7,888`

### Current status
- Data wiring materially improved, but rankings are still not at desired quality thresholds for key anchors.
- Remaining blocker is not basic feature presence; it is coverage and model-behavior calibration under incomplete game coverage.
