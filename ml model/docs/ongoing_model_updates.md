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

## 2026-02-19 — RAPM parser hardening + impact/athleticism wiring patch

### Implemented
- `college_scripts/calculate_historical_rapm.py`
  - Replaced brittle exact-match home-team split with normalized + fuzzy resolver:
    - header-derived home candidate (`| Score |` rows),
    - `onFloor` team-label candidates,
    - normalized/fuzzy matching fallback.
  - Added unresolved-home warning counter for runtime diagnostics.

- `college_scripts/build_college_impact_stack_v1.py`
  - Added ON/OFF-derived impact fields from `fact_player_game` + `fact_team_game` identity:
    - `impact_off_ortg_raw`, `impact_off_drtg_raw`, `impact_off_net_raw`
    - `impact_on_off_ortg_diff_raw`, `impact_on_off_drtg_diff_raw`, `impact_on_off_net_diff_raw`
  - Preserved historical RAPM augmentation path unchanged.

- `nba_scripts/build_unified_training_table.py`
  - Added dunk-rate source loader from `stg_shots` + `fact_play_raw` text (`%dunk%` on made rim shots).
  - Preserved dunk-rate through final-season aggregation and exposed `college_dunk_rate`.
  - Added canonical aliases for ON/OFF impact fields (`college_off_*`, `college_on_off_*`).

- `models/player_encoder.py`
  - Added `college_dunk_rate` to active Tier1 inputs.
  - Kept sparse OFF/ONOFF-diff columns out of active Tier1 until coverage improves.

- `nba_scripts/nba_prospect_inference.py`
  - Added dunk/impact signals to ranking meta-feature builder (`college_dunk_rate`, `college_off_net_rating`, `college_on_off_net_diff`).

### Validation
- Rebuilt impact stack:
  - `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/college_impact_stack_v1.parquet`
  - rows: `34,914`, cols: `45`
- Rebuilt unified training table:
  - `/Users/akashc/my-trankcopy/ml model/data/training/unified_training_table.parquet`
  - rows: `1,065`, cols: `445`
  - `college_dunk_rate` coverage: `98.5%` non-null, `86.29%` non-zero
- DAG input contract check:
  - `python3 nba_scripts/emit_full_input_dag.py` passes.
- Tests:
  - `pytest tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py` -> `5 passed`.

### Current constraint (known)
- ON/OFF-derived OFF columns are populated in impact stack for season `2025` only (source-limited by `fact_player_game` coverage), so they remain passive for core encoder until broader season coverage exists.

## 2026-02-19 — Additional pipeline-alignment coverage pass (activity + threshold proxies)

### Implemented
- Expanded unified activity proxy loader (`stg_shots` + `fact_play_raw`) to materialize:
  - `dunk_rate`, `dunk_freq`, `putback_att_proxy`, `transition_freq`
- Carried activity fields through final-season aggregation into unified columns:
  - `college_dunk_rate`, `college_dunk_freq`, `college_putback_att_proxy`, `college_transition_freq`
- Added derived threshold proxies in unified table:
  - `college_putback_rate`
  - `college_rim_pressure_index`
  - `college_contest_proxy` (provisional form pending foul-attribution closure)
- Promoted to active Tier1 inputs:
  - `college_dunk_rate`, `college_dunk_freq`, `college_putback_rate`, `college_rim_pressure_index`, `college_contest_proxy`

### Coverage snapshot (post rebuild)
- `college_dunk_rate`: 98.5% non-null
- `college_dunk_freq`: 100% non-null
- `college_rim_pressure_index`: 100% non-null
- `college_stl_total_per100poss`: 100% non-null
- `college_blk_total_per100poss`: 100% non-null
- `college_contest_proxy`: 100% non-null
- `college_putback_rate`: 8.3% non-null
- `college_transition_freq`: 100% non-null, 0% non-zero (source-limited)

### Notes
- OFF-side On/Off fields remain source-limited in the current post-2011 cohort and are kept as passive columns until coverage expands.
- Full matrix exported to:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/pipeline_alignment_feature_coverage_2026-02-19.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/pipeline_alignment_feature_coverage_by_season_2026-02-19.csv`

## 2026-02-19 — RAPM split-quality hard gates (season include/exclude)

### Implemented
- `college_scripts/calculate_historical_rapm.py`
  - Added season-level diagnostics for RAPM split integrity:
    - `n_stints`
    - `valid_5v5_rate`
    - `nonempty_split_rate`
    - `unresolved_home_rate`
    - `parse_fail_rate`
    - `unique_players_5v5`
  - Added hard gate logic per season with strict mode default:
    - fails closed when thresholds are violated
    - supports explicit `--include-seasons` and `--exclude-seasons` overrides
  - Added diagnostics artifact output:
    - `--diagnostics-csv` (default `data/audit/historical_rapm_split_quality.csv`)
  - Enforced valid 5v5 split rows for RAPM solve matrix.
  - Moved season include/exclude filtering to run immediately after load for faster diagnostics loops.

### Why this was added
- Prevent silent RAPM degradation in late seasons (2019–2023) from propagating into downstream training targets.
- Make season inclusion explicit and auditable rather than implicit.

### Diagnostic findings (2019–2023)
- Diagnostics artifact:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/historical_rapm_split_quality_latest.csv`
- Current gate results:
  - all seasons 2019–2023 fail strict split-quality gates.
- Key failure pattern:
  - `valid_5v5_rate` decays sharply from `0.431` (2019) to `0.027` (2023),
  - side sizes average ~1–3 players instead of 5.
- Interpretation:
  - this is source lineup-fidelity collapse (non-10-player `onFloor` semantics), not unresolved home/away parsing (`unresolved_home_rate=0`).

## 2026-02-19 — Pre-2025 true 10-player reconstruction (v3) + pre-RAPM lineup gate

### Implemented
- Added new reconstructor:
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/reconstruct_historical_onfloor_v3.py`
- Deterministic game-level state machine:
  - explicit lineup checkpoints (`TEAM For ...`),
  - substitution propagation (`Enters/Leaves`, `SUB IN/OUT`),
  - starter inference from pre-sub activity,
  - strict ghost fill from observed game roster only.
- Added explicit quality/provenance columns to reconstructed output:
  - `lineup_source`, `lineup_confidence`, `lineup_quality_flag`.
- Added audit outputs:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/historical_lineup_quality_by_game.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/historical_lineup_quality_by_season.csv`
- Updated RAPM script defaults and pre-gate wiring:
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/calculate_historical_rapm.py`
  - default input now `fact_play_historical_combined_v2.parquet`
  - added `--lineup-season-audit-csv` + required pre-lineup gate (fail-closed).
- Updated historical orchestrator order:
  - `/Users/akashc/my-trankcopy/ml model/run_historical_pipeline.py`
  - v3 reconstruct -> diagnostics-only RAPM -> gated RAPM solve -> export.

### Validation (bounded sample run)
- Sample run command:
  - `python3 college_scripts/reconstruct_historical_onfloor_v3.py --start-season 2019 --end-season 2023 --max-games-per-season 200 --no-api-append ...`
- Sample season audit result:
  - `pct_rows_len10` >= 0.985 across 2019–2023
  - `pct_rows_placeholder` = 0.0 across sample seasons
  - all sample seasons `gate_pass=True` in lineup quality audit.
- RAPM diagnostics on sample reconstructed artifact:
  - `valid_5v5_rate` ~0.99 across 2019–2023
  - split gate failures on sample were only `n_stints<2000` (expected from bounded sample size), not lineup fidelity.

## 2026-02-20 — Full 2011–2023 reconstruction + gated RAPM + full hardening pipeline run

### Implemented
- Executed full historical reconstruction for manual-scrape seasons `2011..2023` using season-chunked runs of:
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/reconstruct_historical_onfloor_v3.py`
  - mode: `--no-api-append` and one season per invocation.
- Merged per-season outputs into canonical artifacts (DuckDB out-of-core merge):
  - `/Users/akashc/my-trankcopy/ml model/data/fact_play_historical_combined_v2.parquet`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/historical_lineup_quality_by_game.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/historical_lineup_quality_by_season.csv`
- Recomputed RAPM with hard lineup + split gates:
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/calculate_historical_rapm.py`
  - outputs:
    - `/Users/akashc/my-trankcopy/ml model/data/historical_rapm_results_enhanced.csv`
    - `/Users/akashc/my-trankcopy/ml model/data/audit/historical_rapm_split_quality.csv`
- Rebuilt downstream artifacts:
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/build_college_impact_stack_v1.py`
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/build_unified_training_table.py`
- Ran strict hardening runner end-to-end:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/harden_and_run_full_pipeline.py`
  - stages 0–8 all passed; final audit written.

### Key results
- Reconstructed historical combined rows: `19,365,111` (seasons 2011–2023).
- Lineup season audit rows: `13`; all `gate_pass=True`.
- RAPM split diagnostics rows: `13`; all `gate_pass=True`.
- Enhanced RAPM output rows: `319,683` player-season rows.
- Impact stack rebuild:
  - `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/college_impact_stack_v1.parquet`
  - `44,934` rows; `has_impact_ripm` coverage `99.8%`.
- Unified training table:
  - `/Users/akashc/my-trankcopy/ml model/data/training/unified_training_table.parquet`
  - `1,065` rows, `451` columns
  - target coverage:
    - `y_peak_ovr` `87.4%`
    - `year1_epm_tot` `76.8%`
    - `dev_rate_y1_y3_mean` `100%`.

### Validation/tests
- `pytest tests/test_wiring_edge_cases.py -q` -> `4 passed`.
- `pytest tests/test_dev_rate_label_math.py -q` -> `1 passed`.
- `python3 tests/quick_validate.py` -> `12/12 checks passed`.

### Ranking exports
- Inference ranking exports refreshed:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current_qualified.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_tabs.xlsx`
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_by_season_csv/`
- Historical RAPM Excel tabs refreshed:
  - `/Users/akashc/my-trankcopy/ml model/data/historical_rapm_rankings.xlsx`
  - sheets for each season `2011..2023`.

## 2026-02-20 — NBA→NCAA crosswalk hardening with `d_y`/`d_n` + HTML DAG canonical dashboards

### Implemented
- Rebuilt crosswalk builder:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/build_nba_college_crosswalk.py`
- Matching remains NBA-anchored and now includes basketball-excel draft signal enrichment:
  - source: `data/basketball_excel/all_players.parquet`
  - fields used: `pid`, `bbr_pid`, `nm`, `d_y`, `d_n`
- Added deterministic match methods and tiers:
  - `match_method`: `bbr_id_link`, `pid_link`, `name_draft_fuzzy`
  - `match_tier`: `id_exact`, `draft_constrained_high`, `draft_constrained_medium`, `manual_review`
- Added hard publish gates:
  - required schema columns
  - duplicate `nba_id` fail
  - duplicate `athlete_id` fail
  - high-confidence regression tolerance check versus prior snapshot
- Added new audit artifacts:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/crosswalk_nba_to_college_coverage.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/crosswalk_ambiguity_catalog.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/audit/crosswalk_unmatched_nba.csv`

### Output metrics
- Final crosswalk:
  - `/Users/akashc/my-trankcopy/ml model/data/warehouse_v2/dim_player_nba_college_crosswalk.parquet`
  - rows: `1231` (from `1124`)
  - duplicates: `0` on `nba_id`, `0` on `athlete_id`
- Coverage:
  - all NBA: `1231/2461` (`50.0%`)
  - 2011–2024 cohort: `1082/1304` (`83.0%`)
- Tier distribution:
  - `id_exact`: `1119`
  - `draft_constrained_high`: `80`
  - `draft_constrained_medium`: `10`
  - `manual_review`: `21`

### Pipeline integration
- Rebuilt unified table after new crosswalk:
  - `/Users/akashc/my-trankcopy/ml model/data/training/unified_training_table.parquet`
  - rows: `1082`
  - target coverage after rebuild:
    - `y_peak_ovr`: `87.2%`
    - `year1_epm_tot`: `76.4%`
    - `dev_rate_y1_y3_mean`: `100.0%`
- Updated stage-3 hardening validation required crosswalk schema in:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/harden_and_run_full_pipeline.py`

### Canonical HTML dashboards
- Added generator:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/generate_html_dag_dashboards.py`
- Generated canonical HTML artifacts:
  - `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`
  - `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`
  - `/Users/akashc/my-trankcopy/ml model/docs/diagrams/layered_execution_dashboard.html`
  - `/Users/akashc/my-trankcopy/ml model/docs/diagrams/crosswalk_quality_dashboard.html`
- Updated markdown mirror headers in:
  - `/Users/akashc/my-trankcopy/ml model/docs/model_architecture_dag.md`
  - `/Users/akashc/my-trankcopy/ml model/docs/current_inputs_dag_2026-02-18.md`
  - `/Users/akashc/my-trankcopy/ml model/docs/antigravity_full_pipeline_layered_dag_2026-02-19.md`
  - `/Users/akashc/my-trankcopy/ml model/docs/generative_model_dag.md`
  - plus dashboard index entries in `/Users/akashc/my-trankcopy/ml model/docs/INDEX.md`

## 2026-02-20 — Objective pivot to EPM-first (training + inference ranking)

### Implemented
- Updated `/Users/akashc/my-trankcopy/ml model/nba_scripts/train_latent_model.py`:
  - Added objective profiles:
    - `epm_first` (default)
    - `rapm_first`
    - `balanced`
  - Added resolver that applies profile defaults with optional per-lambda overrides.
  - Added EPM evaluation metrics (`epm_rmse`, `epm_mae`, `epm_corr`) alongside RAPM metrics.
  - Added objective-aware monitor metric in epoch logs (`epm_rmse` when EPM-first).
  - Persists `objective_profile` and resolved `objective_weights` in model config.
- Updated `/Users/akashc/my-trankcopy/ml model/nba_scripts/nba_prospect_inference.py`:
  - Reads model objective metadata.
  - If objective is `epm_first`, ranking score defaults to `pred_year1_epm`.
  - Emits:
    - `pred_rank_target`
    - `pred_rank_score`
    - keeps `pred_peak_rapm_rank_score` for backward compatibility.
- Updated `/Users/akashc/my-trankcopy/ml model/scripts/export_inference_rankings.py`:
  - Ranking export now prefers `pred_rank_score` when available.
  - Exports `pred_rank_target` to make ranking objective explicit in CSV/XLSX.

### Validation
- Python compile checks passed for:
  - `train_latent_model.py`
  - `nba_prospect_inference.py`
  - `export_inference_rankings.py`
- CLI help checks confirmed new args:
  - `--objective-profile {epm_first,rapm_first,balanced}`
  - lambda override flags now optional overrides.

### Ranking export extension
- Updated ranking exporter to emit crosswalk-matched cohort views:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current_matched.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current_matched_qualified.csv`
- Added fields:
  - `is_nba_matched`
  - `season_rank_matched`
- XLSX tabs now include matched-qualified sheets (`{season}_mq`).

## 2026-02-20 — EPM ranking-quality optimization loop (matched-qualified cohort)

### Implemented
- Updated `/Users/akashc/my-trankcopy/ml model/nba_scripts/train_latent_model.py`:
  - Added validation ranking metrics for EPM on qualified cohort:
    - `epm_ndcg10`
    - `epm_top10_recall`
    - `epm_spearman`
    - `epm_rank_seasons`
  - EPM-first runs now monitor `epm_ndcg10` for early stopping (`monitor_mode=max`) instead of raw loss-only selection.
  - Persisted monitor metadata (`monitor_metric`, `monitor_mode`) in model config.

### Latest run artifacts
- Model:
  - `/Users/akashc/my-trankcopy/ml model/models/latent_model_20260220_114321`
- Predictions:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/prospect_predictions_20260220_114345.parquet`
- Ranking exports:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current_matched_qualified.csv`
- `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_tabs.xlsx`

## 2026-02-20 — Active learning loop (expanding yearly retrain, warm-started)

### Implemented
- Updated `/Users/akashc/my-trankcopy/ml model/nba_scripts/train_latent_model.py`:
  - Added `--init-model-path` warm-start support for checkpoint chaining.
- Updated `/Users/akashc/my-trankcopy/ml model/nba_scripts/run_rolling_retrain.py`:
  - Added `--expanding-window` (default on): training window grows each anchor year.
  - Added `--base-train-start` for fixed historical start.
  - Added `--warm-start` (default on): anchor `t` initializes from anchor `t-1` best checkpoint.
  - Added objective forwarding (`--objective-profile`).
  - Writes per-anchor report artifacts:
    - `data/audit/rolling_retrain_report_<timestamp>.csv`
    - `data/audit/rolling_retrain_report_<timestamp>.json`

### Behavior
- This now matches the intended loop:
  - train through year `t-1`,
  - validate recent season,
  - test on year `t`,
  - carry posterior weights forward to year `t+1`,
  - repeat with more observed future EPM each year.

## 2026-02-20 — Active loop refinements (cold-start expanding windows + RAPM maturity gate)

### Implemented
- Updated `/Users/akashc/my-trankcopy/ml model/nba_scripts/run_rolling_retrain.py`:
  - expanding-window runs now disable warm-start by default (can override with `--allow-warm-start-expanding`).
  - passes `--asof-year` and `--rapm-min-nba-seasons` through to training.
  - report includes `warm_start_effective`.
- Updated `/Users/akashc/my-trankcopy/ml model/nba_scripts/train_latent_model.py`:
  - RAPM target mask now supports maturity gating:
    - `--asof-year`
    - `--rapm-min-nba-seasons` (default `3`)
  - immature RAPM rows are excluded from RAPM supervision while EPM/dev/survival still train.

### Validation run
- Rolling smoke run (anchors 2020→2021) completed with:
  - `expanding_window=1`
  - `warm_start_effective=0`
  - `objective_profile=epm_first`
  - `monitor_metric=epm_ndcg10`
  - report: `/Users/akashc/my-trankcopy/ml model/data/audit/rolling_retrain_report_20260220_115737.csv`

## 2026-02-20 — Activity Pipeline Restore + Hard Gate Pass

### Implemented
- Restored end-to-end activity feature contract:
  - `college_dunk_rate`
  - `college_dunk_freq`
  - `college_putback_rate`
  - `college_rim_pressure_index`
  - `college_contest_proxy`
- Added strict gate runner:
  - `nba_scripts/run_activity_feature_gate.py`
- Unified table assembly hardening:
  - coalesces suffixed activity merge columns (`_x/_y`) into canonical names
  - explicit provenance/mask columns retained
  - impact alias parity fields emitted for inference:
    - `college_off_net_rating`
    - `college_on_off_net_diff`
    - `college_on_off_ortg_diff`
    - `college_on_off_drtg_diff`

### Coverage outcome (post-fix, unified table)
- core activity non-null: 100% for all 5 core activity fields
- gate status: PASS

### Gate artifacts
- `/Users/akashc/my-trankcopy/ml model/data/audit/activity_restore_stage0_snapshot.json`
- `/Users/akashc/my-trankcopy/ml model/data/audit/activity_feature_gate_report.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/activity_feature_gate_report.json`

### Train + refresh
- Trained model:
  - `/Users/akashc/my-trankcopy/ml model/models/latent_model_20260220_191735/model.pt`
- Refresh audit:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/inseason_refresh_20260220_191810.json`
- Rankings exports:
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current.csv`
  - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_tabs.xlsx`

### Dashboards regenerated (canonical HTML)
- `/Users/akashc/my-trankcopy/ml model/docs/diagrams/model_architecture_dashboard.html`
- `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`
- `/Users/akashc/my-trankcopy/ml model/docs/diagrams/layered_execution_dashboard.html`
- `/Users/akashc/my-trankcopy/ml model/docs/diagrams/crosswalk_quality_dashboard.html`
- `/Users/akashc/my-trankcopy/ml model/docs/diagrams/full_pipeline_active_learning_dashboard.html`
- `/Users/akashc/my-trankcopy/ml model/docs/diagrams/activity_feature_quality_dashboard.html`

## 2026-02-21 — College Physicals Backfill (Season-by-Season) + Hard Gate

### Implemented
- Added new ingest pipeline:
  - `/Users/akashc/my-trankcopy/ml model/college_scripts/ingest_college_physicals.py`
- New raw/canonical/trajectory data contracts:
  - `raw_team_roster_physical` (duckdb, append-only)
  - `fact_college_player_physicals_by_season` (duckdb + parquet mirror)
  - `fact_college_player_physical_trajectory` (duckdb + parquet mirror)
- Provider adapter stack (priority order):
  - `cbd`, `cbbpy`, `sportsipy`, `manual`, `recruiting_fallback`
- Unified + inference parity wiring:
  - joins canonical physicals by final season
  - adds trajectory fields:
    - `college_height_delta_yoy`
    - `college_weight_delta_yoy`
    - `college_height_slope_3yr`
    - `college_weight_slope_3yr`
- Encoder contract update:
  - physical trajectory fields added to `CAREER_BASE_COLUMNS`
- Added hard physical gate:
  - `/Users/akashc/my-trankcopy/ml model/nba_scripts/run_physical_feature_gate.py`
- Added canonical HTML dashboard:
  - `/Users/akashc/my-trankcopy/ml model/docs/diagrams/physical_feature_quality_dashboard.html`

### Gate/Audit artifacts
- `/Users/akashc/my-trankcopy/ml model/data/audit/physicals_unresolved_identity.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/physicals_ambiguous_identity.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/physicals_coverage_by_season.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/physicals_provider_mix.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/physicals_linkage_quality.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/physical_feature_gate_report.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/physical_feature_gate_report.json`

### Current limitation (expected with current provider mix)
- This run used `manual,recruiting_fallback` providers only (no CBD/cbbpy/sportsipy roster source records in environment).
- Result:
  - static physicals (`college_height_in`, `college_weight_lbs`) are high coverage in unified.
  - trajectory fields (`college_height_delta_yoy`, `college_weight_delta_yoy`, `college_height_slope_3yr`, `college_weight_slope_3yr`) are schema-complete but sparse/null in NBA-crosswalk cohort.
- Action to increase trajectory coverage:
  - run ingest with season roster providers (`cbd` and/or `cbbpy`/`sportsipy`) and/or populate `/Users/akashc/my-trankcopy/ml model/data/manual_physicals`.

## 2026-02-21 — Patch Set A Hardening (Readiness Pass)
- Fixed baseline source to supervised surface (`unified_training_table_supervised.parquet`) in `/Users/akashc/my-trankcopy/ml model/scripts/run_stage0_baseline.py`.
- Moved and corrected maturity/target masks in `/Users/akashc/my-trankcopy/ml model/nba_scripts/build_unified_training_table.py`:
  - `has_peak_rapm_target` -> `y_peak_ovr`
  - `has_peak_epm_target` -> `y_peak_epm_window|3y|2y|1y`
  - `has_year1_epm_target` -> `year1_epm_tot`
- Enforced foundation grain (`athlete_id`,`season`) with zero duplicates in foundation contract report.
- Aligned Stage 4 objective to `TARGET_COL = 'y_peak_epm_window'` in `/Users/akashc/my-trankcopy/ml model/scripts/train_2026_model.py`.
- Hardened publish gate in `/Users/akashc/my-trankcopy/ml model/scripts/run_stage5_evaluation.py` to require CV-majority criterion.
- Repaired mixed-type feature coercion in Stage 4/5 training+evaluation matrix assembly.
- Updated combine contract outputs to `warehouse_v2`:
  - `/Users/akashc/my-trankcopy/ml model/data/warehouse_v2/raw_nba_draft_combine.parquet`
  - `/Users/akashc/my-trankcopy/ml model/data/warehouse_v2/fact_player_combine_measurements.parquet`
  - `/Users/akashc/my-trankcopy/ml model/data/warehouse_v2/fact_player_combine_imputed.parquet`
- Added canonical 2026 HTML dashboards under `/Users/akashc/my-trankcopy/ml model/docs/diagrams/`:
  - `advanced_ml_pipeline_dashboard.html`
  - `foundation_data_coverage_dashboard.html`
  - `combine_linkage_quality_dashboard.html`
  - `model_signal_separation_dashboard.html`
