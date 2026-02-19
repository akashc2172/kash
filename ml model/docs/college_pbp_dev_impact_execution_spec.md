# College PBP + Development + Impact Execution Spec

Date: 2026-02-19  
Owner: NBA/College modeling pipeline

## 1. Purpose
This is the canonical implementation spec for assembling college-side modeling inputs beyond raw PBP features:
- RAPM/RIPM-adjacent impact signals
- Year-over-year development signals
- Transfer-aware context shifts
- Unified integration into the NBA training table

This document consolidates the previously scattered planning docs into an execution contract.

## 2. Current State Snapshot (verified)
- `fact_play_raw`: 2010-2025 (all seasons), participants available broadly.
- `stg_shots`: 2010-2025, strong coverage for shot-profile features.
- `stg_lineups` / `bridge_lineup_athletes`: materially available in modern seasons only.
- `college_features_v1.parquet`: broad multi-season feature surface exists.
- `college_impact_v1.parquet`: impact present but sparse (modern-heavy).
- `prospect_career_v1.parquet` and `prospect_career_long_v1.parquet`: year-over-year structure and slope/breakout signals already present.
- `fact_player_development_rate.parquet` (NBA Y1->Y3): implemented and integrated.

## 3. Artifacts to Produce

### 3.1 Impact stack
Output: `data/college_feature_store/college_impact_stack_v1.parquet`  
Grain: athlete-season (`athlete_id`, `season`)

Required blocks:
1. Raw impact proxies:
- `impact_on_net_raw`, `impact_on_ortg_raw`, `impact_on_drtg_raw`

2. Stint PM100 proxies:
- `impact_pm100_stint_raw`
- `impact_pm100_stint_non_garbage`
- `impact_pm100_stint_lev_wt`

3. Adjusted RIPM variants:
- Standard: `rIPM_tot_std`, `rIPM_off_std`, `rIPM_def_std`
- Non-garbage: `rIPM_tot_non_garbage`, `rIPM_off_non_garbage`, `rIPM_def_non_garbage`
- Leverage-weighted: `rIPM_tot_lev_wt`, `rIPM_off_lev_wt`, `rIPM_def_lev_wt`
- Rubber-band adjusted: `rIPM_tot_rubber`, `rIPM_off_rubber`, `rIPM_def_rubber`
- Opponent-adjusted: `rIPM_tot_opp_adj`, `rIPM_off_opp_adj`, `rIPM_def_opp_adj`
- Recency-adjusted: `rIPM_tot_recency`, `rIPM_off_recency`, `rIPM_def_recency`

4. Reliability and masks:
- `impact_poss_total`, `impact_seconds_total`, `impact_stints_total`
- `impact_ripm_sd_tot`, `impact_ripm_sd_off`, `impact_ripm_sd_def`
- `impact_reliability_weight`
- `has_impact_raw`, `has_impact_stint`, `has_impact_ripm`
- `impact_source_mix`, `impact_version`

### 3.2 College development-rate labels
Output: `data/college_feature_store/fact_player_college_development_rate.parquet`  
Grain: one row per athlete final college season (`athlete_id`, `final_college_season`)

Required fields:
- `college_dev_rate_off_mean`, `college_dev_rate_off_sd`
- `college_dev_rate_eff_mean`, `college_dev_rate_eff_sd`
- `college_dev_rate_creation_mean`, `college_dev_rate_creation_sd`
- `college_dev_rate_impact_mean`, `college_dev_rate_impact_sd`
- `college_dev_rate_phys_mean`, `college_dev_rate_phys_sd` (nullable)
- `college_dev_p10`, `college_dev_p50`, `college_dev_p90`
- `college_dev_quality_weight`
- `college_dev_obs_years`, `college_dev_has_transfer`, `college_dev_model_version`

### 3.3 Transfer context
Output: `data/college_feature_store/fact_player_transfer_context.parquet`  
Grain: transfer event

Required fields:
- `athlete_id`, `season_from`, `season_to`, `team_from`, `team_to`
- `transfer_conf_delta`, `transfer_pace_delta`, `transfer_role_delta`
- `transfer_perf_delta_raw`, `transfer_perf_delta_context_adj`
- `transfer_shock_score`

## 4. Statistical Rules
1. Missingness policy: never coerce unknown data to zero for modeling signals. Keep NaN + explicit masks.
2. Reliability weighting: convert uncertainty to capped inverse-variance weights.
3. Transfer handling: detect team changes in long career panel; summarize pre/post deltas.
4. Development timing:
- career-stage breakout and within-season breakout remain separate axes.
- no hardcoded “late breakout bad” rule; model learns sign/magnitude.

## 5. Integration Points
Update unified table builder to join new artifacts:
- `college_impact_stack_v1.parquet`
- `fact_player_college_development_rate.parquet`
- transfer summary from `fact_player_transfer_context.parquet`

Add coverage diagnostics for each new block in builder logs and audit scripts.

## 6. Implementation Checklist
1. Build impact stack script.
2. Build transfer context script.
3. Build college development-rate script.
4. Wire all three into unified training table.
5. Add basic validation checks and run smoke build.

## 7. Validation Requirements
- Unique keys by artifact grain.
- No row explosion in unified join.
- Reasonable non-null rates by season for each block.
- Reliability monotonicity (higher exposure -> lower sd on average).

## 8. Non-goals in this phase
- Perfect historical lineup reconstruction for all early seasons.
- Full physical-dev signal ingestion if physical sources are still incomplete.
- Forcing sparse variants to appear where source coverage is absent.

