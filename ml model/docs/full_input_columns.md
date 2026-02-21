# Full Input Columns (Verified)

**Generated**: 2026-02-20 19:19:28

**Source Table**: `data/training/unified_training_table.parquet`

**Total Columns**: 458

## Model Inputs

### Tier 1 (Universal)

- `college_rim_fg_pct`
- `college_mid_fg_pct`
- `college_three_fg_pct`
- `college_ft_pct`
- `college_rim_share`
- `college_mid_share`
- `college_three_share`
- `college_shots_total`
- `college_fga_total`
- `college_ft_att`
- `college_games_played`
- `college_poss_proxy`
- `college_minutes_total`
- `college_team_pace`
- `college_is_power_conf`
- `college_team_srs`
- `team_strength_srs`
- `college_team_rank`
- `college_ast_total_per100poss`
- `college_tov_total_per100poss`
- `college_stl_total_per100poss`
- `college_blk_total_per100poss`
- `college_orb_total_per100poss`
- `college_drb_total_per100poss`
- `college_trb_total_per100poss`
- `college_dunk_rate`
- `college_dunk_freq`
- `college_putback_rate`
- `college_rim_pressure_index`
- `college_contest_proxy`
- `college_transition_freq`
- `college_deflection_proxy`
- `college_pressure_handle_proxy`
- `college_assisted_share_rim`
- `college_assisted_share_mid`
- `college_assisted_share_three`
- `college_rapm_standard`
- `college_o_rapm`
- `college_d_rapm`
- `college_on_net_rating`
- `college_on_ortg`
- `college_on_drtg`
- `high_lev_att_rate`
- `garbage_att_rate`
- `leverage_poss_share`
- `college_three_fg_pct_z`
- `final_trueShootingPct_z`
- `final_usage_z`
- `college_rim_fg_pct_z`
- `college_mid_fg_pct_z`
- `college_ft_pct_z`
- `final_trueShootingPct_team_resid`
- `final_usage_team_resid`
- `college_three_fg_pct_team_resid`
- `college_recruiting_rank`
- `college_recruiting_stars`
- `college_recruiting_rating`

### Tier 2 (Spatial)

- `college_avg_shot_dist`
- `college_shot_dist_var`
- `college_corner_3_rate`
- `college_corner_3_pct`
- `college_deep_3_rate`
- `college_rim_purity`
- `college_xy_shots`
- `college_xy_3_shots`
- `college_xy_rim_shots`

### Career (Progression)

- `career_years`
- `college_final_season`
- `draft_year_proxy`
- `season_index`
- `class_year`
- `age_at_season`
- `has_age_at_season`
- `college_height_in`
- `college_weight_lbs`
- `has_college_height`
- `has_college_weight`
- `nba_height_change_cm` (observed trajectory field)
- `nba_weight_change_lbs` (observed trajectory field)
- `final_trueShootingPct`
- `final_usage`
- `final_poss_total`
- `final_rim_fg_pct`
- `final_three_fg_pct`
- `final_ft_pct`
- `slope_trueShootingPct`
- `slope_usage`
- `career_wt_trueShootingPct`
- `career_wt_usage`
- `delta_trueShootingPct`
- `delta_usage`
- `slope_rim_fg_pct`
- `slope_three_fg_pct`
- `slope_ft_pct`
- `career_wt_rim_fg_pct`
- `career_wt_three_fg_pct`
- `career_wt_ft_pct`
- `delta_rim_fg_pct`
- `delta_three_fg_pct`
- `delta_ft_pct`
- `breakout_timing_avg`
- `breakout_timing_volume`
- `breakout_timing_usage`
- `breakout_timing_eff`
- `breakout_rank_eff`
- `breakout_rank_volume`
- `breakout_rank_usage`
- `college_dev_p10`
- `college_dev_p50`
- `college_dev_p90`
- `college_dev_quality_weight`
- `transfer_mean_shock`
- `has_transfer_context`
- `transfer_event_count`
- `transfer_max_shock`
- `transfer_conf_delta_mean`
- `transfer_pace_delta_mean`
- `transfer_role_delta_mean`
- `has_within_window_data`

### Within-Season Windows (Star Run)

- `final_has_ws_last10`
- `final_ws_minutes_last10`
- `final_ws_pps_last10`
- `final_ws_delta_pps_last5_minus_prev5`
- `final_has_ws_breakout_timing_eff`
- `final_ws_breakout_timing_eff`

## Targets (Labels Only)

- `y_peak_ovr`
- `y_peak_off`
- `y_peak_def`
- `year1_epm_tot`
- `year1_epm_off`
- `year1_epm_def`
- `gap_ts_legacy`
- `gap_usg_legacy`
- `made_nba`

## Masks / Coverage Flags

- `has_spatial_data`
- `final_has_ws_last5`
- `final_has_ws_last10`
- `final_has_ws_breakout_timing_eff`

## Notes

- Columns marked `(MISSING)` are not present in the current training table and would be silently imputed by naive loaders; fix by adding them upstream or removing them from the model column lists.

- Columns listed in `Dead Inputs` are present but effectively constant/zero and should be treated as unwired until source population is fixed.

- NBA data is used as targets only; see `docs/end_to_end_wiring.md` and `nba_scripts/nba_data_loader.py` for leakage rules.


## Dead Inputs (non-zero < 0.1%)

- `college_transition_freq` (non-zero rate=0.0000)
- `final_has_ws_breakout_timing_eff` (non-zero rate=0.0000)
- `final_has_ws_last10` (non-zero rate=0.0000)
- `final_ws_breakout_timing_eff` (non-zero rate=0.0000)
- `final_ws_delta_pps_last5_minus_prev5` (non-zero rate=0.0000)
- `final_ws_minutes_last10` (non-zero rate=0.0000)
- `final_ws_pps_last10` (non-zero rate=0.0000)
- `has_within_window_data` (non-zero rate=0.0000)
## 2026-02-20 Activity Contract Update

Restored as live contract inputs (unified table):
- `college_dunk_rate`
- `college_dunk_freq`
- `college_putback_rate`
- `college_rim_pressure_index`
- `college_contest_proxy`

Contract masks/provenance confirmed:
- `college_activity_source`
- `has_college_activity_features`
- `college_dunk_rate_missing`
- `college_dunk_freq_missing`
- `college_putback_rate_missing`
- `college_rim_pressure_index_missing`
- `college_contest_proxy_missing`

Note:
- `college_transition_freq` remains a dead near-zero column in current data and is tracked as non-critical until source population improves.
