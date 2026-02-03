# Full Input Columns (Verified)

**Generated**: 2026-02-03 09:28:57

**Source Table**: `data/training/unified_training_table.parquet`

**Total Columns**: 325

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
- `college_minutes_total`
- `college_team_pace`
- `college_is_power_conf`
- `college_ast_total_per40`
- `college_tov_total_per40`
- `college_stl_total_per40`
- `college_blk_total_per40`
- `college_three_fg_pct_z`
- `college_three_share_z`
- `final_trueShootingPct_z`
- `final_usage_z`
- `final_trueShootingPct_team_resid`
- `final_usage_team_resid`

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
- `breakout_timing_avg`
- `breakout_timing_volume`
- `breakout_timing_eff`

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

- NBA data is used as targets only; see `docs/end_to_end_wiring.md` and `nba_scripts/nba_data_loader.py` for leakage rules.
