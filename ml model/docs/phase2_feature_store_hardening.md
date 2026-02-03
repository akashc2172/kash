# Phase 2: Feature Store Hardening - Detailed Implementation Plan

**Date**: 2026-01-29  
**Status**: Planning  
**Owner**: TBD

---

## Overview

Phase 2 focuses on **hardening the feature store** to ensure robust, normalized features that work across the 15-year span (2010-2025). This phase addresses data quality, normalization, and the creation of a unified training table.

---

## 2.1 Finish Historical Backfill

### Objective
Complete the historical box score backfill for all available scraped seasons (2010-2018).

### Current Status
- ✅ **2015**: Completed (1.16M minutes derived)
- ✅ **2017**: Completed (48k minutes derived)
- ⏳ **2010-2014, 2016, 2018**: Pending

### Implementation Steps

**Step 2.1.1: Run Backfill for All Available Seasons**
```bash
# Process all seasons that have cleaned PBP files
python college_scripts/derive_minutes_from_historical_pbp.py --all
```

**Step 2.1.2: Validate Output**
- Check minutes totals per season (should be ~200 min/game × games)
- Check turnover rates (should be ~10-15 per game, ~40-60% capture)
- Spot-check 10-20 high-profile players per season

**Step 2.1.3: Merge into DuckDB**
Create integration script: `college_scripts/merge_backfill_into_warehouse.py`

**Logic**:
1. Load backfill parquet: `fact_player_season_stats_backfill.parquet`
2. Crosswalk `player_name` + `team_name` → `athlete_id` (fuzzy matching)
3. Merge with existing `fact_player_season_stats` in DuckDB
4. Fill missing values for 2006-2024

**Deliverables**:
- `fact_player_season_stats` updated with minutes/turnovers for 2010-2018
- Coverage report: `data/validation_reports/backfill_coverage_{DATE}.csv`

**Success Criteria**:
- Minutes coverage: ≥70% of player-seasons for 2010-2018
- Turnover coverage: ≥40% (acceptable given text extraction limitations)

---

## 2.2 Implement Windowed Activity Ghost Fill

### Objective
Improve lineup reconstruction accuracy by using time-aware activity windows instead of global game-total activity.

### Problem Statement

**Current Approach** (`clean_historical_pbp_v2.py`):
- Uses `roster_counter` (global game-total activity) to fill ghosts
- Problem: In blowouts, starters sit in 2nd half, bench plays more
- At minute 35, we might incorrectly fill with a starter who hasn't played since minute 20

**Example Failure Case**:
- Game: Blowout (team up 30 at halftime)
- Minute 5: Starters on floor (correct)
- Minute 35: Bench players on floor (starters benched)
- Current algorithm: Sees starter has 15 total activities, fills with starter
- **Error**: Starter hasn't played since minute 20, shouldn't be on floor at minute 35

### Solution: Windowed Activity

**Concept**: Track activity in rolling time windows (e.g., last 10 minutes).

**Algorithm**:
1. For each player, track timestamps of all activity events
2. When filling ghost at time T, look at activity in window [T-10min, T]
3. Prioritize players active in that window

**Implementation**:

I've created `college_scripts/utils/clean_historical_pbp_v2_windowed.py` with:
- `WindowedActivityTracker`: Tracks activity in rolling windows
- `WindowedGameSolver`: Improved solver using windowed activity
- `ensure_five_windowed()`: Ghost fill using windowed activity instead of global

### Integration Steps

**Step 2.2.1: Test Windowed Approach on 2015 Validated Data**
```python
# Compare outputs:
# - Original: clean_historical_pbp_v2.py
# - Improved: clean_historical_pbp_v2_windowed.py
# 
# Validation: Check if windowed approach reduces lineup errors
# (e.g., starters appearing in 2nd half of blowouts)
```

**Step 2.2.2: A/B Test**
- Run both approaches on 2015 data
- Compare lineup accuracy (manual spot-checks on known games)
- Measure improvement (e.g., % of games with correct lineups)

**Step 2.2.3: Integrate into Main Pipeline**
- If windowed approach is better, replace `ensure_five()` in `clean_historical_pbp_v2.py`
- Or create hybrid: use windowed for games with >20 point margin, global otherwise

**Deliverables**:
- Windowed ghost fill implementation
- Validation report comparing windowed vs global
- Updated `clean_historical_pbp_v2.py` (if windowed is better)

**Success Criteria**:
- Windowed approach reduces lineup errors by ≥10% (measured on validated games)
- No performance regression (processing time < 2x original)

---

## 2.3 Build Unified Training Table

### Objective
Create a single wide matrix that combines all feature sources for model training.

### Current State
Features are scattered across multiple parquet files:
- `college_features_v1.parquet` (athlete-season-split)
- `prospect_career_v1.parquet` (athlete-level)
- `fact_player_year1_epm.parquet` (NBA Year-1)
- `fact_player_peak_rapm.parquet` (NBA Peak)
- `fact_player_nba_college_gaps.parquet` (Gap features)

### Target Schema

**Grain**: `(athlete_id, season)` (one row per player-season)

**Columns** (grouped by source):

**College Features (Tier 1 - Universal)**:
- `rim_att`, `rim_made`, `rim_fg_pct`, `rim_share`
- `mid_att`, `mid_made`, `mid_fg_pct`, `mid_share`
- `three_att`, `three_made`, `three_fg_pct`, `three_share`
- `ft_att`, `ft_made`, `ft_pct`, `ft_rate`
- `assisted_share_rim`, `assisted_share_three`, `assisted_share_mid`
- `high_lev_att_rate`, `high_lev_fg_pct`
- `garbage_att_rate`
- `on_net_rating`, `on_ortg`, `on_drtg`
- `seconds_on`, `games_played`
- `team_pace`, `conference`, `is_power_conf`
- `opp_rank` (opponent strength)

**College Features (Tier 2 - Spatial, 2019+)**:
- `avg_shot_dist`, `shot_dist_var`
- `corner_3_rate`, `corner_3_pct`
- `deep_3_rate`
- `rim_purity`
- `xy_shots`, `xy_coverage` (coverage flags)

**College Features (Advanced Rates)**:
- `minutes_total` (from backfill or box scores)
- `tov_total` (from backfill or box scores)
- `usage_proxy` (derived: `(FGA + 0.44*FTA + TOV) / poss`)
- `ast_proxy_raw`, `ast_proxy_z` (if ASTz implemented)

**College Career Summary** (from `prospect_career_v1`):
- `final_trueShootingPct`, `final_usage`
- `career_years`
- `slope_*` (trajectory features)
- `career_wt_*` (recency-weighted)

**NBA Targets** (for training only, NOT for inference):
- `year1_epm_tot`, `year1_epm_off`, `year1_epm_def`
- `peak_rapm_ovr`, `peak_rapm_off`, `peak_rapm_def`
- `gap_ts_legacy`, `gap_usg_legacy`

**Metadata**:
- `season`, `athlete_id`, `teamId`
- `draft_year` (if known)
- `has_spatial_data`, `has_athletic_testing` (coverage flags)

### Implementation

**Script**: `nba_scripts/build_unified_training_table.py` (new)

**Logic**:
1. Load college features (default: `split_id == 'ALL__ALL'`)
2. Join career summary (final season features)
3. Join NBA targets via crosswalk (only for historical NBA players)
4. Join gap features (if available)
5. Apply feature transforms (era normalization, stabilization)
6. Add coverage flags
7. Output: `data/training/unified_training_table_{DATE}.parquet`

**Key Design Decisions**:
- **One row per player-season**: Model sees each season as a separate observation
- **Split handling**: Default to `ALL__ALL`, but can add split-specific rows later
- **Missingness**: Explicit `NaN` for missing features (no silent zeros)
- **Leakage prevention**: Only join NBA targets for players with `draft_year < current_year`

**Deliverables**:
- Unified training table builder script
- Sample table: `data/training/unified_training_table_sample.parquet` (first 1000 players)
- Schema documentation: Column descriptions, missingness patterns, coverage stats

**Success Criteria**:
- Table has ≥2000 player-seasons with both college features and NBA targets
- No leakage detected (manual spot-check of high-profile players)
- Missingness is explicit (no silent zeros)

---

## 2.4 Feature Normalization & Era Adjustment

### Objective
Ensure features are normalized so a 2012 player looks like a 2024 player to the model.

### Normalization Strategies

**1. Z-Score by Era (Season-Level)**
- For rates with season drift (AST%, pace, 3P rates)
- Formula: `z = (value - season_mean) / season_std`
- Store: raw value, season baseline, z-score
- **Robust variant**: Use median/MAD instead of mean/std for skewed distributions

**2. Logit Transform (Percentages)**
- For percentage features (FG%, assisted share, etc.)
- Formula: `logit(p) = log(p / (1-p))`
- Handles 0% and 100% gracefully with small epsilon

**3. Beta Prior Stabilization (Rates)**
- For noisy rates (FG% on small samples)
- Formula: `stabilized = (made + α) / (att + α + β)`
- Use league-wide priors per season

**4. Coverage Masks**
- Boolean flags: `has_spatial_data`, `has_athletic_testing`
- Model can learn to handle missingness explicitly

### Implementation

**Script**: `nba_scripts/nba_feature_transforms.py` (already exists, needs updates)

**Updates Needed**:
1. Add college feature transforms (currently only has NBA transforms)
2. Add era normalization for college features
3. Add coverage mask generation

**Deliverables**:
- Updated transform functions
- Validation: transformed features have reasonable distributions

**Success Criteria**:
- Transformed features are approximately normal (or appropriate distribution)
- Era normalization removes season-level drift (correlation with season < 0.1)

---

## 2.5 Enhanced RAPM & Leverage Features (COMPLETED - Jan 2025)

### Objective
Extend RAPM computation with leverage-aware variants and add athleticism/pressure features.

### Implemented Enhancements

**1. Win Probability & Leverage Model** (`calculate_historical_rapm.py`)
- `compute_win_probability()`: Time-weighted logistic model for WP estimation
- `compute_leverage_index()`: Expected WP swing from possession outcomes (pbpstats methodology)
- Leverage buckets: `garbage`, `low`, `medium`, `high`, `very_high`

**2. RAPM Variants** (all per-season):
| Variant | Description | Use Case |
|---------|-------------|----------|
| `rapm_standard` | Possession-weighted (original) | Baseline impact |
| `rapm_leverage_weighted` | Weights by leverage index | Clutch performance |
| `rapm_high_leverage` | High/very_high stints only | Crunch time specialists |
| `rapm_non_garbage` | Excludes garbage time | Cleaner signal |
| `o_rapm` / `d_rapm` | Offensive/Defensive split | Two-way analysis |
| `rapm_rubber_adj` | Rubber-band effect correction | Removes coasting bias |

**3. Rubber Band Adjustment**
- Teams ahead tend to coast; teams behind try harder
- `compute_rubber_band_adjustment()` corrects for this systematic bias
- Adjusts expected margin change based on lead size

**4. New Feature Blocks** (added to `college_feature_registry_v1.csv`):

**Athleticism Block**:
- `dunk_rate`: Dunks / rim attempts (explosiveness)
- `dunk_freq`: Dunks / total FGA
- `putback_rate`: Putback attempts / OREBs (motor + timing)
- `transition_freq`: Transition attempts / total FGA
- `transition_eff`: Transition points / attempts
- `rim_pressure_index`: (rim_fga + 0.44*FTA) / poss (rim gravity)

**Defense Activity Block**:
- `deflection_proxy`: 1.5*stl_rate + 0.5*blk_rate (disruption)
- `contest_proxy`: blk_rate / (blk_rate + foul_rate) (clean contests)

**Decision Discipline Block**:
- `pressure_handle_proxy`: TO_rate(high_lev) - TO_rate(low_lev) (negative = handles pressure)
- `clutch_shooting_delta`: TS%(high_lev) - TS%(low_lev) (positive = rises to occasion)

**Shot Creation Block**:
- `self_creation_rate`: Unassisted FGA / total FGA
- `self_creation_eff`: Unassisted points / (2 * unassisted FGA)

**Context Block**:
- `leverage_poss_share`: High-leverage poss / total poss (clutch usage)

### Success Criteria
- All RAPM variants computed for 2010-2025 seasons
- Correlation analysis shows variants capture different signals
- Feature registry updated with all new features

---

## Dependencies & Blockers

### Critical Blockers
1. **Historical Backfill Completion** (2.1): Blocks usage features for 2010-2018
2. **Windowed Ghost Fill Validation** (2.2): Need to verify improvement before full rollout

### Nice-to-Have (Not Blocking)
1. ASTz implementation (can use raw AST% initially)
2. Recency weighting (can use full-season aggregates initially)

---

## Success Criteria (Overall Phase 2)

**Data Quality**:
- Historical backfill: ≥70% minutes coverage for 2010-2018
- Windowed ghost fill: ≥10% reduction in lineup errors (if implemented)
- Unified table: ≥2000 player-seasons with complete features

**Feature Quality**:
- Era normalization: Season-level drift removed (correlation < 0.1)
- Missingness: Explicit `NaN` (no silent zeros)
- Coverage flags: All Tier 2 features have explicit masks

---

**Plan Author**: cursor  
**Last Updated**: 2026-01-29
