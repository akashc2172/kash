# Review Summary: Antigravity Changes & Improvements

**Date**: 2026-01-29  
**Reviewer**: cursor

---

## Executive Summary

I've reviewed antigravity's work and created detailed implementation plans for all phases. Key improvements:

1. ✅ **Zone B (Ghost Fill)**: Created windowed activity implementation
2. ✅ **Zone C (Loss Function)**: Clarified gap concept (translation, not absolute)
3. ✅ **Zone D (Input Pipeline)**: Created detailed feature lists and phase plans

---

## Zone A: Data Ingestion (1.1 & 1.2)

### Status: ✅ **GOOD**

**1.1 Source Hierarchy**: Fine as-is. No changes needed.

**1.2 Reconstruction Pipeline**: 
- ✅ Period boundary fix implemented (recovers 15-30s per game)
- ✅ Windowed activity improvement created (see Zone B)

**Recommendation**: Proceed with current implementation.

---

## Zone B: Feature Store (Ghost Fill Improvement)

### Current Implementation Review

**Antigravity's Proposal**: Move from "Global Most Active" to "Windowed Activity" filler.

**Problem Identified**: ✅ **CORRECT**
- Current approach uses global game-total activity
- In blowouts, starters sit in 2nd half, bench plays more
- At minute 35, we might incorrectly fill with a starter who hasn't played since minute 20

**My Implementation**: ✅ **CREATED**

I've created `college_scripts/utils/clean_historical_pbp_v2_windowed.py` with:

1. **WindowedActivityTracker**: Tracks activity in rolling time windows (default 10 minutes)
2. **WindowedGameSolver**: Improved solver using windowed activity
3. **ensure_five_windowed()**: Ghost fill using windowed activity instead of global

**Key Features**:
- Tracks activity timestamps for each player
- When filling ghost at time T, looks at activity in window [T-10min, T]
- Prioritizes players active in that window
- Falls back to global activity for early game (window too small)

**Next Steps**:
1. Test windowed approach on 2015 validated data
2. Compare outputs (windowed vs global)
3. If better, integrate into main pipeline

**See**: `docs/phase2_feature_store_hardening.md` Section 2.2 for detailed plan.

---

## Zone C: Model Architecture (Loss Function)

### Clarification: What is "Gap = NBA_Metric - College_Metric"?

**Answer**: We predict **TRANSLATION** (change), not absolute NBA performance.

**Why?**
- College TS% = 0.60, NBA TS% = 0.55
- Gap = 0.55 - 0.60 = -0.05 (player's TS% drops by 5 percentage points)
- Model learns: "NBA is harder, everyone's TS% drops. This player will drop by 0.05."

**Targets**:
- **Primary**: `gap_rapm = NBA_3yr_Peak_RAPM - College_3yr_RAPM`
- **Auxiliary**: `gap_ts_legacy`, `gap_usg_legacy`, `made_nba` (binary)

**Loss Function**:
```
L_total = w1 * MSE(gap_rapm) + 
          w2 * MSE(gap_ts) + 
          w3 * MSE(gap_usg) + 
          w4 * BCE(made_nba)
```

**Why Multi-Task?**
- Primary target (`gap_rapm`) is noisy (small sample for 3yr peak)
- Auxiliary targets provide additional signal
- Binary target (`made_nba`) is easier to predict (more data)

**See**: `docs/zone_c_gap_concept_clarification.md` for detailed explanation.

---

## Zone D: Input Pipeline (Feature Lists & Implementation)

### Problem: Too Vague

**User Request**: "I want more detail about what exactly we're doing. What exactly are we thinking of the plan for building the feature list?"

### Solution: Detailed Phase Documents

I've created **3 detailed phase documents**:

1. **`docs/phase2_feature_store_hardening.md`**: Feature store hardening
2. **`docs/phase3_model_training.md`**: Model training with complete feature lists
3. **`docs/phase4_validation.md`**: Validation & analysis

### Complete Feature List (From Phase 3)

**Tier 1 (Universal - Always Available)**:
- Shot Profile: `rim_att`, `rim_made`, `rim_fg_pct`, `rim_share`, `mid_*`, `three_*`, `ft_*`
- Creation Context: `assisted_share_rim`, `assisted_share_three`, `high_lev_att_rate`, `garbage_att_rate`
- Impact Metrics: `on_net_rating`, `on_ortg`, `on_drtg`, `seconds_on`, `games_played`
- Team Context: `team_pace`, `conference`, `is_power_conf`, `opp_rank`
- Volume & Usage: `minutes_total`, `tov_total`, `usage_proxy`
- Career Summary: `final_trueShootingPct`, `final_usage`, `career_years`, `slope_*`, `career_wt_*`

**Tier 2 (Spatial - 2019+ Only, with Masking)**:
- `avg_shot_dist`, `shot_dist_var`
- `corner_3_rate`, `corner_3_pct`
- `deep_3_rate`
- `rim_purity`
- Coverage flags: `xy_shots`, `xy_3_shots`, `xy_rim_shots`

**Coverage Masks**:
- `has_spatial_data` (0/1)
- `has_athletic_testing` (0/1)

**See**: `docs/phase3_model_training.md` Section 3.1 for complete list with descriptions.

### Feature Selection Strategy

**Phase 3.5: Feature Selection & Ablation Studies**

**Experiments**:
1. **Tier 1 Only vs Tier 1 + Tier 2**: Compare performance
2. **Feature Groups**: Test shot profile only, impact metrics only, etc.
3. **Ablation**: Remove top 10 features one-by-one, measure degradation

**Goal**: Identify top 20 most important features, remove redundant ones.

**See**: `docs/phase3_model_training.md` Section 3.5 for detailed plan.

### Baseline vs Advanced Model

**Baseline (XGBoost)**:
- Fast iteration, easy to debug
- Establishes performance benchmarks
- Feature importance analysis
- **Goal**: RMSE < 2.0, Correlation > 0.4

**Advanced (MLP/Transformer)**:
- MLP first (simple, fast)
- Transformer if MLP underperforms
- **Goal**: Outperform XGBoost (RMSE improvement ≥10%)

**See**: `docs/phase3_model_training.md` Sections 3.3 & 3.4 for detailed plans.

---

## Improvements Made

### 1. Windowed Activity Ghost Fill ✅
- Created implementation: `clean_historical_pbp_v2_windowed.py`
- Detailed plan: `phase2_feature_store_hardening.md` Section 2.2

### 2. Gap Concept Clarification ✅
- Created explanation: `zone_c_gap_concept_clarification.md`
- Updated master plan with clarifications

### 3. Detailed Phase Documents ✅
- Phase 2: Feature Store Hardening
- Phase 3: Model Training (with complete feature lists)
- Phase 4: Validation & Analysis

### 4. Master Plan Updates ✅
- Added links to detailed phase documents
- Clarified Zone C (gap concept)
- Expanded Zone D (input pipeline details)

---

## Recommendations

### Immediate Next Steps

1. **Test Windowed Ghost Fill** (Zone B):
   - Run on 2015 validated data
   - Compare to global approach
   - If better, integrate

2. **Complete Historical Backfill** (Phase 2.1):
   - Run `derive_minutes_from_historical_pbp.py --all`
   - Merge into DuckDB

3. **Build Unified Training Table** (Phase 2.3):
   - Create `build_unified_training_table.py`
   - Merge all feature sources

4. **Start Baseline Training** (Phase 3.3):
   - Train XGBoost model
   - Establish performance benchmarks

### Long-Term

1. **Feature Selection** (Phase 3.5):
   - Run ablation studies
   - Optimize feature set

2. **Advanced Model** (Phase 3.4):
   - Train MLP
   - If needed, train Transformer

3. **Validation** (Phase 4):
   - Walk-forward validation
   - Error analysis
   - Feature importance

---

## Files Created/Updated

### New Files
- `college_scripts/utils/clean_historical_pbp_v2_windowed.py` (Windowed ghost fill)
- `docs/phase2_feature_store_hardening.md` (Phase 2 plan)
- `docs/phase3_model_training.md` (Phase 3 plan with feature lists)
- `docs/phase4_validation.md` (Phase 4 plan)
- `docs/zone_c_gap_concept_clarification.md` (Gap concept explanation)
- `docs/review_summary_2026_01_29.md` (This document)

### Updated Files
- `docs/ml_model_master_plan.md` (Added clarifications, links to phase docs)

---

## Questions Answered

### Q1: "Check if the current implementation is fine" (Zone B)
**A**: ✅ Current implementation is fine, but windowed approach should be better. Created implementation for testing.

### Q2: "Create a better implementation" (Zone B)
**A**: ✅ Created windowed activity implementation. See `clean_historical_pbp_v2_windowed.py`.

### Q3: "What is NBA metric minus College metric?" (Zone C)
**A**: ✅ It's **translation** (change), not absolute performance. See `zone_c_gap_concept_clarification.md`.

### Q4: "I want more detail about the input pipeline" (Zone D)
**A**: ✅ Created detailed phase documents with complete feature lists. See `phase3_model_training.md` Section 3.1.

### Q5: "What exactly are we thinking for building the feature list?"
**A**: ✅ See `phase3_model_training.md` Section 3.5 (Feature Selection & Ablation Studies).

### Q6: "What does baseline vs advanced model mean?"
**A**: ✅ Baseline = XGBoost (fast, interpretable). Advanced = MLP/Transformer (complex, potentially better). See `phase3_model_training.md` Sections 3.3 & 3.4.

---

## Final Verdict

**Status**: ✅ **ALL ZONES REVIEWED & IMPROVED**

- **Zone A**: ✅ Good, no changes needed
- **Zone B**: ✅ Windowed implementation created, ready for testing
- **Zone C**: ✅ Clarified gap concept, loss function makes sense
- **Zone D**: ✅ Detailed phase documents created with complete feature lists

**Ready for**: Testing windowed ghost fill, building unified training table, starting baseline training.

---

**Reviewer**: cursor  
**Date**: 2026-01-29
