# Docs-to-Code Realignment (2026-02-18)

## Why this pass
Recent iterations drifted from documented intent in `docs/`:
- too much post-model ranking logic in inference,
- missing career timing signals in active encoder inputs,
- within-season inputs disabled despite populated fields.

This pass restores alignment **without discarding existing pipeline work**.

## What was checked
- `docs/full_input_columns.md`
- `docs/latent_input_plan.md`
- `docs/data_assembly_and_model_connectivity_plan.md`
- `docs/end_to_end_wiring.md`
- `docs/nba_feeding_plan.md`
- active code:
  - `models/player_encoder.py`
  - `models/archetype_analyzer.py`
  - `nba_scripts/nba_prospect_inference.py`
  - current training table and inference table surfaces

## Mismatches found and fixed

### 1) Within-season branch was disabled
- **Issue**: `WITHIN_COLUMNS` was empty in `models/player_encoder.py`, while `final_ws_*` and `final_has_ws_*` columns are populated and intended for in-season development signals.
- **Fix**: Re-enabled 6 within-season columns:
  - `final_has_ws_last10`
  - `final_ws_minutes_last10`
  - `final_ws_pps_last10`
  - `final_ws_delta_pps_last5_minus_prev5`
  - `final_has_ws_breakout_timing_eff`
  - `final_ws_breakout_timing_eff`

### 2) Career timing inputs were incomplete
- **Issue**: `breakout_timing_usage` and `breakout_timing_eff` were not active in the career branch.
- **Fix**: Added both to `CAREER_BASE_COLUMNS`.

### 3) Era/year context was under-explicit
- **Issue**: season-level normalization exists, but explicit year context was not fed through encoder.
- **Fix**: Added `college_final_season` to `CAREER_BASE_COLUMNS` as draft-time-safe era context.

### 4) Inference ranking drifted to hand-tuned blend
- **Issue**: ranking used hardcoded post-model weighted z-score blend, which drifted from model-based contract.
- **Fix**: kept diagnostic z-score columns but restored ranking to model output:
  - `pred_peak_rapm_rank_score = pred_peak_rapm`

### 5) Archetype narratives referenced non-existent features
- **Issue**: `models/archetype_analyzer.py` templates used `college_on_net_rating` (not a stable active feature).
- **Fix**: replaced with valid high-coverage defensive proxy feature:
  - `college_stl_total_per100poss`

## Coverage verification (post-fix)
- Unified table:
  - Tier1 missing: 0
  - Tier2 missing: 0
  - Career missing: 0
  - Within missing: 0
- Within columns listed above: all present and non-null in current unified table.
- Inference table: all restored timing + within fields present and non-null.

## What is still genuinely missing (not patched here)

### Age/class features
- `age_at_season`, `class_year`, `season_index` are not currently present in the assembled feature store outputs.
- This is a **source-data / builder-surface gap**, not an encoder wiring gap.
- For now, we use:
  - `career_years`
  - `breakout_rank_*`, `breakout_timing_*`
  - `college_final_season` (era context)

### Note on leakage contract
- NBA Year-1 and post-draft signals remain targets/auxiliary only.
- No post-draft NBA performance columns were moved into college input X.

## Next actions (execution order)
1. Retrain latent model with restored within/timing/year context.
2. Run inference + season ranking export.
3. Run granular audit and publish updated coverage + sanity in `docs/data_quality_master_log.md`.
4. If ranking quality still weak, tune model/loss/regularization first (not post-hoc rank blending).
