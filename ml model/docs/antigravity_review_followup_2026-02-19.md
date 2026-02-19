# Antigravity Review Follow-up (2026-02-19)

## Findings validated
- `usage`, `games_played`, and `poss_per_game` were sparse/all-null for most pre-2025 rows in `prospect_career_long_v1.parquet`.
- Builder logic using `df.get(..., np.nan)` could return scalars, causing downstream `.notna()`/`.values` crash risks when columns are absent.
- Transfer deltas could collapse to null if context columns were missing.

## Fixes implemented
- `build_transfer_context_v1.py`
  - Added index-aligned `_series()` helper for safe missing-column handling.
  - Replaced scalar `get()` fallbacks with series-safe paths.
  - Kept `usage` derivation fallback from `fga_total`, `ft_att`, `tov_total`, `poss_total`.
- `build_fact_player_college_development_rate.py`
  - Added index-aligned `_series()` helper.
  - Replaced scalar `get()` fallbacks that could crash in grouped operations.
  - Retained `usage` and creation-rate derivations from populated columns.
- `build_unified_training_table.py`
  - Added wingspan schema scaffolding:
    - `wingspan_in`, `standing_reach_in`, `wingspan_minus_height_in`, `has_wingspan`.
  - If present in crosswalk, wingspan columns are carried through; otherwise nullable defaults are added.
- Model consumption wiring
  - `models/player_encoder.py`: added impact/dev/transfer/wingspan fields to active latent feature lists.
  - `nba_scripts/train_pathway_model.py`: extended `FEATURE_COLS` with impact/dev/transfer/wingspan fields.
  - `nba_scripts/train_generative_model.py`: broadened feature selection to include `impact_*`, `rIPM_*`, `transfer_*`, and wingspan fields.

## Remaining constraints
- Historical impact coverage is still source-limited by available lineup/impact records.
- Physical dev-rate remains null pending actual biometric source ingestion.
