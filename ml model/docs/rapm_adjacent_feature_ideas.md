# RAPM-Adjacent Feature Ideas (Draft-Time Safe)

**Date**: 2026-02-03  
**Goal**: Add “impact-like” signals that are less brittle than full RAPM and more available than full spatial data.

## Why RAPM-Adjacent?

RAPM is powerful but expensive and sensitive to lineup completeness, minutes coverage, and era shifts. These features aim to:

- capture *impact* without requiring full multi-season NBA outcomes
- remain draft-time safe (college-only inputs)
- degrade gracefully when coverage is partial (use `NaN` + masks)

## Candidate Feature Blocks

### 1) On/Off Proxy Features (from player-game)

Source: DuckDB `fact_player_game` (modern years with player-game coverage).

- `on_net_rating` seconds-weighted season mean (already present in some pipelines)
- `on_ortg`, `on_drtg` seconds-weighted means (if available)
- leverage splits if we can compute (high-leverage stints only)

Why it helps:
- much cheaper than RAPM, still impact-flavored

Missingness:
- for years without player-game tables, keep `NaN` and `has_onoff = 0`

### 2) Leverage/Clutch Involvement

Sources:
- `fact_player_game` has `high_lev_att`, `garbage_att` (where available)
- within-season windows (last5/last10)

Features:
- `clutch_share = high_lev_att / fga` (NaN-safe)
- `non_garbage_share = 1 - garbage_att / fga`
- “late-season form”: deltas from `within_season_windows_v1.parquet`

Why it helps:
- distinguishes “big game” roles vs empty-calorie production

### 3) Stint-Based Plus/Minus (Non-Adjusted)

Source:
- historical stints (from reconstructed onFloor / stint builder)

Features:
- raw +/- per 100 possessions (with shrinkage)
- non-garbage raw +/- per 100
- leverage-weighted raw +/-

Why it helps:
- gives impact-ish signal even when full RAPM solve is unstable

### 4) RAPM Variant Decomposition (from historical solver)

Source:
- `calculate_historical_rapm.py` outputs multiple variants (standard, O/D split, leverage-weighted, non-garbage, rubber-band adjusted)

Features:
- include the *variants* (where available) rather than one scalar
- include reliability/exposure (`poss`, minutes, stints)

Why it helps:
- improves robustness and interpretability

### 5) Role + Context Stability

Sources:
- career progression table
- within-season windows

Features:
- role volatility: `std(usage)` across seasons
- late-role-shift: `delta_usage_last5_minus_prev5` (when computable)
- team strength proxy (SRS) / conference strength

Why it helps:
- helps interpret late breakouts and transfer jumps

## Implementation Notes

- Always store missing as `NaN`, never 0, and add a boolean mask.
- Prefer exposure-aware shrinkage (minutes/games) before feeding into the model.
- Treat “impact proxies” as model inputs, not targets, unless explicitly supervised.

