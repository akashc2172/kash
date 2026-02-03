# Long-Term Strategy: NCAA Historical Data & RApM (2010-2025)

## Phase 1: Data Ingestion & Reconstruction (COMPLETED)
- **Scraping**: High-confidence scraping of raw PBP text from `stats.ncaa.org`.
- **Normalization**: Fuzzy-matching NCAA team names to the `dim_teams` warehouse.
- **Lineup Reconstruction**: Implementation of the "Holistic Solver" with Ghost Fill and Backward Propagation to guarantee 5-man lineups.
- **Schema Alignment**: Conversion of historical text into the `fact_play_raw` schema.

## Phase 2: Stint Aggregation & RApM (COMPLETED)
- **Unified Engine**: Created `calculate_historical_rapm.py` which consumes unified Parquet.
- **Stint Detection**: Successfully identifies constant-lineup periods and pace-adjusted possessions using absolute clock.
- **Validation**: Verified the solver on **2015 (VanVleet, Jenkins)** and **2017 (UNC Championship core: Meeks, Jackson, Hicks)**.
- **Solver Settings**: Lambda = 1000.0, Ridge Regression via Conjugate Gradient.

## Phase 2.5: Enhanced RAPM Variants (COMPLETED - Jan 2025)
- **Win Probability Model**: Implemented `compute_win_probability()` using time-weighted logistic function.
- **Leverage Index**: Implemented `compute_leverage_index()` based on pbpstats.com methodology.
  - Calculates expected WP swing from possession outcomes weighted by frequency.
  - Buckets: `garbage`, `low`, `medium`, `high`, `very_high`.
- **RAPM Variants** (all computed per-season):
  - `rapm_standard`: Possession-weighted (original).
  - `rapm_leverage_weighted`: Weights stints by leverage index (clutch performance signal).
  - `rapm_high_leverage`: Only high/very_high leverage stints (crunch time specialists).
  - `rapm_non_garbage`: Excludes garbage time stints.
  - `o_rapm` / `d_rapm`: Offensive and Defensive RAPM split.
  - `rapm_rubber_adj`: Adjusts for rubber-band effect (teams coast when ahead).
- **Rubber Band Adjustment**: `compute_rubber_band_adjustment()` corrects for systematic bias where leading teams lose ground.

## Phase 3: Integration & Global Feature Store (IN PROGRESS)
- **Feature Store**: Updated `build_college_feature_store_v1.py` to pull labels from the merged 2010-2025 RApM library.
- **Spatial Data (Tier 2)**: Implemented "Hybrid Resolution" strategy.
    - **Universal**: Rim/Mid/3 buckets for all history (2010+).
    - **Modern High-Res**: X,Y coordinates (Avg Dist, Corner 3) for 2019+ (Gated by `xy_shots >= 25`).
    - **Normalization**: Confirmed 0-940 scale factor.
- **Usage Gap Fix**: Implemented derivation of Minutes and Turnovers from raw PBP to enable valid Usage Rate calculation for historical prospects.
- **Leakage Protection**: Ensure the allowlist handles 15 years of metadata without future-lookahead bias.
- **New Feature Blocks** (added Jan 2025):
  - **Athleticism**: `dunk_rate`, `dunk_freq`, `putback_rate`, `transition_freq`, `transition_eff`, `rim_pressure_index`.
  - **Defense Activity**: `deflection_proxy`, `contest_proxy` (blocks without fouling).
  - **Decision Discipline**: `pressure_handle_proxy` (TO rate delta under pressure), `clutch_shooting_delta`.
  - **Shot Creation**: `self_creation_rate`, `self_creation_eff`.
  - **Impact Variants**: All RAPM variants from Phase 2.5.
  - **Context**: `leverage_poss_share` (how often player is on court in clutch).

## Phase 4: Model Training Parity
- **Cross-Era Normalization**: Using RApM to compare a 2012 prospect to a 2025 prospect on a neutral baseline.
- **Historical Backtesting**: Evaluating how well the RApM-based `NBAProspectModel` would have predicted the success of current NBA stars.
- **Era Adjustment**: Robust z-score normalization within season to handle 3P revolution, pace changes, etc.

## Phase 5: Advanced Enhancements (PLANNED)
- **Opponent Strength Weighting**: Weight stints by opponent quality (Top 25 = 1.2x, Weak = 0.8x).
- **Team/Conference Style Adjustment**: Normalize steal rates for pressing defenses, etc.
- **Multi-Season RAPM**: Rolling 2-3 year windows for stability.
- **Win Probability Adjusted RAPM (WP-RAPM)**: Full Bayesian model regressing on WP change instead of points.
