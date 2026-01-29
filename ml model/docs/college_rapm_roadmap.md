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

## Phase 3: Integration & Global Feature Store (Next Step)
- **Feature Store**: Update `build_college_feature_store_v1.py` to pull labels from the merged 2010-2025 RApM library.
- **Leakage Protection**: Ensure the allowlist handles 15 years of metadata without future-lookahead bias.

## Phase 4: Model Training Parity
- **Cross-Era Normalization**: Using RApM to compare a 2012 prospect to a 2025 prospect on a neutral baseline.
- **Historical Backtesting**: Evaluating how well the RApM-based `NBAProspectModel` would have predicted the success of current NBA stars.
