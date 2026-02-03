# College Feature Store v1

> **Status**: Active (Populated with 15-Year History + Spatial Suite)
> **Version**: v1.1
> **Date**: 2026-01-29

## Overview

This repository builds the **College Feature Store v1** from the raw DuckDB warehouse. It produces:

1.  `college_features_v1.parquet`: Main feature table with 20 cross-product splits per athlete-season.
2.  `prospect_career_v1.parquet`: Longitudinal features (Slopes, Deltas, Career Aggregates) for prospect modeling.
3.  `college_impact_v1.parquet`: Impact metrics (True RAPM proxy).
4.  `coverage_report_v1.csv`: Metadata on athlete coverage.

## Implementation Plan (Detailed)

For the **full, executable plan** of how college-side stats should be computed, stabilized, split, and fed into the Generative Prospect Model (including leakage rules, temporal windows, and QA gates), see:
- `docs/college_side_stats_implementation_plan.md`

## Key Features

### 1. Split System (20 Splits)
We generate a cross-product of **Leverage** and **Opponent Strength**:
*   **Leverage**: `ALL`, `HIGH_LEVERAGE` (Score <= 10, < 5 min left), `LOW_LEVERAGE`, `GARBAGE`.
*   **Strength**: `ALL`, `VS_TOP50`, `VS_TOP100`, `VS_OTHERS`, `VS_UNKNOWN`.
*   **Combinations**: `ALL__ALL`, `ALL__VS_TOP50`, `HIGH_LEVERAGE__ALL`, etc.

### 2. Opponent Strength (SRS Proxy)
*   **Method**: Margin-based SRS calculated from `dim_games` (2009-2025).
*   **Logic**: `AvgMargin + AvgOpponentMargin`.
*   **Usage**: Maps every game to an opponent rank (1-364+).

### 3. Imperative Additions
*   **Recruiting**: From `fact_recruiting_players` (Rank, Stars, Rating).
*   **Rim Decisions**: Dunk vs Layup counts (joined from `stg_plays` logic if available, currently simplified).
*   **Team Context**: Pace, Efficiency.

### 4. Tier 2 Spatial Suite (New)
*   **Coordinate-based Traits**: Average Shot Distance, Corner 3 Rate, Deep 3 Rate (>27ft), Rim Purity (<4ft).
*   **Coverage-Aware**: Precision gating implemented (`xy_3_shots >= 15` for Corner/Deep; `xy_rim_shots >= 20` for Purity).
*   **Normalization**: 10.0 scale factor (0-940 range).

### 5. Usage Gap Mitigation
*   **Derived Minutes/TOV**: Reconstructed via PBP text traversal for 2015/2017.
*   **Volume Proxy**: `poss_total` (possessions when player records any defensive/offensive event) available for all seasons where minutes are missing.

### 6. True RAPM
*   Solved via `calculate_historical_rapm.py` across 2010-2025.
*   Joined as labels for modeling.

## Usage

### Prerequisites
*   Python 3.10+
*   `duckdb`, `pandas`, `numpy`, `scipy`

### Running the Build
```bash
python college_scripts/build_college_feature_store_v1.py
```

### Outputs
Files are saved to `data/college_feature_store/`.
