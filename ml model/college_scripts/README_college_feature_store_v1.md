# College Feature Store v1

> **Status**: Ready for Build (Pending True RAPM reconstruction)
> **Version**: v1.0
> **Date**: 2026-01-27

## Overview

This repository builds the **College Feature Store v1** from the raw DuckDB warehouse. It produces:

1.  `college_features_v1.parquet`: Main feature table with 20 cross-product splits per athlete-season.
2.  `college_impact_v1.parquet`: Impact metrics (True RAPM proxy).
3.  `coverage_report_v1.csv`: Metadata on athlete coverage.

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

### 4. True RAPM (Planned)
*   currently structured to reconstruct 5v5 lineups from `stg_subs`.
*   **Note**: Requires valid `subIn`/`subOut` event stream.

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
