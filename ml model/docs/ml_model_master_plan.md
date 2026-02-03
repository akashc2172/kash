# ML Model Master Plan: "The Oracle"
**Version**: 1.0  
**Date**: 2026-01-29  
**Status**: Living Architecture Document

---

## 🌎 1. Zone A: Data Ingestion (The Foundation)
**Goal**: Create a seamless 15-year dataset (2010-2025) despite massive structural changes in source data.

### 1.1 The Source Hierarchy
*   **Modern Era (2019-2025)**: Rich CBD API Data.
    *   Full Shot Coordinates (X,Y)
    *   Official Box Scores (Minutes, TOV, Usage)
    *   Play-by-Play with reliable substitutions.
*   **Historical Era (2010-2018)**: "Lossy" Scraped Data.
    *   **Text-Only PBP**: No X,Y coordinates (Spatial Gap).
    *   **Missing Box Scores**: No official Minutes/Turnovers (Usage Gap).
    *   **Ghost Players**: PBP does not list all 5 players on floor.

### 1.2 The Reconstruction Pipeline (Status: ACTIVE)
1.  **Ghost Solver (`clean_historical_pbp_v2.py`)**: 
    *   Reconstructs 5-man lineups from text.
    *   **Improvement (Planned)**: Move from "Global Most Active" filler to "Windowed Activity" filler (e.g., utilize Rolling 10-minute substitution patterns) to reduce error in blowout minutes.
2.  **Volume Backfill (`derive_minutes_from_historical_pbp.py`)**: 
    *   Derives Minutes/Turnovers from PBP text.
    *   **Status**: Implemented with Period Boundary Logic.
    *   **Output**: `fact_player_season_stats_backfill.parquet`.
3.  **Crosswalk**: Links 145k College IDs to NBA IDs (Fuzzy Matching).

---

## 🏗️ 2. Zone B: The Feature Store (Input Layer)
**Goal**: Normalize features so a 2012 player looks like a 2024 player to the model, without "leaking" the fact that they are from the dark ages.

### 2.1 Feature Tier Strategy (Bias Mitigation)
To handle the "Spatial Brick Wall" (2018/2019 boundary), we strictly separate input features.

| Tier | Description | Availability | Strategy |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Universal)** | Usage, efficiency (TS%), shot zones (Rim/Mid/3), height, team strength. | **100% (2010-2025)** | **Always Active**. The backbone of the model. |
| **Tier 2 (Spatial)** | `avg_shot_dist`, `corner_3_rate`, `rim_purity`. | **~25% (2019+)** | **Dropout / Masking**. During training, randomly mask these features even for modern players so the model learns to predict without them. |

### 2.2 Coverage Masks
We must explicitly pass simple boolean flags to the model so it *knows* if data is missing (vs just being zero).
*   `has_spatial_data` (0/1)
*   `has_athletic_testing` (0/1) (Combine/Pro Day)

---

## 🧠 3. Zone C: Model Architecture (The Brain)
**Goal**: A Multi-Task Transfomer/Regression hybrid that predicts *Translation*, not just raw production.

**Key Concept**: We predict the **GAP** (change/translation), not absolute NBA performance. This normalizes across eras and accounts for the fact that NBA is harder than college.

**See**: `docs/zone_c_gap_concept_clarification.md` for detailed explanation.

### 3.1 The Target Variables (Y)
We are predicting **NBA Translation**, defined as:
`Gap = NBA_Metric - College_Metric`

*   **Primary Target**: `gap_rapm` (NBA 3yr Peak RAPM - College 3yr RAPM).
    *   **Why Peak?** Captures "ceiling" (best 3 consecutive years), avoids injury/decline noise.
    *   **Why Gap?** Normalizes across eras (2010 vs 2024), predicts relative change.
*   **Auxiliary Targets**: 
    *   `gap_ts_legacy` (NBA Year-1 TS% - College Final TS%). Efficiency translation.
    *   `gap_usg_legacy` (NBA Year-1 Usage - College Final Usage). Role translation.
    *   `nba_year1_minutes` (Survival Proxy). Binary: `made_nba = (minutes >= 100)`.

### 3.2 Loss Function Strategy
To handle the massive noise in NBA outcomes, we use a compound loss function:

$$ L_{total} = w_1 \cdot MSE(RAPM_{gap}) + w_2 \cdot MSE(TS_{gap}) + w_3 \cdot MSE(USG_{gap}) + w_4 \cdot BCE(MadeNBA) $$

*   **Rationale**: 
    *   Primary target (`gap_rapm`) is noisy (small sample for 3yr peak).
    *   Auxiliary targets provide additional signal (efficiency, role translation).
    *   Binary target (`made_nba`) is easier to predict (more data) and helps regularize.
*   **Heteroscedastic Variance**: Weight by exposure (variance ∝ 1/(minutes+ε)) to downweight noisy observations.

### 3.3 The "Time Machine" Validation
We cannot use standard K-Fold CV. We must use **Walk-Forward Validation**.
*   **Train**: 2010-2017 (8 seasons, ~15k player-seasons)
*   **Val**: 2018-2019 (transition period, ~4k player-seasons)
*   **Test**: 2020-2022 (modern era, ~5k player-seasons)
*   **Excluded**: 2023-2025 (too recent for 3yr NBA targets)

**See**: `docs/phase4_validation.md` for detailed validation plan.

---

## 🚀 4. Zone D: Inference (The Product)
**Goal**: Generate a "Draft Board" ranking for the current class.

### 4.1 Input Pipeline

**Step 1: Data Ingestion**
1.  Ingest current CBD season (`ingest-season`).
2.  Run `build_college_feature_store_v1.py` (Tier 1 & Tier 2 populated).
3.  Run `build_prospect_career_store_v2.py` (final season features).

**Step 2: Feature Preparation**
1.  Load unified training table builder (`nba_scripts/build_unified_training_table.py`).
2.  Apply feature transforms (era normalization, logit, stabilization).
3.  Generate coverage masks (`has_spatial_data`, `has_athletic_testing`).

**Step 3: League Context Injection**
1.  Adjust for current NCAA offensive environment (pace, 3P rate, etc.).
2.  Normalize features to current season baseline.

**See**: `docs/phase3_model_training.md` Section 3.1 for detailed feature list.

### 4.2 Prediction

**Input**:
- Model receives: `[PlayerFeatures, Tier2_Mask, Has_Spatial, Has_Athletic]`
- Features: See `docs/phase3_model_training.md` Section 3.1 for complete list.

**Output**:
- `predicted_gap_rapm`: Predicted translation (change in RAPM).
- `predicted_gap_ts`: Predicted efficiency translation.
- `predicted_gap_usg`: Predicted role translation.
- `predicted_made_nba`: Probability of making NBA (≥100 minutes).

**Derived**:
- `predicted_nba_rapm = college_rapm + predicted_gap_rapm`
- `predicted_nba_ts = college_ts + predicted_gap_ts`

**Uncertainty Quantification**:
- Use Quantile Regression or Dropout-Ensemble to output confidence intervals.
- Output: `[predicted_low, predicted_median, predicted_high]`
- Interpretation: "High Ceiling / Low Floor" vs "Safe Pick"

**See**: `docs/phase3_model_training.md` for detailed implementation.

---

## 🗓️ 5. Implementation Roadmap (The "Next Steps")

**Detailed Plans**: Each phase has a dedicated document with step-by-step implementation details.

### Phase 2: Feature Store Hardening
**See**: `docs/phase2_feature_store_hardening.md` for complete plan.

*   **2.1**: Finish Historical Backfill (Run `derive_minutes` on 2010-2018).
    *   Status: ✅ 2015, 2017 complete. ⏳ 2010-2014, 2016, 2018 pending (scraping in background).
*   **2.2**: Implement "Windowed Activity" for Ghost Fill improvement.
    *   Status: ✅ Implementation created (`clean_historical_pbp_v2_windowed.py`). ⏳ Validation pending.
*   **2.3**: Build the **Unified Training Table** (Merge all Parquets into one wide matrix).
    *   Status: ✅ **COMPLETE** (2026-02-01). Script: `nba_scripts/build_unified_training_table.py`.
*   **2.4**: Feature Normalization & Era Adjustment.
    *   Status: ✅ **COMPLETE**. Integrated into `build_unified_training_table.py`.

### Phase 3: Model Training
**See**: `docs/phase3_model_training.md` for complete plan.
**See**: `docs/model_architecture_dag.md` for visual pipeline DAG.

*   **3.1**: Build `NBADataLoader` (PyTorch).
    *   Status: ⏳ Deferred. Using pandas-based loading for XGBoost baseline.
*   **3.2**: Implement Tier 2 Masking (Dropout).
    *   Status: ✅ **COMPLETE**. `has_spatial_data` mask in unified table, NaN handling in XGBoost.
*   **3.3**: Train Baseline Model (XGBoost) -> Advanced Model (MLP/Transformer).
    *   Status: ✅ **COMPLETE** (2026-02-01). Script: `nba_scripts/train_baseline_xgboost.py`.
    *   Includes: Walk-forward validation, multi-task targets, feature importance.
*   **3.4**: Pipeline Orchestration.
    *   Status: ✅ **COMPLETE** (2026-02-01). Script: `nba_scripts/run_training_pipeline.py`.

### Phase 4: Validation
**See**: `docs/phase4_validation.md` for complete plan.

*   **4.1**: Run Walk-Forward Validation.
    *   Status: ✅ **COMPLETE**. Integrated into `train_baseline_xgboost.py` (train 2010-2017, val 2018-2019, test 2020-2022).
*   **4.2**: Analyze "Misses" (Why did we miss on Jokic? Why did we like Player X?).
    *   Status: ⏳ Pending. Requires trained model output.
*   **4.3**: Feature Importance Analysis.
    *   Status: ✅ **COMPLETE**. Output in `models/xgboost_baseline_*/feature_importance.csv`.
*   **4.4**: Calibration Analysis.
    *   Status: ⏳ Pending. Script: `nba_scripts/analyze_calibration.py` (new).
*   **4.5**: Model Comparison & Selection.
    *   Status: ⏳ Pending. Script: `nba_scripts/compare_models.py` (new).