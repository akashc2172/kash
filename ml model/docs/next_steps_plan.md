# Next Steps Plan: NCAA Data Pipeline & Model Development

**Date**: 2026-01-29  
**Status**: Active Planning  
**Context**: Spatial data integration complete; continuing NCAA.org scraping

---

## Executive Summary

This plan outlines the **next steps** for the NCAA prospect modeling pipeline while we continue scraping historical data from `stats.ncaa.org`. The plan is organized by **priority** and **dependency**, with clear execution milestones.

**Current State**:
- ✅ Spatial Tier 2 features implemented (`corner_3_rate`, `deep_3_rate`, `rim_purity`, `shot_dist_var`)
- ✅ Coverage-aware gating logic validated
- ✅ Historical lineup reconstruction working (2015, 2017 validated)
- ⏳ Ongoing: Historical data scraping from NCAA.org
- ⏳ Pending: Historical box score backfill (2006-2024)

---

## Strict Run Guardrails (2026-02-19 Addendum)

This document now defers hard-run execution policy to:
- `mistake_prevention_retrospective_2026-02-19.md`
- `nba_pretrain_gate.md`
- hardening runner stage audits under `data/audit/hardening_run_*`

### Mandatory before any "big run"
1. Complete stage-gated data-quality hardening (snapshot, integrity sweep, crosswalk QA, DAG reconciliation).
2. Confirm no critical contract drift vs DAG docs.
3. Confirm dead critical branches are either populated or explicitly masked and gated off.
4. Confirm coverage floors are at/above locked baseline for peak RAPM, Year-1 EPM, and dev-rate.
5. Publish final GO/NO-GO audit prior to full training.

---

## Phase 1: Data Ingestion & Quality Assurance (Weeks 1-4)

### 1.1 Historical Scraping Continuation

**Goal**: Expand historical coverage to maximize training data for the 15-year RApM model.

**Tasks**:
1. **Prioritize Missing Seasons**:
   - **High Priority**: 2010-2014 (early modern era, bridges to 2015 validated data)
   - **Medium Priority**: 2016, 2018-2019 (fill gaps between validated years)
   - **Lower Priority**: 2006-2009 (pre-2010, but valuable for era normalization)

2. **Scraping Execution**:
   - Run `college_scripts/scrapers/scrape_ncaa_master.py` for target seasons
   - Store raw CSVs in `data/manual_scrapes/{YEAR}/`
   - **Validation Checkpoint**: After each season batch, verify:
     - File count matches expected game count
     - Sample files contain "Enters Game" / "Leaves Game" text
     - Team names are parseable

3. **Lineup Reconstruction**:
   - Run `college_scripts/utils/clean_historical_pbp_v2.py` for newly scraped seasons
   - Output: `data/fact_play_historical_{YEAR}_clean.parquet`
   - **Quality Gate**: Verify 5-on-floor consistency (no 4-player or 6-player lineups)

**Deliverables**:
- Raw scrapes for 2010-2014, 2016, 2018-2019
- Cleaned historical PBP parquet files
- Coverage report: `data/historical_coverage_report.csv` (seasons × games × players)

**Success Metrics**:
- Historical coverage: 2010-2025 (16 seasons) with ≥80% game coverage per season
- Lineup reconstruction: ≥95% of games have valid 5-on-floor lineups

---

### 1.2 Historical Box Score Backfill (Critical Blocker)

**Problem**: `fact_player_season_stats` in DuckDB only has 2005 and 2025. This blocks:
- Historical usage rate calculation
- Advanced rate stats (AST%, TO%, etc.) for 2006-2024
- Phase 4 `gap_usg_legacy` coverage (currently only 2 players)

**Solution Options**:

**Option A: Derive from Historical PBP** (Recommended)
- **Script**: `college_scripts/derive_minutes_from_historical_pbp.py` (new)
- **Logic**:
  1. Load cleaned historical PBP (`fact_play_historical_{YEAR}_clean.parquet`)
  2. For each player-game, sum `seconds_on` from lineup reconstruction
  3. Aggregate to player-season: `minutes_total = sum(seconds_on) / 60`
  4. Extract turnovers from `playText` patterns (e.g., "Turnover by PLAYER")
  5. Output: `data/warehouse_v2/fact_player_season_stats_backfill_{YEAR}.parquet`
- **Advantages**: Uses existing lineup reconstruction, no new scraping needed
- **Limitations**: TOV extraction from text is noisy; may need manual validation

**Option B: Scrape Box Scores from NCAA.org**
- **Script**: New scraper for `stats.ncaa.org/teams/.../stats`
- **Logic**: Parse HTML tables for per-player season totals
- **Advantages**: More accurate (official box scores)
- **Limitations**: Requires new scraping infrastructure, rate limiting

**Recommendation**: **Start with Option A** (derive from PBP). If TOV extraction is too noisy, supplement with Option B for critical seasons.

**Deliverables**:
- `fact_player_season_stats_backfill_{YEAR}.parquet` for 2006-2024
- Integration script to merge backfill into DuckDB `fact_player_season_stats`
- Validation report: coverage by season, comparison to known totals

**Success Metrics**:
- `minutes_total` coverage: ≥70% of player-seasons for 2010-2024
- `tov_total` coverage: ≥60% (text extraction is harder)
- Usage gap coverage: increase from 2 → 400-500 players

---

### 1.3 Data Quality Validation Suite

**Goal**: Automated checks to catch data quality issues before they propagate to modeling.

**Script**: `college_scripts/utils/validate_feature_store.py` (new)

**Checks**:
1. **Uniqueness**: `(athlete_id, season, split_id)` is unique in `college_features_v1.parquet`
2. **Range Checks**:
   - All percentage columns ∈ [0, 1]
   - All count columns ≥ 0
   - `shots_total >= fga_total + ft_att`
   - `assisted_made_* <= *_made`
3. **Coverage Checks**:
   - Per season: fraction of athletes with non-null PBP features > threshold
   - Spatial coverage: `xy_coverage` distribution by season (should increase 2019+)
4. **Consistency Checks**:
   - `corner_3_att <= three_att` (corner is subset of 3s)
   - `deep_3_att <= three_att` (deep is subset of 3s)
   - `rim_rest_att <= rim_att` (restricted is subset of rim)
5. **Gating Validation**:
   - Tier 2 features are `NaN` when `xy_shots < 25` (not 0)
   - `corner_3_rate` is `NaN` when `xy_3_shots < 15`
   - `rim_purity` is `NaN` when `xy_rim_shots < 20`

**Deliverables**:
- Validation script with configurable thresholds
- Weekly validation report: `data/validation_reports/validation_{DATE}.json`
- Alert system: fail build if critical checks fail

**Success Metrics**:
- Zero uniqueness violations
- <1% range violations (investigate outliers)
- Coverage trends match expectations (spatial increases 2019+)

---

## Phase 2: Feature Store Enhancements (Weeks 5-8)

### 2.1 ASTz Implementation (Era-Normalized Assist Rate)

**Problem**: Raw AST% has shifted across seasons (cohort-level inflation, not player skill). Example: NCAA freshmen with AST% > 25: 2022=16, 2023=16, 2024=21, 2025=13, 2026=33.

**Solution**: Hierarchical normalization (per proposal spec):
- **Baseline Model**: `AST%_{j,s,d} | g_j=g ~ StudentT(ν, μ_{s,d,g}, σ_{s,d,g})`
- **Operational Approximation** (Phase 1): Z-score within `(season, role_bucket)` using robust median/MAD

**Implementation**:
1. **Role Buckets**: Define role buckets (Bigs, Wings, Guards) from shot profile + usage
   - Logic: `rim_share > 0.5` → Big, `three_share > 0.4` → Wing, else Guard
2. **Baseline Calculation**:
   - For each `(season, role_bucket)`, compute median/MAD of AST% (or AST proxy)
   - Store: `ast_baseline_median`, `ast_baseline_mad`
3. **Z-Score**:
   - `ast_z = (ast_raw - baseline_median) / baseline_mad`
   - Store: `ast_proxy_raw`, `ast_proxy_z`, `ast_proxy_n` (sample size)

**Script**: `college_scripts/build_astz_normalization.py` (new)

**Deliverables**:
- `ast_proxy_raw`, `ast_proxy_z`, `ast_proxy_n` columns in `college_features_v1.parquet`
- Baseline lookup table: `data/college_feature_store/ast_baselines_{season}.parquet`
- Validation: ASTz distribution should be ~N(0,1) within each season/role

**Success Metrics**:
- ASTz removes season-level inflation (correlation with season < 0.1)
- ASTz correlates with NBA playmaking traits (validate on historical NBA players)

---

### 2.2 Recency-Weighted Aggregates & "Best 15-Game Window"

**Goal**: Add temporal weighting to capture "recent form" and "peak performance" signals.

**Implementation**:
1. **Recency-Weighted Season Aggregate**:
   - Exponential weights: `w_i = exp(-λ * (games_ago))` with half-life tuned via CV
   - Precompute EWMA aggregates per athlete-season
   - Store as additional `window_id = 'recency_weighted'`

2. **Best 15-Game Window**:
   - For each athlete-season, find the 15-game rolling window with highest impact (e.g., `on_net_rating`)
   - Store as `window_id = 'best_15'`
   - Use-case: "Stress test" features (proposal recommendation)

**Script**: Extend `cbd_pbp/windows.py` or create `college_scripts/build_temporal_windows.py`

**Deliverables**:
- `fact_player_window` extended with `recency_weighted` and `best_15` window_ids
- Feature store integration: add windowed features to `college_features_v1` (optional split axis)

**Success Metrics**:
- Recency-weighted features improve out-of-sample prediction vs full-season aggregates
- Best-15 window identifies "peak performance" players (validated on known breakout seasons)

---

### 2.3 Extended Spatial Features (Left/Right Corner, Shot Dispersion)

**Goal**: Add more granular spatial traits for role fingerprinting.

**New Features**:
1. **Left vs Right Corner 3 Rate**:
   - `left_corner_3_rate`, `right_corner_3_rate`
   - Logic: `(loc_x/10.0 - hoop_x) < 0` → left, else right
   - Gating: `xy_3_shots >= 15` (same as corner_3_rate)

2. **Shot Dispersion (Already Implemented)**:
   - ✅ `shot_dist_var` (variance of distance) - already in feature store
   - Use-case: "Tight shot profile" vs "chaotic" role fingerprint

3. **Rim Arc Purity (Optional)**:
   - Share of rim attempts in restricted area (< 4ft) - ✅ already implemented as `rim_purity`

**Implementation**:
- Add SQL aggregates to `build_college_feature_store_v1.py` (left/right corner counts)
- Python gating: same thresholds as existing corner features

**Deliverables**:
- `left_corner_3_rate`, `right_corner_3_rate` in `college_features_v1.parquet`
- Validation: left/right should be roughly symmetric (no coordinate bias)

**Success Metrics**:
- Left/right rates are uncorrelated (different roles prefer different corners)
- Dispersion correlates with role archetype (validated on known players)

---

## Phase 3: Model Integration & Training Prep (Weeks 9-12)

### 3.1 Data Loader Integration

**Goal**: Update `nba_scripts/nba_data_loader.py` to consume college features correctly.

**Tasks**:
1. **Load College Features**:
   - Function: `load_college_features(warehouse_dir)` → `pd.DataFrame`
   - Grain: `(athlete_id, season, split_id)`
   - Default split: `split_id == 'ALL__ALL'` (full season)

2. **Feature Selection**:
   - Tier 1 (Universal): Always include
   - Tier 2 (Spatial): Include with explicit missingness handling
   - ASTz: Include if available

3. **Leakage Prevention**:
   - Add college feature columns to `FORBIDDEN_FEATURE_COLUMNS` if they encode post-draft info
   - Verify: college features only use games through end of final college season

**Deliverables**:
- Updated `nba_data_loader.py` with college feature loading
- Integration test: verify college features join correctly to NBA targets via crosswalk

**Success Metrics**:
- College features load without errors
- No leakage detected (college features don't correlate with post-draft NBA outcomes)

---

### 3.2 Feature Transform Pipeline Updates

**Goal**: Ensure college features are properly normalized/transformed for model training.

**Tasks**:
1. **Era Normalization**:
   - Z-score by season for rates with drift (AST%, pace, 3P rates)
   - Use baseline stats from feature store (if available)

2. **Missingness Handling**:
   - Tier 2 spatial features: explicit `NaN` (not 0)
   - Model layer: masking or imputation strategy (TBD by model architecture)

3. **Stabilization**:
   - Rate features (FG%, assisted shares): apply Beta prior stabilization if needed
   - Store both raw and stabilized rates

**Script**: Update `nba_scripts/nba_feature_transforms.py`

**Deliverables**:
- Transform functions for college features
- Validation: transformed features have reasonable distributions (no extreme outliers)

**Success Metrics**:
- Transformed features are approximately normal (or appropriate distribution)
- Missingness patterns are handled correctly (no silent zeros)

---

### 3.3 Training Dataset Builder

**Goal**: Create unified training dataset joining college features → NBA targets.

**Script**: `nba_scripts/build_training_dataset.py` (new)

**Logic**:
1. Load college features (default: `split_id == 'ALL__ALL'`)
2. Load NBA targets (`fact_player_peak_rapm`, `fact_player_year1_epm`)
3. Join via `dim_player_nba_college_crosswalk`
4. Apply leakage filters (only use college games through draft date)
5. Apply feature transforms
6. Output: `data/training/train_dataset_{DATE}.parquet`

**Deliverables**:
- Training dataset builder script
- Sample dataset: `data/training/train_dataset_sample.parquet` (first 1000 players)
- Dataset documentation: column descriptions, missingness patterns, coverage stats

**Success Metrics**:
- Training dataset has ≥2000 players with both college features and NBA targets
- No leakage detected (manual spot-check of high-profile players)

---

## Phase 4: Model Architecture & Training (Weeks 13-16)

### 4.1 Model Architecture Design

**Goal**: Design the Generative Prospect Model architecture per proposal spec.

**Key Components**:
1. **Latent Trait Encoder**: `p(z_i | x_{i,t})` - maps college features to latent skills
2. **Measurement Head**: `p(x_{i,t} | z_i, context)` - reconstructs college observables
3. **Impact Head**: `p(y_i,peak | z_i, context)` - predicts peak RAPM
4. **Auxiliary Heads**: `p(a_i,1 | z_i, context)` - predicts year-1 EPM, gap features

**Architecture Options**:
- **Transformer**: Attention over player-seasons, latent bottleneck
- **Tree Ensemble**: XGBoost/LightGBM with multi-task learning
- **Neural Network**: MLP with shared encoder, task-specific heads

**Deliverables**:
- Architecture spec document: `docs/model_architecture_spec.md`
- Prototype implementation: `nba_scripts/nba_model_architecture.py`

**Success Metrics**:
- Architecture supports multi-task learning (RAPM + EPM + gaps)
- Latent space is interpretable (validate on known player archetypes)

---

### 4.2 Training Pipeline

**Goal**: End-to-end training script with proper validation and monitoring.

**Script**: `nba_scripts/nba_training_pipeline.py` (already exists, needs updates)

**Components**:
1. **Data Loading**: Use `build_training_dataset.py` output
2. **Train/Test Split**: Time-based (hold out recent drafts) or player-based
3. **Feature Transforms**: Apply era normalization, stabilization
4. **Model Training**: Multi-task loss (RAPM + EPM + gaps)
5. **Evaluation**: RMSE, MAE, Spearman correlation, calibration metrics
6. **Monitoring**: TensorBoard/MLflow logging

**Deliverables**:
- Updated training pipeline
- Training run outputs: `models/nba_prospect_model_{DATE}/`
- Evaluation report: `reports/training_eval_{DATE}.json`

**Success Metrics**:
- Out-of-sample RAPM prediction: RMSE < 2.0 (minutes-weighted)
- Calibration: PIT histogram is uniform (predicted intervals match actual coverage)

---

### 4.3 Model Validation & Backtesting

**Goal**: Validate model on historical drafts to ensure it would have worked in the past.

**Tasks**:
1. **Historical Backtest**:
   - Train on 2010-2020 drafts, predict 2021-2025 drafts
   - Compare predicted vs actual RAPM for known players
   - Identify systematic biases (e.g., overrating certain archetypes)

2. **Ablation Studies**:
   - Remove Tier 2 spatial features → measure performance drop
   - Remove ASTz normalization → measure era bias
   - Remove recency weighting → measure recency bias

3. **Interpretability Analysis**:
   - Feature importance (which college features predict NBA success?)
   - Latent space visualization (t-SNE of z_i, color by NBA outcome)
   - Archetype discovery (cluster z_i, label clusters with known players)

**Deliverables**:
- Backtest report: `reports/backtest_{DATE}.md`
- Ablation results: `reports/ablation_{DATE}.json`
- Interpretability visualizations: `reports/interpretability_{DATE}.html`

**Success Metrics**:
- Backtest: Top-10 predicted players have ≥70% "success rate" (RAPM > 0)
- Ablation: Tier 2 features improve prediction by ≥5% (RMSE reduction)
- Interpretability: Latent clusters align with known player archetypes

---

## Phase 5: Production Deployment & Monitoring (Weeks 17-20)

### 5.1 Inference Pipeline

**Goal**: Deploy model for prospect-only inference (no NBA data).

**Script**: `nba_scripts/nba_prospect_inference.py` (new)

**Logic**:
1. Load college features for new prospects (2026+)
2. Apply same transforms as training
3. Run model forward pass → get `z_i` and `y_i,peak` predictions
4. Output: `data/inference/prospect_predictions_{DATE}.parquet`

**Deliverables**:
- Inference script
- Sample predictions: `data/inference/prospect_predictions_sample.parquet`
- Prediction format spec: columns, uncertainty intervals, archetype labels

**Success Metrics**:
- Inference runs without errors
- Predictions have calibrated uncertainty (80% intervals contain true outcome 80% of time)

---

### 5.2 Monitoring & Retraining

**Goal**: Set up continuous monitoring and retraining pipeline.

**Tasks**:
1. **Data Drift Detection**:
   - Monitor college feature distributions over time
   - Alert if distributions shift significantly (e.g., 3P rate increases)

2. **Model Performance Tracking**:
   - Track prediction accuracy as new NBA outcomes become available
   - Retrain if performance degrades (e.g., RMSE increases >10%)

3. **Automated Retraining**:
   - Quarterly retrain with latest data
   - A/B test new model vs production model

**Deliverables**:
- Monitoring dashboard: `reports/monitoring_dashboard.html`
- Retraining script: `scripts/retrain_model.sh`
- Alert system: email/Slack notifications for drift/performance issues

**Success Metrics**:
- Drift detected within 1 week of distribution shift
- Retraining improves performance (or maintains if no new signal)

---

## Dependencies & Blockers

### Critical Blockers
1. **Historical Box Score Backfill** (Phase 1.2): Blocks usage gap features and advanced rate stats
2. **Historical Scraping Completion** (Phase 1.1): Need 2010-2019 coverage for full 15-year model

### Nice-to-Have (Not Blocking)
1. ASTz implementation (can use raw AST% initially)
2. Recency weighting (can use full-season aggregates initially)
3. Extended spatial features (left/right corner, etc.)

---

## Success Criteria (Overall)

**Data Quality**:
- Historical coverage: 2010-2025 with ≥80% game coverage per season
- Feature store: ≥2000 players with complete college features + NBA targets
- No data quality violations (uniqueness, range checks, leakage)

**Model Performance**:
- Out-of-sample RAPM prediction: RMSE < 2.0 (minutes-weighted)
- Calibration: 80% prediction intervals contain true outcome 80% of time
- Backtest: Top-10 predicted players have ≥70% success rate

**Production Readiness**:
- Inference pipeline deployed and tested
- Monitoring dashboard operational
- Documentation complete (all .md files up to date)

---

## Next Immediate Actions (This Week)

### ✅ COMPLETED (2026-02-01)

1. **Training Pipeline Scripts Created**
   - ✅ `nba_scripts/build_unified_training_table.py` - Merges college features + NBA targets
   - ✅ `nba_scripts/train_baseline_xgboost.py` - XGBoost baseline with walk-forward validation
   - ✅ `nba_scripts/run_training_pipeline.py` - Pipeline orchestrator with prereq checks
   - ✅ `docs/model_architecture_dag.md` - Visual DAG of full pipeline

2. **Directory Structure Created**
   - ✅ `data/training/` - For unified training table output
   - ✅ `data/warehouse_v2/` - For NBA targets
   - ✅ `data/college_feature_store/` - For college features
   - ✅ `models/` - For trained model outputs

### ⏳ IN PROGRESS (Background)

1. **Historical Data Scraping** (User running in background)
   - Gathering minutes/rotation data for usage and RAPM inputs
   - Target seasons: 2010-2014, 2016, 2018-2019
   - This will feed into the pipeline once complete

### 🔜 NEXT UP (When Data Ready)

1. **Install XGBoost Dependency**
   ```bash
   brew install libomp  # Required for XGBoost on Mac
   ```

2. **Generate/Copy Data Files**
   - Run feature store scripts OR copy from main workspace:
     - `college_features_v1.parquet`
     - `prospect_career_v1.parquet`
     - `dim_player_nba_college_crosswalk.parquet`
     - `fact_player_peak_rapm.parquet`
     - `fact_player_year1_epm.parquet`

3. **Run Training Pipeline**
   ```bash
   python3 nba_scripts/run_training_pipeline.py --all
   ```

4. **Historical Box Score Backfill** (After scraping complete)
   - Run `college_scripts/derive_minutes_from_historical_pbp.py`
   - This unlocks usage gap for historical players

---

**Plan Author**: cursor  
**Last Updated**: 2026-02-01  
**Next Review**: After training pipeline first run
