# Phase 4: Validation & Analysis - Detailed Implementation Plan

**Date**: 2026-01-29  
**Status**: Planning  
**Owner**: TBD

---

## Overview

Phase 4 focuses on **validating the model** using walk-forward validation and analyzing predictions to understand model behavior and identify improvements.

---

## 4.1 Walk-Forward Validation

### Objective
Validate model using time-aware train/val/test splits (no future data leakage).

### Why Walk-Forward?
- Standard K-Fold CV would leak future data (train on 2020, test on 2015)
- We need to simulate real-world usage: train on past, predict future

### Split Strategy

**Train**: 2010-2017 seasons
- **Rationale**: Enough data (8 seasons), before modern era (2019+ spatial data)
- **Players**: ~15,000 player-seasons

**Validation**: 2018-2019 seasons
- **Rationale**: Transition period (some spatial data, but not all)
- **Players**: ~4,000 player-seasons

**Test**: 2020-2022 seasons
- **Rationale**: Modern era (full spatial data), but recent enough for 3yr NBA targets
- **Players**: ~5,000 player-seasons

**Excluded**: 2023-2025
- **Rationale**: Too recent (can't compute 3yr peak RAPM yet)

### Implementation

**Script**: `nba_scripts/walk_forward_validation.py` (new)

**Logic**:
1. Load unified training table
2. Split by `season` (not random)
3. Train model on train set
4. Evaluate on validation set (tune hyperparameters)
5. Evaluate on test set (final performance)
6. Repeat for each model (XGBoost, MLP, Transformer)

**Metrics**:
- **RMSE**: Root mean squared error on `gap_rapm`
- **MAE**: Mean absolute error
- **Correlation**: Pearson correlation (predicted vs actual)
- **R²**: Coefficient of determination
- **Calibration**: Predicted distribution vs actual distribution

**Deliverables**:
- Validation report: Metrics for train/val/test splits
- Predictions: `data/predictions/walk_forward_{MODEL}_{DATE}.parquet`
- Learning curves: Loss over epochs (for deep learning models)

**Success Criteria**:
- Test RMSE < 2.0 (tune based on actual distribution)
- Test Correlation > 0.4 (moderate predictive power)
- No overfitting (val loss ≈ test loss)

---

## 4.2 Analyze "Misses" (Error Analysis)

### Objective
Understand why the model fails on certain players and identify systematic biases.

### Analysis Framework

**1. High-Error Cases**
- Identify players with largest prediction errors (|predicted - actual| > 2.0)
- Group by:
  - Position (if available)
  - Conference (Power 5 vs Mid-Major)
  - Draft position (if known)
  - Era (2010-2017 vs 2018-2022)

**2. False Positives (Overrated)**
- Players predicted high `gap_rapm` but actual low
- Examples: High-usage college players who didn't translate
- Question: Did model over-weight usage/efficiency?

**3. False Negatives (Underrated)**
- Players predicted low `gap_rapm` but actual high
- Examples: Jokic, Draymond Green (low-usage, high-impact)
- Question: Did model miss defensive/playmaking signals?

**4. Era Bias**
- Compare model performance on 2010-2017 vs 2018-2022
- Question: Does model work better on modern era (more data)?

**5. Spatial Data Dependency**
- Compare model performance with/without Tier 2 features
- Question: Does model rely too heavily on spatial features?

### Implementation

**Script**: `nba_scripts/analyze_model_errors.py` (new)

**Analysis Functions**:
```python
def identify_high_error_cases(predictions, actuals, threshold=2.0):
    """Find players with largest prediction errors."""
    errors = np.abs(predictions - actuals)
    high_error = errors > threshold
    return high_error

def analyze_false_positives(predictions, actuals, threshold=1.0):
    """Find overrated players (predicted high, actual low)."""
    fp = (predictions > threshold) & (actuals < -threshold)
    return fp

def analyze_false_negatives(predictions, actuals, threshold=-1.0):
    """Find underrated players (predicted low, actual high)."""
    fn = (predictions < threshold) & (actuals > threshold)
    return fn

def compare_by_era(predictions, actuals, seasons):
    """Compare performance by era."""
    pre_2018 = seasons < 2018
    post_2018 = seasons >= 2018
    # Calculate metrics for each era
    return metrics_pre, metrics_post
```

**Deliverables**:
- Error analysis report: High-error cases, false positives/negatives
- Case studies: Deep dive on 5-10 interesting misses (e.g., Jokic, Draymond)
- Bias analysis: Performance by era, conference, position

**Success Criteria**:
- Identified top 20 high-error cases
- Identified systematic biases (if any)
- Actionable insights for model improvement

---

## 4.3 Feature Importance Analysis

### Objective
Understand which features the model relies on most and verify they make sense.

### Methods

**1. XGBoost Feature Importance**
- Built-in `feature_importances_` attribute
- Shows which features are most predictive

**2. Permutation Importance**
- Shuffle each feature, measure performance degradation
- More robust than built-in importance

**3. SHAP Values** (for deep learning models)
- SHapley Additive exPlanations
- Shows feature contributions for individual predictions

**4. Gradient-Based Importance** (for deep learning models)
- Compute gradients of loss w.r.t. features
- Shows which features model is most sensitive to

### Implementation

**Script**: `nba_scripts/analyze_feature_importance.py` (new)

**Functions**:
```python
def compute_permutation_importance(model, X, y, n_repeats=10):
    """Compute permutation importance for each feature."""
    baseline_score = model.score(X, y)
    importances = []
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        permuted_score = model.score(X_permuted, y)
        importance = baseline_score - permuted_score
        importances.append(importance)
    return importances

def compute_shap_values(model, X_sample):
    """Compute SHAP values for deep learning model."""
    import shap
    explainer = shap.DeepExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    return shap_values
```

**Deliverables**:
- Feature importance report: Top 20 features, grouped by category
- SHAP plots: Feature contributions for sample predictions
- Validation: Do important features make intuitive sense?

**Success Criteria**:
- Top 10 features are interpretable (usage, efficiency, impact metrics)
- No surprising features in top 10 (e.g., random noise)
- Feature importance aligns with domain knowledge

---

## 4.4 Calibration Analysis

### Objective
Verify that predicted probabilities/uncertainties are well-calibrated.

### Methods

**1. Reliability Diagrams**
- Plot predicted probability vs actual frequency
- Well-calibrated: Points lie on diagonal

**2. Expected Calibration Error (ECE)**
- Measure average deviation from perfect calibration
- Lower is better (0 = perfect calibration)

**3. Uncertainty Quantification**
- For regression: Check if predicted uncertainty (std) correlates with actual error
- Well-calibrated: High uncertainty → high error, low uncertainty → low error

### Implementation

**Script**: `nba_scripts/analyze_calibration.py` (new)

**Functions**:
```python
def compute_ece(predictions, actuals, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = actuals[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def plot_reliability_diagram(predictions, actuals, n_bins=10):
    """Plot reliability diagram."""
    # Bin predictions
    # Compute actual frequency in each bin
    # Plot
    pass
```

**Deliverables**:
- Calibration report: ECE, reliability diagrams
- Uncertainty analysis: Predicted uncertainty vs actual error

**Success Criteria**:
- ECE < 0.1 (good calibration)
- Reliability diagram shows points near diagonal
- Uncertainty correlates with actual error

---

## 4.5 Model Comparison & Selection

### Objective
Compare XGBoost, MLP, and Transformer models and select best for production.

### Comparison Metrics

**Performance**:
- RMSE, MAE, Correlation, R² (on test set)

**Interpretability**:
- Feature importance (XGBoost > MLP > Transformer)
- SHAP values (all models)

**Efficiency**:
- Training time
- Inference time
- Memory usage

**Robustness**:
- Performance on different eras
- Performance with/without Tier 2 features

### Implementation

**Script**: `nba_scripts/compare_models.py` (new)

**Logic**:
1. Load all trained models
2. Evaluate on test set
3. Compare metrics
4. Generate comparison report

**Deliverables**:
- Model comparison report: Performance, interpretability, efficiency
- Recommendation: Which model to use for production?

**Success Criteria**:
- Clear winner (or trade-offs documented)
- Production model selected

---

## Dependencies & Blockers

### Critical Blockers
1. **Trained Models** (Phase 3): Must complete before validation
2. **Unified Training Table** (Phase 2.3): Must be complete

### Nice-to-Have (Not Blocking)
1. SHAP library (for interpretability)
2. Position data (for error analysis by position)

---

## Success Criteria (Overall Phase 4)

**Validation**:
- Test RMSE < 2.0, Correlation > 0.4
- No overfitting (val ≈ test)

**Analysis**:
- Identified top 20 high-error cases
- Identified systematic biases
- Feature importance makes sense

**Calibration**:
- ECE < 0.1
- Uncertainty correlates with error

**Model Selection**:
- Production model selected
- Trade-offs documented

---

**Plan Author**: cursor  
**Last Updated**: 2026-01-29
