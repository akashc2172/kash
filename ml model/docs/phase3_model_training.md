# Phase 3: Model Training - Detailed Implementation Plan

**Date**: 2026-01-29  
**Status**: Planning  
**Owner**: TBD

---

## Overview

Phase 3 focuses on **building and training the NBA Prospect Model**. This phase includes data loading, feature transforms, model architecture, and training pipeline.

---

## 3.1 Build NBA Data Loader (PyTorch)

### Objective
Create a PyTorch `DataLoader` that loads the unified training table and prepares batches for training.

### Current State
- `nba_scripts/nba_data_loader.py` exists but is focused on NBA-side data only
- Needs to be extended to load unified training table with college features

### Target Schema

**Input Features (X)** - One row per `(athlete_id, season)`:

**Tier 1 (Universal - Always Available)**:
```
# Shot Profile (Low-Resolution Zones)
- rim_att, rim_made, rim_fg_pct, rim_share
- mid_att, mid_made, mid_fg_pct, mid_share  
- three_att, three_made, three_fg_pct, three_share
- ft_att, ft_made, ft_pct, ft_rate

# Creation Context
- assisted_share_rim, assisted_share_three, assisted_share_mid
- high_lev_att_rate, high_lev_fg_pct
- garbage_att_rate

# Impact Metrics
- on_net_rating, on_ortg, on_drtg
- seconds_on, games_played

# Team Context
- team_pace, conference, is_power_conf
- opp_rank (opponent strength proxy)

# Volume & Usage
- minutes_total, tov_total
- usage_proxy (derived: (FGA + 0.44*FTA + TOV) / poss)

# Career Summary (Final Season)
- final_trueShootingPct, final_usage
- career_years
- slope_* (trajectory features)
- career_wt_* (recency-weighted)
```

**Tier 2 (Spatial - 2019+ Only, with Masking)**:
```
- avg_shot_dist, shot_dist_var
- corner_3_rate, corner_3_pct
- deep_3_rate
- rim_purity
- xy_shots, xy_3_shots, xy_rim_shots (coverage counts)
```

**Coverage Masks**:
```
- has_spatial_data (0/1)
- has_athletic_testing (0/1)
```

**Targets (Y)** - Only for training, NOT for inference:
```
# Primary Target
- gap_rapm (NBA 3yr Peak RAPM - College 3yr RAPM)

# Auxiliary Targets
- gap_ts_legacy (NBA Year-1 TS% - College Final TS%)
- gap_usg_legacy (NBA Year-1 Usage - College Final Usage)
- nba_year1_minutes (Survival proxy)

# Binary Target
- made_nba (0/1) - Did player play ≥100 NBA minutes?
```

### Implementation

**Script**: `nba_scripts/nba_data_loader.py` (extend existing)

**Key Functions**:

```python
class NBAProspectDataset(Dataset):
    """
    PyTorch Dataset for NBA Prospect Model.
    
    Loads unified training table and applies feature transforms.
    """
    def __init__(
        self,
        unified_table_path: str,
        target_seasons: List[int] = None,  # None = all seasons
        include_targets: bool = True,  # False for inference
        apply_transforms: bool = True,
        tier2_dropout_rate: float = 0.0,  # For training: randomly mask Tier 2
    ):
        # Load unified table
        # Filter by target_seasons
        # Apply feature transforms (z-score, logit, etc.)
        # Generate coverage masks
        # Apply Tier 2 dropout (if training)
    
    def __getitem__(self, idx):
        # Return:
        # - X: Feature tensor [num_features]
        # - y_primary: gap_rapm (or None if inference)
        # - y_aux: gap_ts, gap_usg, nba_year1_minutes (or None)
        # - y_binary: made_nba (or None)
        # - masks: coverage masks, target masks
        # - metadata: athlete_id, season, draft_year
```

**Feature Transform Pipeline**:
1. **Era Normalization**: Z-score by season (for rates with drift)
2. **Logit Transform**: For percentages (FG%, assisted share, etc.)
3. **Stabilization**: Beta prior for noisy rates
4. **Missingness**: Explicit `NaN` → `0.0` with coverage mask
5. **Tier 2 Dropout**: Randomly mask Tier 2 features (training only)

**Deliverables**:
- Updated `nba_data_loader.py` with unified table support
- Unit tests: Test data loading, transforms, masking
- Sample batch: Print first batch to verify shapes

**Success Criteria**:
- DataLoader can load unified table
- Feature transforms produce reasonable distributions
- Tier 2 dropout works (random masking during training)

---

## 3.2 Implement Tier 2 Masking (Dropout)

### Objective
Prevent model from over-relying on Tier 2 spatial features (which are missing for 2010-2018).

### Strategy

**Training-Time Dropout**:
- During training, randomly mask Tier 2 features even for modern players (2019+)
- Dropout rate: 30-50% (tune via validation)
- Forces model to learn from Tier 1 features

**Inference-Time**:
- Use actual coverage masks (`has_spatial_data`)
- If `has_spatial_data=0`, set all Tier 2 features to `NaN` (or learned "missing" embedding)

### Implementation

**In DataLoader**:
```python
def apply_tier2_dropout(features, tier2_mask, dropout_rate=0.3):
    """
    Randomly mask Tier 2 features during training.
    
    Args:
        features: Full feature tensor
        tier2_mask: Boolean mask indicating Tier 2 features
        dropout_rate: Probability of masking each Tier 2 feature
    
    Returns:
        features: Features with some Tier 2 masked
        dropout_mask: Which Tier 2 features were masked (for logging)
    """
    if dropout_rate == 0.0:
        return features, None
    
    # Identify Tier 2 feature indices
    tier2_indices = torch.where(tier2_mask)[0]
    
    # Random dropout
    dropout_mask = torch.rand(len(tier2_indices)) < dropout_rate
    masked_indices = tier2_indices[dropout_mask]
    
    # Mask features (set to NaN or learned "missing" value)
    features[masked_indices] = float('nan')  # Or learned embedding
    
    return features, dropout_mask
```

**In Model Architecture**:
- Model should handle `NaN` gracefully (e.g., learned "missing" embeddings)
- Or: Use coverage mask to zero out masked features

**Deliverables**:
- Dropout implementation in DataLoader
- Validation: Model performance with/without Tier 2 dropout
- Tune dropout rate (30%, 40%, 50%) via validation loss

**Success Criteria**:
- Model can train with Tier 2 dropout
- Validation loss doesn't degrade significantly with dropout
- Model learns to rely on Tier 1 features (not just Tier 2)

---

## 3.3 Train Baseline Model (XGBoost)

### Objective
Train a baseline XGBoost model to establish performance benchmarks and validate the pipeline.

### Why XGBoost First?
1. **Fast iteration**: Quick to train, easy to debug
2. **Baseline performance**: Establishes what's achievable with tree models
3. **Feature importance**: Understand which features matter most
4. **Validation**: Verify data pipeline works before complex models

### Implementation

**Script**: `nba_scripts/train_baseline_xgboost.py` (new)

**Model Configuration**:
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
)
```

**Training Setup**:
- **Train**: 2010-2017 seasons
- **Val**: 2018-2019 seasons
- **Test**: 2020-2022 seasons
- **Target**: `gap_rapm` (primary), `gap_ts_legacy`, `gap_usg_legacy` (auxiliary)
- **Multi-task**: Train separate models for each target, or use multi-output

**Feature Engineering**:
- Use transformed features from DataLoader
- Handle missingness: XGBoost handles `NaN` natively
- Coverage masks: Include as features

**Evaluation Metrics**:
- RMSE on `gap_rapm`
- Correlation: Predicted vs Actual
- Feature importance: Top 20 features
- Calibration: Predicted distribution vs Actual distribution

**Deliverables**:
- Trained XGBoost model(s)
- Evaluation report: RMSE, correlation, feature importance
- Predictions on test set: `data/predictions/xgboost_baseline_{DATE}.parquet`

**Success Criteria**:
- RMSE on `gap_rapm` < 2.0 (tune based on actual distribution)
- Correlation > 0.4 (moderate predictive power)
- Feature importance makes sense (e.g., usage, efficiency, impact metrics)

---

## 3.4 Train Advanced Model (MLP/Transformer)

### Objective
Train a deep learning model (MLP or Transformer) that can learn complex feature interactions.

### Architecture Options

**Option A: Multi-Layer Perceptron (MLP)**
- Simple feedforward network
- Pros: Fast, interpretable, good baseline
- Cons: Limited capacity for complex interactions

**Option B: Transformer Encoder**
- Self-attention mechanism
- Pros: Can learn complex feature interactions, state-of-the-art
- Cons: More complex, slower, harder to interpret

**Option C: Hybrid (MLP + Transformer)**
- MLP for Tier 1 features, Transformer for Tier 2
- Pros: Best of both worlds
- Cons: Most complex

### Recommended: MLP First, Then Transformer

**Phase 3.4.1: MLP Baseline**
- Simple 3-4 layer MLP
- Input: All features (flattened)
- Hidden: 256, 128, 64
- Output: Multi-task heads (gap_rapm, gap_ts, gap_usg, made_nba)

**Phase 3.4.2: Transformer (If MLP Underperforms)**
- Transformer encoder with positional encoding
- Input: Feature embeddings (learned)
- Attention: Self-attention over features
- Output: Multi-task heads

### Implementation

**Script**: `nba_scripts/nba_training_pipeline.py` (already exists, needs updates)

**Model Architecture** (MLP):
```python
class NBAProspectMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        # Feature encoder
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Multi-task heads
        self.head_primary = nn.Linear(prev_dim, 1)  # gap_rapm
        self.head_aux_ts = nn.Linear(prev_dim, 1)  # gap_ts
        self.head_aux_usg = nn.Linear(prev_dim, 1)  # gap_usg
        self.head_binary = nn.Linear(prev_dim, 1)  # made_nba (sigmoid)
    
    def forward(self, x):
        z = self.encoder(x)
        return {
            'pred_primary': self.head_primary(z),
            'pred_aux_ts': self.head_aux_ts(z),
            'pred_aux_usg': self.head_aux_usg(z),
            'pred_binary': torch.sigmoid(self.head_binary(z)),
        }
```

**Loss Function** (from `nba_loss_functions.py`):
```python
loss_total = (
    w1 * MSE(gap_rapm_pred, gap_rapm_true) +
    w2 * MSE(gap_ts_pred, gap_ts_true) +
    w3 * MSE(gap_usg_pred, gap_usg_true) +
    w4 * BCE(made_nba_pred, made_nba_true)
)
```

**Training Loop**:
- Optimizer: Adam (lr=1e-3)
- Batch size: 64
- Epochs: 100 (early stopping on validation loss)
- Learning rate schedule: Reduce on plateau

**Deliverables**:
- Trained MLP model
- Training curves: Loss, RMSE over epochs
- Evaluation report: Comparison to XGBoost baseline
- Model checkpoint: `models/nba_prospect_mlp_{DATE}.pt`

**Success Criteria**:
- MLP outperforms XGBoost (lower RMSE, higher correlation)
- Training converges (loss decreases, no overfitting)
- Feature importance (via gradients) makes sense

---

## 3.5 Feature Selection & Ablation Studies

### Objective
Understand which features matter most and optimize feature set.

### Experiments

**Experiment 1: Tier 1 Only vs Tier 1 + Tier 2**
- Train model with Tier 1 only
- Train model with Tier 1 + Tier 2 (with dropout)
- Compare performance

**Experiment 2: Feature Groups**
- Train with different feature groups:
  - Shot profile only
  - Impact metrics only
  - Team context only
  - All features
- Identify which groups are most predictive

**Experiment 3: Ablation**
- Remove top 10 features one-by-one
- Measure performance degradation
- Identify critical features

**Deliverables**:
- Ablation study report: Feature importance, performance by group
- Optimized feature set: Remove redundant/irrelevant features

**Success Criteria**:
- Identified top 20 most important features
- Removed redundant features (correlation > 0.9)
- Performance doesn't degrade with optimized feature set

---

## Dependencies & Blockers

### Critical Blockers
1. **Unified Training Table** (Phase 2.3): Must be complete before training
2. **Feature Transforms** (Phase 2.4): Must be implemented before training

### Nice-to-Have (Not Blocking)
1. Windowed ghost fill (can use global initially)
2. ASTz features (can use raw AST% initially)

---

## Success Criteria (Overall Phase 3)

**Model Performance**:
- XGBoost baseline: RMSE < 2.0, Correlation > 0.4
- MLP: Outperforms XGBoost (RMSE improvement ≥10%)
- Feature importance: Makes intuitive sense

**Pipeline Quality**:
- DataLoader works correctly (no errors, correct shapes)
- Feature transforms produce reasonable distributions
- Tier 2 dropout works (model doesn't over-rely on Tier 2)

---

**Plan Author**: cursor  
**Last Updated**: 2026-01-29
