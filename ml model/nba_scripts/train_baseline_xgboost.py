"""
XGBoost Baseline Model Training
===============================
Trains XGBoost models for NBA prospect prediction using the unified training table.

Key Features:
1. Walk-forward validation (temporal split)
2. Multi-task: separate models for each target
3. Feature importance analysis
4. Heteroscedastic weighting by exposure

Targets:
- Primary: y_peak_ovr (peak 3-year RAPM)
- Auxiliary: gap_ts_legacy, year1_epm_tot, made_nba

Output:
- models/xgboost_baseline_{DATE}/
  - model_y_peak_ovr.json
  - model_gap_ts_legacy.json
  - feature_importance.csv
  - eval_metrics.json
  - predictions.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import logging
import warnings

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Run: pip install xgboost")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
TRAINING_DATA_DIR = BASE_DIR / "data/training"
MODELS_DIR = BASE_DIR / "models"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Walk-forward validation splits (by college final season)
TRAIN_SEASONS = list(range(2010, 2018))  # 2010-2017 (8 seasons)
VAL_SEASONS = list(range(2018, 2020))    # 2018-2019 (2 seasons)
TEST_SEASONS = list(range(2020, 2023))   # 2020-2022 (3 seasons)
# Note: 2023-2025 excluded (too recent for 3yr NBA targets)

# Target definitions
TARGETS = {
    'y_peak_ovr': {
        'type': 'regression',
        'weight_col': 'peak_poss',
        'description': 'Peak 3-year RAPM (primary)',
    },
    'gap_ts_legacy': {
        'type': 'regression',
        'weight_col': 'year1_mp',
        'description': 'TS% translation (NBA Y1 - College Final)',
    },
    'year1_epm_tot': {
        'type': 'regression',
        'weight_col': 'year1_mp',
        'description': 'Year-1 EPM total',
    },
    'made_nba': {
        'type': 'binary',
        'weight_col': None,
        'description': 'Made NBA (Year-1 minutes >= 100)',
    },
}

# Feature column patterns (prefix-based selection)
FEATURE_PREFIXES = [
    'college_',      # Final season features
    'final_',        # Career final values
    'career_',       # Career aggregates
    'slope_',        # Career trajectories
    'delta_',        # Year-over-year changes
]

# Columns to EXCLUDE from features (identity, targets, metadata)
EXCLUDE_COLS = [
    'athlete_id', 'nba_id', 'player_name', 'name',
    'y_peak_ovr', 'y_peak_off', 'y_peak_def', 'peak_poss',
    'gap_ts_legacy', 'gap_usg_legacy', 'gap_dist_leap', 'gap_corner_rate',
    'year1_epm_tot', 'year1_epm_off', 'year1_epm_def',
    'year1_mp', 'year1_tspct', 'year1_usg',
    'made_nba',
    'college_final_season', 'season', 'teamId',
    'split_id',
]

# XGBoost hyperparameters (baseline)
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
}


def load_training_data() -> pd.DataFrame:
    """Load the unified training table."""
    path = TRAINING_DATA_DIR / "unified_training_table.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            f"Run build_unified_training_table.py first!"
        )
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded training data: {len(df):,} rows, {len(df.columns)} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select feature columns based on prefix patterns.
    Excludes identity, target, and metadata columns.
    """
    feature_cols = []
    for col in df.columns:
        # Check if matches feature prefix
        is_feature = any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)
        # Check not excluded
        is_excluded = col in EXCLUDE_COLS
        # Check is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        if is_feature and not is_excluded and is_numeric:
            feature_cols.append(col)
    
    # Add coverage masks
    if 'has_spatial_data' in df.columns:
        feature_cols.append('has_spatial_data')
    
    logger.info(f"Selected {len(feature_cols)} feature columns")
    return sorted(feature_cols)


def create_temporal_splits(
    df: pd.DataFrame,
    season_col: str = 'college_final_season'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create walk-forward temporal splits.
    
    Returns: (train_df, val_df, test_df)
    """
    if season_col not in df.columns:
        logger.warning(f"Season column '{season_col}' not found. Using random split.")
        n = len(df)
        train = df.iloc[:int(0.6*n)]
        val = df.iloc[int(0.6*n):int(0.8*n)]
        test = df.iloc[int(0.8*n):]
        return train, val, test
    
    train = df[df[season_col].isin(TRAIN_SEASONS)]
    val = df[df[season_col].isin(VAL_SEASONS)]
    test = df[df[season_col].isin(TEST_SEASONS)]
    
    logger.info(f"Temporal splits: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


def compute_sample_weights(
    df: pd.DataFrame,
    weight_col: Optional[str],
    weight_ref: float = 2000.0
) -> np.ndarray:
    """
    Compute sample weights based on exposure (heteroscedastic).
    
    Per spec: "σ² ∝ 1/(mp+ε)", so weight ∝ mp
    """
    if weight_col is None or weight_col not in df.columns:
        return np.ones(len(df))
    
    weights = df[weight_col].fillna(0).values
    weights = np.clip(weights / weight_ref, 0.1, 1.0)  # Normalize and clip
    return weights


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    weights_train: np.ndarray,
    target_type: str = 'regression',
) -> xgb.XGBRegressor:
    """
    Train an XGBoost model with early stopping.
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed!")
    
    params = XGB_PARAMS.copy()
    
    if target_type == 'binary':
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
        model = xgb.XGBClassifier(**params)
    else:
        model = xgb.XGBRegressor(**params)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    return model


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    target_type: str = 'regression'
) -> Dict:
    """Compute evaluation metrics."""
    y_pred = model.predict(X)
    y_true = y.values
    
    # Filter out NaN
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'n': 0}
    
    metrics = {'n': len(y_true)}
    
    if target_type == 'binary':
        from sklearn.metrics import roc_auc_score, accuracy_score
        metrics['auc'] = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
        metrics['accuracy'] = accuracy_score(y_true, y_pred > 0.5)
    else:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['spearman'], _ = spearmanr(y_true, y_pred)
    
    return metrics


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importance from XGBoost model."""
    importance = model.feature_importances_
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)


def train_all_models(
    df: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """
    Train models for all targets.
    
    Returns dict of results.
    """
    results = {}
    feature_cols = get_feature_columns(df)
    
    # Create splits
    train_df, val_df, test_df = create_temporal_splits(df)
    
    # Prepare feature matrices
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # Fill NaN with -999 (XGBoost handles this)
    X_train = X_train.fillna(-999)
    X_val = X_val.fillna(-999)
    X_test = X_test.fillna(-999)
    
    all_importance = []
    all_predictions = []
    
    for target_name, target_config in TARGETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for: {target_name}")
        logger.info(f"{'='*60}")
        
        # Check target availability
        if target_name not in df.columns:
            logger.warning(f"Target '{target_name}' not in data, skipping")
            continue
        
        # Get targets
        y_train = train_df[target_name]
        y_val = val_df[target_name]
        y_test = test_df[target_name]
        
        # Filter to non-null targets
        train_mask = y_train.notna()
        val_mask = y_val.notna()
        test_mask = y_test.notna()
        
        n_train = train_mask.sum()
        n_val = val_mask.sum()
        n_test = test_mask.sum()
        
        logger.info(f"Samples: train={n_train:,}, val={n_val:,}, test={n_test:,}")
        
        if n_train < 50 or n_val < 10:
            logger.warning(f"Insufficient samples for {target_name}, skipping")
            continue
        
        # Compute weights
        weights_train = compute_sample_weights(
            train_df[train_mask],
            target_config['weight_col']
        )
        
        # Train model
        model = train_xgboost_model(
            X_train[train_mask],
            y_train[train_mask],
            X_val[val_mask],
            y_val[val_mask],
            weights_train,
            target_config['type'],
        )
        
        # Evaluate
        train_metrics = evaluate_model(model, X_train[train_mask], y_train[train_mask], target_config['type'])
        val_metrics = evaluate_model(model, X_val[val_mask], y_val[val_mask], target_config['type'])
        test_metrics = evaluate_model(model, X_test[test_mask], y_test[test_mask], target_config['type'])
        
        logger.info(f"Train: {train_metrics}")
        logger.info(f"Val:   {val_metrics}")
        logger.info(f"Test:  {test_metrics}")
        
        # Store results
        results[target_name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'config': target_config,
        }
        
        # Feature importance
        importance = get_feature_importance(model, feature_cols)
        importance['target'] = target_name
        all_importance.append(importance)
        
        # Predictions for test set
        if test_mask.sum() > 0:
            preds = pd.DataFrame({
                'nba_id': test_df[test_mask]['nba_id'].values if 'nba_id' in test_df.columns else range(test_mask.sum()),
                'athlete_id': test_df[test_mask]['athlete_id'].values if 'athlete_id' in test_df.columns else range(test_mask.sum()),
                f'{target_name}_true': y_test[test_mask].values,
                f'{target_name}_pred': model.predict(X_test[test_mask]),
            })
            all_predictions.append(preds)
        
        # Save model
        model_path = output_dir / f"model_{target_name}.json"
        model.save_model(model_path)
        logger.info(f"Saved model to {model_path}")
    
    # Save combined outputs
    if all_importance:
        importance_df = pd.concat(all_importance, ignore_index=True)
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        logger.info(f"Saved feature importance to {output_dir / 'feature_importance.csv'}")
    
    if all_predictions:
        predictions_df = all_predictions[0]
        for p in all_predictions[1:]:
            predictions_df = predictions_df.merge(p, on=['nba_id', 'athlete_id'], how='outer')
        predictions_df.to_parquet(output_dir / "predictions.parquet", index=False)
        logger.info(f"Saved predictions to {output_dir / 'predictions.parquet'}")
    
    return results


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("XGBoost Baseline Model Training")
    logger.info("=" * 60)
    
    if not HAS_XGBOOST:
        logger.error("XGBoost not installed! Run: pip install xgboost")
        return
    
    # Load data
    try:
        df = load_training_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODELS_DIR / f"xgboost_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Train models
    results = train_all_models(df, output_dir)
    
    # Save metrics
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    for target, res in results.items():
        if 'test' in res:
            if 'rmse' in res['test']:
                logger.info(f"{target}: Test RMSE={res['test']['rmse']:.4f}, R²={res['test']['r2']:.4f}")
            elif 'auc' in res['test']:
                logger.info(f"{target}: Test AUC={res['test']['auc']:.4f}")


if __name__ == "__main__":
    main()
