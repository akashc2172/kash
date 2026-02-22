#!/usr/bin/env python3
"""Stage 3: Athleticism Imputation
Trains simple light models on the 'measured' combine subset to impute proxies 
for missing combine fields (wingspan, vertical, agility).

Crucially, explicitly flags them with `is_imputed=1` and `_sd` uncertainty values
to prevent silent provenance mixing, as demanded by Codex.

Generates `data/combine/fact_player_combine_imputed.parquet`
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent

COMBINE_MEASUREMENTS = BASE / "data" / "warehouse_v2" / "fact_player_combine_measurements.parquet"
FOUNDATION_FEATURES = BASE / "data" / "training" / "foundation_college_table.parquet"
OUT_PATH = BASE / "data" / "warehouse_v2" / "fact_player_combine_imputed.parquet"

TARGET_COLS = [
    'wingspan_in',
    'standing_reach_in',
    'no_step_vertical_in',
    'max_vertical_in',
    'lane_agility_s',
    'three_quarter_sprint_s'
]

# Physical + Box predictors to inform the imputation model
PREDICTOR_COLS = [
    'college_height_in', 'college_weight_lbs',
    'college_dunk_rate', 'college_putback_rate', 'college_transition_freq',
    'college_rim_pressure_index', 'college_stl_total_per100poss', 'college_blk_total_per100poss',
    'college_ast_total_per100poss', 'college_minutes_total'
]

def train_and_impute(df_train: pd.DataFrame, df_infer: pd.DataFrame, target: str):
    """Train a simple model on the measured rows, predict for the missing rows."""
    
    # Filter to rows that actually have the target measurement
    df_fit = df_train[df_train[target].notna()].copy()
    if len(df_fit) < 100:
        logger.warning(f"Not enough training rows for {target} ({len(df_fit)}). Returning NaNs.")
        return pd.Series(np.nan, index=df_infer.index), pd.Series(np.nan, index=df_infer.index)
        
    # X and y
    use_preds = [c for c in PREDICTOR_COLS if c in df_fit.columns and c in df_infer.columns]
    if len(use_preds) < 3:
        logger.warning("Too few predictor columns for %s (%d). Returning NaNs.", target, len(use_preds))
        return pd.Series(np.nan, index=df_infer.index), pd.Series(np.nan, index=df_infer.index)

    X_train = df_fit[use_preds].fillna(0)
    y_train = df_fit[target]
    
    X_infer = df_infer[use_preds].fillna(0)
    
    # XGBoost with Quantile Regression to get Mean AND Variance (Uncertainty)
    # We approximate SD from the IQR of the 25th and 75th percentiles.
    
    # Check if XGBoost supports quantile out of the box (requires newer versions)
    # If not, we fall back to standard squared error + a constant variance.
    # For now, let's try the modern quantile objective syntax.
    
    params_median = {'objective': 'reg:quantileerror', 'quantile_alpha': 0.5, 'n_estimators': 50, 'random_state': 42, 'verbosity': 0}
    params_p25 = {'objective': 'reg:quantileerror', 'quantile_alpha': 0.25, 'n_estimators': 50, 'random_state': 42, 'verbosity': 0}
    params_p75 = {'objective': 'reg:quantileerror', 'quantile_alpha': 0.75, 'n_estimators': 50, 'random_state': 42, 'verbosity': 0}
    
    try:
        model_median = xgb.XGBRegressor(**params_median).fit(X_train, y_train)
        model_p25 = xgb.XGBRegressor(**params_p25).fit(X_train, y_train)
        model_p75 = xgb.XGBRegressor(**params_p75).fit(X_train, y_train)
        
        pred_median = model_median.predict(X_infer)
        pred_p25 = model_p25.predict(X_infer)
        pred_p75 = model_p75.predict(X_infer)
        pred_sd = (pred_p75 - pred_p25) / 1.349
    except xgb.core.XGBoostError:
        logger.warning("XGBoost version does not fully support quantileerror. Falling back to squared error mean imputation.")
        model_mean = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42, verbosity=0).fit(X_train, y_train)
        pred_median = model_mean.predict(X_infer)
        # Empirical dummy standard deviation (e.g., 2.0 inches)
        pred_sd = np.full_like(pred_median, 2.0)
    
    return pred_median, pred_sd

def main():
    if not COMBINE_MEASUREMENTS.exists() or not FOUNDATION_FEATURES.exists():
        logger.error("Missing prerequisites for imputation")
        return
        
    df_meas = pd.read_parquet(COMBINE_MEASUREMENTS)
    df_found = pd.read_parquet(FOUNDATION_FEATURES)
    
    logger.info(f"Loaded {len(df_meas)} measured rows, {len(df_found)} foundation rows")
    
    # We want to impute for everyone in the foundation table.
    join_cols = ['athlete_id'] + [c for c in TARGET_COLS if c in df_meas.columns]
    keep_predictors = [c for c in PREDICTOR_COLS if c in df_found.columns]
    
    df_master = df_found[['athlete_id', 'college_final_season'] + keep_predictors].copy()
    df_master = df_master.merge(df_meas[join_cols], on='athlete_id', how='left')
    
    out_df = df_master[['athlete_id', 'college_final_season']].copy()
    out_df['is_imputed'] = 1
    
    for target in TARGET_COLS:
        if target not in df_master.columns:
            logger.warning(f"{target} not in measured block")
            continue
            
        logger.info(f"Training imputation model for: {target}")
        pred_val, pred_sd = train_and_impute(df_master, df_master, target)
        
        # We only output the proxies explicitly renamed
        out_col = target.replace('_in', '').replace('_s', '') + '_imputed'
        sd_col = target.replace('_in', '').replace('_s', '') + '_imputed_sd'
        
        out_df[out_col] = pred_val
        out_df[sd_col] = pred_sd
        
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)
    logger.info(f"Imputed combine block saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
