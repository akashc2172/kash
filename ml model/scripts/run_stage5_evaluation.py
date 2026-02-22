#!/usr/bin/env python3
"""
Stage 5-7: Evaluation, Inference Export, and Dashboard Generation
=================================================================
- Walk-forward temporal cross-validation
- Anti-compression & baseline fallback gates
- Export inference with uncertainty bands
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
MODEL_DIR = BASE / "models" / "stack2026"
AUDIT_DIR = BASE / "data" / "audit"
INFERENCE_DIR = BASE / "data" / "inference"

# Import model definitions
import sys
sys.path.insert(0, str(BASE / "scripts"))
from train_2026_model import Stack2026Model, get_feature_columns, TARGET_COL


def walk_forward_cv(df: pd.DataFrame, feat_cols: list, means: np.ndarray, stds: np.ndarray,
                    ckpt: dict, n_folds: int = 3) -> dict:
    """
    Strict walk-forward temporal cross-validation.
    Train ≤ T-2, Val = T-1, Test = T.
    """
    logger.info("=" * 60)
    logger.info("WALK-FORWARD TEMPORAL CROSS-VALIDATION")
    logger.info("=" * 60)

    if 'draft_year_proxy' not in df.columns:
        logger.warning("No draft_year_proxy column for temporal CV. Using random splits.")
        return {}

    draft_years = sorted(df['draft_year_proxy'].dropna().unique())
    if len(draft_years) < 4:
        logger.warning(f"Only {len(draft_years)} draft years, need at least 4 for walk-forward")
        return {}

    results = []
    # Use last n_folds years as test folds
    test_years = draft_years[-n_folds:]

    for test_year in test_years:
        val_year = test_year - 1
        train_mask = df['draft_year_proxy'] <= test_year - 2
        val_mask = df['draft_year_proxy'] == val_year
        test_mask = df['draft_year_proxy'] == test_year

        df_train = df[train_mask]
        df_val = df[val_mask]
        df_test = df[test_mask]

        if len(df_train) < 50 or len(df_test) < 5:
            continue

        x_train_frame = df_train[feat_cols].copy()
        x_test_frame = df_test[feat_cols].copy()
        for c in feat_cols:
            x_train_frame[c] = pd.to_numeric(x_train_frame[c], errors='coerce')
            x_test_frame[c] = pd.to_numeric(x_test_frame[c], errors='coerce')
        X_train = x_train_frame.fillna(0.0).to_numpy(dtype=np.float32)
        X_test = x_test_frame.fillna(0.0).to_numpy(dtype=np.float32)
        X_train = (X_train - means) / stds
        X_test = (X_test - means) / stds

        y_train = df_train[TARGET_COL].astype(np.float32).fillna(0).values
        y_test = df_test[TARGET_COL].astype(np.float32).fillna(0).values

        y_mean = y_train.mean()
        y_std_val = y_train.std()
        if y_std_val < 1e-6:
            y_std_val = 1.0
        y_train_z = (y_train - y_mean) / y_std_val
        y_test_z = (y_test - y_mean) / y_std_val

        # Train neural model
        input_dim = ckpt['input_dim']
        model = Stack2026Model(input_dim=input_dim, use_hypernetwork=False)
        model.autoencoder.encoder.load_state_dict(ckpt['encoder_state_dict'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train_z.reshape(-1, 1))

        model.train()
        for ep in range(50):
            mu, logvar = model(X_t)
            logvar = torch.clamp(logvar, -4.0, 4.0)
            precision = torch.exp(-logvar)
            loss = 0.5 * (precision * (y_t - mu)**2 + logvar).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_pred, logvar_pred = model(torch.FloatTensor(X_test))
        neural_preds = (mu_pred.numpy().flatten() * y_std_val) + y_mean

        # Train XGBoost baseline
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)

        # Metrics
        neural_rmse = np.sqrt(mean_squared_error(y_test, neural_preds))
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        neural_r2 = r2_score(y_test, neural_preds)
        xgb_r2 = r2_score(y_test, xgb_preds)
        neural_spearman = spearmanr(y_test, neural_preds)[0] if len(y_test) > 3 else 0
        xgb_spearman = spearmanr(y_test, xgb_preds)[0] if len(y_test) > 3 else 0

        fold_result = {
            'test_year': int(test_year),
            'train_size': len(df_train),
            'test_size': len(df_test),
            'neural_rmse': float(neural_rmse),
            'xgb_rmse': float(xgb_rmse),
            'neural_r2': float(neural_r2),
            'xgb_r2': float(xgb_r2),
            'neural_spearman': float(neural_spearman),
            'xgb_spearman': float(xgb_spearman),
            'beats_xgb_rmse': bool(neural_rmse <= xgb_rmse),
            'beats_xgb_rank': bool(neural_spearman >= xgb_spearman),
        }
        results.append(fold_result)
        logger.info(f"  Year {int(test_year)}: Neural RMSE={neural_rmse:.4f} vs XGB RMSE={xgb_rmse:.4f} | "
                     f"Neural ρ={neural_spearman:.3f} vs XGB ρ={xgb_spearman:.3f}")

    return {'folds': results}


def export_inference(df: pd.DataFrame, feat_cols: list, means: np.ndarray, stds: np.ndarray,
                     ckpt: dict, model_path: Path):
    """Export full inference rankings with uncertainty bands."""
    logger.info("=" * 60)
    logger.info("EXPORTING INFERENCE RANKINGS")
    logger.info("=" * 60)

    model_ckpt = torch.load(model_path, weights_only=False)
    input_dim = model_ckpt['input_dim']
    context_cols = model_ckpt.get('context_cols', [])

    model = Stack2026Model(input_dim=input_dim, use_hypernetwork=len(context_cols) > 0)
    model.load_state_dict(model_ckpt['model_state_dict'])
    model.eval()

    x_frame = df[feat_cols].copy()
    for c in feat_cols:
        x_frame[c] = pd.to_numeric(x_frame[c], errors='coerce')
    X = x_frame.fillna(0.0).to_numpy(dtype=np.float32)
    X = (X - means) / stds

    # Target stats for rescaling
    y_raw = df[TARGET_COL].astype(np.float32).fillna(0).values
    y_mean = y_raw.mean()
    y_std = y_raw.std()
    if y_std < 1e-6:
        y_std = 1.0

    ctx = None
    if context_cols:
        ccols = [c for c in context_cols if c in df.columns]
        c_frame = df[ccols].copy()
        for c in ccols:
            c_frame[c] = pd.to_numeric(c_frame[c], errors='coerce')
        ctx_data = c_frame.fillna(0.0).to_numpy(dtype=np.float32)
        if ctx_data.shape[1] < 3:
            ctx_data = np.pad(ctx_data, ((0, 0), (0, 3 - ctx_data.shape[1])))
        ctx = torch.FloatTensor(ctx_data)

    with torch.no_grad():
        mu_pred, logvar_pred = model(torch.FloatTensor(X), ctx)
        sd_pred = torch.exp(0.5 * logvar_pred)

    id_priority = [
        'player_name',
        'nba_id',
        'athlete_id',
        'bbr_id',
        'draft_year',
        'draft_year_proxy',
        'college_final_season',
    ]
    id_cols = [c for c in id_priority if c in df.columns]
    if not id_cols:
        id_cols = ['athlete_id'] if 'athlete_id' in df.columns else []
    df_out = df[id_cols].copy()

    df_out['pred_mu'] = (mu_pred.numpy().flatten() * y_std) + y_mean
    df_out['pred_sd'] = sd_pred.numpy().flatten() * y_std
    df_out['pred_upside'] = df_out['pred_mu'] + 1.0 * df_out['pred_sd']
    df_out['pred_floor'] = df_out['pred_mu'] - 1.0 * df_out['pred_sd']
    df_out['pred_rank'] = df_out['pred_mu'].rank(ascending=False).astype(int)

    if TARGET_COL in df.columns:
        df_out['actual'] = df[TARGET_COL]

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(INFERENCE_DIR / 'stack2026_inference_rankings.csv', index=False)
    df_out.to_parquet(INFERENCE_DIR / 'stack2026_inference_rankings.parquet', index=False)
    logger.info(f"Exported {len(df_out)} rankings to inference/stack2026_inference_rankings.*")

    return df_out


def main():
    if not (MODEL_DIR / 'pretrained_encoder.pt').exists():
        logger.error("Run train_2026_model.py --phase full first!")
        return

    ckpt = torch.load(MODEL_DIR / 'pretrained_encoder.pt', weights_only=False)
    feat_cols = ckpt['feature_cols']
    means = np.array(ckpt['means'])
    stds = np.array(ckpt['stds'])

    df = pd.read_parquet(SUPERVISED_PATH)

    # Ensure feature columns exist
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Stage 5: Walk-forward CV
    cv_results = walk_forward_cv(df, feat_cols, means, stds, ckpt, n_folds=3)
    if cv_results:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_DIR / 'stack2026_walkforward_cv.json', 'w') as f:
            json.dump(cv_results, f, indent=2)

    # Stage 6: Export
    if (MODEL_DIR / 'stack2026_supervised.pt').exists():
        df_ranked = export_inference(df, feat_cols, means, stds, ckpt, MODEL_DIR / 'stack2026_supervised.pt')

        # Stage 5 continued: Hard gate checks
        gate_path = AUDIT_DIR / 'stack2026_model_gate.json'
        if gate_path.exists():
            with open(gate_path) as f:
                gate = json.load(f)

            publish_gate = {
                'std_gate': gate.get('std_gate_pass', False),
                'iqr_gate': gate.get('iqr_gate_pass', False),
                'cv_beats_xgb': False,
                'cv_majority_beats_xgb': False,
                'cv_folds': len(cv_results.get('folds', [])) if cv_results else 0,
                'publish_approved': False,
            }
            if cv_results and cv_results.get('folds'):
                folds = cv_results.get('folds', [])
                wins = sum(1 for f in folds if f.get('beats_xgb_rmse', False) and f.get('beats_xgb_rank', False))
                publish_gate['cv_beats_xgb'] = wins > 0
                publish_gate['cv_majority_beats_xgb'] = wins >= max(1, (len(folds) // 2) + 1)
            publish_gate['publish_approved'] = all([
                publish_gate['std_gate'],
                publish_gate['iqr_gate'],
                publish_gate['cv_majority_beats_xgb'],
            ])
            with open(AUDIT_DIR / 'model_publish_gate.json', 'w') as f:
                json.dump(publish_gate, f, indent=2)
            logger.info(f"Publish gate: {publish_gate}")

    logger.info("=" * 60)
    logger.info("STAGES 5-7 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
