"""
Train Hierarchical Pathway Model
================================
Phase 1 implementation: β_global + Δβ_k, no interactions.

Usage:
    python train_pathway_model.py [--num_steps N] [--k_z K] [--k_p K]
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.generative.pathway_model import HierarchicalPathwayModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = BASE_DIR / "data/training"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

FEATURE_COLS = [
    'college_rim_fg_pct', 'college_rim_share',
    'college_mid_fg_pct', 'college_mid_share',
    'college_three_fg_pct', 'college_three_share',
    'college_ft_pct',
    'college_on_net_rating',
    'college_assisted_share_rim', 'college_assisted_share_three',
    'final_trueShootingPct', 'final_usage',
    'college_corner_3_rate', 'college_rim_purity',
    'college_avg_shot_dist', 'college_shot_dist_var',
]

TARGET_COL = 'y_peak_ovr'
MADE_NBA_COL = 'made_nba'
AUX_COLS = ['year1_epm_tot', 'gap_ts_legacy']
POSS_COL = 'peak_poss'


def load_training_data() -> pd.DataFrame:
    """Load unified training table."""
    path = TRAINING_DATA_DIR / "unified_training_table.parquet"
    if not path.exists():
        logger.error(f"Training data not found: {path}")
        logger.info("Run: python nba_scripts/build_unified_training_table.py first")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def prepare_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Prepare feature matrix with missingness indicators (no median imputation).
    
    Design: Missingness is signal, not noise to be hidden.
    - Fill NaN with 0 (after z-scoring)
    - Add *_missing binary indicator for each feature with any NaN
    """
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
    
    X = df[available_cols].copy()
    
    missing_indicators = {}
    for col in X.columns:
        if X[col].isna().any():
            missing_indicators[f"{col}_missing"] = X[col].isna().astype(np.float32).values
            n_missing = X[col].isna().sum()
            logger.info(f"  {col}: {n_missing} missing → added {col}_missing indicator")
    
    X_arr = X.values.astype(np.float32)
    
    col_means = np.nanmean(X_arr, axis=0, keepdims=True)
    col_stds = np.nanstd(X_arr, axis=0, keepdims=True) + 1e-8
    X_arr = (X_arr - col_means) / col_stds
    
    X_arr = np.nan_to_num(X_arr, nan=0.0)
    
    final_cols = list(available_cols)
    if missing_indicators:
        missing_arr = np.column_stack(list(missing_indicators.values()))
        X_arr = np.concatenate([X_arr, missing_arr], axis=1)
        final_cols.extend(list(missing_indicators.keys()))
        logger.info(f"Added {len(missing_indicators)} missingness indicators")
    
    logger.info(f"Prepared features: shape={X_arr.shape} ({len(available_cols)} base + {len(missing_indicators)} missing)")
    return X_arr, final_cols


def prepare_targets(df: pd.DataFrame) -> dict:
    """Prepare target arrays."""
    targets = {}
    
    if TARGET_COL in df.columns:
        targets['y_peak_rapm'] = df[TARGET_COL].values.astype(np.float32)
        n_valid = (~np.isnan(targets['y_peak_rapm'])).sum()
        logger.info(f"y_peak_rapm: {n_valid:,} valid values")
    
    if MADE_NBA_COL in df.columns:
        targets['y_made_nba'] = df[MADE_NBA_COL].values.astype(np.float32)
        n_made = (targets['y_made_nba'] == 1).sum()
        logger.info(f"y_made_nba: {n_made:,} made NBA")
    
    if 'year1_mp' in df.columns:
        targets['y_min_threshold'] = (df['year1_mp'] >= 500).astype(np.float32).values
        targets['y_min_threshold'] = np.where(
            df[MADE_NBA_COL].values == 1,
            targets['y_min_threshold'],
            np.nan
        )
        n_threshold = np.nansum(targets['y_min_threshold'])
        logger.info(f"y_min_threshold: {int(n_threshold):,} with ≥500 min")
    
    y_aux = {}
    for col in AUX_COLS:
        if col in df.columns:
            y_aux[col] = df[col].values.astype(np.float32)
            n_valid = (~np.isnan(y_aux[col])).sum()
            logger.info(f"aux {col}: {n_valid:,} valid values")
    
    if y_aux:
        targets['y_aux'] = y_aux
    
    if POSS_COL in df.columns:
        targets['peak_poss'] = df[POSS_COL].values.astype(np.float32)
        targets['peak_poss'] = np.nan_to_num(targets['peak_poss'], nan=1000.0)
    
    return targets


def plot_training_loss(losses: np.ndarray, output_path: Path):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses)
    ax.set_xlabel('Step')
    ax.set_ylabel('ELBO Loss')
    ax.set_title('Pathway Model Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved loss plot to {output_path}")


def generate_report(
    model: HierarchicalPathwayModel,
    X: np.ndarray,
    targets: dict,
    feature_cols: list,
    output_dir: Path,
) -> dict:
    """Generate training report with diagnostics."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'k_z': model.k_z,
            'k_p': model.k_p,
            'ard_scale': model.ard_scale,
            'beta_global_scale': model.beta_global_scale,
            'delta_beta_tau': model.delta_beta_tau,
        },
        'data': {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_cols': feature_cols,
        },
    }
    
    predictions = model.predict(X, num_samples=200)
    
    y_true = targets.get('y_peak_rapm')
    if y_true is not None:
        mask = ~np.isnan(y_true)
        y_pred = predictions['mean']
        
        residuals = y_true[mask] - y_pred[mask]
        rmse = np.sqrt((residuals ** 2).mean())
        mae = np.abs(residuals).mean()
        
        corr = np.corrcoef(y_true[mask], y_pred[mask])[0, 1]
        
        report['metrics'] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(corr),
            'n_evaluated': int(mask.sum()),
        }
        logger.info(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, Corr: {corr:.3f}")
    
    pathway_summary = model.get_pathway_summary(X, num_samples=200)
    report['pathways'] = pathway_summary
    
    logger.info("Pathway usage:")
    for k, usage in enumerate(pathway_summary['pathway_usage']):
        logger.info(f"  Pathway {k}: {usage:.1%}")
    
    samples = model.get_posterior_samples(X[:1], num_samples=200)
    alpha = np.array(samples['alpha'])
    alpha_mean = alpha.mean(axis=0)
    
    threshold = 0.1
    effective_dims = int((alpha_mean > threshold).sum())
    report['ard'] = {
        'effective_dims': effective_dims,
        'total_dims': model.k_z,
        'alpha_mean': alpha_mean.tolist(),
    }
    logger.info(f"Effective dimensions: {effective_dims} / {model.k_z}")
    
    beta_global = np.array(samples['beta_global']).mean(axis=0)
    delta_beta = np.array(samples['delta_beta']).mean(axis=0)
    
    report['coefficients'] = {
        'beta_global_magnitude': float(np.abs(beta_global).mean()),
        'delta_beta_magnitudes': [float(np.abs(delta_beta[k]).mean()) for k in range(model.k_p)],
    }
    
    report_path = output_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report to {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Pathway Model')
    parser.add_argument('--num_steps', type=int, default=5000, help='SVI iterations')
    parser.add_argument('--k_z', type=int, default=24, help='Latent dimensions')
    parser.add_argument('--k_p', type=int, default=6, help='Number of pathways')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("HIERARCHICAL PATHWAY MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Config: k_z={args.k_z}, k_p={args.k_p}, steps={args.num_steps}")
    
    df = load_training_data()
    if df.empty:
        logger.error("No training data available!")
        return
    
    X, feature_cols = prepare_features(df, FEATURE_COLS)
    targets = prepare_targets(df)
    
    if 'y_peak_rapm' not in targets:
        logger.error("No RAPM target available!")
        return
    
    model = HierarchicalPathwayModel(
        k_z=args.k_z,
        k_p=args.k_p,
        ard_scale=1.0,
        beta_global_scale=0.5,
        delta_beta_tau=0.3,
        gating_scale=0.5,
    )
    
    logger.info("\nStarting SVI training...")
    result = model.fit(
        x=X,
        y_peak_rapm=targets['y_peak_rapm'],
        y_made_nba=targets.get('y_made_nba'),
        y_min_threshold=targets.get('y_min_threshold'),
        y_aux=targets.get('y_aux'),
        peak_poss=targets.get('peak_poss'),
        num_steps=args.num_steps,
        lr=args.lr,
        seed=args.seed,
        progress_bar=True,
    )
    
    logger.info(f"\nFinal loss: {result['final_loss']:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODELS_DIR / f"pathway_model_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_loss(result['losses'], output_dir / "training_loss.png")
    
    report = generate_report(model, X, targets, feature_cols, output_dir)
    
    import pickle
    model_path = output_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'params': model.params,
            'config': {
                'k_z': model.k_z,
                'k_p': model.k_p,
                'ard_scale': model.ard_scale,
                'beta_global_scale': model.beta_global_scale,
                'delta_beta_tau': model.delta_beta_tau,
            },
            'feature_cols': feature_cols,
            'losses': result['losses'],
        }, f)
    logger.info(f"Saved model to {model_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Metrics: RMSE={report['metrics']['rmse']:.3f}, Corr={report['metrics']['correlation']:.3f}")
    
    if len(X) > 0:
        logger.info("\nSample decomposition (first player):")
        decomp = model.decompose(X[:1], player_idx=0, num_samples=100)
        logger.info(f"  Predicted peakRAPM: {decomp['total_mean']:.2f} ± {decomp['total_std']:.2f}")
        logger.info(f"  P(made_nba): {decomp['p_made_nba']:.1%}")
        logger.info(f"  Pathway probs: {[f'{p:.1%}' for p in decomp['pathway_probs']]}")


if __name__ == "__main__":
    main()
