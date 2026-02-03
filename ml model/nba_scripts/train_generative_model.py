"""
Train Generative Prospect Model
===============================
NumPyro-based Bayesian model with ARD and horseshoe priors.

Usage:
    python train_generative_model.py --num-steps 10000 --k-max 32

Output:
    models/generative_model_{DATE}/
    ├── params.pkl          # Fitted parameters
    ├── diagnostics.json    # ARD scales, effective dims
    ├── decomposition.json  # Example player decompositions
    └── report.md           # Human-readable report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import pickle
import argparse
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
TRAINING_DATA_DIR = BASE_DIR / "data/training"
MODELS_DIR = BASE_DIR / "models"


def load_training_data():
    """Load unified training table."""
    mock_path = TRAINING_DATA_DIR / "unified_training_table_mock.parquet"
    real_path = TRAINING_DATA_DIR / "unified_training_table.parquet"
    
    if real_path.exists():
        logger.info(f"Loading real training data: {real_path}")
        return pd.read_parquet(real_path)
    elif mock_path.exists():
        logger.info(f"Loading mock training data: {mock_path}")
        return pd.read_parquet(mock_path)
    else:
        raise FileNotFoundError("No training data found.")


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix and targets."""
    # Feature columns (college stats)
    feature_cols = [c for c in df.columns if c.startswith('college_') or c.startswith('final_') or c.startswith('career_')]
    # Exclude non-numeric and identifier columns
    exclude = ['college_final_season', 'athlete_id', 'nba_id', 'player_name']
    feature_cols = [c for c in feature_cols if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    logger.info(f"Using {len(feature_cols)} feature columns")
    
    # Extract features
    X = df[feature_cols].values.astype(np.float32)
    
    # Fill NaN with 0 (will be handled by model)
    X = np.nan_to_num(X, nan=0.0)
    
    # Standardize features
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - X_mean) / X_std
    
    # Targets
    y_rapm = df['y_peak_ovr'].values.astype(np.float32) if 'y_peak_ovr' in df.columns else None
    
    # Exposure
    exposure = df['peak_poss'].values.astype(np.float32) if 'peak_poss' in df.columns else None
    
    # Auxiliary targets
    y_aux = {}
    if 'year1_epm_tot' in df.columns:
        y_aux['epm_tot'] = df['year1_epm_tot'].values.astype(np.float32)
    if 'gap_ts_legacy' in df.columns:
        y_aux['gap_ts'] = df['gap_ts_legacy'].values.astype(np.float32)
    
    return X, y_rapm, exposure, y_aux, feature_cols, (X_mean, X_std)


def main(args):
    logger.info("=" * 60)
    logger.info("GENERATIVE PROSPECT MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load data
    df = load_training_data()
    logger.info(f"Loaded {len(df)} players")
    
    # Prepare features
    X, y_rapm, exposure, y_aux, feature_cols, norm_stats = prepare_features(df)
    logger.info(f"Features: {X.shape}, Targets: {y_rapm.shape if y_rapm is not None else 'None'}")
    
    # Filter to players with valid RAPM
    if y_rapm is not None:
        valid_mask = ~np.isnan(y_rapm)
        n_valid = valid_mask.sum()
        logger.info(f"Players with valid RAPM: {n_valid}")
    
    # Import model
    from models.generative import GenerativeProspectModel
    
    # Create model
    model = GenerativeProspectModel(
        k_max=args.k_max,
        ard_scale=args.ard_scale,
        tau_main=args.tau_main,
        tau_interaction=args.tau_interaction,
        sigma_residual=0.1,
        include_aux_head=len(y_aux) > 0,
    )
    
    logger.info(f"Model config: K_max={args.k_max}, ARD_scale={args.ard_scale}")
    logger.info(f"Tau_main={args.tau_main}, Tau_interaction={args.tau_interaction}")
    
    # Fit
    logger.info(f"Starting SVI with {args.num_steps} steps...")
    result = model.fit(
        x=X,
        y_rapm=y_rapm,
        y_aux=y_aux if len(y_aux) > 0 else None,
        exposure_rapm=exposure,
        num_steps=args.num_steps,
        lr=args.lr,
        seed=args.seed,
        progress_bar=True,
    )
    
    logger.info(f"Final loss: {result['final_loss']:.4f}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODELS_DIR / f"generative_model_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")
    
    # Save parameters
    with open(output_dir / "params.pkl", "wb") as f:
        pickle.dump({
            'params': model.params,
            'k_max': args.k_max,
            'feature_cols': feature_cols,
            'norm_stats': norm_stats,
        }, f)
    
    # Get predictions
    logger.info("Getting predictions...")
    predictions = model.predict(X, num_samples=200, seed=args.seed)
    
    # Diagnostics
    logger.info("Computing diagnostics...")
    samples = model.get_posterior_samples(X[:1], num_samples=500, seed=args.seed)
    
    alpha_samples = np.array(samples['alpha'])
    alpha_mean = alpha_samples.mean(axis=0)
    alpha_std = alpha_samples.std(axis=0)
    
    # Effective dimensions
    active_mask = alpha_mean > 0.1
    k_eff = int(active_mask.sum())
    logger.info(f"Effective dimensions: {k_eff} / {args.k_max}")
    
    # Beta (main effects)
    beta_samples = np.array(samples['beta'])
    beta_mean = beta_samples.mean(axis=0)
    beta_std = beta_samples.std(axis=0)
    
    # Rho (interactions)
    rho_samples = np.array(samples['rho'])
    rho_mean = rho_samples.mean(axis=0)
    rho_std = rho_samples.std(axis=0)
    
    # Count active interactions
    n_pairs = args.k_max * (args.k_max - 1) // 2
    active_interactions = np.abs(rho_mean) > 0.05
    n_active = int(active_interactions.sum())
    logger.info(f"Active interactions: {n_active} / {n_pairs}")
    
    # Save diagnostics
    diagnostics = {
        'k_max': args.k_max,
        'k_effective': k_eff,
        'n_interactions_total': n_pairs,
        'n_interactions_active': n_active,
        'alpha_mean': alpha_mean.tolist(),
        'alpha_std': alpha_std.tolist(),
        'beta_mean': beta_mean.tolist(),
        'beta_std': beta_std.tolist(),
        'final_loss': result['final_loss'],
    }
    
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    
    # Example decompositions
    logger.info("Computing example decompositions...")
    decompositions = []
    
    # Pick a few players
    sample_indices = [0, len(df)//4, len(df)//2, 3*len(df)//4]
    for idx in sample_indices:
        if idx < len(df):
            decomp = model.decompose(X, player_idx=idx, num_samples=200, seed=args.seed)
            
            player_info = {
                'idx': idx,
                'athlete_id': int(df.iloc[idx].get('athlete_id', idx)),
            }
            decomp['player'] = player_info
            decompositions.append(decomp)
    
    with open(output_dir / "decompositions.json", "w") as f:
        json.dump(decompositions, f, indent=2, default=float)
    
    # Generate report
    report = generate_report(diagnostics, decompositions, predictions, y_rapm, args)
    with open(output_dir / "report.md", "w") as f:
        f.write(report)
    
    # Save losses
    np.save(output_dir / "losses.npy", result['losses'])
    
    logger.info(f"\n✅ Training complete! Results saved to {output_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Effective dimensions: {k_eff} / {args.k_max}")
    print(f"Active interactions: {n_active} / {n_pairs}")
    print(f"Final loss: {result['final_loss']:.4f}")
    
    if y_rapm is not None:
        valid = ~np.isnan(y_rapm)
        pred_mean = predictions['mean'][valid]
        actual = y_rapm[valid]
        rmse = np.sqrt(np.mean((pred_mean - actual) ** 2))
        corr = np.corrcoef(pred_mean, actual)[0, 1]
        print(f"RMSE: {rmse:.4f}")
        print(f"Correlation: {corr:.4f}")
    
    print("\nExample decomposition (Player 0):")
    if decompositions:
        d = decompositions[0]
        print(f"  Predicted RAPM: {d['total_mean']:.2f} ± {d['total_std']:.2f}")
        print(f"  Main effects total: {d['main_total']:.2f}")
        print(f"  Interaction total: {d['interaction_total']:.2f}")
        print(f"  Top traits (80% cumulative):")
        for t in d['top_traits'][:5]:
            print(f"    Trait {t['trait_idx']}: z={t['z_value']:.2f} → {t['contribution']:.3f} RAPM")


def generate_report(diagnostics, decompositions, predictions, y_rapm, args):
    """Generate markdown report."""
    report = f"""# Generative Prospect Model Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Model Configuration

| Parameter | Value |
|-----------|-------|
| K_max (latent dims) | {args.k_max} |
| ARD scale | {args.ard_scale} |
| Tau main | {args.tau_main} |
| Tau interaction | {args.tau_interaction} |
| SVI steps | {args.num_steps} |

## Results

| Metric | Value |
|--------|-------|
| Effective dimensions | {diagnostics['k_effective']} / {diagnostics['k_max']} |
| Active interactions | {diagnostics['n_interactions_active']} / {diagnostics['n_interactions_total']} |
| Final ELBO loss | {diagnostics['final_loss']:.4f} |

"""
    
    if y_rapm is not None:
        valid = ~np.isnan(y_rapm)
        pred_mean = predictions['mean'][valid]
        actual = y_rapm[valid]
        rmse = np.sqrt(np.mean((pred_mean - actual) ** 2))
        corr = np.corrcoef(pred_mean, actual)[0, 1]
        
        report += f"""
## Prediction Performance

| Metric | Value |
|--------|-------|
| RMSE | {rmse:.4f} |
| Correlation | {corr:.4f} |

"""
    
    # ARD scales
    report += """
## ARD Scales (Trait Importance)

| Trait | Scale (mean ± std) | Active |
|-------|-------------------|--------|
"""
    for i, (mean, std) in enumerate(zip(diagnostics['alpha_mean'], diagnostics['alpha_std'])):
        active = "✓" if mean > 0.1 else ""
        report += f"| {i} | {mean:.3f} ± {std:.3f} | {active} |\n"
    
    # Decompositions
    if decompositions:
        report += """
## Example Player Decompositions

"""
        for d in decompositions[:3]:
            report += f"""
### Player {d['player']['idx']}

**Predicted RAPM**: {d['total_mean']:.2f} ± {d['total_std']:.2f}

**Contributions**:
- Intercept: {d['intercept']:.3f}
- Main effects: {d['main_total']:.3f}
- Interactions: {d['interaction_total']:.3f}

**Top Traits** (80% cumulative):

| Trait | z-value | Contribution |
|-------|---------|--------------|
"""
            for t in d['top_traits'][:5]:
                report += f"| {t['trait_idx']} | {t['z_value']:.2f} | {t['contribution']:.3f} |\n"
            
            if d['active_interactions']:
                report += """
**Active Interactions**:

| Pair | Contribution |
|------|--------------|
"""
                for inter in d['active_interactions'][:5]:
                    report += f"| {inter['pair']} | {inter['contribution']:.3f} |\n"
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generative prospect model")
    parser.add_argument('--num-steps', type=int, default=5000, help='SVI iterations')
    parser.add_argument('--k-max', type=int, default=32, help='Max latent dimensions')
    parser.add_argument('--ard-scale', type=float, default=1.0, help='ARD prior scale')
    parser.add_argument('--tau-main', type=float, default=1.0, help='Main effect shrinkage')
    parser.add_argument('--tau-interaction', type=float, default=0.3, help='Interaction shrinkage')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)
