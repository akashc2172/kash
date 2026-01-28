"""
NBA Training Pipeline
=====================
End-to-end training script for the NBA Prospect Model.

Pipeline per spec:
1. Load and join warehouse data
2. Apply feature transforms (z-score, logit, weights)
3. Split train/test (temporal by draft class)
4. Train model with multi-task loss
5. Evaluate on held-out data

Note: This script is a template. Do NOT run until data and config are verified.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Dict, Any

# Local imports (assuming scripts are in nba_scripts/)
from nba_data_loader import build_modeling_dataset, get_feature_columns, prepare_train_test_split
from nba_feature_transforms import apply_all_transforms, filter_reliable_samples
from nba_model_architecture import NBAProspectModel, compute_kl_divergence
from nba_loss_functions import MultiTaskLoss, compute_primary_variance, compute_year1_variance
from nba_evaluation import evaluate_model

# Configuration
CONFIG = {
    # Data
    'warehouse_dir': Path('data/warehouse_v2'),
    'test_seasons': [2022, 2023],  # Hold out recent drafts for testing
    
    # Model
    'latent_dim': 32,
    'hidden_dims': (64, 64),
    'dropout': 0.1,
    
    # Training
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 100,
    'early_stopping_patience': 10,
    
    # Loss weights
    'primary_weight': 1.0,
    'aux_target_weight': 0.3,
    'aux_obs_weight': 0.1,
    'kl_weight': 0.01,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def prepare_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load, transform, and split data.
    
    Returns dict with train/test DataFrames and feature dimensions.
    """
    print("Loading data...")
    df = build_modeling_dataset(config['warehouse_dir'])
    
    print("Applying transforms...")
    df = apply_all_transforms(df)
    
    print("Filtering reliable samples...")
    df = filter_reliable_samples(df)
    
    print("Splitting train/test...")
    train_df, test_df = prepare_train_test_split(
        df, 
        test_seasons=config['test_seasons']
    )
    
    # Get column groupings
    cols = get_feature_columns()
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'feature_cols': cols,
    }


def pick_transformed(df: pd.DataFrame, col: str) -> str:
    """
    Prefer transformed versions of columns (_logit or _z) if present.
    
    This ensures we actually use the output of apply_all_transforms().
    """
    if f"{col}_logit" in df.columns:
        return f"{col}_logit"
    if f"{col}_z" in df.columns:
        return f"{col}_z"
    return col


def df_to_tensors(
    df: pd.DataFrame,
    feature_cols: dict,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Convert DataFrame to model-ready tensors.
    
    Key fixes from critique:
    1. Uses transformed columns (*_logit, *_z) when available
    2. Creates aux_obs_mask for proper missingness handling
    3. Includes missing indicators in feature set
    """
    # Input features (bio only for NBA-side training)
    # Use raw bio cols (no transforms applied to bio)
    bio_cols = [c for c in feature_cols['bio_features'] if c in df.columns]
    X = torch.tensor(df[bio_cols].fillna(0).values, dtype=torch.float32).to(device)
    
    # Primary target (no transform, these are already standardized in warehouse)
    primary_cols = [c for c in feature_cols['primary_target'] if c in df.columns]
    y_primary = torch.tensor(df[primary_cols].fillna(0).values, dtype=torch.float32).to(device)
    primary_mask = torch.tensor(df['has_peak'].values, dtype=torch.bool).to(device)
    
    # Aux targets (year1 EPM - use z-scored versions if available)
    aux_target_base = [c for c in feature_cols['aux_targets'] if c in df.columns]
    aux_target_cols = [pick_transformed(df, c) for c in aux_target_base]
    y_aux_target = torch.tensor(df[aux_target_cols].fillna(0).values, dtype=torch.float32).to(device)
    aux_target_mask = torch.tensor(df['has_year1'].values, dtype=torch.bool).to(device)
    
    # Aux observations (use transformed versions: _logit for pcts, _z for rates)
    aux_obs_base = [c for c in feature_cols['aux_observations'] if c in df.columns]
    aux_obs_cols = [pick_transformed(df, c) for c in aux_obs_base]
    
    # Build aux_obs_mask: True only where ALL aux obs are non-missing
    # Per spec: "do not impute with zeros" - mask them out of loss
    aux_obs_raw = df[aux_obs_base]  # Check missingness on raw cols, not transformed
    aux_obs_mask = torch.tensor(
        aux_obs_raw.notna().all(axis=1).values,
        dtype=torch.bool
    ).to(device)
    
    y_aux_obs = torch.tensor(df[aux_obs_cols].fillna(0).values, dtype=torch.float32).to(device)
    
    # Exposure weights (for variance computation)
    year1_mp = torch.tensor(df['year1_mp'].fillna(0).values, dtype=torch.float32).to(device)
    peak_poss = torch.tensor(df['peak_poss'].fillna(0).values, dtype=torch.float32).to(device)
    
    return {
        'X': X,
        'y_primary': y_primary,
        'primary_mask': primary_mask,
        'y_aux_target': y_aux_target,
        'aux_target_mask': aux_target_mask,
        'y_aux_obs': y_aux_obs,
        'aux_obs_mask': aux_obs_mask,  # NEW: mask for aux observations
        'year1_mp': year1_mp,
        'peak_poss': peak_poss,
        'n_bio_features': len(bio_cols),
        'n_aux_obs': len(aux_obs_cols),
        'aux_obs_cols': aux_obs_cols,  # For debugging
    }



def create_train_loader(data: Dict[str, torch.Tensor], batch_size: int) -> DataLoader:
    """
    Create a DataLoader for minibatch training.
    
    Per critique: "switch to minibatching" for stable gradients and meaningful epochs.
    """
    dataset = TensorDataset(
        data['X'],
        data['y_primary'],
        data['primary_mask'],
        data['y_aux_target'],
        data['aux_target_mask'],
        data['y_aux_obs'],
        data['aux_obs_mask'],
        data['year1_mp'],
        data['peak_poss'],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Train for one epoch with proper minibatching.
    
    Fixed from critique: now iterates over DataLoader batches instead of
    single full-batch forward pass.
    """
    model.train()
    
    epoch_losses = {'loss_primary': 0., 'loss_total': 0., 'kl': 0., 'total_with_kl': 0.}
    n_batches = 0
    
    for batch in train_loader:
        X, y_primary, primary_mask, y_aux_target, aux_target_mask, \
            y_aux_obs, aux_obs_mask, year1_mp, peak_poss = batch
        
        # Forward pass
        outputs = model(X)
        
        # Compute variances using correct exposure signals
        primary_variance = compute_primary_variance(peak_poss).unsqueeze(-1)
        
        # Multi-task loss
        loss, loss_components = loss_fn(
            pred_primary=outputs['pred_primary'],
            target_primary=y_primary,
            primary_variance=primary_variance,
            primary_mask=primary_mask,
            pred_aux_target=outputs['pred_aux_target'],
            target_aux_target=y_aux_target,
            aux_target_weights=None,
            aux_target_mask=aux_target_mask,
            pred_aux_obs=outputs['pred_aux_obs'],
            target_aux_obs=y_aux_obs,
            aux_obs_weights=None,
            aux_obs_mask=aux_obs_mask,
        )
        
        # KL divergence regularization
        kl_loss = compute_kl_divergence(outputs['mu'], outputs['logvar'])
        total_loss = loss + config['kl_weight'] * kl_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate for epoch averaging
        for key in epoch_losses:
            if key in loss_components:
                epoch_losses[key] += loss_components[key]
        epoch_losses['kl'] += kl_loss.item()
        epoch_losses['total_with_kl'] += total_loss.item()
        n_batches += 1
    
    # Average over batches
    for key in epoch_losses:
        epoch_losses[key] /= max(n_batches, 1)
    
    return epoch_losses


def evaluate_test(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    draft_class: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate on test set with both global and per-draft-class metrics.
    
    Per critique: "Top-K / Spearman should be computed within each draft class"
    """
    from nba_evaluation import evaluate_model, evaluate_model_by_draft_class
    
    model.eval()
    
    with torch.no_grad():
        # Get predictions with uncertainty
        mean, std = model.predict_peak(data['X'], num_samples=100)
    
    # Convert to numpy
    y_true = data['y_primary'][:, 0].cpu().numpy()  # Use overall RAPM
    y_pred_mean = mean[:, 0].cpu().numpy()
    y_pred_std = std[:, 0].cpu().numpy()
    mask = data['primary_mask'].cpu().numpy()
    
    # Filter to samples with targets
    y_true_masked = y_true[mask]
    y_pred_mean_masked = y_pred_mean[mask]
    y_pred_std_masked = y_pred_std[mask]
    
    # Weights from exposure
    weights = data['peak_poss'][mask].cpu().numpy()
    weights = weights / (weights.sum() + 1e-6)
    
    # Global metrics
    results = evaluate_model(y_true_masked, y_pred_mean_masked, y_pred_std_masked, weights)
    
    # Per-draft-class metrics (if draft_class provided)
    if draft_class is not None:
        draft_class_masked = draft_class[mask]
        class_results = evaluate_model_by_draft_class(
            y_true_masked, y_pred_mean_masked, draft_class_masked,
            y_pred_std_masked, weights
        )
        # Merge with prefix
        for k, v in class_results.items():
            results[f'by_class_{k}'] = v
    
    return results


def main():
    """Main training loop."""
    print("=" * 60)
    print("NBA PROSPECT MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    
    # Prepare data
    data = prepare_data(CONFIG)
    
    # Convert to tensors
    train_tensors = df_to_tensors(data['train_df'], data['feature_cols'], CONFIG['device'])
    test_tensors = df_to_tensors(data['test_df'], data['feature_cols'], CONFIG['device'])
    
    print(f"Train samples: {len(data['train_df'])}")
    print(f"Test samples: {len(data['test_df'])}")
    print(f"Input features: {train_tensors['n_bio_features']}")
    print(f"Aux observations: {train_tensors['n_aux_obs']}")
    
    # Initialize model
    model = NBAProspectModel(
        input_dim=train_tensors['n_bio_features'],
        num_aux_obs=train_tensors['n_aux_obs'],
        latent_dim=CONFIG['latent_dim'],
        hidden_dims=CONFIG['hidden_dims'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = MultiTaskLoss(
        primary_weight=CONFIG['primary_weight'],
        aux_target_weight=CONFIG['aux_target_weight'],
        aux_obs_weight=CONFIG['aux_obs_weight']
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Training loop
    best_test_rmse = float('inf')
    patience_counter = 0
    
    # Create DataLoader for minibatching
    train_loader = create_train_loader(train_tensors, CONFIG['batch_size'])
    print(f"Batches per epoch: {len(train_loader)}")
    
    print("\nTraining...")
    for epoch in range(CONFIG['num_epochs']):
        train_losses = train_epoch(model, train_loader, loss_fn, optimizer, CONFIG)
        
        if (epoch + 1) % 10 == 0:
            test_metrics = evaluate_test(model, test_tensors, CONFIG)
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_losses['loss_total']:.4f} | "
                  f"Test RMSE: {test_metrics['rmse']:.4f} | Spearman: {test_metrics['spearman']:.3f}")
            
            # Early stopping
            if test_metrics['rmse'] < best_test_rmse:
                best_test_rmse = test_metrics['rmse']
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['early_stopping_patience'] // 10:
                    print("Early stopping triggered.")
                    break
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    final_metrics = evaluate_test(model, test_tensors, CONFIG)
    
    print("\n" + "=" * 60)
    print("FINAL TEST METRICS")
    print("=" * 60)
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nTraining complete. Model saved to best_model.pt")


if __name__ == '__main__':
    main()
