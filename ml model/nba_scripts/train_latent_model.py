"""
Latent Space Model Training
===========================
Trains the ProspectModel with multi-task learning and archetype discovery.

Features:
- Walk-forward temporal validation
- Multi-task loss (RAPM, gaps, survival)
- Tier 2 feature masking (dropout during training)
- Archetype clustering regularization
- Early stopping and checkpointing

Usage:
    python train_latent_model.py --epochs 100 --latent-dim 32 --n-archetypes 8

Output:
    models/latent_model_{DATE}/
    ├── model.pt                 # Model weights
    ├── archetype_profiles.json  # Discovered archetypes
    ├── embeddings.npy           # Player embeddings
    ├── eval_metrics.json        # Training metrics
    └── analysis_report.md       # Archetype analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import argparse
import logging
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    ProspectModel,
    ProspectLoss,
    ArchetypeAnalyzer,
    TIER1_COLUMNS,
    TIER2_COLUMNS,
    CAREER_BASE_COLUMNS,
    WITHIN_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
TRAINING_DATA_DIR = BASE_DIR / "data/training"
MODELS_DIR = BASE_DIR / "models"

# Walk-forward splits
TRAIN_SEASONS = list(range(2010, 2018))
VAL_SEASONS = list(range(2018, 2020))
TEST_SEASONS = list(range(2020, 2023))


class ProspectDataset(Dataset):
    """PyTorch dataset for prospect training data."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tier1_cols: List[str],
        tier2_cols: List[str],
        career_cols: List[str],
        within_cols: List[str],
        tier2_dropout: float = 0.3,
        training: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.tier1_cols = [c for c in tier1_cols if c in df.columns]
        self.tier2_cols = [c for c in tier2_cols if c in df.columns]
        self.career_cols = [c for c in career_cols if c in df.columns]
        self.within_cols = [c for c in within_cols if c in df.columns]
        self.tier2_dropout = tier2_dropout
        self.training = training
        
        # Pre-extract features
        self.tier1 = self._extract_features(tier1_cols)
        self.tier2 = self._extract_features(tier2_cols)
        self.career = self._extract_features(career_cols)
        self.within = self._extract_features(within_cols)
        
        # Tier 2 mask
        if 'has_spatial_data' in df.columns:
            self.tier2_mask = df['has_spatial_data'].values.astype(np.float32)
        else:
            self.tier2_mask = np.ones(len(df), dtype=np.float32)

        # Within-season availability mask (do not infer missingness from zeros)
        ws_flags = []
        for c in ['final_has_ws_last5', 'final_has_ws_last10', 'final_has_ws_breakout_timing_eff']:
            if c in df.columns:
                ws_flags.append(df[c].fillna(0).values.astype(np.float32))
        if ws_flags:
            self.within_mask = (np.max(np.stack(ws_flags, axis=0), axis=0) > 0).astype(np.float32)
        else:
            self.within_mask = np.zeros(len(df), dtype=np.float32)
        
        # Targets
        self.targets = self._extract_targets()
        
        # Metadata
        self.player_ids = df['athlete_id'].values if 'athlete_id' in df.columns else np.arange(len(df))
        self.player_names = df['player_name'].values if 'player_name' in df.columns else [f"P{i}" for i in range(len(df))]
    
    def _extract_features(self, cols: List[str]) -> np.ndarray:
        """Extract and fill missing features."""
        available = [c for c in cols if c in self.df.columns]
        if not available:
            return np.zeros((len(self.df), len(cols)), dtype=np.float32)
        
        data = self.df[available].values.astype(np.float32)
        # Fill NaN with 0 (will be masked for Tier 2)
        data = np.nan_to_num(data, nan=0.0)
        return data
    
    def _extract_targets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Extract targets with masks for missing values."""
        targets = {}
        
        # RAPM (primary)
        if 'y_peak_ovr' in self.df.columns:
            vals = self.df['y_peak_ovr'].values.astype(np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, nan=0.0)
            targets['rapm'] = (vals, mask)
        
        # Gap TS (auxiliary)
        if 'gap_ts_legacy' in self.df.columns:
            vals = self.df['gap_ts_legacy'].values.astype(np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, nan=0.0)
            targets['gap'] = (vals, mask)
        
        # Year 1 EPM (auxiliary)
        if 'year1_epm_tot' in self.df.columns:
            vals = self.df['year1_epm_tot'].values.astype(np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, nan=0.0)
            targets['epm'] = (vals, mask)
        
        # Survival (binary)
        if 'made_nba' in self.df.columns:
            vals = self.df['made_nba'].values.astype(np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, nan=0.0)
            targets['survival'] = (vals, mask)
        
        return targets
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tier2_mask = self.tier2_mask[idx]
        within_mask = self.within_mask[idx]
        
        # Apply Tier 2 dropout during training
        if self.training and tier2_mask > 0 and np.random.random() < self.tier2_dropout:
            tier2_mask = 0.0
        
        item = {
            'tier1': torch.from_numpy(self.tier1[idx]),
            'tier2': torch.from_numpy(self.tier2[idx]),
            'career': torch.from_numpy(self.career[idx]),
            'within': torch.from_numpy(self.within[idx]),
            'tier2_mask': torch.tensor([tier2_mask]),
            'within_mask': torch.tensor([within_mask]),
        }
        
        # Add targets
        for name, (vals, mask) in self.targets.items():
            item[f'{name}_target'] = torch.tensor(vals[idx])
            item[f'{name}_mask'] = torch.tensor(mask[idx])
        
        return item


def load_training_data() -> pd.DataFrame:
    """Load unified training table."""
    # Try mock data first (for testing)
    mock_path = TRAINING_DATA_DIR / "unified_training_table_mock.parquet"
    real_path = TRAINING_DATA_DIR / "unified_training_table.parquet"
    
    if mock_path.exists():
        logger.info(f"Loading mock training data: {mock_path}")
        return pd.read_parquet(mock_path)
    elif real_path.exists():
        logger.info(f"Loading training data: {real_path}")
        return pd.read_parquet(real_path)
    else:
        raise FileNotFoundError(
            f"No training data found. Run build_unified_training_table.py first."
        )


def create_temporal_splits(
    df: pd.DataFrame,
    season_col: str = 'college_final_season',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create walk-forward splits."""
    if season_col not in df.columns:
        logger.warning(f"Season column not found. Using random split.")
        n = len(df)
        idx = np.random.permutation(n)
        train = df.iloc[idx[:int(0.6*n)]]
        val = df.iloc[idx[int(0.6*n):int(0.8*n)]]
        test = df.iloc[idx[int(0.8*n):]]
        return train, val, test
    
    train = df[df[season_col].isin(TRAIN_SEASONS)]
    val = df[df[season_col].isin(VAL_SEASONS)]
    test = df[df[season_col].isin(TEST_SEASONS)]
    
    logger.info(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch])
    return result


def train_epoch(
    model: ProspectModel,
    dataloader: DataLoader,
    criterion: ProspectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {}
    n_batches = 0
    
    for batch in dataloader:
        # Move to device
        tier1 = batch['tier1'].to(device)
        tier2 = batch['tier2'].to(device)
        career = batch['career'].to(device)
        within = batch['within'].to(device)
        tier2_mask = batch['tier2_mask'].to(device)
        within_mask = batch['within_mask'].to(device)
        
        # Forward
        outputs = model(tier1, tier2, career, within, tier2_mask, within_mask)
        
        # Build targets dict
        targets = {}
        for name in ['rapm', 'gap', 'epm', 'survival']:
            if f'{name}_target' in batch:
                targets[name] = (
                    batch[f'{name}_target'].to(device),
                    batch[f'{name}_mask'].to(device).bool(),
                )
        
        # Compute loss
        losses = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1
    
    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    model: ProspectModel,
    dataloader: DataLoader,
    criterion: ProspectLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_losses = {}
    all_preds = []
    all_targets = []
    n_batches = 0
    
    for batch in dataloader:
        tier1 = batch['tier1'].to(device)
        tier2 = batch['tier2'].to(device)
        career = batch['career'].to(device)
        tier2_mask = batch['tier2_mask'].to(device)
        within = batch['within'].to(device)
        within_mask = batch['within_mask'].to(device)
        
        outputs = model(tier1, tier2, career, within, tier2_mask, within_mask)
        
        targets = {}
        for name in ['rapm', 'gap', 'epm', 'survival']:
            if f'{name}_target' in batch:
                targets[name] = (
                    batch[f'{name}_target'].to(device),
                    batch[f'{name}_mask'].to(device).bool(),
                )
        
        losses = criterion(outputs, targets)
        
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1
        
        # Collect predictions for metrics
        if 'rapm' in targets:
            mask = targets['rapm'][1]
            if mask.any():
                all_preds.extend(outputs['rapm_pred'][:, 0][mask].cpu().numpy())
                all_targets.extend(targets['rapm'][0][mask].cpu().numpy())
    
    metrics = {k: v / n_batches for k, v in total_losses.items()}
    
    # Compute additional metrics
    if all_preds:
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        metrics['rapm_rmse'] = np.sqrt(np.mean((preds - targets) ** 2))
        metrics['rapm_mae'] = np.mean(np.abs(preds - targets))
        if len(preds) > 1:
            metrics['rapm_corr'] = np.corrcoef(preds, targets)[0, 1]
    
    return metrics


def train_model(
    model: ProspectModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: ProspectLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    n_epochs: int,
    patience: int = 20,
    output_dir: Path = None,
) -> Dict:
    """Full training loop with early stopping."""
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(n_epochs):
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_losses = evaluate(model, val_loader, criterion, device)
        
        # LR schedule
        scheduler.step(val_losses['total'])
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"Val RMSE: {val_losses.get('rapm_rmse', 0):.4f}"
        )
        
        history['train'].append(train_losses)
        history['val'].append(val_losses)
        
        # Early stopping
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch
            if output_dir:
                torch.save(model.state_dict(), output_dir / "model_best.pt")
        elif epoch - best_epoch >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if output_dir and (output_dir / "model_best.pt").exists():
        model.load_state_dict(torch.load(output_dir / "model_best.pt"))
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'history': history,
    }


def main(args):
    logger.info("=" * 60)
    logger.info("LATENT SPACE MODEL TRAINING")
    logger.info("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load data
    df = load_training_data()
    train_df, val_df, test_df = create_temporal_splits(df)
    
    # Feature columns (use what's available)
    tier1_cols = [c for c in TIER1_COLUMNS if c in df.columns]
    tier2_cols = [c for c in TIER2_COLUMNS if c in df.columns]
    career_cols = [c for c in CAREER_BASE_COLUMNS if c in df.columns]
    within_cols = [c for c in WITHIN_COLUMNS if c in df.columns]
    
    logger.info(
        f"Features: Tier1={len(tier1_cols)}, Tier2={len(tier2_cols)}, "
        f"Career={len(career_cols)}, Within={len(within_cols)}"
    )
    
    # Datasets
    train_dataset = ProspectDataset(train_df, tier1_cols, tier2_cols, career_cols, within_cols, training=True)
    val_dataset = ProspectDataset(val_df, tier1_cols, tier2_cols, career_cols, within_cols, training=False)
    test_dataset = ProspectDataset(test_df, tier1_cols, tier2_cols, career_cols, within_cols, training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    
    # Model
    model = ProspectModel(
        latent_dim=args.latent_dim,
        n_archetypes=args.n_archetypes,
        use_vae=args.use_vae,
        predict_uncertainty=True,
        condition_on_archetypes=getattr(args, 'condition_on_archetypes', False),
    )
    
    # Reinitialize encoder with the exact dimensions actually used by this run.
    # (This allows dropping columns if a table build is missing optional inputs.)
    from models.player_encoder import PlayerEncoder
    model.encoder = PlayerEncoder(
        tier1_dim=len(tier1_cols),
        tier2_dim=len(tier2_cols),
        career_dim=len(career_cols),
        within_dim=len(within_cols),
        latent_dim=args.latent_dim,
        use_vae=args.use_vae,
    )
    
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    criterion = ProspectLoss(
        lambda_rapm=1.0,
        lambda_gap=0.5,
        lambda_epm=0.5,
        lambda_surv=0.3,
        lambda_arch=0.1,
        lambda_kl=0.01 if args.use_vae else 0.0,
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODELS_DIR / f"latent_model_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")
    
    # Train
    results = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args.epochs, args.patience, output_dir
    )
    
    # Test evaluation
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test RMSE: {test_metrics.get('rapm_rmse', 0):.4f}")
    results['test'] = test_metrics
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "model.pt")
    
    # Archetype analysis
    logger.info("Running archetype analysis...")
    analyzer = ArchetypeAnalyzer(model, tier1_cols, tier2_cols, career_cols, within_cols, args.n_archetypes)
    
    # Fit analyzer on training data
    train_tier1 = torch.from_numpy(train_dataset.tier1).to(device)
    train_tier2 = torch.from_numpy(train_dataset.tier2).to(device)
    train_career = torch.from_numpy(train_dataset.career).to(device)
    train_within = torch.from_numpy(train_dataset.within).to(device)
    train_mask = torch.from_numpy(train_dataset.tier2_mask).unsqueeze(1).to(device)
    train_within_mask = torch.from_numpy(train_dataset.within_mask).unsqueeze(1).to(device)
    
    targets_for_analysis = {}
    if 'rapm' in train_dataset.targets:
        targets_for_analysis['rapm'] = train_dataset.targets['rapm'][0]
    if 'survival' in train_dataset.targets:
        targets_for_analysis['survival'] = train_dataset.targets['survival'][0]
    
    analyzer.fit(
        train_tier1, train_tier2, train_career, train_mask, train_within, train_within_mask,
        player_names=list(train_dataset.player_names),
        targets=targets_for_analysis,
    )
    
    # Save archetype profiles
    archetype_summary = analyzer.get_archetype_summary()
    archetype_summary.to_csv(output_dir / "archetype_profiles.csv", index=False)
    logger.info(f"\nArchetype Summary:\n{archetype_summary.to_string()}")
    
    # Save embeddings
    np.save(output_dir / "embeddings.npy", analyzer.embeddings)
    
    # Save metrics
    with open(output_dir / "eval_metrics.json", 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(results, f, indent=2, default=convert)
    
    # Generate analysis report
    report = generate_analysis_report(analyzer, results, output_dir)
    with open(output_dir / "analysis_report.md", 'w') as f:
        f.write(report)
    
    logger.info(f"\n✅ Training complete! Results saved to {output_dir}")


def generate_analysis_report(analyzer: ArchetypeAnalyzer, results: Dict, output_dir: Path) -> str:
    """Generate markdown analysis report."""
    summary = analyzer.get_archetype_summary()
    
    report = f"""# Latent Space Model Analysis Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Model**: {output_dir.name}

## Training Results

- **Best Epoch**: {results['best_epoch']}
- **Best Val Loss**: {results['best_val_loss']:.4f}
- **Test RMSE**: {results['test'].get('rapm_rmse', 'N/A')}
- **Test Correlation**: {results['test'].get('rapm_corr', 'N/A')}

## Discovered Archetypes

"""
    
    for _, row in summary.iterrows():
        report += f"""### {row['name']} (Archetype {row['archetype_id']})

- **Players**: {row['n_players']}
- **Avg RAPM**: {row['avg_rapm']:.2f}
- **Survival Rate**: {row['survival_rate']:.1%}
- **Description**: {row['description']}
- **Examples**: {row['examples']}

"""
    
    report += """## Usage

```python
from models import ProspectModel, ArchetypeAnalyzer

# Load model
model = ProspectModel(latent_dim=32, n_archetypes=8)
model.load_state_dict(torch.load('model.pt'))

# Analyze a prospect
analysis = analyzer.analyze_player(tier1, tier2, career, tier2_mask)
print(analysis.narrative)
```
"""
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train latent space model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--n-archetypes', type=int, default=8, help='Number of archetypes')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--use-vae', action='store_true', help='Use VAE-style encoder')
    parser.add_argument('--condition-on-archetypes', action='store_true', help='Condition decoders on archetype probabilities')
    
    args = parser.parse_args()
    main(args)
