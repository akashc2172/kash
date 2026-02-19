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
YEAR1_INTERACTION_COLUMNS = ['year1_epm_tot', 'year1_epm_off', 'year1_epm_def', 'year1_usg', 'year1_tspct']


class ProspectDataset(Dataset):
    """PyTorch dataset for prospect training data."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tier1_cols: List[str],
        tier2_cols: List[str],
        career_cols: List[str],
        within_cols: List[str],
        year1_cols: List[str],
        tier2_dropout: float = 0.3,
        training: bool = True,
        temporal_decay_half_life: float = 4.0,
        temporal_decay_min: float = 0.2,
    ):
        self.df = df.reset_index(drop=True)
        self.tier1_cols = [c for c in tier1_cols if c in df.columns]
        self.tier2_cols = [c for c in tier2_cols if c in df.columns]
        self.career_cols = [c for c in career_cols if c in df.columns]
        self.within_cols = [c for c in within_cols if c in df.columns]
        self.year1_cols = [c for c in year1_cols if c in df.columns]
        self.tier2_dropout = tier2_dropout
        self.training = training
        self.temporal_decay_half_life = temporal_decay_half_life
        self.temporal_decay_min = temporal_decay_min
        
        # Pre-extract features
        self.tier1 = self._extract_features(tier1_cols)
        self.tier2 = self._extract_features(tier2_cols)
        self.career = self._extract_features(career_cols)
        self.within = self._extract_features(within_cols)
        self.year1, self.year1_mask = self._extract_features_with_mask(year1_cols)
        self.temporal_weight = self._compute_temporal_weights()
        self.adaptive_weight = np.ones(len(self.df), dtype=np.float32)
        self.sample_weight = self.temporal_weight.copy()
        
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
        self.seasons = df['college_final_season'].values if 'college_final_season' in df.columns else np.full(len(df), np.nan)
    
    def _extract_features(self, cols: List[str]) -> np.ndarray:
        """Extract and fill missing features."""
        available = [c for c in cols if c in self.df.columns]
        if not available:
            return np.zeros((len(self.df), len(cols)), dtype=np.float32)
        
        data = self.df[available].values.astype(np.float32)
        # Fill NaN with 0 (will be masked for Tier 2)
        data = np.nan_to_num(data, nan=0.0)
        return data

    def _extract_features_with_mask(self, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features with per-column presence mask for interaction gating."""
        available = [c for c in cols if c in self.df.columns]
        if not available:
            zeros = np.zeros((len(self.df), len(cols)), dtype=np.float32)
            return zeros, zeros

        raw = self.df[available].values.astype(np.float32)
        mask = (~np.isnan(raw)).astype(np.float32)
        data = np.nan_to_num(raw, nan=0.0)

        # Keep fixed output width by padding missing columns from the requested set.
        if len(available) != len(cols):
            col_to_idx = {c: i for i, c in enumerate(available)}
            padded_data = np.zeros((len(self.df), len(cols)), dtype=np.float32)
            padded_mask = np.zeros((len(self.df), len(cols)), dtype=np.float32)
            for j, c in enumerate(cols):
                if c in col_to_idx:
                    idx = col_to_idx[c]
                    padded_data[:, j] = data[:, idx]
                    padded_mask[:, j] = mask[:, idx]
            data, mask = padded_data, padded_mask

        return data, mask
    
    def _extract_targets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Extract targets with masks for missing values."""
        targets = {}
        
        # RAPM (primary)
        if 'y_peak_ovr' in self.df.columns:
            ovr = self.df['y_peak_ovr'].values.astype(np.float32)
            off = self.df['y_peak_off'].values.astype(np.float32) if 'y_peak_off' in self.df.columns else np.full_like(ovr, np.nan)
            deff = self.df['y_peak_def'].values.astype(np.float32) if 'y_peak_def' in self.df.columns else np.full_like(ovr, np.nan)
            vals = np.stack([
                np.nan_to_num(ovr, nan=0.0),
                np.nan_to_num(off, nan=0.0),
                np.nan_to_num(deff, nan=0.0),
            ], axis=1)
            mask = np.stack([
                ~np.isnan(ovr),
                ~np.isnan(off),
                ~np.isnan(deff),
            ], axis=1)
            # Tail-aware supervised weighting: avoid underfitting high-impact outcomes.
            base_w = np.ones(len(self.df), dtype=np.float32)
            valid_ovr = np.isfinite(ovr)
            if valid_ovr.sum() >= 50:
                mu = float(np.nanmean(ovr[valid_ovr]))
                sd = float(np.nanstd(ovr[valid_ovr]))
                sd = max(sd, 1e-6)
                z = (ovr - mu) / sd
                tail = np.clip(z, 0.0, 3.0)
                base_w = np.where(valid_ovr, 1.0 + 0.15 * tail, 1.0).astype(np.float32)
                base_w = np.clip(base_w, 1.0, 2.5).astype(np.float32)
            targets['rapm'] = (vals, mask, base_w)
        
        # Gap TS (auxiliary)
        if 'gap_ts_legacy' in self.df.columns:
            vals = self.df['gap_ts_legacy'].values.astype(np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, nan=0.0)
            base_w = np.ones(len(self.df), dtype=np.float32)
            targets['gap'] = (vals, mask, base_w)
        
        # Year 1 EPM (auxiliary)
        if 'year1_epm_tot' in self.df.columns:
            vals = self.df['year1_epm_tot'].values.astype(np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, nan=0.0)
            base_w = np.ones(len(self.df), dtype=np.float32)
            targets['epm'] = (vals, mask, base_w)

        # Development-rate auxiliary (quality-weighted)
        if 'dev_rate_y1_y3_mean' in self.df.columns:
            vals = self.df['dev_rate_y1_y3_mean'].values.astype(np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, nan=0.0)
            if 'dev_rate_quality_weight' in self.df.columns:
                w = self.df['dev_rate_quality_weight'].values.astype(np.float32)
                w = np.nan_to_num(w, nan=0.0)
            else:
                w = np.ones(len(self.df), dtype=np.float32)
            targets['dev'] = (vals, mask, w.astype(np.float32))
        
        # Survival (binary)
        if 'made_nba' in self.df.columns:
            vals = self.df['made_nba'].values.astype(np.float32)
            mask = ~np.isnan(vals)
            vals = np.nan_to_num(vals, nan=0.0)
            base_w = np.ones(len(self.df), dtype=np.float32)
            targets['survival'] = (vals, mask, base_w)
        
        return targets

    def _compute_temporal_weights(self) -> np.ndarray:
        """Compute recency-based sample weights from college final season."""
        if 'college_final_season' not in self.df.columns:
            return np.ones(len(self.df), dtype=np.float32)
        seasons = pd.to_numeric(self.df['college_final_season'], errors='coerce').values.astype(float)
        if np.all(~np.isfinite(seasons)):
            return np.ones(len(self.df), dtype=np.float32)
        anchor = np.nanmax(seasons[np.isfinite(seasons)])
        delta = np.maximum(0.0, anchor - seasons)
        hl = max(float(self.temporal_decay_half_life), 1e-6)
        w = np.exp(-np.log(2.0) * delta / hl)
        w = np.where(np.isfinite(w), w, 1.0)
        w = np.clip(w, self.temporal_decay_min, 1.0)
        return w.astype(np.float32)

    def set_adaptive_weights(self, adaptive_multiplier: np.ndarray) -> None:
        """Update per-sample adaptive weights (multiplied on top of temporal weights)."""
        m = np.asarray(adaptive_multiplier, dtype=np.float32)
        if len(m) != len(self.df):
            raise ValueError(f"adaptive_multiplier length mismatch: {len(m)} != {len(self.df)}")
        m = np.nan_to_num(m, nan=1.0, posinf=1.0, neginf=1.0)
        m = np.clip(m, 0.5, 2.0)
        self.adaptive_weight = m
        self.sample_weight = (self.temporal_weight * self.adaptive_weight).astype(np.float32)
    
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
            'year1': torch.from_numpy(self.year1[idx]),
            'year1_mask': torch.from_numpy(self.year1_mask[idx]),
            'tier2_mask': torch.tensor([tier2_mask]),
            'within_mask': torch.tensor([within_mask]),
            'sample_weight': torch.tensor(self.sample_weight[idx]),
            'college_final_season': torch.tensor(
                np.nan_to_num(self.seasons[idx], nan=-1.0), dtype=torch.float32
            ),
        }
        
        # Add targets
        for name, target_tuple in self.targets.items():
            if len(target_tuple) == 3:
                vals, mask, base_weights = target_tuple
                item[f'{name}_target'] = torch.tensor(vals[idx])
                item[f'{name}_mask'] = torch.tensor(mask[idx])
                eff_w = base_weights[idx] * self.sample_weight[idx]
                item[f'{name}_weight'] = torch.tensor(eff_w)
            else:
                vals, mask = target_tuple
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
    train_start: Optional[int] = None,
    train_end: Optional[int] = None,
    val_start: Optional[int] = None,
    val_end: Optional[int] = None,
    test_start: Optional[int] = None,
    test_end: Optional[int] = None,
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
    
    if all(v is not None for v in [train_start, train_end, val_start, val_end, test_start, test_end]):
        train = df[(df[season_col] >= train_start) & (df[season_col] <= train_end)]
        val = df[(df[season_col] >= val_start) & (df[season_col] <= val_end)]
        test = df[(df[season_col] >= test_start) & (df[season_col] <= test_end)]
    else:
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
        year1 = batch['year1'].to(device)
        year1_mask = batch['year1_mask'].to(device)
        tier2_mask = batch['tier2_mask'].to(device)
        within_mask = batch['within_mask'].to(device)
        
        # Forward
        outputs = model(tier1, tier2, career, within, tier2_mask, within_mask, year1, year1_mask)
        
        # Build targets dict
        targets = {}
        for name in ['rapm', 'gap', 'epm', 'dev', 'survival']:
            if f'{name}_target' in batch:
                if f'{name}_weight' in batch:
                    targets[name] = (
                        batch[f'{name}_target'].to(device),
                        batch[f'{name}_mask'].to(device).bool(),
                        batch[f'{name}_weight'].to(device),
                    )
                else:
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
        year1 = batch['year1'].to(device)
        year1_mask = batch['year1_mask'].to(device)
        
        outputs = model(tier1, tier2, career, within, tier2_mask, within_mask, year1, year1_mask)
        
        targets = {}
        for name in ['rapm', 'gap', 'epm', 'dev', 'survival']:
            if f'{name}_target' in batch:
                if f'{name}_weight' in batch:
                    targets[name] = (
                        batch[f'{name}_target'].to(device),
                        batch[f'{name}_mask'].to(device).bool(),
                        batch[f'{name}_weight'].to(device),
                    )
                else:
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
            rapm_target, rapm_mask = targets['rapm'][0], targets['rapm'][1]
            # RAPM may be supervised as scalar (legacy) or [ovr, off, def].
            if rapm_mask.dim() == 2:
                mask_ovr = rapm_mask[:, 0]
                if mask_ovr.any():
                    all_preds.extend(outputs['rapm_pred'][:, 0][mask_ovr].cpu().numpy())
                    all_targets.extend(rapm_target[:, 0][mask_ovr].cpu().numpy())
            else:
                if rapm_mask.any():
                    all_preds.extend(outputs['rapm_pred'][:, 0][rapm_mask].cpu().numpy())
                    all_targets.extend(rapm_target[rapm_mask].cpu().numpy())
    
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


@torch.no_grad()
def build_season_recalibration(
    model: ProspectModel,
    dataset: ProspectDataset,
    device: torch.device,
    min_samples_per_season: int = 15,
    shrinkage: float = 25.0,
) -> Dict[str, object]:
    """
    Build season-level residual offsets for RAPM prediction.
    Offset(season) = shrinked mean residual for that season.
    """
    model.eval()
    batch = {
        'tier1': torch.from_numpy(dataset.tier1).to(device),
        'tier2': torch.from_numpy(dataset.tier2).to(device),
        'career': torch.from_numpy(dataset.career).to(device),
        'within': torch.from_numpy(dataset.within).to(device),
        'tier2_mask': torch.from_numpy(dataset.tier2_mask).unsqueeze(1).to(device),
        'within_mask': torch.from_numpy(dataset.within_mask).unsqueeze(1).to(device),
        'year1': torch.from_numpy(dataset.year1).to(device),
        'year1_mask': torch.from_numpy(dataset.year1_mask).to(device),
    }
    out = model(
        batch['tier1'], batch['tier2'], batch['career'], batch['within'],
        batch['tier2_mask'], batch['within_mask'], batch['year1'], batch['year1_mask']
    )

    if 'rapm' not in dataset.targets:
        return {"offsets_by_season": {}, "global_offset": 0.0, "n_samples": 0}

    y, mask = dataset.targets['rapm'][0], dataset.targets['rapm'][1]
    if mask.ndim == 2:
        y = y[:, 0]
        mask = mask[:, 0]
    pred = out['rapm_pred'][:, 0].detach().cpu().numpy()
    seasons = dataset.seasons

    valid = mask & np.isfinite(seasons)
    if valid.sum() == 0:
        return {"offsets_by_season": {}, "global_offset": 0.0, "n_samples": 0}

    resid = y[valid] - pred[valid]
    season_vals = seasons[valid].astype(int)
    global_offset = float(np.mean(resid))

    offsets = {}
    for s in np.unique(season_vals):
        sel = season_vals == s
        n = int(sel.sum())
        if n < min_samples_per_season:
            continue
        m = float(np.mean(resid[sel]))
        # Empirical-Bayes style shrinkage toward global mean.
        shrunk = (n / (n + shrinkage)) * m + (shrinkage / (n + shrinkage)) * global_offset
        offsets[str(int(s))] = {"offset": float(shrunk), "n": n, "raw_mean_resid": m}

    return {
        "offsets_by_season": offsets,
        "global_offset": global_offset,
        "n_samples": int(valid.sum()),
        "min_samples_per_season": int(min_samples_per_season),
        "shrinkage": float(shrinkage),
    }


@torch.no_grad()
def compute_iterative_reweight_multipliers(
    model: ProspectModel,
    dataset: ProspectDataset,
    device: torch.device,
    min_samples_per_group: int = 20,
    shrinkage: float = 20.0,
    strength: float = 0.35,
    min_mult: float = 0.7,
    max_mult: float = 1.4,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute epoch-wise adaptive multipliers from residual error by
    (season, within-availability, class_year_bin).
    Higher residual groups are upweighted (capped), lower residual groups are downweighted.
    """
    n = len(dataset)
    multipliers = np.ones(n, dtype=np.float32)

    if 'rapm' not in dataset.targets:
        return multipliers, {"enabled": 0.0, "n_groups": 0.0, "mean_mult": 1.0}

    model.eval()
    batch = {
        'tier1': torch.from_numpy(dataset.tier1).to(device),
        'tier2': torch.from_numpy(dataset.tier2).to(device),
        'career': torch.from_numpy(dataset.career).to(device),
        'within': torch.from_numpy(dataset.within).to(device),
        'tier2_mask': torch.from_numpy(dataset.tier2_mask).unsqueeze(1).to(device),
        'within_mask': torch.from_numpy(dataset.within_mask).unsqueeze(1).to(device),
        'year1': torch.from_numpy(dataset.year1).to(device),
        'year1_mask': torch.from_numpy(dataset.year1_mask).to(device),
    }
    out = model(
        batch['tier1'], batch['tier2'], batch['career'], batch['within'],
        batch['tier2_mask'], batch['within_mask'], batch['year1'], batch['year1_mask']
    )

    y, mask, _ = dataset.targets['rapm']
    if mask.ndim == 2:
        y = y[:, 0]
        mask = mask[:, 0]
    pred = out['rapm_pred'][:, 0].detach().cpu().numpy()
    seasons = pd.to_numeric(pd.Series(dataset.seasons), errors='coerce').values
    within_bin = (dataset.within_mask >= 0.5).astype(int)
    if 'class_year' in dataset.df.columns:
        cls = pd.to_numeric(dataset.df['class_year'], errors='coerce').fillna(0).values
    else:
        cls = np.zeros(len(dataset.df), dtype=float)
    # 0=unknown, 1=freshman, 2=soph, 3=junior, 4=senior+, clipped
    class_bin = np.clip(np.floor(cls).astype(int), 0, 4)

    valid = mask & np.isfinite(seasons) & np.isfinite(pred)
    if valid.sum() < max(min_samples_per_group, 10):
        return multipliers, {"enabled": 0.0, "n_groups": 0.0, "mean_mult": 1.0}

    abs_err = np.abs(y - pred)
    global_err = float(np.mean(abs_err[valid]))
    global_err = max(global_err, 1e-6)

    n_groups = 0
    for s in np.unique(seasons[valid].astype(int)):
        season_sel = valid & (seasons.astype(int) == s)
        for wb in (0, 1):
            for cb in (0, 1, 2, 3, 4):
                sel = season_sel & (within_bin == wb) & (class_bin == cb)
                n_sel = int(sel.sum())
                if n_sel < min_samples_per_group:
                    continue
                group_err = float(np.mean(abs_err[sel]))
                shrunk = (n_sel / (n_sel + shrinkage)) * group_err + (shrinkage / (n_sel + shrinkage)) * global_err
                rel = max(shrunk / global_err, 1e-6)
                m = float(np.clip(rel ** strength, min_mult, max_mult))
                multipliers[sel] = m
                n_groups += 1

    # Keep average weight stable to avoid implicit LR changes.
    mean_valid = float(np.mean(multipliers[valid])) if valid.any() else 1.0
    if mean_valid > 0:
        multipliers[valid] = multipliers[valid] / mean_valid
    multipliers = np.clip(multipliers, min_mult, max_mult).astype(np.float32)

    diag = {
        "enabled": 1.0,
        "n_groups": float(n_groups),
        "global_abs_err": global_err,
        "mean_mult": float(np.mean(multipliers[valid])) if valid.any() else 1.0,
        "min_mult": float(np.min(multipliers[valid])) if valid.any() else 1.0,
        "max_mult": float(np.max(multipliers[valid])) if valid.any() else 1.0,
    }
    return multipliers, diag


def train_model(
    model: ProspectModel,
    train_loader: DataLoader,
    train_dataset: ProspectDataset,
    val_loader: DataLoader,
    criterion: ProspectLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    n_epochs: int,
    patience: int = 20,
    iterative_reweight: bool = True,
    reweight_min_group: int = 20,
    reweight_shrinkage: float = 20.0,
    reweight_strength: float = 0.35,
    reweight_min_mult: float = 0.7,
    reweight_max_mult: float = 1.4,
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

        if iterative_reweight:
            multipliers, rw_diag = compute_iterative_reweight_multipliers(
                model,
                train_dataset,
                device,
                min_samples_per_group=reweight_min_group,
                shrinkage=reweight_shrinkage,
                strength=reweight_strength,
                min_mult=reweight_min_mult,
                max_mult=reweight_max_mult,
            )
            train_dataset.set_adaptive_weights(multipliers)
            logger.info(
                "Iterative reweight | groups=%d global_abs_err=%.4f mult[min/mean/max]=%.3f/%.3f/%.3f",
                int(rw_diag.get("n_groups", 0)),
                float(rw_diag.get("global_abs_err", 0.0)),
                float(rw_diag.get("min_mult", 1.0)),
                float(rw_diag.get("mean_mult", 1.0)),
                float(rw_diag.get("max_mult", 1.0)),
            )
        
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
    train_df, val_df, test_df = create_temporal_splits(
        df,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    
    # Feature columns (use what's available)
    tier1_cols = [c for c in TIER1_COLUMNS if c in df.columns]
    tier2_cols = [c for c in TIER2_COLUMNS if c in df.columns]
    career_cols = [c for c in CAREER_BASE_COLUMNS if c in df.columns]
    within_cols = [c for c in WITHIN_COLUMNS if c in df.columns]
    year1_cols = [c for c in YEAR1_INTERACTION_COLUMNS if c in df.columns]
    
    logger.info(
        f"Features: Tier1={len(tier1_cols)}, Tier2={len(tier2_cols)}, "
        f"Career={len(career_cols)}, Within={len(within_cols)}, Year1Interaction={len(year1_cols)}"
    )
    
    # Datasets
    train_dataset = ProspectDataset(
        train_df, tier1_cols, tier2_cols, career_cols, within_cols, YEAR1_INTERACTION_COLUMNS,
        training=True,
        temporal_decay_half_life=args.temporal_decay_half_life,
        temporal_decay_min=args.temporal_decay_min,
    )
    val_dataset = ProspectDataset(
        val_df, tier1_cols, tier2_cols, career_cols, within_cols, YEAR1_INTERACTION_COLUMNS,
        training=False,
        temporal_decay_half_life=args.temporal_decay_half_life,
        temporal_decay_min=args.temporal_decay_min,
    )
    test_dataset = ProspectDataset(
        test_df, tier1_cols, tier2_cols, career_cols, within_cols, YEAR1_INTERACTION_COLUMNS,
        training=False,
        temporal_decay_half_life=args.temporal_decay_half_life,
        temporal_decay_min=args.temporal_decay_min,
    )

    within_cov = float(np.mean(train_dataset.within_mask)) if len(train_dataset.within_mask) else 0.0
    if within_cov < 0.01:
        logger.warning(
            "Within-season feature coverage is very low (%.2f%%). "
            "Within branch/freshman modulation will have limited effect until windows are populated.",
            100.0 * within_cov,
        )
    
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
        year1_feature_dim=len(YEAR1_INTERACTION_COLUMNS),
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
        lambda_rapm=args.lambda_rapm,
        lambda_rapm_var=args.lambda_rapm_var,
        lambda_gap=args.lambda_gap,
        lambda_epm=args.lambda_epm,
        lambda_dev=args.lambda_dev,
        lambda_surv=args.lambda_surv,
        lambda_arch=args.lambda_arch,
        lambda_kl=0.01 if args.use_vae else 0.0,
    )
    logger.info(
        "Loss weights: rapm=%.3f rapm_var=%.3f gap=%.3f epm=%.3f dev=%.3f surv=%.3f arch=%.3f",
        args.lambda_rapm, args.lambda_rapm_var, args.lambda_gap, args.lambda_epm, args.lambda_dev, args.lambda_surv, args.lambda_arch
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
        model, train_loader, train_dataset, val_loader, criterion, optimizer, scheduler,
        device, args.epochs, args.patience,
        iterative_reweight=args.iterative_reweight,
        reweight_min_group=args.reweight_min_group,
        reweight_shrinkage=args.reweight_shrinkage,
        reweight_strength=args.reweight_strength,
        reweight_min_mult=args.reweight_min_mult,
        reweight_max_mult=args.reweight_max_mult,
        output_dir=output_dir
    )
    
    # Test evaluation
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test RMSE: {test_metrics.get('rapm_rmse', 0):.4f}")
    results['test'] = test_metrics
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "model.pt")
    model_cfg = {
        "latent_dim": int(args.latent_dim),
        "n_archetypes": int(args.n_archetypes),
        "use_vae": bool(args.use_vae),
        "predict_uncertainty": True,
        "condition_on_archetypes": bool(getattr(args, "condition_on_archetypes", False)),
        "year1_feature_dim": len(YEAR1_INTERACTION_COLUMNS),
        "tier1_dim": len(tier1_cols),
        "tier2_dim": len(tier2_cols),
        "career_dim": len(career_cols),
        "within_dim": len(within_cols),
        "tier1_columns": list(tier1_cols),
        "tier2_columns": list(tier2_cols),
        "career_columns": list(career_cols),
        "within_columns": list(within_cols),
        "year1_interaction_columns": list(YEAR1_INTERACTION_COLUMNS),
    }
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)
    
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
        rapm_vals = train_dataset.targets['rapm'][0]
        targets_for_analysis['rapm'] = rapm_vals[:, 0] if rapm_vals.ndim == 2 else rapm_vals
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

    # Season-level recalibration artifact (inflation/deflation correction).
    recal = build_season_recalibration(
        model,
        train_dataset,
        device,
        min_samples_per_season=args.recal_min_samples,
        shrinkage=args.recal_shrinkage,
    )
    recal["generated_at"] = datetime.now().isoformat()
    with open(output_dir / "season_recalibration.json", "w") as f:
        json.dump(recal, f, indent=2)
    
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
    parser.add_argument('--temporal-decay-half-life', type=float, default=4.0, help='Half-life in seasons for recency weighting')
    parser.add_argument('--temporal-decay-min', type=float, default=0.2, help='Minimum recency weight floor')
    parser.add_argument('--train-start', type=int, default=None, help='Train split season start (inclusive)')
    parser.add_argument('--train-end', type=int, default=None, help='Train split season end (inclusive)')
    parser.add_argument('--val-start', type=int, default=None, help='Validation split season start (inclusive)')
    parser.add_argument('--val-end', type=int, default=None, help='Validation split season end (inclusive)')
    parser.add_argument('--test-start', type=int, default=None, help='Test split season start (inclusive)')
    parser.add_argument('--test-end', type=int, default=None, help='Test split season end (inclusive)')
    parser.add_argument('--recal-min-samples', type=int, default=15, help='Min samples per season for recalibration offsets')
    parser.add_argument('--recal-shrinkage', type=float, default=25.0, help='Shrinkage strength for season offsets')
    parser.add_argument('--iterative-reweight', action=argparse.BooleanOptionalAction, default=True, help='Enable epoch-wise residual reweighting by season/within groups')
    parser.add_argument('--reweight-min-group', type=int, default=20, help='Minimum samples per (season,within) group for adaptive reweight')
    parser.add_argument('--reweight-shrinkage', type=float, default=20.0, help='Shrinkage strength for adaptive group error estimates')
    parser.add_argument('--reweight-strength', type=float, default=0.35, help='Exponent on relative error when mapping to adaptive multipliers')
    parser.add_argument('--reweight-min-mult', type=float, default=0.7, help='Lower bound for adaptive multipliers')
    parser.add_argument('--reweight-max-mult', type=float, default=1.4, help='Upper bound for adaptive multipliers')
    parser.add_argument('--lambda-rapm', type=float, default=1.0, help='Primary RAPM loss weight')
    parser.add_argument('--lambda-rapm-var', type=float, default=0.20, help='RAPM variance-matching regularization weight')
    parser.add_argument('--lambda-gap', type=float, default=0.15, help='Gap auxiliary loss weight')
    parser.add_argument('--lambda-epm', type=float, default=0.20, help='Year-1 EPM auxiliary loss weight')
    parser.add_argument('--lambda-dev', type=float, default=0.20, help='Development-rate auxiliary loss weight')
    parser.add_argument('--lambda-surv', type=float, default=0.10, help='Survival auxiliary loss weight')
    parser.add_argument('--lambda-arch', type=float, default=0.05, help='Archetype regularization weight')
    
    args = parser.parse_args()
    main(args)
