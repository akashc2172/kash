#!/usr/bin/env python3
"""
Stage 4: 2026 Model Stack
=========================
Phases A-D unified in one script:
  A) Masked Tabular Autoencoder (MTAE) pre-training on Foundation surface
  B) Supervised fine-tuning on Joint/Supervised surface
  C) Heteroscedastic head (mu + logvar) for uncertainty
  D) Age-conditioned hypernetwork gating

Usage:
  python train_2026_model.py --phase A        # Pre-train encoder
  python train_2026_model.py --phase B        # Fine-tune supervised
  python train_2026_model.py --phase full     # A → B end-to-end
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
FOUNDATION_PATH = BASE / "data" / "training" / "foundation_college_table.parquet"
SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
JOINT_PATH = BASE / "data" / "training" / "unified_training_table_joint.parquet"
MODEL_DIR = BASE / "models" / "stack2026"
AUDIT_DIR = BASE / "data" / "audit"

# Feature columns to use for pre-training (numeric only, no IDs/targets/masks)
EXCLUDE_PREFIXES = [
    'athlete_id', 'nba_id', 'season', 'split_id', 'college_final_season',
    'y_peak', 'actual_peak', 'actual_year1', 'year1_epm', 'gap_ts', 'dev_rate',
    'made_nba', 'year1_mp', 'peak_poss', 'draft_year', 'rookie_season',
    'target_asof', 'epm_years', 'rapm_years', 'is_epm_mature', 'is_rapm_mature',
    'has_peak', 'has_year1', 'has_nba_link', 'link_method', 'source_combine',
    'source_', 'confidence', 'combine_source', 'bbr_id', 'pid',
    'player_name', 'norm_name', 'match_', 'backfill_', 'hist_',
    'draft_year_proxy', 'college_teamId', 'teamId',
]

EXCLUDE_SUFFIXES = ['_missing', '_source']

TARGET_COL = 'y_peak_epm_window'
AUX_TARGETS = ['year1_epm_tot', 'y_peak_ovr']


def get_feature_columns(df: pd.DataFrame) -> list:
    """Select numeric feature columns, excluding IDs, targets, masks."""
    cols = []
    for c in df.columns:
        if any(c.startswith(p) or c == p for p in EXCLUDE_PREFIXES):
            continue
        if any(c.endswith(s) for s in EXCLUDE_SUFFIXES):
            continue
        if df[c].dtype in ['object', 'bool', 'category']:
            continue
        cols.append(c)
    return cols


class TabularDataset(Dataset):
    """Simple tabular dataset for PyTorch."""
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# =============================================================================
# Phase A: Masked Tabular Autoencoder
# =============================================================================
class MaskedTabularAutoencoder(nn.Module):
    """
    Masked Tabular Autoencoder for self-supervised pre-training.
    Randomly masks input features and reconstructs them.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, mask_ratio=0.3):
        # Create random mask
        mask = torch.bernoulli(torch.full_like(x, mask_ratio)).bool()
        x_masked = x.clone()
        x_masked[mask] = 0.0

        # Encode and decode
        z = self.encoder(x_masked)
        x_hat = self.decoder(z)

        # Loss only on masked positions
        loss = F.mse_loss(x_hat[mask], x[mask])
        return loss, z


# =============================================================================
# Phase B+C: Supervised Head with Heteroscedastic Output
# =============================================================================
class HeteroscedasticHead(nn.Module):
    """
    Supervised fine-tuning head.
    Outputs mu (prediction) and logvar (uncertainty).
    Phase C heteroscedastic Gaussian NLL loss.
    """
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, n_targets: int = 1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.mu_head = nn.Linear(hidden_dim, n_targets)
        self.logvar_head = nn.Linear(hidden_dim, n_targets)

    def forward(self, z):
        h = self.shared(z)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


# =============================================================================
# Phase D: Age-Conditioned Hypernetwork
# =============================================================================
class AgeConditionedHyperNet(nn.Module):
    """
    Generates modulation weights conditioned on age/class/career context.
    Modulates the final hidden layer via element-wise gating.
    """
    def __init__(self, n_context: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.hyper = nn.Sequential(
            nn.Linear(n_context, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
            nn.Sigmoid(),  # Gate values between 0 and 1
        )

    def forward(self, context):
        return self.hyper(context)


# =============================================================================
# Full Model: Encoder + HetHead + HyperNet
# =============================================================================
class Stack2026Model(nn.Module):
    """Complete 2026 model: Pretrained encoder + heteroscedastic head + hypernetwork."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128,
                 n_targets: int = 1, use_hypernetwork: bool = True):
        super().__init__()
        self.autoencoder = MaskedTabularAutoencoder(input_dim, hidden_dim, latent_dim)
        self.head = HeteroscedasticHead(latent_dim, hidden_dim=64, n_targets=n_targets)
        self.use_hypernetwork = use_hypernetwork
        if use_hypernetwork:
            self.hypernet = AgeConditionedHyperNet(n_context=3, hidden_dim=64)

    def forward(self, x, context=None):
        z = self.autoencoder.encode(x)
        mu, logvar = self.head(z)

        if self.use_hypernetwork and context is not None:
            gate = self.hypernet(context)
            # Apply gating to the mu head (modulation)
            mu = mu * gate[:, :mu.shape[1]]

        return mu, logvar

    def predict(self, x, context=None):
        """Inference mode: returns mu, sd."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.forward(x, context)
            sd = torch.exp(0.5 * logvar)
        return mu, sd


def heteroscedastic_loss(mu, logvar, y, beta=1.0):
    """
    Gaussian NLL with variance regularization.
    beta controls how much we penalize large variance predictions.
    Includes a floor on logvar to prevent collapse.
    """
    # Floor logvar at -4 (minimum sd ≈ 0.135) to prevent collapse
    logvar = torch.clamp(logvar, min=-4.0, max=4.0)
    precision = torch.exp(-logvar)
    nll = 0.5 * (precision * (y - mu) ** 2 + logvar)
    # Variance regularization: penalize overly negative logvar to prevent collapse
    var_reg = beta * F.relu(-logvar - 1.0).mean()
    return nll.mean() + var_reg


# =============================================================================
# Training Loops
# =============================================================================
def train_phase_a(foundation_path: Path, model_dir: Path, epochs: int = 50, lr: float = 1e-3):
    """Phase A: Self-supervised masked pre-training on Foundation surface."""
    logger.info("=" * 60)
    logger.info("PHASE A: Masked Tabular Pre-Training")
    logger.info("=" * 60)

    df = pd.read_parquet(foundation_path)
    feat_cols = get_feature_columns(df)
    logger.info(f"Foundation rows: {len(df):,}, Feature cols: {len(feat_cols)}")

    # Robust numeric coercion: object/list-like columns become NaN -> 0
    # so supervised fine-tune doesn't crash on mixed schema artifacts.
    x_frame = df[feat_cols].copy()
    for c in feat_cols:
        x_frame[c] = pd.to_numeric(x_frame[c], errors='coerce')
    X = x_frame.fillna(0.0).to_numpy(dtype=np.float32)

    # Z-score normalize
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-8] = 1.0
    X = (X - means) / stds

    dataset = TabularDataset(X)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

    model = MaskedTabularAutoencoder(input_dim=len(feat_cols))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for batch in loader:
            x = batch
            loss, _ = model(x, mask_ratio=0.3)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'feature_cols': feat_cols,
                'means': means.tolist(),
                'stds': stds.tolist(),
                'input_dim': len(feat_cols),
            }, model_dir / 'pretrained_encoder.pt')

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

    logger.info(f"Phase A complete. Best reconstruction loss: {best_loss:.6f}")

    # Save history
    pd.DataFrame(history).to_csv(model_dir / 'phase_a_training_history.csv', index=False)
    return model


def train_phase_b(supervised_path: Path, model_dir: Path, epochs: int = 100, lr: float = 5e-4):
    """Phase B+C+D: Supervised fine-tuning with heteroscedastic head."""
    logger.info("=" * 60)
    logger.info("PHASE B: Supervised Fine-Tuning + Heteroscedastic Head")
    logger.info("=" * 60)

    # Load pretrained encoder
    ckpt_path = model_dir / 'pretrained_encoder.pt'
    if not ckpt_path.exists():
        logger.error("No pretrained encoder found! Run Phase A first.")
        return

    ckpt = torch.load(ckpt_path, weights_only=False)
    feat_cols = ckpt['feature_cols']
    means = np.array(ckpt['means'])
    stds = np.array(ckpt['stds'])
    input_dim = ckpt['input_dim']

    # Load supervised data
    df = pd.read_parquet(supervised_path)
    logger.info(f"Supervised rows: {len(df):,}")

    # Ensure feature columns exist
    available_cols = [c for c in feat_cols if c in df.columns]
    missing_cols = [c for c in feat_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"  {len(missing_cols)} feature cols missing in supervised data, zero-filling")
        for c in missing_cols:
            df[c] = 0.0

    x_frame = df[feat_cols].copy()
    for c in feat_cols:
        x_frame[c] = pd.to_numeric(x_frame[c], errors='coerce')
    X = x_frame.fillna(0.0).to_numpy(dtype=np.float32)
    X = (X - means) / stds

    # Targets - use mature/observed rows only for primary supervision.
    if TARGET_COL not in df.columns:
        logger.error(f"Target column {TARGET_COL} not found!")
        return
    t = pd.to_numeric(df[TARGET_COL], errors='coerce')
    mature = pd.to_numeric(df.get('is_epm_mature'), errors='coerce').fillna(0).astype(int) if 'is_epm_mature' in df.columns else pd.Series(1, index=df.index)
    keep = t.notna() & (mature == 1)
    if keep.sum() < 100:
        logger.warning("Too few mature rows for strict mask; falling back to non-null target rows.")
        keep = t.notna()
    df = df.loc[keep].reset_index(drop=True)
    X = X[keep.values]
    y_raw = t.loc[keep].astype(np.float32).values
    y_mean = y_raw.mean()
    y_std = y_raw.std()
    if y_std < 1e-6:
        y_std = 1.0
    y = ((y_raw - y_mean) / y_std).reshape(-1, 1)
    logger.info(f"  Target stats: mean={y_mean:.4f}, std={y_std:.4f}")

    # Context features for hypernetwork (age/class/career at college time).
    context_cols = []
    for c in ['age_at_season', 'class_year', 'career_years']:
        if c in df.columns:
            context_cols.append(c)
    has_context = len(context_cols) >= 1

    if has_context:
        c_frame = df[context_cols].copy()
        for c in context_cols:
            c_frame[c] = pd.to_numeric(c_frame[c], errors='coerce')
        ctx = c_frame.fillna(0.0).to_numpy(dtype=np.float32)
        # Pad to 3 if needed
        if ctx.shape[1] < 3:
            ctx = np.pad(ctx, ((0, 0), (0, 3 - ctx.shape[1])))
    else:
        ctx = np.zeros((len(df), 3), dtype=np.float32)

    # Build full model
    model = Stack2026Model(input_dim=input_dim, use_hypernetwork=has_context)

    # Load pretrained encoder weights
    model.autoencoder.encoder.load_state_dict(ckpt['encoder_state_dict'])
    logger.info("  Loaded pretrained encoder weights")

    # Freeze encoder for first 20 epochs, then unfreeze
    for param in model.autoencoder.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )

    dataset = TabularDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        # Unfreeze encoder after warmup
        if epoch == 20:
            for param in model.autoencoder.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            logger.info("  Unfroze encoder at epoch 20")

        model.train()
        epoch_loss = 0
        n_batches = 0

        for i, (x_batch, y_batch) in enumerate(loader):
            ctx_batch = torch.FloatTensor(ctx[i * 64: (i + 1) * 64]) if has_context else None
            if ctx_batch is not None and len(ctx_batch) != len(x_batch):
                ctx_batch = ctx_batch[:len(x_batch)]

            mu, logvar = model(x_batch, ctx_batch)
            loss = heteroscedastic_loss(mu, logvar, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_cols': feat_cols,
                'means': means.tolist(),
                'stds': stds.tolist(),
                'input_dim': input_dim,
                'context_cols': context_cols,
            }, model_dir / 'stack2026_supervised.pt')

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

    logger.info(f"Phase B complete. Best supervised loss: {best_loss:.4f}")
    pd.DataFrame(history).to_csv(model_dir / 'phase_b_training_history.csv', index=False)

    # Generate predictions for audit (rescale back to original target scale)
    model.eval()
    X_tensor = torch.FloatTensor(X)
    ctx_tensor = torch.FloatTensor(ctx) if has_context else None
    with torch.no_grad():
        mu_pred, logvar_pred = model(X_tensor, ctx_tensor)
        sd_pred = torch.exp(0.5 * logvar_pred)

    # Rescale predictions back to original scale
    df['pred_mu'] = (mu_pred.numpy().flatten() * y_std) + y_mean
    df['pred_sd'] = sd_pred.numpy().flatten() * y_std
    df['pred_upside'] = df['pred_mu'] + 1.0 * df['pred_sd']
    df['pred_floor'] = df['pred_mu'] - 1.0 * df['pred_sd']

    # Anti-compression gate
    score_std = df['pred_mu'].std()
    score_iqr = df['pred_mu'].quantile(0.75) - df['pred_mu'].quantile(0.25)
    logger.info(f"  Score std: {score_std:.4f}, IQR: {score_iqr:.4f}")

    gate_report = {
        'score_std': float(score_std),
        'score_iqr': float(score_iqr),
        'std_gate_pass': bool(score_std > 0.5),
        'iqr_gate_pass': bool(score_iqr > 0.3),
        'best_loss': float(best_loss),
        'rows': int(len(df)),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / 'stack2026_model_gate.json', 'w') as f:
        json.dump(gate_report, f, indent=2)
    logger.info(f"  Model gate report: {gate_report}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['A', 'B', 'full'], default='full')
    parser.add_argument('--epochs-a', type=int, default=50)
    parser.add_argument('--epochs-b', type=int, default=100)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ['A', 'full']:
        train_phase_a(FOUNDATION_PATH, MODEL_DIR, epochs=args.epochs_a)

    if args.phase in ['B', 'full']:
        train_phase_b(SUPERVISED_PATH, MODEL_DIR, epochs=args.epochs_b)

    logger.info("=" * 60)
    logger.info("STAGE 4 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
