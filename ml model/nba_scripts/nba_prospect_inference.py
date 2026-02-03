#!/usr/bin/env python3
"""
Prospect Inference (College-Only)
=================================
Builds a college-only inference table and (optionally) runs a trained latent model.

Why this exists:
- Training uses NBA targets and the college->NBA crosswalk (only players with targets).
- Inference must work for *everyone*: UDFAs, current prospects, players without nba_id.

Inputs:
- data/college_feature_store/college_features_v1.parquet
- data/college_feature_store/prospect_career_v1.parquet
- (optional) models/latent_model_*/model.pt

Outputs (optional):
- data/inference/prospect_predictions_{DATE}.parquet

Usage:
  python nba_scripts/nba_prospect_inference.py --build-only
  python nba_scripts/nba_prospect_inference.py --model-path models/latent_model_x/model.pt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
FEATURE_STORE = BASE_DIR / "data/college_feature_store"
INFERENCE_DIR = BASE_DIR / "data/inference"


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_parquet(path)


def build_prospect_inference_table(
    college_features: pd.DataFrame,
    career_features: pd.DataFrame,
    split_id: str = "ALL__ALL",
) -> pd.DataFrame:
    """
    Create a college-only inference table.

    Returns one row per athlete_id with:
    - final college season features (prefixed college_*)
    - career aggregate features (from prospect_career_v1)
    - derived masks (has_spatial_data)
    """
    from nba_scripts.build_unified_training_table import (
        get_final_college_season,
        compute_derived_features,
        apply_era_normalization,
        apply_team_residualization,
    )

    if college_features.empty:
        raise ValueError("college_features is empty")
    if career_features.empty:
        raise ValueError("career_features is empty")

    cf = college_features.copy()
    if "split_id" in cf.columns:
        cf = cf[cf["split_id"] == split_id].copy()

    final_college = get_final_college_season(cf)

    df = final_college.merge(career_features, on="athlete_id", how="left")

    # Derived masks/features that do not require NBA labels.
    df = compute_derived_features(df)
    # Apply the same normalization transforms used in training (college-only).
    df = apply_era_normalization(df, era_col='college_final_season')
    df = apply_team_residualization(df, season_col='college_final_season', team_col='college_teamId')

    # For inference, made_nba isn't known; ensure it doesn't get misused.
    if "made_nba" in df.columns:
        df["made_nba"] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-id", default="ALL__ALL")
    parser.add_argument("--build-only", action="store_true", help="Only build the inference table, do not run a model.")
    parser.add_argument("--model-path", default="", help="Path to a trained model.pt (optional).")
    args = parser.parse_args()

    college_path = FEATURE_STORE / "college_features_v1.parquet"
    career_path = FEATURE_STORE / "prospect_career_v1.parquet"

    logger.info("Loading college feature store...")
    college = _load_parquet(college_path)
    logger.info("Loading career feature store...")
    career = _load_parquet(career_path)

    logger.info("Building inference table...")
    inf = build_prospect_inference_table(college, career, split_id=args.split_id)
    logger.info(f"Inference table: {len(inf):,} rows, {inf.shape[1]} cols")

    if args.build_only or not args.model_path:
        return

    # Optional: run a trained model (best-effort). We keep this lightweight and
    # do not require a specific feature set beyond what the model expects.
    import torch
    from models import ProspectModel, TIER1_COLUMNS, TIER2_COLUMNS, CAREER_BASE_COLUMNS, WITHIN_COLUMNS

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    # Select feature columns that actually exist.
    tier1_cols = [c for c in TIER1_COLUMNS if c in inf.columns]
    tier2_cols = [c for c in TIER2_COLUMNS if c in inf.columns]
    career_cols = [c for c in CAREER_BASE_COLUMNS if c in inf.columns]
    within_cols = [c for c in WITHIN_COLUMNS if c in inf.columns]

    logger.info(
        f"Model inputs: tier1={len(tier1_cols)} tier2={len(tier2_cols)} "
        f"career={len(career_cols)} within={len(within_cols)}"
    )

    def to_tensor(cols: list[str]) -> torch.Tensor:
        if not cols:
            return torch.zeros((len(inf), 0), dtype=torch.float32)
        arr = inf[cols].to_numpy(dtype=np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0)
        return torch.from_numpy(arr)

    tier1 = to_tensor(tier1_cols)
    tier2 = to_tensor(tier2_cols)
    career_t = to_tensor(career_cols)
    within_t = to_tensor(within_cols)

    mask = inf["has_spatial_data"].to_numpy(dtype=np.float32) if "has_spatial_data" in inf.columns else np.ones(len(inf), dtype=np.float32)
    tier2_mask = torch.from_numpy(mask).unsqueeze(1)

    # Within-season windows availability mask (derived from explicit flags).
    ws_flags = []
    for c in ["final_has_ws_last5", "final_has_ws_last10", "final_has_ws_breakout_timing_eff"]:
        if c in inf.columns:
            ws_flags.append(inf[c].fillna(0).to_numpy(dtype=np.float32))
    if ws_flags:
        within_mask_np = (np.max(np.stack(ws_flags, axis=0), axis=0) > 0).astype(np.float32)
    else:
        within_mask_np = np.zeros(len(inf), dtype=np.float32)
    within_mask = torch.from_numpy(within_mask_np).unsqueeze(1)

    model = ProspectModel(latent_dim=32, n_archetypes=8, use_vae=False, predict_uncertainty=True)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        out = model(tier1, tier2, career_t, within_t, tier2_mask, within_mask)

    # Archetype distribution output (top-k + cumulative).
    arch_probs = out["archetype_probs"].cpu().numpy()
    def topk_row(p: np.ndarray, cumulative: float = 0.85, max_k: int = 3):
        order = np.argsort(-p)
        ids = []
        probs = []
        total = 0.0
        for idx in order[: max_k * 2]:
            ids.append(int(idx))
            pr = float(p[idx])
            probs.append(pr)
            total += pr
            if len(ids) >= max_k or total >= cumulative:
                break
        return ids, probs

    top_ids = []
    top_probs = []
    for row in arch_probs:
        ids, probs = topk_row(row)
        top_ids.append(ids)
        top_probs.append(probs)

    preds = pd.DataFrame({
        "athlete_id": inf["athlete_id"].values,
        "college_final_season": inf.get("college_final_season", np.nan),
        "pred_peak_rapm": out["rapm_pred"][:, 0].cpu().numpy(),
        "pred_gap_ts": out["gap_pred"][:, 0].cpu().numpy() if out["gap_pred"].shape[1] > 0 else np.nan,
        "pred_gap_usg": out["gap_pred"][:, 1].cpu().numpy() if out["gap_pred"].shape[1] > 1 else np.nan,
        "pred_year1_epm": out["epm_pred"][:, 0].cpu().numpy() if out["epm_pred"].shape[1] > 0 else np.nan,
        "pred_made_nba_logit": out["survival_logits"][:, 0].cpu().numpy(),
        "archetype_top_ids": top_ids,
        "archetype_top_probs": top_probs,
    })

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = INFERENCE_DIR / f"prospect_predictions_{stamp}.parquet"
    preds.to_parquet(out_path, index=False)
    logger.info(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
