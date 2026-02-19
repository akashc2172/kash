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
import sys
import json
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
FEATURE_STORE = BASE_DIR / "data/college_feature_store"
INFERENCE_DIR = BASE_DIR / "data/inference"
YEAR1_INTERACTION_COLUMNS = ['year1_epm_tot', 'year1_epm_off', 'year1_epm_def', 'year1_usg', 'year1_tspct']


def _season_z(values: pd.Series, seasons: pd.Series) -> pd.Series:
    """Season-wise z-score with robust NaN/zero-std handling."""
    x = pd.to_numeric(values, errors="coerce")
    s = pd.to_numeric(seasons, errors="coerce")
    mu = x.groupby(s).transform("mean")
    sd = x.groupby(s).transform("std").replace(0, np.nan)
    z = (x - mu) / sd
    return z.fillna(0.0).clip(-3.0, 3.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_ridge_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit ridge regression with closed form on standardized features.
    Returns (coef, mean, std). Intercept is folded into coef[0] via bias column.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Xs = (X - mu) / sd
    Xb = np.column_stack([np.ones(len(Xs)), Xs])
    I = np.eye(Xb.shape[1], dtype=np.float64)
    I[0, 0] = 0.0  # don't penalize intercept
    beta = np.linalg.solve(Xb.T @ Xb + l2 * I, Xb.T @ y)
    return beta, mu, sd


def _predict_ridge(
    X: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - mu) / np.where(sd > 1e-8, sd, 1.0)
    Xb = np.column_stack([np.ones(len(Xs)), Xs])
    return Xb @ beta


def _num_array(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=np.float64)


def _build_meta_features(
    pred_peak: np.ndarray,
    pred_surv: np.ndarray,
    pred_year1: np.ndarray,
    pred_dev: np.ndarray,
    src_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build a richer ranking feature matrix to avoid collapsed rank ordering.
    Uses model outputs + high-signal college inputs that are available pre-NBA.
    """
    x = [
        pred_peak,
        pred_surv,
        pred_year1,
        pred_dev,
        pred_peak * pred_surv,
        np.square(pred_peak),
        np.log1p(np.clip(_num_array(src_df, "college_games_played"), 0.0, None)),
        np.log1p(np.clip(_num_array(src_df, "college_poss_proxy"), 0.0, None)),
        _num_array(src_df, "final_usage"),
        _num_array(src_df, "final_trueShootingPct"),
        _num_array(src_df, "college_ast_total_per100poss"),
        _num_array(src_df, "college_tov_total_per100poss"),
        _num_array(src_df, "college_stl_total_per100poss"),
        _num_array(src_df, "college_blk_total_per100poss"),
        _num_array(src_df, "college_orb_total_per100poss"),
        _num_array(src_df, "college_drb_total_per100poss"),
        _num_array(src_df, "college_trb_total_per100poss"),
        _num_array(src_df, "college_team_srs"),
        _num_array(src_df, "college_team_rank"),
        _num_array(src_df, "college_recruiting_rating"),
        -np.log1p(np.clip(_num_array(src_df, "college_recruiting_rank"), 0.0, None)),
        _num_array(src_df, "college_assisted_share_rim"),
        _num_array(src_df, "college_assisted_share_mid"),
        _num_array(src_df, "college_assisted_share_three"),
        _num_array(src_df, "college_rapm_standard"),
        _num_array(src_df, "college_o_rapm"),
        _num_array(src_df, "college_d_rapm"),
        _num_array(src_df, "college_on_net_rating"),
        _num_array(src_df, "high_lev_att_rate"),
        _num_array(src_df, "garbage_att_rate"),
        _num_array(src_df, "leverage_poss_share"),
        _num_array(src_df, "career_years"),
        _num_array(src_df, "class_year"),
        _num_array(src_df, "age_at_season"),
        _num_array(src_df, "slope_usage"),
        _num_array(src_df, "slope_trueShootingPct"),
    ]
    return np.column_stack(x)


def _safe_std(x: np.ndarray, floor: float = 1e-6) -> float:
    s = float(np.nanstd(x))
    return s if np.isfinite(s) and s > floor else floor


def _fit_linear_gaussian_evidence(u: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit y ~= a + b*u and return (a, b, residual_sd).
    """
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu_u = float(np.nanmean(u))
    mu_y = float(np.nanmean(y))
    var_u = float(np.nanvar(u))
    if not np.isfinite(var_u) or var_u < 1e-8:
        b = 0.0
    else:
        cov = float(np.nanmean((u - mu_u) * (y - mu_y)))
        b = cov / var_u
    a = mu_y - b * mu_u
    resid = y - (a + b * u)
    resid_sd = _safe_std(resid, floor=0.25)
    return a, b, resid_sd


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
        load_derived_box_stats,
        load_college_dev_rate,
        load_transfer_context_summary,
        load_college_impact_stack,
        build_final_season_leverage_features,
        load_team_strength_features,
    )

    if college_features.empty:
        raise ValueError("college_features is empty")
    if career_features.empty:
        raise ValueError("career_features is empty")

    cf_all = college_features.copy()
    cf = cf_all.copy()
    if "split_id" in cf.columns:
        cf = cf[cf["split_id"] == split_id].copy()
    # Preserve valid transfer multi-team rows while removing upstream fragment duplicates.
    dedupe_keys = [k for k in ["athlete_id", "season", "split_id", "teamId"] if k in cf.columns]
    if dedupe_keys:
        cf = cf.drop_duplicates(subset=dedupe_keys)

    # Keep train/serve parity: merge derived box stats used by unified training build.
    derived_stats = load_derived_box_stats()
    if not derived_stats.empty:
        cf = cf.merge(derived_stats, on=["athlete_id", "season"], how="left")
        for stat in ["ast", "stl", "blk", "tov"]:
            derived_col = f"college_{stat}_total"
            target_col = f"{stat}_total"
            if derived_col in cf.columns:
                cf[target_col] = cf[derived_col].fillna(0)
        if "college_games_played" in cf.columns:
            cf["games_played"] = cf["college_games_played"].fillna(0)

    final_college = get_final_college_season(cf)
    # Train-serve parity: leverage rates are derived from all splits, not only ALL__ALL.
    lev = build_final_season_leverage_features(cf_all)
    if not lev.empty and "college_final_season" in final_college.columns:
        final_college = final_college.merge(
            lev,
            on=["athlete_id", "college_final_season"],
            how="left",
        )

    df = final_college.merge(career_features, on="athlete_id", how="left")

    # Align with training-side feature surfaces so inference doesn't collapse to defaults.
    dev = load_college_dev_rate()
    if not dev.empty:
        dev_join_keys = ["athlete_id"]
        if "final_college_season" in dev.columns and "college_final_season" in df.columns:
            dev = dev.rename(columns={"final_college_season": "college_final_season"})
            dev_join_keys.append("college_final_season")
        dev_cols = dev_join_keys + [c for c in dev.columns if c not in dev_join_keys]
        df = df.merge(dev[dev_cols], on=dev_join_keys, how="left")

    transfer = load_transfer_context_summary()
    if not transfer.empty:
        ts_cols = ["athlete_id"] + [c for c in transfer.columns if c != "athlete_id"]
        df = df.merge(transfer[ts_cols], on="athlete_id", how="left")

    impact = load_college_impact_stack()
    if not impact.empty and "college_final_season" in df.columns:
        impact = impact.rename(columns={"season": "college_final_season"})
        impact_cols = ["athlete_id", "college_final_season"] + [
            c for c in impact.columns if c not in {"athlete_id", "college_final_season"}
        ]
        df = df.merge(impact[impact_cols], on=["athlete_id", "college_final_season"], how="left")

    # Train-serve parity: team-strength/SRS enrich.
    srs = load_team_strength_features()
    if not srs.empty and {"college_teamId", "college_final_season"}.issubset(df.columns):
        srs = srs.rename(columns={"teamId": "college_teamId", "season": "college_final_season"})
        use_cols = ["college_teamId", "college_final_season", "college_team_srs", "team_strength_srs", "college_team_rank"]
        use_cols = [c for c in use_cols if c in srs.columns]
        if use_cols:
            df = df.merge(srs[use_cols], on=["college_teamId", "college_final_season"], how="left")

    # Derived masks/features that do not require NBA labels.
    df = compute_derived_features(df)
    if "draft_year_proxy" not in df.columns and "college_final_season" in df.columns:
        df["draft_year_proxy"] = pd.to_numeric(df["college_final_season"], errors="coerce") + 1.0
    # Keep inference schema aligned with training wingspan surface.
    if "has_wingspan" not in df.columns:
        src_cols = [c for c in ["wingspan_in", "standing_reach_in", "wingspan_minus_height_in"] if c in df.columns]
        if src_cols:
            df["has_wingspan"] = df[src_cols].notna().any(axis=1).astype(int)
        else:
            df["has_wingspan"] = 0
    for col in ["wingspan_in", "standing_reach_in", "wingspan_minus_height_in"]:
        if col not in df.columns:
            df[col] = np.nan
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
    parser.add_argument("--recalibration-path", default="", help="Optional season_recalibration.json path.")
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
    from models.player_encoder import PlayerEncoder

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    # Keep fixed branch widths to match model dimensions.
    tier1_cols = list(TIER1_COLUMNS)
    tier2_cols = list(TIER2_COLUMNS)
    career_cols = list(CAREER_BASE_COLUMNS)
    within_cols = list(WITHIN_COLUMNS)
    interaction_cols = list(YEAR1_INTERACTION_COLUMNS)

    # Align feature sets with the training table surface so checkpoint input dims match.
    try:
        train_path = BASE_DIR / "data/training/unified_training_table.parquet"
        if train_path.exists():
            train_cols = set(pd.read_parquet(train_path).columns.tolist())
            tier1_cols = [c for c in tier1_cols if c in train_cols]
            tier2_cols = [c for c in tier2_cols if c in train_cols]
            career_cols = [c for c in career_cols if c in train_cols]
            within_cols = [c for c in within_cols if c in train_cols]
            interaction_cols = [c for c in interaction_cols if c in train_cols]
    except Exception as exc:
        logger.warning("Failed training-surface alignment via unified table: %s", exc)

    logger.info(
        "Model inputs (fixed width): tier1=%d tier2=%d career=%d within=%d",
        len(tier1_cols), len(tier2_cols), len(career_cols), len(within_cols),
    )
    logger.info(
        "Available in inference table: tier1=%d tier2=%d career=%d within=%d",
        sum(1 for c in tier1_cols if c in inf.columns),
        sum(1 for c in tier2_cols if c in inf.columns),
        sum(1 for c in career_cols if c in inf.columns),
        sum(1 for c in within_cols if c in inf.columns),
    )

    def to_tensor(frame: pd.DataFrame, cols: list[str]) -> torch.Tensor:
        if not cols:
            return torch.zeros((len(frame), 0), dtype=torch.float32)
        arr = np.zeros((len(frame), len(cols)), dtype=np.float32)
        available = [c for c in cols if c in frame.columns]
        if available:
            vals = frame[available].to_numpy(dtype=np.float32, copy=False)
            vals = np.nan_to_num(vals, nan=0.0)
            col_to_idx = {c: i for i, c in enumerate(cols)}
            for j, c in enumerate(available):
                arr[:, col_to_idx[c]] = vals[:, j]
        return torch.from_numpy(arr)

    def to_tensor_and_mask(frame: pd.DataFrame, cols: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not cols:
            zeros = torch.zeros((len(frame), 0), dtype=torch.float32)
            return zeros, zeros
        arr = frame[cols].to_numpy(dtype=np.float32, copy=False)
        mask = (~np.isnan(arr)).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        return torch.from_numpy(arr), torch.from_numpy(mask)

    tier1 = to_tensor(inf, tier1_cols)
    tier2 = to_tensor(inf, tier2_cols)
    career_t = to_tensor(inf, career_cols)
    within_t = to_tensor(inf, within_cols)
    year1_t, year1_mask = to_tensor_and_mask(inf, [c for c in interaction_cols if c in inf.columns])
    if year1_t.shape[1] != len(interaction_cols):
        # Keep fixed width expected by the model.
        padded = torch.zeros((len(inf), len(interaction_cols)), dtype=torch.float32)
        padded_mask = torch.zeros((len(inf), len(interaction_cols)), dtype=torch.float32)
        col_to_idx = {c: i for i, c in enumerate([c for c in interaction_cols if c in inf.columns])}
        for j, c in enumerate(interaction_cols):
            if c in col_to_idx:
                src = col_to_idx[c]
                padded[:, j] = year1_t[:, src]
                padded_mask[:, j] = year1_mask[:, src]
        year1_t, year1_mask = padded, padded_mask

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

    # Checkpoints may be trained with different decoder conditioning settings.
    # Prefer explicit manifest if available; otherwise fall back to probing.
    model_cfg_path = model_path.parent / "model_config.json"
    condition_candidates = []
    model_cfg = {}
    if model_cfg_path.exists():
        try:
            model_cfg = json.loads(model_cfg_path.read_text(encoding="utf-8"))
            condition_candidates = [bool(model_cfg.get("condition_on_archetypes", False))]
            logger.info("Loaded model config from %s", model_cfg_path)
        except Exception as exc:
            logger.warning("Failed reading model config (%s): %s", model_cfg_path, exc)
    if not condition_candidates:
        condition_candidates = [False, True]

    latent_dim = int(model_cfg.get("latent_dim", 32))
    n_archetypes = int(model_cfg.get("n_archetypes", 8))
    use_vae = bool(model_cfg.get("use_vae", False))
    # Prefer config widths when present; fall back to resolved inference widths.
    cfg_tier1_dim = int(model_cfg.get("tier1_dim", len(tier1_cols)))
    cfg_tier2_dim = int(model_cfg.get("tier2_dim", len(tier2_cols)))
    cfg_career_dim = int(model_cfg.get("career_dim", len(career_cols)))
    cfg_within_dim = int(model_cfg.get("within_dim", len(within_cols)))
    cfg_year1_dim = int(model_cfg.get("year1_feature_dim", len(interaction_cols)))

    ckpt = torch.load(model_path, map_location="cpu")
    last_err = None
    model = None
    for cond_arch in condition_candidates:
        candidate = ProspectModel(
            latent_dim=latent_dim,
            n_archetypes=n_archetypes,
            use_vae=use_vae,
            predict_uncertainty=True,
            year1_feature_dim=cfg_year1_dim,
            condition_on_archetypes=cond_arch,
        )
        # Match encoder input widths to this run's resolved feature columns.
        candidate.encoder = PlayerEncoder(
            tier1_dim=cfg_tier1_dim,
            tier2_dim=cfg_tier2_dim,
            career_dim=cfg_career_dim,
            within_dim=cfg_within_dim,
            latent_dim=latent_dim,
            use_vae=use_vae,
        )
        try:
            candidate.load_state_dict(ckpt, strict=False)
            model = candidate
            logger.info("Loaded model with condition_on_archetypes=%s", cond_arch)
            break
        except RuntimeError as exc:
            last_err = exc
            continue

    if model is None:
        # Last-resort probe across known variants if manifest-guided attempt failed.
        for cond_arch in (False, True):
            candidate = ProspectModel(
                latent_dim=latent_dim,
                n_archetypes=n_archetypes,
                use_vae=use_vae,
                predict_uncertainty=True,
                year1_feature_dim=cfg_year1_dim,
                condition_on_archetypes=cond_arch,
            )
            candidate.encoder = PlayerEncoder(
                tier1_dim=cfg_tier1_dim,
                tier2_dim=cfg_tier2_dim,
                career_dim=cfg_career_dim,
                within_dim=cfg_within_dim,
                latent_dim=latent_dim,
                use_vae=use_vae,
            )
            try:
                candidate.load_state_dict(ckpt, strict=False)
                model = candidate
                logger.info("Loaded model with fallback condition_on_archetypes=%s", cond_arch)
                break
            except RuntimeError as exc:
                last_err = exc
                continue
    if model is None:
        raise RuntimeError(f"Failed to load checkpoint with supported architectures: {last_err}")

    model.eval()

    with torch.no_grad():
        out = model(tier1, tier2, career_t, within_t, tier2_mask, within_mask, year1_t, year1_mask)

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
        "college_games_played": pd.to_numeric(inf.get("college_games_played", np.nan), errors="coerce"),
        "college_poss_proxy": pd.to_numeric(inf.get("college_poss_proxy", np.nan), errors="coerce"),
        "college_minutes_total": pd.to_numeric(inf.get("college_minutes_total", np.nan), errors="coerce"),
        "pred_peak_rapm": out["rapm_pred"][:, 0].cpu().numpy(),
        "pred_gap_ts": out["gap_pred"][:, 0].cpu().numpy() if out["gap_pred"].shape[1] > 0 else np.nan,
        "pred_gap_usg": out["gap_pred"][:, 1].cpu().numpy() if out["gap_pred"].shape[1] > 1 else np.nan,
        "pred_year1_epm": out["epm_pred"][:, 0].cpu().numpy() if out["epm_pred"].shape[1] > 0 else np.nan,
        "pred_dev_rate": out["dev_pred"][:, 0].cpu().numpy() if "dev_pred" in out else np.nan,
        "pred_dev_rate_std": torch.sqrt(out["dev_var"][:, 0].clamp(min=1e-8)).cpu().numpy() if "dev_var" in out else np.nan,
        "pred_made_nba_logit": out["survival_logits"][:, 0].cpu().numpy(),
        "archetype_top_ids": top_ids,
        "archetype_top_probs": top_probs,
    })

    # Optional season-level recalibration (inflation/deflation correction).
    recal_path = Path(args.recalibration_path) if args.recalibration_path else model_path.parent / "season_recalibration.json"
    if recal_path.exists():
        try:
            recal = json.loads(recal_path.read_text(encoding="utf-8"))
            by_season = recal.get("offsets_by_season", {})
            global_offset = float(recal.get("global_offset", 0.0))
            season_offsets = []
            for s in preds["college_final_season"].values:
                key = str(int(s)) if pd.notna(s) else None
                if key and key in by_season:
                    season_offsets.append(float(by_season[key].get("offset", global_offset)))
                else:
                    season_offsets.append(global_offset)
            season_offsets = np.array(season_offsets, dtype=np.float32)
            preds["pred_peak_rapm_raw"] = preds["pred_peak_rapm"].values
            preds["pred_peak_rapm"] = preds["pred_peak_rapm"].values + season_offsets
            preds["pred_peak_rapm_recalibration_offset"] = season_offsets
            logger.info("Applied recalibration from %s", recal_path)
        except Exception as exc:
            logger.warning("Failed to apply recalibration from %s: %s", recal_path, exc)

    # Diagnostic season-normalized signals (kept for analysis only).
    try:
        seasons = pd.to_numeric(preds["college_final_season"], errors="coerce")
        usage = pd.to_numeric(inf.get("final_usage", np.nan), errors="coerce")
        ast_p100 = pd.to_numeric(inf.get("college_ast_total_per100poss", np.nan), errors="coerce")
        tov_p100 = pd.to_numeric(inf.get("college_tov_total_per100poss", np.nan), errors="coerce")
        stl_p100 = pd.to_numeric(inf.get("college_stl_total_per100poss", np.nan), errors="coerce")
        blk_p100 = pd.to_numeric(inf.get("college_blk_total_per100poss", np.nan), errors="coerce")
        rec_rating = pd.to_numeric(inf.get("college_recruiting_rating", np.nan), errors="coerce")
        rec_rank = pd.to_numeric(inf.get("college_recruiting_rank", np.nan), errors="coerce")

        # Recruiting signal fallback: rating preferred, else inverse-log rank.
        rec_signal = rec_rating.copy()
        fill_rank = (-np.log1p(rec_rank)).replace([np.inf, -np.inf], np.nan)
        rec_signal = rec_signal.where(rec_signal.notna(), fill_rank)

        usage_z = _season_z(usage, seasons)
        ast_z = _season_z(ast_p100, seasons)
        tov_z = _season_z(tov_p100, seasons)
        stl_z = _season_z(stl_p100, seasons)
        blk_z = _season_z(blk_p100, seasons)
        recruit_z = _season_z(rec_signal, seasons)

        preds["pred_rank_usage_z"] = usage_z
        preds["pred_rank_ast_z"] = ast_z
        preds["pred_rank_tov_z"] = tov_z
        preds["pred_rank_stl_z"] = stl_z
        preds["pred_rank_blk_z"] = blk_z
        preds["pred_rank_recruit_z"] = recruit_z

        # Compatibility columns expected by downstream exporters/audits.
        preds["pred_peak_rapm_reliability"] = preds["pred_peak_rapm"]
        preds["pred_rank_reliability"] = np.nan
        preds["pred_rank_volume_z"] = np.nan

        # Bayesian Gaussian rank score:
        # combine model mean/std with contextual evidence in a posterior update.
        train_path = BASE_DIR / "data/training/unified_training_table.parquet"
        learned_score = None
        if train_path.exists():
            train_df = pd.read_parquet(train_path)
            if "y_peak_ovr" in train_df.columns:
                # Re-run current model on training rows to get in-distribution model outputs.
                tr_t1 = to_tensor(train_df, tier1_cols)
                tr_t2 = to_tensor(train_df, tier2_cols)
                tr_c = to_tensor(train_df, career_cols)
                tr_w = to_tensor(train_df, within_cols)
                tr_mask = torch.from_numpy(
                    train_df["has_spatial_data"].to_numpy(dtype=np.float32)
                ).unsqueeze(1) if "has_spatial_data" in train_df.columns else torch.ones((len(train_df), 1), dtype=torch.float32)

                tr_ws_flags = []
                for c in ["final_has_ws_last5", "final_has_ws_last10", "final_has_ws_breakout_timing_eff"]:
                    if c in train_df.columns:
                        tr_ws_flags.append(train_df[c].fillna(0).to_numpy(dtype=np.float32))
                if tr_ws_flags:
                    tr_within_mask = torch.from_numpy(
                        (np.max(np.stack(tr_ws_flags, axis=0), axis=0) > 0).astype(np.float32)
                    ).unsqueeze(1)
                else:
                    tr_within_mask = torch.zeros((len(train_df), 1), dtype=torch.float32)

                tr_y1_cols = [c for c in interaction_cols if c in train_df.columns]
                tr_y1, tr_y1_mask = to_tensor_and_mask(train_df, tr_y1_cols)
                if tr_y1.shape[1] != len(interaction_cols):
                    pad = torch.zeros((len(train_df), len(interaction_cols)), dtype=torch.float32)
                    pad_mask = torch.zeros((len(train_df), len(interaction_cols)), dtype=torch.float32)
                    c2i = {c: i for i, c in enumerate(tr_y1_cols)}
                    for j, c in enumerate(interaction_cols):
                        if c in c2i:
                            src = c2i[c]
                            pad[:, j] = tr_y1[:, src]
                            pad_mask[:, j] = tr_y1_mask[:, src]
                    tr_y1, tr_y1_mask = pad, pad_mask

                with torch.no_grad():
                    tr_out = model(tr_t1, tr_t2, tr_c, tr_w, tr_mask, tr_within_mask, tr_y1, tr_y1_mask)

                tr_pred_peak = tr_out["rapm_pred"][:, 0].cpu().numpy()
                tr_pred_surv = _sigmoid(tr_out["survival_logits"][:, 0].cpu().numpy())
                tr_games = pd.to_numeric(train_df.get("college_games_played", np.nan), errors="coerce").fillna(0.0).to_numpy()
                tr_poss = pd.to_numeric(train_df.get("college_poss_proxy", np.nan), errors="coerce").fillna(0.0).to_numpy()
                tr_usage = pd.to_numeric(train_df.get("final_usage", np.nan), errors="coerce").fillna(0.0).to_numpy()
                tr_y = pd.to_numeric(train_df["y_peak_ovr"], errors="coerce").to_numpy()

                tr_pred_epm = tr_out["epm_pred"][:, 0].cpu().numpy()
                tr_pred_dev = tr_out["dev_pred"][:, 0].cpu().numpy() if "dev_pred" in tr_out else np.zeros_like(tr_pred_peak)
                tr_pred_sd = np.sqrt(np.clip(tr_out["rapm_var"][:, 0].cpu().numpy(), 1e-8, None)) if "rapm_var" in tr_out else np.full_like(tr_pred_peak, 1.0)
                tr_usage_z = _season_z(pd.Series(pd.to_numeric(train_df.get("final_usage", np.nan), errors="coerce")), pd.Series(pd.to_numeric(train_df.get("college_final_season", np.nan), errors="coerce"))).to_numpy(dtype=float)
                tr_ast_z = _season_z(pd.Series(pd.to_numeric(train_df.get("college_ast_total_per100poss", np.nan), errors="coerce")), pd.Series(pd.to_numeric(train_df.get("college_final_season", np.nan), errors="coerce"))).to_numpy(dtype=float)
                tr_tov_z = _season_z(pd.Series(pd.to_numeric(train_df.get("college_tov_total_per100poss", np.nan), errors="coerce")), pd.Series(pd.to_numeric(train_df.get("college_final_season", np.nan), errors="coerce"))).to_numpy(dtype=float)
                tr_stl_z = _season_z(pd.Series(pd.to_numeric(train_df.get("college_stl_total_per100poss", np.nan), errors="coerce")), pd.Series(pd.to_numeric(train_df.get("college_final_season", np.nan), errors="coerce"))).to_numpy(dtype=float)
                tr_blk_z = _season_z(pd.Series(pd.to_numeric(train_df.get("college_blk_total_per100poss", np.nan), errors="coerce")), pd.Series(pd.to_numeric(train_df.get("college_final_season", np.nan), errors="coerce"))).to_numpy(dtype=float)
                tr_rec_rating = pd.to_numeric(train_df.get("college_recruiting_rating", np.nan), errors="coerce")
                tr_rec_rank = pd.to_numeric(train_df.get("college_recruiting_rank", np.nan), errors="coerce")
                tr_rec_signal = tr_rec_rating.where(tr_rec_rating.notna(), -np.log1p(tr_rec_rank))
                tr_rec_z = _season_z(tr_rec_signal, pd.Series(pd.to_numeric(train_df.get("college_final_season", np.nan), errors="coerce"))).to_numpy(dtype=float)

                tr_evidence_X = np.column_stack([
                    tr_usage_z,
                    tr_rec_z,
                    tr_ast_z,
                    tr_tov_z,
                    tr_stl_z,
                    tr_blk_z,
                    tr_pred_surv,
                    tr_pred_epm,
                    tr_pred_dev,
                    tr_pred_peak,
                    tr_pred_peak * tr_pred_surv,
                    tr_usage_z * tr_rec_z,
                ])

                valid = np.isfinite(tr_y) & np.isfinite(tr_pred_peak) & np.isfinite(tr_pred_sd) & np.isfinite(tr_evidence_X).all(axis=1)
                if valid.sum() >= 50:
                    # Fully learned evidence model from latent outputs only.
                    # Sequential yearly Bayes update:
                    # season s is learned using only prior seasons (< s), which captures era drift.
                    tr_X = np.column_stack([
                        tr_pred_peak,
                        tr_pred_surv,
                        tr_pred_epm,
                        tr_pred_dev,
                        tr_pred_peak * tr_pred_surv,
                    ])
                    tr_season = pd.to_numeric(train_df.get("college_final_season", np.nan), errors="coerce").to_numpy(dtype=float)

                    inf_peak = preds["pred_peak_rapm"].to_numpy(dtype=float)
                    inf_surv = _sigmoid(preds["pred_made_nba_logit"].to_numpy(dtype=float))
                    inf_epm = preds["pred_year1_epm"].to_numpy(dtype=float)
                    inf_dev = preds["pred_dev_rate"].to_numpy(dtype=float)
                    inf_X = np.column_stack([
                        inf_peak,
                        inf_surv,
                        inf_epm,
                        inf_dev,
                        inf_peak * inf_surv,
                    ])
                    inf_season = pd.to_numeric(preds.get("college_final_season", np.nan), errors="coerce").to_numpy(dtype=float)

                    prior_sd_all = (
                        np.sqrt(np.clip(out["rapm_var"][:, 0].cpu().numpy(), 1e-8, None))
                        if "rapm_var" in out
                        else np.full(len(preds), 1.0, dtype=float)
                    )
                    post_mean = inf_peak.copy()
                    post_sd = prior_sd_all.copy()

                    min_hist = 120
                    seasons = np.sort(np.unique(inf_season[np.isfinite(inf_season)]))
                    for s in seasons:
                        idx = (inf_season == s)
                        hist = valid & np.isfinite(tr_season) & (tr_season < s)
                        if hist.sum() < min_hist:
                            continue
                        try:
                            from sklearn.linear_model import BayesianRidge
                            bayes = BayesianRidge(fit_intercept=True, max_iter=500, tol=1e-4, compute_score=False)
                            bayes.fit(tr_X[hist], tr_y[hist])
                            ev_mean, ev_sd = bayes.predict(inf_X[idx], return_std=True)
                        except Exception:
                            continue

                        prior_mean = inf_peak[idx]
                        prior_sd = prior_sd_all[idx]
                        obs_sd = np.clip(np.asarray(ev_sd, dtype=float), 0.25, 3.0)
                        p0 = 1.0 / np.clip(np.square(prior_sd), 1e-8, None)
                        p1 = 1.0 / np.clip(np.square(obs_sd), 1e-8, None)
                        post_mean[idx] = (p0 * prior_mean + p1 * np.asarray(ev_mean, dtype=float)) / (p0 + p1)
                        post_sd[idx] = np.sqrt(1.0 / (p0 + p1))

                    learned_score = post_mean - 0.10 * post_sd
                    preds["pred_peak_rapm_posterior_mean"] = post_mean
                    preds["pred_peak_rapm_posterior_sd"] = post_sd
                    preds["pred_rank_model"] = "bayes_latent_sequential_v1"
                    preds["pred_rank_model_n"] = int(valid.sum())

        if learned_score is None:
            learned_score = preds["pred_peak_rapm"].to_numpy(dtype=float)
            preds["pred_rank_model"] = "raw_peak_fallback"
            preds["pred_rank_model_n"] = 0

        preds["pred_peak_rapm_rank_score"] = learned_score
    except Exception as exc:
        logger.warning("Failed to compute rank diagnostics: %s", exc)
        preds["pred_peak_rapm_reliability"] = preds["pred_peak_rapm"]
        preds["pred_rank_reliability"] = np.nan
        preds["pred_rank_usage_z"] = np.nan
        preds["pred_rank_ast_z"] = np.nan
        preds["pred_rank_tov_z"] = np.nan
        preds["pred_rank_stl_z"] = np.nan
        preds["pred_rank_blk_z"] = np.nan
        preds["pred_rank_volume_z"] = np.nan
        preds["pred_rank_recruit_z"] = np.nan
        preds["pred_peak_rapm_rank_score"] = preds["pred_peak_rapm"]
        preds["pred_rank_model"] = "raw_peak_error_fallback"
        preds["pred_rank_model_n"] = 0

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = INFERENCE_DIR / f"prospect_predictions_{stamp}.parquet"
    preds.to_parquet(out_path, index=False)
    logger.info(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
