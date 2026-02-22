#!/usr/bin/env python3
"""
Sequential yearly runner for Stack 2026.

Writes per-year artifacts:
- models/stack2026/rolling/model_{year}.pt
- data/inference/rolling_yearly/{year}/rankings_{year}_{cohort}.csv
- data/inference/rolling_yearly/{year}/rankings_{year}.xlsx
- data/audit/rolling_yearly/gate_{year}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

import sys

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from train_2026_model import (  # noqa: E402
    LAMBDA_3Y,
    LAMBDA_RANK,
    LAMBDA_TRAJ,
    N_TARGETS_TRAIN,
    TARGET_COL,
    TARGET_COL_3Y,
    TARGET_COL_TRAJ,
    TARGET_COL_WINDOW_FALLBACK,
    Stack2026Model,
    heteroscedastic_loss,
)

SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
FOUNDATION_PATH = BASE / "data" / "training" / "foundation_college_table.parquet"
MODEL_DIR = BASE / "models" / "stack2026"
ROLLING_MODEL_DIR = MODEL_DIR / "rolling"
DEFAULT_OUTPUT_SUBDIR = "rolling_yearly"


def _as_numeric_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    frame = df[cols].copy()
    for c in cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.fillna(0.0).to_numpy(dtype=np.float32)


def _extract_context(df: pd.DataFrame, context_cols: List[str]) -> np.ndarray:
    if not context_cols:
        return np.zeros((len(df), 3), dtype=np.float32)
    cols = [c for c in context_cols if c in df.columns]
    if not cols:
        return np.zeros((len(df), 3), dtype=np.float32)
    frame = df[cols].copy()
    for c in cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    ctx = frame.fillna(0.0).to_numpy(dtype=np.float32)
    if ctx.shape[1] < 3:
        ctx = np.pad(ctx, ((0, 0), (0, 3 - ctx.shape[1])))
    return ctx


def _rank_export(
    df: pd.DataFrame,
    mu: np.ndarray,
    sd: np.ndarray,
    target_col: str,
    id_cols_priority: List[str],
    extra_cols: List[str] | None = None,
) -> pd.DataFrame:
    id_cols = [c for c in id_cols_priority if c in df.columns]
    base = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    pred_mu = np.asarray(mu, dtype=np.float64)
    pred_sd = np.asarray(sd, dtype=np.float64)
    pred_rank = pd.Series(pred_mu).rank(ascending=False, method="min").astype(int)
    actual_target = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series(np.nan, index=df.index)
    actual_rank = pd.Series(np.nan, index=df.index)
    if actual_target.notna().any():
        actual_rank = actual_target.rank(ascending=False, method="min").astype(float)
        actual_rank = actual_rank.where(actual_target.notna(), np.nan)
    rank_error = np.where(actual_rank.notna(), pred_rank.values - actual_rank.values, np.nan)
    parts = [
        base,
        pd.DataFrame({
            "pred_mu": pred_mu,
            "pred_pathway_integrated_mu": pred_mu,
            "pred_sd": pred_sd,
            "pred_upside": pred_mu + pred_sd,
            "pred_floor": pred_mu - pred_sd,
            "pred_rank": pred_rank.values,
            "pred_rank_top_weighted": pred_rank.values,
            "actual_target": actual_target.values,
            "actual_rank": actual_rank.values,
            "rank_error": rank_error,
        }, index=df.index),
    ]
    if extra_cols:
        add_cols = [c for c in extra_cols if c in df.columns and c not in base.columns]
        if add_cols:
            parts.append(df[add_cols].copy())
    out = pd.concat(parts, axis=1).copy()
    return out.sort_values("pred_rank")


REPLAY_SIZE = 500  # diverse past examples for continual learning
REPLAY_STRATIFY_COL = "draft_year_proxy"
EWC_LAMBDA = 0.01  # L2 penalty toward previous year's weights (mild consolidation)
SEASON_CARD_TOP_MISSES = 20


def _run_year_diagnostics(
    year: int,
    ranked_nba: pd.DataFrame,
    supervised: pd.DataFrame,
    feat_cols: List[str],
) -> Dict:
    """Build season card: top archetype misses, calibration by decile, drift stats."""
    card = {"year": year}
    has_actual = "actual_target" in ranked_nba.columns and ranked_nba["actual_target"].notna().any()
    if has_actual:
        ranked_nba = ranked_nba.copy()
        ranked_nba["_err_abs"] = ranked_nba["rank_error"].abs()
        top_misses = ranked_nba.nlargest(SEASON_CARD_TOP_MISSES, "_err_abs", keep="first").drop(columns=["_err_abs"], errors="ignore")
        cols_miss = [c for c in ["player_name", "pred_mu", "actual_target", "pred_rank", "actual_rank", "rank_error"] if c in top_misses]
        rec = top_misses[cols_miss].replace({np.nan: None}).to_dict(orient="records")
        card["top_archetype_misses"] = rec
        # Calibration by decile (pred_mu binned)
        deciles = pd.qcut(ranked_nba["pred_mu"].rank(method="first"), 10, labels=False, duplicates="drop")
        cal = ranked_nba.assign(decile=deciles).groupby("decile").agg(
            mean_pred=("pred_mu", "mean"),
            mean_actual=("actual_target", "mean"),
            count=("pred_mu", "count"),
        ).reset_index()
        card["calibration_by_decile"] = cal.to_dict(orient="records")
    else:
        card["top_archetype_misses"] = []
        card["calibration_by_decile"] = []
    # Drift: feature means by cohort (draft_year_proxy) for this year's training cohort
    cohort_df = supervised[supervised["draft_year_proxy"] <= year]
    if REPLAY_STRATIFY_COL in cohort_df.columns and len(cohort_df) > 0:
        drift_cols = [c for c in feat_cols if c in cohort_df.columns][:20]
        if drift_cols:
            by_cohort = cohort_df.groupby(REPLAY_STRATIFY_COL)[drift_cols].mean()
            card["drift_feature_means_by_cohort"] = by_cohort.to_dict(orient="index")
    return card


def _year_gate(
    year: int,
    ranked_nba: pd.DataFrame,
    ranked_all: pd.DataFrame,
) -> Dict:
    std = float(ranked_all["pred_mu"].std()) if len(ranked_all) else 0.0
    iqr = float(ranked_all["pred_mu"].quantile(0.75) - ranked_all["pred_mu"].quantile(0.25)) if len(ranked_all) else 0.0
    has_actual = ranked_nba["actual_target"].notna() if "actual_target" in ranked_nba.columns else pd.Series([], dtype=bool)
    metrics = {
        "year": int(year),
        "nba_rows": int(len(ranked_nba)),
        "all_college_rows": int(len(ranked_all)),
        "score_std": std,
        "score_iqr": iqr,
        "std_gate": bool(std >= 0.25),
        "iqr_gate": bool(iqr >= 0.20),
        "actual_rate_nba": float(has_actual.mean()) if len(has_actual) else 0.0,
        "rmse_nba": None,
        "spearman_nba": None,
    }
    if "actual_target" in ranked_nba.columns and has_actual.sum() >= 5:
        y = ranked_nba.loc[has_actual, "actual_target"].to_numpy(dtype=float)
        p = ranked_nba.loc[has_actual, "pred_mu"].to_numpy(dtype=float)
        metrics["rmse_nba"] = float(np.sqrt(mean_squared_error(y, p)))
        metrics["spearman_nba"] = float(spearmanr(y, p)[0])
    metrics["publish_approved"] = bool(metrics["std_gate"] and metrics["iqr_gate"] and metrics["all_college_rows"] > 0)
    return metrics


def run(start_year: int, end_year: int, epochs: int, lr: float, output_subdir: str = DEFAULT_OUTPUT_SUBDIR) -> None:
    out_dir = BASE / "data" / "inference" / output_subdir
    audit_dir = BASE / "data" / "audit" / output_subdir
    ROLLING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    pretrained = torch.load(MODEL_DIR / "pretrained_encoder.pt", weights_only=False)
    base_supervised = torch.load(MODEL_DIR / "stack2026_supervised.pt", weights_only=False)

    feat_cols = pretrained["feature_cols"]
    means = np.array(pretrained["means"], dtype=np.float32)
    stds = np.array(pretrained["stds"], dtype=np.float32)
    input_dim = int(pretrained["input_dim"])
    context_cols = base_supervised.get("context_cols", [])

    supervised = pd.read_parquet(SUPERVISED_PATH)
    foundation = pd.read_parquet(FOUNDATION_PATH)

    if "draft_year_proxy" not in supervised.columns:
        raise ValueError("supervised table missing draft_year_proxy")
    if "season" not in foundation.columns:
        raise ValueError("foundation table missing season")

    for c in feat_cols:
        if c not in supervised.columns:
            supervised[c] = 0.0
        if c not in foundation.columns:
            foundation[c] = 0.0

    prev_ckpt_path = MODEL_DIR / "stack2026_supervised.pt"
    summary_rows = []

    for year in range(start_year, end_year + 1):
        train_df = supervised[supervised["draft_year_proxy"] <= (year - 1)].copy()
        primary_col = TARGET_COL if TARGET_COL in train_df.columns else TARGET_COL_WINDOW_FALLBACK
        y = pd.to_numeric(train_df[primary_col], errors="coerce")
        mature = pd.to_numeric(train_df.get("is_epm_mature"), errors="coerce").fillna(0).astype(int)
        keep = y.notna() & (mature == 1)
        if keep.sum() < 100:
            keep = y.notna()
        train_df = train_df.loc[keep].copy()
        if len(train_df) < 100:
            continue

        # Replay buffer: diverse past examples (draft_year_proxy <= year-2) to reduce forgetting
        replay_df = pd.DataFrame()
        if year - 2 >= start_year and REPLAY_STRATIFY_COL in supervised.columns:
            past = supervised[supervised["draft_year_proxy"] <= (year - 2)].copy()
            y_past = pd.to_numeric(past[primary_col], errors="coerce") if primary_col in past.columns else pd.Series(dtype=float)
            keep_past = y_past.notna() if len(y_past) else pd.Series([], dtype=bool)
            past = past.loc[keep_past] if keep_past.any() else pd.DataFrame()
            if len(past) > 0:
                n_replay = min(REPLAY_SIZE, len(past))
                try:
                    n_per = max(1, n_replay // max(1, past[REPLAY_STRATIFY_COL].nunique()))
                    replay_df = past.groupby(REPLAY_STRATIFY_COL, group_keys=True).apply(
                        lambda g: g.sample(n=min(len(g), n_per), replace=False), include_groups=False
                    ).reset_index(level=0).sample(n=min(n_replay, len(past)), replace=False)
                except Exception:
                    replay_df = past.sample(n=n_replay, replace=False)
        if len(replay_df) > 0:
            train_df = pd.concat([train_df, replay_df], ignore_index=True)

        x_train = _as_numeric_matrix(train_df, feat_cols)
        x_train = (x_train - means) / stds
        ctx_train = _extract_context(train_df, context_cols)

        # Multi-head targets (primary + trajectory + 3y) aligned with Phase B for ranking/trajectory signal
        t_primary = pd.to_numeric(train_df[primary_col], errors="coerce") if primary_col in train_df.columns else pd.Series(np.nan, index=train_df.index)
        t_3y = pd.to_numeric(train_df[TARGET_COL_3Y], errors="coerce") if TARGET_COL_3Y in train_df.columns else pd.Series(np.nan, index=train_df.index)
        t_traj = pd.to_numeric(train_df[TARGET_COL_TRAJ], errors="coerce") if TARGET_COL_TRAJ in train_df.columns else pd.Series(np.nan, index=train_df.index)
        y_raw_1 = t_primary.to_numpy(dtype=np.float32)
        y_raw_3y = t_3y.to_numpy(dtype=np.float32)
        y_raw_traj = t_traj.to_numpy(dtype=np.float32)
        y_mean = float(np.nanmean(y_raw_1))
        y_std = float(np.nanstd(y_raw_1))
        if y_std < 1e-6:
            y_std = 1.0
        y_norm_1 = (y_raw_1 - y_mean) / y_std
        fin_3y = np.isfinite(y_raw_3y)
        y_mean_3y = float(np.nanmean(y_raw_3y[fin_3y])) if fin_3y.any() else 0.0
        y_std_3y = float(np.nanstd(y_raw_3y[fin_3y])) if fin_3y.sum() > 1 else 1.0
        if y_std_3y < 1e-6:
            y_std_3y = 1.0
        y_norm_3y = np.where(fin_3y, (y_raw_3y - y_mean_3y) / y_std_3y, 0.0)
        mask_3y = fin_3y.astype(np.float32)
        fin_traj = np.isfinite(y_raw_traj)
        y_mean_traj = float(np.nanmean(y_raw_traj[fin_traj])) if fin_traj.any() else 0.0
        y_std_traj = float(np.nanstd(y_raw_traj[fin_traj])) if fin_traj.sum() > 1 else 1.0
        if y_std_traj < 1e-6:
            y_std_traj = 1.0
        y_norm_traj = np.where(fin_traj, (y_raw_traj - y_mean_traj) / y_std_traj, 0.0)
        mask_traj = fin_traj.astype(np.float32)
        y_fill = np.stack([y_norm_1, y_norm_traj, mask_traj, y_norm_3y, mask_3y], axis=1)
        y_t = torch.FloatTensor(y_fill)

        prev_ckpt = torch.load(prev_ckpt_path, weights_only=False) if prev_ckpt_path.exists() else {}
        n_targets = max(1, int(prev_ckpt.get("n_targets", N_TARGETS_TRAIN)))
        model = Stack2026Model(input_dim=input_dim, use_hypernetwork=len(context_cols) > 0, n_targets=n_targets)
        if prev_ckpt:
            model.load_state_dict(prev_ckpt["model_state_dict"])
        prev_params = {n: p.clone().detach() for n, p in model.named_parameters()} if prev_ckpt and EWC_LAMBDA > 0 else {}

        # Freeze encoder for first 20 epochs (Phase B alignment)
        for param in model.autoencoder.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        x_t = torch.FloatTensor(x_train)
        c_t = torch.FloatTensor(ctx_train) if len(context_cols) > 0 else None

        for epoch in range(epochs):
            if epoch == 20:
                for param in model.autoencoder.encoder.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            model.train()
            mu, logvar = model(x_t, c_t)
            logvar = torch.clamp(logvar, -4.0, 4.0)
            nll1 = heteroscedastic_loss(mu[:, 0:1], logvar[:, 0:1], y_t[:, 0:1])
            loss = nll1
            mask_traj_b = y_t[:, 2] > 0.5
            if mask_traj_b.any() and mu.shape[1] > 1:
                nll_traj = heteroscedastic_loss(
                    mu[:, 1:2][mask_traj_b], logvar[:, 1:2][mask_traj_b], y_t[:, 1:2][mask_traj_b]
                )
                loss = loss + LAMBDA_TRAJ * nll_traj
            mask_3y_b = y_t[:, 4] > 0.5
            if mask_3y_b.any() and mu.shape[1] > 2:
                nll3y = heteroscedastic_loss(
                    mu[:, 2:3][mask_3y_b], logvar[:, 2:3][mask_3y_b], y_t[:, 3:4][mask_3y_b]
                )
                loss = loss + LAMBDA_3Y * nll3y
            if LAMBDA_RANK > 0 and mu.shape[1] > 0:
                y_prim = y_t[:, 0]
                mu_prim = mu[:, 0]
                n_ = mu_prim.size(0)
                idx = torch.randperm(n_, device=mu_prim.device)
                half = n_ // 2
                i, j = idx[:half], idx[half : half * 2]
                if i.numel() > 0 and j.numel() > 0:
                    margin = 0.1
                    rank_loss = F.relu(margin - (mu_prim[i] - mu_prim[j]) * (y_prim[i] - y_prim[j]).sign()).mean()
                    loss = loss + LAMBDA_RANK * rank_loss
            if prev_params and EWC_LAMBDA > 0:
                ewc = 0.0
                for n, p in model.named_parameters():
                    if n in prev_params:
                        ewc = ewc + (p - prev_params[n]).pow(2).sum()
                loss = loss + EWC_LAMBDA * 0.5 * ewc
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        year_model_path = ROLLING_MODEL_DIR / f"model_{year}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "feature_cols": feat_cols,
                "means": means.tolist(),
                "stds": stds.tolist(),
                "context_cols": context_cols,
                "target_col": TARGET_COL,
                "n_targets": n_targets,
                "year": int(year),
                "train_rows": int(len(train_df)),
                "y_mean": float(y_mean),
                "y_std": float(y_std),
                "y_mean_traj": float(y_mean_traj),
                "y_std_traj": float(y_std_traj),
                "y_mean_3y": float(y_mean_3y),
                "y_std_3y": float(y_std_3y),
            },
            year_model_path,
        )
        prev_ckpt_path = year_model_path

        # Single cohort = full college board (foundation for this year). Merge in target from
        # supervised so NBA-linked players get actual_target/actual_rank/rank_error. Then
        # nba_mapped = same list filtered to rows with nba_id (same order, same pred_rank).
        model.eval()
        all_df = foundation[pd.to_numeric(foundation["season"], errors="coerce") == year].copy()
        if len(all_df) == 0:
            continue
        # Enrich with target from supervised so we have actual_* where available
        sup_year = supervised[supervised["draft_year_proxy"] == year]
        if len(sup_year) and "athlete_id" in all_df.columns and "athlete_id" in sup_year.columns:
            tgt = sup_year[["athlete_id", TARGET_COL]].drop_duplicates(subset=["athlete_id"], keep="first")
            all_df = all_df.merge(tgt, on="athlete_id", how="left", suffixes=("", "_sup"))
            if f"{TARGET_COL}_sup" in all_df.columns:
                all_df[TARGET_COL] = all_df[TARGET_COL].fillna(all_df[f"{TARGET_COL}_sup"])
                all_df = all_df.drop(columns=[f"{TARGET_COL}_sup"])

        with torch.no_grad():
            x_all = _as_numeric_matrix(all_df, feat_cols)
            x_all = (x_all - means) / stds
            ctx_all = _extract_context(all_df, context_cols)
            mu_all, lv_all = model(torch.FloatTensor(x_all), torch.FloatTensor(ctx_all) if len(context_cols) > 0 else None)
            mu_primary = mu_all[:, 0] if mu_all.dim() > 1 and mu_all.shape[1] > 1 else mu_all.flatten()
            lv_primary = lv_all[:, 0] if lv_all.dim() > 1 and lv_all.shape[1] > 1 else lv_all.flatten()
            pred_mu_all = (mu_primary.numpy() * y_std) + y_mean
            pred_sd_all = np.exp(0.5 * lv_primary.numpy()) * y_std

        id_cols = ["player_name", "nba_id", "athlete_id", "bbr_id", "draft_year_proxy", "college_final_season", "season"]
        input_review_cols = [c for c in (feat_cols + context_cols) if c in all_df.columns]
        ranked_all = _rank_export(all_df, pred_mu_all, pred_sd_all, TARGET_COL, id_cols, extra_cols=input_review_cols)
        # NBA-mapped = same board order, only rows with nba_id (so pred_rank is global rank on full board)
        if "nba_id" in ranked_all.columns:
            has_nba = pd.to_numeric(ranked_all["nba_id"], errors="coerce").notna()
        else:
            has_nba = pd.Series(False, index=ranked_all.index)
        ranked_nba = ranked_all.loc[has_nba].copy()
        # Rank within NBA cohort only for comparable eval (pred_rank stays global)
        ranked_nba["pred_rank_nba_only"] = ranked_nba["pred_mu"].rank(ascending=False, method="min").astype(int)

        year_dir = out_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        nba_csv = year_dir / f"rankings_{year}_nba_mapped.csv"
        all_csv = year_dir / f"rankings_{year}_all_college.csv"
        ranked_nba.to_csv(nba_csv, index=False)
        ranked_all.to_csv(all_csv, index=False)
        with pd.ExcelWriter(year_dir / f"rankings_{year}.xlsx", engine="openpyxl") as writer:
            writer.book.properties.title = f"Rankings {year}"
            ranked_nba.head(250).to_excel(writer, sheet_name="nba_mapped", index=False)
            ranked_all.head(250).to_excel(writer, sheet_name="all_college", index=False)

        gate = _year_gate(year, ranked_nba, ranked_all)
        gate["model_path"] = str(year_model_path)
        gate["nba_csv"] = str(nba_csv)
        gate["all_csv"] = str(all_csv)
        with open(audit_dir / f"gate_{year}.json", "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
        summary_rows.append(gate)

        # Season card: top misses, calibration, drift (for "pause and learn")
        year_audit_dir = audit_dir / str(year)
        year_audit_dir.mkdir(parents=True, exist_ok=True)
        season_card = _run_year_diagnostics(year, ranked_nba, supervised, feat_cols)
        with open(year_audit_dir / "season_card.json", "w", encoding="utf-8") as f:
            json.dump(season_card, f, indent=2)

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("year").to_csv(audit_dir / "rolling_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential per-year stack2026 runner")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdir under data/inference and data/audit (e.g. rolling_yearly or other_rolling)",
    )
    args = parser.parse_args()
    run(args.start_year, args.end_year, args.epochs, args.lr, args.output_subdir)


if __name__ == "__main__":
    main()
