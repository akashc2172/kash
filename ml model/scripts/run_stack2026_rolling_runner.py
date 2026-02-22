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
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

import sys

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from train_2026_model import Stack2026Model, TARGET_COL  # noqa: E402

SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
FOUNDATION_PATH = BASE / "data" / "training" / "foundation_college_table.parquet"
MODEL_DIR = BASE / "models" / "stack2026"
ROLLING_MODEL_DIR = MODEL_DIR / "rolling"
AUDIT_DIR = BASE / "data" / "audit" / "rolling_yearly"
OUT_DIR = BASE / "data" / "inference" / "rolling_yearly"


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
) -> pd.DataFrame:
    id_cols = [c for c in id_cols_priority if c in df.columns]
    out = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    out["pred_mu"] = mu
    out["pred_pathway_integrated_mu"] = mu
    out["pred_sd"] = sd
    out["pred_upside"] = out["pred_mu"] + out["pred_sd"]
    out["pred_floor"] = out["pred_mu"] - out["pred_sd"]
    out["pred_rank"] = out["pred_mu"].rank(ascending=False, method="min").astype(int)
    out["pred_rank_top_weighted"] = out["pred_rank"]
    if target_col in df.columns:
        out["actual_target"] = pd.to_numeric(df[target_col], errors="coerce")
        non_null = out["actual_target"].notna()
        out["actual_rank"] = np.nan
        if non_null.any():
            out.loc[non_null, "actual_rank"] = (
                out.loc[non_null, "actual_target"]
                .rank(ascending=False, method="min")
                .astype(int)
            )
        out["rank_error"] = np.where(
            out["actual_rank"].notna(),
            out["pred_rank"] - out["actual_rank"],
            np.nan,
        )
    return out.sort_values("pred_rank")


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


def run(start_year: int, end_year: int, epochs: int, lr: float) -> None:
    ROLLING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

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
        y = pd.to_numeric(train_df[TARGET_COL], errors="coerce")
        mature = pd.to_numeric(train_df.get("is_epm_mature"), errors="coerce").fillna(0).astype(int)
        keep = y.notna() & (mature == 1)
        if keep.sum() < 100:
            keep = y.notna()
        train_df = train_df.loc[keep].copy()
        if len(train_df) < 100:
            continue

        x_train = _as_numeric_matrix(train_df, feat_cols)
        x_train = (x_train - means) / stds
        y_raw = pd.to_numeric(train_df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        y_mean = float(np.nanmean(y_raw))
        y_std = float(np.nanstd(y_raw))
        if y_std < 1e-6:
            y_std = 1.0
        y_train = ((y_raw - y_mean) / y_std).reshape(-1, 1)
        ctx_train = _extract_context(train_df, context_cols)

        model = Stack2026Model(input_dim=input_dim, use_hypernetwork=len(context_cols) > 0)
        prev_ckpt = torch.load(prev_ckpt_path, weights_only=False)
        model.load_state_dict(prev_ckpt["model_state_dict"])
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        x_t = torch.FloatTensor(x_train)
        y_t = torch.FloatTensor(y_train)
        c_t = torch.FloatTensor(ctx_train) if len(context_cols) > 0 else None
        for _ in range(epochs):
            mu, logvar = model(x_t, c_t)
            logvar = torch.clamp(logvar, -4.0, 4.0)
            precision = torch.exp(-logvar)
            loss = 0.5 * (precision * (y_t - mu) ** 2 + logvar).mean()
            optimizer.zero_grad()
            loss.backward()
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
                "year": int(year),
                "train_rows": int(len(train_df)),
            },
            year_model_path,
        )
        prev_ckpt_path = year_model_path

        model.eval()
        with torch.no_grad():
            nba_df = supervised[supervised["draft_year_proxy"] == year].copy()
            all_df = foundation[pd.to_numeric(foundation["season"], errors="coerce") == year].copy()

            x_nba = _as_numeric_matrix(nba_df, feat_cols) if len(nba_df) else np.zeros((0, len(feat_cols)), dtype=np.float32)
            x_all = _as_numeric_matrix(all_df, feat_cols) if len(all_df) else np.zeros((0, len(feat_cols)), dtype=np.float32)
            x_nba = (x_nba - means) / stds if len(x_nba) else x_nba
            x_all = (x_all - means) / stds if len(x_all) else x_all

            ctx_nba = _extract_context(nba_df, context_cols) if len(nba_df) else np.zeros((0, 3), dtype=np.float32)
            ctx_all = _extract_context(all_df, context_cols) if len(all_df) else np.zeros((0, 3), dtype=np.float32)

            if len(nba_df):
                mu_nba, lv_nba = model(torch.FloatTensor(x_nba), torch.FloatTensor(ctx_nba) if len(context_cols) > 0 else None)
                pred_mu_nba = (mu_nba.numpy().flatten() * y_std) + y_mean
                pred_sd_nba = np.exp(0.5 * lv_nba.numpy().flatten()) * y_std
            else:
                pred_mu_nba = np.array([])
                pred_sd_nba = np.array([])

            if len(all_df):
                mu_all, lv_all = model(torch.FloatTensor(x_all), torch.FloatTensor(ctx_all) if len(context_cols) > 0 else None)
                pred_mu_all = (mu_all.numpy().flatten() * y_std) + y_mean
                pred_sd_all = np.exp(0.5 * lv_all.numpy().flatten()) * y_std
            else:
                pred_mu_all = np.array([])
                pred_sd_all = np.array([])

        id_cols = ["player_name", "nba_id", "athlete_id", "bbr_id", "draft_year_proxy", "college_final_season", "season"]
        ranked_nba = _rank_export(nba_df, pred_mu_nba, pred_sd_nba, TARGET_COL, id_cols)
        ranked_all = _rank_export(all_df, pred_mu_all, pred_sd_all, TARGET_COL, id_cols)

        year_dir = OUT_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        nba_csv = year_dir / f"rankings_{year}_nba_mapped.csv"
        all_csv = year_dir / f"rankings_{year}_all_college.csv"
        ranked_nba.to_csv(nba_csv, index=False)
        ranked_all.to_csv(all_csv, index=False)
        with pd.ExcelWriter(year_dir / f"rankings_{year}.xlsx", engine="openpyxl") as writer:
            ranked_nba.head(250).to_excel(writer, sheet_name="nba_mapped", index=False)
            ranked_all.head(250).to_excel(writer, sheet_name="all_college", index=False)

        gate = _year_gate(year, ranked_nba, ranked_all)
        gate["model_path"] = str(year_model_path)
        gate["nba_csv"] = str(nba_csv)
        gate["all_csv"] = str(all_csv)
        with open(AUDIT_DIR / f"gate_{year}.json", "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
        summary_rows.append(gate)

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("year").to_csv(AUDIT_DIR / "rolling_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential per-year stack2026 runner")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    run(args.start_year, args.end_year, args.epochs, args.lr)


if __name__ == "__main__":
    main()
