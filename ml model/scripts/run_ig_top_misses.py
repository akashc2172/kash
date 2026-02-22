#!/usr/bin/env python3
"""
Integrated Gradients (captum) + top-misses report for Stack 2026.

For each season run (or one-off), computes IG attributions for the top rank-error
players and writes a "top misses" report to data/audit/rolling_yearly/{year}/
(or data/audit/ when --year not set). Optionally clusters explanation vectors
into failure modes.

Usage:
  python run_ig_top_misses.py --year 2023
  python run_ig_top_misses.py --rankings-csv data/inference/rolling_yearly/2023/rankings_2023_nba_mapped.csv --year 2023
  python run_ig_top_misses.py  # one-off using main checkpoint + supervised table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from train_2026_model import Stack2026Model, TARGET_COL, get_feature_columns  # noqa: E402

SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
MODEL_DIR = BASE / "models" / "stack2026"
ROLLING_MODEL_DIR = MODEL_DIR / "rolling"
AUDIT_DIR = BASE / "data" / "audit" / "rolling_yearly"
DEFAULT_TOP_N = 30
IG_N_STEPS = 50


def _as_numeric_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    frame = df[[c for c in cols if c in df.columns]].copy()
    for c in cols:
        if c not in frame.columns:
            frame[c] = 0.0
    for c in cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame[cols].fillna(0.0).to_numpy(dtype=np.float32)


def _extract_context(df: pd.DataFrame, context_cols: list) -> np.ndarray:
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


def _load_model_and_data(
    ckpt_path: Path,
    table_path: Path,
    draft_year: int | None,
) -> tuple[Stack2026Model, pd.DataFrame, np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    df = pd.read_parquet(table_path)
    if draft_year is not None and "draft_year_proxy" in df.columns:
        df = df[pd.to_numeric(df["draft_year_proxy"], errors="coerce") == draft_year].copy()
    df = df.reset_index(drop=True)

    ckpt = torch.load(ckpt_path, weights_only=False)
    feature_cols = ckpt.get("feature_cols")
    if not feature_cols:
        raise ValueError("Checkpoint missing feature_cols")
    means = np.array(ckpt["means"], dtype=np.float32)
    stds = np.array(ckpt["stds"], dtype=np.float32)
    context_cols = ckpt.get("context_cols", [])

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = _as_numeric_matrix(df, feature_cols)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    X_norm = (X - means) / stds_safe
    ctx = _extract_context(df, context_cols)

    input_dim = X_norm.shape[1]
    n_targets = int(ckpt.get("n_targets", 1))
    model = Stack2026Model(
        input_dim=input_dim,
        use_hypernetwork=len(context_cols) > 0,
        n_targets=max(1, n_targets),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, df, X_norm, X, feature_cols, ctx, (means, stds_safe)


def _run_ig_top_misses(
    year: int | None,
    rankings_csv: Path | None,
    table_path: Path,
    ckpt_path: Path,
    top_n: int,
    out_dir: Path,
) -> dict:
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ImportError("Install captum: pip install captum")

    draft_year = year
    model, df, X_norm, X_raw, feature_cols, ctx, _ = _load_model_and_data(ckpt_path, table_path, draft_year)

    # If we have a rankings CSV with rank_error, use it; else compute pred and rank_error from model
    if rankings_csv and rankings_csv.exists():
        rank_df = pd.read_csv(rankings_csv)
        if "rank_error" not in rank_df.columns or "player_name" not in rank_df.columns:
            id_col = "athlete_id" if "athlete_id" in rank_df.columns else "nba_id"
            merge_on = [c for c in ["athlete_id", "nba_id", "player_name", "draft_year_proxy"] if c in rank_df.columns and c in df.columns]
            if merge_on:
                df = df.merge(rank_df[merge_on + ["pred_mu", "actual_target", "pred_rank", "actual_rank", "rank_error"]], on=merge_on, how="inner", suffixes=("", "_r"))
            else:
                rank_df = None
        else:
            # Match by row order if same cohort
            rank_df = rank_df.copy()
    else:
        rank_df = None

    # Compute predictions and rank_error from model if not from CSV
    with torch.no_grad():
        x_t = torch.FloatTensor(X_norm)
        ctx_t = torch.FloatTensor(ctx) if ctx.size else None
        mu, _ = model(x_t, ctx_t)
        mu_primary = mu[:, 0].numpy() if mu.dim() > 1 else mu.numpy().flatten()
    df["pred_mu"] = mu_primary
    if TARGET_COL in df.columns:
        df["actual_target"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        valid = df["actual_target"].notna()
        df["actual_rank"] = np.nan
        if valid.any():
            df.loc[valid, "actual_rank"] = df.loc[valid, "actual_target"].rank(ascending=False, method="min").astype(int)
        df["pred_rank"] = df["pred_mu"].rank(ascending=False, method="min").astype(int)
        df["rank_error"] = np.where(valid, df["pred_rank"] - df["actual_rank"], np.nan)
    else:
        df["rank_error"] = np.nan

    # Top misses by |rank_error| (worst rank error)
    has_error = df["rank_error"].notna()
    if not has_error.any():
        report = {"year": year, "n_rows": len(df), "top_misses": [], "message": "No rank_error (missing actual_target)."}
        return report
    err_abs = df["rank_error"].abs()
    top_idx = err_abs.nlargest(min(top_n, has_error.sum())).index.tolist()

    # Wrapper for captum: single scalar output (primary head); accepts (x,) or (x, ctx)
    def forward_fn(*args) -> torch.Tensor:
        x_tensor = args[0]
        ctx_tensor = args[1] if len(args) > 1 else None
        mu, _ = model(x_tensor, ctx_tensor)
        return mu[:, 0] if mu.dim() > 1 else mu.squeeze(-1)

    ig = IntegratedGradients(forward_fn)

    results = []
    for idx in top_idx:
        i = df.index.get_loc(idx)
        xi = torch.FloatTensor(X_norm[i : i + 1]).requires_grad_(True)
        ci = torch.FloatTensor(ctx[i : i + 1]) if ctx.size else None
        baseline_x = torch.zeros_like(xi)
        baseline_c = torch.zeros_like(ci) if ci is not None else None

        inputs = (xi, ci) if ci is not None else (xi,)
        baselines = (baseline_x, baseline_c) if baseline_c is not None else (baseline_x,)
        attr = ig.attribute(inputs=inputs, baselines=baselines, n_steps=IG_N_STEPS)
        attr_x = attr[0].detach().numpy().flatten() if isinstance(attr, tuple) else attr.detach().numpy().flatten()

        order = np.argsort(np.abs(attr_x))[::-1]
        top5 = [(feature_cols[k], float(attr_x[k]), float(X_raw[i, k])) for k in order[:5]]
        row = df.loc[idx]
        results.append({
            "player_name": str(row.get("player_name", row.get("norm_name", "?"))),
            "athlete_id": int(row["athlete_id"]) if pd.notna(row.get("athlete_id")) else None,
            "pred_mu": float(row["pred_mu"]),
            "actual_target": float(row["actual_target"]) if pd.notna(row.get("actual_target")) else None,
            "pred_rank": int(row["pred_rank"]),
            "actual_rank": int(row["actual_rank"]) if pd.notna(row.get("actual_rank")) else None,
            "rank_error": int(row["rank_error"]) if pd.notna(row.get("rank_error")) else None,
            "top5_attributions": top5,
        })

    report = {
        "year": year,
        "n_rows": len(df),
        "n_top_misses": len(results),
        "target_col": TARGET_COL,
        "top_misses": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{year}" if year is not None else ""
    with open(out_dir / f"diagnostics{suffix}.json", "w") as f:
        json.dump(report, f, indent=2)
    # CSV of top attributions per player
    rows_csv = []
    for r in results:
        for feat, attr_val, raw_val in r["top5_attributions"]:
            rows_csv.append({"player_name": r["player_name"], "rank_error": r["rank_error"], "feature": feat, "attribution": attr_val, "raw_value": raw_val})
    if rows_csv:
        pd.DataFrame(rows_csv).to_csv(out_dir / f"feature_importance{suffix}.csv", index=False)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="IG explainer + top-misses report")
    ap.add_argument("--year", type=int, default=None, help="Rolling year (writes to rolling_yearly/{year}/)")
    ap.add_argument("--rankings-csv", type=Path, default=None, help="Optional rankings CSV with rank_error")
    ap.add_argument("--table", type=Path, default=SUPERVISED_PATH, help="Supervised/unified parquet")
    ap.add_argument("--ckpt", type=Path, default=None, help="Model checkpoint (default: rolling/model_{year}.pt or stack2026_supervised.pt)")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of top misses to explain")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: audit/rolling_yearly or audit/rolling_yearly/{year})")
    args = ap.parse_args()

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    elif args.year is not None:
        ckpt_path = ROLLING_MODEL_DIR / f"model_{args.year}.pt"
    else:
        ckpt_path = MODEL_DIR / "stack2026_supervised.pt"

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not args.table.exists():
        print(f"Table not found: {args.table}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir if args.out_dir is not None else (AUDIT_DIR / str(args.year) if args.year is not None else AUDIT_DIR)
    report = _run_ig_top_misses(
        year=args.year,
        rankings_csv=args.rankings_csv,
        table_path=args.table,
        ckpt_path=ckpt_path,
        top_n=args.top_n,
        out_dir=out_dir,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "top_misses"}, indent=2))
    suf = f"_{args.year}" if args.year is not None else ""
    print(f"Wrote {out_dir / ('diagnostics' + suf + '.json')}")


if __name__ == "__main__":
    main()
