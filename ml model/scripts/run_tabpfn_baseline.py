#!/usr/bin/env python3
"""
TabPFN seasonal baseline: same feature/target matrix as Stack2026, run per year.

Diagnostic: if TabPFN ranks Shai-like players high and our model does not,
architecture/training is the bottleneck; if TabPFN also ranks them low,
feature/signal is the bottleneck. Writes to data/audit/rolling_yearly/{year}/tabpfn_baseline.json.

Usage:
  python run_tabpfn_baseline.py --year 2023
  python run_tabpfn_baseline.py --start-year 2015 --end-year 2023
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from train_2026_model import TARGET_COL, get_feature_columns  # noqa: E402

SUPERVISED_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
AUDIT_DIR = BASE / "data" / "audit" / "rolling_yearly"
# TabPFN has sample size limits; cap train size for feasibility
TABPFN_MAX_TRAIN = 1000


def run_year(year: int, df: pd.DataFrame, feat_cols: list) -> dict:
    train_df = df[df["draft_year_proxy"] <= (year - 1)].copy()
    test_df = df[df["draft_year_proxy"] == year].copy()
    y_train = pd.to_numeric(train_df[TARGET_COL], errors="coerce")
    y_test = pd.to_numeric(test_df[TARGET_COL], errors="coerce")
    train_valid = y_train.notna()
    test_valid = y_test.notna()
    train_df = train_df.loc[train_valid]
    test_df = test_df.loc[test_valid]
    if len(train_df) < 20 or len(test_df) < 5:
        return {"year": year, "error": "insufficient train or test rows", "n_train": len(train_df), "n_test": len(test_df)}

    X_train = train_df[feat_cols].fillna(0.0).astype(np.float32)
    X_test = test_df[feat_cols].fillna(0.0).astype(np.float32)
    y_train = y_train.loc[train_valid].astype(np.float32).values
    y_test = y_test.loc[test_valid].astype(np.float32).values

    if len(X_train) > TABPFN_MAX_TRAIN:
        idx = np.random.default_rng(42).choice(len(X_train), TABPFN_MAX_TRAIN, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train[idx]

    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        return {"year": year, "error": "tabpfn not installed (pip install tabpfn)"}

    try:
        model = TabPFNRegressor()
        model.fit(X_train.to_numpy(), y_train)
        pred = model.predict(X_test.to_numpy())
    except Exception as e:
        return {"year": year, "error": str(e), "n_train": len(X_train), "n_test": len(X_test)}

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    sp, _ = spearmanr(y_test, pred)
    spearman = float(sp) if np.isfinite(sp) else None
    report = {
        "year": year,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rmse": rmse,
        "spearman": spearman,
        "target_col": TARGET_COL,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="TabPFN seasonal baseline")
    ap.add_argument("--year", type=int, default=None, help="Single year")
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2023)
    ap.add_argument("--table", type=Path, default=SUPERVISED_PATH)
    args = ap.parse_args()

    if not args.table.exists():
        print(f"Table not found: {args.table}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.table)
    if "draft_year_proxy" not in df.columns:
        print("Missing draft_year_proxy", file=sys.stderr)
        sys.exit(1)
    feat_cols = get_feature_columns(df)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    feat_cols = [c for c in feat_cols if c in df.columns]

    years = [args.year] if args.year is not None else list(range(args.start_year, args.end_year + 1))
    for year in years:
        report = run_year(year, df, feat_cols)
        out_dir = AUDIT_DIR / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "tabpfn_baseline.json", "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
