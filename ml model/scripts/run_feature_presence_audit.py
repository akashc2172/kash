#!/usr/bin/env python3
"""
Feature presence audit for the "Shai bundle" and training feature set.

Per the Trajectory Plan v2: no guard probe; this is a sanity check that the
model can see the signals (AST%, stl, TS/usage, unassisted/self_creation, age,
high_major). Reports:
  - Which bundle columns are in feature_cols?
  - Non-null / non-zero share in cohort (NBA-linked or full supervised table).

Usage:
  python run_feature_presence_audit.py [--table PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Scripts and nba_scripts are siblings under ml model
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from train_2026_model import get_feature_columns  # noqa: E402

# Columns we care about for guard/creation signal (Shai bundle)
SHAI_BUNDLE = [
    "college_ast_total_per100poss",
    "college_stl_total_per100poss",
    "college_assisted_share_rim",
    "college_assisted_share_mid",
    "college_assisted_share_three",
    "college_self_creation_rate",
    "age_at_season",
    "class_year",
    "college_is_power_conf",
    "final_usage",
    "final_trueShootingPct",
    "college_ft_pct",
    "college_tov_total_per100poss",
]
# Also check any column that matches these prefixes (for flexible naming)
SHAI_BUNDLE_PREFIXES = ["college_assisted_share_", "college_ast_", "college_stl_", "final_usage", "final_trueShootingPct"]


def run_audit(table_path: Path) -> dict:
    df = pd.read_parquet(table_path)
    feat_cols = get_feature_columns(df)
    n = len(df)

    # Cohort: rows with at least one target (NBA-linked); fallback full table
    target_candidates = ["y_peak_epm_1y_60gp", "y_peak_epm_3y", "y_peak_epm_window", "y_peak_epm_1y"]
    has_target = pd.Series(False, index=df.index)
    for c in target_candidates:
        if c in df.columns:
            has_target = has_target | pd.to_numeric(df[c], errors="coerce").notna()
    cohort = df.loc[has_target] if has_target.any() else df
    n_cohort = len(cohort)

    # 1) In feature_cols?
    in_feature_cols = {col: col in feat_cols for col in SHAI_BUNDLE}
    # Any feature that matches Shai bundle by prefix
    for c in feat_cols:
        for pre in SHAI_BUNDLE_PREFIXES:
            if c.startswith(pre) and c not in SHAI_BUNDLE:
                in_feature_cols[c] = True

    # 2) Present in table at all
    in_table = {col: col in df.columns for col in SHAI_BUNDLE}

    # 3) Non-null share (full table and cohort)
    non_null_full = {}
    non_null_cohort = {}
    non_zero_cohort = {}
    for col in SHAI_BUNDLE:
        if col not in df.columns:
            non_null_full[col] = None
            non_null_cohort[col] = None
            non_zero_cohort[col] = None
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        non_null_full[col] = float(s.notna().sum() / n) if n else None
        if n_cohort:
            sc = pd.to_numeric(cohort[col], errors="coerce")
            non_null_cohort[col] = float(sc.notna().sum() / n_cohort)
            non_zero_cohort[col] = float((sc.notna() & (sc != 0)).sum() / n_cohort)
        else:
            non_null_cohort[col] = None
            non_zero_cohort[col] = None

    report = {
        "table_path": str(table_path),
        "n_rows": n,
        "n_cohort_nba_linked": n_cohort,
        "n_feature_cols": len(feat_cols),
        "shai_bundle_in_feature_cols": in_feature_cols,
        "shai_bundle_in_table": in_table,
        "non_null_share_full": non_null_full,
        "non_null_share_cohort": non_null_cohort,
        "non_zero_share_cohort": non_zero_cohort,
        "all_feature_cols": feat_cols,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature presence audit for Shai bundle")
    ap.add_argument(
        "--table",
        type=Path,
        default=BASE / "data" / "training" / "unified_training_table_supervised.parquet",
        help="Path to supervised or unified parquet",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write JSON report here (default: data/audit/feature_presence_audit.json)")
    args = ap.parse_args()

    if not args.table.exists():
        print(f"Table not found: {args.table}", file=sys.stderr)
        sys.exit(1)

    report = run_audit(args.table)

    out = args.out if args.out is not None else (BASE / "data" / "audit" / "feature_presence_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({k: v for k, v in report.items() if k != "all_feature_cols"}, f, indent=2)
    with open(out.with_suffix(".full.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out} and .full.json")
    print(json.dumps({k: v for k, v in report.items() if k != "all_feature_cols"}, indent=2))


if __name__ == "__main__":
    main()
