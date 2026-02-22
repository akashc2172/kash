#!/usr/bin/env python3
"""
Generate input manifest: INPUTS_MASTER, INPUTS_USED_{year}, coverage_report_{year}.
Run from repo root: cd "ml model" && python scripts/generate_input_manifest.py [--year YYYY]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MANIFEST_DIR = BASE / "docs" / "model_inputs_manifest"
TRAINING_DIR = BASE / "data" / "training"
DEFAULT_TABLE = TRAINING_DIR / "unified_training_table_supervised.parquet"


def _get_feature_columns(df: pd.DataFrame) -> list:
    """Reuse train_2026_model logic when available; else fallback (no torch)."""
    try:
        import sys
        sys.path.insert(0, str(BASE))
        from scripts.train_2026_model import get_feature_columns as gfc
        return list(gfc(df, auto_drop=True))
    except Exception:
        pass
    # Fallback without torch: numeric columns, exclude IDs/targets/provenance
    exclude_prefixes = (
        "athlete_id", "nba_id", "season", "split_id", "college_final_season",
        "y_peak", "actual_peak", "draft_year", "player_name", "backfill_", "hist_",
        "source_", "confidence", "match_", "link_method",
    )
    exclude_exact = {
        "derived_minutes_total_candidate", "xy_coverage", "deflection_proxy", "contest_proxy",
    }
    cols = []
    for c in df.columns:
        if df[c].dtype.kind not in "iufb":
            continue
        if any(c.startswith(p) or c == p for p in exclude_prefixes):
            continue
        if c.endswith("_missing") or c.endswith("_source"):
            continue
        if c in exclude_exact or (c.startswith("college_") and c[8:] in exclude_exact):
            continue
        cols.append(c)
    return cols


def run(table_path: Path, year: int | None, out_dir: Path) -> None:
    if not table_path.exists():
        print(f"Table not found: {table_path}")
        return
    df = pd.read_parquet(table_path)
    if df.empty:
        print("Table is empty.")
        return

    # Optional year slice (e.g. college_final_season or draft_year_proxy)
    if year is not None:
        if "college_final_season" in df.columns:
            df = df[df["college_final_season"] == year].copy()
        elif "draft_year_proxy" in df.columns:
            df = df[df["draft_year_proxy"] == year].copy()
        if df.empty:
            print(f"No rows for year {year}.")
            return

    feat_cols = _get_feature_columns(df)
    year_suffix = f"_{year}" if year is not None else "_all"

    # INPUTS_MASTER: all numeric candidate columns + include Y/N
    numeric = [c for c in df.columns if df[c].dtype.kind in "iufb"]
    master = pd.DataFrame({
        "column": numeric,
        "category": "feature",
        "source": "unified_training_table",
        "include": ["Y" if c in feat_cols else "N" for c in numeric],
    })
    master_path = out_dir / "INPUTS_MASTER.csv"
    master.to_csv(master_path, index=False)
    print(f"Wrote {master_path} ({len(master)} rows)")

    # INPUTS_USED_{year}
    used = pd.DataFrame({"column": feat_cols})
    used_path = out_dir / f"INPUTS_USED{year_suffix}.csv"
    used.to_csv(used_path, index=False)
    print(f"Wrote {used_path} ({len(feat_cols)} columns)")

    # coverage_report_{year}: %non-null, %non-zero, variance, min, median, max
    rows = []
    for c in feat_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        n = len(s)
        non_null = s.notna().sum()
        non_zero = (s.fillna(0) != 0).sum()
        var = s.var()
        rows.append({
            "column": c,
            "pct_non_null": 100.0 * non_null / n if n else 0,
            "pct_non_zero": 100.0 * non_zero / n if n else 0,
            "variance": var if np.isfinite(var) else np.nan,
            "min": s.min() if non_null else np.nan,
            "median": s.median() if non_null else np.nan,
            "max": s.max() if non_null else np.nan,
        })
    cov = pd.DataFrame(rows)
    cov_path = out_dir / f"coverage_report{year_suffix}.csv"
    cov.to_csv(cov_path, index=False)
    print(f"Wrote {cov_path} ({len(cov)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Generate input manifest CSVs")
    ap.add_argument("--year", type=int, default=None, help="Filter to this year (college_final_season or draft_year_proxy)")
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Path to unified training table parquet")
    ap.add_argument("--out-dir", type=Path, default=MANIFEST_DIR, help="Output directory")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args.table, args.year, args.out_dir)


if __name__ == "__main__":
    main()
