#!/usr/bin/env python3
"""Hard/diagnostic gates for minutes quality and qualified-pool leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CSV_ALL = BASE / "data" / "inference" / "season_rankings_latest_best_current.csv"
AUDIT = BASE / "data" / "audit"


def main() -> int:
    ap = argparse.ArgumentParser(description="Minutes quality gate checks.")
    ap.add_argument("--max-zero-raw-qualified-v2", type=float, default=0.05)
    ap.add_argument("--min-season-median-qualified-v2", type=float, default=200.0)
    args = ap.parse_args()

    AUDIT.mkdir(parents=True, exist_ok=True)
    out_mix = AUDIT / "minutes_source_mix_by_season.csv"
    out_json = AUDIT / "minutes_quality_gate.json"
    out_cmp = AUDIT / "qualified_pool_comparison_v1_vs_v2.csv"

    failures: list[str] = []
    if not CSV_ALL.exists():
        payload = {"passed": False, "failures": [f"missing ranking csv: {CSV_ALL}"]}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    df = pd.read_csv(CSV_ALL)
    req = [
        "college_final_season",
        "college_minutes_total_raw",
        "college_minutes_total_display",
        "is_qualified_pool",
        "is_qualified_pool_v2",
    ]
    miss = [c for c in req if c not in df.columns]
    if miss:
        payload = {"passed": False, "failures": [f"missing required columns: {miss}"]}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    # Source mix (if present)
    if "college_minutes_total_source" in df.columns:
        mix = (
            df.groupby(["college_final_season", "college_minutes_total_source"], dropna=False)
            .size()
            .reset_index(name="rows")
        )
        tot = mix.groupby("college_final_season")["rows"].transform("sum")
        mix["share"] = np.where(tot > 0, mix["rows"] / tot, np.nan)
        mix = mix.sort_values(["college_final_season", "rows"], ascending=[True, False])
        mix.to_csv(out_mix, index=False)

    raw = pd.to_numeric(df["college_minutes_total_raw"], errors="coerce")
    disp = pd.to_numeric(df["college_minutes_total_display"], errors="coerce")
    q1 = pd.to_numeric(df["is_qualified_pool"], errors="coerce").fillna(0).astype(int)
    q2 = pd.to_numeric(df["is_qualified_pool_v2"], errors="coerce").fillna(0).astype(int)

    q2_mask = q2 == 1
    zero_raw_q2_rate = float(((raw.fillna(0) <= 0) & q2_mask).mean()) if len(df) else 0.0
    if zero_raw_q2_rate > args.max_zero_raw_qualified_v2:
        failures.append(
            f"qualified_v2 zero-raw-minutes rate {zero_raw_q2_rate:.3f} exceeds {args.max_zero_raw_qualified_v2:.3f}"
        )

    mod5_rate = float(((disp.dropna() % 5) == 0).mean()) if disp.notna().any() else np.nan

    q2_df = df[q2_mask].copy()
    med = (
        q2_df.groupby("college_final_season")["college_minutes_total_raw"].median()
        if not q2_df.empty else pd.Series(dtype=float)
    )
    low = med[med < args.min_season_median_qualified_v2]
    if not low.empty:
        failures.append(
            "low qualified_v2 season minute medians: "
            + ", ".join([f"{int(k)}={float(v):.1f}" for k, v in low.items()])
        )

    cmp = (
        df.groupby("college_final_season")
        .agg(
            n=("college_final_season", "size"),
            qualified_v1=("is_qualified_pool", "sum"),
            qualified_v2=("is_qualified_pool_v2", "sum"),
        )
        .reset_index()
    )
    cmp["delta_v2_minus_v1"] = cmp["qualified_v2"] - cmp["qualified_v1"]
    cmp.to_csv(out_cmp, index=False)

    payload = {
        "passed": len(failures) == 0,
        "failures": failures,
        "metrics": {
            "qualified_v2_zero_raw_minutes_rate": zero_raw_q2_rate,
            "display_mod5_rate_diagnostic": mod5_rate,
            "qualified_v1_count": int(q1.sum()),
            "qualified_v2_count": int(q2.sum()),
        },
        "season_median_minutes_qualified_v2": {str(int(k)): float(v) for k, v in med.items()},
        "artifacts": {
            "minutes_source_mix_by_season": str(out_mix),
            "qualified_pool_comparison_v1_vs_v2": str(out_cmp),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
