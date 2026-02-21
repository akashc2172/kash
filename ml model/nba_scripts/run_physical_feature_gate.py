#!/usr/bin/env python3
"""Hard gates for college physicals backfill contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
WAREHOUSE = BASE / "data" / "warehouse.duckdb"
WAREHOUSE_V2 = BASE / "data" / "warehouse_v2"
UNIFIED = BASE / "data" / "training" / "unified_training_table.parquet"
AUDIT = BASE / "data" / "audit"

CANON_PQ = WAREHOUSE_V2 / "fact_college_player_physicals_by_season.parquet"
TRAJ_PQ = WAREHOUSE_V2 / "fact_college_player_physical_trajectory.parquet"

REQUIRED_UNIFIED = [
    "college_height_in",
    "college_weight_lbs",
    "has_college_height",
    "has_college_weight",
    "college_height_delta_yoy",
    "college_weight_delta_yoy",
    "college_height_slope_3yr",
    "college_weight_slope_3yr",
    "wingspan_in",
    "standing_reach_in",
    "wingspan_minus_height_in",
    "has_wingspan",
]


def _load_canonical() -> pd.DataFrame:
    if CANON_PQ.exists():
        return pd.read_parquet(CANON_PQ)
    if not WAREHOUSE.exists():
        return pd.DataFrame()
    try:
        con = duckdb.connect(str(WAREHOUSE), read_only=True)
        df = con.execute("SELECT * FROM fact_college_player_physicals_by_season").df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def _load_trajectory() -> pd.DataFrame:
    if TRAJ_PQ.exists():
        return pd.read_parquet(TRAJ_PQ)
    if not WAREHOUSE.exists():
        return pd.DataFrame()
    try:
        con = duckdb.connect(str(WAREHOUSE), read_only=True)
        df = con.execute("SELECT * FROM fact_college_player_physical_trajectory").df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def _nonnull_pct(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").notna().mean() * 100.0)


def _read_linkage_rates() -> Dict[str, float]:
    p = AUDIT / "physicals_linkage_quality.csv"
    if not p.exists():
        return {"unresolved_rate_pct": 100.0, "ambiguous_rate_pct": 100.0}
    try:
        df = pd.read_csv(p)
        if df.empty:
            return {"unresolved_rate_pct": 100.0, "ambiguous_rate_pct": 100.0}
        r = df.iloc[0].to_dict()
        return {
            "unresolved_rate_pct": float(r.get("unresolved_rate_pct", 100.0)),
            "ambiguous_rate_pct": float(r.get("ambiguous_rate_pct", 100.0)),
        }
    except Exception:
        return {"unresolved_rate_pct": 100.0, "ambiguous_rate_pct": 100.0}


def _read_nba_mapped_missing_rate() -> float:
    p = AUDIT / "physicals_nba_mapped_missing_rate.csv"
    if not p.exists():
        return 100.0
    try:
        df = pd.read_csv(p)
        if df.empty:
            return 100.0
        return float(df.iloc[0].get("missing_rate_pct", 100.0))
    except Exception:
        return 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard gate checks for college physicals pipeline.")
    ap.add_argument("--coverage-threshold", type=float, default=90.0)
    ap.add_argument("--max-unresolved-rate", type=float, default=5.0)
    ap.add_argument("--max-nba-mapped-missing-rate", type=float, default=3.0)
    args = ap.parse_args()

    AUDIT.mkdir(parents=True, exist_ok=True)
    out_csv = AUDIT / "physical_feature_gate_report.csv"
    out_json = AUDIT / "physical_feature_gate_report.json"

    failures: List[str] = []
    canon = _load_canonical()
    traj = _load_trajectory()
    unified = pd.read_parquet(UNIFIED) if UNIFIED.exists() else pd.DataFrame()

    if canon.empty:
        failures.append("Canonical physicals table is missing/empty")
    else:
        key_cols = ["athlete_id", "season", "team_id"]
        miss = [c for c in key_cols if c not in canon.columns]
        if miss:
            failures.append(f"Canonical key columns missing: {miss}")
        else:
            dup = int(canon.duplicated(subset=key_cols).sum())
            if dup > 0:
                failures.append(f"Canonical duplicates on (athlete_id, season, team_id): {dup}")

    if traj.empty:
        failures.append("Trajectory table is missing/empty")

    for c in REQUIRED_UNIFIED:
        if c not in unified.columns:
            failures.append(f"Unified missing required column: {c}")

    h_cov = _nonnull_pct(unified, "college_height_in")
    w_cov = _nonnull_pct(unified, "college_weight_lbs")
    if h_cov < args.coverage_threshold:
        failures.append(f"Height coverage below threshold: {h_cov:.2f}% < {args.coverage_threshold:.2f}%")
    if w_cov < args.coverage_threshold:
        failures.append(f"Weight coverage below threshold: {w_cov:.2f}% < {args.coverage_threshold:.2f}%")

    lr = _read_linkage_rates()
    if lr["unresolved_rate_pct"] > args.max_unresolved_rate:
        failures.append(
            f"Unresolved identity rate too high: {lr['unresolved_rate_pct']:.2f}% > {args.max_unresolved_rate:.2f}%"
        )
    nba_missing = _read_nba_mapped_missing_rate()
    if nba_missing > args.max_nba_mapped_missing_rate:
        failures.append(
            f"NBA-mapped seasonal missing rate too high: {nba_missing:.2f}% > {args.max_nba_mapped_missing_rate:.2f}%"
        )

    cov_rows = [
        {"column": "college_height_in", "nonnull_pct": h_cov, "threshold_pct": args.coverage_threshold},
        {"column": "college_weight_lbs", "nonnull_pct": w_cov, "threshold_pct": args.coverage_threshold},
        {
            "column": "unresolved_identity_rate_pct",
            "nonnull_pct": float(lr["unresolved_rate_pct"]),
            "threshold_pct": args.max_unresolved_rate,
        },
        {
            "column": "nba_mapped_missing_rate_pct",
            "nonnull_pct": float(nba_missing),
            "threshold_pct": args.max_nba_mapped_missing_rate,
        },
    ]
    pd.DataFrame(cov_rows).to_csv(out_csv, index=False)

    payload = {
        "passed": len(failures) == 0,
        "failures": failures,
        "coverage_threshold_pct": float(args.coverage_threshold),
        "max_unresolved_rate_pct": float(args.max_unresolved_rate),
        "max_nba_mapped_missing_rate_pct": float(args.max_nba_mapped_missing_rate),
        "artifacts": {
            "canonical": str(CANON_PQ),
            "trajectory": str(TRAJ_PQ),
            "unified": str(UNIFIED),
            "coverage_report_csv": str(out_csv),
            "nba_mapped_missing_report_csv": str(AUDIT / "physicals_nba_mapped_missing_rate.csv"),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
