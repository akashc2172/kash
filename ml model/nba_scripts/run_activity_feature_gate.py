#!/usr/bin/env python3
"""Hard gates for restored activity feature pipeline contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
COLLEGE_FS = BASE / "data" / "college_feature_store" / "college_features_v1.parquet"
UNIFIED = BASE / "data" / "training" / "unified_training_table.parquet"
LATEST_RANK = BASE / "data" / "inference" / "season_rankings_latest_best_current.csv"
AUDIT_DIR = BASE / "data" / "audit"

CORE_RAW = ["dunk_rate", "dunk_freq", "putback_rate", "rim_pressure_index", "contest_proxy"]
CORE_UNIFIED = [
    "college_dunk_rate",
    "college_dunk_freq",
    "college_putback_rate",
    "college_rim_pressure_index",
    "college_contest_proxy",
]
REQUIRED_PROVENANCE = [
    "college_activity_source",
    "has_college_activity_features",
    "college_dunk_rate_missing",
    "college_dunk_freq_missing",
    "college_putback_rate_missing",
    "college_rim_pressure_index_missing",
    "college_contest_proxy_missing",
]
IMPACT_ALIASES = [
    "college_off_net_rating",
    "college_on_off_net_diff",
    "college_on_off_ortg_diff",
    "college_on_off_drtg_diff",
]


def _coverage(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in cols:
        if c not in df.columns:
            out[c] = {"nonnull_pct": 0.0, "nonzero_pct": 0.0, "exists": 0.0}
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = {
            "nonnull_pct": float(s.notna().mean() * 100.0),
            "nonzero_pct": float((s.fillna(0) != 0).mean() * 100.0),
            "exists": 1.0,
        }
    return out


def _inference_required_columns() -> List[str]:
    # Keep explicit and deterministic to avoid brittle AST parsing.
    return [
        "college_dunk_rate",
        "college_dunk_freq",
        "college_putback_rate",
        "college_rim_pressure_index",
        "college_contest_proxy",
        "college_off_net_rating",
        "college_on_off_net_diff",
        "college_team_srs",
        "college_team_rank",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard gate checks for activity feature restoration.")
    ap.add_argument("--coverage-threshold", type=float, default=80.0)
    ap.add_argument("--snapshot-only", action="store_true")
    args = ap.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    snap_path = AUDIT_DIR / "activity_restore_stage0_snapshot.json"
    csv_path = AUDIT_DIR / "activity_feature_gate_report.csv"
    json_path = AUDIT_DIR / "activity_feature_gate_report.json"

    failures: List[str] = []

    cf = pd.read_parquet(COLLEGE_FS) if COLLEGE_FS.exists() else pd.DataFrame()
    ut = pd.read_parquet(UNIFIED) if UNIFIED.exists() else pd.DataFrame()
    rk = pd.read_csv(LATEST_RANK) if LATEST_RANK.exists() else pd.DataFrame()

    snapshot = {
        "paths": {"college_features": str(COLLEGE_FS), "unified": str(UNIFIED), "latest_rank": str(LATEST_RANK)},
        "rows": {"college_features": int(len(cf)), "unified": int(len(ut)), "latest_rank": int(len(rk))},
        "college_feature_coverage": _coverage(cf, CORE_RAW),
        "unified_core_coverage": _coverage(ut, CORE_UNIFIED),
        "unified_required_presence": {
            c: (c in ut.columns) for c in CORE_UNIFIED + REQUIRED_PROVENANCE + IMPACT_ALIASES
        },
    }
    snap_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    if args.snapshot_only:
        print(json.dumps({"snapshot": str(snap_path), "status": "ok"}, indent=2))
        return

    # Hard gates
    for c in CORE_UNIFIED:
        if c not in ut.columns:
            failures.append(f"Missing core unified column: {c}")
    for c in REQUIRED_PROVENANCE:
        if c not in ut.columns:
            failures.append(f"Missing provenance/mask column: {c}")
    for c in IMPACT_ALIASES:
        if c not in ut.columns:
            failures.append(f"Missing impact alias column: {c}")

    cov = _coverage(ut, CORE_UNIFIED)
    for c, d in cov.items():
        if d["nonnull_pct"] < args.coverage_threshold:
            failures.append(
                f"Coverage below threshold for {c}: {d['nonnull_pct']:.2f}% < {args.coverage_threshold:.2f}%"
            )

    if "nba_id" in ut.columns:
        dup = int(ut["nba_id"].duplicated().sum())
        if dup > 0:
            failures.append(f"Duplicate nba_id rows in unified table: {dup}")
    else:
        failures.append("Missing nba_id in unified table")

    # Encoder/inference contract
    from models.player_encoder import TIER1_COLUMNS
    tier1_missing = [c for c in TIER1_COLUMNS if c not in ut.columns]
    if tier1_missing:
        failures.append(f"Encoder columns missing from unified: {len(tier1_missing)}")

    inf_required = _inference_required_columns()
    inf_missing = [c for c in inf_required if c not in ut.columns]
    if inf_missing:
        failures.append(f"Inference-required columns missing from unified: {inf_missing}")

    report_rows = []
    for c in CORE_UNIFIED:
        d = cov.get(c, {"nonnull_pct": 0.0, "nonzero_pct": 0.0})
        report_rows.append(
            {"column": c, "nonnull_pct": d["nonnull_pct"], "nonzero_pct": d["nonzero_pct"], "threshold_pct": args.coverage_threshold}
        )
    pd.DataFrame(report_rows).to_csv(csv_path, index=False)

    gate = {
        "snapshot": str(snap_path),
        "coverage_report_csv": str(csv_path),
        "passed": len(failures) == 0,
        "failures": failures,
        "coverage_threshold_pct": args.coverage_threshold,
    }
    json_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
