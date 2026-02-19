#!/usr/bin/env python3
"""
NBA Pre-Train Readiness Gate
============================
Checks critical coverage and drift signals before model training.

Outputs:
- data/audit/nba_pretrain_gate.json
- data/audit/nba_pretrain_gate_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_PATH = BASE_DIR / "data/training/unified_training_table.parquet"
DIM_PATH = BASE_DIR / "data/warehouse_v2/dim_player_nba.parquet"
Y1_PATH = BASE_DIR / "data/warehouse_v2/fact_player_year1_epm.parquet"
DEV_PATH = BASE_DIR / "data/warehouse_v2/fact_player_development_rate.parquet"
AUDIT_DIR = BASE_DIR / "data/audit"


def _rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(df[col].notna().mean())


def _bool_check(name: str, value: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(value), "detail": detail}


def build_gate_report(min_overlap_recent: int = 5) -> dict:
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(f"Missing training table: {TRAINING_PATH}")
    if not DIM_PATH.exists():
        raise FileNotFoundError(f"Missing dim table: {DIM_PATH}")
    if not Y1_PATH.exists():
        raise FileNotFoundError(f"Missing year1 table: {Y1_PATH}")
    if not DEV_PATH.exists():
        raise FileNotFoundError(f"Missing dev table: {DEV_PATH}")

    train = pd.read_parquet(TRAINING_PATH)
    dim = pd.read_parquet(DIM_PATH)[["nba_id", "rookie_season_year"]]
    y1 = pd.read_parquet(Y1_PATH)
    dev = pd.read_parquet(DEV_PATH)

    n_rows = int(len(train))
    dup_nba = int(train.duplicated(subset=["nba_id"]).sum()) if "nba_id" in train.columns else n_rows

    coverage = {
        "y_peak_ovr_non_null_rate": _rate(train, "y_peak_ovr"),
        "year1_epm_tot_non_null_rate": _rate(train, "year1_epm_tot"),
        "dev_rate_y1_y3_mean_non_null_rate": _rate(train, "dev_rate_y1_y3_mean"),
        "dev_rate_quality_weight_non_null_rate": _rate(train, "dev_rate_quality_weight"),
        "gap_ts_legacy_non_null_rate": _rate(train, "gap_ts_legacy"),
        "gap_usg_legacy_non_null_rate": _rate(train, "gap_usg_legacy"),
    }

    # Dev-rate distribution sanity.
    dev_mean = pd.to_numeric(train.get("dev_rate_y1_y3_mean", pd.Series(dtype=float)), errors="coerce")
    dev_sd = pd.to_numeric(train.get("dev_rate_y1_y3_sd", pd.Series(dtype=float)), errors="coerce")
    dev_w = pd.to_numeric(train.get("dev_rate_quality_weight", pd.Series(dtype=float)), errors="coerce")
    distribution = {
        "dev_rate_mean": float(dev_mean.mean()) if len(dev_mean) else np.nan,
        "dev_rate_std": float(dev_mean.std()) if len(dev_mean) else np.nan,
        "dev_rate_p05": float(dev_mean.quantile(0.05)) if len(dev_mean) else np.nan,
        "dev_rate_p50": float(dev_mean.quantile(0.50)) if len(dev_mean) else np.nan,
        "dev_rate_p95": float(dev_mean.quantile(0.95)) if len(dev_mean) else np.nan,
        "dev_sd_p50": float(dev_sd.quantile(0.50)) if len(dev_sd) else np.nan,
        "dev_sd_p90": float(dev_sd.quantile(0.90)) if len(dev_sd) else np.nan,
        "dev_weight_p95": float(dev_w.quantile(0.95)) if len(dev_w) else np.nan,
        "dev_weight_p99": float(dev_w.quantile(0.99)) if len(dev_w) else np.nan,
    }

    # Drift/overlap checks by rookie season.
    y1_small = y1[["nba_id", "tot"]].copy() if "tot" in y1.columns else y1[["nba_id"]].copy()
    if "tot" in y1_small.columns:
        y1_small["has_y1"] = y1_small["tot"].notna().astype(int)
    else:
        y1_small["has_y1"] = 1

    dev_small = dev[["nba_id", "dev_has_rapm3y", "dev_rate_y1_y3_mean"]].copy()
    dev_small["has_dev"] = dev_small["dev_rate_y1_y3_mean"].notna().astype(int)
    dev_small["has_rapm3y"] = dev_small["dev_has_rapm3y"].fillna(0).astype(int)

    overlap = dim.merge(y1_small[["nba_id", "has_y1"]], on="nba_id", how="left")
    overlap = overlap.merge(dev_small[["nba_id", "has_dev", "has_rapm3y"]], on="nba_id", how="left")
    overlap[["has_y1", "has_dev", "has_rapm3y"]] = overlap[["has_y1", "has_dev", "has_rapm3y"]].fillna(0).astype(int)
    overlap["has_both"] = ((overlap["has_y1"] == 1) & (overlap["has_dev"] == 1)).astype(int)

    by_rookie = (
        overlap.dropna(subset=["rookie_season_year"])
        .groupby("rookie_season_year", as_index=False)
        .agg(
            n_players=("nba_id", "nunique"),
            n_y1=("has_y1", "sum"),
            n_dev=("has_dev", "sum"),
            n_rapm3y=("has_rapm3y", "sum"),
            n_overlap=("has_both", "sum"),
        )
        .sort_values("rookie_season_year")
    )

    recent = by_rookie[by_rookie["rookie_season_year"] >= 2018].copy()
    recent_min_overlap = int(recent["n_overlap"].min()) if len(recent) else 0
    recent_mean_overlap = float(recent["n_overlap"].mean()) if len(recent) else 0.0

    checks = [
        _bool_check(
            "no_duplicate_nba_id_rows",
            dup_nba == 0,
            f"duplicate_nba_id_rows={dup_nba}",
        ),
        _bool_check(
            "peak_target_coverage",
            coverage["y_peak_ovr_non_null_rate"] >= 0.80,
            f"rate={coverage['y_peak_ovr_non_null_rate']:.3f}, min=0.800",
        ),
        _bool_check(
            "year1_epm_coverage",
            coverage["year1_epm_tot_non_null_rate"] >= 0.65,
            f"rate={coverage['year1_epm_tot_non_null_rate']:.3f}, min=0.650",
        ),
        _bool_check(
            "dev_rate_coverage",
            coverage["dev_rate_y1_y3_mean_non_null_rate"] >= 0.85,
            f"rate={coverage['dev_rate_y1_y3_mean_non_null_rate']:.3f}, min=0.850",
        ),
        _bool_check(
            "dev_weight_coverage",
            coverage["dev_rate_quality_weight_non_null_rate"] >= 0.85,
            f"rate={coverage['dev_rate_quality_weight_non_null_rate']:.3f}, min=0.850",
        ),
        _bool_check(
            "dev_distribution_sane",
            bool(np.isfinite(distribution["dev_rate_std"]) and distribution["dev_rate_std"] > 0.01),
            f"dev_rate_std={distribution['dev_rate_std']:.4f}",
        ),
        _bool_check(
            "recent_overlap_not_collapsed",
            recent_min_overlap >= min_overlap_recent,
            f"recent_min_overlap={recent_min_overlap}, threshold={min_overlap_recent}",
        ),
    ]

    non_critical = [
        _bool_check(
            "gap_ts_legacy_coverage",
            coverage["gap_ts_legacy_non_null_rate"] >= 0.60,
            f"rate={coverage['gap_ts_legacy_non_null_rate']:.3f}, target=0.600",
        ),
        _bool_check(
            "gap_usg_legacy_coverage",
            coverage["gap_usg_legacy_non_null_rate"] >= 0.40,
            f"rate={coverage['gap_usg_legacy_non_null_rate']:.3f}, non-critical",
        ),
    ]

    passed = all(c["passed"] for c in checks)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "training_table": str(TRAINING_PATH),
            "dim_player_nba": str(DIM_PATH),
            "fact_player_year1_epm": str(Y1_PATH),
            "fact_player_development_rate": str(DEV_PATH),
        },
        "row_counts": {
            "training_rows": n_rows,
            "dim_rows": int(len(dim)),
            "year1_rows": int(len(y1)),
            "dev_rows": int(len(dev)),
        },
        "coverage": coverage,
        "distribution": distribution,
        "drift_checks": {
            "recent_min_overlap": recent_min_overlap,
            "recent_mean_overlap": recent_mean_overlap,
            "overlap_by_rookie_season": by_rookie.to_dict(orient="records"),
        },
        "quality_gate": {
            "passed": passed,
            "critical_checks": checks,
            "non_critical_checks": non_critical,
            "notes": [
                "gap_usg_legacy remains non-critical by design.",
            ],
        },
    }


def write_gate_report(report: dict) -> Path:
    """Write gate report to latest + timestamped JSON and return latest path."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    latest = AUDIT_DIR / "nba_pretrain_gate.json"
    ts_path = AUDIT_DIR / f"nba_pretrain_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = json.dumps(report, indent=2)
    latest.write_text(payload, encoding="utf-8")
    ts_path.write_text(payload, encoding="utf-8")
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NBA pre-train readiness gate")
    parser.add_argument("--min-overlap-recent", type=int, default=5)
    parser.add_argument("--fail-on-gate", action="store_true", help="Exit non-zero if critical gate fails")
    args = parser.parse_args()

    report = build_gate_report(min_overlap_recent=args.min_overlap_recent)

    latest = write_gate_report(report)

    passed = report["quality_gate"]["passed"]
    logger.info("Saved gate report: %s", latest)
    logger.info("Quality gate passed: %s", passed)
    for check in report["quality_gate"]["critical_checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        logger.info("[%s] %s | %s", status, check["name"], check["detail"])

    if args.fail_on_gate and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
