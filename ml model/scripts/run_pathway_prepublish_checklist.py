#!/usr/bin/env python3
"""
Prepublish checklist for pathway v2 rollout.
Focuses on previously observed failure modes.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
AUDIT = BASE / "data" / "audit" / "rolling_yearly"
TRAIN = BASE / "data" / "training"
OUT = AUDIT / "prepublish_checklist_v21.json"


def _ok(v: bool, detail: str) -> dict:
    return {"pass": bool(v), "detail": detail}


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    checks = {}

    sup = TRAIN / "unified_training_table_supervised.parquet"
    fnd = TRAIN / "foundation_college_table.parquet"
    jnt = TRAIN / "unified_training_table_joint.parquet"
    checks["surfaces_exist"] = _ok(sup.exists() and fnd.exists() and jnt.exists(), "supervised/foundation/joint files present")

    if sup.exists():
        sdf = pd.read_parquet(sup, columns=[c for c in ["nba_id", "epm_years_observed", "has_peak_epm_target", "has_year1_epm_target"] if c in pd.read_parquet(sup, columns=None).columns])
        checks["supervised_unique_nba_id"] = _ok(int(sdf.duplicated(subset=["nba_id"]).sum()) == 0, "no duplicate nba_id")
        if "epm_years_observed" in sdf.columns:
            nn = float(sdf["epm_years_observed"].notna().mean())
            checks["epm_years_observed_non_null"] = _ok(nn > 0.95, f"non-null rate={nn:.4f}")
    else:
        checks["supervised_unique_nba_id"] = _ok(False, "supervised missing")

    if fnd.exists():
        fdf = pd.read_parquet(fnd, columns=[c for c in ["athlete_id", "season", "has_ctx_onoff_core", "has_ctx_velocity"] if c in pd.read_parquet(fnd, columns=None).columns])
        dup = int(fdf.duplicated(subset=["athlete_id", "season"]).sum()) if {"athlete_id", "season"}.issubset(fdf.columns) else -1
        checks["foundation_grain_unique"] = _ok(dup == 0, f"duplicate athlete_id+season={dup}")
    else:
        checks["foundation_grain_unique"] = _ok(False, "foundation missing")

    # Pathway coverage artifact
    cov = AUDIT / "pathway_coverage_by_season.csv"
    checks["pathway_coverage_audit_exists"] = _ok(cov.exists(), "pathway coverage audit emitted")

    # Compile status
    passed = all(v.get("pass", False) for v in checks.values())
    payload = {"passed": passed, "checks": checks}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

