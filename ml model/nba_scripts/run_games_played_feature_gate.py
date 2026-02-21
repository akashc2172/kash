#!/usr/bin/env python3
"""Hard gates for games-played source quality and season coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "training" / "unified_training_table.parquet"
AUDIT_DIR = BASE / "data" / "audit"


def main() -> int:
    ap = argparse.ArgumentParser(description="Games-played quality gates.")
    ap.add_argument("--max-derived-share-season", type=float, default=0.35)
    ap.add_argument("--max-derived-share-overall", type=float, default=0.25)
    ap.add_argument("--min-season-median", type=float, default=15.0)
    ap.add_argument("--season-floor", type=int, default=2011)
    args = ap.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = AUDIT_DIR / "games_played_source_mix_by_season.csv"
    out_json = AUDIT_DIR / "games_played_regression_report.json"

    if not TRAIN_PATH.exists():
        payload = {"passed": False, "failures": [f"Missing training table: {TRAIN_PATH}"]}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    df = pd.read_parquet(TRAIN_PATH)
    required = ["college_final_season", "college_games_played", "college_games_played_source"]
    missing = [c for c in required if c not in df.columns]
    failures: list[str] = []
    if missing:
        failures.append(f"Missing required columns: {missing}")
        payload = {"passed": False, "failures": failures}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    d = df[df["college_final_season"] >= args.season_floor].copy()
    d["college_games_played"] = pd.to_numeric(d["college_games_played"], errors="coerce")

    mix = (
        d.groupby(["college_final_season", "college_games_played_source"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    totals = mix.groupby("college_final_season")["rows"].transform("sum")
    mix["share"] = np.where(totals > 0, mix["rows"] / totals, np.nan)
    mix = mix.sort_values(["college_final_season", "rows"], ascending=[True, False]).reset_index(drop=True)
    mix.to_csv(out_csv, index=False)

    derived_mask = d["college_games_played_source"].eq("derived_proxy")
    overall_derived_share = float(derived_mask.mean()) if len(d) else 0.0
    if overall_derived_share > args.max_derived_share_overall:
        failures.append(
            f"Derived-proxy overall share {overall_derived_share:.3f} exceeds {args.max_derived_share_overall:.3f}"
        )

    by_season = d.groupby("college_final_season")
    med = by_season["college_games_played"].median()
    low_med = med[med < args.min_season_median]
    if not low_med.empty:
        failures.append(
            "Low season medians: "
            + ", ".join([f"{int(k)}={float(v):.1f}" for k, v in low_med.items()])
        )

    derived_season = by_season.apply(lambda x: float((x["college_games_played_source"] == "derived_proxy").mean()))
    hi_derived = derived_season[derived_season > args.max_derived_share_season]
    if not hi_derived.empty:
        failures.append(
            "High derived-proxy season share: "
            + ", ".join([f"{int(k)}={float(v):.3f}" for k, v in hi_derived.items()])
        )

    payload = {
        "passed": len(failures) == 0,
        "failures": failures,
        "overall_derived_share": overall_derived_share,
        "thresholds": {
            "max_derived_share_overall": args.max_derived_share_overall,
            "max_derived_share_season": args.max_derived_share_season,
            "min_season_median": args.min_season_median,
            "season_floor": args.season_floor,
        },
        "season_medians": {str(int(k)): float(v) for k, v in med.items()},
        "derived_share_by_season": {str(int(k)): float(v) for k, v in derived_season.items()},
        "artifacts": {
            "source_mix_csv": str(out_csv),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
