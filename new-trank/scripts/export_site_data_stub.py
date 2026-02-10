#!/usr/bin/env python3
"""
export_site_data_stub.py

Purpose:
- Provide a minimal "new-trank" export path without touching ml model.
- This is intentionally a STUB: it does not yet compute stats from raw games.

What it does today:
- Reads the current site exports from `my-trank/public/data/`.
- Ensures a few site-contract columns exist (`drafted`, `split_id`).
- Writes the results into `new-trank/exports/`.

Why:
This lets the frontend develop against a stable contract now, while we build the
true cbbdata-derived warehouse later.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MY_TRANK_DATA = REPO_ROOT / "my-trank" / "public" / "data"
OUT_DIR = REPO_ROOT / "new-trank" / "exports"


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Drafted flag (1 if pick present)
    if "drafted" not in out.columns:
        if "pick" in out.columns:
            pick = pd.to_numeric(out["pick"], errors="coerce")
            out["drafted"] = (~pick.isna()).astype(int)
        else:
            out["drafted"] = 0

    # Competition split identifier (multi-row in the future)
    if "split_id" not in out.columns:
        out["split_id"] = "ALL"

    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["season.csv", "career.csv", "archive.csv", "weights.csv", "nba_lookup.csv", "br_advanced_stats.csv"]:
        src = MY_TRANK_DATA / name
        if not src.exists():
            continue
        if name.endswith(".csv") and name not in {"weights.csv", "nba_lookup.csv", "br_advanced_stats.csv"}:
            df = pd.read_csv(src)
            df = _ensure_columns(df)
            df.to_csv(OUT_DIR / name, index=False)
        else:
            shutil.copy2(src, OUT_DIR / name)

    # Stat dictionary is produced in the site repo today; copy if present.
    stat_dict = MY_TRANK_DATA / "stat_dictionary.json"
    if stat_dict.exists():
        shutil.copy2(stat_dict, OUT_DIR / "stat_dictionary.json")

    print(f"[ok] exports written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

