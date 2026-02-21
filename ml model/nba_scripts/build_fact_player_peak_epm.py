#!/usr/bin/env python3
"""
Build peak/rolling EPM target table from merged NBA season data.

Output:
  data/warehouse_v2/fact_player_peak_epm.parquet

Columns:
  nba_id
  y_peak_epm_1y
  y_peak_epm_2y
  y_peak_epm_3y
  y_peak_epm_window
  epm_obs_seasons
  epm_obs_minutes
  epm_peak_window_end_year
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1]
NBA_MERGED = BASE / "data" / "nba_merged" / "nba_player_season_merged_2004_2025.parquet"
OUT = BASE / "data" / "warehouse_v2" / "fact_player_peak_epm.parquet"


def _first(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing any of required columns: {candidates}")


def build(min_seasons_for_3y: int = 2) -> pd.DataFrame:
    df = pd.read_parquet(NBA_MERGED)
    id_col = _first(df, ["nba_id", "nid", "epm__player_id", "player_id"])
    season_col = _first(df, ["season_year", "epm__season"])
    epm_col = _first(df, ["epm__tot", "tot"])
    mp_col = _first(df, ["minutes", "epm__mp", "mp"])

    d = pd.DataFrame(
        {
            "nba_id": pd.to_numeric(df[id_col], errors="coerce"),
            "season_year": pd.to_numeric(df[season_col], errors="coerce"),
            "epm_tot": pd.to_numeric(df[epm_col], errors="coerce"),
            "minutes": pd.to_numeric(df[mp_col], errors="coerce"),
        }
    )
    d = d.dropna(subset=["nba_id", "season_year", "epm_tot"]).copy()
    d["nba_id"] = d["nba_id"].astype(int)
    d["season_year"] = d["season_year"].astype(int)
    d["minutes"] = d["minutes"].fillna(0.0).clip(lower=0.0)

    # Deduplicate to one row per player-season by minutes-weighted EPM.
    d["epm_num"] = d["epm_tot"] * d["minutes"]
    grp = (
        d.groupby(["nba_id", "season_year"], as_index=False)
        .agg(epm_num=("epm_num", "sum"), minutes=("minutes", "sum"), epm_fallback=("epm_tot", "mean"))
    )
    grp["epm_tot"] = np.where(grp["minutes"] > 0, grp["epm_num"] / grp["minutes"], grp["epm_fallback"])
    grp = grp.sort_values(["nba_id", "season_year"])

    rows = []
    for pid, g in grp.groupby("nba_id", sort=False):
        s = g["epm_tot"].astype(float).reset_index(drop=True)
        years = g["season_year"].astype(int).reset_index(drop=True)
        mins = g["minutes"].astype(float).reset_index(drop=True)

        r1 = float(s.max()) if len(s) else np.nan
        r2 = float(s.rolling(window=2, min_periods=2).mean().max()) if len(s) >= 2 else np.nan
        r3_series = s.rolling(window=3, min_periods=max(2, int(min_seasons_for_3y))).mean()
        r3 = float(r3_series.max()) if len(s) >= 2 else np.nan

        # choose best available window priority 3y > 2y > 1y
        y = r3 if np.isfinite(r3) else (r2 if np.isfinite(r2) else r1)
        if np.isfinite(r3):
            end_idx = int(r3_series.idxmax())
        elif np.isfinite(r2):
            end_idx = int(s.rolling(window=2, min_periods=2).mean().idxmax())
        else:
            end_idx = int(s.idxmax()) if len(s) else -1
        end_year = int(years.iloc[end_idx]) if end_idx >= 0 else np.nan

        rows.append(
            {
                "nba_id": int(pid),
                "y_peak_epm_1y": r1,
                "y_peak_epm_2y": r2,
                "y_peak_epm_3y": r3,
                "y_peak_epm_window": y,
                "epm_obs_seasons": int(len(s)),
                "epm_obs_minutes": float(mins.sum()),
                "epm_peak_window_end_year": end_year,
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["nba_id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build peak/rolling EPM target fact table")
    ap.add_argument("--min-seasons-for-3y", type=int, default=2)
    args = ap.parse_args()

    out = build(min_seasons_for_3y=args.min_seasons_for_3y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    logger.info("Saved %s rows to %s", len(out), OUT)
    logger.info("Coverage y_peak_epm_3y: %.2f%%", 100.0 * out["y_peak_epm_3y"].notna().mean())


if __name__ == "__main__":
    main()

