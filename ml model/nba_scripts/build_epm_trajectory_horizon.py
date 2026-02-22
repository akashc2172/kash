#!/usr/bin/env python3
"""
Horizon-bounded EPM trajectory labels (no future leakage).

Fits a local-linear-trend state-space model (statsmodels UnobservedComponents)
only on seasons within a fixed horizon (e.g. first 7 seasons). Outputs:
  - latent_peak_within_7y: max of smoothed level (mu_t) over the horizon
  - slope_last_2y: velocity at end of horizon (or over last 2 seasons)
  - plateau_flag: 1 if velocity is below threshold at end (peaking/flat)
  - epm_trajectory_censored: 1 if observed seasons < horizon (career may continue)

Output: data/warehouse_v2/fact_player_epm_trajectory_horizon.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1]
NBA_MERGED = BASE / "data" / "nba_merged" / "nba_player_season_merged_2004_2025.parquet"
OUT = BASE / "data" / "warehouse_v2" / "fact_player_epm_trajectory_horizon.parquet"

HORIZON_YEARS = 7
PLATEAU_VELOCITY_THRESHOLD = 0.05  # EPM per season; below this = plateau
MIN_SEASONS_FOR_FIT = 2


def _first(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing any of required columns: {candidates}")


def _fit_trajectory_within_horizon(
    epm_series: np.ndarray,
    horizon: int = HORIZON_YEARS,
    plateau_threshold: float = PLATEAU_VELOCITY_THRESHOLD,
) -> dict:
    """
    Fit local linear trend on first `horizon` seasons; return latent peak, slope, plateau flag.
    Uses only past data within the window (no leakage).
    """
    y = np.asarray(epm_series, dtype=np.float64)
    T = min(len(y), horizon)
    if T < MIN_SEASONS_FOR_FIT:
        return {
            "latent_peak_within_7y": np.nan,
            "slope_last_2y": np.nan,
            "plateau_flag": np.nan,
            "n_seasons_used": T,
        }
    y_trunc = y[:T]

    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        mod = UnobservedComponents(y_trunc, level=True, trend=True)
        res = mod.fit(disp=0)
        # smoothed_state: (state_dim, T). For level+trend, state_dim=2; row 0 = level, row 1 = trend
        smoothed = res.smoothed_state
        if smoothed is None or smoothed.size == 0:
            latent_peak = float(np.nanmax(y_trunc))
            slope_end = np.nan
            plateau = np.nan
        else:
            # shape (2, T) or (T,) for single state
            if smoothed.ndim == 2:
                level = smoothed[0]
                trend = smoothed[1]
            else:
                level = smoothed
                trend = np.full_like(level, np.nan)
            latent_peak = float(np.nanmax(level))
            slope_end = float(trend[-1]) if len(trend) and np.isfinite(trend[-1]) else (float(level[-1]) - float(level[-2])) if len(level) >= 2 else np.nan
            plateau = 1.0 if np.isfinite(slope_end) and slope_end < plateau_threshold else 0.0
        return {
            "latent_peak_within_7y": latent_peak,
            "slope_last_2y": slope_end,
            "plateau_flag": plateau,
            "n_seasons_used": T,
        }
    except Exception as e:
        logger.debug("Trajectory fit failed for series %s: %s", y_trunc, e)
        # Fallback: raw max and simple slope
        latent_peak = float(np.nanmax(y_trunc))
        slope_end = (float(y_trunc[-1]) - float(y_trunc[-2])) / 1.0 if T >= 2 else np.nan
        plateau = 1.0 if np.isfinite(slope_end) and slope_end < plateau_threshold else 0.0
        return {
            "latent_peak_within_7y": latent_peak,
            "slope_last_2y": slope_end,
            "plateau_flag": plateau,
            "n_seasons_used": T,
        }


def build(
    horizon_years: int = HORIZON_YEARS,
    plateau_threshold: float = PLATEAU_VELOCITY_THRESHOLD,
) -> pd.DataFrame:
    df = pd.read_parquet(NBA_MERGED)
    id_col = _first(df, ["nba_id", "nid", "epm__player_id", "player_id"])
    season_col = _first(df, ["season_year", "epm__season"])
    epm_col = _first(df, ["epm__tot", "tot"])
    mp_col = _first(df, ["minutes", "epm__mp", "mp"])
    gp_col = _first(df, ["gp", "g", "games_played", "games"])

    d = pd.DataFrame(
        {
            "nba_id": pd.to_numeric(df[id_col], errors="coerce"),
            "season_year": pd.to_numeric(df[season_col], errors="coerce"),
            "epm_tot": pd.to_numeric(df[epm_col], errors="coerce"),
            "minutes": pd.to_numeric(df[mp_col], errors="coerce"),
            "games": pd.to_numeric(df[gp_col], errors="coerce").fillna(0).clip(lower=0),
        }
    )
    d = d.dropna(subset=["nba_id", "season_year", "epm_tot"]).copy()
    d["nba_id"] = d["nba_id"].astype(int)
    d["season_year"] = d["season_year"].astype(int)
    d = d.sort_values(["nba_id", "season_year"])

    rows = []
    for pid, g in d.groupby("nba_id", sort=False):
        s = g["epm_tot"].astype(float).values
        # Use first horizon_years seasons only (no future leakage)
        s_trunc = s[:horizon_years]
        out = _fit_trajectory_within_horizon(
            s_trunc,
            horizon=horizon_years,
            plateau_threshold=plateau_threshold,
        )
        n_obs = int(len(g))
        rows.append({
            "nba_id": int(pid),
            "latent_peak_within_7y": out["latent_peak_within_7y"],
            "slope_last_2y": out["slope_last_2y"],
            "plateau_flag": out["plateau_flag"],
            "epm_trajectory_n_seasons_used": out["n_seasons_used"],
            "epm_trajectory_censored": 1 if n_obs < horizon_years else 0,
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["nba_id"])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build horizon-bounded EPM trajectory labels")
    ap.add_argument("--horizon", type=int, default=HORIZON_YEARS, help="Max seasons in window")
    ap.add_argument("--plateau-threshold", type=float, default=PLATEAU_VELOCITY_THRESHOLD, help="Velocity below this = plateau")
    args = ap.parse_args()

    out_df = build(horizon_years=args.horizon, plateau_threshold=args.plateau_threshold)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, index=False)
    logger.info("Saved %s rows to %s", len(out_df), OUT)
    logger.info("latent_peak_within_7y non-null: %.2f%%", 100.0 * out_df["latent_peak_within_7y"].notna().mean())
    logger.info("epm_trajectory_censored=1: %.2f%%", 100.0 * (out_df["epm_trajectory_censored"] == 1).mean())


if __name__ == "__main__":
    main()
