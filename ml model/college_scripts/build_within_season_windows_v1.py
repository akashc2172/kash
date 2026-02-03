#!/usr/bin/env python3
"""Within-Season Windows Builder (v1)

Builds within-season window features from DuckDB player-game data.

Input:
- data/warehouse.duckdb (fact_player_game, dim_games)

Output:
- data/college_feature_store/within_season_windows_v1.parquet

Design notes:
- Missing stays NaN; never impute missing to 0.
- Masks indicate whether a feature is computable (enough games / coverage).
- v1 focuses on last5/last10 aggregates plus a lightweight breakout timing signal.
"""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
WAREHOUSE = BASE_DIR / 'data/warehouse.duckdb'
OUT_PATH = BASE_DIR / 'data/college_feature_store/within_season_windows_v1.parquet'


def load_player_games() -> pd.DataFrame:
    if not WAREHOUSE.exists():
        raise FileNotFoundError(f"Warehouse not found: {WAREHOUSE}")

    import duckdb

    query = """
        SELECT
            f.athleteId AS athlete_id,
            g.season AS season,
            g.startDate AS start_date,
            f.gameId AS game_id,
            f.fga AS fga,
            f.pts AS pts,
            f.seconds_on AS seconds_on,
            f.on_net_rating AS on_net_rating
        FROM fact_player_game f
        JOIN (
            SELECT CAST(id AS VARCHAR) AS gameId, season, startDate
            FROM dim_games
        ) g
        ON g.gameId = f.gameId
        WHERE g.startDate IS NOT NULL
    """

    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    try:
        df = con.sql(query).df()
    finally:
        con.close()

    # Ensure types
    df['season'] = df['season'].astype(int)
    df['start_date'] = pd.to_datetime(df['start_date'], utc=True, errors='coerce')
    for col in ['fga', 'pts', 'seconds_on', 'on_net_rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def _sum_last_n(x: pd.Series, n: int) -> float:
    if x.isna().all() or len(x) < n:
        return np.nan
    return float(x.tail(n).sum(skipna=True))


def _mean_last_n(x: pd.Series, n: int) -> float:
    if x.isna().all() or len(x) < n:
        return np.nan
    return float(x.tail(n).mean(skipna=True))


def _pps(pts: float, fga: float) -> float:
    if pts is None or fga is None or np.isnan(pts) or np.isnan(fga) or fga <= 0:
        return np.nan
    return float(pts) / float(fga)


def _rolling_max_timing(metric: pd.Series, window: int) -> tuple[float, int]:
    """Return (timing, has_flag). timing in [0,1] based on argmax of rolling mean."""
    metric = metric.astype(float)
    n = len(metric)
    if n < max(window, 2) or metric.isna().all():
        return (np.nan, 0)

    roll = metric.rolling(window=window, min_periods=window).mean()
    if roll.isna().all():
        return (np.nan, 0)

    idx = int(roll.idxmax())
    # idx is positional index if metric has RangeIndex; ensure positional timing:
    # We compute position by locating idx within index.
    pos = int(metric.index.get_indexer([idx])[0])
    denom = max(1, n - 1)
    return (float(pos) / float(denom), 1)


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # Sort within each athlete-season by start_date.
    df = df.sort_values(['athlete_id', 'season', 'start_date', 'game_id'])

    # For each athlete-season, compute games_played and window aggregates.
    rows = []
    for (athlete_id, season), g in df.groupby(['athlete_id', 'season'], sort=False):
        g = g.reset_index(drop=True)
        gp = int(len(g))

        minutes = g['seconds_on'] / 60.0

        fga_last5 = _sum_last_n(g['fga'], 5)
        pts_last5 = _sum_last_n(g['pts'], 5)
        min_last5 = _sum_last_n(minutes, 5)
        onnr_last5 = _mean_last_n(g['on_net_rating'], 5)

        fga_last10 = _sum_last_n(g['fga'], 10)
        pts_last10 = _sum_last_n(g['pts'], 10)
        min_last10 = _sum_last_n(minutes, 10)
        onnr_last10 = _mean_last_n(g['on_net_rating'], 10)

        fga_prev5 = _sum_last_n(g['fga'].iloc[:-5], 5) if gp >= 10 else np.nan
        pts_prev5 = _sum_last_n(g['pts'].iloc[:-5], 5) if gp >= 10 else np.nan
        min_prev5 = _sum_last_n(minutes.iloc[:-5], 5) if gp >= 10 else np.nan

        fga_prev10 = _sum_last_n(g['fga'].iloc[:-10], 10) if gp >= 20 else np.nan
        pts_prev10 = _sum_last_n(g['pts'].iloc[:-10], 10) if gp >= 20 else np.nan
        min_prev10 = _sum_last_n(minutes.iloc[:-10], 10) if gp >= 20 else np.nan

        pps_last5 = _pps(pts_last5, fga_last5)
        pps_last10 = _pps(pts_last10, fga_last10)
        pps_prev5 = _pps(pts_prev5, fga_prev5)
        pps_prev10 = _pps(pts_prev10, fga_prev10)

        has_last5 = int(gp >= 5)
        has_last10 = int(gp >= 10)
        has_prev5 = int(gp >= 10)
        has_prev10 = int(gp >= 20)

        ws_delta_pps_5 = (pps_last5 - pps_prev5) if (has_prev5 and not np.isnan(pps_last5) and not np.isnan(pps_prev5)) else np.nan
        ws_delta_min_5 = (min_last5 - min_prev5) if (has_prev5 and not np.isnan(min_last5) and not np.isnan(min_prev5)) else np.nan

        ws_delta_pps_10 = (pps_last10 - pps_prev10) if (has_prev10 and not np.isnan(pps_last10) and not np.isnan(pps_prev10)) else np.nan
        ws_delta_min_10 = (min_last10 - min_prev10) if (has_prev10 and not np.isnan(min_last10) and not np.isnan(min_prev10)) else np.nan

        # Rolling timing (v1): rolling 3-game mean on minutes, fga, pps proxy.
        timing_minutes, has_t_min = _rolling_max_timing(minutes, window=3)
        timing_volume, has_t_vol = _rolling_max_timing(g['fga'], window=3)

        pps_game = g.apply(lambda r: _pps(r['pts'], r['fga']), axis=1)
        timing_eff, has_t_eff = _rolling_max_timing(pps_game, window=3)

        rows.append({
            'athlete_id': int(athlete_id),
            'season': int(season),
            'games_played_pg': gp,

            'has_ws_last5': has_last5,
            'ws_minutes_last5': min_last5,
            'ws_fga_last5': fga_last5,
            'ws_pts_last5': pts_last5,
            'ws_pps_last5': pps_last5,
            'ws_on_net_rating_last5_mean': onnr_last5,

            'has_ws_last10': has_last10,
            'ws_minutes_last10': min_last10,
            'ws_fga_last10': fga_last10,
            'ws_pts_last10': pts_last10,
            'ws_pps_last10': pps_last10,
            'ws_on_net_rating_last10_mean': onnr_last10,

            'has_ws_prev5': has_prev5,
            'ws_delta_pps_last5_minus_prev5': ws_delta_pps_5,
            'ws_delta_minutes_last5_minus_prev5': ws_delta_min_5,

            'has_ws_prev10': has_prev10,
            'ws_delta_pps_last10_minus_prev10': ws_delta_pps_10,
            'ws_delta_minutes_last10_minus_prev10': ws_delta_min_10,

            'has_ws_breakout_timing_minutes': has_t_min,
            'ws_breakout_timing_minutes': timing_minutes,
            'has_ws_breakout_timing_volume': has_t_vol,
            'ws_breakout_timing_volume': timing_volume,
            'has_ws_breakout_timing_eff': has_t_eff,
            'ws_breakout_timing_eff': timing_eff,
        })

    out = pd.DataFrame(rows)

    # Ensure mask/value consistency.
    for m, v in [
        ('has_ws_last5', 'ws_minutes_last5'),
        ('has_ws_last10', 'ws_minutes_last10'),
        ('has_ws_breakout_timing_minutes', 'ws_breakout_timing_minutes'),
        ('has_ws_breakout_timing_volume', 'ws_breakout_timing_volume'),
        ('has_ws_breakout_timing_eff', 'ws_breakout_timing_eff'),
    ]:
        if m in out.columns and v in out.columns:
            out.loc[out[m] == 0, v] = np.nan

    return out


def main() -> None:
    logger.info(f"Loading player-game rows from {WAREHOUSE}...")
    df = load_player_games()
    logger.info(f"Loaded {len(df):,} player-game rows")

    logger.info("Building within-season windows...")
    out = build_windows(df)
    logger.info(f"Built {len(out):,} athlete-season rows")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
