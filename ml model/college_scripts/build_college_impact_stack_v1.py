"""
Build college_impact_stack_v1.parquet.

This is a pragmatic v1 assembler that standardizes impact-like signals at
athlete-season grain and exposes explicit reliability and missingness fields.
"""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import duckdb
import re


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE_DIR / "data/college_feature_store"
OUT_PATH = FEATURE_DIR / "college_impact_stack_v1.parquet"

IMPACT_IN_PATH = FEATURE_DIR / "college_impact_v1.parquet"
HIST_RAPM_PATH = BASE_DIR / "data/historical_rapm_results_lambda1000.csv"
HIST_RAPM_ENHANCED_PATH = BASE_DIR / "data/historical_rapm_results_enhanced.csv"
WAREHOUSE_DUCKDB_PATH = BASE_DIR / "data/warehouse.duckdb"

VERSION = "college_impact_stack_v1"


def _empty_output() -> pd.DataFrame:
    cols = [
        "athlete_id",
        "season",
        "team_id",
        "impact_version",
        "impact_source_mix",
        "impact_on_net_raw",
        "impact_on_ortg_raw",
        "impact_on_drtg_raw",
        "impact_off_net_raw",
        "impact_off_ortg_raw",
        "impact_off_drtg_raw",
        "impact_on_off_net_diff_raw",
        "impact_on_off_ortg_diff_raw",
        "impact_on_off_drtg_diff_raw",
        "impact_pm100_stint_raw",
        "impact_pm100_stint_non_garbage",
        "impact_pm100_stint_lev_wt",
        "rIPM_tot_std",
        "rIPM_off_std",
        "rIPM_def_std",
        "rIPM_tot_non_garbage",
        "rIPM_off_non_garbage",
        "rIPM_def_non_garbage",
        "rIPM_tot_lev_wt",
        "rIPM_off_lev_wt",
        "rIPM_def_lev_wt",
        "rIPM_tot_rubber",
        "rIPM_off_rubber",
        "rIPM_def_rubber",
        "rIPM_tot_opp_adj",
        "rIPM_off_opp_adj",
        "rIPM_def_opp_adj",
        "rIPM_tot_recency",
        "rIPM_off_recency",
        "rIPM_def_recency",
        "impact_poss_total",
        "impact_seconds_total",
        "impact_stints_total",
        "impact_ripm_sd_tot",
        "impact_ripm_sd_off",
        "impact_ripm_sd_def",
        "impact_reliability_weight",
        "has_impact_raw",
        "has_impact_stint",
        "has_impact_ripm",
    ]
    return pd.DataFrame(columns=cols)


def _safe_col(df: pd.DataFrame, col: str, fallback_col: str | None = None) -> pd.Series:
    """Return column as Series, falling back to another column or NaN Series."""
    if col in df.columns:
        return df[col]
    if fallback_col is not None and fallback_col in df.columns:
        return df[fallback_col]
    return pd.Series(np.nan, index=df.index)


def _norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().upper()
    # Historical scrape format is commonly "LAST,FIRST"; reorder to "FIRST LAST".
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            s = " ".join(parts[1:] + [parts[0]])
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


def _load_shooter_name_bridge() -> pd.DataFrame:
    if not WAREHOUSE_DUCKDB_PATH.exists():
        logger.warning("Warehouse DB missing for RAPM name bridge: %s", WAREHOUSE_DUCKDB_PATH)
        return pd.DataFrame(columns=["season", "norm_name", "athlete_id"])

    con = duckdb.connect(str(WAREHOUSE_DUCKDB_PATH), read_only=True)
    try:
        q = """
            SELECT
                g.season AS season,
                shooterAthleteId AS athlete_id,
                s.shooter_name,
                COUNT(*) AS shots
            FROM stg_shots s
            JOIN dim_games g
              ON CAST(s.gameId AS BIGINT) = g.id
            WHERE s.shooterAthleteId IS NOT NULL
              AND s.shooter_name IS NOT NULL
              AND g.season IS NOT NULL
            GROUP BY 1,2,3
        """
        bridge = con.execute(q).fetchdf()
    finally:
        con.close()

    if bridge.empty:
        return pd.DataFrame(columns=["season", "norm_name", "athlete_id"])

    bridge["season"] = pd.to_numeric(bridge["season"], errors="coerce").astype("Int64")
    bridge["athlete_id"] = pd.to_numeric(bridge["athlete_id"], errors="coerce").astype("Int64")
    bridge["norm_name"] = bridge["shooter_name"].map(_norm_name)
    bridge = bridge.dropna(subset=["season", "athlete_id"])
    bridge = bridge[bridge["norm_name"] != ""]
    # Deterministic tie-break: max shot count per (season, normalized_name).
    bridge = (
        bridge.sort_values(["season", "norm_name", "shots", "athlete_id"], ascending=[True, True, False, False])
        .drop_duplicates(subset=["season", "norm_name"], keep="first")
    )
    return bridge[["season", "norm_name", "athlete_id"]]


def _load_historical_rapm_mapped() -> pd.DataFrame:
    hist_path = HIST_RAPM_ENHANCED_PATH if HIST_RAPM_ENHANCED_PATH.exists() else HIST_RAPM_PATH
    if not hist_path.exists():
        logger.info("No historical RAPM file found at %s or %s", HIST_RAPM_ENHANCED_PATH, HIST_RAPM_PATH)
        return pd.DataFrame()

    hist = pd.read_csv(hist_path)
    if hist.empty or "player_name" not in hist.columns or "season" not in hist.columns:
        return pd.DataFrame()

    # Normalize historical schema names into canonical fields.
    if "rapm_standard" in hist.columns:
        hist["rapm_total"] = pd.to_numeric(hist["rapm_standard"], errors="coerce")
    elif "rapm" in hist.columns:
        hist["rapm_total"] = pd.to_numeric(hist["rapm"], errors="coerce")
    else:
        hist["rapm_total"] = np.nan

    hist["o_rapm"] = pd.to_numeric(hist["o_rapm"], errors="coerce") if "o_rapm" in hist.columns else np.nan
    hist["d_rapm"] = pd.to_numeric(hist["d_rapm"], errors="coerce") if "d_rapm" in hist.columns else np.nan
    hist["rapm_non_garbage"] = pd.to_numeric(hist["rapm_non_garbage"], errors="coerce") if "rapm_non_garbage" in hist.columns else hist["rapm_total"]
    hist["rapm_leverage_weighted"] = pd.to_numeric(hist["rapm_leverage_weighted"], errors="coerce") if "rapm_leverage_weighted" in hist.columns else hist["rapm_total"]
    hist["rapm_rubber_adj"] = pd.to_numeric(hist["rapm_rubber_adj"], errors="coerce") if "rapm_rubber_adj" in hist.columns else hist["rapm_total"]
    hist["poss_total"] = pd.to_numeric(hist["poss_total"], errors="coerce")
    
    hist["on_ortg"] = pd.to_numeric(hist["on_ortg"], errors="coerce") if "on_ortg" in hist.columns else np.nan
    hist["on_drtg"] = pd.to_numeric(hist["on_drtg"], errors="coerce") if "on_drtg" in hist.columns else np.nan
    hist["on_net_rating"] = pd.to_numeric(hist["on_net_rating"], errors="coerce") if "on_net_rating" in hist.columns else np.nan
    hist["off_ortg"] = pd.to_numeric(hist["off_ortg"], errors="coerce") if "off_ortg" in hist.columns else np.nan
    hist["off_drtg"] = pd.to_numeric(hist["off_drtg"], errors="coerce") if "off_drtg" in hist.columns else np.nan
    hist["off_net_rating"] = pd.to_numeric(hist["off_net_rating"], errors="coerce") if "off_net_rating" in hist.columns else np.nan
    
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce").astype("Int64")
    hist["norm_name"] = hist["player_name"].map(_norm_name)
    hist = hist.dropna(subset=["season", "poss_total"])
    hist = hist[hist["norm_name"] != ""]

    bridge = _load_shooter_name_bridge()
    if bridge.empty:
        logger.warning("Historical RAPM bridge is empty; cannot map to athlete_id.")
        return pd.DataFrame()

    mapped = hist.merge(bridge, on=["season", "norm_name"], how="inner")
    if mapped.empty:
        logger.warning("Historical RAPM mapping produced 0 rows.")
        return pd.DataFrame()

    # Keep best-exposure row per athlete-season.
    mapped = (
        mapped.sort_values(["athlete_id", "season", "poss_total"], ascending=[True, True, False])
        .drop_duplicates(subset=["athlete_id", "season"], keep="first")
        .reset_index(drop=True)
    )
    logger.info(
        "Historical RAPM mapped rows: %d (file=%s, with o_rapm=%d, with d_rapm=%d)",
        len(mapped),
        hist_path.name,
        int(mapped["o_rapm"].notna().sum()),
        int(mapped["d_rapm"].notna().sum()),
    )
    return mapped


def _load_player_game_on_off_season() -> pd.DataFrame:
    """
    Derive season-level on/off ratings from player-game and team-game tables.

    For each player-game:
      team_metric * team_seconds = on_metric * on_seconds + off_metric * off_seconds
    """
    if not WAREHOUSE_DUCKDB_PATH.exists():
        return pd.DataFrame()

    con = duckdb.connect(str(WAREHOUSE_DUCKDB_PATH), read_only=True)
    try:
        q = """
            WITH pg AS (
              SELECT
                CAST(pg.gameId AS VARCHAR) AS game_id,
                CAST(pg.athleteId AS BIGINT) AS athlete_id,
                CAST(pg.teamId AS BIGINT) AS team_id,
                CAST(pg.seconds_on AS DOUBLE) AS seconds_on,
                CAST(pg.on_ortg AS DOUBLE) AS on_ortg,
                CAST(pg.on_drtg AS DOUBLE) AS on_drtg,
                CAST(pg.on_net_rating AS DOUBLE) AS on_net
              FROM fact_player_game pg
              WHERE pg.athleteId IS NOT NULL
                AND pg.teamId IS NOT NULL
                AND pg.seconds_on IS NOT NULL
            ),
            tg AS (
              SELECT
                CAST(gameId AS VARCHAR) AS game_id,
                CAST(teamId AS BIGINT) AS team_id,
                CAST(seconds_game AS DOUBLE) AS team_seconds,
                CAST(offenseRating AS DOUBLE) AS team_ortg,
                CAST(defenseRating AS DOUBLE) AS team_drtg,
                CAST(netRating AS DOUBLE) AS team_net
              FROM fact_team_game
            ),
            g AS (
              SELECT CAST(id AS VARCHAR) AS game_id, CAST(season AS BIGINT) AS season
              FROM dim_games
              UNION ALL
              SELECT CAST(sourceId AS VARCHAR) AS game_id, CAST(season AS BIGINT) AS season
              FROM dim_games
              WHERE sourceId IS NOT NULL
            ),
            joined AS (
              SELECT
                pg.athlete_id,
                g.season,
                pg.seconds_on,
                GREATEST(tg.team_seconds - pg.seconds_on, 0.0) AS seconds_off,
                pg.on_ortg,
                pg.on_drtg,
                pg.on_net,
                tg.team_ortg,
                tg.team_drtg,
                tg.team_net,
                tg.team_seconds
              FROM pg
              JOIN tg
                ON pg.game_id = tg.game_id
               AND pg.team_id = tg.team_id
              JOIN g
                ON pg.game_id = g.game_id
              WHERE tg.team_seconds > 0
            ),
            game_level AS (
              SELECT
                athlete_id,
                season,
                seconds_on,
                seconds_off,
                on_ortg,
                on_drtg,
                on_net,
                CASE WHEN seconds_off > 0
                     THEN (team_ortg * team_seconds - on_ortg * seconds_on) / seconds_off
                     ELSE NULL END AS off_ortg,
                CASE WHEN seconds_off > 0
                     THEN (team_drtg * team_seconds - on_drtg * seconds_on) / seconds_off
                     ELSE NULL END AS off_drtg,
                CASE WHEN seconds_off > 0
                     THEN (team_net * team_seconds - on_net * seconds_on) / seconds_off
                     ELSE NULL END AS off_net
              FROM joined
            )
            SELECT
              athlete_id,
              season,
              SUM(seconds_on) AS seconds_on_total,
              SUM(seconds_off) AS seconds_off_total,
              CASE WHEN SUM(seconds_on) > 0 THEN SUM(on_ortg * seconds_on) / SUM(seconds_on) END AS impact_on_ortg_raw,
              CASE WHEN SUM(seconds_on) > 0 THEN SUM(on_drtg * seconds_on) / SUM(seconds_on) END AS impact_on_drtg_raw,
              CASE WHEN SUM(seconds_on) > 0 THEN SUM(on_net * seconds_on) / SUM(seconds_on) END AS impact_on_net_raw,
              CASE WHEN SUM(seconds_off) > 0 THEN SUM(off_ortg * seconds_off) / SUM(seconds_off) END AS impact_off_ortg_raw,
              CASE WHEN SUM(seconds_off) > 0 THEN SUM(off_drtg * seconds_off) / SUM(seconds_off) END AS impact_off_drtg_raw,
              CASE WHEN SUM(seconds_off) > 0 THEN SUM(off_net * seconds_off) / SUM(seconds_off) END AS impact_off_net_raw
            FROM game_level
            GROUP BY 1,2
        """
        df = con.execute(q).fetchdf()
    finally:
        con.close()

    if df.empty:
        return df

    for c in [
        "impact_on_ortg_raw",
        "impact_on_drtg_raw",
        "impact_on_net_raw",
        "impact_off_ortg_raw",
        "impact_off_drtg_raw",
        "impact_off_net_raw",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["impact_on_off_ortg_diff_raw"] = df["impact_on_ortg_raw"] - df["impact_off_ortg_raw"]
    # For defense, lower DRTG is better => OFF-ON is positive defensive impact.
    df["impact_on_off_drtg_diff_raw"] = df["impact_off_drtg_raw"] - df["impact_on_drtg_raw"]
    df["impact_on_off_net_diff_raw"] = df["impact_on_net_raw"] - df["impact_off_net_raw"]
    return df


def build_impact_stack() -> pd.DataFrame:
    if not IMPACT_IN_PATH.exists():
        logger.warning("Input missing: %s", IMPACT_IN_PATH)
        return _empty_output()

    src = pd.read_parquet(IMPACT_IN_PATH)
    if src.empty:
        logger.warning("Input is empty: %s", IMPACT_IN_PATH)
        return _empty_output()

    out = pd.DataFrame(
        {
            "athlete_id": src["athlete_id"],
            "season": src["season"].astype("Int64"),
            "team_id": _safe_col(src, "teamId"),
            "impact_version": VERSION,
        }
    )

    # Source mix: this table is mostly modern API-derived. Keep explicit provenance.
    out["impact_source_mix"] = np.where(out["season"] <= 2023, "manual_only", "api_only")

    out["impact_on_net_raw"] = _safe_col(src, "on_net_rating")
    out["impact_on_ortg_raw"] = _safe_col(src, "on_ortg")
    out["impact_on_drtg_raw"] = _safe_col(src, "on_drtg")
    out["impact_off_net_raw"] = np.nan
    out["impact_off_ortg_raw"] = np.nan
    out["impact_off_drtg_raw"] = np.nan
    out["impact_on_off_net_diff_raw"] = np.nan
    out["impact_on_off_ortg_diff_raw"] = np.nan
    out["impact_on_off_drtg_diff_raw"] = np.nan

    # Stint-level proxy fields: v1 re-labels available on-net/rapm as proxies
    # where independent stint-level data is unavailable.  Downstream consumers
    # should NOT treat these as independent features until real stint data lands.
    out["impact_pm100_stint_raw"] = _safe_col(src, "on_net_rating")
    out["impact_pm100_stint_non_garbage"] = _safe_col(src, "rapm_adjusted", "rapm_value")
    out["impact_pm100_stint_lev_wt"] = _safe_col(src, "rapm_value")

    base_ripm = _safe_col(src, "rapm_adjusted", "rapm_value")

    # Off/def split unavailable in current upstream tables; keep nulls explicit.
    for c in ["rIPM_off_std", "rIPM_def_std", "rIPM_off_non_garbage", "rIPM_def_non_garbage",
              "rIPM_off_lev_wt", "rIPM_def_lev_wt", "rIPM_off_rubber", "rIPM_def_rubber",
              "rIPM_off_opp_adj", "rIPM_def_opp_adj", "rIPM_off_recency", "rIPM_def_recency"]:
        out[c] = np.nan

    out["rIPM_tot_std"] = base_ripm
    out["rIPM_tot_non_garbage"] = base_ripm
    out["rIPM_tot_lev_wt"] = _safe_col(src, "rapm_value").fillna(base_ripm)
    out["rIPM_tot_rubber"] = base_ripm
    out["rIPM_tot_opp_adj"] = _safe_col(src, "rapm_adjusted").fillna(base_ripm)
    out["rIPM_tot_recency"] = base_ripm

    out["impact_poss_total"] = _safe_col(src, "poss_est")
    seconds_on = _safe_col(src, "seconds_on")
    out["impact_seconds_total"] = seconds_on
    out["impact_stints_total"] = np.where(seconds_on.notna(), seconds_on / 60.0, np.nan)

    poss = pd.to_numeric(out["impact_poss_total"], errors="coerce")
    # Approximate uncertainty from exposure. Lower poss -> higher sd.
    sd_tot = 1.75 / np.sqrt(np.clip(poss / 100.0, 1e-6, None))
    sd_tot = np.where(np.isfinite(sd_tot), sd_tot, np.nan)
    out["impact_ripm_sd_tot"] = sd_tot
    out["impact_ripm_sd_off"] = np.nan
    out["impact_ripm_sd_def"] = np.nan

    # Inverse-variance reliability, clipped.
    inv_var = 1.0 / np.clip(np.square(sd_tot), 1e-6, None)
    out["impact_reliability_weight"] = np.clip(inv_var, 0.05, 10.0)
    out.loc[np.isnan(sd_tot), "impact_reliability_weight"] = np.nan

    # Derive ON/OFF season features from player-game + team-game where available.
    on_off = _load_player_game_on_off_season()
    if not on_off.empty:
        merge_cols = ["athlete_id", "season"]
        enrich_cols = [
            "impact_on_ortg_raw", "impact_on_drtg_raw", "impact_on_net_raw",
            "impact_off_ortg_raw", "impact_off_drtg_raw", "impact_off_net_raw",
            "impact_on_off_ortg_diff_raw", "impact_on_off_drtg_diff_raw", "impact_on_off_net_diff_raw",
        ]
        out = out.merge(on_off[merge_cols + enrich_cols], on=merge_cols, how="left", suffixes=("", "_pg"))
        for c in enrich_cols:
            pg = f"{c}_pg"
            if pg in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").combine_first(pd.to_numeric(out[pg], errors="coerce"))
                out = out.drop(columns=[pg])

    out["has_impact_raw"] = out[
        [
            "impact_on_net_raw", "impact_on_ortg_raw", "impact_on_drtg_raw",
            "impact_off_net_raw", "impact_off_ortg_raw", "impact_off_drtg_raw",
            "impact_on_off_net_diff_raw", "impact_on_off_ortg_diff_raw", "impact_on_off_drtg_diff_raw",
        ]
    ].notna().any(axis=1).astype(int)
    out["has_impact_stint"] = out[["impact_pm100_stint_raw", "impact_pm100_stint_non_garbage", "impact_pm100_stint_lev_wt"]].notna().any(axis=1).astype(int)
    out["has_impact_ripm"] = out["rIPM_tot_std"].notna().astype(int)

    # Keep one row per athlete-season (max exposure when duplicates exist).
    out = out.sort_values(["athlete_id", "season", "impact_poss_total"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["athlete_id", "season"], keep="first").reset_index(drop=True)

    # Historical RAPM augmentation (from scrape-derived lineups).
    hist = _load_historical_rapm_mapped()
    if not hist.empty:
        hist_rows = pd.DataFrame(
            {
                "athlete_id": hist["athlete_id"].astype("Int64"),
                "season": hist["season"].astype("Int64"),
                "team_id": pd.Series(np.nan, index=hist.index),
                "impact_version": VERSION,
                "impact_source_mix": "historical_scrape",
                "impact_on_net_raw": hist["on_net_rating"],
                "impact_on_ortg_raw": hist["on_ortg"],
                "impact_on_drtg_raw": hist["on_drtg"],
                "impact_off_net_raw": hist["off_net_rating"],
                "impact_off_ortg_raw": hist["off_ortg"],
                "impact_off_drtg_raw": hist["off_drtg"],
                "impact_on_off_net_diff_raw": hist["on_net_rating"] - hist["off_net_rating"],
                "impact_on_off_ortg_diff_raw": hist["on_ortg"] - hist["off_ortg"],
                "impact_on_off_drtg_diff_raw": hist["off_drtg"] - hist["on_drtg"],
                "impact_pm100_stint_raw": hist["rapm_total"],
                "impact_pm100_stint_non_garbage": hist["rapm_non_garbage"],
                "impact_pm100_stint_lev_wt": hist["rapm_leverage_weighted"],
                "rIPM_off_std": hist["o_rapm"],
                "rIPM_def_std": hist["d_rapm"],
                "rIPM_off_non_garbage": hist["o_rapm"],
                "rIPM_def_non_garbage": hist["d_rapm"],
                "rIPM_off_lev_wt": hist["o_rapm"],
                "rIPM_def_lev_wt": hist["d_rapm"],
                "rIPM_off_rubber": hist["o_rapm"],
                "rIPM_def_rubber": hist["d_rapm"],
                "rIPM_off_opp_adj": hist["o_rapm"],
                "rIPM_def_opp_adj": hist["d_rapm"],
                "rIPM_off_recency": hist["o_rapm"],
                "rIPM_def_recency": hist["d_rapm"],
                "rIPM_tot_std": hist["rapm_total"],
                "rIPM_tot_non_garbage": hist["rapm_non_garbage"],
                "rIPM_tot_lev_wt": hist["rapm_leverage_weighted"],
                "rIPM_tot_rubber": hist["rapm_rubber_adj"],
                "rIPM_tot_opp_adj": hist["rapm_total"],
                "rIPM_tot_recency": hist["rapm_total"],
                "impact_poss_total": hist["poss_total"],
                "impact_seconds_total": pd.Series(np.nan, index=hist.index),
                "impact_stints_total": pd.Series(np.nan, index=hist.index),
            }
        )
        poss_h = pd.to_numeric(hist_rows["impact_poss_total"], errors="coerce")
        sd_tot_h = 1.75 / np.sqrt(np.clip(poss_h / 100.0, 1e-6, None))
        hist_rows["impact_ripm_sd_tot"] = sd_tot_h
        hist_rows["impact_ripm_sd_off"] = sd_tot_h
        hist_rows["impact_ripm_sd_def"] = sd_tot_h
        inv_var_h = 1.0 / np.clip(np.square(sd_tot_h), 1e-6, None)
        hist_rows["impact_reliability_weight"] = np.clip(inv_var_h, 0.05, 10.0)
        hist_rows["has_impact_raw"] = 0
        hist_rows["has_impact_stint"] = 1
        hist_rows["has_impact_ripm"] = 1

        out = pd.concat([out, hist_rows], ignore_index=True, sort=False)
        out = (
            out.sort_values(["athlete_id", "season", "impact_poss_total"], ascending=[True, True, False])
            .drop_duplicates(subset=["athlete_id", "season"], keep="first")
            .reset_index(drop=True)
        )

    return out


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out = build_impact_stack()
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %s (%d rows, %d cols)", OUT_PATH, len(out), len(out.columns))
    if not out.empty:
        logger.info("has_impact_ripm coverage: %.1f%%", 100.0 * out["has_impact_ripm"].mean())


if __name__ == "__main__":
    main()
