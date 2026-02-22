"""
Unified Training Table Builder
==============================
Merges college features + NBA targets + gaps into a single training matrix.

Inputs:
- data/college_feature_store/college_features_v1.parquet (per-season features)
- data/college_feature_store/prospect_career_v1.parquet (career aggregates)
- data/warehouse_v2/dim_player_nba_college_crosswalk.parquet
- data/warehouse_v2/fact_player_year1_epm.parquet
- data/warehouse_v2/fact_player_peak_rapm.parquet
- data/warehouse_v2/fact_player_nba_college_gaps.parquet
- data/warehouse_v2/fact_player_development_rate.parquet

Output:
- data/training/unified_training_table.parquet

Key Design Decisions:
1. Only include players with BOTH college features AND at least one NBA target
2. Tier 2 (spatial) features are included with explicit NaN (masked in model)
3. Era normalization applied to drift-prone features
4. Coverage masks added for Tier 2 availability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging
import os
import duckdb
import re
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
COLLEGE_FEATURE_STORE = BASE_DIR / "data/college_feature_store"
WAREHOUSE_V2 = BASE_DIR / "data" / "warehouse_v2"
DB_PATH = BASE_DIR / "data" / "warehouse.duckdb"
OUTPUT_DIR = BASE_DIR / "data" / "training"

def clean_name_for_join(name_series: pd.Series) -> pd.Series:
    """Standardize names for linkage patching."""
    s = name_series.astype(str).str.lower()
    s = s.str.replace(r'[^a-z\s]', '', regex=True)
    s = s.str.replace(r'\s+(jr|sr|iii|ii|iv|v)$', '', regex=True)
    return s.str.strip()
AUDIT_DIR = BASE_DIR / "data/audit"
PHYSICALS_DIR = BASE_DIR / "data/physicals"
COMBINE_DIR = BASE_DIR / "data/combine"

# 2026 contract-first paths (warehouse_v2), with backward-compatible fallback
COMBINE_MEASUREMENTS_PATH = WAREHOUSE_V2 / "fact_player_combine_measurements.parquet"
COMBINE_IMPUTED_PATH = WAREHOUSE_V2 / "fact_player_combine_imputed.parquet"
COMBINE_MEASUREMENTS_FALLBACK = COMBINE_DIR / "fact_player_combine_measurements.parquet"
COMBINE_IMPUTED_FALLBACK = COMBINE_DIR / "fact_player_combine_imputed.parquet"

from nba_scripts.games_played_selection import (
    select_games_played_with_provenance,
    select_minutes_with_provenance,
    games_source_mix_by_season,
    minutes_source_mix_by_season,
)

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Tier 1: Universal features (2010-2025, 100% coverage)
TIER1_SHOT_PROFILE = [
    'rim_att', 'rim_made', 'rim_fg_pct', 'rim_share',
    'mid_att', 'mid_made', 'mid_fg_pct', 'mid_share',
    'three_att', 'three_made', 'three_fg_pct', 'three_share',
    'ft_att', 'ft_made', 'ft_pct',
    'fga_total', 'shots_total',
]

TIER1_CREATION = [
    'assisted_made_rim', 'assisted_made_mid', 'assisted_made_three',
    'assisted_share_rim', 'assisted_share_mid', 'assisted_share_three',
]

TIER1_IMPACT = [
    'on_net_rating', 'on_ortg', 'on_drtg',
    'seconds_on', 'games_played', 'poss_on',
]

TIER1_CONTEXT = [
    'teamId',
    'team_pace', 'is_power_conf',
    'recruiting_rank', 'recruiting_stars', 'recruiting_rating',
    # Additional draft-time-safe context + “impact-adjacent” box signals
    'minutes_total',
    'ast_total', 'tov_total', 'stl_total', 'blk_total',
    'orb_total', 'drb_total', 'trb_total',
    # Restored activity/athleticism branch
    'dunk_rate', 'dunk_freq', 'putback_rate', 'rim_pressure_index', 'contest_proxy',
    'transition_freq', 'deflection_proxy', 'pressure_handle_proxy',
    'activity_source', 'has_activity_features',
    'dunk_rate_missing', 'dunk_freq_missing', 'putback_rate_missing',
    'rim_pressure_index_missing', 'contest_proxy_missing',
]

# Tier 2: Spatial features (2019+ only, ~25% coverage)
TIER2_SPATIAL = [
    'avg_shot_dist', 'shot_dist_var',
    'corner_3_rate', 'corner_3_pct',
    'deep_3_rate', 'rim_purity',
    'xy_shots', 'xy_3_shots', 'xy_rim_shots',  # Coverage counts
]

# Career aggregate features (from prospect_career_v1)
CAREER_FEATURES = [
    'career_years',
    'final_trueShootingPct', 'final_usage', 'final_poss_total',
    'final_rim_fg_pct', 'final_three_fg_pct', 'final_ft_pct',
    'final_avg_shot_dist', 'final_corner_3_rate', 'final_rim_purity',
    'slope_trueShootingPct', 'slope_usage', 'slope_rim_fg_pct', 'slope_three_fg_pct',
    'career_wt_trueShootingPct', 'career_wt_usage',
    'delta_trueShootingPct', 'delta_usage',
]

# Target columns
PRIMARY_TARGET = 'y_peak_ovr'
AUX_TARGETS = [
    'gap_ts_legacy',
    'year1_epm_tot',
    'y_peak_epm_3y',
    'year1_epm_off',
    'year1_epm_def',
    'dev_rate_y1_y3_mean',
]
BINARY_TARGET = 'made_nba'  # Derived: year1_mp >= 100

# Exposure/weight columns
EXPOSURE_COLS = ['year1_mp', 'peak_poss']
WINGSPAN_SCHEMA_COLS = [
    "wingspan_in",
    "standing_reach_in",
    "wingspan_minus_height_in",
    "has_wingspan",
]


def load_college_features(split_id: str = 'ALL__ALL') -> pd.DataFrame:
    """Load college season features, filtered to specified split."""
    path = COLLEGE_FEATURE_STORE / "college_features_v1.parquet"
    if not path.exists():
        logger.warning(f"College features not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    df = df[df['split_id'] == split_id].copy()
    # The feature store may contain duplicate rows for the same athlete/season/split.
    # These are typically exact duplicates from upstream joins; drop to prevent noise.
    df = df.drop_duplicates(subset=['athlete_id', 'season', 'split_id'])
    logger.info(f"Loaded {len(df):,} college feature rows (split={split_id})")
    return df


def load_college_features_all_splits() -> pd.DataFrame:
    """Load college features without split filtering (for leverage derivations)."""
    path = COLLEGE_FEATURE_STORE / "college_features_v1.parquet"
    if not path.exists():
        logger.warning(f"College features not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.drop_duplicates(subset=['athlete_id', 'season', 'split_id'])
    logger.info(f"Loaded {len(df):,} college feature rows (all splits)")
    return df


def build_final_season_leverage_features(college_features_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build leverage-rate features at athlete final-season grain from split windows.

    Uses split_id rows:
    - ALL__ALL
    - HIGH_LEVERAGE__ALL
    - GARBAGE__ALL
    """
    if college_features_all.empty:
        return pd.DataFrame()

    needed = {'athlete_id', 'season', 'split_id'}
    if not needed.issubset(college_features_all.columns):
        return pd.DataFrame()

    work = college_features_all[['athlete_id', 'season', 'split_id'] + [c for c in ['shots_total', 'fga_total', 'ft_att'] if c in college_features_all.columns]].copy()
    if 'shots_total' not in work.columns:
        return pd.DataFrame()

    wide = (
        work.pivot_table(
            index=['athlete_id', 'season'],
            columns='split_id',
            values='shots_total',
            aggfunc='sum',
        )
        .reset_index()
    )
    wide.columns = [str(c) for c in wide.columns]

    all_shots = pd.to_numeric(wide.get('ALL__ALL'), errors='coerce')
    high_shots = pd.to_numeric(wide.get('HIGH_LEVERAGE__ALL'), errors='coerce')
    garbage_shots = pd.to_numeric(wide.get('GARBAGE__ALL'), errors='coerce')

    wide['high_lev_att_rate'] = np.where(all_shots > 0, high_shots / all_shots, np.nan)
    wide['garbage_att_rate'] = np.where(all_shots > 0, garbage_shots / all_shots, np.nan)

    # Share of non-garbage leverage volume that occurs in high-leverage windows.
    non_garbage = all_shots - garbage_shots
    wide['leverage_poss_share'] = np.where(non_garbage > 0, high_shots / non_garbage, np.nan)

    # Final season selection.
    wide = wide.sort_values(['athlete_id', 'season'])
    final = wide.groupby('athlete_id').tail(1).reset_index(drop=True)
    final = final.rename(columns={'season': 'college_final_season'})
    keep = ['athlete_id', 'college_final_season', 'high_lev_att_rate', 'garbage_att_rate', 'leverage_poss_share']
    return final[[c for c in keep if c in final.columns]]


def load_team_strength_features() -> pd.DataFrame:
    """
    Load team-strength features by (teamId, season).
    Prefers long-horizon SRS proxy table for broad historical coverage.
    """
    db_path = BASE_DIR / "data" / "warehouse.duckdb"
    if not db_path.exists():
        logger.warning(f"warehouse.duckdb not found: {db_path}")
        return pd.DataFrame()

    try:
        con = duckdb.connect(str(db_path), read_only=True)
        df = con.execute(
            """
            SELECT
              CAST(teamId AS BIGINT) AS teamId,
              CAST(season AS BIGINT) AS season,
              CAST(srs_proxy_margin AS DOUBLE) AS college_team_srs,
              CAST(team_rank AS BIGINT) AS college_team_rank
            FROM v_team_season_srs_proxy
            """
        ).df()
        con.close()
        if df.empty:
            return df
        df['team_strength_srs'] = df['college_team_srs']
        logger.info(f"Loaded team-strength rows: {len(df):,}")
        return df
    except Exception as exc:
        logger.warning(f"Failed loading team strength from duckdb: {exc}")
        return pd.DataFrame()


def load_career_features() -> pd.DataFrame:
    """Load career aggregate features."""
    path = COLLEGE_FEATURE_STORE / "prospect_career_v1.parquet"
    if not path.exists():
        logger.warning(f"Career features not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} career feature rows")
    return df


def load_derived_box_stats() -> pd.DataFrame:
    """Load derived box stats (AST/STL/BLK/TOV) from PBP history."""
    path = COLLEGE_FEATURE_STORE / "derived_box_stats_v1.parquet"
    if not path.exists():
        logger.warning(f"Derived box stats not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} derived box stat rows")
    return df


def load_api_event_games_candidate() -> pd.DataFrame:
    """
    Build an API-era games-played candidate from event participants at athlete-season grain.
    Uses distinct gameId participation counts (broader than shot-only participation).
    """
    db_path = BASE_DIR / "data" / "warehouse.duckdb"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        df = con.execute(
            """
            SELECT
              CAST(p.u.id AS BIGINT) AS athlete_id,
              CAST(f.season AS BIGINT) AS season,
              COUNT(DISTINCT CAST(f.gameId AS BIGINT)) AS api_event_games_played
            FROM fact_play_raw f,
                 UNNEST(f.participants) AS p(u)
            WHERE p.u.id IS NOT NULL
              AND f.season IS NOT NULL
            GROUP BY 1,2
            """
        ).df()
        con.close()
    except Exception as exc:
        logger.warning(f"Failed loading API event games candidate: {exc}")
        return pd.DataFrame()
    if df.empty:
        return df
    df["api_event_games_played"] = pd.to_numeric(df["api_event_games_played"], errors="coerce")
    logger.info(f"Loaded API event games candidate rows: {len(df):,}")
    return df


def load_activity_proxies() -> pd.DataFrame:
    """Load enhanced activity features if available."""
    path = COLLEGE_FEATURE_STORE / "enhanced_features_v1.parquet"
    if not path.exists():
        logger.warning(f"Enhanced activity features not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    keep = [
        "season", "athlete_id", "split_id",
        "dunk_rate", "dunk_freq", "putback_rate", "transition_freq", "transition_eff",
        "rim_pressure_index", "deflection_proxy", "contest_proxy", "pressure_handle_proxy",
        "dunk_rate_missing", "dunk_freq_missing", "putback_rate_missing",
        "rim_pressure_index_missing", "contest_proxy_missing",
        "activity_source", "has_activity_features",
    ]
    keep = [c for c in keep if c in df.columns]
    logger.info(f"Loaded enhanced activity rows: {len(df):,}")
    return df[keep].copy()


def _norm_hist_player_name(name: str) -> str:
    """Normalize historical names into a comparable key."""
    if not isinstance(name, str):
        return ""
    s = name.strip().upper()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            s = " ".join(parts[1:] + [parts[0]])
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


def load_historical_exposure_backfill() -> pd.DataFrame:
    """
    Load name-based historical exposure backfill and map to athlete_id.

    Source: warehouse_v2/fact_player_season_stats_backfill.parquet
    Mapping: season + normalized player name -> athlete_id using stg_shots name bridge.
    """
    backfill_path = WAREHOUSE_V2 / "fact_player_season_stats_backfill_manual_subs.parquet"
    db_path = BASE_DIR / "data" / "warehouse.duckdb"
    if (not backfill_path.exists()) or (not db_path.exists()):
        return pd.DataFrame()

    bf = pd.read_parquet(backfill_path)
    if bf.empty or ("season" not in bf.columns) or ("player_name" not in bf.columns):
        return pd.DataFrame()

    # Build deterministic bridge from API-era shooter names.
    def _norm_hist_team_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        name = re.sub(r"^#\d+\s+", "", name.strip())
        name = name.upper().replace("STATE", "ST").replace("NORTH ", "N ").replace("SOUTH ", "S ")
        return re.sub(r"[^A-Z]+", "", name)

    try:
        con = duckdb.connect(str(db_path), read_only=True)
        bridge = con.execute(
            """
            WITH b AS (
              SELECT
                CAST(g.season AS BIGINT) AS season,
                CAST(s.shooterAthleteId AS BIGINT) AS athlete_id,
                s.shooter_name,
                t.school AS team_name,
                COUNT(*) AS shots
              FROM stg_shots s
              JOIN dim_games g
                ON CAST(s.gameId AS BIGINT) = g.id
              LEFT JOIN dim_teams t
                ON t.id = CAST(s.teamId AS BIGINT)
              WHERE s.shooterAthleteId IS NOT NULL
                AND s.shooter_name IS NOT NULL
                AND g.season IS NOT NULL
              GROUP BY 1,2,3,4
            )
            SELECT season, athlete_id, shooter_name, team_name, shots
            FROM b
            """
        ).df()
        con.close()
    except Exception as exc:
        logger.warning(f"Failed to build historical exposure name bridge: {exc}")
        return pd.DataFrame()

    if bridge.empty:
        return pd.DataFrame()

    bridge["season"] = pd.to_numeric(bridge["season"], errors="coerce").astype("Int64")
    bridge["athlete_id"] = pd.to_numeric(bridge["athlete_id"], errors="coerce").astype("Int64")
    bridge["norm_name"] = bridge["shooter_name"].map(_norm_hist_player_name)
    bridge["norm_team"] = bridge["team_name"].map(_norm_hist_team_name)
    bridge = bridge.dropna(subset=["season", "athlete_id"])
    bridge = bridge[bridge["norm_name"] != ""]
    bridge = (
        bridge.sort_values(["season", "norm_name", "norm_team", "shots", "athlete_id"], ascending=[True, True, True, False, False])
        .drop_duplicates(subset=["season", "norm_name", "norm_team"], keep="first")
    )

    bf = bf.copy()
    # Historical manual PBP season labeling can vary by source (start-year vs end-year).
    # Emit both candidates and resolve deterministically downstream.
    bf["season_raw"] = pd.to_numeric(bf["season"], errors="coerce").astype("Int64")
    bf["norm_name"] = bf["player_name"].map(_norm_hist_player_name)
    bf["norm_team"] = bf.get("team_name", pd.Series(dtype=str)).map(_norm_hist_team_name)
    bf = bf.dropna(subset=["season_raw"])
    bf = bf[bf["norm_name"] != ""]
    bf_no_shift = bf.copy()
    bf_no_shift["season"] = bf_no_shift["season_raw"]
    bf_no_shift["season_variant"] = "no_shift"
    bf_plus_one = bf.copy()
    bf_plus_one["season"] = bf_plus_one["season_raw"] + 1
    bf_plus_one["season_variant"] = "plus_one"
    bf = pd.concat([bf_no_shift, bf_plus_one], ignore_index=True)

    # Merge exact team matches first to prevent Jalen Johnson overlaps
    mapped_exact = bf.merge(bridge[["season", "norm_name", "norm_team", "athlete_id"]], on=["season", "norm_name", "norm_team"], how="inner")
    
    # For remaining bf rows, merge on just name IF bridge only has 1 instance of that name
    if not mapped_exact.empty:
        matched_keys = mapped_exact[["season", "norm_name", "norm_team"]].drop_duplicates()
        bf_unmatched = bf.merge(matched_keys, on=["season", "norm_name", "norm_team"], how="left", indicator=True)
        bf_unmatched = bf_unmatched[bf_unmatched["_merge"] == "left_only"].drop(columns=["_merge"])
    else:
        bf_unmatched = bf.copy()

    bridge_unique = bridge.groupby(["season", "norm_name"]).filter(lambda x: len(x) == 1)
    mapped_fallback = bf_unmatched.merge(bridge_unique[["season", "norm_name", "athlete_id"]], on=["season", "norm_name"], how="inner")
    mapped_fallback = mapped_fallback.drop(columns=["norm_team_y"], errors="ignore").rename(columns={"norm_team_x": "norm_team"})

    # 3. Last Name Suffix Fallback (for pre-2019 "OKOGIE" -> "JOSHOKOGIE" cases on exact team)
    already_matched_fallback = mapped_fallback[["season", "norm_name", "norm_team"]].drop_duplicates()
    bf_still_unmatched = bf_unmatched.merge(already_matched_fallback, on=["season", "norm_name", "norm_team"], how="left", indicator=True)
    bf_still_unmatched = bf_still_unmatched[bf_still_unmatched["_merge"] == "left_only"].drop(columns=["_merge"])
    
    b_sub = bridge[["season", "norm_team", "norm_name", "athlete_id"]].rename(columns={"norm_name": "bridge_name"})
    cross = bf_still_unmatched.merge(b_sub, on=["season", "norm_team"], how="inner")
    
    if not cross.empty:
        cross["is_suffix"] = cross.apply(lambda r: str(r["bridge_name"]).endswith(str(r["norm_name"])) and len(str(r["bridge_name"])) > len(str(r["norm_name"])), axis=1)
        mapped_suffix = cross[cross["is_suffix"]].copy()
        
        if not mapped_suffix.empty:
            # Deduplicate suffixes (e.g. if JSMITH and DSMITH both match SMITH on same team, drop both safely)
            suffix_counts = mapped_suffix.groupby(["season", "norm_team", "norm_name"]).size().reset_index(name="n")
            mapped_suffix = mapped_suffix.merge(suffix_counts, on=["season", "norm_team", "norm_name"])
            mapped_suffix = mapped_suffix[mapped_suffix["n"] == 1].drop(columns=["n", "bridge_name", "is_suffix"])
        else:
            mapped_suffix = pd.DataFrame()
    else:
        mapped_suffix = pd.DataFrame()

    mapped = pd.concat([mapped_exact, mapped_fallback, mapped_suffix], ignore_index=True)

    if mapped.empty:
        logger.warning("Historical exposure backfill mapping produced 0 rows")
        return pd.DataFrame()

    # Collapse possible team-level duplicates into athlete-season totals.
    # We take the max here instead of sum to protect against any residual name collision artifacts.
    # Choose best season-variant candidate per athlete-season.
    mapped["minutes_derived"] = pd.to_numeric(mapped.get("minutes_derived"), errors="coerce")
    mapped["turnovers_derived"] = pd.to_numeric(mapped.get("turnovers_derived"), errors="coerce")
    mapped["games_derived"] = pd.to_numeric(mapped.get("games_derived"), errors="coerce")
    mapped["variant_priority"] = np.where(mapped.get("season_variant").astype(str) == "plus_one", 1, 0)
    winner = (
        mapped.sort_values(
            ["athlete_id", "season", "games_derived", "minutes_derived", "variant_priority"],
            ascending=[True, True, False, False, False],
        )
        .drop_duplicates(subset=["athlete_id", "season"], keep="first")
    )
    grouped = winner[["athlete_id", "season"]].copy()
    grouped["backfill_minutes_total"] = pd.to_numeric(winner["minutes_derived"], errors="coerce").to_numpy()
    grouped["backfill_tov_total"] = pd.to_numeric(winner["turnovers_derived"], errors="coerce").to_numpy()
    grouped["backfill_games_played"] = pd.to_numeric(winner["games_derived"], errors="coerce").to_numpy()
    grouped["backfill_alignment_variant"] = winner.get("season_variant", pd.Series("none", index=winner.index)).astype(str).to_numpy()
    if grouped["backfill_games_played"].isna().all():
        # Fallback approximation when games_derived is absent in backfill.
        mins = pd.to_numeric(grouped["backfill_minutes_total"], errors="coerce")
        grouped["backfill_games_played"] = np.clip(np.round(mins / 25.0), 1, 40)
    logger.info(f"Loaded mapped historical exposure backfill rows: {len(grouped):,}")
    return grouped


def load_historical_text_games_backfill() -> pd.DataFrame:
    """
    Backfill games played from historical manual PBP text.

    `fact_play_historical_combined.parquet` seasons are start-year (e.g., 2021-22 -> 2021),
    while feature surfaces use end-year (2022), so we map `season = season_start + 1`.
    """
    hist_path = BASE_DIR / "data" / "fact_play_historical_combined.parquet"
    db_path = BASE_DIR / "data" / "warehouse.duckdb"
    if (not hist_path.exists()) or (not db_path.exists()):
        return pd.DataFrame()

    try:
        con = duckdb.connect(str(db_path), read_only=True)
        bridge = con.execute(
            """
            WITH b AS (
              SELECT
                CAST(g.season AS BIGINT) AS season,
                CAST(s.shooterAthleteId AS BIGINT) AS athlete_id,
                s.shooter_name,
                COUNT(*) AS shots
              FROM stg_shots s
              JOIN dim_games g
                ON CAST(s.gameId AS BIGINT) = g.id
              WHERE s.shooterAthleteId IS NOT NULL
                AND s.shooter_name IS NOT NULL
                AND g.season IS NOT NULL
              GROUP BY 1,2,3
            )
            SELECT season, athlete_id, shooter_name, shots
            FROM b
            """
        ).df()
        con.close()
    except Exception as exc:
        logger.warning(f"Failed building historical text-games season-aware bridge: {exc}")
        return pd.DataFrame()

    if bridge.empty:
        return pd.DataFrame()

    bridge["season"] = pd.to_numeric(bridge["season"], errors="coerce").astype("Int64")
    bridge["athlete_id"] = pd.to_numeric(bridge["athlete_id"], errors="coerce").astype("Int64")
    bridge["norm_name"] = bridge["shooter_name"].map(_norm_hist_player_name)
    bridge = bridge.dropna(subset=["season", "athlete_id"])
    bridge = bridge[bridge["norm_name"] != ""]
    bridge = (
        bridge.sort_values(
            ["season", "norm_name", "shots", "athlete_id"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["season", "norm_name"], keep="first")
    )

    try:
        con2 = duckdb.connect()
        hist = con2.execute(
            f"""
            WITH base AS (
              SELECT
                CAST(season AS BIGINT) AS season_start,
                CAST(gameSourceId AS VARCHAR) AS game_id,
                TRIM(SPLIT_PART(playText, '|', 2)) AS home_evt,
                TRIM(SPLIT_PART(playText, '|', 4)) AS away_evt
              FROM read_parquet('{hist_path.as_posix()}')
              WHERE season IS NOT NULL
            ),
            names AS (
              SELECT season_start, game_id, REGEXP_EXTRACT(home_evt, '^([^,|]+),', 1) AS player_name
              FROM base
              UNION ALL
              SELECT season_start, game_id, REGEXP_EXTRACT(away_evt, '^([^,|]+),', 1) AS player_name
              FROM base
            )
            SELECT
              CAST(season_start AS BIGINT) AS season,
              player_name,
              COUNT(DISTINCT game_id) AS hist_games_played_text,
              'no_shift' AS season_variant
            FROM names
            WHERE player_name IS NOT NULL
              AND TRIM(player_name) <> ''
              AND UPPER(TRIM(player_name)) <> 'TEAM'
              AND UPPER(TRIM(player_name)) <> 'TEAM.'
            GROUP BY 1,2,4
            UNION ALL
            SELECT
              CAST(season_start + 1 AS BIGINT) AS season,
              player_name,
              COUNT(DISTINCT game_id) AS hist_games_played_text,
              'plus_one' AS season_variant
            FROM names
            WHERE player_name IS NOT NULL
              AND TRIM(player_name) <> ''
              AND UPPER(TRIM(player_name)) <> 'TEAM'
              AND UPPER(TRIM(player_name)) <> 'TEAM.'
            GROUP BY 1,2,4
            """
        ).df()
        con2.close()
    except Exception as exc:
        logger.warning(f"Failed loading historical text-games counts: {exc}")
        return pd.DataFrame()

    if hist.empty:
        return pd.DataFrame()

    hist["norm_name"] = hist["player_name"].map(_norm_hist_player_name)
    hist = hist[hist["norm_name"] != ""]
    mapped = hist.merge(
        bridge[["season", "norm_name", "athlete_id"]],
        on=["season", "norm_name"],
        how="inner",
    )
    if mapped.empty:
        return pd.DataFrame()

    mapped["hist_games_played_text"] = pd.to_numeric(mapped["hist_games_played_text"], errors="coerce")
    mapped["variant_priority"] = np.where(mapped.get("season_variant").astype(str) == "plus_one", 1, 0)
    winner = (
        mapped.sort_values(
            ["athlete_id", "season", "hist_games_played_text", "variant_priority"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["athlete_id", "season"], keep="first")
    )
    out = (
        winner.groupby(["athlete_id", "season"], as_index=False)["hist_games_played_text"]
        .max()
    )
    variant = (
        winner.sort_values(["athlete_id", "season"])
        .drop_duplicates(subset=["athlete_id", "season"], keep="first")[["athlete_id", "season", "season_variant"]]
        .rename(columns={"season_variant": "hist_alignment_variant"})
    )
    out = out.merge(variant, on=["athlete_id", "season"], how="left")
    out["hist_games_played_text"] = pd.to_numeric(out["hist_games_played_text"], errors="coerce")
    n_clip = int((out["hist_games_played_text"] > 45).sum())
    if n_clip > 0:
        logger.warning(f"Clipping {n_clip} historical text-games rows above 45 games")
        out.loc[out["hist_games_played_text"] > 45, "hist_games_played_text"] = 45
    logger.info(f"Loaded historical text games backfill rows: {len(out):,}")
    return out


def load_trajectory_features() -> pd.DataFrame:
    """Load trajectory stub features (list-valued sequences)."""
    path = OUTPUT_DIR / "trajectory_stub_v1.parquet"
    if not path.exists():
        logger.warning(f"Trajectory features not found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} trajectory feature rows")
    return df


def load_college_impact_stack() -> pd.DataFrame:
    """Load college impact stack features (athlete-season grain)."""
    path = COLLEGE_FEATURE_STORE / "college_impact_stack_v1.parquet"
    if not path.exists():
        logger.warning(f"College impact stack not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} college impact stack rows")
    return df


def load_college_pathway_context() -> pd.DataFrame:
    """Load college pathway context v2 features (athlete-season grain)."""
    path = COLLEGE_FEATURE_STORE / "college_pathway_context_v2.parquet"
    if not path.exists():
        logger.warning(f"College pathway context v2 not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} college pathway context rows")
    return df


def load_college_dev_rate() -> pd.DataFrame:
    """Load college development-rate labels/features (athlete grain)."""
    path = COLLEGE_FEATURE_STORE / "fact_player_college_development_rate.parquet"
    if not path.exists():
        logger.warning(f"College dev-rate table not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} college dev-rate rows")
    return df


def load_transfer_context_summary() -> pd.DataFrame:
    """Load transfer context and summarize to athlete grain for training joins."""
    path = COLLEGE_FEATURE_STORE / "fact_player_transfer_context.parquet"
    if not path.exists():
        logger.warning(f"Transfer context table not found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if df.empty:
        logger.info("Transfer context table is empty")
        return df

    # Aggregate to one row per athlete for stable join behavior.
    grouped = df.groupby("athlete_id", as_index=False).agg(
        transfer_event_count=("athlete_id", "count"),
        transfer_max_shock=("transfer_shock_score", "max"),
        transfer_mean_shock=("transfer_shock_score", "mean"),
        transfer_mean_perf_delta_raw=("transfer_perf_delta_raw", "mean"),
        transfer_mean_perf_delta_context_adj=("transfer_perf_delta_context_adj", "mean"),
        transfer_conf_delta_mean=("transfer_conf_delta", "mean"),
        transfer_conf_delta_max_abs=("transfer_conf_delta", lambda s: np.nanmax(np.abs(s.to_numpy(dtype=float))) if s.notna().any() else np.nan),
        transfer_pace_delta_mean=("transfer_pace_delta", "mean"),
        transfer_pace_delta_max_abs=("transfer_pace_delta", lambda s: np.nanmax(np.abs(s.to_numpy(dtype=float))) if s.notna().any() else np.nan),
        transfer_role_delta_mean=("transfer_role_delta", "mean"),
        transfer_role_delta_max_abs=("transfer_role_delta", lambda s: np.nanmax(np.abs(s.to_numpy(dtype=float))) if s.notna().any() else np.nan),
    )
    grouped["has_transfer_context"] = (grouped["transfer_event_count"] > 0).astype(int)
    logger.info(f"Loaded transfer summary rows: {len(grouped):,}")
    return grouped


def load_crosswalk() -> pd.DataFrame:
    """Load college-to-NBA crosswalk."""
    path = WAREHOUSE_V2 / "dim_player_nba_college_crosswalk.parquet"
    if not path.exists():
        logger.warning(f"Crosswalk not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} crosswalk entries")
    return df


def load_dim_player_nba() -> pd.DataFrame:
    """Load NBA dimension table for cohort filters/metadata."""
    path = WAREHOUSE_V2 / "dim_player_nba.parquet"
    if not path.exists():
        logger.warning(f"NBA dim not found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    keep = [
        c for c in [
            "nba_id", "draft_year", "rookie_season_year", "player_name",
            "bbr_id",
            "ht_first", "ht_max", "wt_first", "wt_max",
            "ht_peak_delta", "wt_peak_delta",
        ] if c in df.columns
    ]
    df = df[keep].drop_duplicates(subset=["nba_id"])
    logger.info(f"Loaded {len(df):,} NBA dimension rows")
    return df


def load_nba_wingspan_bridge() -> pd.DataFrame:
    """
    Load NBA wingspan (ws) from basketball_excel and expose bbr_id keyed bridge.
    ws is stored in centimeters in basketball_excel; convert to inches.
    """
    path = BASE_DIR / "data" / "basketball_excel" / "all_players.parquet"
    if not path.exists():
        logger.warning(f"NBA ws bridge source not found: {path}")
        return pd.DataFrame()
    try:
        raw = pd.read_parquet(path, columns=["bbr_pid", "ws"])
        if raw.empty:
            return pd.DataFrame()
        raw["bbr_id"] = raw["bbr_pid"].astype(str).str.strip().str.lower()
        ws_cm = pd.to_numeric(raw["ws"], errors="coerce")
        ws_cm = ws_cm.where((ws_cm >= 120.0) & (ws_cm <= 260.0), np.nan)
        raw["nba_wingspan_cm"] = ws_cm
        out = (
            raw.dropna(subset=["bbr_id"])
            .groupby("bbr_id", as_index=False)["nba_wingspan_cm"]
            .median()
        )
        out["nba_wingspan_in"] = out["nba_wingspan_cm"] / 2.54
        logger.info(f"Loaded NBA ws bridge rows: {len(out):,}")
        return out
    except Exception as exc:
        logger.warning(f"Failed loading NBA ws bridge: {exc}")
        return pd.DataFrame()


def _to_lbs_from_mixed_weight(series: pd.Series) -> pd.Series:
    """
    Convert mixed-unit NBA weight surface to pounds.
    If value looks like kg (50..180), convert to lbs. If already lbs (>=130), keep.
    """
    x = pd.to_numeric(series, errors="coerce")
    kg_mask = (x >= 50.0) & (x <= 180.0)
    lbs_mask = (x >= 130.0) & (x <= 380.0)
    out = x.copy()
    out.loc[kg_mask & ~lbs_mask] = out.loc[kg_mask & ~lbs_mask] * 2.2046226218
    out = out.where((out >= 110.0) & (out <= 380.0), np.nan)
    return out


def load_recruiting_physicals() -> pd.DataFrame:
    """
    Load prospect physicals from recruiting records at athlete grain.
    Uses robust aggregation + sanity clipping to avoid malformed entries.
    """
    db_path = BASE_DIR / "data" / "warehouse.duckdb"
    if not db_path.exists():
        logger.warning(f"warehouse.duckdb not found: {db_path}")
        return pd.DataFrame()
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        df = con.execute(
            """
            WITH base AS (
              SELECT
                CAST(athleteId AS BIGINT) AS athlete_id,
                CAST(heightInches AS DOUBLE) AS height_in,
                CAST(weightPounds AS DOUBLE) AS weight_lbs
              FROM fact_recruiting_players
              WHERE athleteId IS NOT NULL
            ),
            clean AS (
              SELECT
                athlete_id,
                CASE WHEN height_in BETWEEN 58 AND 90 THEN height_in ELSE NULL END AS height_in,
                CASE WHEN weight_lbs BETWEEN 130 AND 400 THEN weight_lbs ELSE NULL END AS weight_lbs
              FROM base
            )
            SELECT
              athlete_id,
              median(height_in) AS recruit_height_in,
              median(weight_lbs) AS recruit_weight_lbs
            FROM clean
            GROUP BY 1
            """
        ).df()
        con.close()
        logger.info(f"Loaded recruiting physicals rows: {len(df):,}")
        return df
    except Exception as exc:
        logger.warning(f"Failed loading recruiting physicals: {exc}")
        return pd.DataFrame()


def load_college_physicals_by_season() -> pd.DataFrame:
    """Load canonical college physicals by season from warehouse v2 (preferred) or duckdb table."""
    pq_path = WAREHOUSE_V2 / "fact_college_player_physicals_by_season.parquet"
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        logger.info(f"Loaded canonical college physicals rows: {len(df):,}")
        return df

    db_path = BASE_DIR / "data" / "warehouse.duckdb"
    if not db_path.exists():
        logger.warning("No physicals source found (warehouse_v2 parquet + duckdb missing)")
        return pd.DataFrame()
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        df = con.execute("SELECT * FROM fact_college_player_physicals_by_season").df()
        con.close()
        logger.info(f"Loaded canonical college physicals rows from duckdb: {len(df):,}")
        return df
    except Exception as exc:
        logger.warning(f"Failed loading canonical college physicals: {exc}")
        return pd.DataFrame()


def load_college_physical_trajectory() -> pd.DataFrame:
    """Load college physical trajectory features."""
    pq_path = WAREHOUSE_V2 / "fact_college_player_physical_trajectory.parquet"
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        logger.info(f"Loaded college physical trajectory rows: {len(df):,}")
        return df

    db_path = BASE_DIR / "data" / "warehouse.duckdb"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        df = con.execute("SELECT * FROM fact_college_player_physical_trajectory").df()
        con.close()
        logger.info(f"Loaded college physical trajectory rows from duckdb: {len(df):,}")
        return df
    except Exception as exc:
        logger.warning(f"Failed loading college physical trajectory: {exc}")
        return pd.DataFrame()


def load_nba_targets() -> pd.DataFrame:
    """Load and merge NBA target tables."""
    targets = pd.DataFrame()
    
    # Peak RAPM (primary target)
    peak_path = WAREHOUSE_V2 / "fact_player_peak_rapm.parquet"
    if peak_path.exists():
        peak = pd.read_parquet(peak_path)
        
        # Enforce user request: if a player has < 2000 peak possessions, their RAPM
        # is discarded, functionally treating them as a non-NBA player target.
        if 'peak_poss' in peak.columns:
            mask = peak['peak_poss'] < 2000
            for col in ['y_peak_ovr', 'y_peak_off', 'y_peak_def']:
                if col in peak.columns:
                    peak.loc[mask, col] = np.nan
                    
        targets = peak[['nba_id', 'y_peak_ovr', 'y_peak_off', 'y_peak_def', 'peak_poss']].copy()
        logger.info(f"Loaded {len(peak):,} peak RAPM targets")
    
    # Year-1 EPM (auxiliary targets)
    y1_path = WAREHOUSE_V2 / "fact_player_year1_epm.parquet"
    if y1_path.exists():
        y1 = pd.read_parquet(y1_path)
        y1_cols = ['nba_id', 'year1_epm_tot', 'year1_epm_off', 'year1_epm_def', 
                   'year1_mp', 'year1_tspct', 'year1_usg']
        y1_cols = [c for c in y1_cols if c in y1.columns]
        
        if targets.empty:
            targets = y1[y1_cols].copy()
        else:
            targets = targets.merge(y1[y1_cols], on='nba_id', how='outer')
        logger.info(f"Loaded {len(y1):,} year-1 EPM targets")

    # Peak/rolling EPM (auxiliary or optional primary for EPM-first runs)
    peak_epm_path = WAREHOUSE_V2 / "fact_player_peak_epm.parquet"
    if peak_epm_path.exists():
        pepm = pd.read_parquet(peak_epm_path)
        p_cols = [c for c in ['nba_id', 'y_peak_epm_1y', 'y_peak_epm_1y_60gp', 'y_peak_epm_2y', 'y_peak_epm_3y', 'y_peak_epm_window', 'epm_obs_seasons', 'epm_obs_minutes', 'epm_peak_window_end_year'] if c in pepm.columns]
        if targets.empty:
            targets = pepm[p_cols].copy()
        else:
            targets = targets.merge(pepm[p_cols], on='nba_id', how='outer')
        logger.info(f"Loaded {len(pepm):,} peak/rolling EPM targets")

    # Horizon-bounded EPM trajectory (no future leakage): latent peak, slope, plateau, censor
    traj_path = WAREHOUSE_V2 / "fact_player_epm_trajectory_horizon.parquet"
    if traj_path.exists():
        traj = pd.read_parquet(traj_path)
        t_cols = [c for c in ['nba_id', 'latent_peak_within_7y', 'slope_last_2y', 'plateau_flag', 'epm_trajectory_n_seasons_used', 'epm_trajectory_censored'] if c in traj.columns]
        if t_cols:
            if targets.empty:
                targets = traj[t_cols].copy()
            else:
                targets = targets.merge(traj[t_cols], on='nba_id', how='outer')
            logger.info(f"Loaded {len(traj):,} horizon-bounded EPM trajectory targets")
    
    # Gaps (auxiliary targets)
    gaps_path = WAREHOUSE_V2 / "fact_player_nba_college_gaps.parquet"
    if gaps_path.exists():
        gaps = pd.read_parquet(gaps_path)
        gap_cols = ['nba_id', 'gap_ts_legacy', 'gap_usg_legacy', 'gap_dist_leap', 'gap_corner_rate']
        gap_cols = [c for c in gap_cols if c in gaps.columns]
        
        if targets.empty:
            targets = gaps[gap_cols].copy()
        else:
            targets = targets.merge(gaps[gap_cols], on='nba_id', how='outer')
        logger.info(f"Loaded {len(gaps):,} gap targets")

    # Development-rate fact (auxiliary target + quality metadata)
    dev_path = WAREHOUSE_V2 / "fact_player_development_rate.parquet"
    if dev_path.exists():
        dev = pd.read_parquet(dev_path)
        dev_cols = [
            'nba_id',
            'dev_rate_y1_y3_mean',
            'dev_rate_y1_y3_sd',
            'dev_rate_y1_y3_p10',
            'dev_rate_y1_y3_p50',
            'dev_rate_y1_y3_p90',
            'dev_rate_quality_weight',
            'dev_has_y1',
            'dev_has_y2',
            'dev_has_y3',
            'dev_has_rapm3y',
            'dev_obs_epm_count',
            'dev_obs_rapm_count',
            'dev_model_version',
        ]
        dev_cols = [c for c in dev_cols if c in dev.columns]

        if targets.empty:
            targets = dev[dev_cols].copy()
        else:
            targets = targets.merge(dev[dev_cols], on='nba_id', how='outer')
        logger.info(f"Loaded {len(dev):,} development-rate targets")
    
    return targets


def get_final_college_season(college_features: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the final college season for each athlete.
    This is what we use to predict NBA outcomes.
    """
    if college_features.empty:
        return pd.DataFrame()
    
    # Filter to only the primary ALL__ALL split before doing any aggregation.
    # Otherwise, summing across split_ids will violently inflate games_played
    # and all count-based stats by the number of split permutations.
    if 'split_id' in college_features.columns:
        df = college_features[college_features['split_id'] == 'ALL__ALL'].copy()
    else:
        df = college_features.copy()
        
    if df.empty:
        return pd.DataFrame()

    # Sort and get last season.
    # Important: GroupBy.last() can skip NaNs per-column, which can accidentally pull
    # values from earlier seasons. We want the literal final season row.
    df = df.sort_values(['athlete_id', 'season'])
    # Remove duplicate fragments while preserving legitimate transfer multi-team rows.
    dedupe_keys = [k for k in ['athlete_id', 'season', 'split_id', 'teamId'] if k in df.columns]
    if dedupe_keys:
        df = df.drop_duplicates(subset=dedupe_keys)
    else:
        df = df.drop_duplicates()

    # Handle transfers / multi-team seasons:
    # - Sum count columns across rows within (athlete_id, season)
    # - Choose teamId as the team with max minutes_total
    # - Average team_pace weighted by minutes_total (if available)
    sum_cols = [
        'shots_total', 'fga_total', 'ft_att',
        'rim_att', 'rim_made',
        'three_att', 'three_made',
        'mid_att', 'mid_made',
        'ft_made',
        'assisted_made_rim', 'assisted_made_three', 'assisted_made_mid',
        'xy_shots', 'sum_dist_ft', 'corner_3_att', 'corner_3_made',
        'xy_3_shots', 'xy_rim_shots', 'deep_3_att', 'rim_rest_att',
        'sum_dist_sq_ft',
        'ast_total', 'tov_total', 'stl_total', 'blk_total',
        'orb_total', 'drb_total', 'trb_total',
        'minutes_total',
        'games_played',
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]

    grp = df.groupby(['athlete_id', 'season'], as_index=False)
    agg = grp[sum_cols].sum(min_count=1)

    # team_pace should be averaged (not summed), weighted by minutes when available.
    if 'team_pace' in df.columns:
        pace_src = df[['athlete_id', 'season', 'team_pace'] + ([c for c in ['minutes_total', 'games_played'] if c in df.columns])].copy()
        pace_src['team_pace'] = pd.to_numeric(pace_src['team_pace'], errors='coerce')
        if 'minutes_total' in pace_src.columns:
            w = pd.to_numeric(pace_src['minutes_total'], errors='coerce').fillna(0.0).to_numpy()
            if np.any(w > 0):
                pace_src['_w'] = w
            elif 'games_played' in pace_src.columns:
                pace_src['_w'] = pd.to_numeric(pace_src['games_played'], errors='coerce').fillna(0.0)
            else:
                pace_src['_w'] = 1.0
        elif 'games_played' in pace_src.columns:
            pace_src['_w'] = pd.to_numeric(pace_src['games_played'], errors='coerce').fillna(0.0)
        else:
            pace_src['_w'] = 1.0
        pace_src['_wx'] = pace_src['_w'] * pace_src['team_pace']
        pace_agg = (
            pace_src.groupby(['athlete_id', 'season'], as_index=False)[['_wx', '_w']].sum(min_count=1)
        )
        pace_agg['team_pace'] = np.where(pace_agg['_w'] > 0, pace_agg['_wx'] / pace_agg['_w'], np.nan)
        agg = agg.merge(pace_agg[['athlete_id', 'season', 'team_pace']], on=['athlete_id', 'season'], how='left')

    # Meta from max minutes row
    meta_cols = [
        'teamId', 'is_power_conf', 'recruiting_rank', 'recruiting_stars', 'recruiting_rating',
        'college_minutes_total_source', 'college_minutes_total_source_rank',
        'college_minutes_total_conflict_flag', 'college_minutes_total_alignment_variant',
        'college_minutes_total_candidate_api', 'college_minutes_total_candidate_backfill',
        'college_minutes_total_candidate_hist_text', 'college_minutes_total_candidate_derived',
        'college_games_played_source', 'college_games_played_source_rank',
        'college_games_played_conflict_flag', 'college_games_played_alignment_variant',
        'college_games_played_candidate_api', 'college_games_played_candidate_backfill',
        'college_games_played_candidate_hist_text', 'college_games_played_candidate_derived',
        'dunk_rate', 'dunk_freq', 'putback_rate', 'transition_freq', 'rim_pressure_index',
        'deflection_proxy', 'contest_proxy', 'pressure_handle_proxy',
        'activity_source', 'has_activity_features',
        'dunk_rate_missing', 'dunk_freq_missing', 'putback_rate_missing',
        'rim_pressure_index_missing', 'contest_proxy_missing',
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    if 'minutes_total' in df.columns and meta_cols:
        # If minutes are 0, we risk unstable sort. 
        # But games_played might be non-zero now!
        sort_cols = ['athlete_id', 'season', 'minutes_total']
        if 'games_played' in df.columns:
            sort_cols.append('games_played')
            
        meta = (
            df.sort_values(sort_cols, ascending=[True, True, False] + ([False] if 'games_played' in df.columns else []))
            .drop_duplicates(['athlete_id', 'season'])[ ['athlete_id', 'season'] + meta_cols]
        )
        agg = agg.merge(meta, how='left', on=['athlete_id', 'season'])

    # Recompute core rates + spatial fields (match college_features conventions).
    if 'rim_att' in agg.columns and 'rim_made' in agg.columns:
        agg['rim_fg_pct'] = np.where(agg['rim_att'] > 0, agg['rim_made'] / agg['rim_att'], np.nan)
    if 'three_att' in agg.columns and 'three_made' in agg.columns:
        agg['three_fg_pct'] = np.where(agg['three_att'] > 0, agg['three_made'] / agg['three_att'], np.nan)
    if 'mid_att' in agg.columns and 'mid_made' in agg.columns:
        agg['mid_fg_pct'] = np.where(agg['mid_att'] > 0, agg['mid_made'] / agg['mid_att'], np.nan)
    if 'ft_att' in agg.columns and 'ft_made' in agg.columns:
        agg['ft_pct'] = np.where(agg['ft_att'] > 0, agg['ft_made'] / agg['ft_att'], np.nan)

    # Spatial recompute + gating
    if {'shots_total', 'xy_shots'}.issubset(agg.columns):
        agg['xy_coverage'] = np.where(agg['shots_total'] > 0, agg['xy_shots'] / agg['shots_total'], np.nan)
    if {'xy_shots', 'sum_dist_ft'}.issubset(agg.columns):
        gate_xy = agg['xy_shots'] >= 25
        avg_dist = np.where(gate_xy & (agg['xy_shots'] > 0), agg['sum_dist_ft'] / agg['xy_shots'], np.nan)
        agg['avg_shot_dist'] = avg_dist
    if {'xy_shots', 'sum_dist_sq_ft', 'avg_shot_dist'}.issubset(agg.columns):
        gate_xy = agg['xy_shots'] >= 25
        var_dist = np.where(
            gate_xy & (agg['xy_shots'] > 0),
            (agg['sum_dist_sq_ft'] / agg['xy_shots']) - np.square(agg['avg_shot_dist']),
            np.nan,
        )
        agg['shot_dist_var'] = np.where(np.isnan(var_dist), np.nan, np.maximum(0.0, var_dist))

    if {'xy_3_shots', 'three_att', 'corner_3_att'}.issubset(agg.columns):
        gate_xy_3 = agg['xy_3_shots'] >= 15
        agg['corner_3_rate'] = np.where(gate_xy_3 & (agg['three_att'] > 0), agg['corner_3_att'] / agg['three_att'], np.nan)
    if {'xy_3_shots', 'corner_3_att', 'corner_3_made'}.issubset(agg.columns):
        gate_xy_3 = agg['xy_3_shots'] >= 15
        agg['corner_3_pct'] = np.where(gate_xy_3 & (agg['corner_3_att'] > 0), agg['corner_3_made'] / agg['corner_3_att'], np.nan)
    if {'xy_3_shots', 'three_att', 'deep_3_att'}.issubset(agg.columns):
        gate_xy_3 = agg['xy_3_shots'] >= 15
        agg['deep_3_rate'] = np.where(gate_xy_3 & (agg['three_att'] > 0), agg['deep_3_att'] / agg['three_att'], np.nan)
    if {'xy_rim_shots', 'rim_att', 'rim_rest_att'}.issubset(agg.columns):
        gate_xy_rim = agg['xy_rim_shots'] >= 20
        agg['rim_purity'] = np.where(gate_xy_rim & (agg['rim_att'] > 0), agg['rim_rest_att'] / agg['rim_att'], np.nan)

    # Now take the literal final season row per athlete.
    agg = agg.sort_values(['athlete_id', 'season'])
    final = agg.groupby('athlete_id').tail(1).reset_index(drop=True)
    
    # Rename to indicate these are "final season" features
    cols_to_rename = {}
    for col in TIER1_SHOT_PROFILE + TIER1_CREATION + TIER1_IMPACT + TIER1_CONTEXT + TIER2_SPATIAL:
        if col in final.columns:
            cols_to_rename[col] = f'college_{col}'

    # Keep explicit names for games-played provenance contract fields.
    for col in [
        "college_minutes_total_source",
        "college_minutes_total_source_rank",
        "college_minutes_total_conflict_flag",
        "college_minutes_total_alignment_variant",
        "college_minutes_total_candidate_api",
        "college_minutes_total_candidate_backfill",
        "college_minutes_total_candidate_hist_text",
        "college_minutes_total_candidate_derived",
        "college_games_played_source",
        "college_games_played_source_rank",
        "college_games_played_conflict_flag",
        "college_games_played_alignment_variant",
        "college_games_played_candidate_api",
        "college_games_played_candidate_backfill",
        "college_games_played_candidate_hist_text",
        "college_games_played_candidate_derived",
    ]:
        if col in final.columns:
            cols_to_rename[col] = col
    
    final = final.rename(columns=cols_to_rename)
    final = final.rename(columns={'season': 'college_final_season'})
    
    logger.info(f"Extracted {len(final):,} final college seasons")
    return final


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features and coverage masks."""
    df = df.copy()
    
    # Binary target: made_nba (played >= 100 minutes in Year 1)
    if 'year1_mp' in df.columns:
        df['made_nba'] = (df['year1_mp'] >= 100).astype(int)
    else:
        df['made_nba'] = np.nan
    
    # Coverage masks for Tier 2
    if 'college_xy_shots' in df.columns:
        df['has_spatial_data'] = (df['college_xy_shots'] >= 25).astype(int)
    else:
        df['has_spatial_data'] = 0
    
    # Assisted share rates (from counts).
    # Support both naming conventions:
    # - college_assisted_{zone}_made
    # - college_assisted_made_{zone}
    for zone in ['rim', 'mid', 'three']:
        # If source already has share fields, use them directly.
        precomputed_share = f'assisted_share_{zone}'
        if precomputed_share in df.columns:
            df[f'college_assisted_share_{zone}'] = pd.to_numeric(df[precomputed_share], errors='coerce')
            continue
        made_col = f'college_{zone}_made'
        assisted_col = f'college_assisted_{zone}_made'
        assisted_alt_col = f'college_assisted_made_{zone}'
        assisted_unprefixed_col = f'assisted_made_{zone}'
        assisted_src = assisted_col if assisted_col in df.columns else assisted_alt_col
        if made_col in df.columns and assisted_col in df.columns:
            df[f'college_assisted_share_{zone}'] = np.where(
                df[made_col] > 0,
                df[assisted_src] / df[made_col],
                np.nan
            )
        elif made_col in df.columns and assisted_alt_col in df.columns:
            df[f'college_assisted_share_{zone}'] = np.where(
                df[made_col] > 0,
                df[assisted_alt_col] / df[made_col],
                np.nan
            )
        elif made_col in df.columns and assisted_unprefixed_col in df.columns:
            df[f'college_assisted_share_{zone}'] = np.where(
                df[made_col] > 0,
                pd.to_numeric(df[assisted_unprefixed_col], errors='coerce') / df[made_col],
                np.nan
            )

    # Shot-mix shares (derived from counts; missingness-safe).
    # These features are important for era adjustment (3P rates drift a lot).
    if 'college_shots_total' in df.columns:
        shots = df['college_shots_total'].astype(float)
        for zone in ['rim', 'mid', 'three']:
            att = f'college_{zone}_att'
            if att in df.columns:
                df[f'college_{zone}_share'] = np.where(shots > 0, df[att].astype(float) / shots, np.nan)

    # Possession-proxy from final-season counts (available across eras).
    if all(c in df.columns for c in ['college_fga_total', 'college_ft_att', 'college_tov_total']):
        fga = pd.to_numeric(df['college_fga_total'], errors='coerce')
        fta = pd.to_numeric(df['college_ft_att'], errors='coerce')
        tov = pd.to_numeric(df['college_tov_total'], errors='coerce')
        poss_proxy = fga + 0.44 * fta + tov
        df['college_poss_proxy'] = poss_proxy
    else:
        poss_proxy = pd.Series(np.nan, index=df.index)

    # Per-100-possession activity rates (preferred).
    for stat in ['stl_total', 'blk_total', 'tov_total', 'ast_total']:
        col = f'college_{stat}'
        if col in df.columns:
            num = pd.to_numeric(df[col], errors='coerce')
            df[f'{col}_per100poss'] = np.where(poss_proxy > 0, num / poss_proxy * 100.0, np.nan)
    for stat in ['orb_total', 'drb_total', 'trb_total']:
        col = f'college_{stat}'
        if col in df.columns:
            num = pd.to_numeric(df[col], errors='coerce')
            df[f'{col}_per100poss'] = np.where(poss_proxy > 0, num / poss_proxy * 100.0, np.nan)

    # Canonical aliases for impact stack + RAPM-style fields (DAG contract names).
    impact_aliases = {
        'college_rapm_standard': ['rIPM_tot_std', 'rIPM_tot_lev_wt', 'rIPM_tot_non_garbage', 'rIPM_tot_rubber', 'rIPM_tot_recency'],
        'college_o_rapm': ['rIPM_off_std', 'rIPM_off_lev_wt', 'rIPM_off_non_garbage', 'rIPM_off_rubber', 'rIPM_off_recency', 'impact_on_net_raw'],
        'college_d_rapm': ['rIPM_def_std', 'rIPM_def_lev_wt', 'rIPM_def_non_garbage', 'rIPM_def_rubber', 'rIPM_def_recency'],
        'college_on_net_rating': ['impact_on_net_raw'],
        'college_on_ortg': ['impact_on_ortg_raw'],
        'college_on_drtg': ['impact_on_drtg_raw'],
    }
    for dst, src_candidates in impact_aliases.items():
        if dst not in df.columns:
            df[dst] = np.nan
        out = pd.to_numeric(df[dst], errors='coerce')
        for src in src_candidates:
            if src in df.columns:
                src_vals = pd.to_numeric(df[src], errors='coerce')
                out = out.where(out.notna(), src_vals)
        # Defensive fallback from on-court net when explicit defensive RAPM variants are unavailable.
        if dst == 'college_d_rapm' and out.notna().mean() < 0.001 and 'impact_on_net_raw' in df.columns:
            net = pd.to_numeric(df['impact_on_net_raw'], errors='coerce')
            out = out.where(out.notna(), -net)
        df[dst] = out
    
    # Backward-compatibility: keep per40 columns, but avoid near-all-null behavior.
    # If minutes are missing, use a scaled per100 proxy.
    if 'college_minutes_total' in df.columns:
        mins = pd.to_numeric(df['college_minutes_total'], errors='coerce')
        for stat in ['stl_total', 'blk_total', 'tov_total', 'ast_total']:
            col = f'college_{stat}'
            p40_col = f'{col}_per40'
            if col in df.columns:
                p40_true = np.where(mins > 0, pd.to_numeric(df[col], errors='coerce') / mins * 40.0, np.nan)
                p40_proxy = df.get(f'{col}_per100poss', np.nan) * 0.40
                df[p40_col] = np.where(np.isfinite(p40_true), p40_true, p40_proxy)

    # Usage fallback if career-store final_usage is missing.
    if 'final_usage' in df.columns and all(c in df.columns for c in ['college_fga_total', 'college_ft_att', 'college_tov_total']):
        poss_final = pd.to_numeric(df['final_poss_total'], errors='coerce') if 'final_poss_total' in df.columns else poss_proxy
        usage_proxy = (fga + 0.44 * fta + tov) / poss_final.clip(lower=1)
        need_usage = pd.to_numeric(df['final_usage'], errors='coerce').isna()
        df.loc[need_usage, 'final_usage'] = usage_proxy[need_usage]

    # Team pace fallback to same-scale proxy (possessions/game-ish).
    if 'college_team_pace' in df.columns:
        need_pace = pd.to_numeric(df['college_team_pace'], errors='coerce').isna()
        pace_proxy = pd.to_numeric(df.get('final_poss_total', np.nan), errors='coerce') / 30.0
        df.loc[need_pace, 'college_team_pace'] = pace_proxy[need_pace]

    # Physical derived features (used by encoder for interaction learning).
    h_in = pd.to_numeric(df.get('college_height_in'), errors='coerce')
    w_lbs = pd.to_numeric(df.get('college_weight_lbs'), errors='coerce')
    ws_in = pd.to_numeric(df.get('wingspan_in'), errors='coerce')
    h_m = h_in * 0.0254
    w_kg = w_lbs * 0.45359237
    df['college_bmi'] = np.where((h_m > 0) & np.isfinite(h_m) & np.isfinite(w_kg), w_kg / (h_m * h_m), np.nan)
    df['college_wingspan_to_height_ratio'] = np.where((h_in > 0) & np.isfinite(h_in) & np.isfinite(ws_in), ws_in / h_in, np.nan)
    
    # True Shooting % (if not already present)
    # Missingness-safe: do not fabricate 0s when components are missing.
    if 'final_trueShootingPct' not in df.columns:
        rim_made = df['college_rim_made'] if 'college_rim_made' in df.columns else np.nan
        mid_made = df['college_mid_made'] if 'college_mid_made' in df.columns else np.nan
        three_made = df['college_three_made'] if 'college_three_made' in df.columns else np.nan
        ft_made = df['college_ft_made'] if 'college_ft_made' in df.columns else np.nan
        fga = df['college_fga_total'] if 'college_fga_total' in df.columns else np.nan
        fta = df['college_ft_att'] if 'college_ft_att' in df.columns else np.nan

        any_pts = pd.Series(rim_made).notna() | pd.Series(mid_made).notna() | pd.Series(three_made).notna() | pd.Series(ft_made).notna()
        pts = np.where(
            any_pts,
            2 * pd.Series(rim_made).fillna(0) + 2 * pd.Series(mid_made).fillna(0) + 3 * pd.Series(three_made).fillna(0) + pd.Series(ft_made).fillna(0),
            np.nan,
        )
        denom = 2 * (pd.Series(fga).astype(float) + 0.44 * pd.Series(fta).astype(float))
        df['final_trueShootingPct'] = np.where((denom > 0) & pd.Series(pts).notna(), pts / denom, np.nan)

    # Explicit within-season defaults when upstream windows are unavailable.
    # This makes gating intent explicit instead of carrying silent NaNs.
    within_defaults = {
        'final_has_ws_last10': 0.0,
        'final_ws_minutes_last10': 0.0,
        'final_ws_pps_last10': 0.0,
        'final_ws_delta_pps_last5_minus_prev5': 0.0,
        'final_has_ws_breakout_timing_eff': 0.0,
        'final_ws_breakout_timing_eff': 0.0,
    }
    for col, default in within_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    # Explicit within-branch masks for gating/QA.
    ws_signal_cols = [
        'final_has_ws_last10',
        'final_ws_minutes_last10',
        'final_ws_pps_last10',
        'final_ws_delta_pps_last5_minus_prev5',
        'final_has_ws_breakout_timing_eff',
        'final_ws_breakout_timing_eff',
    ]
    present_ws = [c for c in ws_signal_cols if c in df.columns]
    if present_ws:
        ws_arr = np.stack([pd.to_numeric(df[c], errors='coerce').fillna(0).to_numpy() for c in present_ws], axis=1)
        df['has_within_window_data'] = (np.max(np.abs(ws_arr), axis=1) > 0).astype(int)
    else:
        df['has_within_window_data'] = 0

    # DAG contract fields: season index / class proxy / age (nullable if unknown)
    # season_index: sequential college year count (proxy from career_years)
    if 'season_index' not in df.columns:
        if 'career_years' in df.columns:
            df['season_index'] = pd.to_numeric(df['career_years'], errors='coerce')
        else:
            df['season_index'] = np.nan

    # class_year: integer class proxy from season_index (1=freshman .. 5=5th year+)
    if 'class_year' not in df.columns:
        si = pd.to_numeric(df.get('season_index'), errors='coerce')
        cls = np.floor(si).clip(lower=1, upper=5)
        df['class_year'] = np.where(np.isfinite(si), cls, np.nan)

    # age_at_season: use source if present; otherwise deterministic class-year proxy.
    if 'age_at_season' not in df.columns:
        df['age_at_season'] = np.nan
    age_raw = pd.to_numeric(df['age_at_season'], errors='coerce')
    if 'class_year' in df.columns:
        cls = pd.to_numeric(df['class_year'], errors='coerce')
        age_proxy = 17.0 + cls  # freshman~18, sophomore~19, etc.
        age_raw = age_raw.where(age_raw.notna(), age_proxy)
    df['age_at_season'] = age_raw
    df['has_age_at_season'] = pd.to_numeric(df['age_at_season'], errors='coerce').notna().astype(int)

    # Breakout timing efficiency fallback:
    # If timing feature is all-zero/missing in current sources, backfill from breakout rank.
    if 'breakout_timing_eff' in df.columns:
        bte = pd.to_numeric(df['breakout_timing_eff'], errors='coerce')
        if float((bte.fillna(0) != 0).mean()) < 0.001 and 'breakout_rank_eff' in df.columns:
            bre = pd.to_numeric(df['breakout_rank_eff'], errors='coerce')
            df['breakout_timing_eff'] = bre.fillna(0.0).clip(lower=0.0, upper=1.0)
    
    return df


def apply_era_normalization(df: pd.DataFrame, era_col: str = 'college_final_season') -> pd.DataFrame:
    """
    Z-score features by era (season) to handle drift.
    Only applies to features known to drift across eras.
    """
    df = df.copy()
    
    if era_col not in df.columns:
        logger.warning(f"Era column '{era_col}' not found, skipping normalization")
        return df
    
    # Features to normalize (rates that drift)
    drift_features = [
        # shooting + mix
        'college_rim_fg_pct', 'college_mid_fg_pct', 'college_three_fg_pct', 'college_ft_pct',
        'college_rim_share', 'college_mid_share', 'college_three_share',
        # creation-ish
        'college_assisted_share_rim', 'college_assisted_share_three',
        # career-level summary rates
        'final_trueShootingPct', 'final_usage',
        # activity / role
        'college_tov_total_per100poss', 'college_ast_total_per100poss',
        'college_stl_total_per100poss', 'college_blk_total_per100poss',
    ]
    
    for col in drift_features:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            grouped = vals.groupby(df[era_col])
            mean = grouped.transform('mean')
            std = grouped.transform('std').replace(0, 1)
            df[f'{col}_z'] = (vals - mean) / std
            logger.debug(f"Normalized {col} by era")
    
    return df


def apply_team_residualization(
    df: pd.DataFrame,
    season_col: str = 'college_final_season',
    team_col: str = 'college_teamId',
) -> pd.DataFrame:
    """
    Remove team-season means from select features (team context adjustment).
    This is a lightweight alternative to full adjusted-plus-minus for non-point outcomes.
    """
    df = df.copy()
    if season_col not in df.columns or team_col not in df.columns:
        return df

    features = [
        'final_trueShootingPct',
        'final_usage',
        'college_three_fg_pct',
        'college_three_share',
        'college_ast_total_per100poss',
        'college_tov_total_per100poss',
        'college_stl_total_per100poss',
        'college_blk_total_per100poss',
    ]
    for col in features:
        if col not in df.columns:
            continue
        grp = df.groupby([season_col, team_col])[col]
        mu = grp.transform('mean')
        df[f'{col}_team_resid'] = df[col] - mu

    return df


def build_unified_training_table(
    use_career_features: bool = True,
    use_trajectory_features: bool = True,
    apply_normalization: bool = True,
    min_targets: int = 1,
    min_draft_year: Optional[int] = 2011,
    mode: str = "supervised",
) -> pd.DataFrame:
    """
    Build the unified training table.
    
    Args:
        use_career_features: Include career aggregate features
        apply_normalization: Apply era-based z-score normalization
        min_targets: Minimum number of non-null targets required
    
    Returns:
        DataFrame ready for model training
    """
    logger.info("=" * 60)
    logger.info("Building Unified Training Table")
    logger.info("=" * 60)
    
    # 1. Load all data sources
    college_features_all = load_college_features_all_splits()
    college_features = college_features_all[college_features_all['split_id'] == 'ALL__ALL'].copy() if not college_features_all.empty else pd.DataFrame()
    leverage_features = build_final_season_leverage_features(college_features_all)
    team_strength = load_team_strength_features()
    derived_stats = load_derived_box_stats()
    api_event_games = load_api_event_games_candidate()
    activity_proxies = load_activity_proxies()
    exposure_backfill = load_historical_exposure_backfill()
    hist_text_games = load_historical_text_games_backfill()
    college_impact_stack = load_college_impact_stack()
    college_pathway_context = load_college_pathway_context()
    
    # Merge derived stats to fill missing signal (2010-2024 gaps)
    if not derived_stats.empty and not college_features.empty:
        og_len = len(college_features)
        # We join on athlete_id and season
        college_features = college_features.merge(derived_stats, on=['athlete_id', 'season'], how='left')
        
        # Merge semantics: use derived values when available, otherwise keep
        # existing feature-store values (never force nulls to zero here).
        for stat in ['ast', 'stl', 'blk', 'tov']:
            derived_col = f'college_{stat}_total'
            target_col = f'{stat}_total'
            if derived_col in college_features.columns:
                if target_col in college_features.columns:
                    college_features[target_col] = (
                        pd.to_numeric(college_features[derived_col], errors='coerce')
                        .combine_first(pd.to_numeric(college_features[target_col], errors='coerce'))
                    )
                else:
                    college_features[target_col] = pd.to_numeric(college_features[derived_col], errors='coerce')
        
        # Games played: preserve primary games_played first, use derived fallback only when missing.
        if 'college_games_played' in college_features.columns:
            college_features["derived_games_played_candidate"] = pd.to_numeric(
                college_features["college_games_played"], errors="coerce"
            )
            if 'games_played' in college_features.columns:
                college_features['games_played'] = (
                    pd.to_numeric(college_features['games_played'], errors='coerce')
                    .combine_first(pd.to_numeric(college_features['college_games_played'], errors='coerce'))
                )
            else:
                college_features['games_played'] = pd.to_numeric(college_features['college_games_played'], errors='coerce')
            
            # CRITICAL: Drop the pre-named column so it doesn't fatally collide when get_final_college_season()
            # attempts to rename the freshly patched 'games_played' target back to 'college_games_played'!
            college_features = college_features.drop(columns=['college_games_played'])
             
        logger.info(f"Merged derived box stats. Rows: {og_len} -> {len(college_features)}")

    # Preserve baseline minutes candidate before exposure overrides.
    if not college_features.empty and "derived_minutes_total_candidate" not in college_features.columns:
        if "minutes_total" in college_features.columns:
            college_features["derived_minutes_total_candidate"] = pd.to_numeric(
                college_features["minutes_total"], errors="coerce"
            )

    # API event-participation games candidate (athlete-season).
    if not api_event_games.empty and not college_features.empty:
        og_len = len(college_features)
        college_features = college_features.merge(api_event_games, on=["athlete_id", "season"], how="left")
        if "games_played" in college_features.columns:
            gp = pd.to_numeric(college_features["games_played"], errors="coerce")
            api_gp = pd.to_numeric(college_features.get("api_event_games_played"), errors="coerce")
            college_features["games_played"] = gp.combine_first(api_gp)
        else:
            college_features["games_played"] = pd.to_numeric(college_features.get("api_event_games_played"), errors="coerce")
        logger.info(f"Merged API event games candidate. Rows: {og_len} -> {len(college_features)}")

    # Merge enhanced activity proxies (dunk/putback/rim-pressure/contest/etc.).
    if not activity_proxies.empty and not college_features.empty:
        og_len = len(college_features)
        join_keys = [c for c in ["athlete_id", "season", "split_id"] if c in college_features.columns and c in activity_proxies.columns]
        if not join_keys:
            join_keys = [c for c in ["athlete_id", "season"] if c in college_features.columns and c in activity_proxies.columns]
        non_keys = [c for c in activity_proxies.columns if c not in join_keys]
        if join_keys:
            college_features = college_features.merge(
                activity_proxies[join_keys + non_keys].drop_duplicates(subset=join_keys),
                on=join_keys,
                how='left',
            )
            # If base frame already had activity columns, merge can create _x/_y
            # suffixes. Coalesce deterministically back to canonical names.
            activity_cols = [
                "dunk_rate", "dunk_freq", "putback_rate", "transition_freq",
                "rim_pressure_index", "deflection_proxy", "contest_proxy",
                "pressure_handle_proxy", "activity_source", "has_activity_features",
                "dunk_rate_missing", "dunk_freq_missing", "putback_rate_missing",
                "rim_pressure_index_missing", "contest_proxy_missing",
            ]
            for c in activity_cols:
                cx, cy = f"{c}_x", f"{c}_y"
                if cx in college_features.columns or cy in college_features.columns:
                    sx = college_features[cx] if cx in college_features.columns else pd.Series(np.nan, index=college_features.index)
                    sy = college_features[cy] if cy in college_features.columns else pd.Series(np.nan, index=college_features.index)
                    college_features[c] = pd.to_numeric(sy, errors="ignore").combine_first(
                        pd.to_numeric(sx, errors="ignore")
                    )
                    if cx in college_features.columns:
                        college_features = college_features.drop(columns=[cx])
                    if cy in college_features.columns:
                        college_features = college_features.drop(columns=[cy])
            logger.info(f"Merged enhanced activity proxies on {join_keys}. Rows: {og_len} -> {len(college_features)}")

    # Merge mapped historical exposure backfill (minutes/games/tov) to patch
    # known API-era undercoverage in specific seasons/teams.
    if not exposure_backfill.empty and not college_features.empty:
        og_len = len(college_features)
        college_features = college_features.merge(exposure_backfill, on=['athlete_id', 'season'], how='left')
        
        # Inject precise API impact seconds to perfectly match the tracked event denominator!
        if 'impact_seconds_total' in college_impact_stack.columns:
            college_features = college_features.merge(college_impact_stack[['athlete_id', 'season', 'impact_seconds_total']], on=['athlete_id', 'season'], how='left')

        if 'minutes_total' in college_features.columns:
            minutes_existing = pd.to_numeric(college_features['minutes_total'], errors='coerce')
            minutes_backfill = pd.to_numeric(college_features.get('backfill_minutes_total'), errors='coerce')
            
            # Use impact_seconds_total / 60.0 as the most accurate API minute tracking for this dataset's sample!
            if 'impact_seconds_total' in college_features.columns:
                minutes_api = college_features['impact_seconds_total'] / 60.0
                minutes_existing = np.where(minutes_api > 0, minutes_api, minutes_existing)
                
            college_features['minutes_total'] = pd.Series(np.where(np.nan_to_num(pd.to_numeric(minutes_existing, errors='coerce'), nan=0.0) > 0, minutes_existing, minutes_backfill), index=college_features.index)

        if 'games_played' in college_features.columns:
            games_existing = pd.to_numeric(college_features['games_played'], errors='coerce')
            games_backfill = pd.to_numeric(college_features.get('backfill_games_played'), errors='coerce')
            
            # Prefer existing API/participant data UNLESS the backfill indicates a massive gap
            # in API coverage (e.g. 18 games vs 39 known games). Gap >= 5 indicates structural undercoverage.
            college_features['games_played'] = pd.Series(
                np.where(
                    (games_backfill > 0) & ((games_existing.isna()) | (games_existing <= 0) | ((games_backfill - games_existing) >= 5)),
                    games_backfill,
                    games_existing
                ), 
                index=college_features.index
            )
            
            p_mask = college_features['athlete_id'] == 27623
            if p_mask.any():
                logger.info(f"PAOLO TRACE POST-EXPOSURE: games_played={college_features.loc[p_mask, 'games_played'].tolist()}")

        if 'tov_total' in college_features.columns:
            tov_existing = pd.to_numeric(college_features['tov_total'], errors='coerce')
            tov_backfill = pd.to_numeric(college_features.get('backfill_tov_total'), errors='coerce')
            college_features['tov_total'] = pd.Series(np.where(tov_existing.fillna(0) > 0, tov_existing, tov_backfill), index=college_features.index)

        logger.info(f"Merged historical exposure backfill. Rows: {og_len} -> {len(college_features)}")

    # Historical manual text backfill for games played (season start-year -> end-year mapped).
    if not hist_text_games.empty and not college_features.empty:
        og_len = len(college_features)
        college_features = college_features.merge(hist_text_games, on=['athlete_id', 'season'], how='left')
        hist_games = pd.to_numeric(college_features.get('hist_games_played_text'), errors='coerce')
        if 'games_played' in college_features.columns:
            games_existing = pd.to_numeric(college_features['games_played'], errors='coerce')
            # Source priority: API/derived/backfill first, historical text as
            # fallback-only to avoid undercount overrides (e.g., parsed 1 game) UNLESS
            # it exposes a massive >5 game gap in the primary sources.
            college_features['games_played'] = pd.Series(
                np.where(
                    (hist_games > 0) & ((games_existing.isna()) | (games_existing <= 0) | ((hist_games - games_existing) >= 5)),
                    hist_games,
                    games_existing,
                ),
                index=college_features.index,
            )
            
            p_mask = college_features['athlete_id'] == 27623
            if p_mask.any():
                logger.info(f"PAOLO TRACE POST-HIST: games_played={college_features.loc[p_mask, 'games_played'].tolist()}")
        else:
            college_features['games_played'] = hist_games
        logger.info(f"Merged historical text games backfill. Rows: {og_len} -> {len(college_features)}")

    # Canonical minutes selection with provenance.
    if not college_features.empty:
        college_features = select_minutes_with_provenance(
            college_features,
            api_col="minutes_total",
            backfill_col="backfill_minutes_total",
            hist_col="hist_minutes_total",
            derived_col="derived_minutes_total_candidate",
            backfill_variant_col="backfill_alignment_variant",
            hist_variant_col="hist_alignment_variant",
        )

    # Canonical games-played selection with provenance (train/serve parity contract).
    if not college_features.empty:
        college_features = select_games_played_with_provenance(
            college_features,
            api_col="games_played",
            backfill_col="backfill_games_played",
            hist_col="hist_games_played_text",
            derived_col="derived_games_played_candidate",
            backfill_variant_col="backfill_alignment_variant",
            hist_variant_col="hist_alignment_variant",
        )
        
        # Enforce Explicit Provenance Columns
        if 'minutes_source' in college_features.columns:
            college_features = college_features.rename(columns={'minutes_source': 'college_minutes_total_source'})
        if 'games_source' in college_features.columns:
            college_features = college_features.rename(columns={'games_source': 'college_games_played_source'})

    career_features = load_career_features()
    trajectory_features = load_trajectory_features()
    college_dev_rate = load_college_dev_rate()
    transfer_summary = load_transfer_context_summary()
    physicals_by_season = load_college_physicals_by_season()
    physical_traj = load_college_physical_trajectory()
    crosswalk = load_crosswalk()
    dim_nba = load_dim_player_nba()
    nba_ws_bridge = load_nba_wingspan_bridge()
    recruiting_phys = load_recruiting_physicals()
    nba_targets = load_nba_targets()
    
    # Check for required data
    if crosswalk.empty:
        logger.error("Crosswalk is required but not found!")
        return pd.DataFrame()
    
    if nba_targets.empty:
        logger.error("NBA targets are required but not found!")
        return pd.DataFrame()
    
    # 2. Extract final college season features
    if not college_features.empty:
        final_college = get_final_college_season(college_features)
        
        p_mask = final_college['athlete_id'] == 27623
        if p_mask.any():
            cg_val = final_college.loc[p_mask, 'college_games_played'].tolist() if 'college_games_played' in final_college.columns else "MISSING"
            g_val = final_college.loc[p_mask, 'games_played'].tolist() if 'games_played' in final_college.columns else "MISSING"
            logger.info(f"PAOLO TRACE POST-AGGREGATOR: college_games_played={cg_val} / games_played={g_val}")
    else:
        final_college = pd.DataFrame()

    # Add final-season leverage rates.
    if not final_college.empty and not leverage_features.empty:
        final_college = final_college.merge(
            leverage_features,
            on=['athlete_id', 'college_final_season'],
            how='left',
        )
        logger.info("  Added leverage-rate features to final season frame")

    # 2b. **Normalize on the full population** BEFORE crosswalk/cohort filtering.
    # This ensures z-scores and team residuals match the inference pipeline,
    # which also normalizes on the full feature store.
    if apply_normalization and not final_college.empty:
        # Join career features temporarily for normalization of career-level cols
        if not career_features.empty:
            fc_with_career = final_college.merge(career_features, on='athlete_id', how='left', suffixes=('', '_career'))
        else:
            fc_with_career = final_college.copy()
        fc_with_career = apply_era_normalization(fc_with_career)
        fc_with_career = apply_team_residualization(fc_with_career)
        # Carry the normalized columns back to final_college
        norm_cols = [c for c in fc_with_career.columns if c.endswith('_z') or c.endswith('_team_resid')]
        for c in norm_cols:
            final_college[c] = fc_with_career[c].values
        logger.info(f"  Applied era + team normalization on full population ({len(final_college):,} rows, {len(norm_cols)} derived cols)")
    
    # 3. Apply Mode Grain
    logger.info(f"Applying data surface grain for mode: {mode}")
    
    base_cols = ['athlete_id', 'nba_id']
    for c in ['wingspan_in', 'standing_reach_in', 'wingspan_minus_height_in']:
        if c in crosswalk.columns:
            base_cols.append(c)
    cw = crosswalk[base_cols].copy()
    
    if mode in ["foundation", "joint"]:
        # Foundation/joint surfaces are explicit athlete-season training planes.
        # Do NOT collapse to final season here.
        if college_features.empty:
            df = pd.DataFrame(columns=['athlete_id', 'season', 'college_final_season'])
        else:
            df = college_features.copy()
            if "season" not in df.columns:
                df["season"] = np.nan
            # Keep a stable season key used by downstream scripts.
            df["college_final_season"] = pd.to_numeric(df["season"], errors="coerce")
            df = df.drop_duplicates(subset=["athlete_id", "season"])
        # Join leverage-rate features at athlete-season grain.
        if not leverage_features.empty:
            lev = leverage_features.rename(columns={"college_final_season": "season"})
            lev_keep = ["athlete_id", "season"] + [c for c in ["high_lev_att_rate", "garbage_att_rate", "leverage_poss_share"] if c in lev.columns]
            df = df.merge(lev[lev_keep], on=["athlete_id", "season"], how="left")
        df = df.merge(cw, on='athlete_id', how='left')
        df['has_nba_link'] = df['nba_id'].notna().astype(int)
    else:
        df = cw.copy()
        if not final_college.empty:
            df = df.merge(final_college, on='athlete_id', how='left')
        df['has_nba_link'] = 1
        
    # 4. Integrate Combine Signals (Measured & Imputed)
    logger.info("Integrating Combine Athleticism Signals...")
    combine_meas_path = COMBINE_MEASUREMENTS_PATH if COMBINE_MEASUREMENTS_PATH.exists() else COMBINE_MEASUREMENTS_FALLBACK
    combine_imp_path = COMBINE_IMPUTED_PATH if COMBINE_IMPUTED_PATH.exists() else COMBINE_IMPUTED_FALLBACK
    if combine_meas_path.exists() and combine_imp_path.exists():
        df_meas = pd.read_parquet(combine_meas_path)
        df_imp = pd.read_parquet(combine_imp_path)
        
        # Prefix combine measurement columns to avoid collisions with existing physicals
        combine_raw_cols = ['wingspan_in', 'standing_reach_in', 'height_wo_shoes_in',
                            'weight_lbs', 'no_step_vertical_in', 'max_vertical_in',
                            'lane_agility_s', 'three_quarter_sprint_s']
        rename_map = {c: f'combine_{c}' for c in combine_raw_cols if c in df_meas.columns}
        df_meas = df_meas.rename(columns=rename_map)
        
        drop_cols = ['nba_id', 'combine_player_name', 'combine_year', 'link_method']
        df = df.merge(df_meas.drop(columns=[c for c in drop_cols if c in df_meas.columns]), on='athlete_id', how='left')
        df = df.merge(df_imp, on=['athlete_id', 'college_final_season'], how='left')
        
        combine_targets = ['wingspan_in', 'standing_reach_in', 'no_step_vertical_in', 'max_vertical_in', 'lane_agility_s', 'three_quarter_sprint_s']
        
        for t in combine_targets:
            meas_col = f'combine_{t}'
            imp_col = t.replace('_in', '').replace('_s', '') + '_imputed'
            sd_col = t.replace('_in', '').replace('_s', '') + '_imputed_sd'
            final_col = f'combine_{t}_combined'
            
            if meas_col in df.columns and imp_col in df.columns:
                is_meas = df[meas_col].notna()
                df[final_col] = np.where(is_meas, df[meas_col], df[imp_col])
                df[f'combine_{t}_uncertainty'] = np.where(is_meas, 0.0, df.get(sd_col, np.nan))
                df[f'source_combine_{t}'] = np.where(is_meas, 'measured', np.where(df[imp_col].notna(), 'imputed', 'missing'))
                
        # Fill global combine flag
        df['has_combine_measured'] = df.get('has_combine_measured', pd.Series(0, index=df.index)).fillna(0).astype(int)
        df['has_combine_imputed'] = df.get('is_imputed', pd.Series(0, index=df.index)).fillna(0).astype(int)
        
        logger.info(
            f"  Combine integration complete from {combine_meas_path.name}/{combine_imp_path.name}. "
            f"Measured rows matching: {df['has_combine_measured'].sum()}"
        )

    # Join canonical college physicals at final college season grain.
    if not physicals_by_season.empty and "college_final_season" in df.columns:
        phys = physicals_by_season.copy()
        phys = phys.rename(columns={"season": "college_final_season", "team_id": "college_teamId"})
        join_keys = ["athlete_id", "college_final_season"]
        if "college_teamId" in df.columns and "college_teamId" in phys.columns:
            join_keys.append("college_teamId")
        keep_cols = join_keys + [
            c for c in [
                "height_in", "weight_lbs", "wingspan_in", "standing_reach_in",
                "wingspan_minus_height_in", "has_height", "has_weight", "has_wingspan",
                "source_type", "source_provider", "source_url", "confidence",
            ] if c in phys.columns
        ]
        phys = phys[keep_cols].drop_duplicates(subset=join_keys)
        df = df.merge(phys, on=join_keys, how="left")
        # Canonical contract aliases.
        existing_h = pd.to_numeric(df["college_height_in"], errors="coerce") if "college_height_in" in df.columns else pd.Series(np.nan, index=df.index)
        existing_w = pd.to_numeric(df["college_weight_lbs"], errors="coerce") if "college_weight_lbs" in df.columns else pd.Series(np.nan, index=df.index)
        existing_h_mask = pd.to_numeric(df["has_college_height"], errors="coerce") if "has_college_height" in df.columns else pd.Series(np.nan, index=df.index)
        existing_w_mask = pd.to_numeric(df["has_college_weight"], errors="coerce") if "has_college_weight" in df.columns else pd.Series(np.nan, index=df.index)
        if "height_in" in df.columns:
            df["college_height_in"] = existing_h.combine_first(
                pd.to_numeric(df["height_in"], errors="coerce")
            )
        if "weight_lbs" in df.columns:
            df["college_weight_lbs"] = existing_w.combine_first(
                pd.to_numeric(df["weight_lbs"], errors="coerce")
            )
        if "has_height" in df.columns:
            df["has_college_height"] = existing_h_mask.combine_first(
                pd.to_numeric(df["has_height"], errors="coerce")
            )
        if "has_weight" in df.columns:
            df["has_college_weight"] = existing_w_mask.combine_first(
                pd.to_numeric(df["has_weight"], errors="coerce")
            )
        logger.info("  Added canonical college physicals by season")

    # Join physical trajectory values at final college season grain.
    if not physical_traj.empty and "college_final_season" in df.columns:
        ptri = physical_traj.rename(columns={"season": "college_final_season"}).copy()
        t_join = ["athlete_id", "college_final_season"]
        t_cols = t_join + [
            c for c in [
                "height_delta_yoy", "weight_delta_yoy",
                "height_slope_3yr", "weight_slope_3yr",
                "height_change_entry_to_final", "weight_change_entry_to_final",
                "trajectory_obs_count",
            ] if c in ptri.columns
        ]
        ptri = ptri[t_cols].drop_duplicates(subset=t_join)
        df = df.merge(ptri, on=t_join, how="left")
        rename_map = {
            "height_delta_yoy": "college_height_delta_yoy",
            "weight_delta_yoy": "college_weight_delta_yoy",
            "height_slope_3yr": "college_height_slope_3yr",
            "weight_slope_3yr": "college_weight_slope_3yr",
        }
        for src, dst in rename_map.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = pd.to_numeric(df[src], errors="coerce")
        logger.info("  Added college physical trajectory features")

    # Add team-strength/SRS via final season + team.
    if not team_strength.empty and {'college_teamId', 'college_final_season'}.issubset(df.columns):
        srs = team_strength.rename(columns={'teamId': 'college_teamId', 'season': 'college_final_season'})
        df = df.merge(
            srs[['college_teamId', 'college_final_season', 'college_team_srs', 'team_strength_srs', 'college_team_rank']],
            on=['college_teamId', 'college_final_season'],
            how='left',
        )
        logger.info("  Added team-strength/SRS features")
    
    # Join career features
    if use_career_features and not career_features.empty:
        df = df.merge(career_features, on='athlete_id', how='left', suffixes=('', '_career'))

    # Add college-side physicals (works for prospects, draft-safe).
    if not recruiting_phys.empty:
        df = df.merge(recruiting_phys, on="athlete_id", how="left")
    recruit_h = pd.to_numeric(df["recruit_height_in"], errors="coerce") if "recruit_height_in" in df.columns else pd.Series(np.nan, index=df.index)
    recruit_w = pd.to_numeric(df["recruit_weight_lbs"], errors="coerce") if "recruit_weight_lbs" in df.columns else pd.Series(np.nan, index=df.index)
    if "college_height_in" not in df.columns:
        df["college_height_in"] = recruit_h
    else:
        df["college_height_in"] = pd.to_numeric(df["college_height_in"], errors="coerce").combine_first(
            recruit_h
        )
    if "college_weight_lbs" not in df.columns:
        df["college_weight_lbs"] = recruit_w
    else:
        df["college_weight_lbs"] = pd.to_numeric(df["college_weight_lbs"], errors="coerce").combine_first(
            recruit_w
        )

    # Join college impact stack by final season with dual season mapping:
    # exact season and (season + 1) fallback for start-year vs end-year conventions.
    if not college_impact_stack.empty and "college_final_season" in df.columns:
        impact = college_impact_stack.copy()
        impact["season"] = pd.to_numeric(impact["season"], errors="coerce")
        impact_exact = impact.rename(columns={"season": "college_final_season"})
        impact_plus1 = impact.copy()
        impact_plus1["college_final_season"] = impact_plus1["season"] + 1
        impact_plus1 = impact_plus1.drop(columns=["season"], errors="ignore")
        impact_cols = [
            c for c in impact_exact.columns
            if c not in {"athlete_id", "college_final_season"}
        ]
        df = df.merge(
            impact_exact[["athlete_id", "college_final_season"] + impact_cols],
            on=["athlete_id", "college_final_season"],
            how="left",
        )
        p1_cols = ["athlete_id", "college_final_season"] + impact_cols
        p1_rename = {c: f"{c}_p1" for c in impact_cols}
        df = df.merge(
            impact_plus1[p1_cols].rename(columns=p1_rename),
            on=["athlete_id", "college_final_season"],
            how="left",
        )
        for c in impact_cols:
            c_p1 = f"{c}_p1"
            if c_p1 in df.columns:
                base = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
                alt = pd.to_numeric(df[c_p1], errors="coerce")
                df[c] = base.combine_first(alt)
                df.drop(columns=[c_p1], inplace=True, errors="ignore")
        logger.info(f"  Added college impact stack features (dual-season): {len(impact_cols)}")

    # Join college pathway context v2 with dual season mapping.
    if not college_pathway_context.empty and "college_final_season" in df.columns:
        # Linkage Gap Patch:
        # If athlete_id linkage is broken for recent years, try a name-based fallback.
        if mode == "supervised":
            try:
                con = duckdb.connect(str(DB_PATH), read_only=True)
                name_map = con.execute("SELECT shooterAthleteId as athlete_id, mode(shooter_name) as shooter_name FROM stg_shots GROUP BY 1").df()
                con.close()
                
                # Standardize pathway context for matching
                pctx_m = college_pathway_context.merge(name_map, on="athlete_id", how="inner")
                pctx_m["clean_name"] = clean_name_for_join(pctx_m["shooter_name"])
                
                # Standardize supervised table for matching
                # Ensure we have player_name from dim_nba
                if "player_name" not in df.columns and not dim_nba.empty:
                    df = df.merge(dim_nba[["nba_id", "player_name"]], on="nba_id", how="left")
                
                if "player_name" in df.columns:
                    df["clean_name"] = clean_name_for_join(df["player_name"])
                    
                    # Target season for matching: prioritize college_final_season, then draft_year_proxy
                    df["target_season"] = df["college_final_season"].fillna(df.get("draft_year_proxy", np.nan))
                    df["target_season"] = pd.to_numeric(df["target_season"], errors="coerce")
                    
                    # Missing context mask: 
                    # Row needs patching if its current [athlete_id, season] lacks valid context data
                    # OR if the existing athlete_id points to a DIFFERENT person (crosswalk swap).
                    pctx_valid = college_pathway_context[college_pathway_context["ctx_adj_onoff_net"].notna()]
                    valid_links = set(zip(pctx_valid["athlete_id"], pctx_valid["season"]))
                    
                    # Pre-calculate internal name map for name-validation
                    con = duckdb.connect(str(DB_PATH), read_only=True)
                    name_map = con.execute("""
                        SELECT athlete_id, mode(name) as shooter_name FROM (
                            SELECT shooterAthleteId as athlete_id, shooter_name as name FROM stg_shots
                            UNION ALL
                            SELECT athleteId, athlete_name FROM stg_participants
                        ) GROUP BY 1
                    """).df()
                    con.close()
                    
                    # Standardize names in map
                    name_map["clean_name"] = clean_name_for_join(name_map["shooter_name"])
                    aid_to_name = name_map.set_index("athlete_id")["clean_name"].to_dict()

                    def is_missing_link(row):
                        aid, name, fseason = row["athlete_id"], row["clean_name"], row["target_season"]
                        if pd.isna(aid) or pd.isna(fseason): return True
                        
                        # Case A: Current ID/Season has NO data in pctx
                        has_data = (aid, fseason) in valid_links or (aid, fseason - 1) in valid_links
                        if not has_data: return True
                        
                        # Case B: Current ID points to a different name (Crosswalk Corruption)
                        if aid in aid_to_name:
                            if aid_to_name[aid] != name:
                                return True
                        
                        return False
                    
                    missing_mask = df.apply(is_missing_link, axis=1)
                    
                    if missing_mask.any():
                        logger.info(f"  Linkage Patch Debug: missing_mask count = {missing_mask.sum()}")
                        
                        pctx_m_ext = college_pathway_context.merge(name_map, on="athlete_id", how="inner")
                        pctx_m_ext["clean_name"] = clean_name_for_join(pctx_m_ext["shooter_name"])
                        pctx_v_ext = pctx_m_ext[pctx_m_ext["ctx_adj_onoff_net"].notna()]
                        
                        for shift in [0, 1]:
                            patch_c = pctx_v_ext.copy()
                            # Use season + shift to align pctx season with the prospect's draft/final season
                            patch_c["target_season"] = patch_c["season"] + shift
                            patch_map = patch_c.drop_duplicates(subset=["clean_name", "target_season"])
                            
                            patch = df[missing_mask].merge(
                                patch_map[["clean_name", "target_season", "athlete_id"]],
                                on=["clean_name", "target_season"],
                                how="inner",
                                suffixes=("", "_patched")
                            )
                            
                            if not patch.empty:
                                # Trace ALL cohort patches
                                trace_cohort = patch[patch.target_season.isin([2021, 2022, 2023])]
                                for _, p_row in trace_cohort.iterrows():
                                    logger.info(f"  [COHORT PATCH] NBA:{p_row['nba_id']} | NAME:{p_row['clean_name']} | AID:{p_row['athlete_id']} -> {p_row['athlete_id_patched']} | T_SEASON:{p_row['target_season']}")
                                
                                patch_dict = patch.set_index("nba_id")["athlete_id_patched"].to_dict()
                                # Update only the missing links
                                df["athlete_id"] = df["nba_id"].map(patch_dict).combine_first(df["athlete_id"])
                                # Update missing mask for next shift
                                missing_mask = df.apply(is_missing_link, axis=1)
                        
                        logger.info(f"  Applied name-based Linkage Patch: Updated athlete_ids for {len(df) - missing_mask.sum()} players")
                    
                    df = df.drop(columns=["clean_name", "target_season"], errors="ignore")
            except Exception as e:
                logger.warning(f"Failed to apply Linkage Patch: {e}")

        pctx = college_pathway_context.copy()
        pctx["season"] = pd.to_numeric(pctx["season"], errors="coerce")
        pctx_exact = pctx.rename(columns={"season": "college_final_season"})
        pctx_plus1 = pctx.copy()
        pctx_plus1["college_final_season"] = pctx_plus1["season"] + 1
        pctx_plus1 = pctx_plus1.drop(columns=["season"], errors="ignore")
        pctx_cols = [c for c in pctx_exact.columns if c not in {"athlete_id", "college_final_season"}]
        df = df.merge(
            pctx_exact[["athlete_id", "college_final_season"] + pctx_cols],
            on=["athlete_id", "college_final_season"],
            how="left",
        )
        p1_cols = ["athlete_id", "college_final_season"] + pctx_cols
        p1_rename = {c: f"{c}_p1" for c in pctx_cols}
        df = df.merge(
            pctx_plus1[p1_cols].rename(columns=p1_rename),
            on=["athlete_id", "college_final_season"],
            how="left",
        )
        numeric_path_cols = {"path_onoff_poss", "path_onoff_seconds", "path_onoff_reliability_weight"}
        for c in pctx_cols:
            c_p1 = f"{c}_p1"
            if c_p1 not in df.columns:
                continue
            base = df[c] if c in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
            if pd.api.types.is_numeric_dtype(base) or c.startswith("ctx_") or c in numeric_path_cols:
                base = pd.to_numeric(base, errors="coerce").astype(float)
                alt = pd.to_numeric(df[c_p1], errors="coerce").astype(float)
            else:
                alt = df[c_p1]
            # Prefer base, fill with alt where base is NaN (avoids FutureWarning from combine_first with empty/mixed dtypes).
            out = base.fillna(alt)
            df[c] = out
            df.drop(columns=[c_p1], inplace=True, errors="ignore")

        # Prior-season fallback:
        # if final-season context is missing, use nearest prior valid season
        # (real observed data only) within a bounded window.
        left = df[["athlete_id", "college_final_season"]].copy()
        left = left.reset_index(drop=False).rename(columns={"index": "_rid"})
        # Only fallback to rows that actually HAVE data.
        cand = pctx[pctx["ctx_adj_onoff_net"].notna()].copy()
        cand["cand_final_season"] = cand["season"] + 1
        cand = cand.drop(columns=["season"], errors="ignore")
        cand_join = left.merge(cand, on="athlete_id", how="left")
        cand_join["delta"] = (
            pd.to_numeric(cand_join["college_final_season"], errors="coerce")
            - pd.to_numeric(cand_join["cand_final_season"], errors="coerce")
        )
        cand_join = cand_join[
            cand_join["delta"].notna()
            & (cand_join["delta"] >= 0)
            & (cand_join["delta"] <= 4)
        ].copy()
        if not cand_join.empty:
            cand_join = cand_join.sort_values(["_rid", "delta"], ascending=[True, True])
            best = cand_join.drop_duplicates(subset=["_rid"], keep="first")
            best_cols = ["_rid"] + [c for c in pctx_cols if c in best.columns]
            best = best[best_cols].rename(columns={c: f"{c}_prior" for c in pctx_cols if c in best.columns})
            df = df.reset_index(drop=True)
            df["_rid"] = np.arange(len(df))
            df = df.merge(best, on="_rid", how="left")
            for c in pctx_cols:
                c_prior = f"{c}_prior"
                if c_prior not in df.columns:
                    continue
                base = df[c] if c in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
                if pd.api.types.is_numeric_dtype(base) or c.startswith("ctx_") or c in numeric_path_cols:
                    base = pd.to_numeric(base, errors="coerce").astype(float)
                    alt = pd.to_numeric(df[c_prior], errors="coerce").astype(float)
                else:
                    alt = df[c_prior]
                out = base.fillna(alt)
                df[c] = out
                df.drop(columns=[c_prior], inplace=True, errors="ignore")
            df.drop(columns=["_rid"], inplace=True, errors="ignore")
        logger.info(f"  Added college pathway context features (dual-season): {len(pctx_cols)}")

    # Impact aliases used by inference/ranking contract.
    alias_map = {
        "impact_off_net_raw": "college_off_net_rating",
        "impact_on_off_net_diff_raw": "college_on_off_net_diff",
        "impact_on_off_ortg_diff_raw": "college_on_off_ortg_diff",
        "impact_on_off_drtg_diff_raw": "college_on_off_drtg_diff",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    # Teammate quality proxy when on court (team-level fallback; lineup-level can be added later).
    if "college_teammate_quality_on" not in df.columns:
        if "college_team_srs" in df.columns:
            df["college_teammate_quality_on"] = pd.to_numeric(df["college_team_srs"], errors="coerce")
        elif "team_strength_srs" in df.columns:
            df["college_teammate_quality_on"] = pd.to_numeric(df["team_strength_srs"], errors="coerce")
        else:
            df["college_teammate_quality_on"] = np.nan

    # Activity provenance + masks (contract-enforced columns).
    core_cols = [
        "college_dunk_rate", "college_dunk_freq", "college_putback_rate",
        "college_rim_pressure_index", "college_contest_proxy",
    ]
    for c in core_cols:
        if c not in df.columns:
            df[c] = np.nan
    core_signal = df[core_cols].notna().any(axis=1)

    # Enforce explicit mask-first policy, then safe numeric fill to satisfy
    # hard coverage contracts without hiding missingness.
    for c in core_cols:
        mcol = f"{c}_missing"
        if mcol not in df.columns:
            df[mcol] = df[c].isna().astype(int)
        else:
            df[mcol] = pd.to_numeric(df[mcol], errors="coerce").fillna(df[c].isna().astype(int)).astype(int)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "college_activity_source" not in df.columns:
        df["college_activity_source"] = np.where(core_signal, "derived_fallback", "missing")
    else:
        fallback_src = pd.Series(np.where(core_signal, "derived_fallback", "missing"), index=df.index)
        src = df["college_activity_source"]
        src = src.where(src.notna(), fallback_src)
        src = src.where(src.astype(str).str.len() > 0, fallback_src)
        df["college_activity_source"] = src
    if "has_college_activity_features" not in df.columns:
        df["has_college_activity_features"] = core_signal.astype(int)
    else:
        df["has_college_activity_features"] = (
            pd.to_numeric(df["has_college_activity_features"], errors="coerce")
            .fillna(core_signal.astype(int))
            .astype(int)
        )

    mask_map = {
        "college_dunk_rate": "college_dunk_rate_missing",
        "college_dunk_freq": "college_dunk_freq_missing",
        "college_putback_rate": "college_putback_rate_missing",
        "college_rim_pressure_index": "college_rim_pressure_index_missing",
        "college_contest_proxy": "college_contest_proxy_missing",
    }
    for feat, mask in mask_map.items():
        if mask not in df.columns:
            df[mask] = df[feat].isna().astype(int)
        else:
            m = pd.to_numeric(df[mask], errors="coerce")
            df[mask] = np.where(m.notna(), m, df[feat].isna().astype(int)).astype(int)

    # Pathway context v2 contract flags.
    if "has_ctx_onoff_core" not in df.columns:
        ctx_core_cols = [c for c in [
            "ctx_adj_onoff_net", "ctx_adj_onoff_ortg", "ctx_adj_onoff_drtg",
            "ctx_adj_onoff_ast_per100", "ctx_adj_onoff_reb_per100",
            "ctx_adj_onoff_stl_per100", "ctx_adj_onoff_blk_per100", "ctx_adj_onoff_tov_per100",
            "ctx_adj_onoff_transition", "ctx_adj_onoff_dunk_pressure",
        ] if c in df.columns]
        df["has_ctx_onoff_core"] = df[ctx_core_cols].notna().any(axis=1).astype(int) if ctx_core_cols else 0
    else:
        df["has_ctx_onoff_core"] = pd.to_numeric(df["has_ctx_onoff_core"], errors="coerce").fillna(0).astype(int)

    if "has_ctx_velocity" not in df.columns:
        ctx_vel_cols = [c for c in [
            "ctx_vel_net_yoy", "ctx_vel_ortg_yoy", "ctx_vel_drtg_yoy",
            "ctx_vel_ast_yoy", "ctx_vel_reb_yoy", "ctx_vel_transition_yoy", "ctx_vel_dunk_pressure_yoy",
        ] if c in df.columns]
        df["has_ctx_velocity"] = df[ctx_vel_cols].notna().any(axis=1).astype(int) if ctx_vel_cols else 0
    else:
        df["has_ctx_velocity"] = pd.to_numeric(df["has_ctx_velocity"], errors="coerce").fillna(0).astype(int)

    # Join athlete-level college development-rate features.
    if not college_dev_rate.empty:
        dev = college_dev_rate.copy()
        dev_join_keys = ["athlete_id"]
        if "final_college_season" in dev.columns and "college_final_season" in df.columns:
            dev = dev.rename(columns={"final_college_season": "college_final_season"})
            dev_join_keys.append("college_final_season")
        dev_cols = dev_join_keys + [c for c in dev.columns if c not in dev_join_keys]
        df = df.merge(dev[dev_cols], on=dev_join_keys, how="left")
        logger.info(f"  Added college dev-rate features: {len(dev_cols) - len(dev_join_keys)}")

    # Join transfer summary.
    if not transfer_summary.empty:
        ts_cols = ["athlete_id"] + [c for c in transfer_summary.columns if c != "athlete_id"]
        df = df.merge(transfer_summary[ts_cols], on="athlete_id", how="left")
        logger.info(f"  Added transfer summary features: {len(ts_cols) - 1}")
    
    # Join trajectory features (new multi-season encoding)
    if use_trajectory_features and not trajectory_features.empty:
        traj_cols = ['athlete_id'] + [c for c in trajectory_features.columns if c != 'athlete_id']
        df = df.merge(trajectory_features[traj_cols], on='athlete_id', how='left', suffixes=('', '_traj'))
        logger.info(f"  Added {len(trajectory_features.columns)-1} trajectory features")
    
    # Join NBA targets
    if mode == "supervised":
        df = df.merge(nba_targets, on='nba_id', how='inner')
    else:
        df = df.merge(nba_targets, on='nba_id', how='left')
        
    # Drop target columns if foundation mode
    if mode == "foundation":
        drop_targs = [c for c in nba_targets.columns if c != 'nba_id']
        df = df.drop(columns=[c for c in drop_targs if c in df.columns])

    # Enforce Duplicates / Grain Contract
    if mode == "supervised":
        if 'nba_id' in df.columns:
            dup_count = int(df.duplicated(subset=['nba_id']).sum())
            if dup_count > 0:
                sort_cols = ['nba_id']
                ascending = [True]
                if 'college_final_season' in df.columns:
                    sort_cols.append('college_final_season')
                    ascending.append(False)
                df = df.sort_values(sort_cols, ascending=ascending).drop_duplicates(subset=['nba_id'], keep='first')
                logger.warning(f"Supervised Mode: Dropped {dup_count} duplicate nba_id rows.")
    elif mode in ["foundation", "joint"]:
        # Foundation/joint grain is athlete_id + season.
        dedupe_keys = ['athlete_id', 'season'] if 'season' in df.columns else ['athlete_id', 'college_final_season']
        dup_count = int(df.duplicated(subset=dedupe_keys).sum())
        if dup_count > 0:
            df = df.sort_values(dedupe_keys).drop_duplicates(subset=dedupe_keys, keep='first')
            logger.warning(f"{mode.capitalize()} Mode: Dropped {dup_count} duplicate athlete_id+season rows.")

    # Join draft/rookie metadata for cohort filtering + downstream audits.
    if not dim_nba.empty:
        dim_cols = [c for c in ["nba_id", "draft_year", "rookie_season_year", "player_name", "bbr_id", "ht_first", "ht_max", "wt_first", "wt_max", "ht_peak_delta", "wt_peak_delta"] if c in dim_nba.columns]
        df = df.merge(dim_nba[dim_cols], on="nba_id", how="left", suffixes=("", "_dim"))
        if "player_name_dim" in df.columns and "player_name" not in df.columns:
            df = df.rename(columns={"player_name_dim": "player_name"})
        # NBA-side physical fallback surface.
        nba_h_cm = pd.to_numeric(df.get("ht_first"), errors="coerce").combine_first(
            pd.to_numeric(df.get("ht_max"), errors="coerce")
        )
        nba_w_lbs = _to_lbs_from_mixed_weight(
            pd.to_numeric(df.get("wt_first"), errors="coerce")
        ).combine_first(
            _to_lbs_from_mixed_weight(pd.to_numeric(df.get("wt_max"), errors="coerce"))
        )
        df["nba_height_cm"] = nba_h_cm
        df["nba_weight_lbs"] = nba_w_lbs
        # Explicit physical development trajectory fields (NBA observed).
        df["nba_height_change_cm"] = pd.to_numeric(df.get("ht_peak_delta"), errors="coerce")
        df["nba_weight_change_lbs"] = pd.to_numeric(df.get("wt_peak_delta"), errors="coerce")
        if "nba_height_change_cm" in df.columns:
            df["nba_height_change_cm"] = df["nba_height_change_cm"].combine_first(
                pd.to_numeric(df.get("ht_max"), errors="coerce") - pd.to_numeric(df.get("ht_first"), errors="coerce")
            )
        if "nba_weight_change_lbs" in df.columns:
            df["nba_weight_change_lbs"] = df["nba_weight_change_lbs"].combine_first(
                pd.to_numeric(df.get("wt_max"), errors="coerce") - pd.to_numeric(df.get("wt_first"), errors="coerce")
            )
        # Fill college physicals from NBA fallback where missing.
        df["college_height_in"] = pd.to_numeric(df.get("college_height_in"), errors="coerce").combine_first(nba_h_cm / 2.54)
        df["college_weight_lbs"] = pd.to_numeric(df.get("college_weight_lbs"), errors="coerce").combine_first(nba_w_lbs)
        df["has_college_height"] = df["college_height_in"].notna().astype(int)
        df["has_college_weight"] = df["college_weight_lbs"].notna().astype(int)
    if not nba_ws_bridge.empty and "bbr_id" in df.columns:
        ws = nba_ws_bridge.copy()
        ws["bbr_id"] = ws["bbr_id"].astype(str).str.strip().str.lower()
        df["bbr_id"] = df["bbr_id"].astype(str).str.strip().str.lower()
        df = df.merge(ws, on="bbr_id", how="left")
        existing_ws = pd.to_numeric(df.get("wingspan_in"), errors="coerce")
        bridge_ws = pd.to_numeric(df.get("nba_wingspan_in"), errors="coerce")
        df["wingspan_in"] = existing_ws.combine_first(bridge_ws)
        if "wingspan_minus_height_in" not in df.columns:
            df["wingspan_minus_height_in"] = np.nan
        df["wingspan_minus_height_in"] = pd.to_numeric(df["wingspan_minus_height_in"], errors="coerce").combine_first(
            pd.to_numeric(df["wingspan_in"], errors="coerce") - pd.to_numeric(df["college_height_in"], errors="coerce")
        )
        df["has_wingspan"] = pd.to_numeric(df.get("has_wingspan"), errors="coerce").fillna(0)
        df["has_wingspan"] = np.where(pd.to_numeric(df["wingspan_in"], errors="coerce").notna(), 1, df["has_wingspan"]).astype(int)

    # Cohort filter: default to modern era where college-source coverage is aligned.
    # Fallback uses rookie season when draft year is missing.
    if min_draft_year is not None:
        if "draft_year" in df.columns or "rookie_season_year" in df.columns:
            draft_proxy = pd.to_numeric(df.get("draft_year"), errors="coerce")
            if "rookie_season_year" in df.columns:
                rookie_proxy = pd.to_numeric(df["rookie_season_year"], errors="coerce") - 1
                draft_proxy = draft_proxy.where(draft_proxy.notna(), rookie_proxy)
            before = len(df)
            keep_mask = draft_proxy >= int(min_draft_year)
            df = df.loc[keep_mask.fillna(False)].copy()
            dropped = before - len(df)
            logger.info(
                "Applied cohort filter draft_year_proxy >= %s: kept %d / %d (dropped %d)",
                min_draft_year, len(df), before, dropped,
            )
            df["draft_year_proxy"] = draft_proxy.loc[df.index]
        else:
            logger.warning("min_draft_year set but no draft_year/rookie_season_year columns available; skipping filter.")

    # Build target maturity + target-availability masks after rookie metadata join.
    # (Do this here, not earlier, to avoid all-null maturity fields.)
    df['target_asof_year'] = 2024
    if 'rookie_season_year' in df.columns:
        rookie_year = pd.to_numeric(df['rookie_season_year'], errors='coerce')
        df['epm_years_observed'] = (df['target_asof_year'] - rookie_year + 1).clip(lower=0)
        df['rapm_years_observed'] = (df['target_asof_year'] - rookie_year + 1).clip(lower=0)
        df['is_epm_mature'] = (
            pd.to_numeric(df['epm_years_observed'], errors='coerce').ge(3).fillna(False).astype(int)
        )
        df['is_rapm_mature'] = (
            pd.to_numeric(df['rapm_years_observed'], errors='coerce').ge(3).fillna(False).astype(int)
        )
    else:
        df['epm_years_observed'] = np.nan
        df['rapm_years_observed'] = np.nan
        df['is_epm_mature'] = 0
        df['is_rapm_mature'] = 0

    # Availability masks must reference real column names on this table.
    rapm_src = df['y_peak_ovr'] if 'y_peak_ovr' in df.columns else pd.Series(np.nan, index=df.index)
    df['has_peak_rapm_target'] = pd.to_numeric(rapm_src, errors='coerce').notna().astype(int)
    peak_epm_cols = [c for c in ['y_peak_epm_window', 'y_peak_epm_3y', 'y_peak_epm_2y', 'y_peak_epm_1y', 'y_peak_epm_1y_60gp'] if c in df.columns]
    if peak_epm_cols:
        df['has_peak_epm_target'] = (
            pd.concat([pd.to_numeric(df[c], errors='coerce') for c in peak_epm_cols], axis=1).notna().any(axis=1).astype(int)
        )
    else:
        df['has_peak_epm_target'] = 0
    y1_src = df['year1_epm_tot'] if 'year1_epm_tot' in df.columns else pd.Series(np.nan, index=df.index)
    df['has_year1_epm_target'] = pd.to_numeric(y1_src, errors='coerce').notna().astype(int)
    
    logger.info(f"Joined dataset: {len(df):,} rows")
    
    # 4. Compute derived features
    df = compute_derived_features(df)

    # 4b. Wingspan schema scaffolding (nullable by default).
    if "has_wingspan" not in df.columns:
        src_cols = [c for c in ["wingspan_in", "standing_reach_in", "wingspan_minus_height_in"] if c in df.columns]
        if src_cols:
            df["has_wingspan"] = df[src_cols].notna().any(axis=1).astype(int)
        else:
            df["has_wingspan"] = 0
    for col in WINGSPAN_SCHEMA_COLS:
        if col not in df.columns:
            if col == "has_wingspan":
                df[col] = 0
            else:
                df[col] = np.nan

    # Physicals contract columns.
    for col in ["college_height_in", "college_weight_lbs", "nba_height_cm", "nba_weight_lbs"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "has_college_height" not in df.columns:
        df["has_college_height"] = df["college_height_in"].notna().astype(int)
    if "has_college_weight" not in df.columns:
        df["has_college_weight"] = df["college_weight_lbs"].notna().astype(int)
    for col in ["college_height_delta_yoy", "college_weight_delta_yoy", "college_height_slope_3yr", "college_weight_slope_3yr"]:
        if col not in df.columns:
            df[col] = np.nan
    
    # 5. Era normalization was already applied on the full population in step 2b.
    #    Skipping here to avoid recomputing on the filtered cohort (which causes
    #    train-serve skew).
    
    # 6. Filter: require minimum number of targets (Supervised only)
    if mode == "supervised":
        target_cols = [PRIMARY_TARGET] + AUX_TARGETS
        target_cols = [c for c in target_cols if c in df.columns]
        
        if target_cols:
            target_count = df[target_cols].notna().sum(axis=1)
            df = df[target_count >= min_targets]
            logger.info(f"After target filter (min={min_targets}): {len(df):,} rows")
    
    # 7. Log coverage statistics
    log_coverage_stats(df)
    
    # 8. Active Feature Registry Gate
    feature_branches = [
        "dunk_rate", "putback_rate", "rim_pressure_index", "contest_proxy",
        "transition_freq", "deflection_proxy", "pressure_handle_proxy",
        "avg_shot_dist", "corner_3_rate", "deep_3_rate", "rim_purity"
    ]
    branch_health = []
    for branch in feature_branches:
        col = f"college_{branch}" if f"college_{branch}" in df.columns else branch
        if col in df.columns:
            non_null = df[col].notna().sum()
            non_zero = (pd.to_numeric(df[col], errors='coerce').fillna(0) > 0).sum()
            total = len(df)
            health = {
                "branch": branch,
                "feature_col": col,
                "non_null_rate": non_null / total if total > 0 else 0,
                "non_zero_rate": non_zero / total if total > 0 else 0,
            }
            # Gate: Auto-demote/mask any branch with < 1% non-zero rate
            if health['non_zero_rate'] < 0.01:
                health['status'] = 'INACTIVE_AUTO_MASKED'
                df[col] = np.nan
            else:
                health['status'] = 'ACTIVE'
            branch_health.append(health)
            
    if branch_health:
        health_df = pd.DataFrame(branch_health)
        health_df.to_csv(AUDIT_DIR / f"feature_branch_health_{mode}.csv", index=False)
        logger.info(f"Published feature branch health audit: feature_branch_health_{mode}.csv")
    
    # 9. Target Maturity Report
    if mode in ["supervised", "joint"] and "draft_year_proxy" in df.columns:
        mat_df = df.groupby("draft_year_proxy").agg(
            total_players=("athlete_id", "nunique"),
            epm_mature=("is_epm_mature", "sum"),
            rapm_mature=("is_rapm_mature", "sum")
        ).reset_index()
        mat_df.to_csv(AUDIT_DIR / f"target_maturity_report_{mode}.csv", index=False)
        logger.info(f"Published target maturity report: target_maturity_report_{mode}.csv")

    return df


def log_coverage_stats(df: pd.DataFrame) -> None:
    """Log coverage statistics for the training table."""
    logger.info("\n" + "=" * 40)
    logger.info("COVERAGE STATISTICS")
    logger.info("=" * 40)
    
    logger.info(f"Total rows: {len(df):,}")
    
    # Target coverage
    target_cols = [PRIMARY_TARGET] + AUX_TARGETS + [BINARY_TARGET]
    for col in target_cols:
        if col in df.columns:
            n = df[col].notna().sum()
            pct = n / len(df) * 100
            logger.info(f"  {col}: {n:,} ({pct:.1f}%)")

    # Development quality coverage
    for col in ['dev_rate_quality_weight', 'dev_rate_y1_y3_sd']:
        if col in df.columns:
            n = df[col].notna().sum()
            pct = n / len(df) * 100
            logger.info(f"  {col}: {n:,} ({pct:.1f}%)")

    # College development coverage
    for col in ['college_dev_quality_weight', 'college_dev_p50']:
        if col in df.columns:
            n = df[col].notna().sum()
            pct = n / len(df) * 100
            logger.info(f"  {col}: {n:,} ({pct:.1f}%)")

    # College impact coverage
    for col in ['has_impact_raw', 'has_impact_stint', 'has_impact_ripm', 'impact_reliability_weight']:
        if col in df.columns:
            if df[col].dtype.kind in {'i', 'u', 'f'} and col.startswith("has_"):
                pct = 100.0 * df[col].fillna(0).mean()
                logger.info(f"  {col}: {pct:.1f}%")
            else:
                n = df[col].notna().sum()
                pct = n / len(df) * 100
                logger.info(f"  {col}: {n:,} ({pct:.1f}%)")

    # Transfer coverage
    for col in ['has_transfer_context', 'transfer_event_count', 'transfer_max_shock']:
        if col in df.columns:
            n = df[col].notna().sum()
            pct = n / len(df) * 100
            logger.info(f"  {col}: {n:,} ({pct:.1f}%)")
    
    # Tier 2 coverage
    if 'has_spatial_data' in df.columns:
        n = df['has_spatial_data'].sum()
        pct = n / len(df) * 100
        logger.info(f"  has_spatial_data: {n:,} ({pct:.1f}%)")
    
    # Era distribution
    if 'college_final_season' in df.columns:
        logger.info("\nSeason distribution:")
        dist = df['college_final_season'].value_counts().sort_index()
        for season, count in dist.items():
            logger.info(f"  {season}: {count:,}")


def save_training_table(df: pd.DataFrame, filename: str = "unified_training_table.parquet") -> Path:
    """Save the training table to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved training table to {output_path}")
    return output_path


def save_games_played_audits(df: pd.DataFrame) -> None:
    """Persist season-level games-played source mix for gateing/debug."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    mix = games_source_mix_by_season(df, season_col="college_final_season")
    if not mix.empty:
        out = AUDIT_DIR / "games_played_source_mix_by_season.csv"
        mix.to_csv(out, index=False)
        logger.info(f"Saved games-played source mix audit to {out}")

    mmix = minutes_source_mix_by_season(df, season_col="college_final_season")
    if not mmix.empty:
        mout = AUDIT_DIR / "minutes_source_mix_by_season.csv"
        mmix.to_csv(mout, index=False)
        logger.info(f"Saved minutes source mix audit to {mout}")


def main():
    """Main entry point."""
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['supervised', 'foundation', 'joint'], default='supervised')
    args = parser.parse_args()
    
    logger.info(f"Starting build_unified_training_table in {args.mode.upper()} mode")
    
    df = build_unified_training_table(
        use_career_features=True,
        apply_normalization=True,
        min_targets=1,
        min_draft_year=2011 if args.mode == 'supervised' else None,
        mode=args.mode,
    )
    
    if df.empty:
        logger.error("Failed to build training table - no data!")
        return
    
    file_map = {
        'supervised': 'unified_training_table_supervised.parquet',
        'foundation': 'foundation_college_table.parquet',
        'joint': 'unified_training_table_joint.parquet'
    }
    
    filename = file_map[args.mode]
    save_training_table(df, filename=filename)
    save_games_played_audits(df)
    
    # Generate contract report
    if args.mode in ("foundation", "joint"):
        grain_keys = ["athlete_id", "season"] if "season" in df.columns else ["athlete_id", "college_final_season"]
        grain_dups = int(df.duplicated(subset=grain_keys).sum())
    else:
        grain_keys = ["nba_id"]
        grain_dups = int(df.duplicated(subset=["nba_id"]).sum()) if "nba_id" in df.columns else len(df)

    report = {
        "mode": args.mode,
        "row_count": len(df),
        "nba_id_nulls": int(df['nba_id'].isna().sum()) if 'nba_id' in df.columns else 0,
        "nba_id_unique": int(df['nba_id'].nunique()) if 'nba_id' in df.columns else 0,
        "athlete_id_unique": int(df['athlete_id'].nunique()) if 'athlete_id' in df.columns else 0,
        "grain_keys": grain_keys,
        "grain_duplicate_rows": grain_dups,
        "epm_years_observed_non_null_rate": float(pd.to_numeric(df.get("epm_years_observed"), errors="coerce").notna().mean()) if "epm_years_observed" in df.columns else 0.0,
    }
    rpt_path = AUDIT_DIR / f"surface_grain_contract_report_{args.mode}.json"
    with open(rpt_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Contract report saved: {rpt_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"UNIFIED TRAINING TABLE ({args.mode.upper()}) COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
