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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
COLLEGE_FEATURE_STORE = BASE_DIR / "data/college_feature_store"
WAREHOUSE_V2 = BASE_DIR / "data/warehouse_v2"
OUTPUT_DIR = BASE_DIR / "data/training"

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
    # Historical manual PBP seasons are stored as season start-year (e.g., 2021 for 2021-22).
    # Project feature surfaces use season end-year (2022), so shift by +1 here.
    bf["season"] = pd.to_numeric(bf["season"], errors="coerce").astype("Int64") + 1
    bf["norm_name"] = bf["player_name"].map(_norm_hist_player_name)
    bf["norm_team"] = bf.get("team_name", pd.Series(dtype=str)).map(_norm_hist_team_name)
    bf = bf.dropna(subset=["season"])
    bf = bf[bf["norm_name"] != ""]

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

    mapped = pd.concat([mapped_exact, mapped_fallback], ignore_index=True)

    if mapped.empty:
        logger.warning("Historical exposure backfill mapping produced 0 rows")
        return pd.DataFrame()

    # Collapse possible team-level duplicates into athlete-season totals.
    # We take the max here instead of sum to protect against any residual name collision artifacts.
    grouped = mapped.groupby(["athlete_id", "season"], as_index=False).agg(
        backfill_minutes_total=("minutes_derived", "max"),
        backfill_tov_total=("turnovers_derived", "max"),
    )
    if "games_derived" in mapped.columns:
        g2 = mapped.groupby(["athlete_id", "season"], as_index=False).agg(
            backfill_games_played=("games_derived", "max")
        )
        grouped = grouped.merge(g2, on=["athlete_id", "season"], how="left")
    else:
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
              CAST(season_start + 1 AS BIGINT) AS season,
              player_name,
              COUNT(DISTINCT game_id) AS hist_games_played_text
            FROM names
            WHERE player_name IS NOT NULL
              AND TRIM(player_name) <> ''
              AND UPPER(TRIM(player_name)) <> 'TEAM'
              AND UPPER(TRIM(player_name)) <> 'TEAM.'
            GROUP BY 1,2
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

    out = (
        mapped.groupby(["athlete_id", "season"], as_index=False)["hist_games_played_text"]
        .max()
    )
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
            "ht_first", "ht_max", "wt_first", "wt_max",
            "ht_peak_delta", "wt_peak_delta",
        ] if c in df.columns
    ]
    df = df[keep].drop_duplicates(subset=["nba_id"])
    logger.info(f"Loaded {len(df):,} NBA dimension rows")
    return df


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
    activity_proxies = load_activity_proxies()
    exposure_backfill = load_historical_exposure_backfill()
    hist_text_games = load_historical_text_games_backfill()
    college_impact_stack = load_college_impact_stack()
    
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
        
        # Games played: same coalesce behavior (derived first, preserve source fallback).
        if 'college_games_played' in college_features.columns:
            if 'games_played' in college_features.columns:
                college_features['games_played'] = (
                    pd.to_numeric(college_features['college_games_played'], errors='coerce')
                    .combine_first(pd.to_numeric(college_features['games_played'], errors='coerce'))
                )
            else:
                college_features['games_played'] = pd.to_numeric(college_features['college_games_played'], errors='coerce')
            
            # CRITICAL: Drop the pre-named column so it doesn't fatally collide when get_final_college_season()
            # attempts to rename the freshly patched 'games_played' target back to 'college_games_played'!
            college_features = college_features.drop(columns=['college_games_played'])
             
        logger.info(f"Merged derived box stats. Rows: {og_len} -> {len(college_features)}")

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

    career_features = load_career_features()
    trajectory_features = load_trajectory_features()
    college_dev_rate = load_college_dev_rate()
    transfer_summary = load_transfer_context_summary()
    crosswalk = load_crosswalk()
    dim_nba = load_dim_player_nba()
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
    
    # 3. Join college features via crosswalk
    # crosswalk has: athlete_id (college) -> nba_id
    logger.info("Joining college features to NBA targets via crosswalk...")
    
    base_cols = ['athlete_id', 'nba_id']
    for c in ['wingspan_in', 'standing_reach_in', 'wingspan_minus_height_in']:
        if c in crosswalk.columns:
            base_cols.append(c)
    df = crosswalk[base_cols].copy()
    
    # Join final college season
    if not final_college.empty:
        df = df.merge(final_college, on='athlete_id', how='left')

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
    if "college_height_in" not in df.columns:
        df["college_height_in"] = pd.to_numeric(df.get("recruit_height_in"), errors="coerce")
    else:
        df["college_height_in"] = pd.to_numeric(df["college_height_in"], errors="coerce").combine_first(
            pd.to_numeric(df.get("recruit_height_in"), errors="coerce")
        )
    if "college_weight_lbs" not in df.columns:
        df["college_weight_lbs"] = pd.to_numeric(df.get("recruit_weight_lbs"), errors="coerce")
    else:
        df["college_weight_lbs"] = pd.to_numeric(df["college_weight_lbs"], errors="coerce").combine_first(
            pd.to_numeric(df.get("recruit_weight_lbs"), errors="coerce")
        )

    # Join college impact stack by final season when available.
    if not college_impact_stack.empty and "college_final_season" in df.columns:
        impact = college_impact_stack.copy()
        # Align season key to final season key for one-row joins.
        impact = impact.rename(columns={"season": "college_final_season"})
        impact_cols = ["athlete_id", "college_final_season"] + [
            c for c in impact.columns if c not in {"athlete_id", "college_final_season"}
        ]
        df = df.merge(impact[impact_cols], on=["athlete_id", "college_final_season"], how="left")
        logger.info(f"  Added college impact stack features: {len(impact_cols) - 2}")

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
    df = df.merge(nba_targets, on='nba_id', how='inner')  # Inner: must have at least one target

    # Enforce one row per nba_id to keep downstream joins deterministic.
    # Crosswalk noise can occasionally produce duplicate athlete->nba mappings.
    if 'nba_id' in df.columns:
        dup_count = int(df.duplicated(subset=['nba_id']).sum())
        if dup_count > 0:
            sort_cols = ['nba_id']
            ascending = [True]
            if 'college_final_season' in df.columns:
                sort_cols.append('college_final_season')
                ascending.append(False)
            if 'career_years' in df.columns:
                sort_cols.append('career_years')
                ascending.append(False)
            df = df.sort_values(sort_cols, ascending=ascending).drop_duplicates(subset=['nba_id'], keep='first')
            logger.warning(
                "Detected %d duplicate nba_id rows from joins; kept one row per nba_id (now %d rows).",
                dup_count,
                len(df),
            )

    # Join draft/rookie metadata for cohort filtering + downstream audits.
    if not dim_nba.empty:
        dim_cols = [c for c in ["nba_id", "draft_year", "rookie_season_year", "player_name", "ht_first", "ht_max", "wt_first", "wt_max", "ht_peak_delta", "wt_peak_delta"] if c in dim_nba.columns]
        df = df.merge(dim_nba[dim_cols], on="nba_id", how="left", suffixes=("", "_dim"))
        if "player_name_dim" in df.columns and "player_name" not in df.columns:
            df = df.rename(columns={"player_name_dim": "player_name"})
        # NBA-side physical fallback surface.
        nba_h_cm = pd.to_numeric(df.get("ht_first"), errors="coerce").combine_first(
            pd.to_numeric(df.get("ht_max"), errors="coerce")
        )
        nba_w_lbs = pd.to_numeric(df.get("wt_first"), errors="coerce").combine_first(
            pd.to_numeric(df.get("wt_max"), errors="coerce")
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
    
    # 5. Era normalization was already applied on the full population in step 2b.
    #    Skipping here to avoid recomputing on the filtered cohort (which causes
    #    train-serve skew).
    
    # 6. Filter: require minimum number of targets
    target_cols = [PRIMARY_TARGET] + AUX_TARGETS
    target_cols = [c for c in target_cols if c in df.columns]
    
    if target_cols:
        target_count = df[target_cols].notna().sum(axis=1)
        df = df[target_count >= min_targets]
        logger.info(f"After target filter (min={min_targets}): {len(df):,} rows")
    
    # 7. Log coverage statistics
    log_coverage_stats(df)
    
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


def main():
    """Main entry point."""
    # Build the table
    df = build_unified_training_table(
        use_career_features=True,
        apply_normalization=True,
        min_targets=1,
        min_draft_year=2011,
    )
    
    if df.empty:
        logger.error("Failed to build training table - no data!")
        return
    
    # Save
    save_training_table(df)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("UNIFIED TRAINING TABLE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Feature columns: {[c for c in df.columns if c.startswith('college_') or c.startswith('final_') or c.startswith('career_')]}")


if __name__ == "__main__":
    main()
