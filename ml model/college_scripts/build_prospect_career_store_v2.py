"""
Prospect Career Store Build Script (V2 - Phase 4 Ready)
=======================================================
Updates:
- Calculates `trueShootingPct` (TS%) derived from sub-components.
- Calculates `usage` (USG%) using Team Pace proxy.
- Adds specific `final_` columns required for NBA Feeding Plan Phase 4.

Input: data/college_feature_store/college_features_v1.parquet
Output: data/college_feature_store/prospect_career_v1.parquet
"""

import pandas as pd
import numpy as np
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = 'data/college_feature_store/college_features_v1.parquet'
OUTPUT_FILE = 'data/college_feature_store/prospect_career_v1.parquet'
OUTPUT_FILE_LONG = 'data/college_feature_store/prospect_career_long_v1.parquet'
WITHIN_SEASON_WINDOWS_FILE = 'data/college_feature_store/within_season_windows_v1.parquet'
WAREHOUSE_DB = 'data/warehouse.duckdb'

def load_games_played(warehouse_path: str) -> pd.DataFrame | None:
    if not os.path.exists(warehouse_path):
        logger.warning(f"Warehouse not found, skipping games played: {warehouse_path}")
        return None

    try:
        import duckdb
    except Exception as exc:
        logger.warning(f"DuckDB not available, skipping games played: {exc}")
        return None

    query = """
        SELECT
            f.athleteId AS athlete_id,
            g.season AS season,
            COUNT(DISTINCT f.gameId) AS games_played
        FROM fact_player_game f
        JOIN (
            SELECT CAST(id AS VARCHAR) AS gameId, season
            FROM dim_games
        ) g
        ON g.gameId = f.gameId
        GROUP BY 1, 2
    """

    con = duckdb.connect(warehouse_path, read_only=True)
    try:
        return con.sql(query).df()
    finally:
        con.close()


def load_within_season_windows(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        logger.info(f"Within-season windows not found (optional): {path}")
        return None

    df = pd.read_parquet(path)
    required = {'athlete_id', 'season'}
    if not required.issubset(df.columns):
        logger.warning(f"Within-season windows missing required cols {required}, skipping.")
        return None

    return df

def build_career_store():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    logger.info(f"Loading base features from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # Filter to overall stats
    df_all = df[df['split_id'] == 'ALL__ALL'].copy()
    df_all = df_all.sort_values(['athlete_id', 'season'])

    # The feature store may contain duplicate ALL__ALL rows (often exact duplicates).
    # Also, some athletes can appear multiple times within a season (e.g., data quirks or team changes).
    # For career modeling we want a single row per (athlete_id, season).
    df_all = df_all.drop_duplicates()

    # Meta columns (choose the row with max minutes_total per athlete-season).
    meta_cols = [
        'teamId', 'conference', 'is_power_conf', 'team_pace',
        'recruiting_rank', 'recruiting_stars', 'recruiting_rating', 'recruiting_missing'
    ]
    meta_cols = [c for c in meta_cols if c in df_all.columns]
    meta = (
        df_all.sort_values(['athlete_id', 'season', 'minutes_total'], ascending=[True, True, False])
        .drop_duplicates(['athlete_id', 'season'])[['athlete_id', 'season'] + meta_cols]
    )

    # Sum-count columns for athlete-season (safe after drop_duplicates()).
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
    ]
    sum_cols = [c for c in sum_cols if c in df_all.columns]
    agg = df_all.groupby(['athlete_id', 'season'], as_index=False)[sum_cols].sum()

    # Merge back meta and restore split_id.
    df_all = agg.merge(meta, how='left', on=['athlete_id', 'season'])
    df_all['split_id'] = 'ALL__ALL'

    # Recompute core rates and Tier-2 spatial features with explicit gating.
    rim_att = df_all['rim_att'].astype(float)
    rim_made = df_all['rim_made'].astype(float)
    three_att = df_all['three_att'].astype(float)
    three_made = df_all['three_made'].astype(float)
    ft_att = df_all['ft_att'].astype(float)
    ft_made = df_all['ft_made'].astype(float)

    df_all['rim_fg_pct'] = np.where(rim_att > 0, rim_made / rim_att, np.nan)
    df_all['three_fg_pct'] = np.where(three_att > 0, three_made / three_att, np.nan)
    df_all['ft_pct'] = np.where(ft_att > 0, ft_made / ft_att, np.nan)

    shots_total = df_all['shots_total'].astype(float)
    xy_shots = df_all['xy_shots'].astype(float)
    xy_3_shots = df_all['xy_3_shots'].astype(float)
    xy_rim_shots = df_all['xy_rim_shots'].astype(float)

    df_all['xy_coverage'] = np.where(shots_total > 0, xy_shots / shots_total, np.nan)

    # Tier 2 gating thresholds (see PROJECT_MAP.md Spatial Data Dictionary).
    gate_xy = xy_shots >= 25
    gate_xy_3 = xy_3_shots >= 15
    gate_xy_rim = xy_rim_shots >= 20

    sum_dist_ft = df_all['sum_dist_ft'].astype(float)
    sum_dist_sq_ft = df_all['sum_dist_sq_ft'].astype(float)

    avg_dist = np.where(gate_xy & (xy_shots > 0), sum_dist_ft / xy_shots, np.nan)
    var_dist = np.where(
        gate_xy & (xy_shots > 0),
        (sum_dist_sq_ft / xy_shots) - np.square(avg_dist),
        np.nan
    )
    df_all['avg_shot_dist'] = avg_dist
    df_all['shot_dist_var'] = np.where(np.isnan(var_dist), np.nan, np.maximum(0.0, var_dist))

    corner_3_att = df_all['corner_3_att'].astype(float)
    corner_3_made = df_all['corner_3_made'].astype(float)
    deep_3_att = df_all['deep_3_att'].astype(float)

    df_all['corner_3_rate'] = np.where(gate_xy_3 & (three_att > 0), corner_3_att / three_att, np.nan)
    df_all['corner_3_pct'] = np.where(gate_xy_3 & (corner_3_att > 0), corner_3_made / corner_3_att, np.nan)
    df_all['deep_3_rate'] = np.where(gate_xy_3 & (three_att > 0), deep_3_att / three_att, np.nan)

    rim_rest_att = df_all['rim_rest_att'].astype(float)
    df_all['rim_purity'] = np.where(gate_xy_rim & (rim_att > 0), rim_rest_att / rim_att, np.nan)

    # Optional: games played from DuckDB
    games_played = load_games_played(WAREHOUSE_DB)
    if games_played is not None:
        df_all = df_all.merge(games_played, how='left', on=['athlete_id', 'season'])
    else:
        df_all['games_played'] = np.nan

    # Optional: within-season window features (per athlete-season)
    ws = load_within_season_windows(WITHIN_SEASON_WINDOWS_FILE)
    if ws is not None:
        df_all = df_all.merge(ws, how='left', on=['athlete_id', 'season'])
    
    # -------------------------------------------------------------------------
    # 1. Feature Engineering (Phase 4 Additions)
    # -------------------------------------------------------------------------
    logger.info("Deriving Phase 4 metrics (TS%, Usage)...")
    
    # A. Derive Points (NaN-safe)
    # pts = 2*rim_made + 2*mid_made + 3*three_made + ft_made
    # Important: do not convert missing data to 0. Only treat as 0 when a column is present and truly 0.
    rim_made = df_all['rim_made'].astype(float)
    mid_made = df_all['mid_made'].astype(float)
    three_made = df_all['three_made'].astype(float)
    ft_made = df_all['ft_made'].astype(float)

    any_pts_inputs = rim_made.notna() | mid_made.notna() | three_made.notna() | ft_made.notna()
    df_all['points_derived'] = np.where(
        any_pts_inputs,
        2 * rim_made.fillna(0) + 2 * mid_made.fillna(0) + 3 * three_made.fillna(0) + ft_made.fillna(0),
        np.nan
    )
    
    # B. Derive True Shooting %
    # TS% = Pts / (2 * (FGA + 0.44 * FTA))
    # Treat missing attempts as missing (NaN), not 0.
    fga_total = df_all['fga_total'].astype(float)
    fta_total = df_all['ft_att'].astype(float)
    ts_denom = 2 * (fga_total + 0.44 * fta_total)
    
    df_all['trueShootingPct'] = np.where(ts_denom > 0, df_all['points_derived'] / ts_denom, np.nan)
    
    # C. Derive Usage Rate Proxy
    # Formula: (FGA + 0.44*FTA + TOV) / Possessions
    # Possessions Proxy = (Minutes / 40.0) * TeamPace
    # Note: team_pace is usually ~70. If missing, we assume 70.0
    
    team_pace = df_all['team_pace'].fillna(68.0) # Conservative NCAA avg
    minutes = df_all['minutes_total'].astype(float)
    tov = df_all['tov_total'].astype(float)
    
    # NEW: Possession Load (Volume)
    # Critical for bridging gaps when Minutes are missing (2006-2024)
    # Volume proxy: only compute if we have at least one component (prevents NaN -> 0 leakage)
    any_poss_inputs = fga_total.notna() | fta_total.notna() | tov.notna()
    df_all['poss_total'] = np.where(
        any_poss_inputs,
        fga_total.fillna(0) + 0.44 * fta_total.fillna(0) + tov.fillna(0),
        np.nan
    )
    
    est_possessions = (minutes / 40.0) * team_pace
    
    # Usage Rate (Intensity) - NaN if minutes=0
    df_all['usage'] = np.where(
        (est_possessions > 5) & df_all['poss_total'].notna(),
        df_all['poss_total'] / est_possessions,
        np.nan
    )
    
    # Clip usage to reasonable bounds (0.0 to 0.5) to avoid noise from small minutes
    df_all['usage'] = df_all['usage'].clip(0, 0.60)

    # D. Per-Game Rates (when games_played is available)
    gp = df_all['games_played']
    df_all['minutes_per_game'] = np.where(gp > 0, minutes / gp, np.nan)
    df_all['poss_per_game'] = np.where(gp > 0, df_all['poss_total'] / gp, np.nan)

    # -------------------------------------------------------------------------
    # 2. Aggregation Logic
    # -------------------------------------------------------------------------
    
    # Define metrics to track longitudinally
    available_metrics = [
        'minutes_total', 'fga_total', 'ast_total', 'tov_total',
        'stl_total', 'blk_total', 'rim_fg_pct', 'three_fg_pct', 'ft_pct',
        'trueShootingPct', 'usage', 'poss_total', 'games_played', # NEW
        'avg_shot_dist', 'corner_3_rate', 'corner_3_pct', 'xy_coverage', # SPATIAL TIER 2
        'deep_3_rate', 'rim_purity', 'shot_dist_var', # SPATIAL EXTENDED
        # Within-season windows (optional)
        'games_played_pg',
        'has_ws_last5', 'ws_minutes_last5', 'ws_fga_last5', 'ws_pts_last5',
        'ws_pps_last5', 'ws_on_net_rating_last5_mean',
        'has_ws_last10', 'ws_minutes_last10', 'ws_fga_last10', 'ws_pts_last10',
        'ws_pps_last10', 'ws_on_net_rating_last10_mean',
        'has_ws_prev5', 'ws_delta_pps_last5_minus_prev5', 'ws_delta_minutes_last5_minus_prev5',
        'has_ws_prev10', 'ws_delta_pps_last10_minus_prev10', 'ws_delta_minutes_last10_minus_prev10',
        'has_ws_breakout_timing_minutes', 'ws_breakout_timing_minutes',
        'has_ws_breakout_timing_volume', 'ws_breakout_timing_volume',
        'has_ws_breakout_timing_eff', 'ws_breakout_timing_eff',
    ]
    # Filter to what actually exists (safety)
    available_metrics = [m for m in available_metrics if m in df_all.columns]
    
    logger.info(f"Generating vectorized metrics for {len(df_all):,} rows...")

    # Career Metadata
    df_all = df_all.sort_values(['athlete_id', 'season'])
    df_all['season_rank'] = df_all.groupby('athlete_id').cumcount() + 1
    career_counts = df_all.groupby('athlete_id').size().rename('career_years')

    # -------------------------------------------------------------------------
    # 2.1 Breakout Timing Features (Non-Hardcoded, Exposure-Aware)
    # -------------------------------------------------------------------------
    # Late/early breakout is best treated as a continuous timing signal,
    # and its importance can be learned differently across archetypes downstream.
    #
    # We avoid hard thresholds by weighting rate metrics by sqrt(exposure).
    # This reduces the influence of tiny-sample seasons without binning.
    exposure = df_all['minutes_total'].fillna(0).astype(float)
    if 'games_played' in df_all.columns:
        # Prefer minutes as exposure. If minutes are missing but games exist, use games.
        exposure = np.where(exposure > 0, exposure, df_all['games_played'].fillna(0).astype(float) * 20.0)
        exposure = pd.Series(exposure, index=df_all.index)

    expo_w = np.sqrt(np.clip(exposure, 0, None))

    # Scores used only to locate timing of "peak" seasons along different dimensions.
    # Using poss_total (volume) and TS/usage weighted by exposure is robust and cheap.
    poss = df_all['poss_total'].fillna(0).astype(float)
    ts = df_all['trueShootingPct'].astype(float)
    usg = df_all['usage'].astype(float)

    df_all['score_breakout_volume'] = poss
    df_all['score_breakout_usage'] = np.where(
        usg.notna() & poss.notna(),
        usg * np.sqrt(np.clip(poss, 0, None)),
        np.nan
    )
    df_all['score_breakout_eff'] = np.where(
        ts.notna() & expo_w.notna(),
        ts * expo_w,
        np.nan
    )

    def _idxmax(group_col: str) -> pd.Series:
        s = df_all[group_col].replace([np.inf, -np.inf], np.nan)
        # If all seasons are missing for a player, idxmax returns NaN. Fill with a large negative.
        s = s.fillna(-1e18)
        return s.groupby(df_all['athlete_id']).idxmax()

    idx_vol = _idxmax('score_breakout_volume')
    idx_usg = _idxmax('score_breakout_usage')
    idx_eff = _idxmax('score_breakout_eff')

    has_vol = df_all['score_breakout_volume'].notna().groupby(df_all['athlete_id']).any()
    has_usg = df_all['score_breakout_usage'].notna().groupby(df_all['athlete_id']).any()
    has_eff = df_all['score_breakout_eff'].notna().groupby(df_all['athlete_id']).any()

    rank_vol = df_all.loc[idx_vol].set_index('athlete_id')['season_rank'].where(has_vol)
    rank_usg = df_all.loc[idx_usg].set_index('athlete_id')['season_rank'].where(has_usg)
    rank_eff = df_all.loc[idx_eff].set_index('athlete_id')['season_rank'].where(has_eff)

    breakout_ranks = pd.DataFrame({
        'breakout_rank_volume': rank_vol,
        'breakout_rank_usage': rank_usg,
        'breakout_rank_eff': rank_eff,
    }).join(career_counts)
    denom = np.maximum(1, breakout_ranks['career_years'] - 1)
    breakout_ranks['breakout_timing_volume'] = (breakout_ranks['breakout_rank_volume'] - 1) / denom
    breakout_ranks['breakout_timing_usage'] = (breakout_ranks['breakout_rank_usage'] - 1) / denom
    breakout_ranks['breakout_timing_eff'] = (breakout_ranks['breakout_rank_eff'] - 1) / denom

    # A single summary index (still continuous) for convenience.
    breakout_ranks['breakout_timing_avg'] = breakout_ranks[
        ['breakout_timing_volume', 'breakout_timing_usage', 'breakout_timing_eff']
    ].mean(axis=1)
    
    # First/Final season rows must reflect the actual first/last season rows,
    # not the last non-null value (pandas GroupBy.first/last skip NaNs per-column).
    df_first_row = df_all.groupby('athlete_id').head(1).set_index('athlete_id')
    df_final = df_all.groupby('athlete_id').tail(1).set_index('athlete_id')
    
    # Vectorized Deltas (YoY)
    # Do not convert missing to 0; 0 is only valid when both seasons exist and the delta is truly 0.
    for m in available_metrics:
        prev = df_all.groupby('athlete_id')[m].shift(1)
        cur = df_all[m]
        d = df_all.groupby('athlete_id')[m].diff()
        # Keep NaN when either side is missing
        d = d.where(cur.notna() & prev.notna(), np.nan)
        # Define first-season delta as 0 only when the first season value exists
        d = d.where(~((df_all['season_rank'] == 1) & cur.notna()), 0.0)
        df_all[f'delta_{m}'] = d
    
    # Trajectory (Simple Slope: Final - First / Years)
    first_vals = df_first_row[available_metrics]
    last_vals = df_final[available_metrics]
    
    stats = pd.concat([first_vals.add_prefix('first_'), last_vals.add_prefix('last_')], axis=1)
    stats = stats.join(career_counts)
    
    for m in available_metrics:
        stats[f'slope_{m}'] = (stats[f'last_{m}'] - stats[f'first_{m}']) / np.maximum(1, stats['career_years'] - 1)

    # Weighted Career (Recency weighted sum)
    df_all = df_all.merge(career_counts.to_frame('total_years'), left_on='athlete_id', right_index=True)
    df_all['weight'] = 1.0 - (df_all['total_years'] - df_all['season_rank']) * 0.2
    df_all['weight'] = df_all['weight'].clip(lower=0.2)
    
    weighted_stats = pd.DataFrame(index=career_counts.index)
    # Weighted means should ignore missing values rather than treating them as 0.
    for m in available_metrics:
        val = df_all[m]
        w = df_all['weight']
        num = (val * w).where(val.notna(), 0.0)
        den = w.where(val.notna(), 0.0)
        num_s = num.groupby(df_all['athlete_id']).sum()
        den_s = den.groupby(df_all['athlete_id']).sum()
        weighted_stats[f'career_wt_{m}'] = num_s / den_s.replace(0, np.nan)

    # Physical Growth
    if 'heightInches' in df_all.columns:
        stats['height_at_entry'] = df_all.groupby('athlete_id')['heightInches'].first()
        stats['height_final'] = df_all.groupby('athlete_id')['heightInches'].last()
        stats['delta_height'] = stats['height_final'] - stats['height_at_entry']
    else:
        stats['delta_height'] = 0.0

    # -------------------------------------------------------------------------
    # 3. Output Assembly
    # -------------------------------------------------------------------------
    
    # Base: Final Season Context
    final_output = df_final[['season', 'teamId']].join(career_counts)
    
    # Add Computed Slopes
    final_output = final_output.join(stats[[f'slope_{m}' for m in available_metrics] + ['delta_height']])
    
    # Add Weighted Career Averages
    final_output = final_output.join(weighted_stats)
    
    # Add Final YoY Deltas
    deltas_final = df_all.groupby('athlete_id')[[f'delta_{m}' for m in available_metrics]].last()
    final_output = final_output.join(deltas_final)
    
    # Add Final Snapshots (Critical for Gap Analysis)
    # We explicitly map `trueShootingPct` -> `final_trueShootingPct` here
    final_output = final_output.join(last_vals.add_prefix('final_'))

    # Add breakout timing features
    final_output = final_output.join(breakout_ranks[[
        'breakout_rank_volume', 'breakout_rank_usage', 'breakout_rank_eff',
        'breakout_timing_volume', 'breakout_timing_usage', 'breakout_timing_eff',
        'breakout_timing_avg'
    ]])
    
    # Calculate Per-Game Volume (No Minutes Dependency)
    if 'final_games_played' in final_output.columns:
        final_output['final_poss_per_game'] = np.where(
            final_output['final_games_played'] > 0,
            final_output['final_poss_total'] / final_output['final_games_played'],
            np.nan
        )
    else:
        final_output['final_poss_per_game'] = np.nan
    # Wait, career_years is integer count of seasons.
    # We need Games Played count in the final season?
    # We don't have games_played column in college_features!
    # college_features row count is usually 1, but exposure column?
    # shots_total / fga_total is total.
    # We need games played count.
    # We can infer from `v_shots_augmented` grouped by gameId?
    # No, that's expensive.
    # We will assume ~30 games? No.
    # We'll stick to 'final_poss_total' for now, or just leave it as volume.
    
    # Correction: Feature store doesn't have games_played.
    # We will compute 'final_poss_per_shot' = poss_total / fga_total?
    # No, that's meaningless.
    
    # Let's just output final_poss_total. It helps separate high volume vs low.

    final_output = final_output.reset_index()

    # Long format output (per-season, ALL__ALL only)
    long_cols = [
        'athlete_id', 'season', 'teamId', 'season_rank', 'career_years',
        'games_played', 'minutes_total', 'minutes_per_game', 'fga_total',
        'ast_total', 'tov_total', 'stl_total', 'blk_total', 'rim_fg_pct',
        'three_fg_pct', 'ft_pct', 'trueShootingPct', 'usage', 'poss_total',
        'poss_per_game', 'avg_shot_dist', 'corner_3_rate', 'corner_3_pct',
        'xy_coverage', 'deep_3_rate', 'rim_purity', 'shot_dist_var',
        # Within-season windows (optional)
        'games_played_pg',
        'has_ws_last5', 'ws_minutes_last5', 'ws_fga_last5', 'ws_pts_last5',
        'ws_pps_last5', 'ws_on_net_rating_last5_mean',
        'has_ws_last10', 'ws_minutes_last10', 'ws_fga_last10', 'ws_pts_last10',
        'ws_pps_last10', 'ws_on_net_rating_last10_mean',
        'has_ws_prev5', 'ws_delta_pps_last5_minus_prev5', 'ws_delta_minutes_last5_minus_prev5',
        'has_ws_prev10', 'ws_delta_pps_last10_minus_prev10', 'ws_delta_minutes_last10_minus_prev10',
        'has_ws_breakout_timing_minutes', 'ws_breakout_timing_minutes',
        'has_ws_breakout_timing_volume', 'ws_breakout_timing_volume',
        'has_ws_breakout_timing_eff', 'ws_breakout_timing_eff',
    ]
    long_cols = [c for c in long_cols if c in df_all.columns]
    career_long = df_all[long_cols].copy()
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_output.to_parquet(OUTPUT_FILE, index=False)
    career_long.to_parquet(OUTPUT_FILE_LONG, index=False)
    logger.info(f"Saved optimized career store to {OUTPUT_FILE} ({len(final_output):,} athletes)")
    logger.info(f"Saved career long store to {OUTPUT_FILE_LONG} ({len(career_long):,} rows)")

if __name__ == "__main__":
    build_career_store()
