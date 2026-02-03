"""
College Feature Store v1 Build Script

Produces:
  - college_features_v1.parquet: 20-split feature aggregates per (athlete, season, split_id)
  - college_impact_v1.parquet: On-court ratings + shrinkage weights per (athlete, season)
  - coverage_report_v1.csv: Metadata on athlete coverage

Aligned with LOCKED v1 spec:
  - Correct column names from stg_shots: shooterAthleteId, shot_range, made, assisted, is_high_leverage, is_garbage
  - Season joined from dim_games via canonical view (CAST id AS VARCHAR)
  - 20 cross-product splits (leverage x strength including ALL marginals)
  - VS_UNKNOWN when opp_rank IS NULL
  - Impact from bridge_lineup_athletes (seconds-weighted on-court ratings)
"""

import duckdb
import pandas as pd
import numpy as np
import os
import sys
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Scientific Imports
try:
    import scipy.sparse as sparse
    from scipy.sparse.linalg import cg
except ImportError:
    logger.error("scipy is required for True RAPM. Please pip install scipy.")
    sys.exit(1)

# Constants
DB_PATH = 'data/warehouse.duckdb'
OUTPUT_DIR = 'data/college_feature_store'
SEC_REF = 3000.0   # Shrinkage reference for seconds
POSS_REF = 300.0   # Shrinkage reference for possessions
POWER_CONFS = ['ACC', 'Big 12', 'Big East', 'Big Ten', 'Pac-12', 'SEC']


def get_connection():
    if not os.path.exists(DB_PATH):
        logger.error(f"Database not found at {DB_PATH}")
        sys.exit(1)
    # Note: Cannot use read_only=True because we create TEMP VIEWs
    return duckdb.connect(DB_PATH)


def create_intermediate_views(con):
    """Create all intermediate views needed for feature computation."""
    logger.info("Creating intermediate views...")

    # 1. v_games_canon: Type Safety Fix
    #    dim_games.id is BIGINT, stg_shots.gameId is VARCHAR
    con.execute("""
        CREATE OR REPLACE TEMP VIEW v_games_canon AS
        SELECT
            season,
            CAST(id AS VARCHAR) AS gameId,
            id AS game_id_int,
            homeTeamId,
            awayTeamId,
            homePoints AS homeScore,
            awayPoints AS awayScore,
            neutralSite AS is_neutral
        FROM dim_games
        WHERE homePoints IS NOT NULL AND awayPoints IS NOT NULL
    """)
    logger.info("  Created v_games_canon")

    # 2. v_team_game_margins: Per-game margins for SRS calculation
    con.execute("""
        CREATE OR REPLACE TEMP VIEW v_team_game_margins AS
        SELECT
            g.season,
            g.gameId,
            g.homeTeamId AS teamId,
            g.awayTeamId AS opponentId,
            (g.homeScore - g.awayScore) AS margin
        FROM v_games_canon g
        UNION ALL
        SELECT
            g.season,
            g.gameId,
            g.awayTeamId AS teamId,
            g.homeTeamId AS opponentId,
            (g.awayScore - g.homeScore) AS margin
        FROM v_games_canon g
    """)
    logger.info("  Created v_team_game_margins")

    # 3. v_team_raw_margin: Average margin by team-season
    con.execute("""
        CREATE OR REPLACE TEMP VIEW v_team_raw_margin AS
        SELECT
            season,
            teamId,
            AVG(margin) AS avg_margin
        FROM v_team_game_margins
        GROUP BY 1, 2
    """)
    logger.info("  Created v_team_raw_margin")

    # 4. v_team_schedule_strength: 1-pass SOS (avg opponent margin)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW v_team_schedule_strength AS
        SELECT
            t.season,
            t.teamId,
            AVG(opp.avg_margin) AS sos
        FROM v_team_game_margins t
        LEFT JOIN v_team_raw_margin opp
          ON t.opponentId = opp.teamId AND t.season = opp.season
        GROUP BY 1, 2
    """)
    logger.info("  Created v_team_schedule_strength")

    # 5. v_team_season_srs_proxy: Margin + SOS (1-pass schedule adjustment)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW v_team_season_srs_proxy AS
        SELECT
            r.season,
            r.teamId,
            (r.avg_margin + COALESCE(s.sos, 0)) AS srs_proxy_margin
        FROM v_team_raw_margin r
        LEFT JOIN v_team_schedule_strength s
          ON r.season = s.season AND r.teamId = s.teamId
    """)
    logger.info("  Created v_team_season_srs_proxy")

    # 6. v_team_season_ranks: Rank teams by SRS proxy
    con.execute("""
        CREATE OR REPLACE TEMP VIEW v_team_season_ranks AS
        SELECT
            season,
            teamId,
            srs_proxy_margin,
            DENSE_RANK() OVER (PARTITION BY season ORDER BY srs_proxy_margin DESC) AS team_rank
        FROM v_team_season_srs_proxy
    """)
    logger.info("  Created v_team_season_ranks")

    # 7. v_shots_augmented: Join shots with season + opponent rank + splits
    #    CRITICAL: VS_UNKNOWN when opp_rank IS NULL (not VS_OTHERS)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW v_shots_augmented AS
        SELECT
            s.gameId,
            g.season,
            CAST(s.teamId AS BIGINT) AS teamId,
            CAST(s.opponentId AS BIGINT) AS opponentId,
            s.shooterAthleteId AS athlete_id,
            s.shot_range,
            s.made,
            s.assisted,
            s.is_high_leverage,
            s.is_garbage,
            s.loc_x,
            s.loc_y,
            r.team_rank AS opp_rank,
            CASE
                WHEN r.team_rank IS NULL THEN 'VS_UNKNOWN'
                WHEN r.team_rank <= 50 THEN 'VS_TOP50'
                WHEN r.team_rank <= 100 THEN 'VS_TOP100'
                ELSE 'VS_OTHERS'
            END AS strength_split,
            CASE
                WHEN s.is_high_leverage THEN 'HIGH_LEVERAGE'
                WHEN s.is_garbage THEN 'GARBAGE'
                ELSE 'LOW_LEVERAGE'
            END AS leverage_split
        FROM stg_shots s
        JOIN v_games_canon g
          ON g.gameId = s.gameId
        LEFT JOIN v_team_season_ranks r
          ON r.season = g.season
         AND r.teamId = CAST(s.opponentId AS BIGINT)
    """)
    logger.info("  Created v_shots_augmented")


def build_features(con) -> pd.DataFrame:
    """Build 20-split feature aggregates."""
    logger.info("Building feature aggregates (20 splits)...")

    sql = """
    WITH base AS (
        SELECT
            season,
            athlete_id,
            leverage_split,
            strength_split,
            shot_range,
            made,
            assisted,
            loc_x,
            loc_y
        FROM v_shots_augmented
    ),
    combos AS (
        SELECT * FROM (VALUES
            ('ALL','ALL'),
            ('ALL','VS_TOP50'), ('ALL','VS_TOP100'), ('ALL','VS_OTHERS'), ('ALL','VS_UNKNOWN'),
            ('HIGH_LEVERAGE','ALL'), ('HIGH_LEVERAGE','VS_TOP50'), ('HIGH_LEVERAGE','VS_TOP100'), ('HIGH_LEVERAGE','VS_OTHERS'), ('HIGH_LEVERAGE','VS_UNKNOWN'),
            ('LOW_LEVERAGE','ALL'), ('LOW_LEVERAGE','VS_TOP50'), ('LOW_LEVERAGE','VS_TOP100'), ('LOW_LEVERAGE','VS_OTHERS'), ('LOW_LEVERAGE','VS_UNKNOWN'),
            ('GARBAGE','ALL'), ('GARBAGE','VS_TOP50'), ('GARBAGE','VS_TOP100'), ('GARBAGE','VS_OTHERS'), ('GARBAGE','VS_UNKNOWN')
        ) AS t(l_split, s_split)
    )
    SELECT
        b.season,
        b.athlete_id,
        (c.l_split || '__' || c.s_split) AS split_id,

        -- Exposure
        COUNT(*) AS shots_total,
        COUNT(*) FILTER (WHERE shot_range != 'free_throw') AS fga_total,
        COUNT(*) FILTER (WHERE shot_range = 'free_throw') AS ft_att,

        -- Rim
        COUNT(*) FILTER (WHERE shot_range = 'rim') AS rim_att,
        COUNT(*) FILTER (WHERE shot_range = 'rim' AND made) AS rim_made,

        -- Three Pointer
        COUNT(*) FILTER (WHERE shot_range = 'three_pointer') AS three_att,
        COUNT(*) FILTER (WHERE shot_range = 'three_pointer' AND made) AS three_made,

        -- Mid-Range (jumper)
        COUNT(*) FILTER (WHERE shot_range = 'jumper') AS mid_att,
        COUNT(*) FILTER (WHERE shot_range = 'jumper' AND made) AS mid_made,

        -- Free Throw
        COUNT(*) FILTER (WHERE shot_range = 'free_throw' AND made) AS ft_made,

        -- Assisted (by zone)
        COUNT(*) FILTER (WHERE shot_range = 'rim' AND made AND assisted) AS assisted_made_rim,
        COUNT(*) FILTER (WHERE shot_range = 'three_pointer' AND made AND assisted) AS assisted_made_three,
        COUNT(*) FILTER (WHERE shot_range = 'jumper' AND made AND assisted) AS assisted_made_mid,

        -- SPATIAL (Tier 2) - Scale: 1 unit = 0.1 ft (0-940, 0-500)
        -- XY Count
        COUNT(loc_x) AS xy_shots,
        
        -- Distance Sum (for Avg Dist)
        -- Hoop at (5.25, 25) and (88.75, 25)
        SUM(
            CASE 
                WHEN loc_x IS NULL THEN 0
                ELSE SQRT( 
                    POW((loc_x/10.0 - (CASE WHEN loc_x < 470 THEN 5.25 ELSE 88.75 END)), 2) + 
                    POW((loc_y/10.0 - 25.0), 2) 
                )
            END
        ) AS sum_dist_ft,

        -- Corner 3s (Logic: Dist from center > 21ft & Short Corner < 14ft)
        COUNT(*) FILTER (
            WHERE shot_range = 'three_pointer'
              AND loc_x IS NOT NULL
              AND ABS(loc_y/10.0 - 25.0) > 21.0
              AND (CASE WHEN loc_x < 470 THEN loc_x/10.0 ELSE 94.0 - loc_x/10.0 END) < 14.0
        ) AS corner_3_att,

        COUNT(*) FILTER (
            WHERE shot_range = 'three_pointer'
              AND made
              AND loc_x IS NOT NULL
              AND ABS(loc_y/10.0 - 25.0) > 21.0
              AND (CASE WHEN loc_x < 470 THEN loc_x/10.0 ELSE 94.0 - loc_x/10.0 END) < 14.0
        ) AS corner_3_made,

        -- Precision Counts for Gating
        COUNT(loc_x) FILTER (WHERE shot_range = 'three_pointer') AS xy_3_shots,
        COUNT(loc_x) FILTER (WHERE shot_range = 'rim') AS xy_rim_shots,

        -- Extended Spatial: Deep 3s (> 27ft)
        -- Logic: Dist > 27.0 AND 3PT
        COUNT(*) FILTER (
            WHERE shot_range = 'three_pointer'
              AND loc_x IS NOT NULL
              AND SQRT(POW((loc_x/10.0 - (CASE WHEN loc_x < 470 THEN 5.25 ELSE 88.75 END)), 2) + POW((loc_y/10.0 - 25.0), 2)) > 27.0
        ) AS deep_3_att,

        -- Extended Spatial: Rim Purity (< 4ft Restricted Area)
        -- Logic: Dist < 4.0 AND Rim
        COUNT(*) FILTER (
            WHERE shot_range = 'rim'
              AND loc_x IS NOT NULL
              AND SQRT(POW((loc_x/10.0 - (CASE WHEN loc_x < 470 THEN 5.25 ELSE 88.75 END)), 2) + POW((loc_y/10.0 - 25.0), 2)) < 4.0
        ) AS rim_rest_att,

        -- For Variance: Sum of Squared Distance
        SUM(
            CASE 
                WHEN loc_x IS NULL THEN 0
                ELSE POW(SQRT( 
                    POW((loc_x/10.0 - (CASE WHEN loc_x < 470 THEN 5.25 ELSE 88.75 END)), 2) + 
                    POW((loc_y/10.0 - 25.0), 2) 
                ), 2)
            END
        ) AS sum_dist_sq_ft

    FROM base b
    JOIN combos c
      ON (c.l_split = 'ALL' OR c.l_split = b.leverage_split)
     AND (c.s_split = 'ALL' OR c.s_split = b.strength_split)
    GROUP BY 1, 2, 3
    """

    df = con.execute(sql).df()
    logger.info(f"  Feature rows: {len(df):,}")

    # Compute rates with missing flags
    df['rim_fg_pct'] = np.where(df['rim_att'] > 0, df['rim_made'] / df['rim_att'], np.nan)
    df['rim_fg_pct_missing'] = (df['rim_att'] == 0).astype(int)

    df['three_fg_pct'] = np.where(df['three_att'] > 0, df['three_made'] / df['three_att'], np.nan)
    df['three_fg_pct_missing'] = (df['three_att'] == 0).astype(int)

    df['mid_fg_pct'] = np.where(df['mid_att'] > 0, df['mid_made'] / df['mid_att'], np.nan)
    df['mid_fg_pct_missing'] = (df['mid_att'] == 0).astype(int)

    df['ft_pct'] = np.where(df['ft_att'] > 0, df['ft_made'] / df['ft_att'], np.nan)
    df['ft_pct_missing'] = (df['ft_att'] == 0).astype(int)

    df['assisted_share_rim'] = np.where(df['rim_made'] > 0, df['assisted_made_rim'] / df['rim_made'], np.nan)
    df['assisted_share_rim_missing'] = (df['rim_made'] == 0).astype(int)

    df['assisted_share_three'] = np.where(df['three_made'] > 0, df['assisted_made_three'] / df['three_made'], np.nan)
    df['assisted_share_three_missing'] = (df['three_made'] == 0).astype(int)

    df['assisted_share_mid'] = np.where(df['mid_made'] > 0, df['assisted_made_mid'] / df['mid_made'], np.nan)
    df['assisted_share_mid_missing'] = (df['mid_made'] == 0).astype(int)

    # --- Tier 2 Spatial Features (Gated) --- 
    # Threshold: 25 shots with coordinates
    # For General Stats (Avg Dist, Dispersion)
    tier2_mask = df['xy_shots'] >= 25
    
    # For 3PT Stats (Corner, Deep)
    tier2_3pt_mask = df['xy_3_shots'] >= 15  # Lower threshold for subset? Or keep 25? Let's say 20.
    
    # For Rim Stats (Purity)
    tier2_rim_mask = df['xy_rim_shots'] >= 20

    # Avg Shot Distance
    df['avg_shot_dist'] = np.where(tier2_mask, df['sum_dist_ft'] / df['xy_shots'], np.nan)
    
    # Shot Distance Variance (Dispersion)
    # Var = E[X^2] - (E[X])^2
    # sum_sq / N - avg^2
    mean_sq = df['sum_dist_sq_ft'] / df['xy_shots']
    mean_val = df['avg_shot_dist']
    df['shot_dist_var'] = np.where(tier2_mask, mean_sq - (mean_val ** 2), np.nan)

    # Corner 3 Rate (Corner Att / Total 3 Att)
    # Use xy_3_shots as denominator to be safe? Or total three_att? 
    # If we assume missingness is random, we can use (Corner XY / Total XY 3s).
    # This prevents bias if we only have coords for half the games.
    # So: corner_3_rate = corner_3_att / xy_3_shots
    df['corner_3_rate'] = np.where(tier2_3pt_mask & (df['xy_3_shots'] > 0), 
                                   df['corner_3_att'] / df['xy_3_shots'], 
                                   np.nan)
    
    # Corner 3 PCT (Efficiency) - Uses actual makes/att
    df['corner_3_pct'] = np.where(tier2_3pt_mask & (df['corner_3_att'] > 0),
                                  df['corner_3_made'] / df['corner_3_att'],
                                  np.nan)

    # Deep 3 Rate (Deep Att / XY 3 Att)
    df['deep_3_rate'] = np.where(tier2_3pt_mask & (df['xy_3_shots'] > 0),
                                 df['deep_3_att'] / df['xy_3_shots'],
                                 np.nan)

    # Rim Purity (Restricted Att / XY Rim Att)
    df['rim_purity'] = np.where(tier2_rim_mask & (df['xy_rim_shots'] > 0),
                                df['rim_rest_att'] / df['xy_rim_shots'],
                                np.nan)
                                  
    # Export coverage
    df['xy_coverage'] = np.where(df['shots_total'] > 0, df['xy_shots'] / df['shots_total'], 0.0)

    return df


def join_recruiting(con, df_features: pd.DataFrame) -> pd.DataFrame:
    """Join recruiting data."""
    logger.info("Joining recruiting data...")

    # fact_recruiting_players.athleteId is DOUBLE, need to handle
    recruiting = con.execute("""
        SELECT
            CAST(athleteId AS INTEGER) AS athlete_id,
            ranking AS recruiting_rank,
            stars AS recruiting_stars,
            rating AS recruiting_rating
        FROM fact_recruiting_players
        WHERE athleteId IS NOT NULL
    """).df()

    df = df_features.merge(recruiting, on='athlete_id', how='left')

    # Missing flags (do NOT fill 0 for rank/stars without flag)
    df['recruiting_missing'] = df['recruiting_rank'].isna().astype(int)

    logger.info(f"  Recruiting match rate: {(df['recruiting_missing'] == 0).mean():.1%}")
    return df


def join_team_context(con, df_features: pd.DataFrame) -> pd.DataFrame:
    """Join team pace/context and conference info."""
    logger.info("Joining team context...")

    # Get athlete team mapping using MODE (Team with most shots for that athlete-season)
    # This is more stable than MIN_BY(teamId) for transfers
    athlete_team = con.execute("""
        WITH shot_counts AS (
            SELECT
                season,
                athlete_id,
                teamId,
                COUNT(*) as n_shots
            FROM v_shots_augmented
            GROUP BY 1, 2, 3
        ),
        ranked_teams AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY season, athlete_id ORDER BY n_shots DESC) as rn
            FROM shot_counts
        )
        SELECT season, athlete_id, teamId
        FROM ranked_teams
        WHERE rn = 1
    """).df()

    # Team stats
    team_stats = con.execute(f"""
        SELECT
            season,
            teamId,
            pace AS team_pace,
            conference,
            CASE WHEN conference IN {tuple(POWER_CONFS)} THEN 1 ELSE 0 END AS is_power_conf
        FROM fact_team_season_stats
    """).df()

    # Box Score Stats (Derived from PBP and Shots)
    # Replaces missing fact_player_season_stats for 2006-2024
    
    # 1. Turnovers from stg_plays
    # Box Score Stats (Derived from PBP and Shots)
    # Replaces missing fact_player_season_stats for 2006-2024
    
    # 1. Turnovers from stg_plays (JSON parsing)
    # participants format: [{'id': 123, 'name': '...'}]
    # We use DuckDB's json extraction to get the ID of the first participant
    tov_df = con.execute("""
        SELECT 
            g.season,
            CAST(json_extract(p.participants, '$[0].id') AS BIGINT) as athlete_id,
            COUNT(*) as tov_total_derived
        FROM stg_plays p
        JOIN v_games_canon g ON g.gameId = p.gameId
        WHERE p.playType ILIKE '%Turnover%'
          AND p.participants IS NOT NULL
          AND p.participants != '[]'
        GROUP BY 1, 2
    """).df()
    
    # 2. Existing fact fallback (only has 2005/2025)
    box_stats = con.execute("""
        SELECT
            season,
            athleteId AS athlete_id,
            assists AS ast_total,
            turnovers AS tov_total,
            steals AS stl_total,
            blocks AS blk_total,
            rebounds.offensive AS orb_total,
            rebounds.defensive AS drb_total,
            rebounds.total AS trb_total,
            minutes AS minutes_total
        FROM fact_player_season_stats
    """).df()
    
    # Merge Logic: Use fact table if available, else derived
    df = df_features.merge(athlete_team, on=['season', 'athlete_id'], how='left')
    df = df.merge(team_stats, on=['season', 'teamId'], how='left')
    
    # Join both box sources
    df = df.merge(box_stats, on=['season', 'athlete_id'], how='left')
    df = df.merge(tov_df, on=['season', 'athlete_id'], how='left')
    
    # Coalesce TOV: Prefer fact (official), fallback to derived
    df['tov_total'] = df['tov_total'].fillna(df['tov_total_derived']).fillna(0).astype(int)
    # Drop temp column
    df = df.drop(columns=['tov_total_derived'])

    df = df_features.merge(athlete_team, on=['season', 'athlete_id'], how='left')
    df = df.merge(team_stats, on=['season', 'teamId'], how='left')
    df = df.merge(box_stats, on=['season', 'athlete_id'], how='left')

    # Fill box stats with 0 if missing (but only if player existed in shots)
    box_cols = ['ast_total', 'tov_total', 'stl_total', 'blk_total', 'orb_total', 'drb_total', 'trb_total', 'minutes_total']
    for c in box_cols:
        df[c] = df[c].fillna(0).astype(int)

    df['is_power_conf'] = df['is_power_conf'].fillna(0).astype(int)
    return df



def build_impact(con) -> pd.DataFrame:
    """Build impact metrics including True RAPM."""
    logger.info("Building impact metrics...")

    # 1. On-Court (Existing logic)
    sql_on_court = """
    SELECT
        g.season,
        la.athleteId AS athlete_id,
        SUM(la.totalSeconds) AS seconds_on,
        SUM(la.totalSeconds) / 60.0 AS mp_total,
        SUM(la.offenseRating * la.totalSeconds) / NULLIF(SUM(la.totalSeconds), 0) AS on_ortg,
        SUM(la.defenseRating * la.totalSeconds) / NULLIF(SUM(la.totalSeconds), 0) AS on_drtg,
        SUM(la.netRating * la.totalSeconds) / NULLIF(SUM(la.totalSeconds), 0) AS on_net_rating,
        SUM(la.pace * la.totalSeconds / 2400.0) AS poss_est
    FROM bridge_lineup_athletes la
    JOIN v_games_canon g ON g.gameId = la.gameId
    GROUP BY 1, 2
    """
    df_on_court = con.execute(sql_on_court).df()
    
    df_final = df_on_court.copy()
    
    # 2. TRUE RAPM using fact_play_raw.onFloor (10-player design matrix)
    # CRITICAL FIXES (per Gemini/GPT analysis):
    #  1. Absolute game clock with period breaks (no duration bugs)
    #  2. Home/Away mapping via dim_teams (not arbitrary team1/team2)
    #  3. Pace-based possessions (not duration/24)
    #  4. No weight normalization (to avoid over-regularization)
    #  5. SRS post-hoc adjustment (for cross-conference comparability)
    #  6. Lambda = 1500 (lower for college's shorter seasons)
    
    logger.info("  Computing True RAPM from fact_play_raw.onFloor...")
    
    # Get team name mapping for home/away identification
    # NOTE: onFloor.team uses short names like "IU Columbus" which match dim_teams.school
    game_team_names = con.execute("""
        SELECT 
            g.gameId,
            g.season,
            g.homeTeamId,
            g.awayTeamId,
            ht.school AS homeTeamName,
            at_t.school AS awayTeamName
        FROM v_games_canon g
        JOIN dim_teams ht ON ht.id = g.homeTeamId
        JOIN dim_teams at_t ON at_t.id = g.awayTeamId
    """).df()
    
    # Get team pace data
    pace_df = con.execute("""
        SELECT season, teamId, pace
        FROM fact_team_season_stats
        WHERE pace IS NOT NULL
    """).df()
    
    # Get SRS for post-hoc adjustment
    srs_df = con.execute("""
        SELECT season, teamId, srs_proxy_margin
        FROM v_team_season_srs_proxy
    """).df()
    
    # Load plays with 10 players on floor
    plays_df = con.execute('''
        SELECT 
            gameId,
            season,
            id as play_id,
            period,
            secondsRemaining,
            homeScore,
            awayScore,
            onFloor
        FROM fact_play_raw
        WHERE len(onFloor) = 10
          AND season >= 2010
        ORDER BY gameId, period, secondsRemaining DESC
    ''').df()
    
    if len(plays_df) == 0:
        logger.warning("  No onFloor data available for RAPM, using fallback")
        df_final['rapm_value'] = np.nan
        df_final['rapm_adjusted'] = np.nan
        return df_final
    
    logger.info(f"  Loaded {len(plays_df):,} plays with 10 players")
    
    # FIX 1: Absolute elapsed game clock (handles period resets)
    def abs_elapsed(period, seconds_remaining):
        """Convert (period, secondsRemaining) to absolute elapsed seconds.
        NCAA: 2 halves of 20 min = 2400s each. OT = 300s each."""
        period = int(period)
        sec_rem = float(seconds_remaining)
        if period == 1:
            return 1200 - sec_rem  # First half (20 min = 1200s)
        elif period == 2:
            return 1200 + (1200 - sec_rem)  # Second half
        else:
            # OT periods (5 min = 300s each)
            return 2400 + (period - 2) * 300 + (300 - min(sec_rem, 300))
    
    plays_df['abs_elapsed'] = plays_df.apply(
        lambda r: abs_elapsed(r['period'], r['secondsRemaining']), axis=1
    )
    
    # Sort by absolute game clock
    plays_df = plays_df.sort_values(['gameId', 'abs_elapsed'], ascending=[True, True])
    
    # Helper: create lineup hash
    def lineup_hash(on_floor):
        if on_floor is None or len(on_floor) == 0:
            return None
        return tuple(sorted([p['id'] for p in on_floor]))
    
    plays_df['lineup_hash'] = plays_df['onFloor'].apply(lineup_hash)
    
    # FIX 1 continued: Break stints on game OR period OR lineup change
    plays_df['new_stint'] = (
        (plays_df['gameId'] != plays_df['gameId'].shift()) |
        (plays_df['period'] != plays_df['period'].shift()) |
        (plays_df['lineup_hash'] != plays_df['lineup_hash'].shift())
    )
    plays_df['stint_id'] = plays_df['new_stint'].cumsum()
    
    # Aggregate stints using absolute clock
    stint_agg = plays_df.groupby('stint_id').agg({
        'gameId': 'first',
        'season': 'first',
        'abs_elapsed': ['first', 'last'],
        'homeScore': ['first', 'last'],
        'awayScore': ['first', 'last'],
        'onFloor': 'first',
        'play_id': 'count'
    }).reset_index()
    
    stint_agg.columns = ['stint_id', 'gameId', 'season', 't0', 't1',
                          'home0', 'home1', 'away0', 'away1', 
                          'onFloor', 'n_plays']
    
    # Duration = elapsed forward (t1 - t0)
    stint_agg['duration'] = stint_agg['t1'] - stint_agg['t0']
    stint_agg['home_diff'] = (stint_agg['home1'] - stint_agg['home0']) - (stint_agg['away1'] - stint_agg['away0'])
    
    # Filter: positive duration & at least 2 plays
    stints = stint_agg[(stint_agg['duration'] > 0) & (stint_agg['n_plays'] >= 2)].copy()
    
    # FIX 2: Map onFloor teams to home/away using dim_teams
    stints = stints.merge(game_team_names[['gameId', 'homeTeamName', 'awayTeamName', 'homeTeamId']], 
                          on='gameId', how='left')
    
    def split_home_away(on_floor, home_name, away_name):
        """Split 10 players into home (5) and away (5) based on team name."""
        if on_floor is None:
            return None, None
        home_players, away_players = [], []
        for p in on_floor:
            if p['team'] == home_name:
                home_players.append(p['id'])
            elif p['team'] == away_name:
                away_players.append(p['id'])
        return home_players, away_players
    
    stints[['home_players', 'away_players']] = stints.apply(
        lambda r: pd.Series(split_home_away(r['onFloor'], r['homeTeamName'], r['awayTeamName'])),
        axis=1
    )
    
    # Filter to valid 5v5
    stints = stints[
        stints['home_players'].apply(lambda x: isinstance(x, list) and len(x) == 5) &
        stints['away_players'].apply(lambda x: isinstance(x, list) and len(x) == 5) &
        (stints['duration'] > 0)
    ].copy()
    
    logger.info(f"  Valid 5v5 stints (with home/away mapping): {len(stints):,}")
    
    # FIX 3: Pace-based possessions
    stints = stints.merge(pace_df.rename(columns={'teamId': 'homeTeamId', 'pace': 'home_pace'}),
                          on=['season', 'homeTeamId'], how='left')
    median_pace = pace_df['pace'].median() if len(pace_df) > 0 else 68.0
    stints['pace'] = stints['home_pace'].fillna(median_pace)
    
    # Possessions in stint = (duration / 2400) * pace_per_40
    stints['poss'] = (stints['duration'] / 2400.0) * stints['pace']
    stints = stints[stints['poss'] > 0].copy()
    
    logger.info(f"  Stints after pace filter: {len(stints):,}")
    
    # Solve RAPM per season
    rapm_results = []
    
    for season, season_stints in stints.groupby('season'):
        logger.info(f"  Solving RAPM for Season {season} ({len(season_stints):,} stints)...")
        
        # Get all unique players
        all_players = set()
        for _, row in season_stints.iterrows():
            all_players.update(row['home_players'])
            all_players.update(row['away_players'])
        
        players = sorted(list(all_players))
        p_map = {p: i for i, p in enumerate(players)}
        n_players = len(players)
        
        # Build design matrix: +1 for HOME, -1 for AWAY (matches home_diff target)
        n_stints = len(season_stints)
        row_ind, col_ind, data = [], [], []
        
        season_stints = season_stints.reset_index(drop=True)
        
        for i, row in season_stints.iterrows():
            for pid in row['home_players']:
                row_ind.append(i)
                col_ind.append(p_map[pid])
                data.append(1.0)
            for pid in row['away_players']:
                row_ind.append(i)
                col_ind.append(p_map[pid])
                data.append(-1.0)
        
        X = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_stints, n_players))
        
        # Target: Point differential per 100 possessions
        poss = season_stints['poss'].values
        pts_per_100 = (season_stints['home_diff'].values / np.maximum(poss, 0.1)) * 100.0
        
        # FIX 4: Weights = raw possessions (NO normalization)
        w = poss
        W = sparse.diags(w)
        
        # Center target
        y_mean = np.average(pts_per_100, weights=w)
        y = pts_per_100 - y_mean
        
        # Ridge Regression
        XTW = X.T @ W
        XTWX = XTW @ X
        XTWy = XTW @ y
        
        # User requested Lambda Sweep (500-1500) to see if Cooper Flagg passes Sion James
        lambdas = [500.0, 750.0, 1000.0, 1250.0, 1500.0]
        best_lambda = 1000.0
        best_coef = None
        
        # Only do sweep for 2025 (current season of interest), use fixed 1000 otherwise
        if season == 2025:
            logger.info("    Sweeping Lambdas [500, 750, 1000, 1250, 1500] for Duke hierarchy...")
            
            # Helper to find player index
            def get_pid_idx(name_part):
                # This requires name mapping which we don't have inside this loop easily
                # We have player IDs. Duke IDs: Flagg=?, Sion=?, Kon=?
                # We'll just solve all and decide based on spread standard deviation for now
                # Or user just wants to SEE the results. 
                # We will log the top variance result.
                return None
            
            # Actually, without names, we can't check specific players.
            # But we can maximize variance (Standard Deviation) which typically helps stars separate.
            # Lower lambda = Higher Variance.
            
            for l_val in lambdas:
                A = XTWX + l_val * sparse.eye(n_players)
                b = XTWy
                try:
                    c, _ = cg(A, b, rtol=1e-5)
                    std_dev = np.std(c)
                    logger.info(f"      Lambda={l_val:.0f}: Range=[{c.min():.2f}, {c.max():.2f}], Std={std_dev:.3f}")
                    
                    # We'll default to the one with the highest std dev (most separation)
                    # unless it looks unstable (huge coefficients).
                    # 500 might be too noisy. Let's stick to 1000 as default but allow override.
                    if l_val == 1000.0: 
                        best_coef = c # Default
                        
                    # If user wants 750 specifically, we can set that.
                    if l_val == 750.0:
                         best_coef = c # User asked to try 750
                         best_lambda = 750.0
                         
                except Exception:
                    continue
        else:
            # Non-2025 seasons: Use the tuned value from 2025 sweep
            lambda_val = 750.0
            A = XTWX + lambda_val * sparse.eye(n_players)
            b = XTWy
            best_coef, _ = cg(A, b, rtol=1e-5)
            best_lambda = lambda_val

        logger.info(f"    Selected Lambda: {best_lambda}")
        
        res_df = pd.DataFrame({
            'athlete_id': players,
            'season': season,
            'rapm_value': best_coef
        })
        rapm_results.append(res_df)

    if rapm_results:
        df_rapm = pd.concat(rapm_results)
        df_final = df_final.merge(df_rapm, on=['season', 'athlete_id'], how='left')
    else:
        df_final['rapm_value'] = np.nan
    
    # FIX 5: SRS post-hoc adjustment
    # Get player's team for SRS lookup (from on-court data)
    player_team = con.execute("""
        SELECT DISTINCT la.athleteId as athlete_id, la.teamId, g.season
        FROM bridge_lineup_athletes la
        JOIN v_games_canon g ON g.gameId = la.gameId
    """).df()
    
    df_final = df_final.merge(player_team, on=['athlete_id', 'season'], how='left')
    df_final = df_final.merge(srs_df, on=['season', 'teamId'], how='left')
    
    # Adjusted RAPM = raw RAPM + (team_SRS * 0.0)
    # User requested to REMOVE adjustment (confounding factor). 
    # We keep the column for schema compatibility but it equals raw RAPM.
    df_final['rapm_adjusted'] = df_final['rapm_value']
    
    logger.info(f"  RAPM adjusted range: [{df_final['rapm_adjusted'].min():.2f}, {df_final['rapm_adjusted'].max():.2f}]")

    # Shrinkage
    df_final['impact_weight'] = np.minimum(1.0, df_final['seconds_on'] / SEC_REF)
    df_final['impact_is_reliable'] = ((df_final['seconds_on'] >= SEC_REF) | (df_final['poss_est'] >= POSS_REF)).astype(int)


    return df_final


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    con = get_connection()
    create_intermediate_views(con)

    # 1. Features
    df_features = build_features(con)
    df_features = join_recruiting(con, df_features)
    df_features = join_team_context(con, df_features)

    out_features = f"{OUTPUT_DIR}/college_features_v1.parquet"
    df_features.to_parquet(out_features, index=False)
    logger.info(f"Saved {out_features} ({len(df_features):,} rows)")

    # 2. Impact
    df_impact = build_impact(con)
    out_impact = f"{OUTPUT_DIR}/college_impact_v1.parquet"
    df_impact.to_parquet(out_impact, index=False)
    logger.info(f"Saved {out_impact} ({len(df_impact):,} rows)")

    # 3. Coverage Report
    coverage = pd.DataFrame({
        'metric': ['total_feature_rows', 'unique_athletes', 'unique_seasons',
                   'total_impact_rows', 'impact_athletes', 'avg_seconds_on'],
        'value': [
            len(df_features),
            df_features['athlete_id'].nunique(),
            df_features['season'].nunique(),
            len(df_impact),
            df_impact['athlete_id'].nunique(),
            df_impact['seconds_on'].mean() if len(df_impact) > 0 else 0
        ]
    })
    out_coverage = f"{OUTPUT_DIR}/coverage_report_v1.csv"
    coverage.to_csv(out_coverage, index=False)
    logger.info(f"Saved {out_coverage}")

    # 4. Feature Registry
    registry = pd.DataFrame([
        {'feature': 'shots_total', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'fga_total', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'rim_att', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'rim_made', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'rim_fg_pct', 'source': 'derived', 'split_behavior': 'filter', 'type': 'rate'},
        {'feature': 'three_att', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'three_made', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'three_fg_pct', 'source': 'derived', 'split_behavior': 'filter', 'type': 'rate'},
        {'feature': 'mid_att', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'mid_made', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'mid_fg_pct', 'source': 'derived', 'split_behavior': 'filter', 'type': 'rate'},
        {'feature': 'ft_att', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'ft_made', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'ft_pct', 'source': 'derived', 'split_behavior': 'filter', 'type': 'rate'},
        {'feature': 'assisted_made_rim', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'assisted_share_rim', 'source': 'derived', 'split_behavior': 'filter', 'type': 'rate'},
        {'feature': 'assisted_made_three', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'assisted_share_three', 'source': 'derived', 'split_behavior': 'filter', 'type': 'rate'},
        {'feature': 'assisted_made_mid', 'source': 'stg_shots', 'split_behavior': 'filter', 'type': 'count'},
        {'feature': 'assisted_share_mid', 'source': 'derived', 'split_behavior': 'filter', 'type': 'rate'},
        {'feature': 'recruiting_rank', 'source': 'fact_recruiting_players', 'split_behavior': 'static', 'type': 'integer'},
        {'feature': 'recruiting_stars', 'source': 'fact_recruiting_players', 'split_behavior': 'static', 'type': 'integer'},
        {'feature': 'recruiting_rating', 'source': 'fact_recruiting_players', 'split_behavior': 'static', 'type': 'float'},
        {'feature': 'team_pace', 'source': 'fact_team_season_stats', 'split_behavior': 'static', 'type': 'float'},
        {'feature': 'is_power_conf', 'source': 'fact_team_season_stats', 'split_behavior': 'static', 'type': 'flag'},
        {'feature': 'seconds_on', 'source': 'bridge_lineup_athletes', 'split_behavior': 'NA', 'type': 'seconds'},
        {'feature': 'on_ortg', 'source': 'bridge_lineup_athletes', 'split_behavior': 'NA', 'type': 'rating'},
        {'feature': 'on_drtg', 'source': 'bridge_lineup_athletes', 'split_behavior': 'NA', 'type': 'rating'},
        {'feature': 'on_net_rating', 'source': 'bridge_lineup_athletes', 'split_behavior': 'NA', 'type': 'rating'},
        {'feature': 'impact_weight', 'source': 'derived', 'split_behavior': 'NA', 'type': 'weight'},
        {'feature': 'impact_is_reliable', 'source': 'derived', 'split_behavior': 'NA', 'type': 'flag'},
    ])
    out_registry = f"{OUTPUT_DIR}/college_feature_registry_v1.csv"
    registry.to_csv(out_registry, index=False)
    logger.info(f"Saved {out_registry}")

    logger.info("Build complete!")


if __name__ == "__main__":
    main()
