import pandas as pd
import numpy as np
import json
import os
import scipy.sparse as sparse
from scipy.sparse.linalg import cg
import logging
from typing import Tuple, Dict, List, Optional

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_PARQUET = 'data/fact_play_historical_combined.parquet'
OUTPUT_CSV = 'data/historical_rapm_results_enhanced.csv'
LAMBDA = 1000.0

# Leverage thresholds (based on pbpstats research)
LEVERAGE_THRESHOLDS = {
    'garbage': 0.05,      # Below this = garbage time
    'low': 0.10,          # Low leverage
    'medium': 0.20,       # Medium leverage  
    'high': 0.35,         # High leverage
    'very_high': 0.50     # Very high leverage (crunch time)
}

def parse_clock(clock_str: str) -> int:
    """Parse MM:SS clock string to seconds remaining."""
    try:
        parts = clock_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except:
        return 0


def compute_win_probability(margin: int, seconds_remaining: int, 
                            total_game_seconds: int = 2400) -> float:
    """
    Estimate win probability for home team using logistic model.
    
    Based on empirical NBA/NCAA research:
    - At game start (t=2400s), margin has minimal impact
    - At game end (t=0s), margin is decisive
    - Uses time-weighted logistic function
    
    Args:
        margin: Home team lead (positive = home winning)
        seconds_remaining: Seconds left in game
        total_game_seconds: Total game length (2400 for 40-min college)
    
    Returns:
        Win probability for home team [0, 1]
    """
    if seconds_remaining <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    
    # Time factor: how much margin matters (0 at start, 1 at end)
    time_factor = 1.0 - (seconds_remaining / total_game_seconds)
    time_factor = max(0.0, min(1.0, time_factor))  # Clamp to [0, 1]
    
    # Effective margin scales with time remaining
    # At game start, need huge lead to matter; at end, small lead decisive
    # Empirical: ~0.15 points per second of game time for significance
    effective_margin = margin * (1.0 + 2.0 * time_factor)
    
    # Logistic function with empirically-tuned scale
    # Scale of ~4 means 8-point lead at halftime ≈ 75% win prob
    scale = 4.0 + 6.0 * time_factor  # More decisive late
    wp = 1.0 / (1.0 + np.exp(-effective_margin / scale))
    
    return wp


def compute_leverage_index(margin: int, seconds_remaining: int,
                           total_game_seconds: int = 2400) -> float:
    """
    Compute leverage index for a game state.
    
    Leverage = expected win probability swing from possession outcome.
    Based on pbpstats.com methodology:
    - Calculate WP change for each possible outcome (0-4 pts)
    - Weight by outcome frequency
    - Take sqrt of weighted sum of squared changes
    
    Args:
        margin: Home team lead
        seconds_remaining: Seconds remaining
        total_game_seconds: Total game length
    
    Returns:
        Leverage index [0, 1] where higher = more impactful
    """
    if seconds_remaining <= 0:
        return 0.0
    
    # Current win probability
    wp_current = compute_win_probability(margin, seconds_remaining, total_game_seconds)
    
    # Assume possession takes ~18 seconds on average
    time_after = max(0, seconds_remaining - 18)
    
    # Possible outcomes and their frequencies (empirical from NCAA data)
    outcomes = [
        (0, 0.51),   # No score: 51%
        (1, 0.09),   # 1 point (FT): 9%
        (2, 0.28),   # 2 points: 28%
        (3, 0.11),   # 3 points: 11%
        (4, 0.01),   # 4 points (and-1 or 2 FTs): 1%
    ]
    
    # Calculate weighted sum of squared WP changes
    weighted_sq_sum = 0.0
    for pts, freq in outcomes:
        # WP after scoring pts (from home perspective, assume home has ball)
        wp_after = compute_win_probability(margin + pts, time_after, total_game_seconds)
        wp_change = wp_after - wp_current
        weighted_sq_sum += freq * (wp_change ** 2)
    
    leverage = np.sqrt(weighted_sq_sum)
    
    # Normalize to [0, 1] range (max theoretical leverage ~0.5)
    leverage_normalized = min(1.0, leverage / 0.5)
    
    return leverage_normalized


def classify_leverage(leverage: float) -> str:
    """
    Classify leverage into buckets for filtering/analysis.
    
    Returns: 'garbage', 'low', 'medium', 'high', or 'very_high'
    """
    if leverage < LEVERAGE_THRESHOLDS['garbage']:
        return 'garbage'
    elif leverage < LEVERAGE_THRESHOLDS['low']:
        return 'low'
    elif leverage < LEVERAGE_THRESHOLDS['medium']:
        return 'medium'
    elif leverage < LEVERAGE_THRESHOLDS['high']:
        return 'high'
    else:
        return 'very_high'


def compute_rubber_band_adjustment(margin_at_start: int, 
                                   expected_regression_rate: float = 0.02) -> float:
    """
    Compute expected margin regression for rubber band effect.
    
    Teams ahead tend to coast; teams behind try harder.
    This creates systematic bias in raw +/- that we can adjust for.
    
    Args:
        margin_at_start: Score differential at stint start
        expected_regression_rate: Points of regression per point of lead per minute
    
    Returns:
        Expected margin change due to rubber band effect (negative when ahead)
    """
    return -expected_regression_rate * margin_at_start

def solve_rapm(season_stints: pd.DataFrame, lambda_val: float,
               weight_col: str = 'poss',
               use_leverage_weights: bool = False,
               use_rubber_band: bool = False) -> pd.DataFrame:
    """
    Solve RAPM using ridge regression with configurable weighting.
    
    Args:
        season_stints: DataFrame with stint data
        lambda_val: Ridge regularization parameter
        weight_col: Column to use for base weights ('poss' or 'leverage_weight')
        use_leverage_weights: If True, multiply weights by leverage index
        use_rubber_band: If True, adjust target for rubber band effect
    
    Returns:
        DataFrame with player RAPM values
    """
    all_players = set()
    for _, row in season_stints.iterrows():
        all_players.update(row['home_players'])
        all_players.update(row['away_players'])
    
    players = sorted(list(all_players))
    p_map = {p: i for i, p in enumerate(players)}
    n_players = len(players)
    
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
    margin_diff = season_stints['margin_diff'].values.copy()
    
    # Apply rubber band adjustment if requested
    if use_rubber_band and 'margin_start' in season_stints.columns:
        rubber_band_adj = season_stints['margin_start'].apply(
            lambda m: compute_rubber_band_adjustment(m)
        ).values
        # Adjust for stint duration (in minutes)
        duration_mins = season_stints['duration'].values / 60.0
        margin_diff = margin_diff - (rubber_band_adj * duration_mins)
    
    pts_per_100 = (margin_diff / np.maximum(poss, 0.1)) * 100.0
    
    # Compute weights
    w = poss.copy()
    if use_leverage_weights and 'leverage' in season_stints.columns:
        # Multiply by leverage (higher leverage = more weight)
        leverage = season_stints['leverage'].values
        w = w * (0.5 + leverage)  # Floor of 0.5 to not completely ignore low-leverage
    
    W = sparse.diags(w)
    
    # Center target
    y_mean = np.average(pts_per_100, weights=w)
    y = pts_per_100 - y_mean
    
    # Normal Equations: (X'WX + lambda*I)c = X'Wy
    XTW = X.T @ W
    XTWX = XTW @ X
    XTWy = XTW @ y
    
    A = XTWX + lambda_val * sparse.eye(n_players)
    b = XTWy
    
    coef, _ = cg(A, b, rtol=1e-5)
    
    res = pd.DataFrame({
        'player_name': players,
        'rapm': coef,
        'poss_total': np.abs(X).T @ w
    })
    return res


def solve_rapm_split(season_stints: pd.DataFrame, lambda_val: float,
                     use_leverage_weights: bool = False) -> pd.DataFrame:
    """
    Solve for Offensive and Defensive RAPM separately.
    
    O-RAPM: Points scored per 100 possessions (home perspective)
    D-RAPM: Points allowed per 100 possessions (home perspective, inverted)
    
    Args:
        season_stints: DataFrame with stint data including pts_scored_home, pts_allowed_home
        lambda_val: Ridge regularization parameter
        use_leverage_weights: If True, weight by leverage index
    
    Returns:
        DataFrame with player O-RAPM, D-RAPM, and Net RAPM
    """
    all_players = set()
    for _, row in season_stints.iterrows():
        all_players.update(row['home_players'])
        all_players.update(row['away_players'])
    
    players = sorted(list(all_players))
    p_map = {p: i for i, p in enumerate(players)}
    n_players = len(players)
    n_stints = len(season_stints)
    
    # Build design matrix (same for O and D)
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
    
    poss = season_stints['poss'].values
    w = poss.copy()
    if use_leverage_weights and 'leverage' in season_stints.columns:
        leverage = season_stints['leverage'].values
        w = w * (0.5 + leverage)
    
    W = sparse.diags(w)
    
    def solve_single_target(target_values):
        pts_per_100 = (target_values / np.maximum(poss, 0.1)) * 100.0
        y_mean = np.average(pts_per_100, weights=w)
        y = pts_per_100 - y_mean
        
        XTW = X.T @ W
        XTWX = XTW @ X
        XTWy = XTW @ y
        
        A = XTWX + lambda_val * sparse.eye(n_players)
        b = XTWy
        
        coef, _ = cg(A, b, rtol=1e-5)
        return coef
    
    # O-RAPM: Points scored by home team in stint
    # For home players: positive = good offense
    # For away players: their offense is home's defense, so sign flips
    if 'pts_scored_home' in season_stints.columns:
        o_coef = solve_single_target(season_stints['pts_scored_home'].values)
        d_coef = solve_single_target(-season_stints['pts_allowed_home'].values)  # Negative = good defense
    else:
        # Fall back to margin-based approximation
        # Assume roughly equal split of margin to O and D
        margin = season_stints['margin_diff'].values
        o_coef = solve_single_target(margin / 2)
        d_coef = solve_single_target(margin / 2)
    
    res = pd.DataFrame({
        'player_name': players,
        'o_rapm': o_coef,
        'd_rapm': d_coef,
        'rapm_net': o_coef + d_coef,
        'poss_total': np.abs(X).T @ w
    })
    return res

def main():
    if not os.path.exists(INPUT_PARQUET):
        logger.error(f"Input file not found: {INPUT_PARQUET}")
        return

    logger.info(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    # 1. Period Detection & Clock Normalization
    logger.info("Parsing clocks and detecting periods...")
    df['seconds_rem'] = df['clock'].apply(parse_clock)
    
    # Logic: If clock increases by > 60s from prev row in same game, it's a new period
    df['clock_diff'] = df.groupby('gameSourceId')['seconds_rem'].diff()
    df['new_period'] = (df['clock_diff'] > 60).fillna(False)
    df['period'] = df.groupby('gameSourceId')['new_period'].cumsum() + 1
    
    def abs_elapsed(period, sec_rem):
        if period == 1: return 1200 - sec_rem
        return 1200 + (period - 1) * 1200 + (1200 - sec_rem) # Simple 20m periods logic
    
    df['abs_time'] = df.apply(lambda r: abs_elapsed(r['period'], r['seconds_rem']), axis=1)
    
    # 2. Stint Building
    logger.info("Detecting stints...")
    # onFloor is a JSON string of a list of dicts. Hash it for changes.
    df['lineup_hash'] = df['onFloor'].apply(lambda x: hash(x))
    
    df['stint_change'] = (
        (df['gameSourceId'] != df['gameSourceId'].shift()) |
        (df['lineup_hash'] != df['lineup_hash'].shift())
    )
    df['stint_id'] = df['stint_change'].cumsum()
    
    # Aggregate Stints
    stint_agg = df.groupby('stint_id').agg({
        'gameSourceId': 'first',
        'season': 'first',
        'abs_time': ['min', 'max'],
        'homeScore': ['first', 'last'],
        'awayScore': ['first', 'last'],
        'onFloor': 'first'
    })
    stint_agg.columns = ['gameId', 'season', 't0', 't1', 'h0', 'h1', 'a0', 'a1', 'onFloor']
    
    # Calculate Stint Stats
    stint_agg['duration'] = stint_agg['t1'] - stint_agg['t0']
    stint_agg['margin_diff'] = (stint_agg['h1'] - stint_agg['h0']) - (stint_agg['a1'] - stint_agg['a0'])
    stint_agg['pts_scored_home'] = stint_agg['h1'] - stint_agg['h0']
    stint_agg['pts_allowed_home'] = stint_agg['a1'] - stint_agg['a0']
    
    # Possessions Estimate: (duration / 2400) * 68 (avg pace)
    stint_agg['poss'] = (stint_agg['duration'] / 2400.0) * 68.0
    
    # Compute margin at stint start (for leverage and rubber band)
    stint_agg['margin_start'] = stint_agg['h0'] - stint_agg['a0']
    
    # Compute seconds remaining at stint start
    # t0 is absolute elapsed time, so seconds_remaining = 2400 - t0
    stint_agg['seconds_remaining'] = 2400 - stint_agg['t0']
    stint_agg['seconds_remaining'] = stint_agg['seconds_remaining'].clip(lower=0)
    
    # Compute leverage index for each stint
    logger.info("Computing leverage indices...")
    stint_agg['leverage'] = stint_agg.apply(
        lambda row: compute_leverage_index(
            int(row['margin_start']), 
            int(row['seconds_remaining'])
        ), axis=1
    )
    stint_agg['leverage_bucket'] = stint_agg['leverage'].apply(classify_leverage)
    
    # Filter valid stints
    stints = stint_agg[stint_agg['duration'] >= 0].copy()
    
    # Parse onFloor into home/away sets
    def get_lineups(on_floor_json):
        data = json.loads(on_floor_json)
        # Use the 'team' labels from the first row of the game if possible, 
        # but the JSON actually already labels them 'HOME' or 'AWAY' or the Team Name.
        # In clean_historical_pbp_v2, the 'team' field stores the raw name.
        # We need a way to distinguish home vs away.
        # Wait, the clean script put self.h_team and self.a_team in there.
        # Let's check a sample.
        return data
        
    logger.info("Partitioning players into Home/Away...")
    # We need to know which team in 'onFloor' is home vs away.
    # The first 'onFloor' in a game has 10 players. 
    # Let's find the home team name from the game header again? 
    # Or just assume the first 5 in the JSON are home? 
    # In clean_historical_pbp_v2:
    # for p in h_set: on_floor.append({'team': self.h_team})
    # for p in a_set: on_floor.append({'team': self.a_team})
    # So the JSON has the team names. 
    
    # To correctly split, we need the home team name for each game.
    # I'll get it from the raw playText of the first row containing "| Score |"
    game_meta = df[df['playText'].str.contains("\| Score \|")].groupby('gameSourceId').first()['playText'].apply(
        lambda x: [p.strip() for p in x.split("|")][1] # Index 1 is Home Team
    ).to_dict()
    
    def split_players(on_floor_json, home_name):
        data = json.loads(on_floor_json)
        h_players = []
        a_players = []
        for p in data:
            if p['team'] == home_name:
                h_players.append(p['name'])
            else:
                a_players.append(p['name'])
        return h_players, a_players

    # Apply split
    # This might be slow on 1.5M rows, but we only have ~100k stints.
    logger.info(f"Processing {len(stints)} stints...")
    
    # Pre-caching home names for speed
    home_names = stint_agg['gameId'].map(game_meta)
    
    h_p_list = []
    a_p_list = []
    for i, row in stints.iterrows():
        hp, ap = split_players(row['onFloor'], home_names[i])
        h_p_list.append(hp)
        a_p_list.append(ap)
    
    stints['home_players'] = h_p_list
    stints['away_players'] = a_p_list
    
    # Log leverage distribution
    logger.info(f"Leverage distribution: {stints['leverage_bucket'].value_counts().to_dict()}")
    
    # 3. Solve per Season with multiple RAPM variants
    all_rapm = []
    unique_seasons = sorted(stints['season'].unique())
    logger.info(f"Target seasons found: {unique_seasons}")
    
    for season in unique_seasons:
        logger.info(f"Solving RApM variants for {season}...")
        season_stints = stints[(stints['season'] == season) & (stints['poss'] > 0.1)]
        if len(season_stints) == 0:
            logger.warning(f"No data for {season}")
            continue
        
        # 1. Standard RAPM (possession-weighted)
        res_standard = solve_rapm(season_stints, LAMBDA, use_leverage_weights=False)
        res_standard = res_standard.rename(columns={'rapm': 'rapm_standard'})
        
        # 2. Leverage-weighted RAPM
        res_leverage = solve_rapm(season_stints, LAMBDA, use_leverage_weights=True)
        res_leverage = res_leverage[['player_name', 'rapm']].rename(columns={'rapm': 'rapm_leverage_weighted'})
        
        # 3. High-leverage only RAPM (filter to high/very_high stints)
        high_lev_stints = season_stints[season_stints['leverage_bucket'].isin(['high', 'very_high'])]
        if len(high_lev_stints) > 100:  # Need minimum sample
            res_high_lev = solve_rapm(high_lev_stints, LAMBDA)
            res_high_lev = res_high_lev[['player_name', 'rapm']].rename(columns={'rapm': 'rapm_high_leverage'})
        else:
            res_high_lev = pd.DataFrame({'player_name': res_standard['player_name'], 'rapm_high_leverage': np.nan})
        
        # 4. Non-garbage RAPM (exclude garbage time)
        non_garbage_stints = season_stints[season_stints['leverage_bucket'] != 'garbage']
        res_non_garbage = solve_rapm(non_garbage_stints, LAMBDA)
        res_non_garbage = res_non_garbage[['player_name', 'rapm']].rename(columns={'rapm': 'rapm_non_garbage'})
        
        # 5. O/D Split RAPM
        res_split = solve_rapm_split(season_stints, LAMBDA, use_leverage_weights=False)
        res_split = res_split[['player_name', 'o_rapm', 'd_rapm']]
        
        # 6. Rubber-band adjusted RAPM
        res_rubber = solve_rapm(season_stints, LAMBDA, use_rubber_band=True)
        res_rubber = res_rubber[['player_name', 'rapm']].rename(columns={'rapm': 'rapm_rubber_adj'})
        
        # Merge all variants
        res = res_standard.merge(res_leverage, on='player_name', how='left')
        res = res.merge(res_high_lev, on='player_name', how='left')
        res = res.merge(res_non_garbage, on='player_name', how='left')
        res = res.merge(res_split, on='player_name', how='left')
        res = res.merge(res_rubber, on='player_name', how='left')
        res['season'] = season
        
        all_rapm.append(res)
        
    if all_rapm:
        final_df = pd.concat(all_rapm)
        final_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Results saved to {OUTPUT_CSV}")
        
        # Summary statistics
        logger.info(f"Total players: {len(final_df)}")
        logger.info(f"Columns: {list(final_df.columns)}")
        
        # Display top 10 for most recent season
        latest_season = final_df['season'].max()
        print(f"\n--- TOP 10 RApM {latest_season} (Standard) ---")
        print(final_df[final_df['season'] == latest_season].sort_values('rapm_standard', ascending=False).head(10)[
            ['player_name', 'rapm_standard', 'rapm_leverage_weighted', 'o_rapm', 'd_rapm', 'poss_total']
        ])
        
        # Correlation analysis between RAPM variants
        print("\n--- RAPM Variant Correlations ---")
        rapm_cols = ['rapm_standard', 'rapm_leverage_weighted', 'rapm_non_garbage', 'rapm_rubber_adj']
        print(final_df[rapm_cols].corr().round(3))
    else:
        logger.error("No results generated.")

if __name__ == "__main__":
    main()
