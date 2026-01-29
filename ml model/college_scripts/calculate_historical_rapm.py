import pandas as pd
import numpy as np
import json
import os
import scipy.sparse as sparse
from scipy.sparse.linalg import cg
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_PARQUET = 'data/fact_play_historical_combined.parquet'
OUTPUT_CSV = 'data/historical_rapm_results_lambda1000.csv'
LAMBDA = 1000.0

def parse_clock(clock_str):
    try:
        parts = clock_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except:
        return 0

def solve_rapm(season_stints, lambda_val):
    # Unique players
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
    pts_per_100 = (season_stints['margin_diff'].values / np.maximum(poss, 0.1)) * 100.0
    
    # Weight by sqrt(possessions)
    w = poss
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
        'poss_total': np.abs(X).T @ w # Total possessions player was on floor for
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
    
    # Possessions Estimate: (duration / 2400) * 68 (avg pace)
    stint_agg['poss'] = (stint_agg['duration'] / 2400.0) * 68.0
    
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
    
    # 3. Solve per Season
    all_rapm = []
    for season in [2015, 2017]:
        logger.info(f"Solving RApM for {season}...")
        season_stints = stints[(stints['season'] == season) & (stints['poss'] > 0.1)]
        if len(season_stints) == 0:
            logger.warning(f"No data for {season}")
            continue
        res = solve_rapm(season_stints, LAMBDA)
        res['season'] = season
        all_rapm.append(res)
        
    if all_rapm:
        final_df = pd.concat(all_rapm)
        final_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Results saved to {OUTPUT_CSV}")
        
        # Display top 10 for 2015
        print("\n--- TOP 10 RApM 2015 ---")
        print(final_df[final_df['season'] == 2015].sort_values('rapm', ascending=False).head(10))
    else:
        logger.error("No results generated.")

if __name__ == "__main__":
    main()
