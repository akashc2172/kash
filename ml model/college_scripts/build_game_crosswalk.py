"""
Build Game Crosswalk
====================
Matches CBD games (from dim_games) to Manual Scrape games (fact_play_historical_combined)
using Date + Fuzzy Team Matching.

Output:
  data/warehouse.duckdb -> game_crosswalk table
  columns: cbd_game_id, manual_game_id, match_confidence, match_type
"""

import duckdb
import pandas as pd
from rapidfuzz import process, fuzz
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

DB_PATH = "data/warehouse.duckdb"
MANUAL_FILE = "data/fact_play_historical_combined.parquet"

def normalize(s):
    if not isinstance(s, str): return ""
    return s.strip().lower().replace(".","").replace(" st"," state")

def main():
    con = duckdb.connect(DB_PATH)
    
    # 1. Load CBD Games
    logger.info("Loading CBD Games...")
    cbd_games = con.execute("""
        SELECT 
            CAST(id AS VARCHAR) as gameId,
            season,
            STRFTIME(date, '%Y-%m-%d') as date_str,
            homeTeamId,
            awayTeamId,
            homePoints,
            awayPoints
        FROM dim_games
        WHERE date IS NOT NULL
    """).df()
    
    # Get Team Names for CBD
    teams = con.execute("SELECT id, school FROM dim_teams").df().set_index('id')['school'].to_dict()
    
    cbd_games['home_name'] = cbd_games['homeTeamId'].map(teams).fillna('').apply(normalize)
    cbd_games['away_name'] = cbd_games['awayTeamId'].map(teams).fillna('').apply(normalize)
    
    # 2. Load Manual Games
    logger.info("Loading Manual Scrapes...")
    try:
        manual_df = pd.read_parquet(MANUAL_FILE)
    except Exception as e:
        logger.error(f"Could not load {MANUAL_FILE}: {e}")
        return

    # Aggregate to Game Level (manual file is play-level)
    # We need one row per game
    cols = ['gameSourceId', 'season', 'date', 'playText']
    # Use aggregation to extract teams? 
    # The manual cleaner doesn't explicitly save team names in a column?
    # Ah, clean_historical_pbp_v2 saves playText, but onFloor has team names in JSON.
    # We can peek at JSON or parse playText?
    # Actually, cleaner v2 DOES NOT save team names as columns in output. 
    # But it saves `homeScore` / `awayScore`.
    # AND `onFloor` has `team` field.
    # We can extract team names from onFloor of the first row per game.
    
    # Efficient extraction:
    # Group by gameSourceId, take first row's onFloor
    import json
    
    logger.info("Extracting metadata from manual scrapes...")
    
    # Group by game
    manual_games = manual_df.groupby('gameSourceId').first().reset_index()
    
    game_meta = []
    for _, row in manual_games.iterrows():
        try:
            floor = json.loads(row['onFloor'])
            if floor and len(floor) > 0:
                # Naive: Extract two distinct team names
                teams_in_floor = list(set([p['team'] for p in floor]))
                # But we don't know which is home/away easily from onFloor alone 
                # (cleaner preserves home/away in logic but output JSON just says 'team': 'NAME')
                # Wait, cleaner v2 passes `h_team` and `a_team` to GameSolver.
                # But it doesn't write them to output columns?
                # Inspect clean_historical_pbp_v2 again...
                # It writes: gameSourceId, season, date, clock, playText, homeScore, awayScore, onFloor.
                # NO explicit home_team_name / away_team_name column.
                
                # RECOVERY: 
                # 1. We can guess from `playText`? No.
                # 2. Use `onFloor` teams.
                # 3. Match against CBD details (score, date).
                
                # If we have score, date, and "participants", we can match.
                t1 = teams_in_floor[0]
                t2 = teams_in_floor[1] if len(teams_in_floor) > 1 else ""
                
                game_meta.append({
                    'manual_game_id': row['gameSourceId'],
                    'date': row['date'],
                    'season': row['season'],
                    'manual_teams': [normalize(t1), normalize(t2)],
                    'manual_h_score': row['homeScore'],
                    'manual_a_score': row['awayScore']
                })
        except:
            continue
            
    manual_meta_df = pd.DataFrame(game_meta)
    
    # 3. Matching Logic
    logger.info(f"Matching {len(manual_meta_df)} manual games against {len(cbd_games)} CBD games...")
    
    matches = []
    
    # Index CBD by date for speed
    cbd_by_date = cbd_games.groupby('date_str')
    
    matched_cbd_ids = set()
    
    for _, m_row in manual_meta_df.iterrows():
        date = m_row['date']
        if date not in cbd_by_date.groups:
            continue
            
        candidates = cbd_by_date.get_group(date)
        
        # Method 1: Exact Score Match (Highly reliable)
        score_match = candidates[
            (candidates['homePoints'] == m_row['manual_h_score']) & 
            (candidates['awayPoints'] == m_row['manual_a_score'])
        ]
        
        # Verify teams if score matches
        final_match = None
        confidence = 0.0
        match_type = 'none'
        
        if len(score_match) == 1:
            # High confidence
            final_match = score_match.iloc[0]
            confidence = 1.0
            match_type = 'score_exact'
        elif len(score_match) > 1:
            # Multiple games with same score? Rare. Check teams.
            # Fuzzy match teams
            m_teams = m_row['manual_teams']
            best_score = 0
            best_cand = None
            
            for _, cand in score_match.iterrows():
                # Check fuzzy intersection
                c_teams = [cand['home_name'], cand['away_name']]
                # Simple check: do both manual teams fuzzy match the candidate teams?
                # We simply sum the best match scores
                s1 = process.extractOne(c_teams[0], m_teams, scorer=fuzz.token_sort_ratio)
                s2 = process.extractOne(c_teams[1], m_teams, scorer=fuzz.token_sort_ratio)
                avg_score = (s1[1] + s2[1]) / 2.0
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_cand = cand
            
            if best_score > 80:
                final_match = best_cand
                confidence = 0.95
                match_type = 'score_fuzzy_team'
                
        else:
            # No score match (maybe scraping error or different source). 
            # Try fuzzy team match on all games that day.
            m_teams = m_row['manual_teams']
            best_score = 0
            best_cand = None
            
            for _, cand in candidates.iterrows():
                # Check fuzzy intersection
                c_teams = [cand['home_name'], cand['away_name']]
                # Scoring: Max(Match T1->C_T1 + Match T2->C_T2, Match T1->C_T2 + Match T2->C_T1)
                
                # Direct
                d1 = fuzz.token_sort_ratio(m_teams[0], c_teams[0])
                d2 = fuzz.token_sort_ratio(m_teams[1], c_teams[1])
                direct_score = (d1 + d2) / 2.0
                
                # Swap
                x1 = fuzz.token_sort_ratio(m_teams[0], c_teams[1])
                x2 = fuzz.token_sort_ratio(m_teams[1], c_teams[0])
                swap_score = (x1 + x2) / 2.0
                
                score = max(direct_score, swap_score)
                
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            if best_score > 85: # High threshold for team-only match
                final_match = best_cand
                confidence = best_score / 100.0
                match_type = 'team_fuzzy'
        
        if final_match is not None:
            matches.append({
                'cbd_game_id': final_match['gameId'],
                'manual_game_id': m_row['manual_game_id'],
                'match_confidence': confidence,
                'match_type': match_type,
                'season': m_row['season']
            })
            matched_cbd_ids.add(final_match['gameId'])

    # 4. Save Crosswalk
    logger.info(f"Found {len(matches)} matches.")
    
    if matches:
        match_df = pd.DataFrame(matches)
        
        # Create table
        con.execute("CREATE OR REPLACE TABLE game_crosswalk AS SELECT * FROM match_df")
        con.execute("COPY game_crosswalk TO 'data/game_crosswalk.csv' (HEADER, DELIMITER ',')")
        logger.info("Saved data/game_crosswalk.csv and formatted table in DuckDB.")
    else:
        logger.warning("No matches found.")

if __name__ == "__main__":
    main()
