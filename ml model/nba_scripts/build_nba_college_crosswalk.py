"""
Build NBA-College Crosswalk (Phase 4 Prerequisite)
==================================================
Purpose: Link `nba_id` (from Basketball-Excel/EPM) to `athlete_id` (from College Feature Store).
Source of Truth:
    - NBA: `dim_player_crosswalk` (ID, Name, Draft Year)
    - College: `warehouse.duckdb` (stg_shots.shooter_name -> athlete_id)

Method:
    1. Extract Dictionary `(athlete_id -> name, latest_season)` from College DB.
    2. Extract Dictionary `(nba_id -> name, draft_year)` from NBA Warehouse.
    3. Fuzzy Match with Draft Year Constraint:
       - Match Name Score > 0.85
       - `abs(Draft_Year - College_Final_Season) <= 2` (Allow for redshirting/gap years)

Output:
    - data/warehouse_v2/dim_player_nba_college_crosswalk.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
import re
from difflib import SequenceMatcher
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
WAREHOUSE_DIR = Path("data/warehouse_v2")
DB_PATH = 'data/warehouse.duckdb'
OUT_FILE = WAREHOUSE_DIR / "dim_player_nba_college_crosswalk.parquet"

def normalize_name(name):
    """Normalize names for matching: lowercase, remove punctuation/suffixes."""
    if pd.isna(name): return ""
    name = str(name).lower()
    name = re.sub(r'[^a-z\s]', '', name)
    name = re.sub(r'\s+(jr|sr|ii|iii|iv)$', '', name)
    return name.strip()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_college_players():
    """Extract distinct athlete_id -> name from stg_shots."""
    logger.info("Extracting college player directory from DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # We take the most common name for an ID (handle variations)
    # Also get max season to help matching
    df = con.execute("""
        SELECT 
            shooterAthleteId as athlete_id,
            mode(shooter_name) as athlete_name,
            MAX(g.season) as final_season
        FROM stg_shots s
        JOIN dim_games g ON g.id = CAST(s.gameId AS BIGINT)
        WHERE shooterAthleteId IS NOT NULL
        GROUP BY 1
    """).df()
    
    df['norm_name'] = df['athlete_name'].apply(normalize_name)
    logger.info(f"Loaded {len(df):,} college athletes.")
    con.close()
    return df

def get_nba_players():
    """Extract NBA players with draft context."""
    logger.info("Loading NBA players...")
    crosswalk = pd.read_parquet(WAREHOUSE_DIR / "dim_player_crosswalk.parquet")
    dim_nba = pd.read_parquet(WAREHOUSE_DIR / "dim_player_nba.parquet")
    
    # Merge to get Draft Year
    df = pd.merge(crosswalk[['nba_id', 'player_name']], 
                  dim_nba[['nba_id', 'draft_year', 'rookie_season_year']], 
                  on='nba_id', how='inner')
    
    df['norm_name'] = df['player_name'].apply(normalize_name)
    
    # Use rookie_season - 1 as proxy for draft year if missing
    df['draft_year_proxy'] = df['draft_year'].fillna(df['rookie_season_year'] - 1)
    
    logger.info(f"Loaded {len(df):,} NBA players.")
    return df

def match_players(college_df, nba_df):
    logger.info("Running optimized fuzzy matching...")
    
    # Pre-calculate exact match lookups (Blocked by Letter for speed)
    college_lookup = {}
    for r in college_df.to_dict('records'):
        if not r['norm_name']: continue
        letter = r['norm_name'][0]
        if letter not in college_lookup: college_lookup[letter] = []
        college_lookup[letter].append(r)

    matches = []
    
    for _, nba in nba_df.iterrows():
        n_name = nba['norm_name']
        n_draft = nba['draft_year_proxy']
        n_id = nba['nba_id']
        
        if pd.isna(n_draft) or not n_name: continue
        
        letter = n_name[0]
        if letter not in college_lookup: continue
        
        best_score = 0
        best_match = None
        
        # Scan candidates with same first letter
        for college in college_lookup[letter]:
            # 1. Temporal Constraints
            if abs(college['final_season'] - n_draft) > 2:
                continue
            
            # 2. Name Matching
            if college['norm_name'] == n_name:
                score = 1.0
            else:
                # Fuzzy is slow, only do it if basics match
                # e.g. length within 3 chars
                if abs(len(n_name) - len(college['norm_name'])) > 3:
                    continue
                score = similar(n_name, college['norm_name'])
            
            if score > 0.88 and score > best_score:
                best_score = score
                best_match = college
        
        if best_match:
            matches.append({
                'nba_id': n_id,
                'athlete_id': best_match['athlete_id'],
                'nba_name': nba['player_name'],
                'college_name': best_match['athlete_name'],
                'match_score': best_score,
                'draft_year': n_draft,
                'college_final': best_match['final_season']
            })
            
    return pd.DataFrame(matches)

def main():
    col_df = get_college_players()
    nba_df = get_nba_players()
    
    matches_df = match_players(col_df, nba_df)
    
    # Validation
    match_rate = len(matches_df) / len(nba_df) if len(nba_df) > 0 else 0
    logger.info(f"Matched {len(matches_df)} players ({match_rate:.1%} of NBA cohort).")
    
    # CURSOR NOTE: Validate match quality
    high_confidence = matches_df[matches_df['match_score'] >= 0.95]
    logger.info(f"  High confidence matches (score >= 0.95): {len(high_confidence)} ({len(high_confidence)/len(matches_df):.1%} of matches)")
    
    # Check for duplicate matches (one NBA player matched to multiple college players)
    dup_nba = matches_df[matches_df.duplicated(subset=['nba_id'], keep=False)]
    if len(dup_nba) > 0:
        logger.warning(f"  ⚠️  {len(dup_nba)} NBA players matched to multiple college athletes - review needed")
        logger.warning(f"      Example duplicates: {dup_nba[['nba_id', 'nba_name', 'college_name', 'match_score']].head(5).to_dict('records')}")
    
    # Check for duplicate college matches (one college player matched to multiple NBA players)
    dup_college = matches_df[matches_df.duplicated(subset=['athlete_id'], keep=False)]
    if len(dup_college) > 0:
        logger.warning(f"  ⚠️  {len(dup_college)} college athletes matched to multiple NBA players - review needed")
    
    # Save full match details for debugging, then save minimal crosswalk
    debug_file = WAREHOUSE_DIR / "dim_player_nba_college_crosswalk_debug.parquet"
    matches_df.to_parquet(debug_file, index=False)
    logger.info(f"Saved full match details to {debug_file}")
    
    # Save minimal crosswalk (just IDs and score)
    matches_df[['nba_id', 'athlete_id', 'match_score']].to_parquet(OUT_FILE, index=False)
    logger.info(f"Saved crosswalk to {OUT_FILE}")

if __name__ == "__main__":
    main()
