
"""
NCAA History Parser
-------------------
Parses raw scraped PBP text to restore substitution events.

Input: data/scraped_history/scraped_pbp_*.parquet
Output: data/scraped_history/stg_subs_restored.parquet

Logic:
1. Regex parse "Player X substitution in/out"
2. Fuzzy match Player X to fact_player_season_stats roster for that team/year.
3. Generate standard stg_subs format (gameId, athleteId, type, period, time).
"""

import duckdb
import pandas as pd
import re
import argparse
import logging
from thefuzz import process, fuzz

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = 'data/warehouse.duckdb'
INPUT_DIR = 'data/scraped_history'

def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

def normalize_name(name):
    """Normalize names for matching."""
    if not name: return ""
    return name.lower().replace('.', '').replace(',', '').strip()

def load_rosters(con, season):
    """Load player roster for the season for name matching."""
    logger.info(f"Loading rosters for {season}...")
    df = con.execute(f"""
        SELECT 
            athleteId as athlete_id,
            name as athlete_name,
            team as team_text,
            season
        FROM fact_player_season_stats
        WHERE season = {season}
    """).df()
    # Normalize names
    df['search_name'] = df['athlete_name'].apply(normalize_name)
    return df

def parse_pbp_text(text):
    """Extract sub type and player name from text."""
    # Text: "Smith, John substitution in"
    # Regex: ^(.+),?\s+substitution\s+(in|out)
    text = text.lower().strip()
    
    # Pattern 1: "Name substitution in"
    match = re.search(r'(.+?)[\.,]?\s+substitution\s+(in|out)', text)
    if match:
        name = match.group(1).strip()
        sub_type = 'IN' if match.group(2) == 'in' else 'OUT'
        return name, sub_type
        
    return None, None

def resolve_player(name, team_roster):
    """Fuzzy match name against specific team roster."""
    if team_roster.empty:
        return None
        
    # clean name
    clean_name = normalize_name(name)
    
    # Exact match first
    exact = team_roster[team_roster['search_name'] == clean_name]
    if not exact.empty:
        return exact.iloc[0]['athlete_id']
        
    # Fuzzy match
    choices = team_roster['search_name'].tolist()
    best_match, score = process.extractOne(clean_name, choices, scorer=fuzz.token_sort_ratio)
    
    if score > 80:
        return team_roster[team_roster['search_name'] == best_match].iloc[0]['athlete_id']
        
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, default=2023)
    args = parser.parse_args()
    
    con = get_connection()
    
    # 1. Load Scraped Data
    try:
        pbp_path = f"{INPUT_DIR}/scraped_pbp_{args.season}.parquet"
        map_path = f"{INPUT_DIR}/ncaa_game_map_{args.season}.parquet"
        
        if not os.path.exists(pbp_path):
             logger.warning(f"File not found: {pbp_path}")
             return

        logger.info(f"Loading {pbp_path}...")
        df_pbp = pd.read_parquet(pbp_path)
        df_map = pd.read_parquet(map_path)
    except Exception as e:
        logger.error(f"Could not load input files: {e}")
        return

    # 2. Load Rosters
    roster_df = load_rosters(con, args.season)
    
    # Get Team Info for Games to filter rosters
    # Join with map to handle gameId type
    games_meta = con.execute(f"""
        SELECT 
            CAST(g.id AS VARCHAR) as gameId, 
            ht.school as home_team, 
            at_t.school as away_team
        FROM dim_games g
        JOIN dim_teams ht ON g.homeTeamId = ht.id
        JOIN dim_teams at_t ON g.awayTeamId = at_t.id
        WHERE g.season = {args.season}
    """).df()
    
    # 3. Process Rows
    parsed_subs = []
    
    logger.info("Parsing substitutions...")
    
    # Group by game for roster context
    grouped = df_pbp.groupby('gameId')
    
    for game_id, group in grouped:
        # Get Team Names
        meta = games_meta[games_meta['gameId'] == str(game_id)]
        if meta.empty:
            continue
        
        home_team = meta.iloc[0]['home_team']
        away_team = meta.iloc[0]['away_team']
        
        # Filter Rosters
        home_roster = roster_df[roster_df['team_text'].str.contains(re.escape(home_team), case=False, na=False)]
        away_roster = roster_df[roster_df['team_text'].str.contains(re.escape(away_team), case=False, na=False)]
        
        # Iterate PBP rows
        for _, row in group.iterrows():
            raw_cols = row['raw_cols'] # list
            # Usually: Time, HomeScore, AwayScore, PlayText?
            # Or Time, Team, Play?
            
            # Combine all cols to search for "substitution"
            # Some PBP lists are ["19:22", "Duke", "Smith substitution in"]
            full_text = " ".join([str(c) for c in raw_cols])
            
            if "substitution" in full_text.lower():
                name, sub_type = parse_pbp_text(full_text)
                if name:
                    # Resolve to Athlete ID
                    # Try both rosters
                    ath_id = resolve_player(name, home_roster)
                    
                    if not ath_id:
                        ath_id = resolve_player(name, away_roster)
                    
                    if ath_id:
                        # Extract Time
                        time_str = "00:00"
                        for c in raw_cols:
                             if ':' in str(c) and len(str(c)) <= 5:
                                 time_str = c
                                 break
                        
                        parsed_subs.append({
                            'gameId': game_id,
                            'athlete_id': ath_id,
                            'type': sub_type, # SUB_IN / SUB_OUT
                            'period': 1, # Placeholder
                            'time': time_str,
                            'text': full_text
                        })

    # Save
    if parsed_subs:
        df_subs = pd.DataFrame(parsed_subs)
        out_path = f"{INPUT_DIR}/stg_subs_restored_{args.season}.parquet"
        df_subs.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df_subs)} restored substitutions to {out_path}")
    else:
        logger.warning("No substitutions parsed.")

if __name__ == "__main__":
    main()
