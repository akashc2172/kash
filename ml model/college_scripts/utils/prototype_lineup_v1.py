import pandas as pd
import duckdb
import re
from rapidfuzz import process, fuzz

# Configuration
PBP_FILE = "data/manual_scrapes/2015/ncaa_pbp_2015-11-13.csv"
DB_PATH = "data/warehouse.duckdb"

def load_teams(con):
    teams = con.execute("SELECT id, school, displayName, abbreviation FROM dim_teams").fetchdf()
    # Create a mapping dictionary for fuzzy matching
    # We will match against 'school' and 'displayName'
    return teams

def get_game_header(df, contest_id):
    # Find the "Time | Team A | Score | Team B" line
    # Usually it's the very first row or the one with "Score" in it
    rows = df[df['contest_id'] == contest_id].sort_index()
    
    for text in rows['raw_text']:
        if "| Score |" in text:
            # Format: Time | San Fran. St. | Score | Saint Mary's (CA)
            parts = text.split("|")
            if len(parts) >= 4:
                team1 = parts[1].strip()
                team2 = parts[3].strip()
                return team1, team2
    return None, None

def fuzzy_match_team(name, team_df):
    # Simple fuzzy match against 'school'
    choices = team_df['school'].tolist()
    match, score, idx = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    
    # Try displayName if school score is low
    if score < 85:
        choices_dn = team_df['displayName'].tolist()
        match_dn, score_dn, idx_dn = process.extractOne(name, choices_dn, scorer=fuzz.token_sort_ratio)
        if score_dn > score:
            match = match_dn
            score = score_dn
            # Find the row corresponding to this displayName
            matched_row = team_df[team_df['displayName'] == match_dn].iloc[0]
            return matched_row, score

    # Find row by school
    matched_row = team_df[team_df['school'] == match].iloc[0]
    return matched_row, score

def parse_lineup_summary(text):
    # Example: "TEAM For SFSU:#01 Jackson, Warren, #05 Jones, Andre..."
    # Regex to find "#XX Name, Name"
    # Returns a list of player names (or jersey + name)
    if "TEAM For" not in text:
        return None
    
    # Extract the part after the colon or the team code
    # "TEAM For SFSU:#01 Jackson..."
    try:
        content = text.split(":", 1)[1]
    except:
        return None
    
    # Split by comma-space-hash usually separates players? 
    # Actually looking at the example: "#01 Jackson, Warren, #05 Jones, Andre"
    # This is tricky because names have commas.
    # Player pattern: #\d+ [A-Za-z, ]+
    
    # Better approach: Split by '#' and clean up
    # "01 Jackson, Warren, " "05 Jones, Andre, "
    
    players = []
    # Split by the # symbol
    segments = content.split("#")
    for seg in segments:
        seg = seg.strip().strip(",").strip(".")
        if not seg:
            continue
        # seg is "01 Jackson, Warren"
        # We might want to just keep this raw string as the player ID for now
        players.append(seg)
        
    return players[:5] # Should be 5 players

def main():
    print("🦊 Loading Data...")
    con = duckdb.connect(DB_PATH)
    teams_df = load_teams(con)
    pbp_df = pd.read_csv(PBP_FILE)
    
    # Pick the first game (San Fran St vs St Marys)
    # contest_id: 239560
    contest_id = 239560
    
    print(f"\nAnalyzing Contest ID: {contest_id}")
    
    # 1. Match Teams
    raw_t1, raw_t2 = get_game_header(pbp_df, contest_id)
    print(f"Raw Header Teams: '{raw_t1}' vs '{raw_t2}'")
    
    match1, score1 = fuzzy_match_team(raw_t1, teams_df)
    match2, score2 = fuzzy_match_team(raw_t2, teams_df)
    
    print(f"Match 1: {raw_t1} -> {match1['school']} (ID: {match1['id']}) [Score: {score1}]")
    print(f"Match 2: {raw_t2} -> {match2['school']} (ID: {match2['id']}) [Score: {score2}]")
    
    # 2. Iterate and Parse Events
    game_rows = pbp_df[pbp_df['contest_id'] == contest_id]['raw_text'].tolist()
    
    on_floor_home = []
    on_floor_away = []
    
    print("\n--- Play Flow Sample ---")
    for i, row in enumerate(game_rows):
        # Check for explicit lineup checks
        # "16:29 | TEAM For SFSU:#01 Jackson..."
        
        if "TEAM For" in row:
            # Is it Home or Away?
            # We need to map the "SFSU" code to one of our teams.
            # Usually the code is in the text "For SFSU:"
            
            lineup = parse_lineup_summary(row)
            print(f"[{i}] explicit_lineup_check: {lineup}")
            
        elif "Enters Game" in row:
             print(f"[{i}] SUB IN: {row}")
             
        elif "Leaves Game" in row:
             print(f"[{i}] SUB OUT: {row}")
             
        if i > 50: break # Just showing head

if __name__ == "__main__":
    main()
