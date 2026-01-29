import pandas as pd
import duckdb
import re
from rapidfuzz import process, fuzz

# Configuration
PBP_FILE = "data/manual_scrapes/2015/ncaa_pbp_2015-11-13.csv"
DB_PATH = "data/warehouse.duckdb"

def load_teams(con):
    teams = con.execute("SELECT id, school, displayName, abbreviation FROM dim_teams").fetchdf()
    return teams

def parse_game_header(df, contest_id):
    rows = df[df['contest_id'] == contest_id]['raw_text'].tolist()
    for text in rows:
        if "| Score |" in text:
            # Time | Home | Score | Away
            parts = [p.strip() for p in text.split("|")]
            if len(parts) >= 4:
                return parts[1], parts[3]
    return "Unknown", "Unknown"

def clean_name(n):
    # Remove (CA), (OH), etc if strictly needed, but let's try fuzzy first
    return n.strip()

def match_teams(home_raw, away_raw, team_df):
    # Create a dict for faster lookups? No, fuzzy is O(N) anyway.
    # Check normalized first
    
    # Try to find "Saint Mary's" in the list if "Saint Mary's (CA)" fails
    def get_best(name, candidates):
        # 1. Exact match on 'school' or 'displayName'
        exact = team_df[team_df['school'] == name]
        if not exact.empty: return exact.iloc[0], 100
        
        # 2. Token Set Ratio (handles extra words/reordering)
        match, score, idx = process.extractOne(name, candidates['school'], scorer=fuzz.token_set_ratio)
        return team_df.iloc[idx], score

    home_match, h_score = get_best(home_raw, team_df)
    away_match, a_score = get_best(away_raw, team_df)
    
    return (home_match, h_score), (away_match, a_score)

def parse_lineup_explicit(text):
    # Regex to capture #XX Name, First
    # Pattern: #\d+\s+[^#,]+
    # Example: #01 Jackson, Warren
    import re
    # This pattern looks for # followed by digits, space, and then text until the next # or end
    # Note: names have commas like "Jackson, Warren"
    pattern = r"#(\d+)\s+([A-Za-z' \-\.]+)(?:,|$)" 
    # Actually the text is "#01 Jackson, Warren, #05..."
    # So we split by '#' might be safer?
    
    clean_text = text.split(":", 1)[-1] # Remove "TEAM For SFSU:"
    
    # Split by the hash that denotes a new player
    # But skip the empty first split if it starts with #
    raw_players = [x for x in clean_text.split("#") if x.strip()]
    
    players = []
    for p in raw_players:
        # p is like "01 Jackson, Warren, "
        # Remove trailing commas/spaces
        p_clean = p.rstrip(",. ")
        players.append(p_clean) # e.g. "01 Jackson, Warren"
        
    return players

def main():
    con = duckdb.connect(DB_PATH)
    teams_df = load_teams(con)
    pbp_df = pd.read_csv(PBP_FILE)
    contest_id = 239671
    
    # 1. Match Headers
    h_raw, a_raw = parse_game_header(pbp_df, contest_id)
    print(f"Header: {h_raw} (Home) vs {a_raw} (Away)")
    
    (h_obj, h_sc), (a_obj, a_sc) = match_teams(h_raw, a_raw, teams_df)
    print(f"  Home Match: {h_obj['school']} ({h_sc}) ID: {h_obj['id']}")
    print(f"  Away Match: {a_obj['school']} ({a_sc}) ID: {a_obj['id']}")
    
    # 2. Parse Events with Column Logic
    rows = pbp_df[pbp_df['contest_id'] == contest_id]['raw_text'].tolist()
    
    print("\n--- Parsing Flow ---")
    current_lineup_home = set()
    current_lineup_away = set()
    
    for i, row in enumerate(rows):
        parts = [p.strip() for p in row.split("|")]
        # Standard row: Time | HomeEvent | Score | AwayEvent
        # Sometimes 4 parts, sometimes 5? 
        # "19:37 | WORMLEY,FLOYD Assist | 2-0 | " matches 4 parts
        
        if len(parts) < 4: continue
        
        time = parts[0]
        home_evt = parts[1]
        score = parts[2]
        away_evt = parts[3]
        
        # Check Explicit Lineups (usually in Home or Away event slot)
        # "TEAM For SFSU:#01..."
        if "TEAM For" in home_evt:
             players = parse_lineup_explicit(home_evt)
             print(f"[{time}] EXPLICIT HOME ({len(players)}): {players}")
             current_lineup_home = set(players)
             
        if "TEAM For" in away_evt:
             players = parse_lineup_explicit(away_evt)
             print(f"[{time}] EXPLICIT AWAY ({len(players)}): {players}")
             current_lineup_away = set(players)

        # Check Subs
        # "JACKSON,WARREN Leaves Game"
        if "Leaves Game" in home_evt:
            name = home_evt.replace("Leaves Game", "").strip()
            print(f"[{time}] SUB OUT HOME: {name}")
            # current_lineup_home.discard(name) # Need fuzzy name match likely
            
        if "Enters Game" in home_evt:
            name = home_evt.replace("Enters Game", "").strip()
            print(f"[{time}] SUB IN HOME: {name}")
            # current_lineup_home.add(name)

        if "Leaves Game" in away_evt:
            name = away_evt.replace("Leaves Game", "").strip()
            print(f"[{time}] SUB OUT AWAY: {name}")
            
        if "Enters Game" in away_evt:
            name = away_evt.replace("Enters Game", "").strip()
            print(f"[{time}] SUB IN AWAY: {name}")

        if i > 60: break

if __name__ == "__main__":
    main()
