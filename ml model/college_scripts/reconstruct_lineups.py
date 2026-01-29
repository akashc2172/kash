import pandas as pd
import duckdb
import os
import re
from rapidfuzz import process, fuzz
from collections import defaultdict

# --- Configuration ---
SOURCE_DIRS = [
    "data/manual_scrapes/2015",
    "data/manual_scrapes/2017",
    # Add 2012/2013 when ready
]
DB_PATH = "data/warehouse.duckdb"
OUTPUT_FILE = "data/rapm_stints_historical.csv"

# --- 1. Team Matching Logic ---
def load_teams(con):
    print("  Loading team dictionary...")
    return con.execute("SELECT id, school, displayName, abbreviation FROM dim_teams").fetchdf()

def match_team_name(raw_name, team_df):
    """Fuzzy match raw name to DB team."""
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, 0
    
    # 1. Exact Match (School)
    exact = team_df[team_df['school'] == raw_name]
    if not exact.empty: return exact.iloc[0], 100
    
    # 2. Exact Match (DisplayName)
    exact_dn = team_df[team_df['displayName'] == raw_name]
    if not exact_dn.empty: return exact_dn.iloc[0], 100

    # 3. Fuzzy Match
    choices = team_df['school'].tolist()
    match, score, idx = process.extractOne(raw_name, choices, scorer=fuzz.token_set_ratio)
    
    # Heuristic: If < 80, try displayName
    if score < 85:
        match_dn, score_dn, idx_dn = process.extractOne(raw_name, team_df['displayName'].tolist(), scorer=fuzz.token_set_ratio)
        if score_dn > score:
            return team_df.iloc[idx_dn], score_dn
            
    return team_df.iloc[idx], score

# --- 2. Parsing Helpers ---
def parse_header(df):
    """Extract 'Home' and 'Away' team names from the first 'Score' row."""
    # Look for: "Time | Home | Score | Away"
    for text in df['raw_text']:
        if "| Score |" in text:
            parts = [p.strip() for p in text.split("|")]
            if len(parts) >= 4:
                return parts[1], parts[3] # Home, Away
    return None, None

def parse_explicit_lineup(text):
    """Extracts ['Player A', 'Player B'...] from 'TEAM For TEAMCODE: #01 Player A...'"""
    # Remove everything before the colon
    try:
        clean_text = text.split(":", 1)[-1]
        raw_players = [x for x in clean_text.split("#") if x.strip()]
        players = []
        for p in raw_players:
            # "01 Jackson, Warren, " -> "JACKSON,WARREN"
            # Remove digits and extra punct
            p_clean = re.sub(r'^\d+\s+', '', p).split(",")[0] + "," + p.split(",")[1].rstrip(",. ") 
            # Actually, let's keep it simple: "JACKSON,WARREN" to match usual PBP format
            name_part = re.sub(r'^\d+\s+', '', p).strip().rstrip(",. ")
            players.append(name_part.upper().replace(" ", ""))
        return set(players)
    except:
        return set()

def clean_player_name(text):
    """Extracts player name from 'NAME,FIRST Leaves/Enters Game'."""
    # "JACKSON,WARREN Leaves Game" -> "JACKSON,WARREN"
    name = text.split("Leaves")[0].split("Enters")[0].strip()
    return name.upper().replace(" ", "")

# --- 3. Reconstruction Logic ---
def process_game(contest_id, df, team_df):
    # Sort by index to ensure chronological order
    df = df.sort_index()
    rows = df['raw_text'].tolist()
    
    # 1. Identify Teams
    h_raw, a_raw = parse_header(df)
    if not h_raw: return None # Bad Game
    
    (h_team, h_score), (a_team, a_score) = match_team_name(h_raw, team_df), match_team_name(a_raw, team_df)
    
    if h_score < 70 or a_score < 70:
        # print(f"    ⚠️ Low confidence match: {h_raw}->{h_team['school']}({h_score}), {a_raw}->{a_team['school']}({a_score})")
        pass

    # 2. Scan Events
    events = []
    
    # State Tracking
    home_roster = set()
    away_roster = set()
    
    # Checkpoints: (index, 'HOME'/'AWAY', {players})
    checkpoints = [] 
    
    for i, row in enumerate(rows):
        parts = [p.strip() for p in row.split("|")]
        if len(parts) < 4: continue
        
        home_evt = parts[1]
        away_evt = parts[3]
        
        # Explicit Lineups
        if "TEAM For" in home_evt:
            ps = parse_explicit_lineup(home_evt)
            if ps: checkpoints.append((i, 'HOME', ps))
        if "TEAM For" in away_evt:
            ps = parse_explicit_lineup(away_evt)
            if ps: checkpoints.append((i, 'AWAY', ps))
            
        # Subs & Roster Building
        if "Enters Game" in home_evt:
            p = clean_player_name(home_evt)
            home_roster.add(p)
            events.append({'i': i, 'team': 'HOME', 'type': 'IN', 'player': p})
        if "Leaves Game" in home_evt:
            p = clean_player_name(home_evt)
            home_roster.add(p)
            events.append({'i': i, 'team': 'HOME', 'type': 'OUT', 'player': p})
            
        if "Enters Game" in away_evt:
            p = clean_player_name(away_evt)
            away_roster.add(p)
            events.append({'i': i, 'team': 'AWAY', 'type': 'IN', 'player': p})
        if "Leaves Game" in away_evt:
            p = clean_player_name(away_evt)
            away_roster.add(p)
            events.append({'i': i, 'team': 'AWAY', 'type': 'OUT', 'player': p})
            
        # Stat Events (for heuristic inference)
        # Any text in home_evt that isn't sub/timeout implies player on floor
        if home_evt and "Enters" not in home_evt and "Leaves" not in home_evt:
             # Extract potential name "NAME,FIRST action"
             first_word = home_evt.split(" ")[0]
             if "," in first_word: # Rough check for "LAST,FIRST"
                 events.append({'i': i, 'team': 'HOME', 'type': 'STAT', 'player': first_word.strip()})
        
        if away_evt and "Enters" not in away_evt and "Leaves" not in away_evt:
             first_word = away_evt.split(" ")[0]
             if "," in first_word:
                 events.append({'i': i, 'team': 'AWAY', 'type': 'STAT', 'player': first_word.strip()})

    # 3. Solver: Determine Lineup at t=0
    def solve_start_lineup(team_code, checkpoints, team_events):
        team_checks = [c for c in checkpoints if c[1] == team_code]
        
        # Strategy A: Use Checkpoint
        if team_checks:
            # Use the first checkpoint and back-propagate to 0
            idx, _, lineup = team_checks[0]
            curr = set(lineup)
            
            # Events BEFORE checkpoint, reversed
            pre_events = sorted([e for e in team_events if e['i'] < idx], key=lambda x: x['i'], reverse=True)
            
            for e in pre_events:
                if e['type'] == 'IN':
                    curr.discard(e['player']) # If they entered, they weren't in before
                elif e['type'] == 'OUT':
                    curr.add(e['player']) # If they left, they WAS in before
            return curr
            
        # Strategy B: Inference (First Event Heuristic)
        # Filter to IN/OUT/STAT events
        # A player is a STARTER if:
        # 1. They have a STAT event BEFORE their first IN event.
        # 2. They have an OUT event BEFORE their first IN event.
        
        starters = set()
        seen_players = set()
        
        # Sort chronologically
        chron_events = sorted(team_events, key=lambda x: x['i'])
        
        for e in chron_events:
            p = e['player']
            if p in seen_players: continue
            
            if e['type'] == 'IN':
                # First event is IN -> NOT a starter
                seen_players.add(p)
            elif e['type'] in ['OUT', 'STAT']:
                # First event is OUT or STAT -> MUST be a starter
                starters.add(p)
                seen_players.add(p)
                
            if len(starters) >= 5: break # Cap at 5?
            
        return starters

    home_start = solve_start_lineup('HOME', checkpoints, [e for e in events if e['team'] == 'HOME'])
    away_start = solve_start_lineup('AWAY', checkpoints, [e for e in events if e['team'] == 'AWAY'])
    
    # 4. Generate Stint Log
    stints = []
    
    curr_h = set(home_start)
    curr_a = set(away_start)
    
    # We need to emit a row for every 'score' change or period end?
    # Or just "Stint X: Start Time, End Time, Lineup H, Lineup A"
    # Actually, for RApM we usually want "Play X: Lineup H, Lineup A".
    # Let's attach lineups to the rows we have.
    
    for i, row in enumerate(rows):
        # Apply subs that happen at this index
        these_events = [e for e in events if e['i'] == i]
        for e in these_events:
            if e['type'] == 'IN':
                if e['team'] == 'HOME': curr_h.add(e['player'])
                else: curr_a.add(e['player'])
            elif e['type'] == 'OUT':
                if e['team'] == 'HOME': curr_h.discard(e['player'])
                else: curr_a.discard(e['player'])
        
        # If this row is a play (has score/time), record the state
        # We can just record the state for every row
        stints.append({
            'game_id': contest_id,
            'season': 2015, # Extract from filename ideally
            'row_idx': i,
            'home_team_id': h_team['id'],
            'away_team_id': a_team['id'],
            'home_lineup': sorted(list(curr_h)),
            'away_lineup': sorted(list(curr_a)),
            'raw_text': row
        })
        
    return stints

def main():
    print("🦊 Starting Lineup Reconstruction...")
    con = duckdb.connect(DB_PATH)
    teams_df = load_teams(con)
    
    all_stints = []
    
    for folder in SOURCE_DIRS:
        if not os.path.exists(folder): continue
        files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        
        print(f"\n📂 Processing {folder} ({len(files)} files)...")
        
        for f in files:
            path = os.path.join(folder, f)
            try:
                df = pd.read_csv(path)
                # Group by contest (the files contain multiple games)
                contest_ids = df['contest_id'].unique()
                
                # print(f"  > {f}: {len(contest_ids)} games")
                
                for cid in contest_ids:
                    game_df = df[df['contest_id'] == cid]
                    game_stints = process_game(cid, game_df, teams_df)
                    if game_stints:
                        all_stints.extend(game_stints)
                        
            except Exception as e:
                print(f"    ❌ Error reading {f}: {e}")

    # Save
    print(f"\n💾 Saving {len(all_stints)} stint rows to {OUTPUT_FILE}...")
    res_df = pd.DataFrame(all_stints)
    
    # Convert lists to strings for CSV
    res_df['home_lineup'] = res_df['home_lineup'].apply(lambda x: "|".join(x))
    res_df['away_lineup'] = res_df['away_lineup'].apply(lambda x: "|".join(x))
    
    res_df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()
