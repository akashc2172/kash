import pandas as pd
import duckdb
import re
from rapidfuzz import process, fuzz

PBP_FILE = "data/manual_scrapes/2015/ncaa_pbp_2015-11-13.csv"
DB_PATH = "data/warehouse.duckdb"

def parse_lineup_explicit(text):
    # Pattern: #01 Jackson, Warren
    clean_text = text.split(":", 1)[-1]
    raw_players = [x for x in clean_text.split("#") if x.strip()]
    players = []
    for p in raw_players:
        p_clean = p.rstrip(",. ")
        players.append(p_clean) 
    return set(players)

def clean_sub_name(text):
    # "JACKSON,WARREN Leaves Game" -> "Jackson, Warren"
    # Need to match the format "01 Jackson, Warren" ideally?
    # PBP says "JACKSON,WARREN". Explicit says "01 Jackson, Warren".
    # We need a Player Name Normalizer. 
    # For now, let's just use the name part uppercase?
    # "Jackson, Warren" -> "JACKSON,WARREN" check
    # Or just keep consistent keys.
    return text.split("Leaves")[0].split("Enters")[0].strip()

def normalize_name(n):
    # "01 Jackson, Warren" -> "JACKSON,WARREN" (approx)
    # Remove number, uppercase, remove space?
    # "Jackson, Warren" -> "JACKSON,WARREN"
    n = re.sub(r'^\d+\s+', '', n) # Remove leading number
    return n.upper().replace(" ", "")

def main():
    pbp_df = pd.read_csv(PBP_FILE)
    contest_id = 239671
    rows = pbp_df[pbp_df['contest_id'] == contest_id]['raw_text'].tolist()
    
    # 1. First Pass: Find Explicit Lineups relative to an index
    checkpoints = [] # (index, is_home, set_of_players)
    
    events = []
    
    print("--- 1. Scanning Events ---")
    for i, row in enumerate(rows):
        parts = [p.strip() for p in row.split("|")]
        if len(parts) < 4: continue
        
        home_evt = parts[1]
        away_evt = parts[3]
        
        # Explicit Lineups
        if "TEAM For" in home_evt:
             players = parse_lineup_explicit(home_evt)
             # Normalize players
             norm_players = {normalize_name(p) for p in players}
             checkpoints.append( (i, 'HOME', norm_players) )
             print(f"[{i}] Checkpoint HOME: {norm_players}")
             
        if "TEAM For" in away_evt:
             players = parse_lineup_explicit(away_evt)
             norm_players = {normalize_name(p) for p in players}
             checkpoints.append( (i, 'AWAY', norm_players) )
             
        # Subs
        # { 'index': i, 'team': 'HOME', 'type': 'OUT', 'player': 'NAME' }
        if "Leaves Game" in home_evt:
            name = clean_sub_name(home_evt)
            events.append({'index': i, 'team': 'HOME', 'type': 'OUT', 'player': name})
            
        if "Enters Game" in home_evt:
            name = clean_sub_name(home_evt)
            events.append({'index': i, 'team': 'HOME', 'type': 'IN', 'player': name})
            
        if "Leaves Game" in away_evt:
            name = clean_sub_name(away_evt)
            events.append({'index': i, 'team': 'AWAY', 'type': 'OUT', 'player': name})
            
        if "Enters Game" in away_evt:
            name = clean_sub_name(away_evt)
            events.append({'index': i, 'team': 'AWAY', 'type': 'IN', 'player': name})

    # 2. Back Propagation for HOME Team
    # Find first checkpoint
    home_checks = [c for c in checkpoints if c[1] == 'HOME']
    if not home_checks:
        print("No home checkpoint found!")
        return
        
    start_idx, _, current_lineup = home_checks[0]
    print(f"\n--- Back Prop from Index {start_idx} ---")
    print(f"Known Lineup at {start_idx}: {current_lineup}")
    
    # Walk backwards from start_idx to 0
    # Relevant events: those < start_idx
    relevant_events = sorted([e for e in events if e['team'] == 'HOME' and e['index'] < start_idx], key=lambda x: x['index'], reverse=True)
    
    # REVERSE LOGIC:
    # Event "IN P1" at T means P1 was NOT in at T-1. -> Remove P1
    # Event "OUT P2" at T means P2 WAS in at T-1. -> Add P2
    
    for ev in relevant_events:
        p = ev['player'].replace(" ", "") # Normalize space
        
        if ev['type'] == 'IN':
            # Player entered, so before this they were OUT.
            if p in current_lineup:
                current_lineup.remove(p)
            else:
                print(f"Warning: Saw IN for {p} but they weren't in lineup?")
                
        elif ev['type'] == 'OUT':
            # Player left, so before this they were IN.
            current_lineup.add(p)
            
    print(f"\n✅ INFERRED STARTING LINEUP (HOME): {current_lineup}")
    print(f"Count: {len(current_lineup)}")

if __name__ == "__main__":
    main()
