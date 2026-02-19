import pandas as pd
import duckdb
import os
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

# --- Configuration ---
RAW_FILE = "data/manual_scrapes/2015/ncaa_pbp_2015-11-13.csv"
OUTPUT_FILE = "data/fact_play_historical_2015_clean.csv"
DB_PATH = "data/warehouse.duckdb"

def normalize_name(n):
    # Standardize names: "01 Jackson, Warren" -> "JACKSON,WARREN"
    # Remove #XX, remove spaces, uppercase
    n = re.sub(r'#\d+\s*', '', n) # Remove #01
    n = re.sub(r'^\d+\s*', '', n) # Remove leading numbers 01
    return n.strip().upper().replace(" ", "")

def get_header_teams(df):
    for text in df['raw_text']:
        if "| Score |" in text:
            parts = [p.strip() for p in text.split("|")]
            if len(parts) >= 4:
                return parts[1], parts[3]
    return "HOME", "AWAY"

class GameSolver:
    def __init__(self, game_id, rows, h_team, a_team, season):
        self.game_id = game_id
        self.rows = rows
        self.h_team = h_team
        self.a_team = a_team
        self.season = season
        
        # Tracking
        self.events = []
        self.roster_home = Counter()
        self.roster_away = Counter()
        self.checkpoints = [] # (idx, team, set_players)
        
        # Final State
        self.on_floor_history = [] # list of (h_set, a_set)

    def parse_rows(self):
        """Pass 1: Parse Loop"""
        for i, row in enumerate(self.rows):
            parts = [p.strip() for p in row.split("|")]
            if len(parts) < 4: 
                self.events.append(None)
                continue
                
            h_evt, a_evt = parts[1], parts[3]
            
            row_evts = []
            
            # Helper to process an event string
            def process_text(text, team_label, roster_counter):
                if not text: return
                
                # Check Explicit
                if "TEAM For" in text:
                    # Parse players
                    raw = text.split(":", 1)[-1]
                    ps = {normalize_name(x) for x in raw.split("#") if x.strip()}
                    self.checkpoints.append((i, team_label, ps))
                    # Also add to roster
                    for p in ps: roster_counter[p] += 1
                    return

                # Check Subs
                if "Enters" in text:
                    p = normalize_name(text.split("Enters")[0])
                    roster_counter[p] += 1
                    row_evts.append({'type': 'IN', 'team': team_label, 'player': p})
                elif "Leaves" in text:
                    p = normalize_name(text.split("Leaves")[0])
                    roster_counter[p] += 1
                    row_evts.append({'type': 'OUT', 'team': team_label, 'player': p})
                else:
                    # Stat? "JACKSON,WARREN made"
                    first = text.split(" ")[0]
                    if "," in first:
                        p = normalize_name(first)
                        roster_counter[p] += 1  # Activity!
            
            process_text(h_evt, 'HOME', self.roster_home)
            process_text(a_evt, 'AWAY', self.roster_away)
            
            self.events.append(row_evts)

    def solve_timeline(self):
        """Pass 2/3: Propagation"""
        
        # Initialize with best guess (Starters)
        # Use first checkpoint if available, else inference
        
        def get_initial_lineup(team_label, roster):
            # Check for cp
            cps = [c for c in self.checkpoints if c[1] == team_label]
            if cps:
                # Backprop from first CP
                idx, _, current = cps[0]
                starter_set = set(current)
                # Walk back from idx to 0
                for r in range(idx-1, -1, -1):
                    evts = self.events[r]
                    if not evts: continue
                    for e in reversed(evts):
                        if e['team'] != team_label: continue
                        if e['type'] == 'IN': starter_set.discard(e['player'])
                        elif e['type'] == 'OUT': starter_set.add(e['player'])
                return starter_set
            else:
                # Inference: Anyone with activity before IN
                starters = set()
                seen_sub = set()
                # Scan forward
                for r_evts in self.events:
                    if not r_evts: continue
                    for e in r_evts:
                        if e['team'] != team_label: continue
                        if e['type'] == 'IN':
                            seen_sub.add(e['player'])
                        elif e['type'] == 'OUT':
                            if e['player'] not in seen_sub: starters.add(e['player'])
                            seen_sub.add(e['player'])
                
                # Fill gaps with most active
                if len(starters) < 5:
                    sorted_roster = [k for k,v in roster.most_common() if k not in starters]
                    for cand in sorted_roster:
                        if len(starters) >= 5: break
                        starters.add(cand)
                return starters

        curr_h = get_initial_lineup('HOME', self.roster_home)
        curr_a = get_initial_lineup('AWAY', self.roster_away)
        
        # Propagate Forward
        for i, evts in enumerate(self.events):
            # Record State BEFORE subs? Or AFTER? 
            # Usually events happen with the lineup at that moment.
            # But substitution lines themselves change the state for the *next* play.
            
            # Let's record state at start of row
            
            # GHOST FILL ENFORCEMENT
            self.ensure_five(curr_h, self.roster_home)
            self.ensure_five(curr_a, self.roster_away)
            
            self.on_floor_history.append((set(curr_h), set(curr_a)))
            
            if evts:
                for e in evts:
                    tgt = curr_h if e['team'] == 'HOME' else curr_a
                    if e['type'] == 'IN': tgt.add(e['player'])
                    elif e['type'] == 'OUT': tgt.discard(e['player'])

    def ensure_five(self, lineup_set, roster_counter):
        """Pass 4: The Ghost Fix"""
        if len(lineup_set) == 5: return
        
        # If > 5, remove least active? (Rare error)
        if len(lineup_set) > 5:
            # Try to remove players who have 'enters' events later? 
            # For now, simplistic: keep most active
            # sort by roster activity count
            sorted_p = sorted(list(lineup_set), key=lambda x: roster_counter[x], reverse=True)
            # Clip to 5
            to_remove = sorted_p[5:]
            for p in to_remove: lineup_set.discard(p)
            return

        # If < 5, ADD most active not in set
        if len(lineup_set) < 5:
            # Candidates: In roster, not in set
            cands = [p for p,v in roster_counter.most_common() if p not in lineup_set]
            for c in cands:
                lineup_set.add(c)
                if len(lineup_set) == 5: break
                
    def export_rows(self):
        output = []
        for i, row_text in enumerate(self.rows):
            h_set, a_set = self.on_floor_history[i]
            
            # Format onFloor JSON
            on_floor = []
            for p in h_set:
                on_floor.append({'id': None, 'name': p, 'team': self.h_team})
            for p in a_set:
                on_floor.append({'id': None, 'name': p, 'team': self.a_team})
            
            # Parse clock/score
            parts = [p.strip() for p in row_text.split("|")]
            clock = parts[0] if len(parts) > 0 else "00:00"
            score = parts[2] if len(parts) > 2 else "0-0"
            if "-" in score:
                hs, as_ = score.split("-")
            else:
                hs, as_ = 0, 0
            
            # Extract date from filename
            # Filename format: ncaa_pbp_YYYY-MM-DD.csv
            match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(self.filename))
            date_str = match.group(0) if match else "1900-01-01"

            output.append({
                "gameSourceId": str(self.game_id),
                "season": self.season,
                "date": date_str,
                "clock": clock,
                "playText": row_text,
                "homeScore": hs,
                "awayScore": as_,
                "onFloor": json.dumps(on_floor)
            })
        return output

def main():
    print("🦊 Starting Full Scale Holistic Cleaner...")
    con = duckdb.connect(DB_PATH)
    # team_df = con.execute("SELECT id, school, displayName FROM dim_teams").fetchdf() # Not used yet
    
    parts_root = Path("data/fact_play_historical_parts")
    parts_root.mkdir(parents=True, exist_ok=True)

    # Dynamic Directory Scan
    base_scrape_dir = "data/manual_scrapes"
    SOURCE_DIRS = []
    
    if os.path.exists(base_scrape_dir):
        # Find all year folders (supports both YYYY and YYYY-YYYY formats)
        for item in os.listdir(base_scrape_dir):
            full_path = os.path.join(base_scrape_dir, item)
            if os.path.isdir(full_path):
                # Check if it's a valid season folder (YYYY or YYYY-YYYY)
                if (item.isdigit() and len(item) == 4) or re.match(r'^\d{4}-\d{4}$', item):
                    SOURCE_DIRS.append(full_path)
    
    SOURCE_DIRS.sort() # Process in order
    print(f"🌍 Found {len(SOURCE_DIRS)} historical seasons to process: {SOURCE_DIRS}")
    
    season_part_paths = []
    total_rows_written = 0

    for folder in SOURCE_DIRS:
        if not os.path.exists(folder): continue
        files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        print(f"📂 Scanning {folder}: {len(files)} files found.")
        season_rows = []

        for f in files:
            path = os.path.join(folder, f)
            try:
                df = pd.read_csv(path)
                contest_ids = df['contest_id'].unique()
                
                for cid in contest_ids:
                    game_rows = df[df['contest_id'] == cid].sort_index()
                    h_team, a_team = get_header_teams(game_rows)
                    
                    # Extract season from folder name
                    folder_name = os.path.basename(folder)
                    if '-' in folder_name:
                        season = int(folder_name.split('-')[0])  # 2012-2013 -> 2012
                    else:
                        season = int(folder_name)  # 2012 -> 2012
                    
                    solver = GameSolver(cid, game_rows['raw_text'].tolist(), h_team, a_team, season)
                    solver.filename = f # Pass filename for date extraction
                    solver.parse_rows()
                    solver.solve_timeline()
                    clean = solver.export_rows()
                    season_rows.extend(clean)

            except Exception as e:
                print(f"    ❌ Error on {f}: {e}")

        if season_rows:
            season_df = pd.DataFrame(season_rows)
            season_df['season'] = season_df['season'].astype(int)
            season_df['homeScore'] = pd.to_numeric(season_df['homeScore'], errors='coerce').fillna(0).astype(int)
            season_df['awayScore'] = pd.to_numeric(season_df['awayScore'], errors='coerce').fillna(0).astype(int)
            season_df['date'] = season_df['date'].astype(str)

            season_year = int(season_df['season'].iloc[0])
            out_part = parts_root / f"season={season_year}.parquet"
            season_df.to_parquet(out_part, index=False)
            season_part_paths.append(str(out_part))
            total_rows_written += len(season_df)
            print(f"   💾 wrote season part {out_part} ({len(season_df)} rows)")

    # Save combined parquet from parts using DuckDB (avoids giant in-memory dataframe).
    out_parquet = Path("data/fact_play_historical_combined.parquet")
    print(f"\n💾 Materializing combined parquet from {len(season_part_paths)} parts -> {out_parquet}...")

    if not season_part_paths:
        print("⚠️ No rows generated. Exiting.")
        return

    con.execute(
        f"""
        COPY (
          SELECT *
          FROM read_parquet('{parts_root.as_posix()}/season=*.parquet')
        )
        TO '{out_parquet.as_posix()}'
        (FORMAT PARQUET);
        """
    )
    con.close()
    print(f"✅ Full Batch Complete! total_rows={total_rows_written}")

if __name__ == "__main__":
    main()
