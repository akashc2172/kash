
import pandas as pd
import glob
import os
import re

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history')
TARGET_FILE = os.path.join(DATA_DIR, 'internationalplayerarchive.csv')

def parse_height(h_str):
    """Parses height string like '6-2 (188cm)' to inches."""
    if pd.isna(h_str) or not isinstance(h_str, str):
        return None
    # Try to match feet-inches format first
    match_ft = re.search(r'(\d+)-(\d+)', h_str)
    if match_ft:
        feet = int(match_ft.group(1))
        inches = int(match_ft.group(2))
        return feet * 12 + inches
    # Fallback to cm if needed, but the format seems consistent
    return None

def parse_weight(w_str):
    """Parses weight string like '200 lbs (91kg)' to lbs."""
    if pd.isna(w_str) or not isinstance(w_str, str):
        return None
    match_lbs = re.search(r'(\d+)\s*lbs', w_str)
    if match_lbs:
        return int(match_lbs.group(1))
    return None

def main():
    print(f"Loading target archive: {TARGET_FILE}")
    try:
        archive_df = pd.read_csv(TARGET_FILE)
        archive_cols = list(archive_df.columns)
    except FileNotFoundError:
        print("Archive file not found. Creating new schema based on requirements.")
        # This acts as a fallback or if we were starting fresh, but we expect the file to exist
        return

    # Find LNB Espoirs files
    files = glob.glob(os.path.join(DATA_DIR, 'LNB_Espoirs_*.csv'))
    print(f"Found {len(files)} LNB Espoirs files to process.")

    new_rows = []

    for f_path in files:
        print(f"Processing {os.path.basename(f_path)}...")
        df = pd.read_csv(f_path)
        
        # Determine year from filename or column (file content has 'Year' column)
        # We will use the 'Year' column if available, else filename
        
        for _, row in df.iterrows():
            # Create a dictionary for the new row with keys matching archive columns
            new_row = {col: None for col in archive_cols}
            
            # Direct mapping
            new_row['key'] = row.get('Player')
            new_row['team'] = row.get('Team')
            new_row['conf'] = '(INTL) FRA ESP' # Mapped as requested
            new_row['g'] = row.get('GP')
            new_row['min'] = row.get('MPG') * row.get('GP') if pd.notna(row.get('MPG')) and pd.notna(row.get('GP')) else None
            new_row['mpg'] = row.get('MPG')
            new_row['ppg'] = row.get('PPG')
            
            # Shooting
            new_row['fgm'] = row.get('FGM')
            new_row['fga'] = row.get('FGA')
            new_row['fg_pct'] = row.get('FG%')
            new_row['three_m'] = row.get('3PM')
            new_row['three_a'] = row.get('3PA')
            new_row['3p%'] = row.get('3P%')
            new_row['ftm'] = row.get('FTM')
            new_row['fta'] = row.get('FTA')
            new_row['ft_pct'] = row.get('FT%')
            
            # Derived shooting
            if pd.notna(new_row['fga']) and pd.notna(new_row['three_a']):
                new_row['2pa'] = new_row['fga'] - new_row['three_a']
            if pd.notna(new_row['fgm']) and pd.notna(new_row['three_m']):
                new_row['two_m'] = new_row['fgm'] - new_row['three_m']
            
            if pd.notna(new_row.get('two_m')) and pd.notna(new_row.get('2pa')) and new_row['2pa'] > 0:
                 new_row['2p%'] = new_row['two_m'] / new_row['2pa']
            
            # Rebounding & Other Stats
            new_row['oreb'] = row.get('ORB')
            new_row['dreb'] = row.get('DRB')
            new_row['rpg'] = row.get('RPG')
            new_row['apg'] = row.get('APG')
            new_row['spg'] = row.get('SPG')
            new_row['bpg'] = row.get('BPG')
            new_row['tov'] = row.get('TOV')
            new_row['pfr'] = row.get('PF')
            
            # Advanced / Rates
            # archive expects mapped rates often in specific formats, assumed to be same scale (usually 0-100 or 0-1)
            # LNB file has 'ORB%', 'DRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%'
            # Archive has 'oreb_rate', 'dreb_rate', 'usg' (usually whole numbers in archive based on preview)
            # Checked archive: 'usg' e.g. 23.2. LNB 'USG%' e.g. 23.0. Matches.
            
            new_row['oreb_rate'] = row.get('ORB%')
            new_row['dreb_rate'] = row.get('DRB%')
            new_row['usg'] = row.get('USG%')
            new_row['ortg'] = row.get('ORtg')
            new_row['drtg'] = row.get('DRtg')
            
            # Percentages ast/tov/blk/stl might need mapping if columns exist in archive
            # Archive has: 'ast' (total?), 'to' (total?), 'blk' (total?), 'stl' (total?)
            # Archive also has 'stls/100', 'blks/100' etc in some rows.
            # We'll map standard per game stats.
            
            # Metadata
            new_row['torvik_id'] = row.get('Player Link')
            new_row['torvik_year'] = row.get('Year')
            
            # Bios
            new_row['hoop_hgt_in'] = parse_height(row.get('Height'))
            new_row['weight'] = parse_weight(row.get('Weight'))
            
            # FTR
            if pd.notna(new_row['fta']) and pd.notna(new_row['fga']) and new_row['fga'] > 0:
                new_row['ftr'] = new_row['fta'] / new_row['fga']
            
            # TS/eFG - LNB has TS%, eFG%. Archive has 'ts', 'efg'.
            new_row['ts'] = row.get('TS%')
            new_row['efg'] = row.get('eFG%')
            
            new_rows.append(new_row)

    if not new_rows:
        print("No new rows generated.")
        return

    new_df = pd.DataFrame(new_rows)
    
    # Concatenate
    print(f"Appending {len(new_df)} rows to archive...")
    final_df = pd.concat([archive_df, new_df], ignore_index=True)
    
    # Save
    final_df.to_csv(TARGET_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
