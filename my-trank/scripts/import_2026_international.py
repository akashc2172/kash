import pandas as pd
import glob
import os
import re

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history')
TARGET_FILE = os.path.join(DATA_DIR, '2026_records.csv')

# Mapping for 2026_records.csv specific conference names
LEAGUE_CONF_MAP = {
    'French_Jeep_Elite': '(INTL) FRENCH JEEP',
    'Euroleague': '(INTL) EUROLEAGUE',
    'French_LNB_Espoirs': '(INTL) FRENCH LNB',
    'French_LNB_Pro_B': '(INTL) FRENCH PRO B',
    'Australian_NBL': '(INTL) NBL',
    'NBL_Blitz': '(INTL) NBL BLITZ',
    'Spanish_ACB': '(INTL) ACB',
    'Turkish_BSL': '(INTL) TURK. BSL',
    'Eurocup': '(INTL) EUROCUP',
    'Adriatic_League_Liga_ABA': '(INTL) ABA',
}

def parse_height(h_str):
    if pd.isna(h_str) or not isinstance(h_str, str): return None
    match = re.search(r'(\d+)-(\d+)', h_str)
    if match:
        return int(match.group(1)) * 12 + int(match.group(2))
    return None

def parse_weight(w_str):
    if pd.isna(w_str) or not isinstance(w_str, str): return None
    match = re.search(r'(\d+)\s*lbs', w_str)
    if match: return int(match.group(1))
    return None

def main():
    print(f"Loading target: {TARGET_FILE}")
    try:
        target_df = pd.read_csv(TARGET_FILE)
        # Ensure PER column exists
        if 'per' not in target_df.columns:
            target_df['per'] = None
        target_cols = list(target_df.columns)
        print(f"  Existing records: {len(target_df)}")
    except FileNotFoundError:
        print("⚠️  Target not found. Exiting.")
        return

    pattern = os.path.join(DATA_DIR, 'RealGM_*.csv')
    files = glob.glob(pattern)
    # If both full and AvgOnly exist for a league/year, prefer full
    full_set = set(f for f in files if not f.endswith('_AvgOnly.csv'))
    filtered = []
    for f in files:
        if f.endswith('_AvgOnly.csv'):
            base = f.replace('_AvgOnly.csv', '.csv')
            if base in full_set:
                continue
        filtered.append(f)
    files = filtered
    print(f"Found {len(files)} RealGM files.")
    
    if not files:
        print("No RealGM files found. Run realgm_scrape.py first.")
        return

    all_new_rows = []
    
    for f_path in files:
        filename = os.path.basename(f_path)
        print(f"Processing {filename}...")
        
        # Extract league name from RealGM_League_Name_2026.csv
        match = re.search(r'RealGM_(.+)_2026\.csv', filename)
        if not match: continue
        league_id_name = match.group(1)
        
        # Match against our mapping
        conf = LEAGUE_CONF_MAP.get(league_id_name)
        if not conf: conf = f"(INTL) {league_id_name.upper().replace('-', ' ')}"

        df = pd.read_csv(f_path)
        
        for _, row in df.iterrows():
            # Basic mapping
            new_row = {col: None for col in target_cols}
            
            new_row['key'] = row.get('Player')
            new_row['conf'] = conf
            new_row['team'] = row.get('Team')
            
            # Stat mapping
            gp = row.get('GP', 0)
            mpg = row.get('MPG', 0)
            
            new_row['g'] = gp
            new_row['mpg'] = mpg
            new_row['min'] = gp * mpg
            new_row['total minutes'] = gp * mpg
            new_row['ppg'] = row.get('PPG')
            
            new_row['fgm'] = row.get('FGM')
            new_row['fga'] = row.get('FGA')
            new_row['fg_pct'] = row.get('FG%')
            new_row['three_m'] = row.get('3PM')
            new_row['three_a'] = row.get('3PA')
            new_row['3p%'] = row.get('3P%')
            new_row['ftm'] = row.get('FTM')
            new_row['fta'] = row.get('FTA')
            new_row['ft_pct'] = row.get('FT%')
            
            # Derived
            if pd.notna(new_row['fga']) and pd.notna(new_row['three_a']):
                new_row['two_a'] = new_row['fga'] - new_row['three_a']
                new_row['2pa'] = new_row['two_a']
            if pd.notna(new_row['fgm']) and pd.notna(new_row['three_m']):
                new_row['two_m'] = new_row['fgm'] - new_row['three_m']
            
            if pd.notna(new_row.get('two_m')) and pd.notna(new_row.get('2pa')) and new_row['2pa'] > 0:
                new_row['2p%'] = new_row['two_m'] / new_row['2pa']
            
            if pd.notna(new_row['apg']) and pd.notna(new_row['tov']) and new_row['tov'] > 0:
                new_row['ast_to'] = new_row['apg'] / new_row['tov']
            
            if pd.notna(new_row['fta']) and pd.notna(new_row['fga']) and new_row['fga'] > 0:
                new_row['ftr'] = new_row['fta'] / new_row['fga']
                
            new_row['oreb'] = row.get('ORB')
            new_row['dreb'] = row.get('DRB')
            new_row['rpg'] = row.get('RPG')
            new_row['apg'] = row.get('APG')
            new_row['spg'] = row.get('SPG')
            new_row['bpg'] = row.get('BPG')
            new_row['tov'] = row.get('TOV')
            new_row['pfr'] = row.get('PF')
            
            # Advanced
            new_row['usg'] = row.get('USG%')
            new_row['ortg'] = row.get('ORtg')
            new_row['drtg'] = row.get('DRtg')
            new_row['ts'] = row.get('TS%')
            new_row['efg'] = row.get('eFG%')
            new_row['per'] = row.get('PER')
            
            new_row['oreb_rate'] = row.get('ORB%')
            new_row['dreb_rate'] = row.get('DRB%')
            new_row['ast'] = row.get('AST%') 
            new_row['to'] = row.get('TOV%')
            new_row['blk'] = row.get('BLK%')
            new_row['stl'] = row.get('STL%')

            # Metadata
            new_row['torvik_id'] = row.get('Player Link')
            new_row['torvik_year'] = 2026
            
            # Bio
            new_row['hoop_hgt_in'] = parse_height(row.get('Height'))
            new_row['weight'] = parse_weight(row.get('Weight'))
            
            all_new_rows.append(new_row)

    if not all_new_rows:
        print("No rows to import.")
        return

    new_df = pd.DataFrame(all_new_rows)
    print(f"Total scrapped rows: {len(new_df)}")

    # Deduplicate new rows by torvik_id+year when possible (prefer rows with more non-null data)
    def row_score(r):
        return r.notna().sum()

    new_df['__dedupe_key__'] = new_df.apply(
        lambda r: f"{r.get('torvik_id')}_{r.get('torvik_year')}"
        if pd.notna(r.get('torvik_id'))
        else f"{r.get('key')}_{r.get('team')}_{r.get('conf')}_{r.get('torvik_year')}",
        axis=1
    )
    new_df = new_df.loc[new_df.groupby('__dedupe_key__').apply(lambda g: g.loc[g.apply(row_score, axis=1).idxmax()]).reset_index(drop=True).index]
    new_df = new_df.drop(columns=['__dedupe_key__'])

    # Standardize player names for merging
    target_df['key_upper'] = target_df['key'].str.upper()
    new_df['key_upper'] = new_df['key'].str.upper()
    
    updated_count = 0
    newly_added_count = 0
    
    # We'll use a copy to avoid fragmentation warnings if possible, but simple is fine
    for idx, new_row in new_df.iterrows():
        # Prefer matching on torvik_id + year when available
        matches = pd.DataFrame()
        if pd.notna(new_row.get('torvik_id')) and 'torvik_id' in target_df.columns:
            matches = target_df[(target_df['torvik_id'] == new_row['torvik_id']) & (target_df['torvik_year'] == new_row['torvik_year'])]

        if matches.empty:
            # Match on name + team
            matches = target_df[(target_df['key_upper'] == new_row['key_upper']) & (target_df['team'] == new_row['team'])]

        if matches.empty:
            # Fallback: name-only match
            matches = target_df[target_df['key_upper'] == new_row['key_upper']]

        if not matches.empty:
            target_idx = matches.index[0]
            for col in new_row.index:
                if col != 'key_upper' and pd.notna(new_row[col]):
                    target_df.at[target_idx, col] = new_row[col]
            updated_count += 1
        else:
            # Truly new player
            row_to_add = new_row.drop('key_upper')
            target_df = pd.concat([target_df, pd.DataFrame([row_to_add])], ignore_index=True)
            newly_added_count += 1
            
    if 'key_upper' in target_df.columns:
        target_df = target_df.drop(columns=['key_upper'])

    # Final dedupe for 2026 on torvik_id (keep row with most non-null values)
    if 'torvik_id' in target_df.columns and 'torvik_year' in target_df.columns:
        target_df['__key__'] = target_df.apply(
            lambda r: f"{r.get('torvik_id')}_{r.get('torvik_year')}" if pd.notna(r.get('torvik_id')) else None,
            axis=1
        )
        def keep_best(group):
            return group.loc[group.apply(lambda r: r.notna().sum(), axis=1).idxmax()]
        deduped = []
        for k, g in target_df.groupby('__key__', dropna=True):
            if k is None:
                continue
            if len(g) > 1 and (g['torvik_year'] == 2026).any():
                deduped.append(keep_best(g))
            else:
                deduped.append(g.iloc[0])
        # add rows without torvik_id
        no_id = target_df[target_df['__key__'].isna()]
        target_df = pd.concat([pd.DataFrame(deduped), no_id], ignore_index=True)
        target_df = target_df.drop(columns=['__key__'])
    
    print(f"Updated {updated_count} existing records.")
    print(f"Added {newly_added_count} new records.")
    
    target_df.to_csv(TARGET_FILE, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()
