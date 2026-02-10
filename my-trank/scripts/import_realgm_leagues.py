"""
Import RealGM League Data into Archive
======================================

Generic importer for all RealGM league CSV files into internationalplayerarchive.csv.
Handles multiple leagues with appropriate conference mappings.
"""

import pandas as pd
import glob
import os
import re

# Define paths - use public/data to match other scripts and website
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history')
TARGET_FILE = os.path.join(DATA_DIR, 'internationalplayerarchive.csv')

# League to conference mapping
LEAGUE_CONF_MAP = {
    'French_Jeep_Elite': '(INTL) FRA',
    'Euroleague': '(INTL) EUR',
    'French_LNB_Espoirs': '(INTL) FRA ESP',
    'French_LNB_Pro_B': '(INTL) FRA PRO B',
    'Australian_NBL': '(INTL) AUS',
    'NBL_Blitz': '(INTL) AUS BLITZ',
    'Spanish_ACB': '(INTL) ESP',
    'Turkish_BSL': '(INTL) TUR',
    'Eurocup': '(INTL) EUR',
    'Adriatic_League_Liga_ABA': '(INTL) ABA',
}

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
    return None

def parse_weight(w_str):
    """Parses weight string like '200 lbs (91kg)' to lbs."""
    if pd.isna(w_str) or not isinstance(w_str, str):
        return None
    match_lbs = re.search(r'(\d+)\s*lbs', w_str)
    if match_lbs:
        return int(match_lbs.group(1))
    return None

def detect_league_from_filename(filename):
    """Extract league name from filename like 'RealGM_French_Jeep_Elite_2026.csv'."""
    basename = os.path.basename(filename)
    # Pattern: RealGM_LeagueName_Year.csv
    match = re.search(r'RealGM_(.+?)_\d{4}', basename)
    if match:
        return match.group(1)
    return None

def main():
    print(f"Loading target archive: {TARGET_FILE}")
    try:
        archive_df = pd.read_csv(TARGET_FILE)
        # Ensure PER column exists
        if 'per' not in archive_df.columns:
            archive_df['per'] = None
        archive_cols = list(archive_df.columns)
        print(f"  Archive has {len(archive_df)} existing rows")
    except FileNotFoundError:
        print("⚠️  Archive file not found. Creating new schema.")
        # Would need to create schema, but for now we expect it to exist
        return

    # Find all RealGM CSV files
    pattern = os.path.join(DATA_DIR, 'RealGM_*.csv')
    files = glob.glob(pattern)
    print(f"\nFound {len(files)} RealGM files to process:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    if not files:
        print("No RealGM files found. Run realgm_scrape.py first.")
        return

    new_rows = []

    for f_path in files:
        filename = os.path.basename(f_path)
        print(f"\nProcessing {filename}...")
        
        # Detect league from filename
        league_name = detect_league_from_filename(f_path)
        if not league_name:
            print(f"  ⚠️  Could not detect league from filename. Skipping.")
            continue
        
        # Get conference mapping
        conf = LEAGUE_CONF_MAP.get(league_name)
        if not conf:
            print(f"  ⚠️  No conference mapping for '{league_name}'. Using '(INTL) UNK'")
            conf = '(INTL) UNK'
        
        print(f"  League: {league_name} -> Conference: {conf}")
        
        try:
            df = pd.read_csv(f_path)
            print(f"  Loaded {len(df)} rows")
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
            continue
        
        # Determine year from filename or column
        year = None
        if 'Year' in df.columns:
            year = df['Year'].iloc[0] if len(df) > 0 else None
        else:
            # Extract from filename
            match = re.search(r'_(\d{4})\.csv$', filename)
            if match:
                year = int(match.group(1))
        
        if not year:
            print(f"  ⚠️  Could not determine year. Skipping.")
            continue
        
        print(f"  Year: {year}")
        
        for idx, row in df.iterrows():
            # Create a dictionary for the new row with keys matching archive columns
            new_row = {col: None for col in archive_cols}
            
            # Direct mapping
            new_row['key'] = row.get('Player')
            new_row['team'] = row.get('Team')
            new_row['conf'] = conf
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
            new_row['oreb_rate'] = row.get('ORB%')
            new_row['dreb_rate'] = row.get('DRB%')
            new_row['usg'] = row.get('USG%')
            new_row['ortg'] = row.get('ORtg')
            new_row['drtg'] = row.get('DRtg')
            new_row['per'] = row.get('PER')
            
            # Metadata
            new_row['torvik_id'] = row.get('Player Link')
            new_row['torvik_year'] = year
            
            # Bios
            new_row['hoop_hgt_in'] = parse_height(row.get('Height'))
            new_row['weight'] = parse_weight(row.get('Weight'))
            
            # FTR
            if pd.notna(new_row['fta']) and pd.notna(new_row['fga']) and new_row['fga'] > 0:
                new_row['ftr'] = new_row['fta'] / new_row['fga']
            
            # TS/eFG
            new_row['ts'] = row.get('TS%')
            new_row['efg'] = row.get('eFG%')
            
            new_rows.append(new_row)

    if not new_rows:
        print("\n⚠️  No new rows generated.")
        return

    new_df = pd.DataFrame(new_rows)
    print(f"\n✅ Generated {len(new_df)} new rows")
    
    # Upsert by (key, team, year, conf) so we can refresh PER and other fields
    print("\nUpserting rows (key, team, year, conf)...")
    if 'key' in archive_df.columns and 'torvik_year' in archive_df.columns:
        archive_df['__key__'] = list(zip(
            archive_df['key'].fillna(''),
            archive_df['team'].fillna(''),
            archive_df['torvik_year'].fillna(0).astype(int),
            archive_df['conf'].fillna('')
        ))
        new_df['__key__'] = list(zip(
            new_df['key'].fillna(''),
            new_df['team'].fillna(''),
            new_df['torvik_year'].fillna(0).astype(int),
            new_df['conf'].fillna('')
        ))

        updates = 0
        adds = 0
        for _, row in new_df.iterrows():
            k = row['__key__']
            matches = archive_df[archive_df['__key__'] == k]
            if not matches.empty:
                idx = matches.index[0]
                for col in archive_cols:
                    if col in row and pd.notna(row[col]):
                        archive_df.at[idx, col] = row[col]
                updates += 1
            else:
                row_to_add = row.drop(labels='__key__')
                archive_df = pd.concat([archive_df, pd.DataFrame([row_to_add])], ignore_index=True)
                adds += 1

        archive_df = archive_df.drop(columns=['__key__'])
        final_df = archive_df
        print(f"  ✅ Updated {updates} rows, added {adds} new rows.")
    else:
        print("  ⚠️  Missing 'key' or 'torvik_year' columns; falling back to append.")
        final_df = pd.concat([archive_df, new_df], ignore_index=True)

    # Final dedupe for 2026 on torvik_id (keep row with most non-null values)
    if 'torvik_id' in final_df.columns and 'torvik_year' in final_df.columns:
        df2026 = final_df[final_df['torvik_year'] == 2026].copy()
        other = final_df[final_df['torvik_year'] != 2026].copy()

        df2026['__key__'] = df2026.apply(
            lambda r: f"{r.get('torvik_id')}_{r.get('torvik_year')}" if pd.notna(r.get('torvik_id')) else None,
            axis=1
        )

        def best_row(group):
            return group.loc[group.apply(lambda r: r.notna().sum(), axis=1).idxmax()]

        deduped = []
        for k, g in df2026.groupby('__key__', dropna=True):
            if k is None:
                continue
            if len(g) > 1:
                deduped.append(best_row(g))
            else:
                deduped.append(g.iloc[0])
        no_id = df2026[df2026['__key__'].isna()]
        df2026 = pd.concat([pd.DataFrame(deduped), no_id], ignore_index=True)
        df2026 = df2026.drop(columns=['__key__'])

        final_df = pd.concat([other, df2026], ignore_index=True)
    
    # Save
    final_df.to_csv(TARGET_FILE, index=False)
    print(f"✅ Done! Archive now has {len(final_df)} total rows.")
    print(f"   Saved to: {TARGET_FILE}")

if __name__ == "__main__":
    main()
