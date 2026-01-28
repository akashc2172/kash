#!/usr/bin/env python3
"""
Fix pick data propagation: 
- Use Tankathon as source of truth for (player_name, draft_year) -> pick
- For each player, propagate their pick to ALL their seasons
- This ensures all seasons of a drafted player show the correct pick

The logic:
1. Load Tankathon picks to get the definitive (player, draft_year) -> pick mapping
2. For each data file, group rows by player name
3. For each player, find their draft info from Tankathon
4. Apply that pick to ALL their seasons (or NA if undrafted)
"""

import csv
import os
import unicodedata
import re

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TANKATHON_CSV = os.path.join(BASE_DIR, 'data', 'tankathon_draft_picks.csv')
INTL_CSV = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history', 'internationalplayerarchive.csv')
INTL_2026_CSV = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history', '2026_records.csv')
SEASON_CSV = os.path.join(BASE_DIR, 'public', 'data', 'season.csv')
ARCHIVE_CSV = os.path.join(BASE_DIR, 'public', 'data', 'archive.csv')
CAREER_CSV = os.path.join(BASE_DIR, 'public', 'data', 'career.csv')
NBA_LOOKUP_CSV = os.path.join(BASE_DIR, 'public', 'data', 'nba_lookup.csv')

def normalize_name(name):
    if not name:
        return ""
    # Normalize unicode characters to ASCII for matching
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # Remove quotes
    name = name.replace('"', '').replace("'", "")
    # Convert "Last, First" to "First Last" format
    if ',' in name:
        parts = name.split(',', 1)
        if len(parts) == 2:
            last = parts[0].strip()
            first = parts[1].strip()
            name = f"{first} {last}"
    # Remove extra spaces and make lower
    name = ' '.join(name.lower().split())
    return name

def load_tankathon_picks():
    """Load picks returning:
    - picks_by_name: {normalized_name: (pick, draft_year)} - for players who were drafted
    """
    picks_by_name = {}
    
    if not os.path.exists(TANKATHON_CSV):
        print(f"Error: {TANKATHON_CSV} not found.")
        return picks_by_name
    
    with open(TANKATHON_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_lower = row['player_name_lower']
            year = int(row['year'])
            pick = int(row['pick'])
            
            # Store latest draft year for each player (in case of multiple entries)
            if name_lower not in picks_by_name or year > picks_by_name[name_lower][1]:
                picks_by_name[name_lower] = (pick, year)
    
    return picks_by_name

def fix_file_picks(file_path, tankathon_picks, name_col='key', year_col='torvik_year', is_2026=False):
    """Fix pick values in a CSV file by propagating correct picks to all seasons."""
    if not os.path.exists(file_path):
        print(f"  Skipping {os.path.basename(file_path)} (not found)")
        return 0

    temp_path = file_path + '.tmp'
    updated_count = 0

    with open(file_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        if 'pick' not in fieldnames:
            print(f"  Skipping {os.path.basename(file_path)} (no 'pick' column)")
            return 0
            
        rows = list(reader)

    with open(temp_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            name_raw = row.get(name_col) or ''
            name_norm = normalize_name(name_raw)
            row_year = row.get(year_col)
            
            try:
                row_year = int(row_year) if row_year else 0
            except ValueError:
                row_year = 0
            
            # 2026 players shouldn't have picks yet
            if row_year == 2026 or is_2026:
                if row.get('pick', '') not in ('NA', ''):
                    row['pick'] = 'NA'
                    updated_count += 1
            elif name_norm in tankathon_picks:
                # Player was drafted - get their pick info
                pick, draft_year = tankathon_picks[name_norm]
                
                # Set pick for all seasons (the pick is a career achievement)
                old_pick = row.get('pick', '')
                if str(old_pick) != str(pick):
                    row['pick'] = pick
                    updated_count += 1
            else:
                # Player not in Tankathon = undrafted
                old_pick = row.get('pick', '')
                if old_pick not in ('NA', '', 'nan') and old_pick is not None:
                    # Clear incorrect pick values for undrafted players
                    row['pick'] = 'NA'
                    updated_count += 1
            
            writer.writerow(row)
            
    os.replace(temp_path, file_path)
    return updated_count

def fix_nba_lookup(tankathon_picks):
    """Fix nba_lookup.csv with correct picks."""
    if not os.path.exists(NBA_LOOKUP_CSV):
        print(f"  Skipping nba_lookup.csv (not found)")
        return 0
    
    temp_path = NBA_LOOKUP_CSV + '.tmp'
    updated_count = 0
    
    with open(NBA_LOOKUP_CSV, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    with open(temp_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            name_lower = row.get('name_lower', '')
            name_norm = normalize_name(name_lower)
            
            if name_norm in tankathon_picks:
                pick, draft_year = tankathon_picks[name_norm]
                old_pick = row.get('pick', '')
                
                try:
                    if float(old_pick) != float(pick):
                        row['pick'] = float(pick)
                        row['status'] = 'drafted'
                        updated_count += 1
                except (ValueError, TypeError):
                    row['pick'] = float(pick)
                    row['status'] = 'drafted'
                    updated_count += 1
            
            writer.writerow(row)
    
    os.replace(temp_path, NBA_LOOKUP_CSV)
    return updated_count

def main():
    print("Loading Tankathon draft picks...")
    tankathon_picks = load_tankathon_picks()
    print(f"  Loaded {len(tankathon_picks)} unique drafted players\n")
    
    total_updated = 0
    
    # 1. International archive
    print("Fixing internationalplayerarchive.csv...")
    u = fix_file_picks(INTL_CSV, tankathon_picks)
    print(f"  Updated: {u}")
    total_updated += u
    
    # 2. 2026 records
    print("Fixing 2026_records.csv...")
    u = fix_file_picks(INTL_2026_CSV, tankathon_picks, is_2026=True)
    print(f"  Updated: {u}")
    total_updated += u
    
    # 3. Season data
    print("Fixing season.csv...")
    u = fix_file_picks(SEASON_CSV, tankathon_picks)
    print(f"  Updated: {u}")
    total_updated += u
    
    # 4. Archive data
    print("Fixing archive.csv...")
    u = fix_file_picks(ARCHIVE_CSV, tankathon_picks)
    print(f"  Updated: {u}")
    total_updated += u
    
    # 5. Career data
    print("Fixing career.csv...")
    u = fix_file_picks(CAREER_CSV, tankathon_picks, year_col=None)
    print(f"  Updated: {u}")
    total_updated += u
    
    # 6. NBA Lookup
    print("Fixing nba_lookup.csv...")
    u = fix_nba_lookup(tankathon_picks)
    print(f"  Updated: {u}")
    total_updated += u
    
    print(f"\n=== SUMMARY ===")
    print(f"Total picks fixed: {total_updated}")

if __name__ == '__main__':
    main()
