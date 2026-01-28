#!/usr/bin/env python3
"""
Update pick data across ALL data files using Tankathon data.
This script updates:
- international_stat_history/internationalplayerarchive.csv
- international_stat_history/2026_records.csv
- season.csv
- archive.csv  
- career.csv
- nba_lookup.csv
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
    # Remove common prefixes/suffixes that might differ
    name = re.sub(r'^(dr\.|mr\.|ms\.)\s*', '', name)
    return name

def load_tankathon_picks():
    """Load picks keyed by (normalized_name, year) and also by just name for career data."""
    picks_by_year = {}
    picks_by_name = {}  # For career.csv (no year column)
    
    if not os.path.exists(TANKATHON_CSV):
        print(f"Error: {TANKATHON_CSV} not found.")
        return picks_by_year, picks_by_name
    
    with open(TANKATHON_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_lower = row['player_name_lower']
            year = row['year']
            pick = row['pick']
            
            # Store by (name, year) for season/archive data
            picks_by_year[(name_lower, year)] = pick
            
            # Store by name only for career data (use most recent year's pick)
            # Overwrite to get the latest draft year's pick for each player
            picks_by_name[name_lower] = pick
    
    return picks_by_year, picks_by_name

def update_file(file_path, tankathon_picks_by_year, tankathon_picks_by_name, name_col='key', year_col='torvik_year', is_career=False, is_2026=False):
    """Update pick column in a CSV file."""
    if not os.path.exists(file_path):
        print(f"  Skipping {os.path.basename(file_path)} (not found)")
        return 0, 0

    temp_path = file_path + '.tmp'
    updated_count = 0
    cleared_count = 0

    with open(file_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        if 'pick' not in fieldnames:
            print(f"  Skipping {os.path.basename(file_path)} (no 'pick' column)")
            return 0, 0
            
        rows = list(reader)

    with open(temp_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            original_pick = row.get('pick', '')
            year = str(row.get(year_col) or '')
            name_raw = row.get(name_col) or ''
            name_norm = normalize_name(name_raw)
            
            # Ensure 2026 records don't have picks
            if year == '2026' or is_2026:
                if original_pick and original_pick not in ('NA', '', 'nan'):
                    row['pick'] = 'NA'
                    cleared_count += 1
            else:
                # Try to find pick in Tankathon data
                new_pick = None
                
                if is_career:
                    # For career data, use name-only lookup
                    if name_norm in tankathon_picks_by_name:
                        new_pick = tankathon_picks_by_name[name_norm]
                else:
                    # For season/archive data, use (name, year) lookup
                    search_key = (name_norm, year)
                    if search_key in tankathon_picks_by_year:
                        new_pick = tankathon_picks_by_year[search_key]
                
                if new_pick and str(row.get('pick', '')) != str(new_pick):
                    row['pick'] = new_pick
                    updated_count += 1
            
            writer.writerow(row)
            
    os.replace(temp_path, file_path)
    return updated_count, cleared_count

def update_nba_lookup(tankathon_picks_by_name):
    """Update nba_lookup.csv with Tankathon data."""
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
            
            if name_norm in tankathon_picks_by_name:
                new_pick = float(tankathon_picks_by_name[name_norm])
                old_pick = row.get('pick', '')
                
                # Don't update if already NA (undrafted)
                if old_pick not in ('NA', ''):
                    try:
                        if float(old_pick) != new_pick:
                            row['pick'] = new_pick
                            row['status'] = 'drafted'
                            updated_count += 1
                    except ValueError:
                        row['pick'] = new_pick
                        row['status'] = 'drafted'
                        updated_count += 1
            
            writer.writerow(row)
    
    os.replace(temp_path, NBA_LOOKUP_CSV)
    return updated_count

def main():
    print("Loading Tankathon draft picks...")
    picks_by_year, picks_by_name = load_tankathon_picks()
    print(f"  Loaded {len(picks_by_year)} year-specific picks and {len(picks_by_name)} unique players\n")
    
    total_updated = 0
    total_cleared = 0
    
    # 1. International archive
    print("Updating internationalplayerarchive.csv...")
    u, c = update_file(INTL_CSV, picks_by_year, picks_by_name)
    print(f"  Updated: {u}, Cleared: {c}")
    total_updated += u
    total_cleared += c
    
    # 2. 2026 records
    print("Updating 2026_records.csv...")
    u, c = update_file(INTL_2026_CSV, picks_by_year, picks_by_name, is_2026=True)
    print(f"  Updated: {u}, Cleared: {c}")
    total_updated += u
    total_cleared += c
    
    # 3. Season data
    print("Updating season.csv...")
    u, c = update_file(SEASON_CSV, picks_by_year, picks_by_name)
    print(f"  Updated: {u}, Cleared: {c}")
    total_updated += u
    total_cleared += c
    
    # 4. Archive data
    print("Updating archive.csv...")
    u, c = update_file(ARCHIVE_CSV, picks_by_year, picks_by_name)
    print(f"  Updated: {u}, Cleared: {c}")
    total_updated += u
    total_cleared += c
    
    # 5. Career data
    print("Updating career.csv...")
    u, c = update_file(CAREER_CSV, picks_by_year, picks_by_name, is_career=True)
    print(f"  Updated: {u}, Cleared: {c}")
    total_updated += u
    total_cleared += c
    
    # 6. NBA Lookup
    print("Updating nba_lookup.csv...")
    u = update_nba_lookup(picks_by_name)
    print(f"  Updated: {u}")
    total_updated += u
    
    print(f"\n=== SUMMARY ===")
    print(f"Total picks updated: {total_updated}")
    print(f"Total 2026 picks cleared: {total_cleared}")

if __name__ == '__main__':
    main()
