#!/usr/bin/env python3
"""
Integrate Basketball Reference profile URLs into nba_lookup.csv.
Also creates a consolidated drafted players file with all info.
"""

import csv
import os
import unicodedata
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BR_PROFILES_CSV = os.path.join(BASE_DIR, 'data', 'br_profile_urls.csv')
NBA_LOOKUP_CSV = os.path.join(BASE_DIR, 'public', 'data', 'nba_lookup.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'public', 'data', 'nba_lookup.csv')

def normalize_name(name):
    if not name:
        return ""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.replace('"', '').replace("'", "")
    if ',' in name:
        parts = name.split(',', 1)
        if len(parts) == 2:
            last = parts[0].strip()
            first = parts[1].strip()
            name = f"{first} {last}"
    name = ' '.join(name.lower().split())
    return name

def load_br_profiles():
    """Load BR profiles keyed by normalized name."""
    profiles = {}
    
    with open(BR_PROFILES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_norm = normalize_name(row['player_name'])
            profiles[name_norm] = {
                'pick': int(row['pick']),
                'draft_year': int(row['year']),
                'profile_url': row['profile_url']
            }
    
    return profiles

def main():
    print("Loading BR profile URLs...")
    br_profiles = load_br_profiles()
    print(f"  Loaded {len(br_profiles)} profiles\n")
    
    # Read existing nba_lookup
    print("Updating nba_lookup.csv with BR profile URLs...")
    
    with open(NBA_LOOKUP_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        
        # Add new columns if needed
        if 'br_profile_url' not in fieldnames:
            fieldnames.append('br_profile_url')
        if 'draft_year' not in fieldnames:
            fieldnames.append('draft_year')
            
        rows = list(reader)
    
    updated_count = 0
    for row in rows:
        name_norm = normalize_name(row.get('name', ''))
        
        if name_norm in br_profiles:
            profile = br_profiles[name_norm]
            row['br_profile_url'] = profile['profile_url']
            row['draft_year'] = profile['draft_year']
            
            # Also ensure pick is correct
            if row.get('status') == 'drafted':
                row['pick'] = float(profile['pick'])
            
            updated_count += 1
    
    # Write updated file
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  Updated {updated_count} players with BR profile URLs")
    print(f"  Saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
