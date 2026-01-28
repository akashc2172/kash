#!/usr/bin/env python3
"""
Audit missing draft picks from 2019-2025.
Find which picks are missing, identify if players are eligible (played NCAA/International),
and fix name matching issues.
"""

import csv
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TANKATHON_CSV = os.path.join(BASE_DIR, 'data', 'tankathon_draft_picks.csv')
NBA_LOOKUP_CSV = os.path.join(BASE_DIR, 'public', 'data', 'nba_lookup.csv')
SEASON_CSV = os.path.join(BASE_DIR, 'public', 'data', 'season.csv')
ARCHIVE_CSV = os.path.join(BASE_DIR, 'public', 'data', 'archive.csv')
INTL_CSV = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history', 'internationalplayerarchive.csv')

def normalize_name(name):
    """Normalize name for comparison."""
    if not name:
        return ""
    name = name.lower().strip()
    # Convert "Last, First" to "first last"
    if ',' in name:
        parts = name.split(',', 1)
        if len(parts) == 2:
            name = f"{parts[1].strip()} {parts[0].strip()}"
    return name

# Load Tankathon picks
print("Loading Tankathon draft data...")
tankathon_picks = {}
with open(TANKATHON_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        year = int(row['year'])
        pick = int(row['pick'])
        if year not in tankathon_picks:
            tankathon_picks[year] = {}
        tankathon_picks[year][pick] = {
            'name': row['player_name'],
            'name_lower': row['player_name_lower']
        }

# Load NBA Lookup
print("Loading NBA lookup data...")
nba_lookup_picks = defaultdict(set)
with open(NBA_LOOKUP_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('pick') and row['pick'] not in ('NA', ''):
            try:
                pick = int(float(row['pick']))
                draft_year = row.get('draft_year')
                if draft_year:
                    nba_lookup_picks[int(draft_year)].add(pick)
            except (ValueError, TypeError):
                pass

# Load NCAA/International players
print("Loading player databases...")
ncaa_players = set()
intl_players = set()

# Season data
with open(SEASON_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = normalize_name(row.get('key', ''))
        if name:
            ncaa_players.add(name)

# Archive data
with open(ARCHIVE_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = normalize_name(row.get('key', ''))
        if name:
            ncaa_players.add(name)

# International data
with open(INTL_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = normalize_name(row.get('key', ''))
        if name:
            intl_players.add(name)

# Analyze missing picks for each year
print("\n" + "="*80)
print("MISSING DRAFT PICKS ANALYSIS (2019-2025)")
print("="*80)

for year in range(2019, 2026):
    print(f"\n### {year} NBA Draft ###")
    
    if year not in tankathon_picks:
        print(f"  No Tankathon data for {year}")
        continue
    
    missing_picks = []
    for pick in range(1, 61):
        if pick not in nba_lookup_picks.get(year, set()):
            if pick in tankathon_picks[year]:
                player_name = tankathon_picks[year][pick]['name']
                player_name_lower = tankathon_picks[year][pick]['name_lower']
                
                # Check if player is in our databases
                in_ncaa = player_name_lower in ncaa_players
                in_intl = player_name_lower in intl_players
                
                status = "✓ ELIGIBLE" if (in_ncaa or in_intl) else "✗ NOT IN DB"
                
                missing_picks.append({
                    'pick': pick,
                    'name': player_name,
                    'name_lower': player_name_lower,
                    'in_ncaa': in_ncaa,
                    'in_intl': in_intl,
                    'status': status
                })
    
    if missing_picks:
        print(f"  Missing {len(missing_picks)} picks:")
        for item in missing_picks:
            location = "NCAA" if item['in_ncaa'] else ("INTL" if item['in_intl'] else "N/A")
            print(f"    Pick #{item['pick']:2d}: {item['name']:30s} [{item['status']}] {location}")
    else:
        print(f"  ✓ All picks 1-60 present in database")

print("\n" + "="*80)
