#!/usr/bin/env python3
"""
Find and fix name mismatches for eligible drafted players.
Check for apostrophe, hyphen, Jr., spacing differences.
"""

import csv
import os
import unicodedata
import difflib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TANKATHON_CSV = os.path.join(BASE_DIR, 'data', 'tankathon_draft_picks.csv')
SEASON_CSV = os.path.join(BASE_DIR, 'public', 'data', 'season.csv')
ARCHIVE_CSV = os.path.join(BASE_DIR, 'public', 'data', 'archive.csv')
INTL_CSV = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history', 'internationalplayerarchive.csv')
NBA_LOOKUP_CSV = os.path.join(BASE_DIR, 'public', 'data', 'nba_lookup.csv')

def normalize_for_matching(name):
    """Aggressive name normalization for fuzzy matching."""
    if not name:
        return ""
    # Remove unicode accents
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # Convert "Last, First" to "first last"
    if ',' in name:
        parts = name.split(',', 1)
        if len(parts) == 2:
            name = f"{parts[1].strip()} {parts[0].strip()}"
    # Remove apostrophes, hyphens, periods, Jr/Sr/III etc
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    name = name.replace(" Jr", "").replace(" Sr", "").replace(" II", "").replace(" III", "")
    # lowercase and strip extra spaces
    return ' '.join(name.lower().split())

# Load tank picks we need to fix (eligible only)
eligible_players_2019_2025 = [
    # 2019
    (2019, 8, "Jaxson Hayes"),
    (2019, 11, "Cam Reddish"),
    (2019, 13, "Tyler Herro"),
    (2019, 15, "Sekou Doumbouya"),
    (2019, 21, "Brandon Clarke"),
    (2019, 22, "Grant Williams"),
    (2019, 23, "Darius Bazley"),
    (2019, 24, "Darius Bazley"),
    (2019, 26, "Carsen Edwards"),
    (2019, 27, "Mfiondu Kabengele"),
    (2019, 29, "KZ Okpala"),
    (2019, 30, "Keldon Johnson"),
    (2019, 31, "Nicolas Claxton"),
    (2019, 35, "Isaiah Roby"),
    (2019, 37, "Ty Jerome"),
    (2019, 40, "Scottie Lewis"),
    (2019, 41, "Justin James"),
    (2019, 44, "Miye Oni"),
    (2019, 45, "Carsen Edwards"),
    (2019, 51, "Jalen Lecque"),
    (2019, 52, "Tremont Waters"),
    (2019, 54, "Shamorie Ponds"),
    (2019, 56, "Jordan Bone"),
    (2019, 57, "Jaylen Nowell"),
    # 2020 (skipping most as many are G-League)
    (2020, 6, "Onyeka Okongwu"),
    (2020, 11, "Devin Vassell"),
    (2020, 14, "Aaron Nesmith"),
    (2020, 17, "Tyrese Maxey"),
    (2020, 19, "Saddiq Bey"),
    (2020, 21, "Tyrell Terry"),
    (2020, 22, "Zeke Nnaji"),
    (2020, 24, "RJ Hampton"),
    (2020, 25, "Immanuel Quickley"),
    (2020, 27, "Udoka Azubuike"),
    (2020, 29, "Malachi Flynn"),
    (2020, 34, "Theo Maledon"),
    (2020, 37, "Tre Jones"),
    (2020, 38, "Saben Lee"),
    (2020, 39, "Tyler Bey"),
    (2020, 41, "Ty-Shon Alexander"),
    (2020, 42, "Elijah Hughes"),
    (2020, 43, "Cassius Winston"),
    (2020, 45, "Jordan Nwora"),
    (2020, 47, "Yam Madar"),
    (2020, 48, "Nico Mannion"),
    (2020, 51, "Justinian Jessup"),
    (2020, 56, "Grant Riller"),
    (2020, 59, "Jalen Harris"),
    # 2021
    (2021, 26, "Nah'Shon Hyland"),
    (2021, 27, "Cameron Thomas"),
    (2021, 43, "Greg Brown"),
    (2021, 57, "Balsa Koprivica"),
    # 2022
    (2022, 3, "Jabari Smith"),
    # 2024 - MANY
    (2024, 3, "Reed Sheppard"),
    (2024, 4, "Stephon Castle"),
    (2024, 7, "Donovan Clingan"),
    (2024, 8, "Rob Dillingham"),
    (2024, 10, "Cody Williams"),
    (2024, 13, "Devin Carter"),
    (2024, 14, "Carlton Carrington"),
    (2024, 15, "Kel'el Ware"),
    (2024, 16, "Jared McCain"),
    (2024, 17, "Dalton Knecht"),
    (2024, 18, "Tristan da Silva"),
    (2024, 19, "Ja'Kobe Walter"),
    (2024, 20, "Jaylon Tyson"),
    (2024, 21, "Yves Missi"),
    (2024, 22, "DaRon Holmes II"),
    (2024, 26, "Dillon Jones"),
    (2024, 27, "Terrence Shannon Jr."),
    (2024, 28, "Ryan Dunn"),
    (2024, 29, "Isaiah Collier"),
    (2024, 30, "Baylor Scheierman"),
    (2024, 31, "Jonathan Mogbo"),
    (2024, 32, "Kyle Filipowski"),
    (2024, 33, "Tyler Smith"),
    (2024, 34, "Tyler Kolek"),
    (2024, 35, "Johnny Furphy"),
    (2024, 38, "Ajay Mitchell"),
    (2024, 39, "Jaylen Wells"),
    (2024, 40, "Oso Ighodaro"),
    (2024, 41, "Adem Bona"),
    (2024, 42, "KJ Simpson"),
    (2024, 44, "Pelle Larsson"),
    (2024, 45, "Jamal Shead"),
    (2024, 46, "Cam Christie"),
    (2024, 47, "Antonio Reeves"),
    (2024, 48, "Harrison Ingram"),
    (2024, 49, "Tristen Newton"),
    (2024, 50, "Enrique Freeman"),
    (2024, 52, "Quinten Post"),
    (2024, 53, "Cam Spencer"),
    (2024, 54, "Anton Watson"),
    (2024, 55, "Bronny James"),
    (2024, 56, "Kevin McCullar"),
    # 2025 - MANY (future draft, most projections)
    (2025, 1, "Cooper Flagg"),
    (2025, 2, "Dylan Harper"),
    (2025, 3, "V.J. Edgecombe"),
    (2025, 4, "Kon Knueppel"),
    (2025, 5, "Ace Bailey"),
    (2025, 6, "Tre Johnson"),
    (2025, 7, "Jeremiah Fears"),
    (2025, 8, "Egor Demin"),
    (2025, 9, "Collin Murray-Boyles"),
    (2025, 10, "Khaman Maluach"),
    (2025, 11, "Cedric Coward"),
    (2025, 13, "Derik Queen"),
    (2025, 14, "Carter Bryant"),
    (2025, 15, "Thomas Sorber"),
    (2025, 18, "Walter Clayton Jr."),
    (2025, 20, "Kasparas Jakucionis"),
    (2025, 21, "Will Riley"),
    (2025, 22, "Drake Powell"),
    (2025, 23, "Asa Newell"),
    (2025, 24, "Nique Clifford"),
    (2025, 25, "Jase Richardson"),
    (2025, 27, "Danny Wolf"),
    (2025, 29, "Liam McNeeley"),
    (2025, 31, "Rasheer Fleming"),
    (2025, 33, "Sion James"),
    (2025, 34, "Ryan Kalkbrenner"),
    (2025, 35, "Johni Broome"),
    (2025, 36, "Adou Thiero"),
    (2025, 37, "Chaz Lanier"),
    (2025, 38, "Kam Jones"),
    (2025, 39, "Alijah Martin"),
    (2025, 40, "Micah Peavy"),
    (2025, 41, "Koby Brea"),
    (2025, 42, "Maxime Raynaud"),
    (2025, 43, "Jamir Watkins"),
    (2025, 44, "Brooks Barnhizer"),
    (2025, 46, "Amari Williams"),
    (2025, 48, "Javon Small"),
    (2025, 49, "Tyrese Proctor"),
    (2025, 50, "Kobe Sanders"),
    (2025, 53, "John Tonje"),
    (2025, 54, "Taelon Peter"),
    (2025, 56, "Will Richard"),
    (2025, 57, "Max Shulga"),
    (2025, 59, "Jahmai Mashack"),
]

# Load all players from databases
print("Loading player databases...")
all_players = {}

for csv_file in [SEASON_CSV, ARCHIVE_CSV, INTL_CSV]:
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('key', '').strip() if csv_file != INTL_CSV else row.get('key', '').strip()
            if name:
                name_norm = normalize_for_matching(name)
                if name_norm not in all_players:
                    all_players[name_norm] = []
                all_players[name_norm].append({
                    'original_name': name,
                    'file': csv_file,
                    'year': row.get('torvik_year', '')
                })

print(f"Loaded {len(all_players)} unique normalized names\n")

# Try to match each eligible player
print("="*80)
print("FINDING NAME MATCHES")
print("="*80 + "\n")

matches_found = []
no_match = []

for year, pick, player_name in eligible_players_2019_2025:
    player_norm = normalize_for_matching(player_name)
    
    if player_norm in all_players:
        # Exact match (after normalization)
        for entry in all_players[player_norm]:
            matches_found.append({
                'year': year,
                'pick': pick,
                'tankathon_name': player_name,
                'db_name': entry['original_name'],
                'file': entry['file'],
                'db_year': entry['year'],
                'match_type': 'EXACT'
            })
    else:
        # Try fuzzy match
        close_matches = difflib.get_close_matches(player_norm, all_players.keys(), n=3, cutoff=0.8)
        if close_matches:
            best_match = close_matches[0]
            for entry in all_players[best_match]:
                matches_found.append({
                    'year': year,
                    'pick': pick,
                    'tankathon_name': player_name,
                    'db_name': entry['original_name'],
                    'file': entry['file'],
                    'db_year': entry['year'],
                    'match_type': f'FUZZY ({best_match})'
                })
        else:
            no_match.append((year, pick, player_name))

print(f"✓ Found matches for {len(set((m['year'], m['pick']) for m in matches_found))} players")
print(f"✗ No match for {len(no_match)} players\n")

if matches_found:
    print("MATCHES FOUND:")
    for m in matches_found[:20]:  # Show first 20
        print(f"  {m['year']} Pick #{m['pick']:2d}: {m['tankathon_name']:30s} -> {m['db_name']:30s} [{m['match_type']}]")
    if len(matches_found) > 20:
        print(f"  ... and {len(matches_found) - 20} more")

if no_match:
    print("\nNO MATCH:")
    for year, pick, name in no_match:
        print(f"  {year} Pick #{pick:2d}: {name}")

print("\n" + "="*80)
print(f"Total to update: {len(matches_found)} player-season combinations")
