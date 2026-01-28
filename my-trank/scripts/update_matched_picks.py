#!/usr/bin/env python3
"""
Update CSV files with correct pick numbers for matched players.
"""

import csv
import os
import unicodedata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON_CSV = os.path.join(BASE_DIR, 'public', 'data', 'season.csv')
CAREER_CSV = os.path.join(BASE_DIR, 'public', 'data', 'career.csv')
ARCHIVE_CSV = os.path.join(BASE_DIR, 'public', 'data', 'archive.csv')
INTL_CSV = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history', 'internationalplayerarchive.csv')
NBA_LOOKUP_CSV = os.path.join(BASE_DIR, 'public', 'data', 'nba_lookup.csv')

def normalize_for_matching(name):
    """Aggressive name normalization for fuzzy matching."""
    if not name:
        return ""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    if ',' in name:
        parts = name.split(',', 1)
        if len(parts) == 2:
            name = f"{parts[1].strip()} {parts[0].strip()}"
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    name = name.replace(" Jr", "").replace(" Sr", "").replace(" II", "").replace(" III", "")
    return ' '.join(name.lower().split())

# Hardcoded eligible players with picks from audit
player_picks = {
    # 2019 - HIGH PRIORITY LOTTERY PICKS THAT WERE MISSING
    ("rj barrett", "2019"): 3,
    ("deandre hunter", "2019"): 4,
    ("jarrett culver", "2019"): 6,
    ("coby white", "2019"): 7,
    ("pj washington", "2019"): 12,
    ("romeo langford", "2019"): 14,
    # 2019 - Rest
    ("jaxson hayes", "2019"): 8,
    ("cam reddish", "2019"): 11,
    ("tyler herro", "2019"): 13,
    ("sekou doumbouya", "2019"): 15,
    ("brandon clarke", "2019"): 21,
    ("grant williams", "2019"): 22,
    ("carsen edwards", "2019"): 26,
    ("mfiondu kabengele", "2019"): 27,
    ("kz okpala", "2019"): 29,
    ("keldon johnson", "2019"): 30,
    ("nicolas claxton", "2019"): 31,
    ("isaiah roby", "2019"): 35,
    ("ty jerome", "2019"): 37,
    ("justin james", "2019"): 41,
    ("miye oni", "2019"): 44,
    ("jalen lecque", "2019"): 51,
    ("tremont waters", "2019"): 52,
    ("shamorie ponds", "2019"): 54,
    ("jordan bone", "2019"): 56,
    ("jaylen nowell", "2019"): 57,
    # 2020  
    ("onyeka okongwu", "2020"): 6,
    ("devin vassell", "2020"): 11,
    ("aaron nesmith", "2020"): 14,
    ("tyrese maxey", "2020"): 17,
    ("saddiq bey", "2020"): 19,
    ("tyrell terry", "2020"): 21,
    ("zeke nnaji", "2020"): 22,
    ("immanuel quickley", "2020"): 25,
    ("udoka azubuike", "2020"): 27,
    ("malachi flynn", "2020"): 29,
    ("tre jones", "2020"): 37,
    ("saben lee", "2020"): 38,
    ("tyler bey", "2020"): 39,
    ("tyshon alexander", "2020"): 41,
    ("elijah hughes", "2020"): 42,
    ("cassius winston", "2020"): 43,
    ("jordan nwora", "2020"): 45,
    ("nico mannion", "2020"): 48,
    ("justinian jessup", "2020"): 51,
    ("grant riller", "2020"): 56,
    ("jalen harris", "2020"): 59,
    # 2021
    ("nahshon hyland", "2021"): 26,
    ("cameron thomas", "2021"): 27,
    ("greg brown", "2021"): 43,
    ("balsa koprivica", "2021"): 57,
    # 2022
    ("jabari smith", "2022"): 3,
    # 2024
    ("reed sheppard", "2024"): 3,
    ("stephon castle", "2024"): 4,
    ("donovan clingan", "2024"): 7,
    ("rob dillingham", "2024"): 8,
    ("cody williams", "2024"): 10,
    ("devin carter", "2024"): 13,
    ("carlton carrington", "2024"): 14,
    ("kelel ware", "2024"): 15,
    ("jared mccain", "2024"): 16,
    ("dalton knecht", "2024"): 17,
    ("tristan da silva", "2024"): 18,
    ("jakobe walter", "2024"): 19,
    ("jaylon tyson", "2024"): 20,
    ("yves missi", "2024"): 21,
    ("daron holmes ii", "2024"): 22,
    ("dillon jones", "2024"): 26,
    ("terrence shannon jr", "2024"): 27,
    ("ryan dunn", "2024"): 28,
    ("isaiah collier", "2024"): 29,
    ("baylor scheierman", "2024"): 30,
    ("jonathan mogbo", "2024"): 31,
    ("kyle filipowski", "2024"): 32,
    ("tyler smith", "2024"): 33,
    ("tyler kolek", "2024"):  34,
    ("johnny furphy", "2024"): 35,
    ("ajay mitchell", "2024"): 38,
    ("jaylen wells", "2024"): 39,
    ("oso ighodaro", "2024"): 40,
    ("adem bona", "2024"): 41,
    ("kj simpson", "2024"): 42,
    ("pelle larsson", "2024"): 44,
    ("jamal shead", "2024"): 45,
    ("cam christie", "2024"): 46,
    ("antonio reeves", "2024"): 47,
    ("harrison ingram", "2024"): 48,
    ("tristen newton", "2024"): 49,
    ("enrique freeman", "2024"): 50,
    ("quinten post", "2024"): 52,
    ("cam spencer", "2024"): 53,
    ("anton watson", "2024"): 54,
    ("bronny james", "2024"): 55,
    ("kevin mccullar", "2024"): 56,
    # 2025
    ("cooper flagg", "2025"): 1,
    ("dylan harper", "2025"): 2,
    ("vj edgecombe", "2025"): 3,
    ("kon knueppel", "2025"): 4,
    ("ace bailey", "2025"): 5,
    ("tre johnson", "2025"): 6,
    ("jeremiah fears", "2025"): 7,
    ("egor demin", "2025"): 8,
    ("collin murrayboyles", "2025"): 9,
    ("khaman maluach", "2025"): 10,
    ("cedric coward", "2025"): 11,
    ("derik queen", "2025"): 13,
    ("carter bryant", "2025"): 14,
    ("thomas sorber", "2025"): 15,
    ("walter clayton jr", "2025"): 18,
    ("kasparas jakucionis", "2025"): 20,
    ("will riley", "2025"): 21,
    ("drake powell", "2025"): 22,
    ("asa newell", "2025"): 23,
    ("nique clifford", "2025"): 24,
    ("jase richardson", "2025"): 25,
    ("danny wolf", "2025"): 27,
    ("liam mcneeley", "2025"): 29,
    ("rasheer fleming", "2025"): 31,
    ("sion james", "2025"): 33,
    ("ryan kalkbrenner", "2025"): 34,
    ("johni broome", "2025"): 35,
    ("adou thiero", "2025"): 36,
    ("chaz lanier", "2025"): 37,
    ("kam jones", "2025"): 38,
    ("alijah martin", "2025"): 39,
    ("micah peavy", "2025"): 40,
    ("koby brea", "2025"): 41,
    ("maxime raynaud", "2025"): 42,
    ("jamir watkins", "2025"): 43,
    ("brooks barnhizer", "2025"): 44,
    ("amari williams", "2025"): 46,
    ("javon small", "2025"): 48,
    ("tyrese proctor", "2025"): 49,
    ("kobe sanders", "2025"): 50,
    ("john tonje", "2025"): 53,
    ("taelon peter", "2025"): 54,
    ("will richard", "2025"): 56,
    ("max shulga", "2025"): 57,
    ("jahmai mashack", "2025"): 59,
}

def update_file(file_path, name_col='key'):
    """Update pick column in a CSV file."""
    temp_path = file_path + '.tmp'
    updated_count = 0

    with open(file_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        rows = list(reader)

    with open(temp_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            name = row.get(name_col, '').strip()
            year = str(row.get('torvik_year', '')).strip()
            
            name_norm = normalize_for_matching(name)
            key = (name_norm, year)
            
            if key in player_picks:
                new_pick = player_picks[key]
                if str(row.get('pick', '')) != str(new_pick):
                    row['pick'] = new_pick
                    updated_count += 1
            
            writer.writerow(row)
            
    os.replace(temp_path, file_path)
    return updated_count

# Update all files
print("Updating CSV files with correct picks...")
total_updated = 0

for file_path in [SEASON_CSV, CAREER_CSV, ARCHIVE_CSV, INTL_CSV]:
    count = update_file(file_path)
    print(f"  {os.path.basename(file_path)}: {count} updates")
    total_updated += count

print(f"\nTotal updates: {total_updated}")

# Also update nba_lookup
print("\nUpdating nba_lookup.csv...")
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
        name = row.get('name_lower', '').strip()
        draft_year = row.get('draft_year', '').strip()
        
        name_norm = normalize_for_matching(name)
        key = (name_norm, draft_year)
        
        if key in player_picks:
            new_pick = float(player_picks[key])
            if row.get('pick', '') != str(new_pick):
                row['pick'] = str(new_pick)
                row['status'] = 'drafted'
                updated_count += 1
        
        writer.writerow(row)

os.replace(temp_path, NBA_LOOKUP_CSV)
print(f"  nba_lookup.csv: {updated_count} updates")

print("\n✓ All files updated successfully")
