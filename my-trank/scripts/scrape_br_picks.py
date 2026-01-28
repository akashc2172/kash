#!/usr/bin/env python3
"""
Scrape NBA draft picks from Basketball Reference (2008-2025).
This is the authoritative source for draft pick numbers.
"""

import requests
import csv
import os
import time
from bs4 import BeautifulSoup
import unicodedata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'basketball_reference_picks.csv')

def normalize_name(name):
    if not name:
        return ""
    # Normalize unicode characters to ASCII for matching
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # Remove extra spaces and make lower
    return name.lower().strip()

def scrape_draft_year(year):
    """Scrape all draft picks for a given year from Basketball Reference."""
    url = f"https://www.basketball-reference.com/draft/NBA_{year}.html"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching {year}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the stats table
    table = soup.find('table', {'id': 'stats'})
    if not table:
        print(f"  No draft table found for {year}")
        return []
    
    picks = []
    tbody = table.find('tbody')
    if not tbody:
        return []
    
    for row in tbody.find_all('tr'):
        # Skip header rows
        if row.get('class') and 'thead' in row.get('class'):
            continue
        
        cells = row.find_all(['th', 'td'])
        if len(cells) < 4:
            continue
        
        # Get pick number from first cell
        pick_cell = cells[0]
        pick_num = pick_cell.get_text(strip=True)
        
        # Skip if not a valid pick number
        if not pick_num.isdigit():
            continue
        
        # Get player name and link
        player_cell = cells[3]  # Player column
        player_link = player_cell.find('a')
        
        if player_link:
            player_name = player_link.get_text(strip=True)
            profile_url = "https://www.basketball-reference.com" + player_link.get('href', '')
        else:
            player_name = player_cell.get_text(strip=True)
            profile_url = ''
        
        if player_name:
            picks.append({
                'year': year,
                'pick': int(pick_num),
                'player_name': player_name,
                'player_name_lower': normalize_name(player_name),
                'profile_url': profile_url
            })
    
    return picks

def main():
    all_picks = []
    
    # Scrape 2008-2025
    for year in range(2008, 2026):
        print(f"Scraping {year}...")
        picks = scrape_draft_year(year)
        all_picks.extend(picks)
        print(f"  Found {len(picks)} picks")
        time.sleep(3)  # Be respectful to the server
    
    # Write to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['year', 'pick', 'player_name', 'player_name_lower', 'profile_url'])
        writer.writeheader()
        writer.writerows(all_picks)
    
    print(f"\nTotal picks scraped: {len(all_picks)}")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
