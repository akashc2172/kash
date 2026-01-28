#!/usr/bin/env python3
"""
Scrape NBA draft pick data from Tankathon to fix incorrect pick numbers
in international player CSV files.

The issue: Round picks were recorded as just the pick number within the round
(e.g., Round 2 Pick 15 was recorded as "15" instead of "45").

Tankathon displays picks 1-60 sequentially on the same page.
"""

import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import re

# Years to scrape (2008-2025)
YEARS = list(range(2008, 2026))  # 2008-2025

# Output file
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'tankathon_draft_picks.csv')


def scrape_draft_year(year: int) -> list[dict]:
    """Scrape all picks for a given draft year from Tankathon."""
    url = f"https://www.tankathon.com/past_drafts/{year}"
    print(f"Scraping {url}...")
    
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching {year}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    picks = []
    
    # Find all mock-row elements
    for row in soup.select('.mock-row'):
        # Get pick number
        pick_number_el = row.select_one('.mock-row-pick-number')
        if not pick_number_el:
            continue
            
        # Extract only the first text node (the pick number)
        # Avoid movement numbers in child divs
        pick_text = ""
        for child in pick_number_el.children:
            if isinstance(child, str):
                pick_text = child.strip()
                if pick_text:
                    break
        
        # Fallback if first text node is empty
        if not pick_text:
            pick_text = re.sub(r'[^0-9].*', '', pick_number_el.get_text(strip=True))
        
        if not pick_text.isdigit():
            continue
            
        pick_num = int(pick_text)
        
        # Get player name
        name_el = row.select_one('.mock-row-name')
        if not name_el:
            continue
            
        player_name = name_el.get_text(strip=True)
        if not player_name:
            continue
            
        # Normalize name: remove accents, extra spaces, etc.
        import unicodedata
        def normalize_name(name):
            name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
            # Remove positions like PF, PG if they somehow got in
            # But with .mock-row-name, they should be excluded
            return name.strip()

        player_name_normalized = normalize_name(player_name)
        
        picks.append({
            'year': year,
            'pick': pick_num,
            'player_name': player_name_normalized,
            'player_name_lower': player_name_normalized.lower()
        })
    
    # Remove duplicates (sometimes Tankathon has weird rows)
    unique_picks = []
    seen = set()
    for p in picks:
        key = (p['year'], p['pick'])
        if key not in seen:
            seen.add(key)
            unique_picks.append(p)
    
    print(f"  Found {len(unique_picks)} unique picks for {year}")
    return unique_picks


def main():
    all_picks = []
    
    for year in YEARS:
        picks = scrape_draft_year(year)
        all_picks.extend(picks)
        time.sleep(1)  # Be polite to the server
    
    # Write to CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['year', 'pick', 'player_name', 'player_name_lower'])
        writer.writeheader()
        writer.writerows(all_picks)
    
    print(f"\nWrote {len(all_picks)} picks to {OUTPUT_FILE}")
    
    # Print sample of round 2 picks to verify
    print("\nSample Round 2 picks (31-60):")
    round2 = [p for p in all_picks if p['pick'] >= 31][:10]
    for p in round2:
        print(f"  {p['year']} Pick {p['pick']}: {p['player_name']}")


if __name__ == '__main__':
    main()
