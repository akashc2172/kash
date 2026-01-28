import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re

def format_name(name):
    if not name or not isinstance(name, str): return name
    if ',' in name:
        parts = [p.strip() for p in name.split(',')]
        if len(parts) >= 2:
            return f"{parts[1]} {parts[0]}"
    return name

def get_bbref_stats(raw_name):
    player_name = format_name(raw_name)
    name_lower = player_name.lower().strip()
    
    # Construct possible URLs
    # Pattern: /players/x/lastfi01.html
    # last: first 5 chars of last name
    # fi: first 2 chars of first name
    
    name_parts = name_lower.split()
    if len(name_parts) < 2:
        return None
        
    first = name_parts[0]
    last = name_parts[-1]
    
    # Remove juniors, seniors, III, etc from last name if they are tokens
    suffixes = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
    if last in suffixes and len(name_parts) > 2:
        last = name_parts[-2]
    
    fi = re.sub(r'[^a-zA-Z]', '', first)[:2]
    la = re.sub(r'[^a-zA-Z]', '', last)[:5]
    initial = la[0]
    
    stats = None
    
    # Try 01 to 02
    for i in range(1, 3):
        player_id = f"{la}{fi}0{i}"
        url = f"https://www.basketball-reference.com/players/{initial}/{player_id}.html"
        print(f"Trying {url} for {player_name}...")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Verify name
            h1 = soup.find('h1')
            if h1:
                found_name = h1.get_text().strip().lower()
                # Simple loose match
                clean_player_name = re.sub(r'[^a-z ]', '', name_lower)
                clean_found_name = re.sub(r'[^a-z ]', '', found_name)
                
                parts1 = set(clean_player_name.split())
                parts2 = set(clean_found_name.split())
                
                # If they share at least 2 significant name parts
                overlap = parts1.intersection(parts2)
                if len(overlap) < 2 and len(parts1) >= 2:
                    print(f"  Name mismatch: Found '{found_name}', expected '{name_lower}'")
                    continue
            
            # Find Advanced Stats table
            advanced_table = soup.find('table', id='advanced')
            if not advanced_table:
                import bs4
                comments = soup.find_all(string=lambda text: isinstance(text, bs4.Comment))
                for comment in comments:
                    if 'id="advanced"' in comment:
                        c_soup = BeautifulSoup(comment, 'html.parser')
                        advanced_table = c_soup.find('table', id='advanced')
                        break
            
            if advanced_table:
                footer = advanced_table.find('tfoot')
                career_row = None
                if footer:
                    career_row = footer.find('tr')
                else:
                    rows = advanced_table.find_all('tr')
                    for row in rows:
                        th = row.find('th')
                        if th and 'career' in th.get_text().lower():
                            career_row = row
                            break
                
                if career_row:
                    cells = career_row.find_all(['td', 'th'])
                    row_data = {}
                    for cell in cells:
                        stat_name = cell.get('data-stat')
                        if stat_name:
                            row_data[stat_name] = cell.get_text().strip()
                    
                    print(f"  Successfully found stats for {player_name}")
                    stats = row_data
                    break
                    
        except Exception as e:
            print(f"  Error scraping {url}: {e}")
            
        time.sleep(1)
        
    return stats

def scrape_nba_stats():
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    LOOKUP_CSV = os.path.join(BASE_DIR, 'data', 'nba_player_lookup.csv')
    
    if not os.path.exists(LOOKUP_CSV):
        print(f"{LOOKUP_CSV} not found. Run identify_nba_players.py first.")
        return
        
    df = pd.read_csv(LOOKUP_CSV)
    
    results = []
    
    # Filter to only drafted or undrafted players that actually have a name
    df = df[df['name'].notna()]
    
    print(f"Starting scrape for {len(df)} players...")
    
    for idx, row in df.iterrows():
        player_stats = get_bbref_stats(row['name'])
        if player_stats:
            player_stats['player_name'] = row['name']
            player_stats['player_name_lower'] = row['name'].lower()
            results.append(player_stats)
        
        if len(results) % 10 == 0 and len(results) > 0:
            pd.DataFrame(results).to_csv(os.path.join(BASE_DIR, 'data', 'nba_career_advanced_stats_temp.csv'), index=False)
            
    final_df = pd.DataFrame(results)
    output_path = os.path.join(BASE_DIR, 'data', 'nba_career_advanced_stats.csv')
    final_df.to_csv(output_path, index=False)
    print(f"Scraping complete. Saved {len(final_df)} players to {output_path}")

if __name__ == "__main__":
    scrape_nba_stats()
