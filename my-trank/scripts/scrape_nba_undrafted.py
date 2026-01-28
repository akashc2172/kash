import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re

def scrape_undrafted_players():
    url = "https://en.wikipedia.org/wiki/List_of_undrafted_NBA_players"
    print(f"Scraping {url}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch Wikipedia page: {response.status_code}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    undrafted_players = []
    
    # The page has multiple tables for different years/categories
    tables = soup.find_all('table', class_='wikitable')
    print(f"Found {len(tables)} wikitables")
    
    for table_idx, table in enumerate(tables):
        rows = table.find_all('tr')
        if not rows: continue
        
        # Identify headers
        header_row = rows[0]
        header_cells = header_row.find_all(['th', 'td'])
        headers = [h.get_text().strip().lower() for h in header_cells]
        
        if 'player' not in headers:
            continue
            
        player_idx = headers.index('player')
        first_col_is_draft = headers[0] in ['draft', 'year first played']
        
        print(f"Processing table {table_idx} with headers {headers}")
        
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if not cells: continue
            
            target_idx = player_idx
            # Handle rowspans: if the first column (Draft) is missing in this row,
            # the Player column will be at index 0 (if player_idx was 1).
            if first_col_is_draft and len(cells) < len(headers):
                target_idx = player_idx - 1
            
            if 0 <= target_idx < len(cells):
                name = cells[target_idx].get_text().strip()
                name = re.sub(r'\[.*?\]', '', name) # Remove citations
                name = name.split('\n')[0].strip() # Handle multiline
                if name and len(name) > 1 and not name.lower().startswith('draft'):
                    undrafted_players.append(name)
    
    # Deduplicate
    undrafted_players = sorted(list(set(undrafted_players)))
    print(f"Found {len(undrafted_players)} unique undrafted players")
    
    # Save to CSV
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    output_path = os.path.join(data_dir, 'undrafted_nba_players.csv')
    df = pd.DataFrame(undrafted_players, columns=['player_name'])
    df['player_name_lower'] = df['player_name'].str.lower()
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    scrape_undrafted_players()
