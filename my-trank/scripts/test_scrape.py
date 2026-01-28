import requests
from bs4 import BeautifulSoup
import re
import time

def format_name(name):
    if not name or not isinstance(name, str): return name
    if ',' in name:
        parts = [p.strip() for p in name.split(',')]
        if len(parts) >= 2:
            return f"{parts[1]} {parts[0]}"
    return name

def test_scrape_player(raw_name):
    player_name = format_name(raw_name)
    name_lower = player_name.lower().strip()
    
    name_parts = name_lower.split()
    if len(name_parts) < 2:
        return None
        
    first = name_parts[0]
    last = name_parts[-1]
    
    fi = re.sub(r'[^a-zA-Z]', '', first)[:2]
    la = re.sub(r'[^a-zA-Z]', '', last)[:5]
    initial = la[0]
    
    for i in range(1, 3):
        player_id = f"{la}{fi}0{i}"
        url = f"https://www.basketball-reference.com/players/{initial}/{player_id}.html"
        print(f"Trying {url} for {player_name}...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            print(f"  Response Code: {response.status_code}")
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Verify name
            h1 = soup.find('h1')
            if h1:
                found_name = h1.get_text().strip().lower()
                print(f"  Found Page Name: {found_name}")
            
            # Find Advanced Stats table
            advanced_table = soup.find('table', id='advanced')
            if not advanced_table:
                print("  Advanced table not found in main DOM, checking comments...")
                # BBRef often hides tables in comments to save load time
                import bs4
                comments = soup.find_all(string=lambda text: isinstance(text, bs4.Comment))
                for comment in comments:
                    if 'id="advanced"' in comment:
                        print("  Found 'advanced' table in a comment!")
                        c_soup = BeautifulSoup(comment, 'html.parser')
                        advanced_table = c_soup.find('table', id='advanced')
                        break
            
            if advanced_table:
                # Find the 'Career' row
                # Often it's in <tfoot>
                footer = advanced_table.find('tfoot')
                career_row = None
                if footer:
                    career_row = footer.find('tr')
                    print("  Found career row in tfoot")
                else:
                    rows = advanced_table.find_all('tr')
                    for row in rows:
                        th = row.find('th')
                        if th and 'career' in th.get_text().lower():
                            career_row = row
                            print("  Found career row in tbody/thead")
                            break
                
                if career_row:
                    cells = career_row.find_all(['td', 'th'])
                    row_data = {}
                    for cell in cells:
                        stat_name = cell.get('data-stat')
                        if stat_name:
                            row_data[stat_name] = cell.get_text().strip()
                    print(f"  STATS: {row_data}")
                    return row_data
            else:
                print("  Advanced table NOT found anywhere on page.")
                    
        except Exception as e:
            print(f"  Error: {e}")
            
        time.sleep(2)
    return None

if __name__ == "__main__":
    test_scrape_player("Johnson, Cameron")
