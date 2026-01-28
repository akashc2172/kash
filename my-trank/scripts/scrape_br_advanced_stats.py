#!/usr/bin/env python3
"""
Scrape advanced career stats from Basketball Reference player profiles.
Extracts: AST%, STL%, BLK%, BPM, and other advanced metrics.
"""

import csv
import os
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BR_PROFILES_CSV = os.path.join(BASE_DIR, 'data', 'br_profile_urls.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'br_advanced_stats.csv')

def setup_driver():
    """Setup Chrome driver with options to avoid detection."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
    
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def scrape_player_stats(driver, profile_url, player_name, pick, year):
    """Scrape advanced stats from a player's BR profile."""
    try:
        driver.get(profile_url)
        time.sleep(2)
        
        stats = {
            'year': year,
            'pick': pick,
            'player_name': player_name,
            'profile_url': profile_url,
            'career_ast_pct': None,
            'career_stl_pct': None,
            'career_blk_pct': None,
            'career_bpm': None,
            'career_obpm': None,
            'career_dbpm': None,
            'career_vorp': None,
            'career_ws': None,
            'career_ws48': None,
            'career_per': None,
            'career_ts_pct': None,
            'career_usg_pct': None,
        }
        
        # Try to find the advanced stats table
        try:
            # Look for career totals in the advanced table
            advanced_table = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, "advanced"))
            )
            
            # Find the career row (tfoot)
            career_row = advanced_table.find_element(By.CSS_SELECTOR, "tfoot tr")
            cells = career_row.find_elements(By.TAG_NAME, "td")
            
            # Parse the cells - BR advanced table structure:
            # Age, G, MP, PER, TS%, 3PAr, FTr, ORB%, DRB%, TRB%, AST%, STL%, BLK%, TOV%, USG%, OWS, DWS, WS, WS/48, OBPM, DBPM, BPM, VORP
            if len(cells) >= 23:
                def safe_float(val):
                    try:
                        return float(val) if val and val.strip() else None
                    except:
                        return None
                
                stats['career_per'] = safe_float(cells[2].text)
                stats['career_ts_pct'] = safe_float(cells[3].text)
                stats['career_ast_pct'] = safe_float(cells[9].text)
                stats['career_stl_pct'] = safe_float(cells[10].text)
                stats['career_blk_pct'] = safe_float(cells[11].text)
                stats['career_usg_pct'] = safe_float(cells[13].text)
                stats['career_ws'] = safe_float(cells[16].text)
                stats['career_ws48'] = safe_float(cells[17].text)
                stats['career_obpm'] = safe_float(cells[18].text)
                stats['career_dbpm'] = safe_float(cells[19].text)
                stats['career_bpm'] = safe_float(cells[20].text)
                stats['career_vorp'] = safe_float(cells[21].text)
                
        except (TimeoutException, NoSuchElementException):
            # Player might not have NBA stats yet (rookies, international)
            pass
        
        return stats
        
    except Exception as e:
        print(f"    Error scraping {player_name}: {e}")
        return {
            'year': year,
            'pick': pick,
            'player_name': player_name,
            'profile_url': profile_url,
            'career_ast_pct': None,
            'career_stl_pct': None,
            'career_blk_pct': None,
            'career_bpm': None,
            'career_obpm': None,
            'career_dbpm': None,
            'career_vorp': None,
            'career_ws': None,
            'career_ws48': None,
            'career_per': None,
            'career_ts_pct': None,
            'career_usg_pct': None,
        }

def main():
    # Load profiles
    print("Loading BR profile URLs...")
    profiles = []
    with open(BR_PROFILES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        profiles = list(reader)
    
    print(f"  Loaded {len(profiles)} profiles to scrape\n")
    
    all_stats = []
    
    print("Setting up Chrome driver...")
    driver = setup_driver()
    
    try:
        for i, profile in enumerate(profiles):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(profiles)} players scraped...")
            
            stats = scrape_player_stats(
                driver,
                profile['profile_url'],
                profile['player_name'],
                profile['pick'],
                profile['year']
            )
            all_stats.append(stats)
            
            # Rate limiting
            time.sleep(3)
            
    finally:
        driver.quit()
    
    # Write to CSV
    fieldnames = [
        'year', 'pick', 'player_name', 'profile_url',
        'career_ast_pct', 'career_stl_pct', 'career_blk_pct',
        'career_bpm', 'career_obpm', 'career_dbpm', 'career_vorp',
        'career_ws', 'career_ws48', 'career_per', 'career_ts_pct', 'career_usg_pct'
    ]
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)
    
    # Count successful scrapes
    success_count = sum(1 for s in all_stats if s['career_bpm'] is not None)
    
    print(f"\nTotal players processed: {len(all_stats)}")
    print(f"Players with career stats: {success_count}")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
