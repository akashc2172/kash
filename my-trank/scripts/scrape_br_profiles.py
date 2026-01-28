#!/usr/bin/env python3
"""
Scrape Basketball Reference draft pages to get player profile URLs.
Uses Selenium to handle JavaScript and anti-scraping measures.
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
from selenium.webdriver.chrome.service import Service

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'br_profile_urls.csv')

def setup_driver():
    """Setup Chrome driver with options to avoid detection."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def scrape_draft_year(driver, year):
    """Scrape a single draft year from Basketball Reference."""
    url = f"https://www.basketball-reference.com/draft/NBA_{year}.html"
    print(f"Scraping {url}...")
    
    try:
        driver.get(url)
        time.sleep(3)  # Wait for page to load
        
        # Find the stats table
        table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "stats"))
        )
        
        picks = []
        rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
        
        for row in rows:
            try:
                # Skip header rows
                if 'thead' in row.get_attribute('class') or '':
                    continue
                
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 4:
                    th = row.find_element(By.TAG_NAME, "th")
                    pick_num = th.text.strip()
                    if not pick_num.isdigit():
                        continue
                    cells = row.find_elements(By.TAG_NAME, "td")
                else:
                    th = row.find_element(By.TAG_NAME, "th")
                    pick_num = th.text.strip()
                
                if not pick_num.isdigit():
                    continue
                
                # Find player link (usually in the 4th cell or look for player link)
                player_link = None
                player_name = None
                
                for cell in cells:
                    links = cell.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        href = link.get_attribute("href") or ""
                        if "/players/" in href:
                            player_link = href
                            player_name = link.text.strip()
                            break
                    if player_link:
                        break
                
                if player_name and player_link:
                    picks.append({
                        'year': year,
                        'pick': int(pick_num),
                        'player_name': player_name,
                        'profile_url': player_link
                    })
                    
            except Exception as e:
                continue
        
        print(f"  Found {len(picks)} players with profile URLs")
        return picks
        
    except Exception as e:
        print(f"  Error: {e}")
        return []

def main():
    all_picks = []
    
    print("Setting up Chrome driver...")
    driver = setup_driver()
    
    try:
        for year in range(2008, 2026):
            picks = scrape_draft_year(driver, year)
            all_picks.extend(picks)
            time.sleep(5)  # Be respectful between requests
            
    finally:
        driver.quit()
    
    # Write to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['year', 'pick', 'player_name', 'profile_url'])
        writer.writeheader()
        writer.writerows(all_picks)
    
    print(f"\nTotal players scraped: {len(all_picks)}")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
