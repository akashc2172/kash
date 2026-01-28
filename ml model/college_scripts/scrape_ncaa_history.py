
"""
NCAA History Scraper (Selenium Version)
---------------------------------------
Scrapes historical Play-by-Play data from stats.ncaa.org using Headless Chrome
to bypass 403 Forbidden anti-bot protections.

Phase 1: Game Mapper (Local gameId -> NCAA contest_id)
Phase 2: PBP Scraper (Contest ID -> Raw PBP Text)

Usage:
    python3 college_scripts/scrape_ncaa_history.py --season 2023 --limit 50 --no-headless
"""

import duckdb
import pandas as pd
import time
import random
import logging
import argparse
import os
from datetime import datetime
from thefuzz import fuzz

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Constants
DB_PATH = 'data/warehouse.duckdb'
OUTPUT_DIR = 'data/scraped_history'
BASE_URL = 'https://stats.ncaa.org'

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scrape_ncaa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

def random_sleep(min_sec=1.5, max_sec=3.0):
    time.sleep(random.uniform(min_sec, max_sec))

def setup_driver(headless=True):
    """Setup Chrome Driver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
        
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    return driver

def normalize_name(name):
    """Normalize school names for matching (e.g. 'State' -> 'St.')."""
    if not name: return ""
    n = name.lower().replace("state", "st.").replace("saint", "st.").replace("univ.", "")
    return "".join(c for c in n if c.isalnum() or c.isspace()).strip()

def map_games_for_date(driver, target_date, local_games):
    """
    Load schedule using direct URL with discovered Season ID.
    """
    dt = datetime.strptime(target_date, '%Y-%m-%d')
    date_text = dt.strftime('%m/%d/%Y')
    
    # Calculate Season Text
    # 2022-11-07 -> 2022-23
    if dt.month > 7:
        start_year = dt.year
    else:
        start_year = dt.year - 1
    season_text = f"{start_year}-{str(start_year + 1)[-2:]}"
    
    # Lazy Load Season ID
    if not hasattr(driver, 'cached_season_id'):
        logger.info(f"  Discovering Season ID for {season_text}...")
        try:
            driver.get("https://stats.ncaa.org/")
            time.sleep(2)
            
            # HUMAN IN THE LOOP: Ask user to set correct context
            print(f"\n\n{'='*60}")
            print(f"⚠️  MANUAL SETUP REQUIRED  ⚠️")
            print(f"1. In the Chrome window, Select Sport: 'Men's Basketball'")
            print(f"2. Select Season: '{season_text}' (or similar)")
            print(f"3. Wait for the page/dropdowns to update.")
            print(f"4. Press ENTER in this terminal to continue...")
            print(f"{'='*60}\n")
            input() # Wait for user
            
            # Find Option Value from the NOW POPULATED dropdown
            select = driver.find_element(By.ID, "game_sport_year_ctl_id")
            for opt in select.find_elements(By.TAG_NAME, "option"):
                if season_text in opt.text:
                    driver.cached_season_id = opt.get_attribute("value")
                    logger.info(f"    Found Season ID: {driver.cached_season_id}")
                    break
            
            if not hasattr(driver, 'cached_season_id'):
                 # Fallback: Just grab the currently selected value if user picked it
                 driver.cached_season_id = select.get_attribute("value")
                 logger.info(f"    Using Selected Season ID: {driver.cached_season_id}")

        except Exception as e:
            logger.error(f"    Failed to find Season ID: {e}")
            return []
            
    # Direct Navigation
    if hasattr(driver, 'cached_season_id'):
        url = f"{BASE_URL}/contests/livestream_scoreboards?game_sport_year_ctl_id={driver.cached_season_id}&game_date={date_text}"
        logger.info(f"  Navigating to Direct URL: {url}")
        driver.get(url)
        time.sleep(3)
        
        box_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='box_score']")
        logger.info(f"    Found {len(box_links)} Box Score links.")
        
        ncaa_games = []
        for link in box_links:
            try:
                href = link.get_attribute('href')
                contest_id = href.split('/')[-2]
                try:
                    text = link.find_element(By.XPATH, "./ancestor::tr").text
                except:
                    text = link.find_element(By.XPATH, "./..").text
                ncaa_games.append({'contest_id': contest_id, 'text': text, 'url': href})
            except:
                continue

        # Fuzzy Match
        mapped = []
        for lg in local_games:
            best_score = 0
            best_match = None
            l_home = normalize_name(lg['home_team'])
            l_away = normalize_name(lg['away_team'])
            
            for ng in ncaa_games:
                n_text = normalize_name(ng['text'])
                s1 = fuzz.partial_ratio(l_home, n_text)
                s2 = fuzz.partial_ratio(l_away, n_text)
                avg_score = (s1 + s2) / 2
                
                if avg_score > 75 and avg_score > best_score:
                    best_score = avg_score
                    best_match = ng
            
            if best_match:
                mapped.append({'local_game_id': lg['gameId'], 'ncaa_contest_id': best_match['contest_id'], 'score': best_score})
                logger.info(f"    Match: {lg['home_team']} vs {lg['away_team']} -> {best_match['contest_id']}")
                
        return mapped
    return []

def scrape_pbp(driver, contest_id):
    """Scrape PBP for a contest ID."""
    url = f"{BASE_URL}/contests/{contest_id}/play_by_play"
    
    try:
        driver.get(url)
        tables = driver.find_elements(By.TAG_NAME, "table")
        target_table = None
        max_rows = 0
        
        for t in tables:
            rows = t.find_elements(By.TAG_NAME, "tr")
            if len(rows) > max_rows:
                max_rows = len(rows)
                target_table = t
        
        if not target_table or max_rows < 10:
            logger.warning(f"  No valid PBP table found for {contest_id}")
            return None
            
        pbp_rows = []
        rows = target_table.find_elements(By.TAG_NAME, "tr")
        for r in rows:
            cols = r.find_elements(By.TAG_NAME, "td")
            if len(cols) > 0:
                row_data = [c.text for c in cols]
                pbp_rows.append(row_data)
                
        return pbp_rows
        
    except Exception as e:
        logger.error(f"  Error scraping PBP {contest_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, default=2023)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--date', type=str, help='Specific date YYYY-MM-DD')
    parser.add_argument('--no-headless', action='store_true', help='Run browser visibly')
    args = parser.parse_args()
    
    con = get_connection()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.date:
        query = f"SELECT CAST(g.id AS VARCHAR) as gameId, CAST(g.startDate AS VARCHAR) as date, ht.school as home_team, at_t.school as away_team FROM dim_games g JOIN dim_teams ht ON g.homeTeamId = ht.id JOIN dim_teams at_t ON g.awayTeamId = at_t.id WHERE CAST(g.startDate AS DATE) = '{args.date}'"
    else:
        query = f"SELECT CAST(g.id AS VARCHAR) as gameId, CAST(g.startDate AS VARCHAR) as date, ht.school as home_team, at_t.school as away_team FROM dim_games g JOIN dim_teams ht ON g.homeTeamId = ht.id JOIN dim_teams at_t ON g.awayTeamId = at_t.id WHERE g.season = {args.season} LIMIT {args.limit}"
    
    games_df = con.execute(query).df()
    logger.info(f"Targeting {len(games_df)} games using Selenium...")
    
    # Init Driver
    driver = setup_driver(headless=not args.no_headless)
    
    try:
        date_groups = games_df.groupby(games_df['date'].astype(str).str[:10])
        all_mappings = []
        
        for date_str, group in date_groups:
            logger.info(f"Processing {date_str}...")
            matches = map_games_for_date(driver, date_str, group.to_dict('records'))
            all_mappings.extend(matches)
            random_sleep(1, 2)
            
        # Save Mappings
        if all_mappings:
            df_map = pd.DataFrame(all_mappings)
            df_map.to_parquet(f"{OUTPUT_DIR}/ncaa_game_map_{args.season}.parquet", index=False)
            logger.info(f"Saved {len(df_map)} game mappings.")
            
            # 2. Scrape PBP
            logger.info("Starting PBP Scrape...")
            pbp_data = []
            
            for _, row in df_map.iterrows():
                cid = row['ncaa_contest_id']
                logger.info(f"  Scraping Contest {cid}...")
                
                rows = scrape_pbp(driver, cid)
                if rows:
                    for r in rows:
                        pbp_data.append({
                            'contest_id': cid,
                            'gameId': row['local_game_id'],
                            'raw_cols': r
                        })
                random_sleep(0.5, 1.5)
                
            if pbp_data:
                df_pbp = pd.DataFrame(pbp_data)
                out_file = f"{OUTPUT_DIR}/scraped_pbp_{args.season}.parquet"
                df_pbp.to_parquet(out_file, index=False)
                logger.info(f"Success! Saved {len(df_pbp)} PBP rows to {out_file}")
            else:
                logger.warning("No PBP data obtained.")
        else:
            logger.warning("No games mapped successfully.")
            
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
