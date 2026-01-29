import asyncio
import pandas as pd
import random
import os
import re
from datetime import datetime, timedelta
from camoufox.async_api import AsyncCamoufox

# --- SEASON CONFIGURATIONS ---
# You can add or edit seasons here. 
# The script will use the configuration for the active season.
SEASONS_CONFIG = {
    "2012": {
        "season_id": "10480",
        "start_date": datetime(2011, 11, 11),
        "end_date": datetime(2012, 4, 2),
        "output_dir": "data/scraped_history/2012"
    },
    "2013": {
        "season_id": "10883",
        "start_date": datetime(2012, 11, 9),
        "end_date": datetime(2013, 4, 8),
        "output_dir": "data/scraped_history/2013"
    },
    "2014": {
        "season_id": "12020",
        "start_date": datetime(2014, 11, 14),
        "end_date": datetime(2015, 4, 6),
        "output_dir": "data/manual_scrapes/2014"
    },
    "2015": {
        "season_id": "12700",
        "start_date": datetime(2015, 11, 13),
        "end_date": datetime(2016, 4, 4),
        "output_dir": "data/manual_scrapes/2015"
    },
    "2017": {
        "season_id": "13100",
        "start_date": datetime(2016, 11, 11),
        "end_date": datetime(2017, 4, 3),
        "output_dir": "data/manual_scrapes/2017"
    }
}

# --- ACTIVE RUN SETTINGS ---
# Change THIS to the season you want to run
ACTIVE_SEASON = "2017" 

BASE_URL = "https://stats.ncaa.org"

# --- Helper: Scrape PBP ---
async def scrape_game_pbp(page, contest_id: str, game_date_iso: str) -> list:
    """Navigates directly to the PBP page and extracts table data."""
    url = f"{BASE_URL}/contests/{contest_id}/play_by_play"
    
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(random.uniform(2, 4)) # Settle
        
        # Extract Table
        rows_data = await page.evaluate("""() => {
            const tables = Array.from(document.querySelectorAll('table'));
            let bestTable = null;
            let maxRows = 0;
            
            for (const table of tables) {
                const rowCount = table.rows.length;
                if (rowCount > maxRows) {
                    maxRows = rowCount;
                    bestTable = table;
                }
            }
            
            if (!bestTable || bestTable.rows.length < 5) return [];
            
            return Array.from(bestTable.rows).map(row => 
                Array.from(row.cells).map(cell => cell.innerText.trim())
            );
        }""")
        
        if not rows_data:
            return []
            
        game_data = []
        for r in rows_data:
            if len(r) > 1:
                game_data.append({
                    "contest_id": contest_id,
                    "date": game_date_iso,
                    "raw_text": " | ".join(r)
                })
        return game_data

    except Exception as e:
        print(f"      ! Error scraping contest {contest_id}: {e}")
        return []

# --- Main Scraper Loop ---
async def run_scraper(season_name: str):
    config = SEASONS_CONFIG[season_name]
    season_id = config["season_id"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    output_dir = config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    
    async with AsyncCamoufox(
        humanize=True,
        window=(1920, 1080),
    ) as browser:
        print(f"🦊 MASTER SCRAPER START: {season_name} (ID {season_id})")
        page = await browser.new_page()
        
        current_date = start_date
        while current_date <= end_date:
            date_str_url = current_date.strftime("%m/%d/%Y").replace("/", "%2F")
            date_str_iso = current_date.strftime("%Y-%m-%d")
            
            scoreboard_url = f"{BASE_URL}/season_divisions/{season_id}/livestream_scoreboards?utf8=%E2%9C%93&season_division_id=&game_date={date_str_url}&conference_id=0&tournament_id=&commit=Submit"
            
            print(f"\n📅 [{date_str_iso}] Fetching: {scoreboard_url}")
            try:
                await page.goto(scoreboard_url, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(random.uniform(3, 5))
                
                # Extract Contest IDs
                links = await page.locator("a[href*='box_score']").all()
                if not links:
                    links = await page.locator("a", has_text=re.compile("Box Score", re.I)).all()
                
                contest_ids = []
                for link in links:
                    href = await link.get_attribute("href")
                    if href:
                        match = re.search(r'/contests/(\d+)/', href)
                        if match:
                            contest_ids.append(match.group(1))
                
                contest_ids = list(set(contest_ids))
                print(f"    Found {len(contest_ids)} games.")
                
                if not contest_ids:
                    # Holiday check/Diagnostic
                    if current_date.month != 12 or current_date.day not in [24, 25]:
                        img_path = f"debug_empty_{season_name}_{date_str_iso}.png"
                        await page.screenshot(path=img_path)
                        print(f"    ⚠️ 0 games found. Diagnostic screenshot saved.")
                else:
                    daily_pbp = []
                    for cid in contest_ids:
                        print(f"    👉 Scraping Game ID: {cid}...")
                        pbp_data = await scrape_game_pbp(page, cid, date_str_iso)
                        daily_pbp.extend(pbp_data)
                        
                        await asyncio.sleep(random.uniform(5.0, 10.0))
                    
                    if daily_pbp:
                        df = pd.DataFrame(daily_pbp)
                        outfile = f"{output_dir}/ncaa_pbp_{date_str_iso}.csv"
                        df.to_csv(outfile, index=False)
                        print(f"    ✅ Saved {len(df)} rows to {outfile}")
                
            except Exception as e:
                print(f"    ❌ Error on {date_str_iso}: {e}")
                await asyncio.sleep(30)
            
            current_date += timedelta(days=1)
            day_sleep = random.uniform(10.0, 20.0)
            print(f"  💤 Rest period: {day_sleep:.1f}s...")
            await asyncio.sleep(day_sleep)

    print(f"\n🏁 Finished scraping {season_name}!")

if __name__ == "__main__":
    if ACTIVE_SEASON in SEASONS_CONFIG:
        asyncio.run(run_scraper(ACTIVE_SEASON))
    else:
        print(f"Error: Season '{ACTIVE_SEASON}' not found in configuration.")
