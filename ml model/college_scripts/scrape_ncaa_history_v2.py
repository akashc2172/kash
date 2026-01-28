import asyncio
import pandas as pd
import random
import os
import re
from datetime import datetime, timedelta
from camoufox.async_api import AsyncCamoufox

# --- Configuration ---
OUTPUT_DIR = "data/scraped_history/2015"
# Season 12700 is 2015-16
SEASON_ID = "12700"
START_DATE = datetime(2015, 11, 13)
END_DATE = datetime(2016, 4, 4)
BASE_URL = "https://stats.ncaa.org"

# --- Helper: Scrape PBP ---
async def scrape_game_pbp(page, contest_id: str, game_date_iso: str) -> list:
    """Navigates directly to the PBP page and extracts table data."""
    url = f"{BASE_URL}/contests/{contest_id}/play_by_play"
    
    try:
        # print(f"      Visiting PBP: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(random.uniform(2, 4)) # Settle
        
        # Extract Table
        rows_data = await page.evaluate("""() => {
            const tables = Array.from(document.querySelectorAll('table'));
            let bestTable = null;
            let maxRows = 0;
            
            for (const table of tables) {
                // Heuristic: PBP table is usually the largest one
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
async def run_scraper():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Simple browser config to avoid crashes
    async with AsyncCamoufox(
        humanize=True,
        window=(1920, 1080),
    ) as browser:
        print(f"🦊 Scraper started for Season {SEASON_ID}")
        page = await browser.new_page()
        
        current_date = START_DATE
        while current_date <= END_DATE:
            date_str_url = current_date.strftime("%m/%d/%Y").replace("/", "%2F")
            date_str_iso = current_date.strftime("%Y-%m-%d")
            
            # 1. Load Scoreboard
            scoreboard_url = f"{BASE_URL}/season_divisions/{SEASON_ID}/livestream_scoreboards?utf8=%E2%9C%93&season_division_id=&game_date={date_str_url}&conference_id=0&tournament_id=&commit=Submit"
            
            print(f"\n📅 [{date_str_iso}] Fetching: {scoreboard_url}")
            try:
                await page.goto(scoreboard_url, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(random.uniform(3, 5))
                
                # 2. Extract Contest IDs
                # Try both href and text-based locators to be robust
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
                    # Diagnostics: Take a screenshot if we get 0 games on a day that should have them
                    # (Jan 1 is often empty, but Jan 2+ shouldn't be)
                    if current_date.month == 1 and current_date.day > 1:
                        img_path = f"debug_empty_{date_str_iso}.png"
                        await page.screenshot(path=img_path)
                        print(f"    ⚠️ 0 games found. Screenshot saved to {img_path}")
                else:
                    # 3. Scrape each game sequentially
                    daily_pbp = []
                    for cid in contest_ids:
                        print(f"    👉 Scraping Game ID: {cid}...")
                        pbp_data = await scrape_game_pbp(page, cid, date_str_iso)
                        daily_pbp.extend(pbp_data)
                        
                        # BE VERY CONSERVATIVE: 5-10 seconds between games
                        sleep_time = random.uniform(5.0, 10.0)
                        await asyncio.sleep(sleep_time)
                    
                    # 4. Save Daily Data
                    if daily_pbp:
                        df = pd.DataFrame(daily_pbp)
                        outfile = f"{OUTPUT_DIR}/ncaa_pbp_{date_str_iso}.csv"
                        df.to_csv(outfile, index=False)
                        print(f"    ✅ Saved {len(df)} rows to {outfile}")
                
            except Exception as e:
                print(f"    ❌ Error on {date_str_iso}: {e}")
                await asyncio.sleep(30)
            
            # --- IMPORTANT: Always increment and sleep to avoid 'zooming' through empty days ---
            current_date += timedelta(days=1)
            day_sleep = random.uniform(10.0, 20.0)
            print(f"  💤 Rest period: {day_sleep:.1f}s...")
            await asyncio.sleep(day_sleep)

    print("\n🏁 Season Scrape Complete!")

if __name__ == "__main__":
    asyncio.run(run_scraper())
