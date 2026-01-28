import asyncio
import pandas as pd
import random
import os
import re
from datetime import datetime, timedelta
from camoufox.async_api import AsyncCamoufox

# Configuration
OUTPUT_DIR = "data/scraped_history"
BASE_URL = "https://stats.ncaa.org"

# --- 1. Helper: Scrape PBP via the "Human Path" ---
async def scrape_game_pbp(page, box_score_url: str, game_date: str) -> list:
    """
    Follows the buttons: Go to Box Score -> Click 'Play by Play' -> Scrape.
    """
    try:
        # 1. Go to the Box Score first
        if not box_score_url.startswith("http"):
            box_score_url = f"{BASE_URL}{box_score_url}"
            
        # print(f"      Visiting Box Score: {box_score_url}")
        await page.goto(box_score_url, timeout=30000, wait_until="domcontentloaded")

        # 2. Click the "Play by Play" tab/link
        try:
            # Look for the link that says "Play by Play" (using case insensitive regex or partial match)
            pbp_link = page.locator("a", has_text="Play by Play").first
            
            # Check if it exists before clicking
            if await pbp_link.count() > 0:
                await pbp_link.click()
                await page.wait_for_load_state("domcontentloaded")
            else:
                # Fallback: maybe it's just "PBP" or we are already there?
                pass
        except Exception as e:
            # If we can't find the tab, maybe we are already there, or no PBP exists
            print(f"      Warning: Could not click 'Play by Play' tab: {e}")

        # 3. Wait for the PBP table
        try:
            await page.wait_for_selector("table", timeout=5000)
        except:
            print("      No table found on PBP page.")
            return []

        # 4. Find the correct table (largest one is usually the PBP)
        tables = await page.locator("table").all()
        best_table = None
        max_rows = 0

        for t in tables:
            count = await t.locator("tr").count()
            if count > max_rows:
                max_rows = count
                best_table = t

        if not best_table or max_rows < 5:
            # print("      Table too small or empty.")
            return []

        # 5. Extract Data
        rows_data = await best_table.evaluate("""(table) => {
            const rows = Array.from(table.rows);
            return rows.map(row => Array.from(row.cells).map(cell => cell.innerText.trim()));
        }""")

        game_data = []
        # Get Contest ID from URL for reference
        # url looks like .../contests/12345/play_by_play
        try:
            contest_id = page.url.split("/contests/")[-1].split("/")[0]
        except:
            contest_id = "unknown"

        for r in rows_data:
            if len(r) > 1:
                game_data.append({
                    "contest_id": contest_id,
                    "date": game_date,
                    "raw_text": " | ".join(r)
                })

        return game_data

    except Exception as e:
        print(f"    ! Error scraping game {box_score_url}: {e}")
        return []


# --- 2. Setup Filters ---
async def setup_scoreboard_filters(page, season_year):
    print(f"  ⚙️ Setting up filters for Season {season_year}...")
    await page.goto(f"{BASE_URL}/contests/livestream_scoreboards", wait_until="domcontentloaded")
    
    # --- FIX: ROBUST CHOSEN UI INTERACTION ---
    
    # 1. Select Sport: "Men's Basketball"
    try:
        # Check if we need to expand the dropdown
        # The 'chosen' widget is a div with id matching the select id + '_chosen'
        # But usually we can click the visible text container.
        
        # Try finding the chosen container for sport
        chosen_sport = page.locator("#sport_list_chosen")
        if await chosen_sport.count() > 0:
            await chosen_sport.click()
            # Click the option
            await page.locator("#sport_list_chosen .active-result", has_text="Men's Basketball").click()
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(2) # Allow for reload
        else:
            # Fallback to standard select if chosen isn't there (unlikely on main site)
             await page.locator("select:has(option:text-is('Men\\'s Basketball'))").select_option(label="Men's Basketball")
    except Exception as e:
        print(f"    Warning during Sport selection: {e}")

    # 2. Select Season
    prev_year = season_year - 1
    short_current = str(season_year)[-2:]
    season_label = f"{prev_year}-{short_current}" # e.g. "2014-15"
    
    try:
        # Check for Season Chosen container
        chosen_season = page.locator("#game_sport_year_ctl_id_chosen")
        if await chosen_season.count() > 0:
            await chosen_season.click()
            await page.locator("#game_sport_year_ctl_id_chosen .active-result", has_text=season_label).click()
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(1)
        else:
            season_select = page.locator("select[name*='game_sport_year_ctl_id']")
            await season_select.select_option(label=season_label)
            await page.wait_for_load_state("domcontentloaded")
            
    except Exception as e:
        print(f"    ❌ Failed to select season {season_label}: {e}")
        return False

    # 3. Select Division (D-I)
    try:
        chosen_div = page.locator("#division_chosen")
        if await chosen_div.count() > 0:
            await chosen_div.click()
            await page.locator("#division_chosen .active-result", has_text="D-I").click()
            await page.wait_for_load_state("domcontentloaded")
    except:
        pass 
        
    print("  ✅ Filters set.")
    return True


# --- 3. Main Loop ---
async def scrape_season_interactive(browser, season_year):
    context = await browser.new_context()
    page = await context.new_page()

    if not await setup_scoreboard_filters(page, season_year):
        await context.close()
        return

    # Date Range
    # start_date = datetime(season_year - 1, 11, 14) 
    # Use a bit wider range or custom
    start_date = datetime(season_year - 1, 11, 7)
    end_date = datetime(season_year, 4, 8)
    current_date = start_date

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n=== BROWSING SEASON {season_year} ===")

    while current_date <= end_date:
        date_str = current_date.strftime("%m/%d/%Y")
        iso_date = current_date.strftime("%Y-%m-%d")
        
        print(f"  > [{iso_date}] Processing {date_str}...")

        try:
            # 1. Enter Date
            # Use specific locator for the date input
            date_input = page.locator("#game_date")
            await date_input.click()
            await date_input.fill(date_str)
            await date_input.press("Enter")
            
            # --- FIX: Wait for the result to actually update ---
            # Just waiting for networkidle is often not enough.
            # We explicitly wait for the value in the input to be stable or the page to reload.
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except:
                pass

            # 2. Validate that we are on the right date
            # Does the page display the date somewhere? 
            # Often the input value itself is the best check.
            
            # 3. Find all "Box Score" links
            # We get the HREFs so we can open them in new tabs. 
            box_score_links = await page.locator("a", has_text="Box Score").all()
            
            game_urls = []
            for link in box_score_links:
                href = await link.get_attribute("href")
                if href:
                    game_urls.append(href)
            
            # Deduplicate
            game_urls = list(set(game_urls))
            print(f"    Found {len(game_urls)} games.")

            # 4. Visit each game (Human Path)
            if game_urls:
                daily_pbp = []
                pbp_page = await context.new_page()
                
                for url in game_urls:
                    data = await scrape_game_pbp(pbp_page, url, iso_date)
                    daily_pbp.extend(data)
                    
                    # Way way more conservative: 3 to 7 seconds per game
                    sleep_time = random.uniform(3.0, 7.0)
                    # print(f"      Sleeping {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time) 
                
                await pbp_page.close()

                if daily_pbp:
                    df = pd.DataFrame(daily_pbp)
                    outfile = f"{OUTPUT_DIR}/ncaa_pbp_{iso_date}.csv"
                    df.to_csv(outfile, index=False)
                    print(f"    ✅ Saved {len(df)} rows.")
            else:
                 print(f"    (No games found for {iso_date})")

        except Exception as e:
            print(f"    ! Error on {iso_date}: {e}")

        current_date += timedelta(days=1)
        # Conservative day sleep: 10 to 15 seconds
        day_sleep = random.uniform(10.0, 15.0)
        print(f"  💤 Resting {day_sleep:.1f}s before next date...")
        await asyncio.sleep(day_sleep)

    await context.close()


# --- 4. Run ---
async def main():
    async with AsyncCamoufox(humanize=True, disable_coop=True, window=(1920, 1080)) as browser:
        # Run for 2023 Season (2022-23)
        # Can be parameterized
        await scrape_season_interactive(browser, 2023)

if __name__ == "__main__":
    asyncio.run(main())
