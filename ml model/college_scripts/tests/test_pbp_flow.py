import asyncio
import re
from camoufox.async_api import AsyncCamoufox

# Base URL for 2015-16 season scoreboard (Jan 2, 2016)
SCOREBOARD_URL = "https://stats.ncaa.org/season_divisions/12700/livestream_scoreboards?utf8=%E2%9C%93&season_division_id=&game_date=01%2F02%2F2016&conference_id=0&tournament_id=&commit=Submit"
BASE_URL = "https://stats.ncaa.org"

async def test_pbp_navigation():
    async with AsyncCamoufox(humanize=True, window=(1920, 1080)) as browser:
        print("🦊 Camoufox initialized.")
        page = await browser.new_page()
        
        # 1. Step 1: Nav to Scoreboard
        print(f"👉 Navigating to Scoreboard: Jan 2, 2016...")
        await page.goto(SCOREBOARD_URL, wait_until="domcontentloaded")
        await asyncio.sleep(3)
        
        # 2. Step 2: Extract Box Score links
        # Using href selector because text might be split across lines (rendering issue)
        box_score_links = await page.locator("a[href*='box_score']").all()
        print(f"   Found {len(box_score_links)} links matching 'box_score'.")
        
        if not box_score_links:
            print("   ❌ No games found. Exiting.")
            return

        # 3. Step 3: Parse Contest ID from the first link
        # Example href: /contests/12345/box_score
        target_link = box_score_links[0]
        href = await target_link.get_attribute("href")
        print(f"   Sample href: {href}")
        
        # Extract ID using regex
        match = re.search(r'/contests/(\d+)/', href)
        if not match:
            print("   ❌ Could not parse Contest ID. Exiting.")
            return
            
        contest_id = match.group(1)
        print(f"   ✅ Extracted Contest ID: {contest_id}")
        
        # 4. Step 4: Construct and Navigate to Play-by-Play
        pbp_url = f"{BASE_URL}/contests/{contest_id}/play_by_play"
        print(f"👉 Navigating to PBP: {pbp_url}")
        
        await page.goto(pbp_url, wait_until="domcontentloaded")
        await asyncio.sleep(2)
        
        # 5. Step 5: Verify PBP page loaded
        print(f"   Current URL: {page.url}")
        
        # Look for headers or common PBP tables
        # Since rendering is weird, we check for 'Play by Play' anywhere in the body or page title
        title = await page.title()
        print(f"   Page Title: {title}")
        
        pbp_header = await page.locator("h1, h2, .card-header", has_text=re.compile("Play by Play", re.I)).first.count()
        if pbp_header > 0 or "Play by Play" in title:
            print("   ✅ Play by Play page confirmed!")
        else:
            print("   ⚠️ Header not found, but checking for tables...")
            
        # 6. Step 6: Extract Table Data
        print("👉 Extracting PBP table data...")
        
        # Use JS to extract all rows from all tables, filtering for the one that looks like PBP
        # PBP tables usually have headers like "Time", "Score", etc.
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
            
            if (!bestTable) return [];
            
            return Array.from(bestTable.rows).map(row => 
                Array.from(row.cells).map(cell => cell.innerText.trim())
            );
        }""")
        
        if rows_data:
            print(f"   ✅ Extracted {len(rows_data)} rows.")
            
            # Save to a temporary CSV for verification
            import pandas as pd
            import os
            
            os.makedirs("data/scraped_history", exist_ok=True)
            df = pd.DataFrame(rows_data)
            output_path = f"data/scraped_history/test_pbp_{contest_id}.csv"
            df.to_csv(output_path, index=False, header=False)
            print(f"   💾 Saved to: {output_path}")
            
            # Print a few rows
            print("\n   --- SAMPLE DATA ---")
            for row in rows_data[:10]:
                print(f"   {row}")
        else:
            print("   ❌ Failed to extract any table data.")
        
        print("\n✅ Scraping test complete!")
        print("⏳ Keeping browser open for 60 seconds for visual inspection...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(test_pbp_navigation())
