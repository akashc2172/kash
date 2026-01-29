import asyncio
from camoufox.async_api import AsyncCamoufox

BASE_URL = "https://stats.ncaa.org/contests/livestream_scoreboards"

async def get_season_ids():
    # Use a single window, human-like behavior
    async with AsyncCamoufox(humanize=True, disable_coop=True, window=(1280, 800)) as browser:
        print("🦊 Camoufox initialized.")
        page = await browser.new_page()
        
        # 1. Go to Scoreboard
        print(f"👉 Navigating to {BASE_URL}...")
        await page.goto(BASE_URL, wait_until="domcontentloaded")
        
        # 2. Select Men's Basketball to populate the Season dropdown
        print("🏀 Selecting 'Men's Basketball'...")
        try:
            # Try Clicking the Chosen UI
            chosen_sport = page.locator("#sport_list_chosen")
            if await chosen_sport.count() > 0:
                await chosen_sport.click()
                await page.locator("#sport_list_chosen .active-result", has_text="Men's Basketball").click()
                await page.wait_for_load_state("domcontentloaded")
                await asyncio.sleep(2) # Wait for reload/hydration
            else:
                # Fallback
                 await page.locator("select:has(option:text-is('Men\\'s Basketball'))").select_option(label="Men's Basketball")
        except Exception as e:
            print(f"FAILED to select sport: {e}")
            return

        # 3. Extract Options from "game_sport_year_ctl_id"
        # Even if hidden, the <select> usually exists and is populated.
        print("🔍 Scanning Season IDs...")
        
        # Use JS to get all options text/value
        options = await page.eval_on_selector_all("#game_sport_year_ctl_id option", """
            opts => opts.map(o => ({text: o.innerText.trim(), value: o.value}))
        """)
        
        print("\n=== SEASON ID MAP ===")
        found_2015 = None
        for opt in options:
            print(f"  {opt['text']}: {opt['value']}")
            if "2014-15" in opt['text']:
                found_2015 = opt['value']
                
        # 4. Test Navigation
        if found_2015:
            print(f"\n🧪 Testing Direct Navigation to 2014-15 (ID: {found_2015})...")
            test_url = f"{BASE_URL}?game_sport_year_ctl_id={found_2015}"
            
            await page.goto(test_url, wait_until="domcontentloaded")
            await asyncio.sleep(2)
            
            # Verify if it worked by reading the selected value or text on page
            current_val = await page.eval_on_selector("#game_sport_year_ctl_id", "el => el.value")
            print(f"  Current Season Select Value: {current_val}")
            
            if str(current_val) == str(found_2015):
                print("  ✅ SUCCESS: URL parameter correctly switched the season!")
            else:
                print(f"  ❌ FAILURE: Expected {found_2015}, got {current_val}")
        
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(get_season_ids())
