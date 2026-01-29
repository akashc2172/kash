import asyncio
from camoufox.async_api import AsyncCamoufox

# The user-provided "Seeded" URL for 2014-15 Season (ID 12260)
# Note: Division 1 is selected in the params
SEED_URL = "https://stats.ncaa.org/contests/livestream_scoreboards?utf8=%E2%9C%93&game_sport_year_ctl_id=12260&conference_id=0&conference_id=0&tournament_id=&division=1&commit=Submit"

async def test_date_entry():
    # Full browser config to avoid bot detection
    async with AsyncCamoufox(
        humanize=True,
        geoip=True,  # Use realistic geolocation
        window=(1920, 1080),
        block_images=False,  # Ensure images load
        block_webrtc=False,  # Don't block webrtc (can cause detection)
    ) as browser:
        print("🦊 Camoufox initialized (Full Browser Mode).")
        
        # Create context with more realistic settings
        context = await browser.new_context(
            locale="en-US",
            timezone_id="America/New_York",
            color_scheme="light",
        )
        page = await context.new_page()
        
        print(f"👉 Navigating to Seeded Season URL...")
        await page.goto(SEED_URL, wait_until="domcontentloaded")
        await asyncio.sleep(4) # Allow JS/Hydration to complete
        
        # Test Date: Jan 4, 2015
        target_date = "01/04/2015"
        print(f"\n📅 Attempting to change date to: {target_date}")
        
        try:
            # 1. Clear and Type Date
            date_input = page.locator("#game_date")
            print("   Clicking and focusing date input...")
            await date_input.click()
            await asyncio.sleep(1)
            
            # Select All and Delete
            print("   Clearing existing date...")
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Meta+A") # Compatibility for Mac environments
            await page.keyboard.press("Backspace")
            await asyncio.sleep(0.5)
            
            # Type slowly
            print(f"   Typing {target_date}...")
            await page.keyboard.type(target_date, delay=150)
            
            # IMPORTANT: Wait a bit before Enter. 
            # Often the date-picker JS needs a moment to 'register' the change.
            await asyncio.sleep(2)
            
            # 2. Trigger Submit via Enter
            print("   Searching (Enter)...")
            await page.keyboard.press("Enter")
            
            # 3. Wait for the page to reload or the content to change
            print("   Waiting for page refresh...")
            try:
                # We wait for the table or the date string to appear in the content
                await page.wait_for_load_state("networkidle", timeout=15000)
            except:
                pass
            
            await asyncio.sleep(3) # Extra buffer
            
            # 4. Final Verification
            # Look for the date text in the main content area
            content = await page.content()
            if target_date in content or "01/04/2015" in content:
                print("   ✅ SUCCESS: Results for Jan 4 found!")
            else:
                print("   ⚠️ WARNING: Page still showing old data? Check the browser window.")
            
            # Count games
            games_count = await page.locator("a", has_text="Box Score").count()
            print(f"   Games found on screen: {games_count}")
                
            if games_count > 0:
                print("   ✅ Games loaded.")
            else:
                print("   ⚠️ No games found (might be empty day).")
                
        except Exception as e:
            print(f"   result: ERROR {e}")

        print("\n⏳ Keeping browser open for 60 seconds for visual inspection...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(test_date_entry())
