import asyncio
from camoufox.async_api import AsyncCamoufox

# Base URL pattern for 2015-16 season (ID 12700)
# game_date format: MM%2FDD%2FYYYY (URL-encoded slashes)
BASE_URL = "https://stats.ncaa.org/season_divisions/12700/livestream_scoreboards?utf8=%E2%9C%93&season_division_id=&game_date={date}&conference_id=0&tournament_id=&commit=Submit"

def make_url(month: int, day: int, year: int) -> str:
    """Construct the URL with the date baked in."""
    # URL encode the slashes: / -> %2F
    date_str = f"{month:02d}%2F{day:02d}%2F{year}"
    return BASE_URL.format(date=date_str)

async def test_url_navigation():
    async with AsyncCamoufox(humanize=True, window=(1920, 1080)) as browser:
        print("🦊 Camoufox initialized.")
        page = await browser.new_page()
        
        # Test 1: January 2, 2016
        url1 = make_url(1, 2, 2016)
        print(f"\n📅 Test 1: January 2, 2016")
        print(f"   URL: {url1}")
        await page.goto(url1, wait_until="domcontentloaded")
        await asyncio.sleep(3)
        
        games1 = await page.locator("a", has_text="Box Score").count()
        print(f"   Games found: {games1}")
        
        # Test 2: February 4, 2016 (different date, same season)
        url2 = make_url(2, 4, 2016)
        print(f"\n📅 Test 2: February 4, 2016")
        print(f"   URL: {url2}")
        await page.goto(url2, wait_until="domcontentloaded")
        await asyncio.sleep(3)
        
        games2 = await page.locator("a", has_text="Box Score").count()
        print(f"   Games found: {games2}")
        
        # Test 3: Get actual Box Score links
        if games2 > 0:
            print(f"\n🔗 Sample Box Score links from Feb 4:")
            links = await page.locator("a", has_text="Box Score").all()
            for i, link in enumerate(links[:3]):  # First 3
                href = await link.get_attribute("href")
                print(f"   {i+1}. {href}")
        
        print("\n✅ URL-based navigation test complete!")
        print("⏳ Keeping browser open for 30 seconds...")
        await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(test_url_navigation())
