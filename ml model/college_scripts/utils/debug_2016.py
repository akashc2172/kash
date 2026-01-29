import asyncio
from camoufox.async_api import AsyncCamoufox
import re

async def debug_2016():
    # Jan 2, 2016 Scoreboard URL
    url = "https://stats.ncaa.org/season_divisions/12700/livestream_scoreboards?utf8=%E2%9C%93&season_division_id=&game_date=01%2F02%2F2016&conference_id=0&tournament_id=&commit=Submit"
    
    async with AsyncCamoufox(humanize=True, window=(1920, 1080)) as browser:
        print(f"Checking URL: {url}")
        page = await browser.new_page()
        
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        # Check different locators
        count_text = await page.locator("a", has_text="Box Score").count()
        count_href = await page.locator("a[href*='box_score']").count()
        count_all_links = await page.locator("a").count()
        
        print(f"Found with text 'Box Score': {count_text}")
        print(f"Found with href '*box_score*': {count_href}")
        print(f"Total links on page: {count_all_links}")
        
        # Take a screenshot to see what's going on
        await page.screenshot(path="debug_2016_jan2.png")
        print("Screenshot saved to debug_2016_jan2.png")
        
        # Print first few links
        links = await page.locator("a").all()
        print("\nFirst 10 links:")
        for i, link in enumerate(links[:10]):
            text = await link.inner_text()
            href = await link.get_attribute("href")
            print(f"{i+1}. [{text}] -> {href}")

        # Check for specific Pitt game ID from user screenshot: 245898
        pitt_game = await page.locator("a[href*='245898']").count()
        print(f"\nSpecific Game 245898 found: {pitt_game}")

if __name__ == "__main__":
    asyncio.run(debug_2016())
