import asyncio
import os
import random
from playwright.async_api import async_playwright, Page, expect

# --- Configuration ---
OUTPUT_DIR = "data/synergy"
LOGIN_URL = "https://auth.synergysportstech.com/Account/Login?ReturnUrl=%2F"
LEADERBOARD_URL = "https://apps.synergysports.com/basketball/leaderboards"

# Full range of seasons
SEASONS_TO_SCRAPE = [f"{y}-{y+1}" for y in range(2008, 2026)] 

REPORT_TYPES = [
    "Team Offensive",
    "Team Defensive",
    "Player Offensive",
    "Player Defensive"
]

async def wait_for_buffer(page: Page, msg="Buffering..."):
    """
    Waits for the site to process changes (buffer).
    """
    print(f"    ⏳ {msg} (Waiting 10s)", flush=True)
    try:
        # Wait for potential loading overlays to disappear if we can find them
        # await page.wait_for_selector(".loading-spinner", state="hidden", timeout=5000)
        pass
    except:
        pass
    
    # Explicit wait to be safe as user said it takes a while
    await page.wait_for_timeout(10000) 

async def find_ng_select(page: Page, label_text: str):
    """
    Finds an ng-select based on a nearby label or internal text.
    """
    print(f"  Searching for dropdown: {label_text}...", flush=True)
    
    # Strategy 1: Look for a container that has the label and an ng-select
    try:
        # Try finding the specific label element first
        label = page.locator(f"text={label_text}")
        if await label.count() > 0:
            # Strategy 1b: Common layout "Label \n Select"
            dropdown = page.locator("ng-select").filter(has=page.locator(f"xpath=preceding::*[contains(text(), '{label_text}')]")).first
            if await dropdown.count() > 0:
                 return dropdown
    except:
        pass

    # Strategy 2: If finding by label fails, assume specific order? 
    if "Season" in label_text:
        import re
        all_selects = await page.locator("ng-select").all()
        for s in all_selects:
            txt = await s.inner_text()
            if re.search(r"20\d\d-20\d\d", txt):
                return s
                
    # Strategy 3: Report Type
    if "Report Type" in label_text:
         all_selects = await page.locator("ng-select").all()
         for s in all_selects:
            txt = await s.inner_text()
            if "Offensive" in txt or "Defensive" in txt:
                 return s

    return page.locator("ng-select").first

async def select_ng_option(page: Page, label_text: str, option_text: str):
    """
    Selects an option from an Angular ng-select dropdown identified by label.
    """
    print(f"Selecting '{option_text}' in '{label_text}' dropdown...", flush=True)
    try:
        dropdown = await find_ng_select(page, label_text)
        if not dropdown:
            print(f"  Error: Dropdown {label_text} not found.", flush=True)
            return

        # Scroll into view
        await dropdown.scroll_into_view_if_needed()
        
        # 1. Click to open
        # Sometimes click doesn't register if buffering, retry once
        try:
            await dropdown.click(timeout=3000)
        except:
            print("    Click timed out, retrying force click...", flush=True)
            await dropdown.click(force=True)
        
        # 2. Wait for options
        # Increase timeout for panel to appear
        await page.wait_for_selector("ng-dropdown-panel", state="visible", timeout=5000)
        
        # 3. Click Option
        option_locator = page.locator("div.ng-option")
        
        # Try exact match first
        exact_option = option_locator.filter(has_text=f"^{option_text}$").first
        if await exact_option.count() > 0:
            await exact_option.click()
        else:
            # Try contains
            print(f"  Exact match not found for {option_text}, trying partial...", flush=True)
            partial = option_locator.filter(has_text=option_text).first
            if await partial.count() > 0:
                 await partial.click()
            else:
                 print(f"  Warning: Option '{option_text}' not found.", flush=True)
                 await page.keyboard.press("Escape")
            
    except Exception as e:
        print(f"  Error selecting {option_text}: {e}", flush=True)
        await page.keyboard.press("Escape")

async def run_scraper():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 Starting Robust Chromium Scraper...", flush=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        context = await browser.new_context(
            viewport={"width": 1600, "height": 1000},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        page = await context.new_page()
        
        # 1. Login
        print(f"Navigating to Login: {LOGIN_URL}", flush=True)
        await page.goto(LOGIN_URL)
        
        print("\n" + "="*60, flush=True)
        print("ACTION REQUIRED:", flush=True)
        print("1. Log in to Synergy Sports.", flush=True)
        print(f"2. Navigate to: {LEADERBOARD_URL}", flush=True)
        print("3. Ensure the correct League is selected.", flush=True)
        print("="*60 + "\n", flush=True)
        
        try:
            input("Press ENTER here when you are on the Leaderboards page and ready to scrape...")
        except EOFError:
            pass
            
        print("Starting automation...", flush=True)
        
        # 3. Iterate Seasons
        for season in SEASONS_TO_SCRAPE:
            print(f"\nPROCESSING SEASON: {season}", flush=True)
            
            # Select Season
            await select_ng_option(page, "Season", season)
            await wait_for_buffer(page, "Waiting for Season load")
            
            # 4. Iterate Report Types
            for report_type in REPORT_TYPES:
                print(f"  > Report Type: {report_type}", flush=True)
                
                await select_ng_option(page, "Report Type", report_type)
                await wait_for_buffer(page, "Waiting for Report Type load")
                
                # 5. Iterate Situations
                try:
                    # Retry finding the button
                    situation_btn = page.locator(".situation-button-container button")
                    
                    # Wait for it to exist (with retry)
                    try:
                        await situation_btn.wait_for(state="attached", timeout=5000)
                    except:
                        print("    Situation button not found immediately. Waiting longer...", flush=True)
                        await page.wait_for_timeout(5000)
                    
                    if await situation_btn.count() == 0:
                        print("    Situation button STILL not found. Skipping...", flush=True)
                        continue
                            
                    await situation_btn.click()
                    await page.wait_for_selector(".modal-content", state="visible", timeout=8000)
                    
                    # Scrape
                    print("    Scraping available situations...", flush=True)
                    situation_items = page.locator(".modal-content li")
                    if await situation_items.count() == 0:
                         print("    No items found in modal!", flush=True)
                         await page.keyboard.press("Escape")
                         continue
                         
                    situation_texts = await situation_items.all_inner_texts()
                    situation_texts = [s.strip() for s in situation_texts if s.strip()]
                    
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(1000)
                    
                    for situation in situation_texts:
                        print(f"    - Processing Situation: {situation}", flush=True)
                        
                        await situation_btn.click()
                        await page.wait_for_selector(".modal-content", state="visible")
                        
                        try:
                            # Use exact text match
                            item = page.locator(".modal-content li").filter(has_text=f"^{situation}$").first
                            if await item.count() == 0:
                                item = page.locator(".modal-content li").filter(has_text=situation).first
                                
                            await item.click()
                        except:
                            print(f"      Could not select {situation}", flush=True)
                            await page.keyboard.press("Escape")
                            continue
                        
                        # Wait for Table Reload - Situation change is usually faster but let's be safe
                        await page.wait_for_timeout(4000)
                        
                        # 6. Export
                        try:
                            csv_btn = page.locator("button").filter(has_text="CSV").first
                            if await csv_btn.count() == 0:
                                csv_btn = page.locator(".btn-outline-secondary").filter(has_text="CSV").first
                                
                            if await csv_btn.is_enabled():
                                try:
                                    async with page.expect_download(timeout=15000) as download_info:
                                        await csv_btn.click()
                                    download = await download_info.value
                                    
                                    safe_situation = "".join([c for c in situation if c.isalnum() or c in (' ', '-', '_')]).strip()
                                    safe_report = report_type.replace(" ", "_")
                                    target_dir = f"{OUTPUT_DIR}/{season}/{safe_report}"
                                    os.makedirs(target_dir, exist_ok=True)
                                    target_path = f"{target_dir}/{safe_situation}.csv"
                                    
                                    await download.save_as(target_path)
                                    print(f"      Saved: {target_path}", flush=True)
                                except Exception as e:
                                     print(f"      Download failed/timed out: {e}", flush=True)
                            else:
                                print(f"      CSV Export disabled for {situation}", flush=True)

                        except Exception as e:
                            print(f"      Error exporting {situation}: {e}", flush=True)

                except Exception as e:
                    print(f"    Error handling situations: {e}", flush=True)
                    await page.keyboard.press("Escape")
                    continue

    print("\nDone!", flush=True)

if __name__ == "__main__":
    asyncio.run(run_scraper())
