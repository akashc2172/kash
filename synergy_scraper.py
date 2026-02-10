import asyncio
import os
import random
from camoufox.async_api import AsyncCamoufox
from playwright.async_api import Page, expect

# --- Configuration ---
# --- Configuration ---
OUTPUT_DIR = "data/synergy"
LOGIN_URL = "https://auth.synergysportstech.com/Account/Login?ReturnUrl=%2F"
LEADERBOARD_URL = "https://apps.synergysports.com/basketball/leaderboards"

SEASONS_TO_SCRAPE = [f"{y}-{y+1}" for y in range(2008, 2026)] 

REPORT_TYPES = [
    "Team Offensive",
    "Team Defensive",
    "Player Offensive",
    "Player Defensive"
]

async def safe_click(page: Page, selector: str):
    try:
        await page.wait_for_selector(selector, state="visible", timeout=5000)
        await page.click(selector)
    except Exception as e:
        print(f"Error clicking {selector}: {e}")
        # Don't raise, just log. 
        pass

async def select_ng_option(page: Page, control_name: str, option_text: str):
    """
    Selects an option from an Angular ng-select dropdown.
    """
    print(f"Selecting '{option_text}' in {control_name}...", flush=True)
    try:
        # 1. Click the specific ng-select to open it
        dropdown = page.locator(f"ng-select[formcontrolname='{control_name}']")
        await dropdown.click()
        
        # 2. Wait for the dropdown panel to appear
        # The options are usually in a 'ng-dropdown-panel'
        await page.wait_for_selector("ng-dropdown-panel", state="visible", timeout=5000)
        
        # 3. Find the option with exact text and click it
        # Options are usually 'div.ng-option'
        option = page.locator("div.ng-option").filter(has_text=f"^{option_text}$").first
        if await option.count() > 0:
            await option.click()
        else:
            print(f"  Warning: Option '{option_text}' not found in dropdown.", flush=True)
            # Try approximate match if exact fails?
            # await page.locator("div.ng-option").filter(has_text=option_text).first.click()
            # Press Escape to close if invalid
            await page.keyboard.press("Escape")
            
    except Exception as e:
        print(f"  Error selecting {option_text}: {e}", flush=True)
        # Attempt to close dropdown if stuck
        await page.keyboard.press("Escape")

async def run_scraper():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    async with AsyncCamoufox(
        headless=False,
        humanize=True,
        window=(1600, 1000)
    ) as browser:
        print("🦊 Camoufox Browser Started", flush=True)
        page = await browser.new_page()
        
        # 1. Go to Login Page
        print(f"Navigating to Login: {LOGIN_URL}", flush=True)
        await page.goto(LOGIN_URL)
        
        # 2. Manual Login & Navigation Wait
        print("\n" + "="*60, flush=True)
        print("ACTION REQUIRED:", flush=True)
        print("1. Log in to Synergy Sports in the opened browser window.", flush=True)
        print("   NOTE: If the page displays strange characters/garbage text, REFRESH the page manually.", flush=True)
        print("2. Once logged in, navigate to the 'Leaderboards' page if not redirected there.", flush=True)
        print(f"   (Target: {LEADERBOARD_URL})", flush=True)
        print("3. Ensure the correct League (e.g. 'College Men') is selected on the left.", flush=True)
        print("="*60 + "\n", flush=True)
        
        try:
            input("Press ENTER here when you are on the Leaderboards page and ready to scrape...")
        except EOFError:
            pass
            
        print("Starting automation on current page...", flush=True)
        
        # 3. Iterate Seasons
        for season in SEASONS_TO_SCRAPE:
            print(f"\nPROCESSING SEASON: {season}", flush=True)
            
            # Select Season
            await select_ng_option(page, "seasonId", season)
            await page.wait_for_timeout(2000) # Wait for data reload
            
            # 4. Iterate Report Types
            for report_type in REPORT_TYPES:
                print(f"  > Report Type: {report_type}", flush=True)
                
                # Select Report Type
                await select_ng_option(page, "reportType", report_type)
                await page.wait_for_timeout(2000)
                
                # 5. Iterate Situations
                # The situation selector is a button that opens a modal
                try:
                    situation_btn = page.locator(".situation-button-container button")
                    await situation_btn.click()
                    
                    # Wait for modal
                    await page.wait_for_selector(".modal-content", state="visible", timeout=5000)
                    
                    # Scrape all available situations from the modal
                    # They are likely <li> items in the modal content
                    # We grab texts first, then iterate.
                    print("    Scraping available situations...", flush=True)
                    situation_items = page.locator(".modal-content li")
                    situation_texts = await situation_items.all_inner_texts()
                    
                    # Clean up texts (remove counts if present, e.g., "Transition (123)") 
                    # Assuming nice text for now.
                    situation_texts = [s.strip() for s in situation_texts if s.strip()]
                    
                    # Close modal to start clean loop (click backdrop or Escape)
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(500)
                    
                    for situation in situation_texts:
                        # Skip "All" if user wants distinct ones, but user said "every situation"
                        # "All" might be the default.
                        print(f"    - Processing Situation: {situation}", flush=True)
                        
                        # Open Modal Again
                        await situation_btn.click()
                        await page.wait_for_selector(".modal-content", state="visible")
                        
                        # Click the specific Situation <li>
                        # Use exact text match to avoid bad clicks
                        await page.locator(".modal-content li").filter(has_text=f"^{situation}$").first.click()
                        
                        # Wait for Table Reload
                        # We can look for a loading spinner or just wait generic time
                        await page.wait_for_timeout(1500)
                        
                        # 6. Export CSV
                        try:
                            # Click CSV button. It might be an icon or text.
                            csv_btn = page.locator("button").filter(has_text="CSV").first
                            if await csv_btn.count() == 0:
                                # Try class approach if text fails (sometimes it's 'Export CSV' or just icon)
                                csv_btn = page.locator(".btn-outline-secondary").filter(has_text="CSV").first
                                
                            if await csv_btn.is_enabled():
                                async with page.expect_download(timeout=10000) as download_info:
                                    await csv_btn.click()
                                
                                download = await download_info.value
                                
                                # Prepare Filename
                                safe_situation = "".join([c for c in situation if c.isalnum() or c in (' ', '-', '_')]).strip()
                                safe_report = report_type.replace(" ", "_")
                                target_dir = f"{OUTPUT_DIR}/{season}/{safe_report}"
                                os.makedirs(target_dir, exist_ok=True)
                                
                                target_path = f"{target_dir}/{safe_situation}.csv"
                                await download.save_as(target_path)
                                print(f"      Saved: {target_path}", flush=True)
                            else:
                                print(f"      CSV Export disabled for {situation}", flush=True)

                        except Exception as e:
                            print(f"      Error exporting {situation}: {e}", flush=True)

                except Exception as e:
                    print(f"    Error handling situations: {e}", flush=True)
                    # Try to recover by hitting Escape
                    await page.keyboard.press("Escape")
                    continue

    print("\nDone!", flush=True)

if __name__ == "__main__":
    asyncio.run(run_scraper())
