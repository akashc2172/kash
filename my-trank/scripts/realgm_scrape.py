"""
RealGM Multi-League Scraper for 2026
====================================

Iteratively scrapes multiple RealGM leagues for 2026 season data.
Uses camoufox for browser automation.

Leagues to scrape:
- French Jeep Elite (league 12)
- Euroleague (league 1) - special URL structure
- French LNB Espoirs (league 114)
- Australian NBL (league 5)
- Spanish ACB (league 4)
- Turkish BSL (league 7)
- Eurocup (league 2)
- Adriatic League Liga ABA (league 18)
"""

import asyncio
import pandas as pd
import random
import re
import os
from camoufox.async_api import AsyncCamoufox

# League configurations
LEAGUES = [
    {
        "id": 12,
        "name": "French-Jeep-Elite",
        "display_name": "French Jeep Elite",
        "conf": "(INTL) FRA",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 1,
        "name": "Euroleague",
        "display_name": "Euroleague",
        "conf": "(INTL) EUR",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 114,
        "name": "French-LNB-Espoirs",
        "display_name": "French LNB Espoirs",
        "conf": "(INTL) FRA ESP",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 50,
        "name": "French-LNB-Pro-B",
        "display_name": "French LNB Pro B",
        "conf": "(INTL) FRA PRO B",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 5,
        "name": "Australian-NBL",
        "display_name": "Australian NBL",
        "conf": "(INTL) AUS",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 136,
        "name": "NBL-Blitz",
        "display_name": "NBL Blitz",
        "conf": "(INTL) AUS BLITZ",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 4,
        "name": "Spanish-ACB",
        "display_name": "Spanish ACB",
        "conf": "(INTL) ESP",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 7,
        "name": "Turkish-BSL",
        "display_name": "Turkish BSL",
        "conf": "(INTL) TUR",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 2,
        "name": "Eurocup",
        "display_name": "Eurocup",
        "conf": "(INTL) EUR",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
    {
        "id": 18,
        "name": "Adriatic-League-Liga-ABA",
        "display_name": "Adriatic League Liga ABA",
        "conf": "(INTL) ABA",
        "years": [2026],
        "url_suffix_by_year": {
            2026: "Draft/points/All/desc/1/Regular_Season",
        },
    },
]

# Output directory - use public/data to match other scripts and website
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Robust Bio Extraction Helper ---
async def _extract_player_bio(page) -> dict:
    """Extract player bio information from profile page."""
    bio_data = {"Height": None, "Weight": None, "Born": None, "NBA Draft": None}
    try:
        await page.wait_for_selector("div.profile-box", timeout=5000)
        texts = await page.locator("div.profile-box p").all_inner_texts()

        for raw_text in texts:
            text = raw_text.replace('\xa0', ' ').strip()

            if "Height:" in text:
                bio_data["Height"] = text.split("Height:", 1)[1].strip()
            elif "Weight:" in text:
                bio_data["Weight"] = text.split("Weight:", 1)[1].strip()
            elif "Born:" in text:
                val = text.split("Born:", 1)[1].strip()
                match = re.search(r"([A-Za-z]+\s+\d+,\s+\d+)", val)
                bio_data["Born"] = match.group(1) if match else val
            elif "NBA Draft:" in text:
                bio_data["NBA Draft"] = text.split("NBA Draft:", 1)[1].strip()
    except Exception:
        pass
    return bio_data


# --- 2. Scrape Season for a League ---
async def scrape_league_season(browser, league: dict, year: int, stat_type: str, get_bios: bool = False) -> pd.DataFrame:
    """
    Scrape a specific league/season/stat_type combination.
    
    Args:
        browser: AsyncCamoufox browser instance
        league: League configuration dict
        year: Season year (e.g., 2026)
        stat_type: "Averages" or "Advanced_Stats"
        get_bios: Whether to fetch player bio data
    
    Returns:
        DataFrame with scraped stats
    """
    # Build URL based on league configuration
    url_suffix = league.get("url_suffix_by_year", {}).get(year, "Draft")
    base_url = f"https://basketball.realgm.com/international/league/{league['id']}/{league['name']}/stats/{year}/{stat_type}/Qualified/{url_suffix}"
    
    print(f"  > [{league['display_name']} {year}] Loading {stat_type}...")
    print(f"     URL: {base_url}")

    context = await browser.new_context(extra_http_headers={"Accept-Encoding": "identity"})
    page = await context.new_page()

    try:
        await page.goto(base_url, timeout=60000, wait_until="domcontentloaded")

        # Check if table exists
        try:
            await page.wait_for_selector("div.fixed-table-container table tbody tr", timeout=10000)
        except:
            print(f"  > [{league['display_name']} {year}] No data found for {stat_type}. Skipping.")
            await context.close()
            return pd.DataFrame()

        # Sort by Player to ensure alignment across different stat pages
        try:
            player_header = page.locator("div.th-inner", has_text="Player").first
            await player_header.click()
            await asyncio.sleep(2.0)
        except:
            print(f"  > [{league['display_name']} {year}] Could not sort by player. Proceeding with default order.")

        # Scrape Table
        table_locator = page.locator("div.fixed-table-container table").nth(1)
        table_html = await table_locator.evaluate("el => el.outerHTML")
        dfs = pd.read_html(table_html)
        df = dfs[0] if dfs else pd.DataFrame()

        if df.empty:
            await context.close()
            return df

        print(f"  > [{league['display_name']} {year}] Captured {len(df)} rows.")

        if get_bios:
            print(f"  > [{league['display_name']} {year}] Grabbing player links...")
            links = await table_locator.locator("tbody tr td:nth-child(2) a").evaluate_all(
                "list => list.map(e => e.href)")

            if len(links) == len(df):
                df["Player Link"] = links
            else:
                print(f"  ! Warning: Row count {len(df)} != Link count {len(links)}")

            await context.close()  # Close main page context to save memory

            print(f"  > [{league['display_name']} {year}] Visiting {len(links)} profiles one-by-one...")

            bios = []
            for i, link in enumerate(links):
                player_context = await browser.new_context(extra_http_headers={"Accept-Encoding": "identity"})
                player_page = await player_context.new_page()

                try:
                    await player_page.goto(link, timeout=45000, wait_until="domcontentloaded")
                    data = await _extract_player_bio(player_page)
                    bios.append(data)
                except Exception as e:
                    print(f"    ! Error visiting link {i + 1}: {e}")
                    bios.append({"Height": None, "Weight": None, "Born": None, "NBA Draft": None})
                finally:
                    await player_context.close()

                await asyncio.sleep(random.uniform(0.4, 0.7))

            bio_df = pd.DataFrame(bios)
            df = pd.concat([df, bio_df], axis=1)
        else:
            await context.close()

    except Exception as e:
        print(f"  ! CRITICAL ERROR in {league['display_name']} {year} {stat_type}: {e}")
        await context.close()
        return pd.DataFrame()

    # Cleanup numeric types
    for c in df.columns:
        if c not in ("Player", "Team", "Height", "Weight", "Born", "NBA Draft", "Player Link"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# --- 3. Main Loop - Iterate through all leagues ---
async def main():
    """
    Main scraping function. Iterates through all configured leagues/years.
    """
    async with AsyncCamoufox(humanize=True, disable_coop=True, window=(1920, 1080)) as browser:
        for league in LEAGUES:
            years = league.get("years", [2026])
            for year in years:
                print(f"\n{'='*60}")
                print(f"=== PROCESSING {league['display_name']} {year} (League ID: {league['id']}) ===")
                print(f"{'='*60}")

                # Phase 1: Averages + Bios
                df_avg = await scrape_league_season(browser, league, year, "Averages", get_bios=True)

                if df_avg.empty:
                    print(f"⚠️  Skipping {league['display_name']} {year} - No data found")
                    continue

                df_avg["Year"] = year
                df_avg["League"] = league['display_name']
                df_avg["League_ID"] = league['id']
                df_avg["Conf"] = league['conf']

                # Phase 2: Advanced Stats (Fast - No Bios)
                df_adv = await scrape_league_season(browser, league, year, "Advanced_Stats", get_bios=False)

                if not df_adv.empty:
                    print(f"  > [{league['display_name']} {year}] Merging data...")
                    # Trim first 3 columns from Advanced (Rank, Player, Team) to avoid duplicates
                    df_adv_trimmed = df_adv.iloc[:, 3:]
                    final_df = pd.concat([df_avg, df_adv_trimmed], axis=1)

                    # Clean filename
                    safe_league_name = league['name'].replace('-', '_')
                    filename = os.path.join(OUTPUT_DIR, f"RealGM_{safe_league_name}_{year}.csv")
                    final_df.to_csv(filename, index=False)
                    print(f"✅ SUCCESS: Saved {filename} ({len(final_df)} rows)")
                else:
                    print(f"⚠️  Warning: {league['display_name']} {year} had Averages but no Advanced Stats.")
                    safe_league_name = league['name'].replace('-', '_')
                    filename = os.path.join(OUTPUT_DIR, f"RealGM_{safe_league_name}_{year}_AvgOnly.csv")
                    df_avg.to_csv(filename, index=False)
                    print(f"✅ Saved partial file: {filename} ({len(df_avg)} rows)")

                # Small delay between leagues to be respectful
                await asyncio.sleep(random.uniform(1.0, 2.0))

    print(f"\n{'='*60}")
    print("✅ ALL LEAGUES PROCESSED!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review the CSV files in the output directory")
    print("2. Import into internationalplayerarchive.csv using appropriate import script")
    print("3. Rebuild the website data if needed")


if __name__ == "__main__":
    asyncio.run(main())
