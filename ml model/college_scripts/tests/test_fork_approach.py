import requests
from bs4 import BeautifulSoup
import sys

# Using a standard Browser User-Agent to avoid immediate 403s (if basic blocking is in place)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
}

BASE_URL = "https://stats.ncaa.org"

def run_test():
    print("--- 1. Fetching Main Page to Find Season ID for 2014-15 ---")
    try:
        # We need to land on the men's basketball page to get the right dropdowns
        # Usually /sports/men/basketball
        r = requests.get(f"{BASE_URL}/", headers=HEADERS)
        if r.status_code != 200:
            print(f"FAILED to fetch main page: {r.status_code}")
            return
            
        soup = BeautifulSoup(r.text, "html.parser")
        
        # The dropdown usually has id 'game_sport_year_ctl_id' or similar, 
        # but it might only populate after selecting sport. 
        # However, typically strict HTTP scrapers might just see the default (which might be generic).
        # Let's check if we can find the list of seasons.
        
        # In the real browser, one clicks "Sport" first. 
        # But often the hidden select 'game_sport_year_ctl_id' contains all valid combinations or is loaded via JS.
        # Let's try to query the filter directly if we can guess the ID or find it.
        
        # A common ID for Men's Basketball 2022-23 was 16060.
        # Let's try to search the HTML for "2014-15"
        
        # Actually, let's just try to hit the scoreboard with a known working ID from 2015 if we can find one online,
        # OR just iterate/search the select options if they exist.
        
        season_select = soup.find("select", {"id": "game_sport_year_ctl_id"})
        target_id = None
        
        if season_select:
            print("Found Season Select!")
            for opt in season_select.find_all("option"):
                text = opt.get_text().strip()
                val = opt.get("value")
                # print(f"  Option: {text} -> {val}")
                if "2014-15" in text and "Men's Basketball" in text:
                    target_id = val
                    print(f"  -> MATCH FOUND: {text} = {val}")
                    break
                elif "2014-15" in text:
                     # Fallback match
                     print(f"  -> Potential Match: {text} = {val}")
                     target_id = val
        else:
            print("Season select not found in raw HTML (likely JS populated).")
            # If JS populated, we might need to rely on the camoufox discovery or hardcoded knowledge.
            # But let's try to hit the endpoint with a GUESS or just proceed to see if we get ANY response.
            pass

        # If we didn't find it, we can't easily proceed without the robust scraper. 
        # BUT, the user's fork uses `season_division_id`.
        # Let's try to mimic the fork's request structure.
        
        # The fork uses: stats.ncaa.org/contests/livestream_scoreboards
        # Params: season_division_id, game_date
        
        # Let's assume we can't find the ID easily via requests and focus on Step 2:
        # Can we even Hit the URL without 403?
        
        print("\n--- 2. Testing Access to Scoreboard Endpoint ---")
        # I'll just use a recent known ID or a random one to test connectivity.
        # If this 403s, the whole requests approach is dead.
        
        test_url = f"{BASE_URL}/contests/livestream_scoreboards"
        params = {
            "game_date": "02/07/2015",
            # We skip season_id for a moment to see if it defaults or errors
        }
        
        r2 = requests.get(test_url, params=params, headers=HEADERS)
        print(f"Status Code: {r2.status_code}")
        
        if r2.status_code == 403:
            print("❌ 403 Forbidden - Standard requests are blocked.")
            print("The fork likely runs in an environment with different IP rep or headers, or we need to mimic better.")
            return
            
        if "0 games" in r2.text:
            print("Accessed page, but 0 games (expected without valid ID).")
        else:
            print("Accessed page and got content!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_test()
