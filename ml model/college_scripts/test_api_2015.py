import requests
import json
import random

def test_api():
    # 1. Get Scoreboard for a random date in 2015
    date_url = "https://ncaa-api.henrygd.me/scoreboard/basketball-men/d1/2015/02/07"
    print(f"Fetching: {date_url}")
    
    try:
        resp = requests.get(date_url)
        games = resp.json()
        
        if not games or 'games' not in games:
            print("No games found or invalid format.")
            print(str(games)[:500])
            return

        game_list = games['games']
        print(f"Found {len(game_list)} games.")
        
        if not game_list:
            return

        # 2. Pick a random game
        target_game = random.choice(game_list)
        
        print("\n--- RAW GAME OBJECT ---")
        print(json.dumps(target_game, indent=2))
        
        # Try to find numeric ID
        # Common patterns: 'game': {'id': 123} or 'game': {'url': '/game/123'}
        # Previous run showed url was text slug.
        
        # Try Schedule Endpoint
        schedule_url = "https://ncaa-api.henrygd.me/schedule/basketball-men/d1/2015/02"
        print(f"Fetching Schedule: {schedule_url}")
        r = requests.get(schedule_url)
        if r.status_code == 200:
             print("Schedule Data:")
             print(json.dumps(r.json(), indent=2)[:1000])
        else:
             print(f"Schedule Failed: {r.status_code}")


        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
