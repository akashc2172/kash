import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cbbd

key = None
with open(".env") as f:
    for line in f:
        if line.startswith("CBD_API_KEY"):
            key = line.split("=")[1].strip().strip('"').strip("'")

config = cbbd.Configuration(host="https://api.collegebasketballdata.com", access_token=key)
with cbbd.ApiClient(config) as client:
    api = cbbd.PlayersApi(client)
    res = api.search_players(search_term="Paolo Banchero")
    if res:
        print("Player Search Data:", res[0].to_dict())
    
    s_api = cbbd.StatsApi(client)
    stats = s_api.get_player_season_stats(season=2022, team="Duke")
    for s in stats:
        if "Banchero" in s.name:
            print("Player Season Stats Data:", s.to_dict())
