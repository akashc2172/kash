import sys, re
import pandas as pd
import duckdb

sys.path.insert(0, '/Users/akashc/my-trankcopy/ml model')
from nba_scripts.build_unified_training_table import _norm_hist_player_name

def _norm_hist_team_name(name):
    if not isinstance(name, str): return ""
    name = re.sub(r"^#\d+\s+", "", name.strip())
    name = name.upper().replace("STATE", "ST").replace("NORTH ", "N ").replace("SOUTH ", "S ")
    return re.sub(r"[^A-Z]+", "", name)

con = duckdb.connect('/Users/akashc/my-trankcopy/ml model/data/warehouse.duckdb', read_only=True)
bridge = con.execute("""
    WITH b AS (
        SELECT CAST(g.season AS BIGINT) AS season, CAST(s.shooterAthleteId AS BIGINT) AS athlete_id, s.shooter_name, t.school AS team_name, COUNT(*) AS shots
        FROM stg_shots s
        JOIN dim_games g ON CAST(s.gameId AS BIGINT) = g.id
        LEFT JOIN dim_teams t ON t.id = CAST(s.teamId AS BIGINT)
        WHERE s.shooterAthleteId IS NOT NULL AND s.shooter_name IS NOT NULL AND g.season IS NOT NULL
        GROUP BY 1,2,3,4
    )
    SELECT * FROM b WHERE shooter_name ILIKE '%Zach Edey%' OR shooter_name ILIKE '%Jalen Johnson%'
""").df()

bf = pd.read_parquet('/Users/akashc/my-trankcopy/ml model/data/warehouse_v2/fact_player_season_stats_backfill_manual_subs.parquet')
bf = bf[bf['player_name'].str.contains('Edey', case=False, na=False) | bf['player_name'].str.contains('Jalen Johnson', case=False, na=False)].copy()
bf["season"] = pd.to_numeric(bf["season"], errors="coerce").astype("Int64") + 1
bf["norm_name"] = bf["player_name"].map(_norm_hist_player_name)
bf["norm_team"] = bf.get("team_name", pd.Series(dtype=str)).map(_norm_hist_team_name)

bridge["season"] = pd.to_numeric(bridge["season"], errors="coerce").astype("Int64")
bridge["athlete_id"] = pd.to_numeric(bridge["athlete_id"], errors="coerce").astype("Int64")
bridge["norm_name"] = bridge["shooter_name"].map(_norm_hist_player_name)
bridge["norm_team"] = bridge["team_name"].map(_norm_hist_team_name)

print("BF Edey:", bf[bf['norm_name'] == 'ZACHEDEY'][['season', 'team_name']])
print("Bridge Edey lengths per season:")
print(bridge[bridge['norm_name'] == 'ZACHEDEY'].groupby('season').size())

bridge_unique = bridge.groupby(["season", "norm_name"]).filter(lambda x: len(x) == 1)
print("Bridge Unique Edey:")
print(bridge_unique[bridge_unique['norm_name'] == 'ZACHEDEY'])
mapped_exact = bf.merge(bridge[["season", "norm_name", "norm_team", "athlete_id"]], on=["season", "norm_name", "norm_team"], how="inner")
print("Mapped Exact Edey:")
print(mapped_exact[mapped_exact['norm_name'] == 'ZACHEDEY'])
