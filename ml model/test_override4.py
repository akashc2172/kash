import pandas as pd
cf = pd.read_parquet('/Users/akashc/my-trankcopy/ml model/data/college_feature_store/college_features_v1.parquet')
print("CF at start of pipeline for Paolo:")
print(cf[(cf['athlete_id']==27623) & (cf['split_id']=='ALL__ALL')][['season', 'split_id']].to_dict('records'))

# Let's bypass to check the unified_training table
un = pd.read_parquet('/Users/akashc/my-trankcopy/ml model/data/training/unified_training_table.parquet')
p = un[un['athlete_id']==27623]
print("unified table Paolo:", p[['college_games_played', 'delta_games_played', 'final_games_played']].to_dict('records'))

import duckdb
con = duckdb.connect('/Users/akashc/my-trankcopy/ml model/data/warehouse.duckdb', read_only=True)
res = con.execute("SELECT * FROM stg_participants WHERE athleteId='27623'").df()
print("Stg participants Paolo games count:", len(res))

