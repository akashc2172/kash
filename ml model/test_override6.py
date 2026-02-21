import pandas as pd
import numpy as np
from nba_scripts.build_unified_training_table import (
    load_college_features, load_historical_exposure_backfill, load_derived_box_stats,
    load_historical_text_games_backfill, get_final_college_season
)

cf = pd.read_parquet('/Users/akashc/my-trankcopy/ml model/data/college_feature_store/college_features_v1.parquet')
cf = cf[(cf['athlete_id']==27623) & (cf['split_id']=='ALL__ALL')]

der = load_derived_box_stats()
cf = cf.merge(der, on=['athlete_id', 'season'], how='left')

for stat in ['ast', 'stl', 'blk', 'tov']:
    derived_col = f'college_{stat}_total'
    target_col = f'{stat}_total'
    cf[target_col] = pd.to_numeric(cf.get(target_col, np.nan)).combine_first(pd.to_numeric(cf.get(derived_col, np.nan)))

if 'college_games_played' in cf.columns:
    cf['games_played'] = pd.to_numeric(cf['college_games_played'], errors='coerce').combine_first(pd.to_numeric(cf.get('games_played', np.nan), errors='coerce'))
    cf = cf.drop(columns=['college_games_played'])

bak = load_historical_exposure_backfill()
cf = cf.merge(bak, on=['athlete_id', 'season'], how='left')

if 'games_played' in cf.columns:
    games_existing = pd.to_numeric(cf['games_played'], errors='coerce')
    games_backfill = pd.to_numeric(cf.get('backfill_games_played'), errors='coerce')
    cf['games_played'] = pd.Series(
        np.where((games_backfill > 0) & ((games_existing.isna()) | (games_existing <= 0) | ((games_backfill - games_existing) >= 5)),
                 games_backfill, games_existing), 
        index=cf.index
    )

hist = load_historical_text_games_backfill()
cf = cf.merge(hist, on=['athlete_id', 'season'], how='left')
hist_games = pd.to_numeric(cf.get('hist_games_played_text'), errors='coerce')
if 'games_played' in cf.columns:
    games_existing = pd.to_numeric(cf['games_played'], errors='coerce')
    cf['games_played'] = pd.Series(
        np.where((hist_games > 0) & ((games_existing.isna()) | (games_existing <= 0) | ((hist_games - games_existing) >= 5)),
                 hist_games, games_existing),
        index=cf.index
    )

print("Games after full pipeline:", cf['games_played'].tolist())

final = get_final_college_season(cf)
print("Final college games played:", final['college_games_played'].tolist() if 'college_games_played' in final.columns else final['games_played'].tolist())
