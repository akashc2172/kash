from nba_scripts.build_unified_training_table import (
    load_college_features, load_historical_exposure_backfill, load_derived_box_stats, load_historical_text_games_backfill
)
import pandas as pd
import numpy as np

b = load_historical_exposure_backfill()
try:
    print("Exposure Backfill Paolo:", b[b['athlete_id']==27623][['athlete_id', 'season', 'backfill_games_played']].to_dict('records'))
except Exception as e:
    print("Exposure backfill games_played error:", e)

t = load_historical_text_games_backfill()
try:
    print("Text Backfill Paolo:", t[t['athlete_id']==27623][['athlete_id', 'season', 'hist_games_played_text']].to_dict('records'))
except Exception as e:
    print("Text backfill error:", e)
