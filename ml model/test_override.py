from nba_scripts.build_unified_training_table import (
    load_college_features, load_historical_exposure_backfill, load_derived_box_stats
)
import pandas as pd
import numpy as np

try:
    cf = load_college_features()
    cf = cf[(cf['athlete_id']==27623) & (cf['split_id']=='ALL__ALL')]
    print("CF after load:", len(cf))

    der = load_derived_box_stats()
    cf = cf.merge(der, on=['athlete_id', 'season'], how='left')
    print("CF after der:", len(cf))
    if 'college_games_played' in cf.columns:
        cf['games_played'] = pd.to_numeric(cf['college_games_played'], errors='coerce')

    print("Calling history backfill...")
    bak = load_historical_exposure_backfill()
    print("Backfill length:", len(bak))

    cf = cf.merge(bak, on=['athlete_id', 'season'], how='left')
    print("CF after bak:", len(cf))

    if 'games_played' in cf:
        games_existing = pd.to_numeric(cf['games_played'], errors='coerce')
        games_backfill = pd.to_numeric(cf.get('backfill_games_played'), errors='coerce')
        print("Existing:", games_existing.tolist())
        print("Backfill:", games_backfill.tolist())

        cond = (games_backfill > 0) & ((games_existing.isna()) | (games_existing <= 0) | ((games_backfill - games_existing) >= 5))
        games_new = np.where(cond, games_backfill, games_existing)
        print("Resulting games:", games_new.tolist())
except Exception as e:
    import traceback
    traceback.print_exc()
