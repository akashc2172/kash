from __future__ import annotations
import pandas as pd
from typing import List
from .warehouse import Warehouse

def export_player_asof_wide(wh: Warehouse, season: int, season_type: str, window_ids: List[str], dest: str):
    # pivot player_window into wide columns: feature_window
    pw = wh.query_df("""
        SELECT * FROM fact_player_window
        WHERE season = ? AND seasonType = ? AND window_id IN ({})
    """.format(",".join(["?"]*len(window_ids))),
    {"1": season, "2": season_type, **{str(i+3): wid for i,wid in enumerate(window_ids)}})
    if pw.empty:
        return
    key_cols = ["athleteId","teamId","season","seasonType","asOfGameId"]
    feat_cols = [c for c in pw.columns if c not in key_cols + ["window_id"]]
    wide = pw.pivot_table(index=key_cols, columns="window_id", values=feat_cols, aggfunc="first")
    # flatten columns
    wide.columns = [f"{feat}_{wid}" for feat, wid in wide.columns]
    wide = wide.reset_index()
    wide.to_parquet(dest, index=False)
