from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .warehouse import Warehouse

def build_windows_player(wh: Warehouse, season: int, season_type: str, window_ids: List[str]):
    # canonical: long table keyed by athleteId/teamId/season/asOfGameId/window_id
    pg = wh.query_df("""
        SELECT pg.*, g.season, g.seasonType, g.startTime, g.homeTeamId, g.awayTeamId
        FROM fact_player_game pg
        JOIN dim_games g ON g.id = pg.gameId
        WHERE g.season = ? AND g.seasonType = ?
    """, {"1": season, "2": season_type})
    if pg.empty:
        return

    pg["startTime"] = pd.to_datetime(pg["startTime"], errors="coerce")
    pg = pg.sort_values(["athleteId","startTime","gameId"])

    # We compute rolling windows per athleteId (teamId-season aware)
    keys = ["athleteId","teamId","season","seasonType"]
    features = [c for c in pg.columns if c not in ("athleteId","teamId","season","seasonType","startTime","gameId","homeTeamId","awayTeamId")]
    # Identify which features are counts vs rates. Rule: columns ending with _att/_made/_assisted_att/_unassisted_att etc are counts; rates recompute.
    count_like = [c for c in features if any(c.endswith(suf) for suf in ["_att","_made","_assisted_att","_assisted_made","_unassisted_att","_unassisted_made","shots_att","shots_made","possessions_on","seconds_on"])]
    # We'll roll sums for count_like and also keep sums for denominators, then recompute rate-like columns where possible by leaving NaN (you can fill later).
    denom_cols = [c for c in ["total_fga","possessions_on","seconds_on","minutes","rim_att","jump_att","three_pt_jumper_att"] if c in pg.columns]

    rows=[]
    for (aid, tid, s, stype), grp in pg.groupby(keys):
        grp = grp.sort_values(["startTime","gameId"]).reset_index(drop=True)
        game_ids = grp["gameId"].tolist()
        for i, asof_gid in enumerate(game_ids):
            # define index set for season_to_date
            idx_std = list(range(0,i+1))
            # rolling10
            idx_r10 = list(range(max(0,i-9), i+1))
            idx_r5 = list(range(max(0,i-4), i+1))
            idx_r15 = list(range(max(0,i-14), i+1))

            window_map = {
                "season_to_date": idx_std,
                "rolling10": idx_r10,
                "rolling5": idx_r5,
                "rolling15": idx_r15,
            }
            for wid in window_ids:
                if wid not in window_map:
                    # filters handled elsewhere (conference/top quartile etc) - stub
                    continue
                idx = window_map[wid]
                sub = grp.iloc[idx]
                rec = {"athleteId": aid, "teamId": tid, "season": s, "seasonType": stype, "asOfGameId": asof_gid, "window_id": wid}
                # sum counts
                for c in count_like:
                    rec[c] = float(pd.to_numeric(sub[c], errors="coerce").fillna(0).sum())
                # recompute some rates if present
                if "rim_share" in pg.columns and "rim_att" in pg.columns and "total_fga" in pg.columns:
                    rim_att = float(pd.to_numeric(sub["rim_att"], errors="coerce").fillna(0).sum())
                    total = float(pd.to_numeric(sub["total_fga"], errors="coerce").fillna(0).sum())
                    rec["rim_share"] = rim_att / total if total>0 else np.nan
                if "three_share" in pg.columns and "three_pt_jumper_att" in pg.columns and "total_fga" in pg.columns:
                    three = float(pd.to_numeric(sub["three_pt_jumper_att"], errors="coerce").fillna(0).sum())
                    total = float(pd.to_numeric(sub["total_fga"], errors="coerce").fillna(0).sum())
                    rec["three_share"] = three / total if total>0 else np.nan
                # per100 recompute
                if "tov_per100" in pg.columns and "possessions_on" in pg.columns and "turnovers" in sub.columns:
                    pass
                rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        wh.ensure_table("fact_player_window", out, pk=["athleteId","teamId","season","seasonType","asOfGameId","window_id"])

def build_windows_team(wh: Warehouse, season: int, season_type: str, window_ids: List[str]):
    tg = wh.query_df("""
        SELECT tg.*, g.season, g.seasonType, g.startTime
        FROM fact_team_game tg
        JOIN dim_games g ON g.id = tg.gameId
        WHERE g.season = ? AND g.seasonType = ?
    """, {"1": season, "2": season_type})
    if tg.empty:
        return
    tg["startTime"] = pd.to_datetime(tg["startTime"], errors="coerce")
    tg = tg.sort_values(["teamId","startTime","gameId"])
    keys = ["teamId","season","seasonType"]
    count_like = [c for c in tg.columns if c in ["possessions_game","seconds_game"]]
    rows=[]
    for (tid,s,stype), grp in tg.groupby(keys):
        grp = grp.sort_values(["startTime","gameId"]).reset_index(drop=True)
        game_ids = grp["gameId"].tolist()
        for i, asof_gid in enumerate(game_ids):
            idx_std = list(range(0,i+1))
            idx_r10 = list(range(max(0,i-9), i+1))
            window_map={"season_to_date":idx_std,"rolling10":idx_r10}
            for wid in window_ids:
                if wid not in window_map: 
                    continue
                sub = grp.iloc[window_map[wid]]
                rec={"teamId":tid,"season":s,"seasonType":stype,"asOfGameId":asof_gid,"window_id":wid}
                for c in count_like:
                    rec[c]=float(pd.to_numeric(sub[c], errors="coerce").fillna(0).sum())
                for c in ["pace","offenseRating","defenseRating","netRating"]:
                    if c in grp.columns and "possessions_game" in grp.columns:
                        poss = pd.to_numeric(sub["possessions_game"], errors="coerce").fillna(0)
                        val = pd.to_numeric(sub[c], errors="coerce")
                        rec[c]=float((val*poss).sum()/poss.sum()) if poss.sum()>0 else np.nan
                rows.append(rec)
    out=pd.DataFrame(rows)
    if not out.empty:
        wh.ensure_table("fact_team_window", out, pk=["teamId","season","seasonType","asOfGameId","window_id"])
