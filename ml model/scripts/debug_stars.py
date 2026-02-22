#!/usr/bin/env python3
"""
Forensic trace for Banchero/Holmgren: find which raw input is NaN and causes
ctx_adj_onoff_net to be NaN in the pathway context.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE / "data" / "college_feature_store"
SUP_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"

IMPACT_PATH = FEATURE_DIR / "college_impact_stack_v1.parquet"
FEATURES_PATH = FEATURE_DIR / "college_features_v1.parquet"
PATHWAY_PATH = FEATURE_DIR / "college_pathway_context_v2.parquet"


def main():
    # Resolve athlete_id and college_final_season from supervised table (has names)
    if not SUP_PATH.exists():
        print("Supervised table not found; using known athlete_ids for 2022 draft.")
        aid_seasons = {(4432158, 2022), (4432158, 2021)}
        print("Using athlete_id 4432158 (Banchero) for seasons 2021, 2022.")
    else:
        sup = pd.read_parquet(SUP_PATH, columns=["player_name", "athlete_id", "college_final_season"])
        stars = sup[sup["player_name"].str.contains("Banchero|Holmgren", case=False, na=False)]
        if stars.empty:
            print("No Banchero/Holmgren in supervised table.")
            return
        aid_seasons = set()
        for _, r in stars.iterrows():
            aid, s = r["athlete_id"], r["college_final_season"]
            if pd.notna(aid) and pd.notna(s):
                aid_seasons.add((int(aid), int(s)))
                aid_seasons.add((int(aid), int(s) - 1))
        print("Stars from supervised table:")
        print(stars[["player_name", "athlete_id", "college_final_season"]].to_string(index=False))
        print()

    impact_cols = ["athlete_id", "season", "impact_on_off_net_diff_raw", "impact_on_net_raw",
                   "impact_poss_total", "impact_seconds_total", "has_impact_raw",
                   "rIPM_tot_std", "rIPM_off_std", "rIPM_def_std"]
    feat_cols = ["athlete_id", "season", "minutes_total", "poss_total", "team_pace",
                 "fga_total", "ast_total", "trb_total", "stl_total", "blk_total", "tov_total",
                 "rim_made", "three_made", "mid_made", "ft_made"]
    pctx_cols = ["athlete_id", "season", "ctx_adj_onoff_net", "path_onoff_source", "ctx_quality_flag",
                 "path_onoff_reliability_weight", "ctx_reliability_weight"]

    impact = pd.read_parquet(IMPACT_PATH)
    impact = impact[[c for c in impact_cols if c in impact.columns]]
    feats = pd.read_parquet(FEATURES_PATH)
    feats = feats[[c for c in feat_cols if c in feats.columns]]
    pctx = pd.read_parquet(PATHWAY_PATH)
    pctx = pctx[[c for c in pctx_cols if c in pctx.columns]]

    if "rim_made" in feats.columns and "three_made" in feats.columns:
        r = feats["rim_made"].fillna(0)
        m = feats["mid_made"].fillna(0)
        t = feats["three_made"].fillna(0)
        f = feats["ft_made"].fillna(0)
        feats["points_proxy"] = 2 * (r + m) + 3 * t + f
    else:
        feats["points_proxy"] = np.nan

    def filter_stars(df):
        if "athlete_id" not in df.columns or "season" not in df.columns:
            return pd.DataFrame()
        mask = pd.Series(False, index=df.index)
        for (aid, s) in aid_seasons:
            mask = mask | ((df["athlete_id"] == aid) & (df["season"] == s))
        return df[mask].drop_duplicates(subset=["athlete_id", "season"])

    imp = filter_stars(impact)
    fe = filter_stars(feats)
    pc = filter_stars(pctx)

    print("=== 1. college_impact_stack_v1 (raw impact inputs) ===")
    if imp.empty:
        print("No impact rows for these athlete_id/season.")
    else:
        print(imp.to_string())

    print("\n=== 2. college_features_v1 (minutes, poss, box) ===")
    if fe.empty:
        print("No feature rows for these athlete_id/season.")
    else:
        print(fe.to_string())

    print("\n=== 3. college_pathway_context_v2 (output) ===")
    if pc.empty:
        print("No pathway rows for these athlete_id/season.")
    else:
        print(pc.to_string())

    print("\n=== NaN summary ===")
    for name, frame in [("impact", imp), ("feats", fe), ("pathway", pc)]:
        if frame.empty:
            continue
        for c in frame.columns:
            n = frame[c].isna().sum()
            if n > 0:
                print(f"  {name}.{c}: {n} NaN")

    # Infer poison: pathway has ctx_adj_onoff_net = onoff_net * rel; if rel is NaN, result is NaN.
    print("\n=== Inferred cause ===")
    print("ctx_adj_onoff_net = onoff_net * rel. If rel is NaN (from poss), product is NaN.")
    print("Fix: use safe rel (e.g. fillna(0.5)) when onoff_net is valid so value is not poisoned.")


if __name__ == "__main__":
    main()
