#!/usr/bin/env python3
"""
Verify that Banchero and Holmgren (2022 superstars) are no longer null
for ctx_adj_onoff_net and path_onoff_source in the supervised training table.
Run after rebuilding pathway context and unified training table.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
TABLE_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"

def main():
    if not TABLE_PATH.exists():
        print(f"Table not found: {TABLE_PATH}")
        return
    import pyarrow.parquet as pq
    schema = pq.read_schema(TABLE_PATH)
    all_cols = schema.names
    cols = ["player_name", "college_final_season", "ctx_adj_onoff_net", "path_onoff_source"]
    need = [c for c in cols if c in all_cols]
    if not need:
        need = [c for c in ["athlete_id", "college_final_season", "ctx_adj_onoff_net", "path_onoff_source"] if c in all_cols]
    df = pd.read_parquet(TABLE_PATH, columns=need)
    if "player_name" not in df.columns:
        print("player_name not in table; checking by college_final_season=2022 and athlete_id sample.")
        sub = df[df["college_final_season"] == 2022].head(20)
        print(sub.to_string())
        return
    stars = df[df["player_name"].str.contains("Banchero|Holmgren", case=False, na=False)]
    print("Superstar rescue verification (Banchero / Holmgren):")
    print(stars[need].to_string(index=False))
    if stars.empty:
        print("No rows found for Banchero or Holmgren.")
        return
    nonnull = stars["ctx_adj_onoff_net"].notna().all() and stars["path_onoff_source"].notna().all()
    if nonnull:
        print("\nPASS: ctx_adj_onoff_net and path_onoff_source are non-null for these players.")
    else:
        print("\nFAIL: Some ctx_adj_onoff_net or path_onoff_source are still null.")

if __name__ == "__main__":
    main()
