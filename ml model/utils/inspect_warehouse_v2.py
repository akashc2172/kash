
import pandas as pd
import os

base_dir = "data/warehouse_v2"
files = [
    "dim_player_crosswalk.parquet",
    "dim_player_nba.parquet",
    "fact_player_year1_epm.parquet",
    "fact_player_peak_rapm.parquet"
]

print("--- NBA WAREHOUSE V2 SCHEMA ---")
for f in files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"\n### {f}")
        print(f"Rows: {len(df)}")
        print("Columns:")
        for col in df.columns:
            # Get sample non-null value
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "NULL"
            dtype = df[col].dtype
            print(f"- {col} ({dtype}): e.g. {sample}")
    else:
        print(f"\n### {f} NOT FOUND")
