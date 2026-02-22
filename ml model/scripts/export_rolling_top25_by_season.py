#!/usr/bin/env python3
"""
Export one XLSX with one sheet per season: Top 25 (NBA-mapped) from the rolling runner.

Reads data/inference/rolling_yearly/{year}/rankings_{year}_nba_mapped.csv for each year,
takes top 25 by pred_rank, and writes data/inference/rolling_yearly/rolling_all_seasons_top25.xlsx
with sheets named by year (e.g. 2011, 2012, ...). Also writes a summary sheet from
rolling_summary.csv if present.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROLLING_DIR = BASE / "data" / "inference" / "rolling_yearly"
SUMMARY_CSV = BASE / "data" / "audit" / "rolling_yearly" / "rolling_summary.csv"
OUT_XLSX = ROLLING_DIR / "rolling_all_seasons_top25.xlsx"


def main() -> None:
    if not ROLLING_DIR.exists():
        print(f"Rolling dir not found: {ROLLING_DIR}")
        return
    year_dirs = sorted([d for d in ROLLING_DIR.iterdir() if d.is_dir() and d.name.isdigit()])
    if not year_dirs:
        print("No year subdirs in", ROLLING_DIR)
        return
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        for year_dir in year_dirs:
            year = int(year_dir.name)
            csv_path = year_dir / f"rankings_{year}_nba_mapped.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if "pred_rank" in df.columns:
                df = df.sort_values("pred_rank").head(25)
            else:
                df = df.head(25)
            df.to_excel(writer, sheet_name=str(year), index=False)
        if SUMMARY_CSV.exists():
            try:
                summary = pd.read_csv(SUMMARY_CSV)
                summary.to_excel(writer, sheet_name="rolling_summary", index=False)
            except Exception as e:
                print("Summary sheet skip:", e)
    print(f"Wrote {OUT_XLSX} ({len(year_dirs)} season sheets)")


if __name__ == "__main__":
    main()
