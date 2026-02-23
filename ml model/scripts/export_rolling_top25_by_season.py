#!/usr/bin/env python3
"""
Export one XLSX with one sheet per season: Top 25 (NBA-mapped) from the rolling runner.

Reads data/inference/rolling_yearly/{year}/rankings_{year}_nba_mapped.csv for each year,
takes top 25 by pred_rank, and writes data/inference/rolling_yearly/rolling_all_seasons_top25.xlsx
with sheets named by year (e.g. 2011, 2012, ...). Also writes a summary sheet from
rolling_summary.csv if present.
"""
from pathlib import Path

import argparse
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_SUBDIR = "rolling_yearly"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top 25 per season from rolling rankings")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdir under data/inference and data/audit (e.g. rolling_yearly or other_rolling)",
    )
    args = parser.parse_args()
    rolling_dir = BASE / "data" / "inference" / args.output_subdir
    summary_csv = BASE / "data" / "audit" / args.output_subdir / "rolling_summary.csv"
    out_xlsx = rolling_dir / "rolling_all_seasons_top25.xlsx"

    if not rolling_dir.exists():
        print(f"Rolling dir not found: {rolling_dir}")
        return
    year_dirs = sorted([d for d in rolling_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not year_dirs:
        print("No year subdirs in", rolling_dir)
        return
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for year_dir in year_dirs:
            year = int(year_dir.name)
            csv_path = year_dir / f"rankings_{year}_nba_mapped.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            rank_col = "pred_rank_nba_only" if "pred_rank_nba_only" in df.columns else "pred_rank"
            if rank_col in df.columns:
                df = df.sort_values(rank_col).head(25)
            else:
                df = df.head(25)
            df.to_excel(writer, sheet_name=str(year), index=False)
        if summary_csv.exists():
            try:
                summary = pd.read_csv(summary_csv)
                summary.to_excel(writer, sheet_name="rolling_summary", index=False)
            except Exception as e:
                print("Summary sheet skip:", e)
    print(f"Wrote {out_xlsx} ({len(year_dirs)} season sheets)")


if __name__ == "__main__":
    main()
