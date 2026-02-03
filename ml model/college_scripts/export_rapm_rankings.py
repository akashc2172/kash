import pandas as pd
import os
import argparse

# Configuration
INPUT_CSV = 'data/historical_rapm_results_lambda1000.csv'
OUTPUT_XLSX = 'data/historical_rapm_rankings.xlsx'

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.")
        return

    print(f"Reading RApM results from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    if 'season' not in df.columns:
        print("Error: 'season' column missing in input CSV.")
        return

    # Create Excel writer
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        unique_seasons = sorted(df['season'].unique())
        print(f"Found seasons: {unique_seasons}")

        for season in unique_seasons:
            season_df = df[df['season'] == season].copy()
            # Sort by RApM descending
            season_df = season_df.sort_values(by='rapm', ascending=False)
            
            # Write to sheet
            sheet_name = str(season)
            season_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  > Wrote {len(season_df)} rows to sheet '{sheet_name}'")

    print(f"Successfully saved rankings to {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
