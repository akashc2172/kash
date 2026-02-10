# Data Pipeline Overview

## Sources
- **Hoop Explorer** (RAPM and core play-by-play derived stats)
- **Bart Torvik** (box score + advanced metrics, 2026 CSV export)
- **RealGM** (international players)
- **Basketball Reference** (NBA career stats)

## Outputs used by the site
- `public/data/season.csv`
- `public/data/career.csv`
- `public/data/international_stat_history/2026_records.csv`
- `public/data/international_stat_history/internationalplayerarchive.csv`
- `public/data/br_advanced_stats.csv`

## High-level flow
1. Update Torvik 2026 CSV
2. Run `scripts/build_data.R` (generates `season.csv` + `career.csv`)
3. Update RealGM data (scrape + import scripts)
4. Build site (Vite)

## Known gotchas
- `npm run build` does **not** rebuild the data CSVs.
- Always run `Rscript scripts/build_data.R` after Torvik changes.
- International duplicates should not exist for `torvik_year=2026` after running the import scripts.
