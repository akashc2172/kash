# RealGM Multi-League Scraper

## Overview

Updated `realgm_scrape.py` to iteratively scrape multiple RealGM leagues for 2026 season data. The scraper uses `camoufox` (browser automation) to fetch stats from RealGM.

## Leagues Scraped

The scraper processes these leagues:

1. **French Jeep Elite** (League ID: 12) → Conference: `(INTL) FRA`
2. **Euroleague** (League ID: 1) → Conference: `(INTL) EUR`
3. **French LNB Espoirs** (League ID: 114) → Conference: `(INTL) FRA ESP`
4. **Australian NBL** (League ID: 5) → Conference: `(INTL) AUS`
5. **Spanish ACB** (League ID: 4) → Conference: `(INTL) ESP`
6. **Turkish BSL** (League ID: 7) → Conference: `(INTL) TUR`
7. **Eurocup** (League ID: 2) → Conference: `(INTL) EUR`
8. **Adriatic League Liga ABA** (League ID: 18) → Conference: `(INTL) ABA`

## Conference Names

Based on `internationalplayerarchive.csv`:
- **Espoirs**: `(INTL) FRA ESP` (as seen in `import_lnb_espoirs.py`)
- Other leagues use standard `(INTL) [COUNTRY_CODE]` format

## Usage

### Step 1: Run the Scraper

```bash
cd my-trank/scripts
python realgm_scrape.py
```

This will:
- Open a browser (camoufox)
- Visit each league's stats page
- Scrape "Averages" and "Advanced_Stats" tables
- Extract player bio data (height, weight, birth date, NBA draft info)
- Save CSV files to `public/data/international_stat_history/`

**Output files**: `RealGM_[League_Name]_2026.csv`

### Step 2: Import into Archive

```bash
python import_realgm_leagues.py
```

This will:
- Find all `RealGM_*.csv` files
- Map them to appropriate conference names
- Import into `internationalplayerarchive.csv`
- Handle duplicates (appends, may need manual deduplication)

### Step 3: Update Website Data (if needed)

After importing, you may need to rebuild the website data:

```bash
# If you have a build script that processes international data
# Run it here
```

## File Structure

```
my-trank/
├── scripts/
│   ├── realgm_scrape.py          # Main scraper (updated)
│   └── import_realgm_leagues.py   # Generic importer (new)
└── public/
    └── data/
        └── international_stat_history/
            ├── internationalplayerarchive.csv  # Target archive
            └── RealGM_*.csv                    # Scraped files
```

## Technical Details

### URL Formats

Most leagues use standard format:
```
https://basketball.realgm.com/international/league/{ID}/{Name}/stats/{Year}/Averages/Qualified/All
```

**Euroleague** uses special format with sorting:
```
https://basketball.realgm.com/international/league/1/Euroleague/stats/2026/Averages/Qualified/Draft/points/All/desc/1/Regular_Season
```

The scraper handles both formats automatically.

### Browser Automation

Uses `camoufox` (similar to Playwright) for:
- Rendering JavaScript-heavy pages
- Extracting table data
- Visiting player profile pages for bio data
- Human-like delays to avoid rate limiting

### Data Mapping

The import script maps RealGM columns to archive format:
- `Player` → `key`
- `Team` → `team`
- `GP` → `g`
- `MPG` → `mpg`
- `PPG` → `ppg`
- `FGM/FGA/FG%` → `fgm/fga/fg_pct`
- `3PM/3PA/3P%` → `three_m/three_a/3p%`
- `FTM/FTA/FT%` → `ftm/fta/ft_pct`
- Advanced stats (ORB%, DRB%, USG%, ORtg, DRtg, etc.)
- Bio data (Height → `hoop_hgt_in`, Weight → `weight`)

## Notes

- **Duplicates**: The import script appends all rows. You may want to manually deduplicate if re-running.
- **Rate Limiting**: The scraper includes delays between requests. Be patient - it may take 30-60 minutes to complete all leagues.
- **Errors**: If a league fails, the scraper continues with the next league. Check the output for warnings.
- **Euroleague**: Special URL handling is included for the Euroleague's unique URL structure.

## Troubleshooting

### "No data found"
- Check if the URL is correct for 2026
- Some leagues may not have 2026 data yet
- Try visiting the URL manually in a browser

### "camoufox not found"
```bash
pip install camoufox
```

### Import errors
- Check that `internationalplayerarchive.csv` exists
- Verify column names match expected format
- Check for encoding issues in CSV files

## Next Steps

After scraping and importing:
1. Review the imported data in `internationalplayerarchive.csv`
2. Rebuild website data if your pipeline processes international stats
3. Test the website to ensure new data appears correctly
