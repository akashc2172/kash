# Torvik Data Overlay (Chrome Extension)

This extension injects your hosted dataset columns into Bart Torvik player tables.

## What it does
- Runs on `https://barttorvik.com/playerstat.php?...`
- Reads `year` from the URL
- Finds your configured dataset for that season
- Fetches JSON or CSV from your URL
- Matches players by name
- Appends your selected columns directly into the Torvik table

Because it appends columns to the existing Torvik table, Torvik's own filters/date ranges still apply to the rows shown.

## Install locally
1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select this folder:
   - `/Users/akashc/my-trankcopy/chrome-extension/torvik-data-overlay`

## Configure
1. Click extension icon -> **Torvik Overlay** popup.
2. Add one dataset per season:
   - Dataset name: any label
   - Season year: e.g. `2026`
   - Dataset URL: your hosted file (JSON or CSV), e.g. `https://checkthesheets.com/torvik-datasets/2026.json`
   - Match key: column used for player names in your dataset, e.g. `player`
   - Team key (optional): recommended `team` for better match quality
   - Columns to inject: comma-separated column names, e.g. `proj_nba, intl_usage, age`
3. Open a Torvik playerstat URL and refresh.

## Dataset format
### JSON (recommended)
```json
[
  {
    "player": "John Doe",
    "proj_nba": 0.62,
    "intl_usage": 24.1,
    "age": 19.4
  }
]
```

### CSV
```csv
player,proj_nba,intl_usage,age
John Doe,0.62,24.1,19.4
```

## Notes
- Matching uses `player + team` first, then falls back to `player` (accents/punctuation ignored).
- If names differ a lot, add alias rows in your dataset.
- If your host blocks cross-origin requests, enable CORS for Bart Torvik pages.
