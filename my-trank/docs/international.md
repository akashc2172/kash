# International (RealGM) Updates

## Scrape
Run the scraper in an environment with browser automation enabled (Camoufox).

```bash
python3 scripts/realgm_scrape.py
```

## Import into archives
```bash
python3 scripts/import_2026_international.py
python3 scripts/import_realgm_leagues.py
```

Outputs:
- `public/data/international_stat_history/2026_records.csv`
- `public/data/international_stat_history/internationalplayerarchive.csv`

## Dedupe behavior (2026)
- Imports now dedupe on `torvik_id + torvik_year` for 2026.
- If duplicates appear, the row with the most non-null fields is kept.
