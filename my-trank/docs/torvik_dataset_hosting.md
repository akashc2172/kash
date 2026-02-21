# Torvik Dataset Hosting

This site now hosts static files for the Torvik Chrome extension at:

- `/torvik-datasets/manifest.json`
- `/torvik-datasets/2026.json`
- `/torvik-datasets/torvik-column-map.json`
- `/torvik-datasets/<year>.json` for 2008-2026

## Why this path

It does not touch existing data pipelines (`/data/*`) and is safe to deploy as static assets.

## Build / refresh datasets

Datasets are generated from:

- `public/data/international_stat_history/internationalplayerarchive.csv`
- `public/data/international_stat_history/2026_records.csv`

Run:

```bash
python3 scripts/build_torvik_overlay_datasets.py
```

## Expected extension config

- URL: `https://checkthesheets.com/torvik-datasets/2026.json`
- Match key: `player`
- Team key: `team`
- Columns: any custom fields in your dataset (for example `cts_proj,cts_role,cts_notes`)
