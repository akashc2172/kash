# new-trank (CTS NCAA Data Warehouse)

This folder is the **future-proof NCAA data pipeline** for Check The Sheets.

Goals:
- Compute *our own* NCAA player/team stats (instead of depending on archived CSVs).
- Publish a stable, versioned export format for the website (`my-trank/public/data/`).
- Keep the ML work (`ml model/`) isolated: **no writes, no imports, no coupling**.

Non-goals (for now):
- Replacing the international pipeline (RealGM stays as-is).
- Forcing site deploys automatically. Publishing is **explicit**.

## Folder Layout
- `new-trank/data/`:
  - Local DuckDB and raw ingests (not committed; add your own `.gitignore` as needed).
- `new-trank/exports/`:
  - Site-ready exports like `season.csv`, `career.csv`, `archive.csv`, `stat_dictionary.json`.
- `new-trank/scripts/`:
  - Ingest + build + export + publish scripts.

## Current Status
This is a **scaffold** that lets us start wiring the site to a future pipeline.

As lineup/PBP rotation data becomes available, we’ll add:
- RAPM (ridge regression on stints) + box-score priors.
- On/Off panels for derived stats.
- Competition splits (`split_id` = `ALL`, `VS_TOP50`, `VS_TOP100`).

## Publishing (Explicit)
We will only publish to the site when you run a publish script, e.g.:
- `python new-trank/scripts/publish_to_mytrank.py`

This should copy from `new-trank/exports/` into `my-trank/public/data/`.

