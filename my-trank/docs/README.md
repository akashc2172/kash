# My-Trank Docs

Start here. This folder describes the data pipeline and the site build.

## Key docs
- `docs/data-pipeline.md`: End-to-end data flow
- `docs/torvik.md`: Bart Torvik updates
- `docs/international.md`: RealGM international updates
- `docs/site.md`: Running/building the site

## Quick commands (from `/Users/akashc/my-trankcopy/my-trank`)
- Download Bart Torvik 2026 CSV: `bash scripts/download_torvik_2026.sh`
- Build merged data: `Rscript scripts/build_data.R`
- Build site: `npm run build`
- Dev server: `npm run dev -- --host 127.0.0.1 --port 5174`
