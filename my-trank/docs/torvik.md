# Bart Torvik Updates

## Download 2026 export
```bash
cd /Users/akashc/my-trankcopy/my-trank
bash scripts/download_torvik_2026.sh
```

## Regenerate merged data
```bash
Rscript scripts/build_data.R
```

This updates:
- `public/data/season.csv`
- `public/data/career.csv`

## Build site
```bash
npm run build
```
