# Data Update Flow - How Website Updates Work

## ✅ YES - Your Website Updates Dynamically!

Your website loads CSV files **directly from the `public/data/` folder** using client-side fetching. This means:

### How It Works

1. **Website loads data on page load**:
   - Uses `fetchCsv()` function with Papa Parse
   - Loads from `/data/international_stat_history/internationalplayerarchive.csv`
   - No build step required for data updates

2. **File paths**:
   - Code references: `/data/international_stat_history/internationalplayerarchive.csv`
   - Actual location: `public/data/international_stat_history/internationalplayerarchive.csv`
   - Vite serves `public/` files at root `/` path

3. **Update process**:
   ```
   Run realgm_scrape.py
        ↓
   Saves CSVs to public/data/international_stat_history/
        ↓
   Run import_realgm_leagues.py
        ↓
   Updates internationalplayerarchive.csv
        ↓
   Refresh website → New data appears!
   ```

## Local Development

**Fully Dynamic** - Just refresh the page:
```bash
# 1. Scrape new data
python scripts/realgm_scrape.py

# 2. Import into archive
python scripts/import_realgm_leagues.py

# 3. Refresh browser (F5 or Cmd+R)
# → New data appears immediately!
```

## Production (Cloudflare Pages)

**Requires Rebuild** - But data is still dynamic:
```bash
# 1. Scrape & import (same as above)
python scripts/realgm_scrape.py
python scripts/import_realgm_leagues.py

# 2. Commit & push changes
git add public/data/international_stat_history/internationalplayerarchive.csv
git commit -m "Update international player data"
git push

# 3. Cloudflare Pages auto-rebuilds
# → New data appears after deployment (~2-3 minutes)
```

**OR** if you want to update without rebuilding:
- You'd need to manually upload the CSV to Cloudflare Pages
- Or use Cloudflare Workers to fetch from an external source
- Current setup requires rebuild for production updates

## What Gets Updated

When you run the import script, these files are updated:
- ✅ `public/data/international_stat_history/internationalplayerarchive.csv`
- ✅ Website loads this file on every page load
- ✅ New players from RealGM leagues appear automatically

## Testing Updates

1. **Before update**: Note a player count or specific player
2. **Run scraper + import**: Add new data
3. **Refresh page**: Check if new data appears
4. **Verify**: Search for a newly added player

## Notes

- **No cache issues**: The `fetchCsv` function doesn't use aggressive caching
- **Browser cache**: If you see old data, do a hard refresh (Cmd+Shift+R / Ctrl+Shift+R)
- **File watching**: Vite dev server watches `public/` folder, so changes are detected
- **Production**: After Cloudflare rebuild, new data is live

## Summary

✅ **Local Dev**: Fully dynamic - just refresh  
✅ **Production**: Requires rebuild (automatic via git push)  
✅ **Data Format**: CSV files loaded directly, no preprocessing needed  
✅ **Update Speed**: Instant in dev, ~2-3 min in production after push
