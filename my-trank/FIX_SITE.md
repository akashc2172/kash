# Fixing my-trank Site Issues

## Problem
The site isn't going live. This could be due to:
1. Dev server not running
2. Build issues
3. Deployment configuration issues
4. Missing dependencies

## Quick Fixes

### Option 1: Run Development Server Locally

```bash
cd my-trank
npm install  # If dependencies are missing
npm run dev
```

The site will be available at `http://localhost:5173`

### Option 2: Build and Preview

```bash
cd my-trank
npm run build
npm run preview
```

### Option 3: Deploy to Cloudflare Pages

Since you have `wrangler.jsonc`, this is configured for Cloudflare Workers/Pages:

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Deploy
wrangler pages deploy dist
```

Or use Cloudflare Dashboard:
1. Go to Cloudflare Dashboard → Pages
2. Create new project
3. Connect your repository
4. Build command: `npm run build`
5. Output directory: `dist`

## Current Status

✅ **Dependencies**: Installed (node_modules exists)
✅ **Build Output**: Exists (dist/ folder has built files)
✅ **Data Files**: Present in public/data/
✅ **Configuration**: Vite config looks correct

## Common Issues

### Issue: npm log file errors
**Solution**: This is a permissions issue with npm logs. The actual commands should still work. Try:
- Using `./node_modules/.bin/vite` directly
- Or running with `npm run dev` (ignore the log warnings)

### Issue: Site not loading data
**Check**: Make sure `public/data/` has all required CSV files:
- season.csv
- career.csv
- weights.csv
- archive.csv
- br_advanced_stats.csv
- nba_lookup.csv
- international_stat_history/ files

### Issue: TypeScript errors
**Solution**: The project uses `@ts-nocheck` in App.tsx, so TypeScript errors are ignored. If you want to fix them:
```bash
npx tsc --noEmit
```

## Testing Locally

1. **Start dev server**:
   ```bash
   cd my-trank
   ./start-dev.sh
   # Or: npm run dev
   ```

2. **Open browser**: Navigate to `http://localhost:5173`

3. **Check console**: Open browser DevTools (F12) and check for errors

## Deployment Checklist

- [ ] Build succeeds: `npm run build`
- [ ] Preview works: `npm run preview`
- [ ] All data files in `public/data/`
- [ ] Cloudflare Pages configured (if deploying)
- [ ] Environment variables set (if needed)
