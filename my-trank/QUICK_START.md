# Quick Start Guide - my-trank

## 🚀 Start Development Server

```bash
cd my-trank
npm run dev
```

Then open: **http://localhost:5173**

## 🔧 Alternative: Direct Vite Command

If `npm run dev` has issues, use:

```bash
cd my-trank
./node_modules/.bin/vite --host
```

## 📦 Build for Production

```bash
npm run build
```

Output will be in `dist/` folder.

## 🌐 Preview Production Build

```bash
npm run build
npm run preview
```

## ☁️ Deploy to Cloudflare Pages

This project is configured for Cloudflare Pages (see `wrangler.jsonc`):

1. **Via CLI**:
   ```bash
   npm install -g wrangler
   wrangler login
   wrangler pages deploy dist
   ```

2. **Via Dashboard**:
   - Go to Cloudflare Dashboard → Pages
   - Create new project
   - Connect repository
   - Build command: `npm run build`
   - Output directory: `dist`

## ✅ Current Status

- ✅ Dependencies installed
- ✅ Build output exists (`dist/`)
- ✅ Data files present (`public/data/`)
- ✅ Vite configured correctly

## 🐛 Troubleshooting

### npm log errors
These are just warnings about log file permissions. Commands still work.

### Port already in use
Change port:
```bash
npm run dev -- --port 3000
```

### Missing data files
Check `public/data/` has:
- season.csv
- career.csv
- weights.csv
- archive.csv
- br_advanced_stats.csv
- nba_lookup.csv
