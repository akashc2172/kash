#!/usr/bin/env bash
set -euo pipefail

YEAR="2026"
URL="https://barttorvik.com/getadvstats.php?year=${YEAR}&csv=1"

# pick where YOU want it to live in the repo
OUT_DIR="data/torvik_raw"
OUT_FILE="${OUT_DIR}/torvik_advstats_${YEAR}.csv"

mkdir -p "${OUT_DIR}"

echo "[torvik] downloading: ${URL}"
curl -L \
  -A "Mozilla/5.0 (compatible; my-trank-bot/1.0)" \
  --retry 3 --retry-delay 2 \
  -o "${OUT_FILE}" \
  "${URL}"

# quick sanity check: if you accidentally downloaded HTML, fail
if head -n 3 "${OUT_FILE}" | grep -qi "<html"; then
  echo "[torvik] ERROR: downloaded HTML instead of CSV (blocked or changed endpoint)."
  exit 1
fi

echo "[torvik] saved: ${OUT_FILE}"
