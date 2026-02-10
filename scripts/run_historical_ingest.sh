#!/bin/bash
# -----------------------------------------------------------------------------
# Historical Data Ingest Wrapper
# -----------------------------------------------------------------------------
# Usage: ./scripts/run_historical_ingest.sh
# 
# This script manages the long-running ingestion of CBD data (Phases 1 & 2).
# It handles directory switching, logging, and error resilience.

LOG_FILE="../ingest.log"
FAIL_LOG="../ingest_failures.log"

# Determine script directory to handle relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT/ml model" || exit 1

echo "Starting Ingest at $(date)" | tee -a "$LOG_FILE"

# --- Phase 1: Postseason (2011-2025) ---
echo "--- Phase 1: Postseason Ingest ---" | tee -a "$LOG_FILE"
for Y in {2011..2025}; do
    echo "Processing Postseason $Y..." | tee -a "$LOG_FILE"
    if ! python3.13 -m cbd_pbp.cli fetch-ingest --season "$Y" --season-type postseason --no-include-lineups --out ../data/warehouse.duckdb >> "$LOG_FILE" 2>&1; then
        echo "❌ Failed: Postseason $Y" | tee -a "$FAIL_LOG"
    else
        echo "✅ Success: Postseason $Y" | tee -a "$LOG_FILE"
    fi
    sleep 1
done

# --- Phase 2: Regular Season (2010-2025) ---
echo "--- Phase 2: Regular Season Ingest ---" | tee -a "$LOG_FILE"
for Y in {2010..2025}; do
    echo "Processing Regular Season $Y..." | tee -a "$LOG_FILE"
    if ! python3.13 -m cbd_pbp.cli fetch-ingest --season "$Y" --season-type regular --no-include-lineups --out ../data/warehouse.duckdb >> "$LOG_FILE" 2>&1; then
        echo "❌ Failed: Regular Season $Y" | tee -a "$FAIL_LOG"
    else
        echo "✅ Success: Regular Season $Y" | tee -a "$LOG_FILE"
    fi
    sleep 1
done

echo "Ingest Complete at $(date)" | tee -a "$LOG_FILE"
