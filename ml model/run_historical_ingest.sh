#!/bin/bash
# Historical Ingest Script - Fixed Version
# Ingests regular AND postseason for seasons 2010-2025
# Uses fetch-ingest command which doesn't touch static tables (avoids schema issues)

set -e

cd "/Users/akashc/my-trankcopy/ml model"

# 1) Create run log dir
TS=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="logs/ingest_runs/$TS"
mkdir -p "$RUN_DIR"
export RUN_DIR  # Export for Python script

echo "=== Historical Ingest Started at $(date) ===" | tee "$RUN_DIR/run_summary.log"
echo "Log directory: $RUN_DIR" | tee -a "$RUN_DIR/run_summary.log"

# 2) Seasons to ingest (2010-2025, skip 2026)
SEASON_START=${1:-2010}
SEASON_END=${2:-2025}

echo "Season range: $SEASON_START to $SEASON_END" | tee -a "$RUN_DIR/run_summary.log"

# 3) Ingest using fetch-ingest for regular + postseason (no lineups to save API calls)
for Y in $(seq $SEASON_START $SEASON_END); do
  for ST in regular postseason; do
    echo "" | tee -a "$RUN_DIR/run_summary.log"
    echo "=== RUN: season=$Y type=$ST ===" | tee -a "$RUN_DIR/run_summary.log"
    echo "Started: $(date)" | tee -a "$RUN_DIR/run_summary.log"
    
    python3.13 -m cbd_pbp.cli resume-ingest \
      --season "$Y" \
      --season-type "$ST" \
      --no-include-lineups \
      --out data/warehouse.duckdb \
      2>&1 | tee "$RUN_DIR/ingest_${Y}_${ST}.log"
    
    echo "Finished: $(date)" | tee -a "$RUN_DIR/run_summary.log"
    
    # Brief pause between seasons to avoid hammering API
    sleep 2
  done
done

# 4) Record diagnostics + failures snapshot
echo "" | tee -a "$RUN_DIR/run_summary.log"
echo "=== Generating Diagnostics ===" | tee -a "$RUN_DIR/run_summary.log"

python3.13 - <<'PY'
import duckdb
import pathlib
import datetime
import os

run_dir = pathlib.Path(os.environ.get("RUN_DIR", "logs/ingest_runs/latest"))
con = duckdb.connect("data/warehouse.duckdb", read_only=True)

# Basic diagnostics
with open(run_dir / "diagnostics.txt", "w") as f:
    f.write(f"generated_at={datetime.datetime.now().isoformat()}\n")
    try:
        with open("api_call_count.txt") as c:
            f.write(f"api_call_count={c.read().strip()}\n")
    except Exception as e:
        f.write(f"api_call_count_error={e}\n")

    for t in ["fact_play_raw", "fact_substitution_raw", "fact_lineup_stint_raw", "ingest_failures"]:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            f.write(f"{t}_rows={n}\n")
        except Exception as e:
            f.write(f"{t}_rows_error={e}\n")

# Failures summary csv
try:
    con.execute("""
    COPY (
      SELECT season, seasonType, endpoint, COUNT(*) AS failures
      FROM ingest_failures
      GROUP BY 1,2,3
      ORDER BY 1,2,3
    ) TO ? (HEADER, DELIMITER ',')
    """, [str(run_dir / "ingest_failures_summary.csv")])
except Exception as e:
    print(f"Failures summary error: {e}")

# Detailed failures csv
try:
    con.execute("""
    COPY (
      SELECT gameId, season, seasonType, endpoint, error, loggedAt
      FROM ingest_failures
      ORDER BY loggedAt DESC
    ) TO ? (HEADER, DELIMITER ',')
    """, [str(run_dir / "ingest_failures_detail.csv")])
except Exception as e:
    print(f"Failures detail error: {e}")

# Coverage snapshot by season/type - FIXED table and column names
try:
    con.execute("""
    COPY (
      WITH g AS (
        SELECT CAST(id AS VARCHAR) AS gameId, season, seasonType FROM dim_games
      ),
      p AS (SELECT DISTINCT gameId FROM fact_play_raw),
      s AS (SELECT DISTINCT gameId FROM fact_substitution_raw),
      l AS (SELECT DISTINCT gameId FROM fact_lineup_stint_raw)
      SELECT
        g.season,
        g.seasonType,
        COUNT(*) AS games_total,
        COUNT(p.gameId) AS games_with_plays,
        COUNT(s.gameId) AS games_with_subs,
        COUNT(l.gameId) AS games_with_lineups
      FROM g
      LEFT JOIN p ON g.gameId = p.gameId
      LEFT JOIN s ON g.gameId = s.gameId
      LEFT JOIN l ON g.gameId = l.gameId
      GROUP BY 1,2
      ORDER BY 1,2
    ) TO ? (HEADER, DELIMITER ',')
    """, [str(run_dir / "coverage_by_season_type.csv")])
except Exception as e:
    print(f"Coverage snapshot error: {e}")

print(f"Wrote diagnostics to {run_dir}")
con.close()
PY

echo "" | tee -a "$RUN_DIR/run_summary.log"
echo "=== DONE at $(date) ===" | tee -a "$RUN_DIR/run_summary.log"
echo "Logs in: $RUN_DIR"
