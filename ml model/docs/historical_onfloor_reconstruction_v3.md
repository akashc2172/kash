# Historical onFloor Reconstruction v3 (Pre-2025)

**Date**: 2026-02-19  
**Owner**: NBA/College data pipeline  
**Purpose**: Replace degraded historical `onFloor` artifact with deterministic lineup reconstruction from manual scrapes, then hard-gate RAPM publication.

## Why this exists

Late-era historical artifacts had severe lineup collapse (`len(onFloor)==2` dominated by `TEAM,` placeholders), which corrupted RAPM seasons.  
This v3 pipeline reconstructs per-event 10-player lineups and emits explicit quality gates.

## Inputs

- Manual scrape CSVs: `data/manual_scrapes/2011-2012` ... `data/manual_scrapes/2023-2024`
- Optional API plays for 2025+: `data/warehouse.duckdb` (`fact_play_raw`)

## Outputs

1. Reconstructed combined parquet:
- `data/fact_play_historical_combined_v2.parquet`

2. Lineup quality audits:
- `data/audit/historical_lineup_quality_by_game.csv`
- `data/audit/historical_lineup_quality_by_season.csv`

3. RAPM split diagnostics:
- `data/audit/historical_rapm_split_quality.csv`

## Core reconstruction logic (v3)

- Event parsing from raw text (`clock | home_evt | score | away_evt`)
- Name normalization preserving apostrophes/hyphens, removing jersey noise
- Explicit lineup checkpoints (`TEAM For ...`)
- Sub propagation (`Enters/Leaves`, `SUB IN/OUT`)
- Starter inference from pre-sub activity
- Strict ghost fill only from observed game roster candidates
- Team-only events (`TEAM,`, bench/deadball) excluded from player identity

Each row emits:
- `onFloor`
- `lineup_source`
- `lineup_confidence`
- `lineup_quality_flag`

## Hard gates

### Pre-RAPM lineup gate (season audit)

Season must satisfy:
- `pct_rows_len10 >= 0.80`
- `pct_rows_placeholder <= 0.05`
- `pct_games_pass >= 0.80`
- `avg_unique_players_game >= 10`

### RAPM split gate

Season must satisfy:
- `valid_5v5_rate >= 0.80`
- `unresolved_home_rate <= 0.20`
- `parse_fail_rate <= 0.05`
- `n_stints >= 2000`
- `unique_players_5v5 >= 700`

Publication rule: season must pass both lineup gate and split gate.

## Commands

## 1) Build reconstructed historical artifact

```bash
python3 "college_scripts/reconstruct_historical_onfloor_v3.py" \
  --start-season 2011 \
  --end-season 2024 \
  --output-combined "data/fact_play_historical_combined_v2.parquet" \
  --output-game-audit "data/audit/historical_lineup_quality_by_game.csv" \
  --output-season-audit "data/audit/historical_lineup_quality_by_season.csv"
```

## 2) Diagnostics-only RAPM gate pass

```bash
python3 "college_scripts/calculate_historical_rapm.py" \
  --input-parquet "data/fact_play_historical_combined_v2.parquet" \
  --lineup-season-audit-csv "data/audit/historical_lineup_quality_by_season.csv" \
  --diagnostics-only \
  --diagnostics-csv "data/audit/historical_rapm_split_quality.csv"
```

## 3) Solve RAPM on gate-pass seasons only

```bash
python3 "college_scripts/calculate_historical_rapm.py" \
  --input-parquet "data/fact_play_historical_combined_v2.parquet" \
  --lineup-season-audit-csv "data/audit/historical_lineup_quality_by_season.csv" \
  --diagnostics-csv "data/audit/historical_rapm_split_quality.csv" \
  --output-csv "data/historical_rapm_results_enhanced.csv"
```

## Fast validation mode (bounded)

```bash
python3 "college_scripts/reconstruct_historical_onfloor_v3.py" \
  --start-season 2019 \
  --end-season 2023 \
  --max-games-per-season 200 \
  --no-api-append \
  --output-combined "data/fact_play_historical_combined_v2_sample.parquet" \
  --output-game-audit "data/audit/historical_lineup_quality_by_game_sample.csv" \
  --output-season-audit "data/audit/historical_lineup_quality_by_season_sample.csv"
```

