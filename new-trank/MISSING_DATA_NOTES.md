# Missing Data Notes (Site Warehouse)

Last updated: 2026-02-04

## Confirmed gaps

- Postseason data is currently missing in `ml model/data/warehouse.duckdb` for player-season/game coverage used by the site pipeline.
- Root cause: ingest paths were run with `season_type="regular"` only (no postseason ingest/backfill).
- This is a data coverage issue, not a deletion issue.

## Impact

- Season totals for players/teams in warehouse-backed outputs can be short vs official totals.
- Example: 2024-25 Purdue players will miss NCAA tournament contributions unless postseason is ingested.

## Also not yet available in the current warehouse season table

In `fact_player_season_stats`, these requested model-style outputs are not currently present as stored season columns:

- BPM / OBPM / DBPM
- RAPM / ORAPM / DRAPM

(These require either separate model outputs or additional derived build steps.)

## Planned fix (not run yet)

1. Backfill postseason ingest by season.
2. Rebuild derived tables after postseason ingest.
3. Validate player game counts/season totals against expected values.
4. Add/attach impact model outputs (BPM/RAPM family) to site exports once available.

## Evidence from ingest logs

### `ml model/logs/ingest_history.log`
- Repeated API rate-limit backoff loops (`[429] Rate limit. Waiting ...`).
- Repeated hard failures after retries:
  - `Subs error: Max retries exceeded ...`
  - `Plays error: Max retries exceeded ...`
- Parsed counts from this log snapshot:
  - `max_retry_total`: 56
  - `subs_errors`: 28
  - `plays_errors`: 28
  - `unique_game_ids_with_max_retry`: 29

### `ml model/logs/full_ingest.log`
- Extensive quota pressure warnings (`[QUOTA] Reached ...`).
- Parsed count from this log snapshot:
  - `quota_warnings_full_ingest`: 10073

### One-time ingest/schema issue also present
- Early log includes a conversion error:
  - `Conversion Error: Could not convert string 'Murphy Center' to INT64 when casting from source column venue`

Interpretation: missing coverage is consistent with repeated API throttling/retry exhaustion during per-game play/substitution ingest.

## Coverage backlog snapshot (from warehouse coverage check)

Reference: `ml model/data/warehouse.duckdb` coverage comparison of `dim_games` vs `stg_participants`.

- Total expected regular-season games in `dim_games`: `51,004`
- Games with participant coverage: `39,278`
- Missing regular-season games to backfill: `11,726`

Season-level missing regular games:

- 2009: 3000
- 2010: 1100
- 2011: 1314
- 2012: 1428
- 2013: 1261
- 2014: 492
- 2015: 336
- 2016: 279
- 2017: 241
- 2018: 188
- 2019: 346
- 2020: 234
- 2021: 828
- 2022: 382
- 2023: 88
- 2024: 68
- 2025: 141

Player-impact example:

- Zeek Woodley (`athleteId=52717`) 2015 currently resolves to 14 games in `stg_participants` (expected much higher), confirming real under-coverage.

Note: postseason ingest remains a separate missing block not counted in the 11,726 figure above.

## RAPM scope mismatch note

- Historical RAPM outputs used pre-2025 include regular-season + postseason possessions/stints.
- Current warehouse ingest path is regular-season only (`seasonType='regular'`) in `dim_games`/staging for the pipeline inspected here.
- Therefore, RAPM comparability is currently mismatched until postseason is ingested/backfilled for warehouse-based recomputation.

## Additional explicit gaps discovered (coverage + feature completeness)

### Warehouse fact-layer sparsity

- `fact_player_season_stats` currently has only seasons:
  - 2005 (`4541` rows)
  - 2025 (`9793` rows)
- It does **not** contain 2010–2024 season rows yet.

- `fact_player_game` / player shot/impact facts are currently only for 2025 and only a subset of games:
  - `fact_player_game`: 204 games
  - `fact_player_game_impact`: 202 games

### Lineup/substitution unevenness

- `fact_lineup_stint_raw` coverage:
  - 2024: 295 games
  - 2025: 435 games
- `fact_substitution_raw` coverage:
  - 2013: 212 games
  - 2024: 2882 games
  - 2025: 2992 games

This means lineup-based metrics (RAPM/on-off) are not uniformly available across seasons/teams.

### Feature-store integrity issues (college features parquet)

File: `ml model/data/college_feature_store/college_features_v1.parquet`

- Duplicate key groups on `(season, athlete_id, split_id)`: `287,589` groups with duplicates.
- Some groups appear duplicated 4x, indicating repeated ingest/build merge artifacts.

- Metadata/context fields frequently missing:
  - `team_pace` nulls: `1,088,583 / 1,255,109`
  - `conference` nulls: `1,129,823 / 1,255,109`

Core stat totals are present (minutes/ast/tov/stl/blk non-null in aggregate), but context enrichment is incomplete.

### Historical RAPM file coverage gaps

File: `ml model/data/historical_rapm_results_lambda1000.csv`

- Seasons present: 2012, 2013, 2014, 2015, 2017
- Missing from this file: 2016 and other years outside that set.

## High-value tests to run next (diagnostic-only)

1. **Missing game ID export test**
   - Diff `dim_games` vs `stg_participants` by season, export exact missing game IDs per season.

2. **Postseason completeness test**
   - Verify `seasonType='postseason'` counts in `dim_games`, `stg_*`, and `fact_*` tables.

3. **Player totals reconciliation test**
   - For sampled players, compare warehouse totals vs expected official games/minutes/points.

4. **Duplicate collapse test for feature store**
   - Deduplicate `(season, athlete_id, split_id)` and compare aggregate deltas before/after.

5. **Lineup-link integrity test**
   - Validate mapping between lineup athlete IDs and season athlete IDs for each season/team.

6. **API-failure inventory test**
   - Parse logs into a structured CSV of failed game IDs + endpoint type (`plays` vs `subs`) for targeted re-ingest.

## Ingest efficiency fixes implemented

Code updates applied in `ml model/cbd_pbp/ingest.py` + `ml model/cbd_pbp/cli.py`:

- `resume_ingest` now scopes `dim_games` by requested `season` + `seasonType` (no all-season sweep).
- Per-game ingest is now endpoint-aware (calls only missing endpoint types per game):
  - plays missing set
  - substitutions missing set
  - lineups missing set
- Lineups re-enabled for resume/full ingest via `include_lineups` option.
  - CLI defaults:
    - `ingest_season_cmd`: `include_lineups=True`
    - `resume_ingest`: `include_lineups=False` (safer for API budget on historical years)
- Lineups HTTP fetch aligned to PDF-style path first:
  - `/lineups/game/{gameId}`
  - fallback to `/lineups/game?gameId=` for compatibility.
- Added structured failure logging table on endpoint errors:
  - `ingest_failures(gameId, season, seasonType, endpoint, error, loggedAt)`

These changes reduce wasted API calls and make missing-data backfill targeted.
