# API-Aligned Ingest Playbook (Regular + Postseason + Bio)

## Goal
Pull **all required NCAA data** for site/model use using endpoints documented in:
`ml model/cbd_pbp/api commands.pdf`

This playbook is aligned to the PDF endpoint set and prioritizes:
1. Full regular + postseason game coverage
2. Missing ingest backfill (plays/subs/lineups)
3. Player bio enrichment (`dateOfBirth`, height, weight)
4. Deterministic QA checks

---

## 1) Required Endpoint Contract (from API PDF)

## Core game/event endpoints
- `GET /games`
- `GET /games/players`
- `GET /plays/game/{gameId}`
- `GET /substitutions/game/{gameId}`
- `GET /lineups/game/{gameId}`

## Season stats endpoints
- `GET /stats/player/season`
- `GET /stats/player/shooting/season`
- `GET /stats/team/season`
- `GET /stats/team/shooting/season`

## Team/bio endpoints
- `GET /teams`
- `GET /teams/roster`  <-- contains `dateOfBirth`, `height`, `weight`, `position`, hometown fields
- `GET /conferences`, `GET /conferences/history`
- `GET /venues`

## Optional context endpoints
- `GET /ratings/adjusted`, `GET /ratings/srs`, `GET /rankings`, `GET /lines`
- `GET /recruiting/players`, `GET /draft/picks`

---

## 2) Pre-Run Code Fixes (must do first)

Current blockers in `ml model/cbd_pbp/ingest.py`:

1. `ingest_games_only(...)` does **not** filter `dim_games` by season/seasonType.
- It currently loads all game IDs from `dim_games`.
- Fix: query `WHERE season = ? AND seasonType = ?`.

2. `ingest_games_only(...)` has lineups disabled.
- It sets `ldf = pd.DataFrame()` and skips `lineups` endpoint entirely.
- Fix: re-enable lineup pull using `_fetch_lineups_raw(...)` (or SDK call with null-safe handling).

3. Roster endpoint not ingested.
- Add `teams/roster` ingest to a new table, e.g. `dim_team_roster` and exploded `dim_player_bio`.

4. Failed pulls are only logged to text.
- Add structured failure table, e.g. `ingest_failures(gameId, season, seasonType, endpoint, error, ts)`.

---

## 3) Canonical Pull Order

## Step A: Static dimensions (once)
Run static pulls before season loops:
- teams, conferences/history, venues, play types, lines providers, draft dims

Command:
```bash
cd 'ml model'
python -m cbd_pbp.cli ingest_season_cmd --season 2025 --season-type regular --out data/warehouse.duckdb
```

(First run seeds static dims + one season. Additional seasons below.)

## Step B: Season loop (regular + postseason)
For each target season `Y`:
1. Pull `regular`
2. Pull `postseason`
3. Build derived tables

Commands:
```bash
cd 'ml model'
python -m cbd_pbp.cli ingest_season_cmd --season Y --season-type regular --out data/warehouse.duckdb
python -m cbd_pbp.cli ingest_season_cmd --season Y --season-type postseason --out data/warehouse.duckdb
python -m cbd_pbp.cli build_derived --season Y --season-type regular --out data/warehouse.duckdb
```

Notes:
- For historical years where API lineup/sub coverage is weak, merge NCAA.org reconstructed lineups after this pass.
- If API supports `preseason` and you need it, run an explicit third pass.

## Step C: Resume/backfill missing game events
After initial season pulls, run a targeted resume on missing IDs only (not whole-table blind rerun).

Command (after fixing season filter + lineup skip):
```bash
cd 'ml model'
python -m cbd_pbp.cli resume_ingest --season Y --season-type regular --out data/warehouse.duckdb
python -m cbd_pbp.cli resume_ingest --season Y --season-type postseason --out data/warehouse.duckdb
```

---

## 4) NCAA.org Historical Lineup Merge (pre-2024/25)

When API lineup coverage is incomplete for older seasons:
1. Use `data/manual_scrapes/{YEAR}/` raw files
2. Reconstruct stints/lineups
3. Map game IDs to `dim_games.id/sourceId`
4. Map athlete name -> athleteId crosswalk
5. Upsert into:
- `fact_lineup_stint_raw`
- `bridge_lineup_athletes`

Keep provenance fields:
- `source_system` (`api` vs `ncaa_scrape`)
- `reconstruction_version`
- `lineup_confidence`

---

## 5) Bio Ingest Spec (`teams/roster`)

Create two tables:

1. `dim_team_roster`
- keys: `(teamId, season, player_source_id)`
- fields: team/conference/season, player id/sourceId

2. `dim_player_bio`
- keys: `(player_source_id, season)`
- fields: `firstName`, `lastName`, `position`, `height`, `weight`, `dateOfBirth`, hometown, `startSeason`, `endSeason`

Rules:
- Preserve raw DOB string and parsed DOB timestamp.
- Never infer DOB from draft/recruiting tables if missing.

---

## 6) QA Gates (must pass)

## Coverage gate: games
For each `(season, seasonType)`:
- `dim_games.count` vs `stg_participants.distinct(gameId)`
- `dim_games.count` vs `stg_plays.distinct(gameId)`
- `dim_games.count` vs `stg_subs.distinct(gameId)`
- `dim_games.count` vs `stg_lineups.distinct(gameId)`

## Coverage gate: facts
- `fact_play_raw`, `fact_substitution_raw`, `fact_lineup_stint_raw` distinct game counts
- `fact_player_game*` game counts should track play/sub/lineup readiness

## Completeness gate: player seasons
- `fact_player_season_stats` must include all intended seasons (not just 2005/2025)
- no duplicate `(season, athleteId, teamId)` unless justified by transfer logic

## Bio gate
- `% rows with non-null dateOfBirth` by season
- `% rows with non-null height/weight`

## RAPM gate
- Tag RAPM output with `season_scope` (`regular_only` vs `regular_postseason`)
- Reject cross-scope comparison in dashboards by default

---

## 7) SQL Checks (copy/paste)

```sql
-- Missing games by season/type
WITH dg AS (
  SELECT season, seasonType, COUNT(*) AS dim_games
  FROM dim_games
  GROUP BY 1,2
), sp AS (
  SELECT g.season, g.seasonType, COUNT(DISTINCT s.gameId) AS games_with_participants
  FROM stg_participants s
  JOIN dim_games g ON g.id = TRY_CAST(s.gameId AS BIGINT)
  GROUP BY 1,2
)
SELECT dg.season, dg.seasonType, dg.dim_games,
       COALESCE(sp.games_with_participants,0) AS games_with_participants,
       dg.dim_games - COALESCE(sp.games_with_participants,0) AS missing_games
FROM dg LEFT JOIN sp USING (season, seasonType)
ORDER BY 1,2;
```

```sql
-- Missing game IDs list for targeted backfill
SELECT g.id, g.sourceId, g.season, g.seasonType
FROM dim_games g
LEFT JOIN (
  SELECT DISTINCT TRY_CAST(gameId AS BIGINT) AS gid FROM stg_participants
) p ON p.gid = g.id
WHERE p.gid IS NULL
ORDER BY g.season, g.seasonType, g.id;
```

```sql
-- Bio coverage once roster ingest is added
SELECT season,
       COUNT(*) AS rows,
       SUM(CASE WHEN dateOfBirth IS NULL THEN 1 ELSE 0 END) AS null_dob,
       SUM(CASE WHEN height IS NULL THEN 1 ELSE 0 END) AS null_height,
       SUM(CASE WHEN weight IS NULL THEN 1 ELSE 0 END) AS null_weight
FROM dim_player_bio
GROUP BY 1
ORDER BY 1;
```

---

## 8) Operational Notes

- Keep retries + exponential backoff for `429`, but persist failures to table (not only logs).
- Avoid full reruns once quota pressure starts; do targeted `gameId` retries.
- After every batch: run QA SQL above and checkpoint counts in a coverage report.
- Treat provider-empty responses (`HTTP 200` with empty payload) as terminal by default to avoid quota burn.
- Recheck provider-empty IDs only in explicit revalidation runs.

## 8.1) Quota-Safe Retry Policy

Use audit artifacts in `data/audit/`:
- `source_void_games.csv`
- `retry_policy_cache.json`
- `ingest_attempts.csv`

Rules:
1. If `(game_id, endpoint)` is `provider_empty`, skip by default.
2. If retryable failure has active cooldown, skip until `cooldown_until`.
3. Use `--force-recheck` only for explicit source revalidation.
4. Always pass targeted game lists to endpoint ingest to avoid season-wide re-queries.

Example targeted resume:
```bash
python -m cbd_pbp.cli resume-ingest-endpoints \
  --season 2018 \
  --season-type regular \
  --endpoints plays \
  --only-game-ids-file data/audit/some_ids.txt \
  --out data/warehouse.duckdb
```

## 8.2) Dual-Gate Readiness

- `model_readiness_gate.json`: API execution gate (pipeline health).
- `model_readiness_dual_source.json`: data availability gate (API or manual source).

Interpretation:
- API gate may fail while dual-source gate passes.
- Production modeling should key off dual-source gate for required families.

---

## 9) Definition of Done

Done when all are true:
- Regular + postseason ingested for target seasons.
- Missing-game backlog reduced to accepted threshold.
- Lineup tables populated for years where API is weak using NCAA.org reconstruction.
- `teams/roster` bio (`dateOfBirth`, height, weight) available and queryable.
- RAPM outputs clearly labeled by season scope and reproducible.
