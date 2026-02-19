# Mistake Prevention Retrospective (2026-02-19)

## Purpose
- Record concrete mistakes made in recent runs.
- Convert each mistake into an explicit guard for the new strict run.
- Prevent drift from DAG/document contracts during future implementation.

## Mistakes Committed
1. **DAG contract drift**
- What happened: core DAG-required fields/features were not fully wired (`team_strength/SRS`, age/class/season-index, leverage family), while training proceeded.
- Why this hurt: model trained on an incomplete narrative surface and ranking quality degraded.
- Prevention guard:
  - Block training if critical DAG nodes are missing.
  - Require stage report mapping each DAG node to `implemented/partial/missing/dead`.

2. **Dead branch accepted as active**
- What happened: within-season features were present in schema but dead (`0%` nonzero), without hard-stop policy.
- Why this hurt: false confidence that the branch contributed signal.
- Prevention guard:
  - If branch is dead, either (a) populate from source, or (b) explicitly mask and gate it off.
  - Readiness gate fails if dead critical branches are not explicitly disabled.

3. **Insufficient full-surface audits before major runs**
- What happened: not enough per-column/per-season integrity checks before full training/inference.
- Why this hurt: silent issues propagated to model outputs.
- Prevention guard:
  - Run large assertion pack (schema/cardinality/coverage/distribution/contract/regression) before full run.
  - Hard-stop on critical integrity failures.

4. **Crosswalk confidence not enforced tightly enough**
- What happened: ambiguity risk remained around name normalization edge cases.
- Why this hurt: potential identity noise in college-to-NBA linkage.
- Prevention guard:
  - Multi-pass crosswalk QA with punctuation/suffix normalization diagnostics.
  - Regression checks on high-confidence matches.

5. **Proceeding with quality concerns despite user emphasis**
- What happened: execution moved ahead while quality concerns were still open.
- Why this hurt: user trust and model reliability.
- Prevention guard:
  - Strict gate policy: no full run unless all critical checks are green.
  - Every stage emits machine-readable + markdown audit artifacts.

## Mandatory Checks for This Run
- `CHECK-A`: DAG critical-path contract reconciliation complete and green.
- `CHECK-B`: `fact_player_peak_rapm` unique `nba_id` cardinality remains zero-duplicate.
- `CHECK-C`: unified table is unique at `nba_id` and `draft_year_proxy >= 2011`.
- `CHECK-D`: target coverage floors not regressed (`y_peak_ovr`, `year1_epm_tot`, `dev_rate`).
- `CHECK-E`: dead critical branches either populated or explicitly masked/gated off.
- `CHECK-F`: crosswalk ambiguity report generated and no high-confidence regression.

## Run-Loop Reference Policy
This document must be referenced at least twice in strict runs:
1. Before Stage 0 snapshot (to confirm guard scope).
2. Before Stage 6 training (to confirm no-repeat conditions still hold).

## Non-Negotiable Rule
If any critical guard fails, stop the run and produce a NO-GO audit. Do not continue to full training/inference.

## New Mistakes Added During Current Pass
6. **Historical cleaner memory blow-up and silent long-run uncertainty**
- What happened: `clean_historical_pbp_v2.py` attempted to hold all seasons in one in-memory list before writing parquet; this caused unstable long runs and uncertainty about completion.
- Why this hurt: incomplete historical assembly led to stale `fact_play_historical_combined.parquet` and missing lineup-era coverage.
- Fix applied:
  - Reworked cleaner to write per-season parquet parts incrementally and materialize final parquet via DuckDB `COPY` from parts.
- Prevention guard:
  - Any pipeline stage processing >5M rows must stream or partition writes.
  - Add completion check: row count + season count + file mtime must change in the same run.

7. **Hidden background process holding DuckDB lock**
- What happened: non-TTY run continued in background and held `warehouse.duckdb` lock, causing follow-up runs to fail.
- Why this hurt: debugging noise and rerun delays.
- Fix applied:
  - Terminated stale PID and switched to visible unbuffered/TTY run for long steps.
- Prevention guard:
  - For long-running steps, use unbuffered output and verify lock release before next stage.
  - Record PID and stage in audit notes when runtime exceeds expected window.

8. **API coverage assumption was too optimistic**
- What happened: initial pipeline quality assumed broad API completeness despite known missing team-season game coverage in `fact_play_raw`.
- Why this hurt: exposure features (`games`, `minutes`, usage context) were undercounted for some cohorts.
- Fix applied:
  - Promoted manual-scrape historical assembly to required input for coverage closure.
- Prevention guard:
  - Add explicit `covered_either` audit (`api OR manual`) by season/game before full train.
  - NO-GO if uncovered games exceed threshold for target seasons.

9. **Derived-stat merge bug zeroed valid features**
- What happened: unified build used `fillna(0)` when applying derived AST/STL/BLK/TOV/GP, which overwrote existing valid values with zeros on missing derived rows.
- Why this hurt: collapsed signal for key box features and degraded ranking quality.
- Fix applied:
  - replaced with strict coalesce: `derived.combine_first(existing)`.
- Prevention guard:
  - add merge regression test that verifies non-null source values are never reduced after enrichment joins.

10. **Cross-source race in ranking export**
- What happened: export script read an older prediction artifact when invoked in parallel with inference.
- Why this hurt: stale rankings despite new model run.
- Fix applied:
  - rerun export after inference completion; validate latest file timestamp.
- Prevention guard:
  - serialize inference -> export steps in final run and assert exported `predictions=` path equals latest artifact.

11. **Interpreter mismatch (`python3` vs `python3.13`) broke CLI runs**
- What happened: pipeline commands were launched with default `python3` where some required modules were unavailable (`cbd_pbp`), while the working environment was `python3.13` (with `typer` and `cbd_pbp` installed).
- Why this hurt: run interruptions, false failure signals, and wasted debugging time.
- Fix applied:
  - switched closure/ingest execution to `python3.13` for all `cbd_pbp.cli`-dependent steps.
- Prevention guard:
  - add preflight interpreter check before any ingest/closure run:
    - verify `python3.13 -c "import typer, cbbd"` succeeds (or project-required module set).
    - emit explicit runtime banner in logs with interpreter path/version and module availability.
  - treat interpreter mismatch as a hard-stop before stage execution.
