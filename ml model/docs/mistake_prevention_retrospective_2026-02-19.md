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

12. **Wrong recovery source for historical `subs/lineups` (API retries instead of manual reconstruction)**
- What happened: closure was retrying large historical `subs/lineups` queues against API for seasons where manual scrape reconstruction is the intended source.
- Why this hurt: wasted API budget, slow/noisy runs, and policy mismatch with documented pipeline architecture.
- Fix applied:
  - audit manifests now enforce historical boundary for API `subs/lineups` retries (`--subs-lineups-api-min-season`, default `2024`).
  - subs/lineups readiness floor now evaluates effective dual-source coverage (`API ∪ manual bridge`) on modern seasons.
- Prevention guard:
  - before any closure execute run, assert reingest manifests for `subs/lineups` exclude seasons below API-min-season.
  - treat any historical `subs/lineups` API queue inflation as a pipeline bug, not a data-gap signal.

13. **Gate-status ambiguity from stale audit artifact reads**
- What happened: console-reported pass state and previously read JSON state disagreed due to reading stale artifacts from a prior run context.
- Why this hurt: conflicting status interpretation and unnecessary re-debug cycles.
- Fix applied:
  - reran audit with explicit policy flags and re-read generated JSON immediately after write.
- Prevention guard:
  - always pair gate checks with same-run timestamp verification (`generated_at`) before announcing status.
  - never report gate status from cached/opened files without confirming they were produced by the current execution.

14. **Insufficient post-change regression sweep before status updates**
- What happened: after pipeline script edits, status was reported before a broad enough local regression sweep (tests + quick validation + contract files).
- Why this hurt: increased risk of silently carrying breakage into subsequent closure/training steps.
- Fix applied:
  - executed full local suite (`pytest` wiring/dev/gating + `tests/quick_validate.py`) and contract-file existence checks in same pass.
- Prevention guard:
  - any closure/audit script change requires this minimum post-change battery before reporting readiness:
    - `pytest tests/test_wiring_edge_cases.py tests/test_dev_rate_label_math.py tests/test_encoder_gating.py`
    - `python3 tests/quick_validate.py`
    - direct read of gate artifacts with current `generated_at`.

15. **Buffered pipeline wrapper risk (`capture_output=True`) for long verbose subprocesses**
- What happened: `college_scripts/run_full_pipeline.py` uses `subprocess.run(..., capture_output=True)`. For very verbose long-running cleaners this can create opaque hangs and no incremental visibility.
- Why this hurt: execution appeared stalled, reduced observability, and required manual process termination.
- Fix applied:
  - switched execution plan to direct stage commands (clean -> derive -> RAPM) so logs stream and progress is monitorable.
- Prevention guard:
  - avoid wrapped `capture_output=True` for long historical reconstruction stages.
  - run heavy stages directly with streamed stdout and explicit process health checks.

16. **Endpoint probe ambiguity (`requests` request-layer errors) during queue classification**
- What happened: direct endpoint probing for `subs/lineups` returned request-layer errors in this runtime, preventing clean API-empty vs actionable classification from HTTP alone.
- Why this hurt: queue classification could remain inconclusive and trigger repeated retry churn.
- Fix applied:
  - added DB-observed fallback classification in `scripts/classify_endpoint_retry_queue.py`:
    - if probes fail uniformly at request layer, classify from post-retry DB coverage as `source_limited_after_retry` vs `actionable_resolved_in_db`.
  - applied classification results to `retry_policy_cache.json` terminal states.
- Prevention guard:
  - classify retry queues with dual evidence:
    - primary: endpoint probe class,
    - fallback: post-retry DB coverage delta.
  - never leave persistent queues unclassified when DB evidence is available.

17. **Used full closure verify pass when audit-only validation was sufficient**
- What happened: after queue terminalization, a full `run_missing_data_closure.py --execute` verify run was started, which re-triggered expensive bridge rebuild work.
- Why this hurt: unnecessary runtime cost and delayed feedback loop.
- Fix applied:
  - terminated redundant full closure verify run and switched to direct artifact/gate validation.
- Prevention guard:
  - for post-classification verification, prefer:
    - `run_missing_data_audit.py` + queue artifact checks,
    - only run full closure when ingestion/rebuild is actually required.

18. **Over-trusted participant-event game counts as true games played**
- What happened: `college_games_played` from `derived_box_stats_v1.parquet` (event-participation based) was used as canonical exposure in unified/inference.
- Why this hurt: undercounted true season exposure for players like Paolo Banchero (18 vs 39) and Chet Holmgren (16 vs 32), degrading qualification logic and user trust.
- Fix applied:
  - changed coalesce order to preserve existing `games_played` before event-derived fallback,
  - added historical text-games backfill (`fact_play_historical_combined.parquet`) mapped to athlete IDs with season start-year -> end-year conversion,
  - applied max(existing, historical_text_games) merge policy.
- Prevention guard:
  - treat event-participation games as low-confidence unless corroborated by lineup/box source,
  - run spot checks on known players each run (Paolo, Chet, Zion, Ja) before export.

19. **Qualified-only export hid legitimate players from user view**
- What happened: main season rankings CSV filtered to qualified pool only, so players like Chet were absent entirely.
- Why this hurt: looked like linkage failure despite player existing in inference output.
- Fix applied:
  - main export now writes all rows,
  - separate qualified CSV is emitted,
  - both `season_rank_all` and `season_rank_qualified` are kept.
- Prevention guard:
  - export contract must always include all-player file + qualified file,
  - never use qualified-only as the sole user-facing artifact.

20. **Export qualification logic referenced derived minutes column before it was created**
- What happened: `college_minutes_total_display` was used in qualification gating prior to assignment.
- Why this hurt: export script crashed and blocked refreshed rankings output.
- Fix applied:
  - reordered export steps so display minutes are computed before qualification gating.
- Prevention guard:
  - add a minimal export smoke test that validates required intermediate columns before rank/gate computation.

21. **Assumed historical minutes backfill could map star players without validating `onFloor` fidelity**
- What happened: expected `fact_player_season_stats_backfill` to contain player-level rows for modern stars, but many historical rows use team placeholders in `onFloor`, so true player minutes were not recoverable from that path.
- Why this hurt: minutes stayed zero in raw fields; confidence in exposure columns was overstated.
- Fix applied:
  - retained games-played correction via historical text game counts,
  - switched export to transparent minutes display fallback with `minutes_is_estimated` + raw column.
- Prevention guard:
  - before using any backfill as canonical exposure, run named-player spot checks (Paolo/Chet/Zion/Ja) and reject source if player-level rows are absent.

22. **Used historical `onFloor`-based backfill where substitution-text reconstruction was required**
- What happened: prior minutes backfill relied on sources that still had team-placeholder lineup context in many historical games.
- Why this hurt: raw minutes remained zero for key players despite having full manual scrape text.
- Fix applied:
  - added substitution-driven minute/game reconstruction script:
    - `ml model/college_scripts/derive_minutes_from_historical_subs.py`
  - switched exposure loader priority to prefer:
    - `fact_player_season_stats_backfill_manual_subs.parquet`
- Prevention guard:
  - exposure backfill source is not accepted until named-player raw-minute checks pass (Paolo, Chet, Zion, Ja).

23. **RAPM home/away split depended on brittle score-header string equality**
- What happened: `calculate_historical_rapm.py` split `onFloor` players into home/away using exact string match between `onFloor.team` and parsed header home team (`"| Score |"` line). Small naming format differences (abbr/punctuation) caused misassignment and season drop-off behavior.
- Why this hurt: corrupt design matrix signs (+/- side), unstable RAPM estimates, and random player disappearance in recent seasons.
- Fix applied:
  - added robust team normalization + fuzzy fallback (`difflib`) in home-team resolver,
  - resolves home team per game from both score-header and observed `onFloor` team labels,
  - added unresolved-home warning counter.
- Prevention guard:
  - before RAPM production runs, run a split sanity probe and fail if too many stints have unresolved/imbalanced home-away partitions.

24. **Promoted sparse ON/OFF-derived features into active encoder path before coverage check**
- What happened: off-court ON/OFF derivatives were wired into active Tier1 inputs before verifying cohort-level coverage under draft filter.
- Why this hurt: dead/near-dead columns in always-on branch increase noise and can silently degrade ranking behavior.
- Fix applied:
  - kept ON/OFF derivatives in feature table for audit/forward compatibility,
  - removed sparse OFF/ONOFF-diff columns from active Tier1 model inputs until coverage clears threshold.
- Prevention guard:
  - any new feature family must pass a cohort-level non-zero coverage threshold before entering active encoder lists.
  - otherwise keep as passive/audit-only columns with explicit TODO.

25. **Dunk-rate source merged but dropped in final-season aggregation path**
- What happened: dunk-rate was correctly derived and merged at season grain, but not preserved through final-season aggregation, causing DAG hard-fail (`college_dunk_rate` missing).
- Why this hurt: model missed requested athleticism proxy and gate checks failed.
- Fix applied:
  - carried `dunk_rate` through final-season selection and renamed to `college_dunk_rate`.
- Prevention guard:
  - for each new feature source, add a post-aggregation presence assertion in unified-table build tests.

26. **New proxy fields were added upstream but not carried through final-season rename path**
- What happened: `dunk_freq/putback_att_proxy/transition_freq` were merged at season grain but initially omitted from final-season aggregation/rename lists, causing missing Tier1 columns in DAG gate.
- Why this hurt: created a false-positive “wired” state in code while model input contract still failed.
- Fix applied:
  - added these columns to final-season context carry-through so unified columns are emitted.
- Prevention guard:
  - every new source feature must be validated at three points: pre-final-season frame, post-final-season frame, and final unified table schema.

27. **Assumed transition signal existed in source taxonomy without proving non-zero support**
- What happened: transition proxy was exposed from text path before validating source vocabulary prevalence.
- Why this hurt: produced a dead feature (`0%` non-zero), risking noisy feature expansion.
- Fix applied:
  - kept transition feature in table for audit only, but documented as blocked and excluded from active Tier1 branch.
- Prevention guard:
  - for any new event proxy, require pre-wire support test (`non-zero > threshold`) before promoting to active model input.

28. **Allowed RAPM season degradation to pass without hard split-quality publication gates**
- What happened: RAPM output generation had no enforced season-level split QA gates, so degraded seasons (home/away split issues, malformed roster partitions) could still publish and silently contaminate downstream targets.
- Why this hurt: player coverage collapsed year-over-year without a hard stop, and failures were discovered only after rank-quality drift.
- Fix applied:
  - added season-level split diagnostics and hard gate support in `college_scripts/calculate_historical_rapm.py`:
    - `valid_5v5_rate`, `unresolved_home_rate`, `parse_fail_rate`, `unique_players_5v5`, `n_stints`
  - added hard include/exclude behavior with strict gate mode.
  - diagnostics are written to `data/audit/historical_rapm_split_quality*.csv`.
- Prevention guard:
  - RAPM publish step must fail closed for any season that does not pass split-quality thresholds unless explicitly overridden by `--include-seasons`.

29. **Applied season include/exclude too late in RAPM pipeline (after expensive global processing)**
- What happened: include/exclude filters were evaluated after full dataset stint construction, causing very slow runs and higher failure surface.
- Why this hurt: long runtime blocked rapid diagnostics and increased debugging latency.
- Fix applied:
  - moved include/exclude filtering to immediately after parquet load in `calculate_historical_rapm.py`.
- Prevention guard:
  - all season-scoped scripts must apply season filters before any heavy transforms (clock parsing, stint aggregation, matrix assembly).

30. **Accidentally treated `--include-seasons` as a gate bypass**
- What happened: first pass of season-gate logic marked included seasons as forced pass, which weakened hard-gate safety.
- Why this hurt: could allow known-bad seasons to be reported as pass under strict mode.
- Fix applied:
  - removed include-as-bypass behavior; include now only scopes which seasons to evaluate.
- Prevention guard:
  - only `exclude` may hard-block; `include` must never override failing quality criteria unless an explicit separate `force-include` flag is introduced.

31. **Clock parser assumed only `MM:SS` and silently collapsed modern formats to zero**
- What happened: `parse_clock` accepted only `MM:SS`. Newer provider formats (e.g., `PT19M32S`, decimal variants) parsed to `0`, causing distorted absolute times and massive stint-duration artifacts.
- Why this hurt: RAPM stints degraded badly, masking the true lineup quality issue and causing season collapse in downstream outputs.
- Fix applied:
  - expanded clock parsing to support `MM:SS`, `MM:SS.s`, and `PT..M..S` formats.
- Prevention guard:
  - clock parser coverage test must include representative samples from each era/provider format before RAPM publish.

32. **Accepted degraded historical `onFloor` artifact without a hard pre-RAPM lineup-fidelity gate**
- What happened: pre-2025 historical artifact was allowed into RAPM despite heavy `TEAM,` placeholders and non-10-player rows.
- Why this hurt: downstream RAPM collapsed by season and silently degraded model targets.
- Fix applied:
  - implemented v3 reconstruction (`reconstruct_historical_onfloor_v3.py`) with explicit quality flags and season/game audits.
  - made RAPM require lineup-season gate pass before split/solve steps.
- Prevention guard:
  - no RAPM publication from historical artifacts unless lineup season audit exists and `gate_pass=True`.

33. **Legacy cleaner could treat team-token events as player activity**
- What happened: `TEAM,` token could be counted as a player in roster activity inference.
- Why this hurt: ghost-fill polluted lineups with placeholder entities.
- Fix applied:
  - patched `clean_historical_pbp_v2.py` to block team tokens from activity roster updates.
- Prevention guard:
  - include parser unit checks asserting team-event tokens never enter player sets.

34. **Ran full multi-season reconstructor in monolithic mode that stalls in final materialization**
- What happened: executed `reconstruct_historical_onfloor_v3.py` for all seasons at once (with API append in first attempt), which repeatedly spent excessive time in end-stage materialization without producing outputs in reasonable time.
- Why this hurt: delayed recovery and consumed runtime without incremental artifacts.
- Fix applied:
  - switched execution to deterministic per-season runs (`--start-season s --end-season s --no-api-append`) and merged outputs afterward.
  - preserved per-season progress logs and row counts for every run.
- Prevention guard:
  - historical reconstruction must run season-chunked by default; monolithic mode is diagnostics-only.

35. **Used in-memory pandas concat for very large seasonal merge**
- What happened: attempted to merge all per-season reconstruction outputs via pandas concat into memory.
- Why this hurt: merge hung and required manual kill, delaying downstream RAPM and training stages.
- Fix applied:
  - replaced merge operation with DuckDB out-of-core `read_parquet/read_csv_auto` + `COPY`.
- Prevention guard:
  - for multi-million-row merges, use DuckDB/SQL path by default; pandas-only merge is disallowed unless row budget is explicitly small.

36. **Used unsupported `nrows` argument in parquet reader during dashboard generation**
- What happened: initial HTML dashboard generator attempted `pd.read_parquet(..., nrows=1)`, which is not supported in the local pandas/pyarrow runtime.
- Why this hurt: dashboard generation failed and blocked canonical HTML artifact emission.
- Fix applied:
  - removed `nrows` usage and switched to schema-column discovery via standard parquet read path.
- Prevention guard:
  - for parquet schema checks, avoid row-limited parquet reads unless explicitly verified in the environment; prefer schema APIs or full-column subset reads.

37. **Crosswalk composite score exceeded contract bounds**
- What happened: first pass of draft-signal composite scoring allowed `match_score > 1.0`, violating stage-3 distribution contract (`match_score` must be in `[0,1]`).
- Why this hurt: hardening stage-3 crosswalk validation failed despite otherwise correct matching behavior.
- Fix applied:
  - clamped `match_score` to `[0.0, 1.0]` before persistence.
- Prevention guard:
  - any new composite score must include explicit range clamp + contract test before publishing artifacts.

38. **Left trainer defaults RAPM-primary while user requested EPM-priority runs**
- What happened: training defaults stayed `lambda_rapm=1.0` / `lambda_epm=0.2`, while ranking complaints were explicitly about Year-1 translation quality.
- Why this hurt: optimization target and user objective were misaligned, so ranking quality looked systematically wrong for EPM-oriented expectations.
- Fix applied:
  - added explicit objective profiles (`epm_first`, `rapm_first`, `balanced`) in `train_latent_model.py`.
  - switched default profile to `epm_first`.
  - persisted `objective_profile` + resolved weights to `model_config.json`.
  - updated inference ranking to read objective profile and rank by `pred_year1_epm` when profile is EPM-first.
- Prevention guard:
  - every training run must log objective profile + resolved lambdas in artifacts and pre-run console output.
  - inference ranking must consume model objective metadata instead of assuming RAPM-first.

39. **Introduced rank-metric early-stop without threading validation dataset into trainer signature**
- What happened: first patch added `epm_ndcg10` computation inside `train_model` but forgot to pass `val_dataset`.
- Why this hurt: would have failed at runtime on the first training epoch.
- Fix applied:
  - updated `train_model` signature to include `val_dataset` and passed it from `main`.
- Prevention guard:
  - when adding dataset-dependent metrics to training loops, include compile + one-epoch runtime smoke before full run.

40. **Enabled warm-start together with expanding-window active loop by default**
- What happened: rolling retrain initially allowed warm-start and expanding-window simultaneously.
- Why this hurt: can bias optimization toward last year’s basin instead of fully re-optimizing the larger expanded dataset.
- Fix applied:
  - expanding-window mode now disables warm-start by default.
  - explicit override added (`--allow-warm-start-expanding`) for controlled experiments.
- Prevention guard:
  - orchestration scripts must encode incompatible-default safeguards rather than relying on run-time operator judgment.

41. **Mermaid decision-node braces were eaten by Python f-string interpolation**
- What happened: dashboard generator emitted `E{"Gate pass"}` inside an f-string without escaped braces, producing broken Mermaid syntax in HTML.
- Why this hurt: canonical detailed dashboard rendered with syntax error.
- Fix applied:
  - escaped braces (`{{ ... }}`) in generator template and regenerated dashboards.
- Prevention guard:
  - any HTML template that embeds brace-based DSLs must be rendered via escaped braces or non-f-string templating.

42. **Historical games backfill used global name mapping (cross-player contamination)**
- What happened: `load_historical_text_games_backfill()` mapped manual-text names to athlete IDs by `norm_name` globally, not season-aware.
- Why this hurt: same-name players could inherit another player’s game counts, producing impossible single-season values (e.g., 60 games) in inference exports.
- Fix applied:
  - rebuilt bridge as season-aware (`season + norm_name -> athlete_id`) from `stg_shots` + `dim_games`.
  - changed merge key to `["season", "norm_name"]`.
  - added guardrail clip (`>45` games) with warning log.
- Prevention guard:
  - all name-based historical backfills must include season-constrained identity keys; global name-only merges are disallowed.

43. **Backfill merge policy used `max(existing, backfill)` and amplified bad outliers**
- What happened: exposure merge paths in training/inference used `np.maximum` when combining base values with backfill values.
- Why this hurt: any overestimated backfill value (e.g., games) overrode plausible existing values and propagated into ranked outputs.
- Fix applied:
  - changed merge semantics to `combine_first` (prefer existing source; backfill only fills missing).
  - added explicit exposure plausibility guardrails (`games_played` outside `0..45`, `minutes_total` outside `0..2000` => null).
- Prevention guard:
  - backfill sources are fallback-only; they must never dominate primary sources without a reliability score.

44. **Dropped helper export during refactor (`load_historical_text_games_backfill`)**
- What happened: while hardening backfill logic, `load_historical_text_games_backfill` was removed from `build_unified_training_table.py` exports.
- Why this hurt: inference imports failed at runtime (`ImportError`) and refresh runs reported failure despite prior model completion.
- Fix applied:
  - restored `load_historical_text_games_backfill` with season-aware mapping + clip guard.
  - recompiled and reran refresh pipeline.
- Prevention guard:
  - after refactors to shared builder modules, run `py_compile` and a direct import smoke (`from ... import ...`) before any long run.

45. **The Missing Minutes Bug (combine_first silently ignoring zeros)**
- What happened: The fallback logic `combine_first` accidentally ignored exact `0.0` values, causing 85% of NBA-mapped prospects to train on null exposure data grids.
- Why this hurt: Feature decay and distorted volume mappings.
- Fix applied:
  - Refactored the backfill scripts to aggressively override missing columns with precise literal values using pure numpy routing (`np.where`).
- Prevention guard:
  - Backfill checks should verify precise match overrides instead of relying on null-ignoring methods when dealing with explicit 0.0 metrics.

46. **Matrix Dimensional Mismatch due to Phantom Features**
- What happened: 5 hardcoded phantom box stats (e.g. `college_dunk_rate`) were left over in the `player_encoder.py` architecture but weren't built in the pipeline.
- Why this hurt: Matrix collisions (59078x49 vs 54 expected) crashed the inference pipeline.
- Fix applied:
  - Purged the unimplemented dimensions from the model architecture to match the compiled pipeline.
- Prevention guard:
  - Ensure training schemas and inference schemas dimensionally match before running pipeline prediction.

47. **Isolating RAPM Sleepers via Forced Discovery Weights**
- What happened: Target functions rewarded padding stats and generalized models, failing to strongly rank genuine on-court impact sleepers. 
- Why this hurt: Prospects like Paolo Banchero ranked higher than Zach Edey/Austin Reaves.
- Fix applied:
  - Completely disabled non-RAPM gradient priorities, and cranked lambda_rapm to 5.0 to forcefully prioritize true on-court impact.
- Prevention guard:
  - Tune profile loss priorities explicitly according to target evaluation criteria (e.g., RAPM-first).

48. **Activity merge created `_x/_y` suffixes and silently nulled core dunk/impact fields**
- What happened: `build_unified_training_table.py` merged `enhanced_features_v1.parquet` even when activity columns already existed in `college_features_v1.parquet`, producing suffixed duplicates (`dunk_rate_x/dunk_rate_y`) while downstream logic expected canonical names.
- Why this hurt: unified table emitted null `college_dunk_rate/college_dunk_freq/college_putback_rate/college_rim_pressure_index/college_contest_proxy`, causing gate failure and weak model inputs.
- Fix applied:
  - Added deterministic coalesce of `_y -> _x` into canonical activity column names.
  - Added mask-first contract enforcement then numeric fill for core activity columns.
- Prevention guard:
  - Any feature merge with overlapping columns must include explicit coalesce/drop logic and post-merge canonical-column assertions.

49. **Activity gate script import path bug (`ModuleNotFoundError: models`)**
- What happened: `run_activity_feature_gate.py` imported `models.player_encoder` without adding project root to `sys.path`.
- Why this hurt: hard gate could not run despite data being present.
- Fix applied:
  - Added root path insertion using `Path(__file__).resolve().parents[1]`.
- Prevention guard:
  - All standalone scripts that import project modules must set deterministic path bootstrap and run a direct invocation smoke.

50. **Enhanced features artifact exploded to 5.6M rows (duplicate-key amplification)**
- What happened: enhanced feature generation produced many duplicate rows at (`season`,`athlete_id`,`split_id`) due split-level merges.
- Why this hurt: expensive joins and unstable candidate row selection.
- Fix applied:
  - Added one-row-per-key collapse in `compute_all_enhanced_features` by ranking rows on non-null feature count and dropping duplicates.
- Prevention guard:
  - Enforce primary-key uniqueness on every intermediate parquet used by unified training assembly.

51. **`contest_proxy` fallback never triggered when `blk_rate` column existed but was fully null**
- What happened: logic used branch condition `if 'blk_rate' in df.columns`, which short-circuited rim-based fallback even when `blk_rate` had no signal.
- Why this hurt: `contest_proxy` remained fully missing pre-fill.
- Fix applied:
  - Branch conditions now require `blk_rate.notna().any()` (and same for foul rate), then fallback to rim-defense proxy when blocks are unavailable.
- Prevention guard:
  - Column-presence checks for branching must include non-null signal checks in sparse historical pipelines.

52. **Qualified Top-25 XLSX was not guaranteed as a dedicated mainstay artifact**
- What happened: export runs produced mixed-tab workbooks and CSVs, but a dedicated qualified-only Top-25 XLSX was not enforced every run.
- Why this hurt: user-facing review flow repeatedly missed the required file and caused avoidable back-and-forth.
- Fix applied:
  - `scripts/export_inference_rankings.py` now always writes:
    - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_qualified_only_tabs.xlsx`
  - includes per-season `YYYY_q` tabs and `overall_top25_q`.
- Prevention guard:
  - qualified-only XLSX is now a required export contract printed at runtime (`xlsx_qualified=`) and must be validated in release checks.

53. **Matched ranking files displayed global-qualified rank instead of matched-cohort rank**
- What happened: `season_rank` in matched exports (`*_matched*.csv`) was inherited from `season_rank_qualified`, not `season_rank_matched`.
- Why this hurt: strong NBA-matched prospects looked artificially buried (e.g., rank 600+) even when their matched-cohort rank was much better, creating false regression alarms.
- Fix applied:
  - `scripts/export_inference_rankings.py` now rewires `season_rank` to `season_rank_matched` for matched outputs.
  - Added dedicated matched-qualified workbook:
    - `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_matched_qualified_tabs.xlsx`
- Prevention guard:
  - Export contract check must assert that `season_rank == season_rank_matched` for matched files.

54. **Inference width mismatch after encoder-column expansion**
- What happened: adding new active columns (physicals) changed runtime feature widths, but refresh used an older checkpoint trained on narrower `career_dim`.
- Why this hurt: refresh failed with matrix multiply shape error (`59078x51` vs `47x64`) and blocked deployment outputs.
- Fix applied:
  - `nba_prospect_inference.py` now prioritizes checkpoint `model_config.json` feature lists (`tier1_columns`, `tier2_columns`, `career_columns`, `within_columns`, `year1_interaction_columns`) before tensor assembly.
- Prevention guard:
  - Inference must always use checkpoint-persisted feature contracts, never infer branch widths from latest table schema alone.

55. **Physicals were discussed repeatedly but not wired as first-class inputs**
- What happened: `college_height_in` / `college_weight_lbs` were missing from active feature contracts despite user priority.
- Why this hurt: the model ignored key body-profile signal and blocked physical-development pathway work.
- Fix applied:
  - Added recruiting physical ingest in unified/inference, active encoder wiring, and explicit trajectory fields (`nba_height_change_cm`, `nba_weight_change_lbs`) in unified.
- Prevention guard:
  - Any user-priority signal (physicals, wingspan, exposure) must be tracked in input-contract docs and validated in rebuild audits before deployment.
