# Pipeline Alignment & Quality Fix — Coverage Update (2026-02-19)

## Scope
Additional update to the implementation plan focused on requested feature families:
- RAPM drop-off validation
- On/Off impact coverage
- Athleticism + disruption + threshold-of-goodness proxies
- Full-population wiring quality (source -> unified table -> active model inputs)

## What Was Checked
Artifacts:
- `/Users/akashc/my-trankcopy/ml model/data/historical_rapm_results_enhanced.csv`
- `/Users/akashc/my-trankcopy/ml model/data/college_feature_store/college_impact_stack_v1.parquet`
- `/Users/akashc/my-trankcopy/ml model/data/training/unified_training_table.parquet`
- `/Users/akashc/my-trankcopy/ml model/data/audit/pipeline_alignment_feature_coverage_2026-02-19.csv`
- `/Users/akashc/my-trankcopy/ml model/data/audit/pipeline_alignment_feature_coverage_by_season_2026-02-19.csv`

Code paths audited/updated:
- `/Users/akashc/my-trankcopy/ml model/college_scripts/calculate_historical_rapm.py`
- `/Users/akashc/my-trankcopy/ml model/college_scripts/build_college_impact_stack_v1.py`
- `/Users/akashc/my-trankcopy/ml model/nba_scripts/build_unified_training_table.py`
- `/Users/akashc/my-trankcopy/ml model/models/player_encoder.py`
- `/Users/akashc/my-trankcopy/ml model/nba_scripts/nba_prospect_inference.py`

## Current Reality (Post-Patch)

### 1) RAPM Drop-Off (still partially unresolved)
Historical RAPM player counts by season (enhanced file):
- 2018: 5,811
- 2019: 3,025
- 2020: 1,194
- 2021: 920
- 2022: 655
- 2023: 260

Interpretation:
- The previous parser hardening removed one brittle failure mode (home/away exact team string match), but the late-year collapse is still present.
- This is now likely multi-factor: lineup fidelity drift (onFloor quality/identity), not just header parsing.

### 2) On/Off Impact Coverage
In unified training table (post-2011 cohort):
- `college_on_net_rating`: 6.29% non-null
- `college_on_ortg`: 6.29% non-null
- `college_on_drtg`: 6.38% non-null
- `college_off_*` and `college_on_off_*_diff`: 0.0% in this cohort

Interpretation:
- OFF-side fields are computed in impact stack but currently source-limited by season/provider coverage and cohort overlap.
- They are wired and available in the table schema, but not yet broadly populated for active learning value.

### 3) Athleticism / Defensive Activity / Threshold Proxies
In unified training table:
- `college_dunk_rate`: 98.50% non-null (86.29% non-zero)
- `college_dunk_freq`: 100% non-null (87.32% non-zero)
- `college_rim_pressure_index`: 100% non-null (99.44% non-zero)
- `college_stl_total_per100poss`: 100% non-null (97.65% non-zero)
- `college_blk_total_per100poss`: 100% non-null (92.39% non-zero)
- `college_contest_proxy`: 100% non-null (92.39% non-zero)
- `college_putback_rate`: 8.26% non-null (5.92% non-zero)
- `college_transition_freq`: 100% non-null but 0% non-zero (dead under current source taxonomy)

Interpretation:
- Dunks/steals/blocks/rim-pressure are now robustly populated and linked.
- Putback and transition are source-limited with current tags.

## Fixes Implemented in This Pass
1. Added activity-proxy extraction in unified builder from shot + play text taxonomy:
- `dunk_rate`, `dunk_freq`, `putback_att_proxy`, `transition_freq`
2. Preserved these through final-season aggregation and unified rename surface:
- `college_dunk_rate`, `college_dunk_freq`, `college_putback_att_proxy`, `college_transition_freq`
3. Added derived features in unified table:
- `college_putback_rate`
- `college_rim_pressure_index`
- `college_contest_proxy` (provisional until foul-attribution is complete)
4. Added to active model input branch (Tier1):
- `college_dunk_rate`, `college_dunk_freq`, `college_putback_rate`, `college_rim_pressure_index`, `college_contest_proxy`
5. Extended inference meta features to consume these same fields.

## Population Status Matrix

### Fully usable now (high coverage)
- `college_dunk_rate`, `college_dunk_freq`
- `college_rim_pressure_index`
- `college_stl_total_per100poss`, `college_blk_total_per100poss`
- `college_contest_proxy` (provisional definition)

### Partially usable now (sparse)
- `college_putback_rate` (depends on TipShot proxy prevalence)
- `college_on_*` impact fields (limited cohort overlap)

### Blocked / effectively dead under current source
- `college_transition_freq` (no transition/fast-break taxonomy in source tags)
- `college_off_*` and `college_on_off_*_diff` in post-2011 cohort (source-season overlap issue)

## Required Next Approaches to Reach “Fully Populated”

### A) RAPM 2019–2023 collapse (critical)
1. Add stint split diagnostics per season:
- unresolved home-team rate
- mean home/away roster size
- empty-side stint share
- unique player count pre/post split
2. Hard gate RAPM output publication if split-quality thresholds fail.
3. If split fidelity fails, switch to season fallback path:
- use historical RAPM only for seasons passing split QA,
- use on/off proxy impacts for failed seasons until reconstruction is fixed.

### B) Off-court impact broad population
1. Backfill player-game availability across seasons where possible.
2. Keep OFF and ON/OFF-diff columns wired but out of mandatory gates until non-null coverage clears threshold.
3. Add explicit availability mask columns for OFF metrics to prevent silent misuse.

### C) Transition / putback proxy quality
1. Transition: build sequence-based transition detector from possession chain (steal/rebound -> shot within window) rather than playType text.
2. Putback: replace TipShot proxy with rebound-to-shot linkage within N seconds and same offense possession.

### D) Contest proxy finalization
1. Introduce foul-attribution by play-level role extraction (or trusted foul season stats backfill).
2. Replace provisional proxy with target form:
- `contest_proxy = blk_rate / (blk_rate + foul_rate + eps)`

## Gates for Next Big Run
1. `college_dunk_*`, `stl/blk/rim_pressure`: non-null >= 95%
2. No dead active Tier1 features (non-zero floor > 0.5%)
3. RAPM split-quality diagnostics pass for each included season
4. If RAPM season fails QA, explicitly drop that season from RAPM features and log fallback
5. On/Off OFF metrics remain non-blocking until source coverage threshold is met

## Update: RAPM split diagnostics + hard gates implemented

Implemented in `/Users/akashc/my-trankcopy/ml model/college_scripts/calculate_historical_rapm.py`:
- season-level split-quality diagnostics emission (`--diagnostics-csv`)
- strict hard-gate behavior (`--strict-gates` default on)
- explicit season controls (`--include-seasons`, `--exclude-seasons`)
- solve matrix now restricted to valid 5v5 split stints.

Default gate dimensions now tracked:
- `n_stints`
- `valid_5v5_rate`
- `unresolved_home_rate`
- `parse_fail_rate`
- `unique_players_5v5`

This prevents future silent degradation from being published into RAPM target artifacts.

### First diagnostic result (2019–2023)
Command run:
- `python3 /Users/akashc/my-trankcopy/ml model/college_scripts/calculate_historical_rapm.py --input-parquet /Users/akashc/my-trankcopy/ml model/data/fact_play_historical_combined.parquet --include-seasons 2019,2020,2021,2022,2023 --diagnostics-only --diagnostics-csv /Users/akashc/my-trankcopy/ml model/data/audit/historical_rapm_split_quality_latest.csv`

Observed gate table (after clock parser hardening):
- 2019: fail (`valid_5v5_rate=0.431`)
- 2020: fail (`valid_5v5_rate=0.223`)
- 2021: fail (`valid_5v5_rate=0.108`)
- 2022: fail (`valid_5v5_rate=0.059`, `unique_players_5v5=655`)
- 2023: fail (`valid_5v5_rate=0.027`, `unique_players_5v5=260`)

Interpretation:
- Main degradation is now explicit:
  - `valid_5v5_rate` collapses year-over-year,
  - average detected side sizes are far below 5 (about 1.1 vs 1.1 by 2023),
  - implying post-2019 `onFloor` in this artifact is not consistently full 10-player lineups.
- This is why RAPM falls off each year: source lineup fidelity collapses, and strict gates now block those seasons from publication.

## Validation Commands Used
- `python3 /Users/akashc/my-trankcopy/ml model/nba_scripts/build_unified_training_table.py`
- `python3 /Users/akashc/my-trankcopy/ml model/nba_scripts/emit_full_input_dag.py`
- `pytest -q /Users/akashc/my-trankcopy/ml model/tests/test_wiring_edge_cases.py /Users/akashc/my-trankcopy/ml model/tests/test_dev_rate_label_math.py`
