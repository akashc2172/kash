# Model Inputs Manifest

Canonical list of model inputs and wiring for external review (e.g. ChatGPT deep research).

- **INPUTS_MASTER.csv** – Union of all possible feature columns + wiring/source (generated).
- **INPUTS_USED_{year}.csv** – Feature columns that actually made it into training for that year (after exclusions + auto-drop).
- **coverage_report_{year}.csv** – Per-feature coverage: %non-null, %non-zero, variance, min/median/max for that cohort.

**Wiring:** See `docs/diagrams/input_wiring_dashboard.html` for per-column source.

**Design notes:**
- Primary inputs are **rates** (e.g. `college_*_per100poss`). Totals are for context/exposure only. One trusted **exposure** column (e.g. `college_games_played` + `college_minutes_total`) is kept on purpose.
- **Physicals:** Canonical inputs are `college_height_in`, `college_weight_lbs`, `wingspan_in` (as-of-draft).
- **Known gaps:** orb/drb/trb from fact table may be 0 in some seasons (e.g. 2017); prefer rebound rates when available. `clutch_shooting_delta` is not populated for everyone.

Regenerate artifacts by running from repo root:
```bash
cd "ml model" && python scripts/generate_input_manifest.py [--year YYYY] [--table path/to/table.parquet]
```
