# Agent Guidelines for NCAA Prospect Model

> **For AI Agents (Cascade, Claude, GPT) working on this codebase**

## Quick Links

- **Architecture**: See `docs/model_architecture_dag.md` for visual pipeline
- **Master Plan**: `docs/ml_model_master_plan.md`
- **Next Steps**: `docs/next_steps_plan.md`

## Core Principles

### 1. Test-First Bug Fixes

**When I report a bug, don't start by trying to fix it. Instead, start by writing a test that reproduces the bug. Then, have subagents try to fix the bug and prove it with a passing test.**

This means:
1. Create a minimal reproduction case
2. Verify the test fails (confirms the bug)
3. Fix the bug
4. Verify the test passes
5. Run existing tests to prevent regressions

### 2. Never Break the Bridge

Any change to `clean_historical_pbp_v2.py` MUST output the `onFloor` JSON struct exactly as defined in `PROJECT_MAP.md` Section 3. The unified model depends on it.

### 3. Data Constraints

- **Team IDs**: Use fuzzy matching against `dim_teams` (DuckDB) before inventing new team IDs
- **No Partial Lineups**: RApM regression crashes with 4 or 6 players. Strictly enforce `len(lineup) == 5`.
- **Coverage Gates**: Tier 2 spatial features are `NaN` (not 0) when thresholds not met

### 4. Leakage Prevention

Never use post-draft NBA stats as features:
- Year-1 stats are TARGETS, not features
- Peak RAPM is the PRIMARY target
- College features are always safe (pre-draft by definition)

See `nba_scripts/nba_data_loader.py` `FORBIDDEN_FEATURE_COLUMNS` for the explicit list.

## Project Structure

```
ml model/
├── college_scripts/          # College feature engineering
│   ├── build_college_feature_store_v1.py
│   ├── build_prospect_career_store_v2.py
│   └── derive_minutes_from_historical_pbp.py
├── nba_scripts/              # Model training (Zone D)
│   ├── build_unified_training_table.py
│   ├── train_baseline_xgboost.py
│   └── run_training_pipeline.py
├── cbd_pbp/                  # CBD API ingestion
├── docs/                     # Architecture docs
└── data/                     # Data directories
    ├── college_feature_store/
    ├── warehouse_v2/
    └── training/
```

## Data Tiers

| Tier | Availability | Strategy |
|------|--------------|----------|
| Tier 1 (Universal) | 100% (2010-2025) | Always use |
| Tier 2 (Spatial) | ~25% (2019+) | Mask with `has_spatial_data` |

## Common Tasks

### Running the Pipeline

```bash
# Check prerequisites
python3 nba_scripts/run_training_pipeline.py --check

# Run with mock data (for testing)
python3 tests/test_pipeline_mock.py

# Full pipeline (when data ready)
python3 nba_scripts/run_training_pipeline.py --all
```

### Adding Features

1. Add to `build_college_feature_store_v1.py` SQL
2. Document in `docs/phase3_model_training.md` Section 3.1
3. Update `PROJECT_MAP.md` Section 6 if spatial
4. Add gating logic if Tier 2

### Adding Targets

1. Add to `build_fact_nba_college_gaps.py` or similar
2. Update `train_baseline_xgboost.py` `TARGETS` dict
3. Document in `docs/phase4_execution_analysis.md`

## Testing

- Use `tests/test_pipeline_mock.py` for integration tests
- Write unit tests for complex transformations
- Always test with `NaN` inputs (Tier 2 features will have them)

## Documentation

When making significant changes:
1. Update relevant `.md` files
2. Add docstrings to functions
3. Log important decisions in git commits

## Questions?

Check `docs/` first. If unclear, ask the user for clarification rather than making assumptions.
