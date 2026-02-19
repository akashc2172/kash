# NBA Pre-Train Gate

## Purpose
Run critical readiness checks before training NBA models:
- target coverage thresholds
- dev-rate distribution sanity
- source-overlap drift checks by rookie season
- duplicate-key guard (`nba_id` uniqueness)

## Script
`/Users/akashc/my-trankcopy/ml model/nba_scripts/run_nba_pretrain_gate.py`

## Usage
- Run gate and fail on critical checks:
  - `python3 "/Users/akashc/my-trankcopy/ml model/nba_scripts/run_nba_pretrain_gate.py" --fail-on-gate`
- Run from pipeline runner:
  - `python3 "/Users/akashc/my-trankcopy/ml model/nba_scripts/run_training_pipeline.py" --gate`

## Output
- Latest report:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/nba_pretrain_gate.json`
- Timestamped snapshot:
  - `/Users/akashc/my-trankcopy/ml model/data/audit/nba_pretrain_gate_YYYYMMDD_HHMMSS.json`

## Notes
- `gap_usg_legacy` is non-critical by design.
- The gate is intended to block training only on critical failures.
