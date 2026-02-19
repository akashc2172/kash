# DAG Contract Gap Audit (Targeted: 3 Requested DAG Files)

Audited files:
- `/Users/akashc/my-trankcopy/ml model/docs/generative_model_dag.md`
- `/Users/akashc/my-trankcopy/ml model/docs/model_architecture_dag.md`
- `/Users/akashc/my-trankcopy/ml model/nba_scripts/emit_full_input_dag.py`

## Direct Findings

## 1) `model_architecture_dag.md` includes feature families that are not wired in current training surface

The following named columns/families from DAG text are **absent** in `unified_training_table.parquet`:
- `rapm_standard`, `o_rapm`, `d_rapm`, `rapm_leverage`, `rapm_non_garbage`
- `high_lev_att_rate`, `garbage_att_rate`, `leverage_poss_share`
- `on_net_rating`, `on_ortg`, `on_drtg`, `seconds_on`, `poss_on`, `opp_rank`
- `gap_rapm`

Implication: current model is not consuming several impact/leverage blocks described in the DAG.

## 2) `generative_model_dag.md` is conceptual; many described latent/interaction diagnostics are not emitted as concrete artifacts

Documented concepts (trait decomposition, horseshoe-selected interactions, residual decomposition) are not currently exposed as persisted per-player outputs in inference artifacts.

Implication: the narrative-level explainability DAG is ahead of current concrete output contracts.

## 3) `emit_full_input_dag.py` validates only column presence, not signal health

It hard-fails on missing columns, but does not fail when columns are present but effectively dead (all zeros).

This allowed dead within-season fields to look “wired” while carrying no information in training.

## 4) Confirmed dead-but-present fields (training surface)

- `breakout_timing_eff`
- `final_has_ws_breakout_timing_eff`
- `final_has_ws_last10`
- `final_has_ws_last5`
- `final_ws_breakout_timing_eff`
- `final_ws_delta_pps_last5_minus_prev5`
- `final_ws_minutes_last10`
- `final_ws_pps_last10`

All currently have near-zero non-zero rate in training.

## 5) Ranking export behavior fixed

Primary ranking outputs now use qualified pool only, preventing 0-game rows from appearing in the main ranking file:
- `/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current.csv`

Current hard gate:
- `college_games_played >= 14`
- `college_poss_proxy >= 200`

## Patch Priorities from this audit

1. Reconcile DAG names to active schema (either derive columns or update DAG docs to canonical names).
2. Add non-zero signal gate into `emit_full_input_dag.py` (or companion gate) so dead columns fail readiness.
3. Populate within-season windows from true player-game coverage so within branch is not pseudo-wired.
4. Decide whether to reintroduce impact/leverage block columns from older DAG contract (and map to canonical names).
