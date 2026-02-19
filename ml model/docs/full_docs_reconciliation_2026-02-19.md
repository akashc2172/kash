# Full Docs Reconciliation (2026-02-19)

## Scope
Reviewed all documents listed by user under `/Users/akashc/my-trankcopy/ml model/docs` and reconciled against current code paths:
- table assembly: `nba_scripts/build_unified_training_table.py`
- encoder contract: `models/player_encoder.py`
- training loop: `nba_scripts/train_latent_model.py`
- inference assembly: `nba_scripts/nba_prospect_inference.py`
- contract verifier: `nba_scripts/emit_full_input_dag.py`

## Direct Answers

### Are we using class/freshman as a factor?
Yes, now in three places:
1. **Input surface**: `class_year`, `season_index`, `age_at_season` are in `CAREER_BASE_COLUMNS`.
2. **Encoder gating**: early-career modulation uses career context to modulate within-season branch.
3. **Active reweight loop**: iterative residual reweighting now groups by `(season, within_mask, class_year_bin)`.

### Are we correctly tying college to NBA side?
Yes for supervised training:
- College features feed encoder branches.
- NBA outcomes remain targets/auxiliary heads (`y_peak_ovr`, Year-1 EPM, gaps, dev-rate).
- Inference uses college-only features and does not leak NBA labels.

## What Was Wrong (Docs vs Code)

1. **Train/serve parity gap on feature assembly**
- Training table had leverage/SRS enrichments; inference build initially did not.
- Effect: missing Tier1 columns at serve-time, weaker/unstable rankings.
- Fix applied: inference now merges leverage features (all splits) and team-strength/SRS exactly like training.

2. **Missing canonical impact aliases in active inputs**
- Docs and DAGs expect RAPM/impact-style canonical names.
- Effect: signals existed upstream but were not exposed under model contract names.
- Fix applied: canonical aliases added and wired to `TIER1_COLUMNS`.

3. **Rebounding signal not in per-100 activity family**
- Docs emphasize broad activity/role signal; ORB/DRB/TRB were present as totals but not standardized like AST/STL/BLK/TOV.
- Fix applied: `college_orb/drb/trb_total_per100poss` added and activated in Tier1.

4. **Within-season branch state misunderstood**
- Docs include a within-season pathway; source is still mostly unpopulated.
- Effect: branch shows as dead unless explicitly treated as masked/deferred.
- Current state: branch is intentionally mask-gated; not removed, but source remains sparse.

5. **Off/def RAPM source incompleteness**
- `college_impact_stack_v1.parquet` currently has `rIPM_off_*`/`rIPM_def_*` mostly null.
- Effect: canonical off/def RAPM can be dead if strict source-only mapping is used.
- Current workaround: deterministic fallback mapping keeps columns live; upstream impact builder still needs true off/def population.

## What Is Now Wired (Implemented)

- `build_unified_training_table.py`
  - `college_orb_total_per100poss`, `college_drb_total_per100poss`, `college_trb_total_per100poss`
  - `college_rapm_standard`, `college_o_rapm`, `college_d_rapm`
  - `college_on_net_rating`, `college_on_ortg`, `college_on_drtg`
  - `college_team_srs`, `team_strength_srs`, `college_team_rank`
  - `class_year`, `season_index`, `age_at_season`, `has_age_at_season`
- `player_encoder.py`
  - New features added to active `TIER1_COLUMNS`
- `nba_prospect_inference.py`
  - Train-serve parity for leverage + SRS/team-strength + impact aliases
- `train_latent_model.py`
  - Iterative reweight grouping expanded to include class-year bins

## Remaining Structural Gaps (Still Real)

1. **Within-season source population**  
`final_ws_*` remains mostly 0 due to upstream windows coverage, not wiring.

2. **True off/def RAPM in impact stack**  
Need upstream `college_impact_stack_v1` regeneration with non-null `rIPM_off_*` and `rIPM_def_*`.

3. **Small effective supervised sample**
- Current train split has ~483 rows in walk-forward setting, which limits stability for a 63k-parameter model.
- This is a data regime issue, not just wiring.

## Recommended Next Fixes (Priority)
1. Rebuild `college_impact_stack_v1` so off/def RAPM components are populated from real lineup outputs (not fallback).
2. Populate within-season windows from player-game pipeline and re-run strict DAG gate.
3. Add a strict train-serve parity unit test asserting `missing_count==0` for `TIER1/TIER2/CAREER/WITHIN` in inference table.
4. Re-run strict hardening + full training with the refreshed impact/within sources.

## Status
- Wiring drift is significantly reduced.
- Class/freshman and college->NBA linkage are active.
- Model quality still constrained by sparse within-season and incomplete off/def impact source fields.
