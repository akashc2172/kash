# Docs Index

**Updated**: 2026-02-19

## Getting Oriented

- `WORKSPACE_STATUS.md` — current status and next steps
- `ml_model_master_plan.md` — master architecture plan
- `next_steps_plan.md` — operational checklist
- `missing_data_closure_runbook.md` — quota-safe closure operations + dual-gate readiness
- `PROJECT_MAP.md` (in repo root) — full project atlas
- `end_to_end_wiring.md` — what’s wired together (college→targets→models) + what NBA data is/ isn’t used
- `full_input_columns.md` — verified explicit column list for every model input + targets/masks
- `connectivity_proposal_v1.md` — proposal-style explanation of how *each* input feeds the encoder/latent DAG (full arrows + gating logic)
- `data_assembly_and_model_connectivity_plan.md` — nuanced connectivity plan (typed branches, future upgrades)

## Model Architecture

- `model_architecture_dag.md` — end-to-end pipeline diagram
- `generative_model_dag.md` — generative model DAG
- `latent_space_architecture.md` — latent model architecture
- `latent_input_plan.md` — input plan for latent model (multi-season handling)
- `within_season_breakout_pipeline.md` — within-season windows + breakout timing pipeline (March breakout)

## Feature Store & Data

- `college_side_stats_implementation_plan.md` — feature store plan
- `college_pbp_dev_impact_execution_spec.md` — canonical execution contract for PBP + impact stack + dev-rate + transfer context
- `career_feature_spec.md` — career store schema spec
- `phase2_feature_store_hardening.md` — feature store hardening plan
- `spatial_integration_review.md` — spatial feature integration
- `rapm_adjacent_feature_ideas.md` — additional impact-like feature ideas (draft-time safe)

## Training & Validation

- `phase3_model_training.md` — training plan
- `phase4_validation.md` — validation plan
- `phase4_execution_analysis.md` — execution analysis
- `nba_pretrain_gate.md` — pre-train critical QA gate (coverage + drift + key integrity)
- `mistake_prevention_retrospective_2026-02-19.md` — concrete mistakes + mandatory no-repeat guards for strict runs

## Historical Data & RApM

- `historical_data_pipeline.md` — historical pipeline
- `college_rapm_roadmap.md` — RApM roadmap
- `antigravity_review_2012_integration.md` — 2012 integration review

## Reviews

- `review_summary_2026_01_29.md` — summary review
- `antigravity_review_followup_2026-02-19.md` — fixes and follow-up for builder/wiring issues
- `ongoing_model_updates.md` — running log of implemented modeling changes and next candidates
- `data_quality_master_log.md` — stage-by-stage hardening outcomes + open integrity items
- `full_docs_reconciliation_2026-02-19.md` — docs-vs-code reconciliation and current structural gaps
