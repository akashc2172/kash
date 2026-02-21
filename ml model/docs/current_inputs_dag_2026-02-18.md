<!-- CANONICAL_HTML_MIRROR -->
# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`

Summary: Input contract DAG mirrored; canonical contract in HTML dashboard.

Last mirror refresh: 2026-02-20 20:55:19


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`

Summary: Input contract DAG mirrored; canonical contract in HTML dashboard.

Last mirror refresh: 2026-02-20 20:52:26


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`

Summary: Input contract DAG mirrored; canonical contract in HTML dashboard.

Last mirror refresh: 2026-02-20 19:18:17


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`

Summary: Input contract DAG mirrored; canonical contract in HTML dashboard.

Last mirror refresh: 2026-02-20 11:52:26


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`

Summary: Input contract DAG mirrored; canonical contract in HTML dashboard.

Last mirror refresh: 2026-02-20 11:51:36


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`

Summary: Input contract DAG mirrored; canonical contract in HTML dashboard.

Last mirror refresh: 2026-02-20 11:50:14


# Markdown Mirror

This file is a mirror. Canonical visual artifact: `/Users/akashc/my-trankcopy/ml model/docs/diagrams/input_data_contract_dashboard.html`

Summary: Input contract DAG mirrored; canonical contract in HTML dashboard.

Last mirror refresh: 2026-02-20 11:18:49


# Current Input DAG (As-Wired)

This is the **current implementation DAG**, aligned to active code and table surfaces.

## End-to-end graph
```mermaid
flowchart TD
  A["warehouse.duckdb / manual scrapes"] --> B["college_features_v1.parquet"]
  A --> C["prospect_career_v1.parquet + prospect_career_long_v1.parquet"]
  A --> D["derived_box_stats_v1.parquet"]
  A --> E["fact_player_transfer_context.parquet"]
  A --> F["fact_player_college_development_rate.parquet"]
  A --> G["college_impact_stack_v1.parquet"]

  B --> H["build_unified_training_table.py"]
  C --> H
  D --> H
  E --> H
  F --> H
  G --> H
  I["dim_player_nba_college_crosswalk.parquet"] --> H
  J["fact_player_peak_rapm.parquet"] --> H
  K["fact_player_year1_epm.parquet"] --> H
  L["fact_player_nba_college_gaps.parquet"] --> H
  M["fact_player_development_rate.parquet"] --> H
  N["dim_player_nba.parquet"] --> H

  H --> U["unified_training_table.parquet"]
  U --> T["train_latent_model.py"]
  T --> O["ProspectModel / PlayerEncoder"]
  O --> P["model_best.pt + season_recalibration.json"]

  B --> Q["nba_prospect_inference.py (final-season assembly)"]
  C --> Q
  D --> Q
  E --> Q
  F --> Q
  G --> Q
  P --> Q
  Q --> R["prospect_predictions_*.parquet"]
  R --> S["export_inference_rankings.py"]
  S --> V["season_rankings_latest_best_current.csv / xlsx / per-season csvs"]
```

## Active encoder branches
- `Tier1`:
  - shot profile + volume + context
  - per-100 activity rates
  - assisted-share
  - era z-scores
  - team residuals
  - recruiting priors
- `Tier2`:
  - spatial shot geometry (mask-gated)
- `Career`:
  - final anchors
  - slopes / recency-weighted means / deltas
  - breakout timing/rank features
  - dev-rate summaries
  - transfer summaries
  - explicit year context (`college_final_season`, `draft_year_proxy`)
- `Within`:
  - final within-season window features (mask-gated)

## Targets and supervision (no leakage into X)
- Primary: `y_peak_ovr`
- Aux: `gap_ts_legacy`, `year1_epm_tot`, `dev_rate_y1_y3_mean`, `made_nba`
- Year-1 NBA interactions are used as gated auxiliary pathway, not draft-time inputs.

## Known gaps from docs vs current data
- True `age_at_season` / `class_year` are still not present in active assembled surfaces.
- Within-season fields are currently populated but mostly zero-valued in training rows (source population gap).
- `breakout_timing_eff` is effectively dead (all-zero in current unified table).

## Ranking behavior (current)
- Raw model output is preserved: `pred_peak_rapm`.
- Rank score is learned with a small ridge meta-layer on known NBA outcomes:
  - base model outputs + survival probability + exposure terms.
- Exported rankings are now **qualified-pool only by default** (no 0-game rows in primary output).
