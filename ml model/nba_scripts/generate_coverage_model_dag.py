#!/usr/bin/env python3
"""
Generate Coverage Model DAG
===========================
Builds a DAG from coverage_report_all.csv explaining linkage of inputs and labels
under proposals (tier1/tier2/career/within/physical → latent z → labels).

Outputs:
  - docs/diagrams/coverage_model_dag.html
  - docs/model_inputs_manifest/coverage_dag_proposal_mapping.csv (optional)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent
COVERAGE_CSV = BASE / "docs/model_inputs_manifest/coverage_report_all.csv"
OUT_HTML = BASE / "docs/diagrams/coverage_model_dag.html"
OUT_MAPPING_CSV = BASE / "docs/model_inputs_manifest/coverage_dag_proposal_mapping.csv"

# Canonical branch column sets (from models/player_encoder.py)
TIER1_COLUMNS = [
    'college_rim_fg_pct', 'college_mid_fg_pct', 'college_three_fg_pct', 'college_ft_pct',
    'college_rim_share', 'college_mid_share', 'college_three_share',
    'college_shots_total', 'college_fga_total', 'college_ft_att',
    'college_games_played', 'college_poss_proxy', 'college_minutes_total',
    'college_team_pace', 'college_is_power_conf',
    'college_team_srs', 'team_strength_srs', 'college_team_rank',
    'college_ast_total_per100poss', 'college_tov_total_per100poss',
    'college_stl_total_per100poss', 'college_blk_total_per100poss',
    'college_orb_total_per100poss', 'college_drb_total_per100poss', 'college_trb_total_per100poss',
    'college_dunk_rate', 'college_dunk_freq', 'college_putback_rate',
    'college_rim_pressure_index', 'college_contest_proxy',
    'college_transition_freq', 'college_deflection_proxy', 'college_pressure_handle_proxy',
    'college_assisted_share_rim', 'college_assisted_share_mid', 'college_assisted_share_three',
    'college_rapm_standard', 'college_o_rapm', 'college_d_rapm',
    'college_on_net_rating', 'college_on_ortg', 'college_on_drtg',
    'high_lev_att_rate', 'garbage_att_rate', 'leverage_poss_share',
    'college_three_fg_pct_z', 'final_trueShootingPct_z', 'final_usage_z',
    'college_rim_fg_pct_z', 'college_mid_fg_pct_z', 'college_ft_pct_z',
    'final_trueShootingPct_team_resid', 'final_usage_team_resid',
    'college_three_fg_pct_team_resid',
    'college_recruiting_rank', 'college_recruiting_stars', 'college_recruiting_rating',
]
TIER2_COLUMNS = [
    'college_avg_shot_dist', 'college_shot_dist_var',
    'college_corner_3_rate', 'college_corner_3_pct',
    'college_deep_3_rate', 'college_rim_purity',
    'college_xy_shots', 'college_xy_3_shots', 'college_xy_rim_shots',
]
CAREER_BASE_COLUMNS = [
    'career_years', 'college_final_season', 'draft_year_proxy',
    'season_index', 'class_year', 'age_at_season', 'has_age_at_season',
    'college_height_in', 'college_weight_lbs', 'has_college_height', 'has_college_weight',
    'wingspan_in', 'wingspan_minus_height_in', 'has_wingspan',
    'college_bmi', 'college_wingspan_to_height_ratio',
    'college_height_delta_yoy', 'college_weight_delta_yoy',
    'college_height_slope_3yr', 'college_weight_slope_3yr',
    'final_trueShootingPct', 'final_usage', 'final_poss_total',
    'final_rim_fg_pct', 'final_three_fg_pct', 'final_ft_pct',
    'slope_trueShootingPct', 'slope_usage',
    'career_wt_trueShootingPct', 'career_wt_usage',
    'delta_trueShootingPct', 'delta_usage',
    'slope_rim_fg_pct', 'slope_three_fg_pct', 'slope_ft_pct',
    'career_wt_rim_fg_pct', 'career_wt_three_fg_pct', 'career_wt_ft_pct',
    'delta_rim_fg_pct', 'delta_three_fg_pct', 'delta_ft_pct',
    'breakout_timing_avg', 'breakout_timing_volume', 'breakout_timing_usage', 'breakout_timing_eff',
    'breakout_rank_eff', 'breakout_rank_volume', 'breakout_rank_usage',
    'college_dev_p10', 'college_dev_p50', 'college_dev_p90', 'college_dev_quality_weight',
    'transfer_mean_shock', 'has_transfer_context',
    'transfer_event_count', 'transfer_max_shock',
    'transfer_conf_delta_mean', 'transfer_pace_delta_mean', 'transfer_role_delta_mean',
    'has_within_window_data',
]
WITHIN_COLUMNS = [
    'final_has_ws_last10', 'final_ws_minutes_last10', 'final_ws_pps_last10',
    'final_ws_delta_pps_last5_minus_prev5',
    'final_has_ws_breakout_timing_eff', 'final_ws_breakout_timing_eff',
]

# Physical proposal: columns that are "physical/measurements" (subset of career in code)
def _is_physical(col: str) -> bool:
    if col in {'college_height_in', 'college_weight_lbs', 'college_bmi', 'college_wingspan_to_height_ratio',
               'wingspan_in', 'wingspan_minus_height_in', 'has_wingspan', 'has_college_height', 'has_college_weight',
               'college_height_delta_yoy', 'college_weight_delta_yoy', 'college_height_slope_3yr', 'college_weight_slope_3yr'}:
        return True
    return any(x in col for x in ['combine_', 'height_w_shoes', 'wingspan_imputed', 'standing_reach',
                                   'vertical_imputed', 'max_vertical', 'lane_agility', 'three_quarter',
                                   'nba_height_change', 'nba_weight_change', 'nba_wingspan',
                                   'height_delta_yoy', 'weight_delta_yoy', 'height_slope', 'weight_slope',
                                   'height_change_entry', 'weight_change_entry', 'trajectory_obs_count',
                                   'has_combine_measured', 'has_weight', 'has_wingspan'])

# Labels (targets) — not inputs
LABEL_COLUMNS = [
    'y_peak_epm_1y_60gp', 'y_peak_epm_3y', 'latent_peak_within_7y',
    'y_peak_ovr', 'y_peak_off', 'y_peak_def',
    'year1_epm_tot', 'year1_epm_off', 'year1_epm_def',
    'gap_ts_legacy', 'gap_usg_legacy', 'made_nba',
    'y_peak_epm_window',
]


def assign_proposal(column: str) -> str:
    """Assign coverage column to proposal: tier1, tier2, career, within, physical, or label."""
    if column in LABEL_COLUMNS or column.startswith('y_peak_') or column.startswith('year1_epm') or column in ('made_nba', 'gap_ts_legacy', 'gap_usg_legacy'):
        return "label"
    if column in TIER1_COLUMNS:
        return "tier1"
    if column in TIER2_COLUMNS:
        return "tier2"
    if column in WITHIN_COLUMNS:
        return "within"
    if column in CAREER_BASE_COLUMNS:
        return "career" if not _is_physical(column) else "physical"
    # Heuristics for columns in coverage but not in player_encoder (e.g. extra manifest columns)
    if _is_physical(column):
        return "physical"
    if any(x in column for x in ['final_has_ws_', 'final_ws_']):
        return "within"
    if any(x in column for x in ['college_avg_shot_dist', 'college_shot_dist_var', 'college_corner_3', 'college_deep_3', 'college_rim_purity', 'college_xy_']):
        return "tier2"
    if any(x in column for x in ['career_years', 'final_', 'slope_', 'career_wt_', 'delta_', 'breakout_', 'college_dev_', 'transfer_', 'has_within_window']):
        return "career"
    if any(x in column for x in ['college_rim_fg_pct', 'college_mid_fg_pct', 'college_three_fg_pct', 'college_ft_pct',
                                  'college_rim_share', 'college_mid_share', 'college_three_share',
                                  'college_shots_total', 'college_fga_total', 'college_ft_att', 'college_games_played',
                                  'college_poss_proxy', 'college_minutes_total', 'college_team_', 'college_recruiting',
                                  'college_ast_total', 'college_tov_total', 'college_stl_total', 'college_blk_total',
                                  'college_orb_total', 'college_drb_total', 'college_trb_total',
                                  'college_dunk_', 'college_putback_', 'college_rim_pressure', 'college_pressure_handle',
                                  'college_assisted_', 'college_rapm_', 'college_on_net', 'college_on_ortg', 'college_on_drtg',
                                  'high_lev_att_rate', 'garbage_att_rate', 'leverage_poss_share',
                                  'final_trueShootingPct_z', 'final_usage_z', 'college_three_fg_pct_z', 'college_rim_fg_pct_z',
                                  'college_mid_fg_pct_z', 'college_ft_pct_z', 'final_trueShootingPct_team_resid',
                                  'final_usage_team_resid', 'college_three_fg_pct_team_resid']):
        return "tier1"
    return "other"


def main() -> None:
    if not COVERAGE_CSV.exists():
        raise FileNotFoundError(f"Coverage report not found: {COVERAGE_CSV}")
    with open(COVERAGE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cov_rows = list(reader)
    if not cov_rows or "column" not in cov_rows[0]:
        raise ValueError("coverage_report_all.csv must have a 'column' column")
    pct_col = "pct_non_null" if "pct_non_null" in cov_rows[0] else list(cov_rows[0].keys())[1]
    col_to_pct = {r["column"]: r.get(pct_col) for r in cov_rows}

    # Assign proposal and coverage
    rows = []
    for col in col_to_pct:
        prop = assign_proposal(col)
        try:
            pct = float(col_to_pct[col]) if col_to_pct.get(col) not in (None, "") else None
        except (TypeError, ValueError):
            pct = None
        rows.append({"column": col, "proposal": prop, "pct_non_null": pct})
    # Sort for stable output
    rows.sort(key=lambda r: (r["proposal"], r["column"]))

    # Build Mermaid DAG: proposals → Encoder → z → Labels
    # Node IDs must be alphanumeric for Mermaid
    mermaid_lines = [
        "flowchart LR",
        "  subgraph inputs[Input proposals]",
        "    T1[Tier 1\nfinal snapshot]",
        "    T2[Tier 2\nspatial]",
        "    CR[Career\nprogression]",
        "    WH[Within\nstar run]",
        "    PH[Physical\nmeasurements]",
        "  end",
        "  subgraph encoder[Encoder]",
        "    FUS[Fusion MLP]",
        "    Z[latent z]",
        "  end",
        "  subgraph labels[Labels]",
        "    L1[y_peak_epm_1y_60gp]",
        "    L2[y_peak_epm_3y]",
        "    L3[latent_peak_within_7y]",
        "    L4[year1_epm / made_nba]",
        "  end",
        "  T1 --> FUS",
        "  T2 --> FUS",
        "  CR --> FUS",
        "  WH --> FUS",
        "  PH --> FUS",
        "  FUS --> Z",
        "  Z --> L1",
        "  Z --> L2",
        "  Z --> L3",
        "  Z --> L4",
    ]

    # Second diagram: input columns grouped by proposal (summary) — use unique node IDs
    def safe_id(prefix: str, i: int, col: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9]", "_", col)[:24]
        return f"{prefix}_{i}_{base}" if base else f"{prefix}_{i}"

    detail_lines = [
        "flowchart TB",
        "  subgraph tier1[Tier 1: final-season snapshot]",
    ]
    t1_cols = [r["column"] for r in rows if r["proposal"] == "tier1"][:12]
    for i, c in enumerate(t1_cols):
        nid = safe_id("t1", i, c)
        detail_lines.append(f"    {nid}[\"{c}\"]")
    detail_lines.append("  end")
    detail_lines.append("  subgraph tier2[Tier 2: spatial]")
    t2_cols = [r["column"] for r in rows if r["proposal"] == "tier2"]
    for i, c in enumerate(t2_cols[:6]):
        nid = safe_id("t2", i, c)
        detail_lines.append(f"    {nid}[\"{c}\"]")
    detail_lines.append("  end")
    detail_lines.append("  subgraph career[Career: progression]")
    cr_cols = [r["column"] for r in rows if r["proposal"] == "career"][:8]
    for i, c in enumerate(cr_cols):
        nid = safe_id("cr", i, c)
        detail_lines.append(f"    {nid}[\"{c}\"]")
    detail_lines.append("  end")
    detail_lines.append("  subgraph within[Within: star run]")
    wh_cols = [r["column"] for r in rows if r["proposal"] == "within"]
    for i, c in enumerate(wh_cols[:6]):
        nid = safe_id("wh", i, c)
        detail_lines.append(f"    {nid}[\"{c}\"]")
    detail_lines.append("  end")
    detail_lines.append("  subgraph physical[Physical]")
    ph_cols = [r["column"] for r in rows if r["proposal"] == "physical"][:8]
    for i, c in enumerate(ph_cols):
        nid = safe_id("ph", i, c)
        detail_lines.append(f"    {nid}[\"{c}\"]")
    detail_lines.append("  end")
    detail_lines.append("  tier1 --> Z2[latent z]")
    detail_lines.append("  tier2 --> Z2")
    detail_lines.append("  career --> Z2")
    detail_lines.append("  within --> Z2")
    detail_lines.append("  physical --> Z2")
    detail_lines.append("  Z2 --> Y1[Primary: y_peak_epm_1y_60gp]")
    detail_lines.append("  Z2 --> Y2[3y: y_peak_epm_3y]")
    detail_lines.append("  Z2 --> Y3[Trajectory: latent_peak_within_7y]")

    # Summary table rows
    table_rows = []
    for prop in ["tier1", "tier2", "career", "within", "physical", "label"]:
        sub = [r for r in rows if r["proposal"] == prop]
        if not sub:
            continue
        n = len(sub)
        pcts = [r["pct_non_null"] for r in sub if r["pct_non_null"] is not None]
        avg_cov = sum(pcts) / len(pcts) if pcts else 0
        ex = ", ".join(r["column"] for r in sub[:5]) + ("…" if n > 5 else "")
        table_rows.append(f"<tr><td><strong>{prop}</strong></td><td>{n}</td><td>{avg_cov:.1f}%</td><td>{ex}</td></tr>")

    html_body = f"""
<div class="grid">
  <div class="card span-12">
    <h3>Coverage model DAG: inputs → latent z → labels</h3>
    <p>Linkage of columns in <code>coverage_report_all.csv</code> under proposals (tier1, tier2, career, within, physical) to encoder branches and labels.</p>
    <pre class="mermaid">
{chr(10).join(mermaid_lines)}
    </pre>
  </div>
  <div class="card span-12">
    <h3>Inputs by proposal (sample columns)</h3>
    <pre class="mermaid">
{chr(10).join(detail_lines)}
    </pre>
  </div>
  <div class="card span-12">
    <h3>Proposal summary (from coverage_report_all.csv)</h3>
    <table class="wiring-table">
      <thead><tr><th>Proposal</th><th># columns</th><th>Mean coverage %</th><th>Example columns</th></tr></thead>
      <tbody>
        {chr(10).join(table_rows)}
      </tbody>
    </table>
  </div>
  <div class="card span-12">
    <h3>Labels (targets)</h3>
    <ul>
      <li><strong>Primary</strong>: <code>y_peak_epm_1y_60gp</code> — peak 1-year EPM (≥60 GP)</li>
      <li><strong>3y</strong>: <code>y_peak_epm_3y</code> — peak 3-year EPM</li>
      <li><strong>Trajectory</strong>: <code>latent_peak_within_7y</code> — horizon-bounded latent peak (trajectory head)</li>
      <li><strong>Aux</strong>: <code>year1_epm_tot</code>, <code>y_peak_ovr</code>, <code>made_nba</code>, gaps</li>
    </ul>
  </div>
</div>
"""
    html_shell = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Coverage Model DAG</title>
  <script type="module">import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs'; mermaid.initialize({{ startOnLoad: true, theme: 'base' }});</script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #f8f9fa; }}
    .card {{ background: #fff; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .span-12 {{ grid-column: span 12; }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 1rem; }}
    pre.mermaid {{ background: #f0f4f8; padding: 1rem; border-radius: 6px; overflow: auto; }}
    table.wiring-table {{ border-collapse: collapse; width: 100%; }}
    .wiring-table th, .wiring-table td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
    .wiring-table th {{ background: #e9ecef; }}
    code {{ background: #e9ecef; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Coverage Model DAG</h1>
  <p>Input–label linkage under proposals (latent branches: tier1, tier2, career, within, physical).</p>
  {html_body}
</body>
</html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html_shell, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")

    # Optional: write mapping CSV
    OUT_MAPPING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["column", "proposal", "pct_non_null"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_MAPPING_CSV}")


if __name__ == "__main__":
    main()
