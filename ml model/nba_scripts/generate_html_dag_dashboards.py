#!/usr/bin/env python3
"""Generate canonical HTML architecture dashboards and markdown mirror headers."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DOCS = BASE / "docs"
DIAGRAMS = DOCS / "diagrams"
AUDIT = BASE / "data" / "audit"
WAREHOUSE = BASE / "data" / "warehouse_v2"
TRAINING = BASE / "data" / "training"
INFERENCE = BASE / "data" / "inference"


def _load_crosswalk_metrics() -> Dict[str, str]:
    p = AUDIT / "crosswalk_nba_to_college_coverage.csv"
    if not p.exists():
        return {"status": "coverage audit missing"}
    row = pd.read_csv(p).iloc[0].to_dict()
    return {
        "total_nba": f"{int(row['total_nba']):,}",
        "matched_nba": f"{int(row['matched_nba']):,}",
        "match_rate_all": f"{100*float(row['match_rate_nba_all']):.1f}%",
        "cohort_total": f"{int(row['total_nba_2011_2024']):,}",
        "cohort_matched": f"{int(row['matched_nba_2011_2024']):,}",
        "cohort_rate": f"{100*float(row['match_rate_nba_2011_2024']):.1f}%",
        "id_exact": f"{int(row['id_exact_count']):,}",
        "tier_high": f"{int(row['draft_constrained_high_count']):,}",
        "tier_medium": f"{int(row['draft_constrained_medium_count']):,}",
        "manual": f"{int(row['manual_review_count']):,}",
        "unmatched": f"{int(row['unmatched_count']):,}",
    }


def _load_pipeline_metrics() -> Dict[str, str]:
    m = {}
    xw = WAREHOUSE / "dim_player_nba_college_crosswalk.parquet"
    ut = TRAINING / "unified_training_table.parquet"
    if xw.exists():
        d = pd.read_parquet(xw)
        m["crosswalk_rows"] = f"{len(d):,}"
    if ut.exists():
        schema_cols = pd.read_parquet(ut).columns.tolist()
        keep_cols = [c for c in ["nba_id", "y_peak_ovr", "year1_epm_tot"] if c in schema_cols]
        t = pd.read_parquet(ut, columns=keep_cols) if keep_cols else pd.read_parquet(ut)
        m["training_rows"] = f"{len(t):,}"
        if "y_peak_ovr" in t:
            m["peak_cov"] = f"{100*t['y_peak_ovr'].notna().mean():.1f}%"
        if "year1_epm_tot" in t:
            m["epm_cov"] = f"{100*t['year1_epm_tot'].notna().mean():.1f}%"
    return m


def _latest_rolling_report() -> pd.DataFrame:
    files = sorted(AUDIT.glob("rolling_retrain_report_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return pd.DataFrame()
    try:
        return pd.read_csv(files[0])
    except Exception:
        return pd.DataFrame()


def _latest_prediction_sample(limit: int = 12) -> pd.DataFrame:
    files = sorted(INFERENCE.glob("season_rankings_latest_best_current_matched_qualified.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[0])
        score_col = "pred_rank_score" if "pred_rank_score" in df.columns else ("pred_peak_rapm_rank_score" if "pred_peak_rapm_rank_score" in df.columns else None)
        if score_col is None:
            return pd.DataFrame()
        keep = [c for c in ["college_final_season", "season_rank_matched", "player_name", "pred_rank_target", score_col] if c in df.columns]
        if not keep:
            return pd.DataFrame()
        out = df[keep].copy()
        if "season_rank_matched" in out.columns:
            out = out.sort_values(["college_final_season", "season_rank_matched"], ascending=[False, True]).head(limit)
        return out
    except Exception:
        return pd.DataFrame()


def _html_shell(title: str, subtitle: str, body: str) -> str:
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{title}</title>
<style>
:root {{
  --bg:#0f1115;
  --panel:#171a21;
  --text:#ebeff7;
  --muted:#9aa6b2;
  --accent:#26b3a8;
  --ok:#42d392;
  --warn:#f6c177;
  --danger:#ef6f6c;
}}
body {{margin:0;background:radial-gradient(1200px 700px at 20% -10%,#1b2638 0%,#0f1115 45%),linear-gradient(180deg,#0f1115,#10141b);color:var(--text);font-family:Manrope,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;}}
.wrap {{max-width:1400px;margin:0 auto;padding:24px;}}
.header {{display:flex;justify-content:space-between;align-items:end;gap:16px;flex-wrap:wrap;}}
.h1 {{font-size:30px;font-weight:800;margin:0;}}
.sub {{color:var(--muted);margin-top:6px;}}
.gen {{color:var(--muted);font-size:13px;}}
.grid {{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:18px;}}
.card {{background:var(--panel);border:1px solid #273041;border-radius:12px;padding:14px;}}
.span-3{{grid-column:span 3;}} .span-4{{grid-column:span 4;}} .span-6{{grid-column:span 6;}} .span-8{{grid-column:span 8;}} .span-12{{grid-column:span 12;}}
.k {{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}}
.v {{font-size:24px;font-weight:800;margin-top:6px;}}
.badge {{display:inline-block;background:#1e2f2d;color:#8eeadf;padding:3px 8px;border-radius:999px;font-size:12px;margin-right:6px;}}
.metric-ok {{color:var(--ok);}}
.metric-warn {{color:var(--warn);}}
.metric-danger {{color:var(--danger);}}
pre.mermaid {{background:#121722;border-radius:10px;padding:8px;overflow:auto;}}
a {{color:#8fd3ff;}}
ul {{margin:8px 0 0 16px;color:var(--muted);}}
table {{width:100%;border-collapse:collapse;font-size:13px;}}
th,td {{padding:8px 10px;border-bottom:1px solid #273041;text-align:left;}}
th {{color:#bcd2e8;font-weight:700;}}
</style>
<script type=\"module\">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose', flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}, layout: 'elk' }});
</script>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"header\">
      <div>
        <h1 class=\"h1\">{title}</h1>
        <div class=\"sub\">{subtitle}</div>
      </div>
      <div class=\"gen\">Generated: {generated}</div>
    </div>
    {body}
  </div>
</body>
</html>
"""


def _write(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def build_crosswalk_dashboard() -> None:
    m = _load_crosswalk_metrics()
    body = f"""
<div class=\"grid\">
  <div class=\"card span-3\"><div class=\"k\">Total NBA</div><div class=\"v\">{m.get('total_nba','n/a')}</div></div>
  <div class=\"card span-3\"><div class=\"k\">Matched NBA</div><div class=\"v\">{m.get('matched_nba','n/a')}</div></div>
  <div class=\"card span-3\"><div class=\"k\">Match Rate (All)</div><div class=\"v\">{m.get('match_rate_all','n/a')}</div></div>
  <div class=\"card span-3\"><div class=\"k\">Match Rate (2011-2024)</div><div class=\"v\">{m.get('cohort_rate','n/a')}</div></div>

  <div class=\"card span-4\"><div class=\"k\">ID Exact</div><div class=\"v\">{m.get('id_exact','n/a')}</div></div>
  <div class=\"card span-4\"><div class=\"k\">Draft-Constrained High</div><div class=\"v\">{m.get('tier_high','n/a')}</div></div>
  <div class=\"card span-4\"><div class=\"k\">Draft-Constrained Medium</div><div class=\"v\">{m.get('tier_medium','n/a')}</div></div>

  <div class=\"card span-12\">
    <span class=\"badge\">Canonical</span><span class=\"badge\">NBA→NCAA</span><span class=\"badge\">d_y/d_n aware</span>
    <pre class=\"mermaid\">flowchart LR
      A[dim_player_crosswalk + dim_player_nba] --> B[Seed by bbr_id/pid]
      C[all_players.parquet d_y d_n] --> B
      B --> D[Candidate pool: ID-seeded + fuzzy name]
      E[stg_shots athlete_id + final season] --> D
      D --> F[Score: name + year_gap + draft_signal]
      F --> G[Tiering]
      G --> H[Deterministic one-to-one]
      H --> I[dim_player_nba_college_crosswalk.parquet]
      H --> J[crosswalk_debug + ambiguity + unmatched + coverage]
    </pre>
  </div>
</div>
"""
    _write(
        DIAGRAMS / "crosswalk_quality_dashboard.html",
        _html_shell("Crosswalk Quality Dashboard", "Canonical NBA→NCAA linkage quality and gates", body),
    )


def build_model_arch_dashboard() -> None:
    p = _load_pipeline_metrics()
    body = f"""
<div class=\"grid\">
  <div class=\"card span-4\"><div class=\"k\">Crosswalk Rows</div><div class=\"v\">{p.get('crosswalk_rows','n/a')}</div></div>
  <div class=\"card span-4\"><div class=\"k\">Unified Training Rows</div><div class=\"v\">{p.get('training_rows','n/a')}</div></div>
  <div class=\"card span-4\"><div class=\"k\">Peak/Year1 Coverage</div><div class=\"v\">{p.get('peak_cov','n/a')} / {p.get('epm_cov','n/a')}</div></div>
  <div class=\"card span-12\">
    <pre class=\"mermaid\">flowchart TB
      A[College feature store] --> U[Unified training table]
      B[NBA target facts] --> U
      C[Crosswalk NBA→NCAA] --> U
      U --> M[Latent/Generative/Pathway models]
      M --> O[Inference + season rankings]
    </pre>
  </div>
</div>
"""
    _write(
        DIAGRAMS / "model_architecture_dashboard.html",
        _html_shell("Model Architecture Dashboard", "Input contracts, model layers, and outputs", body),
    )


def build_input_contract_dashboard() -> None:
    body = """
<div class=\"grid\">
  <div class=\"card span-12\">
    <pre class=\"mermaid\">flowchart LR
      A[College: stg_shots / career / impact / transfer] --> C[Final college season + trajectory]
      B[NBA: dim_player_nba + targets] --> D[Target surface]
      E[Crosswalk (nba_id, athlete_id)] --> F[Join plane]
      C --> F
      D --> F
      F --> G[unified_training_table.parquet]
    </pre>
    <ul>
      <li>Canonical linkage direction: NBA→NCAA.</li>
      <li>Draft signals in matching: <code>d_y</code>, <code>d_n</code>, <code>draft_year_proxy</code>.</li>
      <li>One-to-one publish gate enforced on <code>nba_id</code> and <code>athlete_id</code>.</li>
    </ul>
  </div>
</div>
"""
    _write(
        DIAGRAMS / "input_data_contract_dashboard.html",
        _html_shell("Input Data Contract Dashboard", "Data contracts and join semantics", body),
    )


def build_layered_execution_dashboard() -> None:
    body = """
<div class=\"grid\">
  <div class=\"card span-12\">
    <pre class=\"mermaid\">flowchart TB
      S1[Stage 1: Build crosswalk + audits] --> S2[Stage 2: Rebuild unified table]
      S2 --> S3[Stage 3: Gate checks]
      S3 --> S4[Stage 4: Train smoke/full]
      S4 --> S5[Stage 5: Inference + exports]
      S5 --> S6[Stage 6: Final audit publish]
    </pre>
    <ul>
      <li>Fail closed on duplicate keys, missing required columns, or confidence regression.</li>
      <li>Artifacts are emitted every run to warehouse + audit paths.</li>
    </ul>
  </div>
</div>
"""
    _write(
        DIAGRAMS / "layered_execution_dashboard.html",
        _html_shell("Layered Execution Dashboard", "Stage-gated runbook and hard-stop policy", body),
    )


def build_detailed_pipeline_dashboard() -> None:
    m = _load_pipeline_metrics()
    cx = _load_crosswalk_metrics()
    rr = _latest_rolling_report()
    sample = _latest_prediction_sample(limit=16)

    if not rr.empty:
        recent_rows = []
        for _, r in rr.sort_values("anchor_season").iterrows():
            recent_rows.append(
                f"<tr><td>{int(r.get('anchor_season', 0))}</td>"
                f"<td>{r.get('objective_profile','n/a')}</td>"
                f"<td>{r.get('monitor_metric','n/a')}</td>"
                f"<td>{float(r.get('test_epm_rmse', float('nan'))):.3f}</td>"
                f"<td>{float(r.get('test_rapm_rmse', float('nan'))):.3f}</td>"
                f"<td>{int(r.get('returncode', 1))}</td></tr>"
            )
        rolling_table = (
            "<table><thead><tr><th>Anchor</th><th>Objective</th><th>Monitor</th>"
            "<th>Test EPM RMSE</th><th>Test RAPM RMSE</th><th>RC</th></tr></thead>"
            f"<tbody>{''.join(recent_rows)}</tbody></table>"
        )
    else:
        rolling_table = "<div class='metric-warn'>No rolling_retrain_report found yet.</div>"

    if not sample.empty:
        cols = sample.columns.tolist()
        head = "".join(f"<th>{c}</th>" for c in cols)
        rows = []
        for _, r in sample.iterrows():
            rows.append("<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>")
        sample_table = f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    else:
        sample_table = "<div class='metric-warn'>No matched-qualified ranking sample found.</div>"

    body = f"""
<div class=\"grid\">
  <div class=\"card span-3\"><div class=\"k\">Crosswalk Coverage (2011-2024)</div><div class=\"v metric-ok\">{cx.get('cohort_rate','n/a')}</div></div>
  <div class=\"card span-3\"><div class=\"k\">Unified Rows</div><div class=\"v\">{m.get('training_rows','n/a')}</div></div>
  <div class=\"card span-3\"><div class=\"k\">Peak Coverage</div><div class=\"v\">{m.get('peak_cov','n/a')}</div></div>
  <div class=\"card span-3\"><div class=\"k\">Year1 EPM Coverage</div><div class=\"v\">{m.get('epm_cov','n/a')}</div></div>

  <div class=\"card span-12\">
    <span class=\"badge\">Canonical</span><span class=\"badge\">Active Learning</span><span class=\"badge\">EPM-first</span>
    <pre class=\"mermaid\">flowchart TB
      A["Data Contracts"] --> B["Unified Training Table"]
      B --> C["Train using prior years"]
      C --> D["Validate rank quality: epm_ndcg10"]
      D --> E{{"Gate pass"}}
      E -->|No| F["Stop and write audit"]
      E -->|Yes| G["Test next year out of time"]
      G --> H["Save model and metrics"]
      H --> I["Warm start next yearly run"]
      I --> C
      H --> J["Inference and matched qualified rankings"]
    </pre>
  </div>

  <div class=\"card span-6\">
    <div class=\"k\">What This Solves</div>
    <ul>
      <li>No manual player anchors: future outcomes are the anchor.</li>
      <li>Year-over-year adaptation: each season adds supervision.</li>
      <li>Era handling: sequential out-of-time retraining captures drift.</li>
      <li>Fail-closed: gate failures stop publication.</li>
    </ul>
  </div>
  <div class=\"card span-6\">
    <div class=\"k\">Current Contracts</div>
    <ul>
      <li>Crosswalk direction: <code>NBA → NCAA</code>.</li>
      <li>Rank exports include <code>matched_qualified</code> cohort.</li>
      <li>Inference emits <code>pred_rank_target</code> + <code>pred_rank_score</code>.</li>
      <li>Objective metadata persisted in model config.</li>
    </ul>
  </div>

  <div class=\"card span-12\">
    <div class=\"k\">Rolling Retrain (Latest Report)</div>
    {rolling_table}
  </div>

  <div class=\"card span-12\">
    <div class=\"k\">Latest Matched-Qualified Ranking Sample</div>
    {sample_table}
  </div>

  <div class=\"card span-12\">
    <div class=\"k\">Artifact Paths</div>
    <ul>
      <li><code>/Users/akashc/my-trankcopy/ml model/data/audit/rolling_retrain_report_*.csv</code></li>
      <li><code>/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_latest_best_current_matched_qualified.csv</code></li>
      <li><code>/Users/akashc/my-trankcopy/ml model/data/inference/season_rankings_top25_best_current_tabs.xlsx</code></li>
      <li><code>/Users/akashc/my-trankcopy/ml model/models/latent_model_*/model_config.json</code></li>
    </ul>
  </div>
</div>
"""
    _write(
        DIAGRAMS / "full_pipeline_active_learning_dashboard.html",
        _html_shell(
            "Full Pipeline Active-Learning Dashboard",
            "Canonical detailed architecture + yearly retrain behavior + live metrics",
            body,
        ),
    )


def build_activity_quality_dashboard() -> None:
    gate_json = AUDIT / "activity_feature_gate_report.json"
    gate_csv = AUDIT / "activity_feature_gate_report.csv"
    snap_json = AUDIT / "activity_restore_stage0_snapshot.json"
    gate = {}
    rows = pd.DataFrame()
    if gate_json.exists():
        try:
            gate = json.loads(gate_json.read_text(encoding="utf-8"))
        except Exception:
            gate = {}
    if gate_csv.exists():
        try:
            rows = pd.read_csv(gate_csv)
        except Exception:
            rows = pd.DataFrame()
    status = "PASS" if gate.get("passed") else "FAIL"
    status_cls = "metric-ok" if status == "PASS" else "metric-danger"
    failure_list = gate.get("failures", [])
    failures_html = (
        "<ul>" + "".join(f"<li>{f}</li>" for f in failure_list) + "</ul>"
        if failure_list else "<div class='metric-ok'>No hard-gate failures.</div>"
    )
    if not rows.empty:
        head = "".join(f"<th>{c}</th>" for c in rows.columns)
        body_rows = []
        for _, r in rows.iterrows():
            body_rows.append("<tr>" + "".join(f"<td>{r[c]}</td>" for c in rows.columns) + "</tr>")
        cov_table = f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    else:
        cov_table = "<div class='metric-warn'>Coverage report not found.</div>"

    body = f"""
<div class=\"grid\">
  <div class=\"card span-3\"><div class=\"k\">Hard Gate Status</div><div class=\"v {status_cls}\">{status}</div></div>
  <div class=\"card span-3\"><div class=\"k\">Coverage Threshold</div><div class=\"v\">{gate.get('coverage_threshold_pct','n/a')}%</div></div>
  <div class=\"card span-6\"><div class=\"k\">Artifacts</div><div class=\"v\" style=\"font-size:14px;\">{gate_csv}<br>{gate_json}<br>{snap_json}</div></div>

  <div class=\"card span-12\">
    <span class=\"badge\">Hard Fail</span><span class=\"badge\">Activity Branch</span><span class=\"badge\">Contract Gate</span>
    <pre class=\"mermaid\">flowchart LR
      A["college_features_v1.parquet"] --> B["unified_training_table.parquet"]
      B --> C["core activity columns present?"]
      C --> D["coverage >= threshold?"]
      D --> E["encoder and inference column parity?"]
      E --> F{{"Publish?"}}
      F -->|No| G["Fail closed + write audit"]
      F -->|Yes| H["Train/refresh allowed"]
    </pre>
  </div>

  <div class=\"card span-12\">
    <div class=\"k\">Core Activity Coverage</div>
    {cov_table}
  </div>

  <div class=\"card span-12\">
    <div class=\"k\">Gate Failures</div>
    {failures_html}
  </div>
</div>
"""
    _write(
        DIAGRAMS / "activity_feature_quality_dashboard.html",
        _html_shell("Activity Feature Quality Dashboard", "Dunk/activity branch gate status and live coverage", body),
    )


def build_physical_quality_dashboard() -> None:
    gate_json = AUDIT / "physical_feature_gate_report.json"
    gate_csv = AUDIT / "physical_feature_gate_report.csv"
    cov_csv = AUDIT / "physicals_coverage_by_season.csv"
    linkage_csv = AUDIT / "physicals_linkage_quality.csv"

    gate = {}
    rows = pd.DataFrame()
    season_cov = pd.DataFrame()
    linkage = pd.DataFrame()
    if gate_json.exists():
        try:
            gate = json.loads(gate_json.read_text(encoding="utf-8"))
        except Exception:
            gate = {}
    if gate_csv.exists():
        try:
            rows = pd.read_csv(gate_csv)
        except Exception:
            rows = pd.DataFrame()
    if cov_csv.exists():
        try:
            season_cov = pd.read_csv(cov_csv)
        except Exception:
            season_cov = pd.DataFrame()
    if linkage_csv.exists():
        try:
            linkage = pd.read_csv(linkage_csv)
        except Exception:
            linkage = pd.DataFrame()

    status = "PASS" if gate.get("passed") else "FAIL"
    status_cls = "metric-ok" if status == "PASS" else "metric-danger"
    failure_list = gate.get("failures", [])
    failures_html = (
        "<ul>" + "".join(f"<li>{f}</li>" for f in failure_list) + "</ul>"
        if failure_list else "<div class='metric-ok'>No hard-gate failures.</div>"
    )

    def _table_html(df: pd.DataFrame, empty_msg: str) -> str:
        if df.empty:
            return f"<div class='metric-warn'>{empty_msg}</div>"
        head = "".join(f"<th>{c}</th>" for c in df.columns)
        body_rows = []
        for _, r in df.head(25).iterrows():
            body_rows.append("<tr>" + "".join(f"<td>{r[c]}</td>" for c in df.columns) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    gate_table = _table_html(rows, "Physical gate coverage report not found.")
    season_table = _table_html(season_cov, "Season coverage report not found.")
    linkage_table = _table_html(linkage, "Linkage report not found.")

    body = f"""
<div class=\"grid\">
  <div class=\"card span-3\"><div class=\"k\">Hard Gate Status</div><div class=\"v {status_cls}\">{status}</div></div>
  <div class=\"card span-3\"><div class=\"k\">Height/Weight Threshold</div><div class=\"v\">{gate.get('coverage_threshold_pct','n/a')}%</div></div>
  <div class=\"card span-3\"><div class=\"k\">Max Unresolved Rate</div><div class=\"v\">{gate.get('max_unresolved_rate_pct','n/a')}%</div></div>
  <div class=\"card span-3\"><div class=\"k\">Artifacts</div><div class=\"v\" style=\"font-size:13px;\">{gate_csv}<br>{gate_json}<br>{cov_csv}<br>{linkage_csv}</div></div>

  <div class=\"card span-12\">
    <span class=\"badge\">Hard Fail</span><span class=\"badge\">Physicals</span><span class=\"badge\">Season-by-Season</span>
    <pre class=\"mermaid\">flowchart LR
      A["raw_team_roster_physical"] --> B["identity resolution"]
      B --> C["fact_college_player_physicals_by_season"]
      C --> D["fact_college_player_physical_trajectory"]
      D --> E["unified_training_table physical columns"]
      E --> F{{"Coverage + unresolved gates pass?"}}
      F -->|No| G["Fail closed + audit artifacts"]
      F -->|Yes| H["Train/inference publish allowed"]
    </pre>
  </div>

  <div class=\"card span-12\"><div class=\"k\">Gate Metrics</div>{gate_table}</div>
  <div class=\"card span-12\"><div class=\"k\">Coverage By Season</div>{season_table}</div>
  <div class=\"card span-12\"><div class=\"k\">Linkage Quality</div>{linkage_table}</div>
  <div class=\"card span-12\"><div class=\"k\">Gate Failures</div>{failures_html}</div>
</div>
"""
    _write(
        DIAGRAMS / "physical_feature_quality_dashboard.html",
        _html_shell("Physical Feature Quality Dashboard", "Season-by-season height/weight pipeline coverage and gates", body),
    )


def update_markdown_mirror(md_path: Path, html_path: Path, summary: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mirror_header = (
        f"# Markdown Mirror\n\n"
        f"This file is a mirror. Canonical visual artifact: `{html_path}`\n\n"
        f"Summary: {summary}\n\n"
        f"Last mirror refresh: {stamp}\n\n"
    )
    original = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    marker = "<!-- CANONICAL_HTML_MIRROR -->"
    body = f"{marker}\n{mirror_header}"
    if marker in original:
        # Replace only mirror block above prior content.
        parts = original.split(marker)
        if len(parts) >= 2:
            # Keep content after first two newlines following previous header block.
            trailing = parts[-1]
            # best effort: keep existing content after first blank line sequence.
            idx = trailing.find("\n#")
            if idx != -1:
                body += trailing[idx:]
            else:
                body += "\n" + trailing
    else:
        body += "\n" + original
    md_path.write_text(body, encoding="utf-8")


def main() -> None:
    DIAGRAMS.mkdir(parents=True, exist_ok=True)

    build_crosswalk_dashboard()
    build_model_arch_dashboard()
    build_input_contract_dashboard()
    build_layered_execution_dashboard()
    build_detailed_pipeline_dashboard()
    build_activity_quality_dashboard()
    build_physical_quality_dashboard()

    update_markdown_mirror(
        DOCS / "model_architecture_dag.md",
        DIAGRAMS / "model_architecture_dashboard.html",
        "End-to-end architecture now maintained as canonical HTML dashboard.",
    )
    update_markdown_mirror(
        DOCS / "current_inputs_dag_2026-02-18.md",
        DIAGRAMS / "input_data_contract_dashboard.html",
        "Input contract DAG mirrored; canonical contract in HTML dashboard.",
    )
    update_markdown_mirror(
        DOCS / "antigravity_full_pipeline_layered_dag_2026-02-19.md",
        DIAGRAMS / "layered_execution_dashboard.html",
        "Layered runbook DAG mirrored; stage gates and hard-stop policy canonicalized in HTML.",
    )
    update_markdown_mirror(
        DOCS / "generative_model_dag.md",
        DIAGRAMS / "model_architecture_dashboard.html",
        "Generative model DAG mirrored to canonical HTML architecture dashboard.",
    )

    index_path = DOCS / "INDEX.md"
    if index_path.exists():
        txt = index_path.read_text(encoding="utf-8")
        needle = "## Canonical HTML Dashboards\n"
        section = (
            "## Canonical HTML Dashboards\n"
            "- `docs/diagrams/model_architecture_dashboard.html`\n"
            "- `docs/diagrams/input_data_contract_dashboard.html`\n"
            "- `docs/diagrams/layered_execution_dashboard.html`\n"
            "- `docs/diagrams/crosswalk_quality_dashboard.html`\n"
            "- `docs/diagrams/full_pipeline_active_learning_dashboard.html`\n"
            "- `docs/diagrams/activity_feature_quality_dashboard.html`\n"
            "- `docs/diagrams/physical_feature_quality_dashboard.html`\n"
        )
        if needle in txt:
            start = txt.index(needle)
            rest = txt[start:]
            end = rest.find("\n## ", len(needle))
            if end == -1:
                txt = txt[:start] + section
            else:
                txt = txt[:start] + section + rest[end + 1 :]
        else:
            txt += "\n\n" + section
        index_path.write_text(txt, encoding="utf-8")

    print(json.dumps({
        "generated": [
            str(DIAGRAMS / "model_architecture_dashboard.html"),
            str(DIAGRAMS / "input_data_contract_dashboard.html"),
            str(DIAGRAMS / "layered_execution_dashboard.html"),
            str(DIAGRAMS / "crosswalk_quality_dashboard.html"),
            str(DIAGRAMS / "full_pipeline_active_learning_dashboard.html"),
            str(DIAGRAMS / "activity_feature_quality_dashboard.html"),
            str(DIAGRAMS / "physical_feature_quality_dashboard.html"),
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
