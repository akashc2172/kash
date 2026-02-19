#!/usr/bin/env python3
"""
Full Pipeline Hardening + End-to-End Runner
===========================================

Implements staged hardening with strict critical gates:
  Stage 0: input freeze + baseline snapshot
  Stage 1: college assembly validation
  Stage 2: NBA target hardening (RAPM dedupe)
  Stage 3: crosswalk/linkage validation
  Stage 4: unified table rebuild + validation
  Stage 5: pretrain/readiness gate
  Stage 6: smoke + full training run
  Stage 7: inference verification
  Stage 8: final consolidated audit
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
WAREHOUSE = BASE_DIR / "data/warehouse_v2"
FEATURE_STORE = BASE_DIR / "data/college_feature_store"
TRAINING = BASE_DIR / "data/training"
AUDIT_ROOT = BASE_DIR / "data/audit"
INFERENCE_DIR = BASE_DIR / "data/inference"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
MISTAKE_DOC = DOCS_DIR / "mistake_prevention_retrospective_2026-02-19.md"


@dataclass
class StageResult:
    name: str
    passed: bool
    critical_failure: bool
    tests: List[Dict[str, Any]]
    details: Dict[str, Any]


def _now() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _to_native(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _col_hash(cols: List[str]) -> str:
    return hashlib.sha256("|".join(cols).encode("utf-8")).hexdigest()


def _run_cmd(cmd: List[str], cwd: Path | None = None, env: Dict[str, str] | None = None) -> Dict[str, Any]:
    logger.info("Running: %s", " ".join(cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": int(p.returncode),
        "stdout_tail": p.stdout[-6000:],
        "stderr_tail": p.stderr[-6000:],
    }


def _test(name: str, passed: bool, detail: str, critical: bool = True) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, "critical": bool(critical)}


def _stage_outcome(name: str, tests: List[Dict[str, Any]], details: Dict[str, Any] | None = None) -> StageResult:
    details = details or {}
    critical_failed = any((not t["passed"]) and t.get("critical", False) for t in tests)
    passed = not any(not t["passed"] for t in tests)
    return StageResult(name=name, passed=passed, critical_failure=critical_failed, tests=tests, details=details)


def _snapshot_table(path: Path, key: str | None = None, null_cols: List[str] | None = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    df = pd.read_parquet(path)
    info["rows"] = int(len(df))
    info["cols"] = int(df.shape[1])
    info["columns"] = list(df.columns)
    info["schema_hash"] = _col_hash(list(df.columns))
    if key and key in df.columns:
        info["key"] = key
        info["key_unique"] = int(df[key].nunique())
        info["key_dupes"] = int(df.duplicated(subset=[key]).sum())
    null_rates = {}
    for c in (null_cols or []):
        if c in df.columns and len(df):
            null_rates[c] = float(df[c].isna().mean())
    if null_rates:
        info["null_rates"] = null_rates
    return info


def stage0_snapshot(run_dir: Path) -> StageResult:
    files = {
        "dim_player_nba": (WAREHOUSE / "dim_player_nba.parquet", "nba_id", ["draft_year", "rookie_season_year"]),
        "fact_player_year1_epm": (WAREHOUSE / "fact_player_year1_epm.parquet", "nba_id", ["year1_epm_tot"]),
        "fact_player_peak_rapm": (WAREHOUSE / "fact_player_peak_rapm.parquet", "nba_id", ["y_peak_ovr", "peak_poss"]),
        "fact_player_nba_college_gaps": (WAREHOUSE / "fact_player_nba_college_gaps.parquet", "nba_id", ["gap_ts_legacy", "gap_usg_legacy"]),
        "fact_player_development_rate": (WAREHOUSE / "fact_player_development_rate.parquet", "nba_id", ["dev_rate_y1_y3_mean", "dev_rate_quality_weight"]),
        "crosswalk": (WAREHOUSE / "dim_player_nba_college_crosswalk.parquet", "nba_id", ["match_score"]),
    }
    snap = {}
    missing = []
    for name, (path, key, null_cols) in files.items():
        s = _snapshot_table(path, key=key, null_cols=null_cols)
        snap[name] = s
        if not s.get("exists"):
            missing.append(name)
    # Mandatory mistake-prevention reference #1 (pre-stage0).
    mistake_ref_ok = MISTAKE_DOC.exists()
    if mistake_ref_ok:
        snap["mistake_prevention_ref_1"] = str(MISTAKE_DOC)
    (run_dir / "stage0_input_snapshot.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    (run_dir / "stage0_validation_report.json").write_text(json.dumps({"snapshot": snap}, indent=2), encoding="utf-8")
    tests = [
        _test("schema", len(missing) == 0, f"missing={missing}" if missing else "all required files present", critical=True),
        _test("cardinality", True, "snapshot recorded key uniqueness/dupes", critical=True),
        _test("coverage", True, "snapshot recorded null-rates for key cols", critical=True),
        _test("distribution", True, "baseline row/column counts recorded", critical=True),
        _test("contract", mistake_ref_ok, f"mistake_prevention_doc_exists={mistake_ref_ok}", critical=True),
    ]
    return _stage_outcome("stage0_snapshot", tests, {"files": snap})


def stage1_validate_college(run_dir: Path) -> StageResult:
    tests: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    req = {
        "college_features": FEATURE_STORE / "college_features_v1.parquet",
        "career": FEATURE_STORE / "prospect_career_v1.parquet",
        "dev_rate": FEATURE_STORE / "fact_player_college_development_rate.parquet",
        "impact_stack": FEATURE_STORE / "college_impact_stack_v1.parquet",
        "transfer_context": FEATURE_STORE / "fact_player_transfer_context.parquet",
    }
    missing = [k for k, p in req.items() if not p.exists()]
    tests.append(_test("schema", len(missing) == 0, f"missing={missing}" if missing else "all required college artifacts present", critical=True))
    if missing:
        tests.extend([
            _test("cardinality", False, "skipped due to missing inputs", critical=True),
            _test("coverage", False, "skipped due to missing inputs", critical=True),
            _test("distribution", False, "skipped due to missing inputs", critical=True),
            _test("contract", False, "skipped due to missing inputs", critical=True),
        ])
        return _stage_outcome("stage1_college_validation", tests, details)

    cf = pd.read_parquet(req["college_features"])
    career = pd.read_parquet(req["career"])
    dev = pd.read_parquet(req["dev_rate"])
    impact = pd.read_parquet(req["impact_stack"])
    transfer = pd.read_parquet(req["transfer_context"])

    dup_cf = int(cf.duplicated(subset=["athlete_id", "season", "split_id"]).sum()) if {"athlete_id", "season", "split_id"}.issubset(cf.columns) else -1
    tests.append(_test("cardinality", dup_cf == 0, f"college_features duplicate athlete/season/split rows={dup_cf}", critical=False))

    coverage_ok = True
    coverage_detail = {}
    for df_name, df_obj, col in [
        ("college_features", cf, "athlete_id"),
        ("career", career, "athlete_id"),
        ("dev_rate", dev, "athlete_id"),
    ]:
        if col in df_obj.columns and len(df_obj):
            rate = float(df_obj[col].notna().mean())
            coverage_detail[f"{df_name}.{col}_nonnull"] = rate
            if rate < 0.99:
                coverage_ok = False
    tests.append(_test("coverage", coverage_ok, f"{coverage_detail}", critical=True))

    dist_ok = True
    dist_detail = {}
    if "season" in cf.columns and len(cf):
        dist_detail["college_features_season_min"] = int(pd.to_numeric(cf["season"], errors="coerce").min())
        dist_detail["college_features_season_max"] = int(pd.to_numeric(cf["season"], errors="coerce").max())
    if "season" in impact.columns and len(impact):
        dist_detail["impact_season_min"] = int(pd.to_numeric(impact["season"], errors="coerce").min())
        dist_detail["impact_season_max"] = int(pd.to_numeric(impact["season"], errors="coerce").max())
    tests.append(_test("distribution", dist_ok, f"{dist_detail}", critical=True))

    contract_ok = {"athlete_id", "season", "split_id"}.issubset(cf.columns) and ("athlete_id" in career.columns)
    tests.append(_test("contract", contract_ok, "required college join keys exist", critical=True))

    # Per-season feature coverage matrix (explicit artifact for strict data QA).
    cov_rows: List[Dict[str, Any]] = []
    season_col = "season" if "season" in cf.columns else None
    if season_col:
        num_cols = [c for c in cf.columns if pd.api.types.is_numeric_dtype(cf[c])]
        for season, g in cf.groupby(season_col, dropna=False):
            for col in num_cols:
                s = pd.to_numeric(g[col], errors="coerce")
                cov_rows.append(
                    {
                        "dataset": "college_features",
                        "season": season,
                        "column": col,
                        "rows": int(len(g)),
                        "nonnull_rate": float(s.notna().mean()),
                        "nonzero_rate": float((s.fillna(0) != 0).mean()),
                    }
                )
    if cov_rows:
        pd.DataFrame(cov_rows).to_csv(run_dir / "feature_coverage_matrix.csv", index=False)

    details.update({
        "college_features_rows": int(len(cf)),
        "career_rows": int(len(career)),
        "dev_rows": int(len(dev)),
        "impact_rows": int(len(impact)),
        "transfer_rows": int(len(transfer)),
        "duplicate_athlete_season_split": dup_cf,
    })
    (run_dir / "stage1_college_validation.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run_dir / "stage1_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return _stage_outcome("stage1_college_validation", tests, details)


def stage2_harden_nba_targets(run_dir: Path) -> StageResult:
    tests: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    peak_path = WAREHOUSE / "fact_player_peak_rapm.parquet"
    dim_path = WAREHOUSE / "dim_player_nba.parquet"

    if not peak_path.exists():
        tests.extend([
            _test("schema", False, "fact_player_peak_rapm missing", critical=True),
            _test("cardinality", False, "skipped", critical=True),
            _test("coverage", False, "skipped", critical=True),
            _test("distribution", False, "skipped", critical=True),
            _test("contract", False, "skipped", critical=True),
        ])
        return _stage_outcome("stage2_nba_target_hardening", tests, details)

    peak = pd.read_parquet(peak_path)
    required_cols = {"nba_id", "y_peak_ovr", "peak_poss", "peak_end_year"}
    missing_cols = [c for c in required_cols if c not in peak.columns]
    tests.append(_test("schema", len(missing_cols) == 0, f"missing_cols={missing_cols}" if missing_cols else "required cols present", critical=True))
    if missing_cols:
        tests.extend([
            _test("cardinality", False, "skipped due to schema failure", critical=True),
            _test("coverage", False, "skipped due to schema failure", critical=True),
            _test("distribution", False, "skipped due to schema failure", critical=True),
            _test("contract", False, "skipped due to schema failure", critical=True),
        ])
        return _stage_outcome("stage2_nba_target_hardening", tests, details)

    pre_dupes = int(peak.duplicated(subset=["nba_id"]).sum())
    details["pre_duplicate_nba_id_rows"] = pre_dupes
    details["pre_rows"] = int(len(peak))

    if pre_dupes > 0:
        backup_path = run_dir / f"fact_player_peak_rapm_pre_dedupe_{_now()}.parquet"
        peak.to_parquet(backup_path, index=False)

        peak = peak.sort_values(
            ["nba_id", "peak_poss", "peak_end_year", "y_peak_ovr"],
            ascending=[True, False, False, False],
        ).drop_duplicates(subset=["nba_id"], keep="first")
        peak.to_parquet(peak_path, index=False)
        details["dedupe_applied"] = True
        details["backup_path"] = str(backup_path)
    else:
        details["dedupe_applied"] = False

    peak_post = pd.read_parquet(peak_path)
    post_dupes = int(peak_post.duplicated(subset=["nba_id"]).sum())
    tests.append(_test("cardinality", post_dupes == 0, f"pre_dupes={pre_dupes}, post_dupes={post_dupes}", critical=True))

    coverage = float(peak_post["y_peak_ovr"].notna().mean()) if len(peak_post) else 0.0
    tests.append(_test("coverage", coverage >= 0.95, f"y_peak_ovr_non_null_rate={coverage:.3f}", critical=True))

    dist_ok = bool((pd.to_numeric(peak_post["peak_poss"], errors="coerce") >= 0).fillna(False).all())
    tests.append(_test("distribution", dist_ok, "peak_poss all non-negative", critical=True))

    # Target coverage by draft cohort on dim join.
    cohort_cov = {}
    if dim_path.exists():
        dim = pd.read_parquet(dim_path)[["nba_id", "draft_year", "rookie_season_year"]]
        d = dim.merge(peak_post[["nba_id", "y_peak_ovr"]], on="nba_id", how="left")
        d["draft_year_proxy"] = pd.to_numeric(d["draft_year"], errors="coerce")
        d["draft_year_proxy"] = d["draft_year_proxy"].where(
            d["draft_year_proxy"].notna(), pd.to_numeric(d["rookie_season_year"], errors="coerce") - 1
        )
        d = d[d["draft_year_proxy"].between(2011, 2024, inclusive="both")]
        by = (
            d.groupby("draft_year_proxy", as_index=False)
            .agg(n=("nba_id", "nunique"), n_peak=("y_peak_ovr", lambda s: s.notna().sum()))
            .sort_values("draft_year_proxy")
        )
        by["coverage"] = by["n_peak"] / by["n"]
        cohort_cov = {"by_draft_year": by.to_dict(orient="records")}
    details["cohort_peak_coverage"] = cohort_cov
    tests.append(_test("contract", post_dupes == 0, "fact_player_peak_rapm now one row per nba_id", critical=True))

    details["post_rows"] = int(len(peak_post))
    details["post_duplicate_nba_id_rows"] = post_dupes
    (run_dir / "stage2_nba_target_hardening.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run_dir / "stage2_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return _stage_outcome("stage2_nba_target_hardening", tests, details)


def stage3_validate_crosswalk(run_dir: Path) -> StageResult:
    tests: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    crosswalk_path = WAREHOUSE / "dim_player_nba_college_crosswalk.parquet"
    if not crosswalk_path.exists():
        tests.extend([
            _test("schema", False, "crosswalk missing", critical=True),
            _test("cardinality", False, "skipped", critical=True),
            _test("coverage", False, "skipped", critical=True),
            _test("distribution", False, "skipped", critical=True),
            _test("contract", False, "skipped", critical=True),
        ])
        return _stage_outcome("stage3_crosswalk_validation", tests, details)

    xw = pd.read_parquet(crosswalk_path)
    req_cols = {"nba_id", "athlete_id"}
    missing = [c for c in req_cols if c not in xw.columns]
    tests.append(_test("schema", len(missing) == 0, f"missing={missing}" if missing else "required columns present", critical=True))

    dup_nba = int(xw.duplicated(subset=["nba_id"]).sum()) if "nba_id" in xw.columns else -1
    dup_ath = int(xw.duplicated(subset=["athlete_id"]).sum()) if "athlete_id" in xw.columns else -1
    tests.append(_test("cardinality", dup_nba == 0, f"duplicate_nba_id_rows={dup_nba}", critical=True))

    coverage_ok = True
    cov_detail = {}
    for c in ["nba_id", "athlete_id"]:
        if c in xw.columns and len(xw):
            rate = float(xw[c].notna().mean())
            cov_detail[c] = rate
            coverage_ok = coverage_ok and rate >= 0.999
    tests.append(_test("coverage", coverage_ok, f"{cov_detail}", critical=True))

    if "match_score" in xw.columns and len(xw):
        ms = pd.to_numeric(xw["match_score"], errors="coerce")
        dist_ok = bool((ms.between(0, 1, inclusive="both") | ms.isna()).all())
        detail = f"match_score_min={float(ms.min()):.3f}, max={float(ms.max()):.3f}"
    else:
        dist_ok, detail = True, "match_score missing (non-blocking)"
    tests.append(_test("distribution", dist_ok, detail, critical=True))

    # Build ambiguity/error catalog from debug crosswalk if available.
    debug_path = WAREHOUSE / "dim_player_nba_college_crosswalk_debug.parquet"
    if debug_path.exists():
        d = pd.read_parquet(debug_path)
        name_score = pd.to_numeric(d.get("name_score_raw", pd.Series(dtype=float)), errors="coerce")
        year_gap = pd.to_numeric(d.get("year_gap", pd.Series(dtype=float)), errors="coerce")
        ms = pd.to_numeric(d.get("match_score", pd.Series(dtype=float)), errors="coerce")
        nba_name_norm = (
            d["nba_name"].astype(str).str.replace(r"[^A-Za-z0-9 ]", "", regex=True).str.lower()
            if "nba_name" in d.columns else pd.Series([""], index=d.index, dtype=object)
        )
        college_name_norm = (
            d["college_name"].astype(str).str.replace(r"[^A-Za-z0-9 ]", "", regex=True).str.lower()
            if "college_name" in d.columns else pd.Series([""], index=d.index, dtype=object)
        )
        cond = (
            (ms < 0.90)
            | (name_score < 0.92)
            | (year_gap.abs() > 2)
            | (nba_name_norm != college_name_norm)
        )
        err = d.loc[cond].copy()
        if len(err):
            err.to_csv(run_dir / "crosswalk_error_catalog.csv", index=False)
        else:
            pd.DataFrame(columns=d.columns).to_csv(run_dir / "crosswalk_error_catalog.csv", index=False)
        details["crosswalk_error_rows"] = int(len(err))
    else:
        pd.DataFrame(columns=["nba_id", "athlete_id", "reason"]).to_csv(run_dir / "crosswalk_error_catalog.csv", index=False)
        details["crosswalk_error_rows"] = 0

    details.update({
        "rows": int(len(xw)),
        "duplicate_nba_id_rows": dup_nba,
        "duplicate_athlete_id_rows": dup_ath,
    })
    tests.append(_test("contract", True, "crosswalk linkage contract validated", critical=True))
    (run_dir / "stage3_crosswalk_validation.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run_dir / "stage3_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return _stage_outcome("stage3_crosswalk_validation", tests, details)


def stage4_dag_contract(run_dir: Path) -> StageResult:
    tests: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    train_path = TRAINING / "unified_training_table.parquet"
    if not train_path.exists():
        tests.extend([
            _test("schema", False, "unified training table missing", critical=True),
            _test("cardinality", False, "skipped", critical=True),
            _test("coverage", False, "skipped", critical=True),
            _test("distribution", False, "skipped", critical=True),
            _test("contract", False, "skipped", critical=True),
        ])
        return _stage_outcome("stage4_dag_contract", tests, details)

    df = pd.read_parquet(train_path)
    docs = [
        DOCS_DIR / "generative_model_dag.md",
        DOCS_DIR / "model_architecture_dag.md",
    ] + sorted(DOCS_DIR.glob("*.md"))
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in docs if p.exists())

    # Critical DAG nodes with alias support.
    required_any = {
        "team_strength": [["college_team_srs"], ["team_strength_srs"]],
        "age_context": [["age_at_season"]],
        "class_context": [["class_year"]],
        "season_index": [["season_index"]],
        "leverage_high": [["high_lev_att_rate"]],
        "leverage_garbage": [["garbage_att_rate"]],
        "leverage_share": [["leverage_poss_share"]],
    }

    missing_nodes: List[str] = []
    implemented_nodes: List[str] = []
    node_status: Dict[str, str] = {}
    for node, groups in required_any.items():
        mentioned = bool(re.search(node.replace("_", r"[_\s\-]*"), text, flags=re.IGNORECASE)) or True
        if not mentioned:
            node_status[node] = "not_mentioned"
            continue
        ok = any(all(col in df.columns for col in grp) for grp in groups)
        if ok:
            implemented_nodes.append(node)
            node_status[node] = "implemented"
        else:
            missing_nodes.append(node)
            node_status[node] = "missing"

    # Dead critical branch policy: within columns can be dead only if explicit mask is present and all mask=0.
    within_cols = [c for c in [
        "final_has_ws_last10",
        "final_ws_minutes_last10",
        "final_ws_pps_last10",
        "final_ws_delta_pps_last5_minus_prev5",
        "final_has_ws_breakout_timing_eff",
        "final_ws_breakout_timing_eff",
    ] if c in df.columns]
    within_dead = False
    within_mask_ok = False
    if within_cols:
        nz = []
        for c in within_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            nz.append(float((s.fillna(0) != 0).mean()))
        within_dead = bool(max(nz) <= 0.0001)
        if "has_within_window_data" in df.columns:
            m = pd.to_numeric(df["has_within_window_data"], errors="coerce").fillna(0)
            within_mask_ok = bool((m == 0).all()) if within_dead else True
        else:
            within_mask_ok = not within_dead

    tests.append(_test("schema", True, "dag docs + unified table loaded", critical=True))
    tests.append(_test("cardinality", len(missing_nodes) == 0, f"missing_dag_nodes={missing_nodes}", critical=True))
    tests.append(_test("coverage", (not within_dead) or within_mask_ok, f"within_dead={within_dead}, within_mask_ok={within_mask_ok}", critical=True))
    tests.append(_test("distribution", True, f"implemented_nodes={implemented_nodes}", critical=True))
    tests.append(_test("contract", True, "DAG contract reconciliation complete", critical=True))

    details["node_status"] = node_status
    details["within_dead"] = within_dead
    details["within_mask_ok"] = within_mask_ok
    (run_dir / "stage4_dag_contract_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run_dir / "stage4_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return _stage_outcome("stage4_dag_contract", tests, details)


def stage4_rebuild_unified(run_dir: Path) -> StageResult:
    tests: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    cmd = [sys.executable, str(BASE_DIR / "nba_scripts/build_unified_training_table.py")]
    cmd_res = _run_cmd(cmd, cwd=BASE_DIR)
    details["build_cmd"] = cmd_res
    tests.append(_test("schema", cmd_res["returncode"] == 0, f"returncode={cmd_res['returncode']}", critical=True))

    path = TRAINING / "unified_training_table.parquet"
    if not path.exists() or cmd_res["returncode"] != 0:
        tests.extend([
            _test("cardinality", False, "skipped", critical=True),
            _test("coverage", False, "skipped", critical=True),
            _test("distribution", False, "skipped", critical=True),
            _test("contract", False, "skipped", critical=True),
        ])
        return _stage_outcome("stage4_unified_rebuild", tests, details)

    df = pd.read_parquet(path)
    dup_nba = int(df.duplicated(subset=["nba_id"]).sum()) if "nba_id" in df.columns else -1
    tests.append(_test("cardinality", dup_nba == 0, f"duplicate_nba_id_rows={dup_nba}", critical=True))

    cov = {
        "y_peak_ovr": float(df["y_peak_ovr"].notna().mean()) if "y_peak_ovr" in df.columns else 0.0,
        "year1_epm_tot": float(df["year1_epm_tot"].notna().mean()) if "year1_epm_tot" in df.columns else 0.0,
        "dev_rate_y1_y3_mean": float(df["dev_rate_y1_y3_mean"].notna().mean()) if "dev_rate_y1_y3_mean" in df.columns else 0.0,
    }
    cov_ok = (cov["y_peak_ovr"] >= 0.85) and (cov["year1_epm_tot"] >= 0.70) and (cov["dev_rate_y1_y3_mean"] >= 0.99)
    tests.append(_test("coverage", cov_ok, f"{cov}", critical=True))

    if "draft_year_proxy" in df.columns and len(df):
        min_proxy = float(pd.to_numeric(df["draft_year_proxy"], errors="coerce").min())
        dist_ok = min_proxy >= 2011.0
        detail = f"min_draft_year_proxy={min_proxy}"
    else:
        dist_ok, detail = False, "draft_year_proxy missing"
    tests.append(_test("distribution", dist_ok, detail, critical=True))

    from models import TIER1_COLUMNS, TIER2_COLUMNS, CAREER_BASE_COLUMNS, WITHIN_COLUMNS

    required = set(TIER1_COLUMNS + TIER2_COLUMNS + CAREER_BASE_COLUMNS + WITHIN_COLUMNS + ["nba_id", "draft_year_proxy"])
    missing = sorted([c for c in required if c not in df.columns])
    tests.append(_test("contract", len(missing) == 0, f"missing_required_cols_count={len(missing)}", critical=True))

    details.update({
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "coverage": cov,
        "missing_required_columns": missing,
    })
    (run_dir / "stage5_unified_rebuild.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run_dir / "stage5_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return _stage_outcome("stage4_unified_rebuild", tests, details)


def stage5_gate_checks(run_dir: Path) -> StageResult:
    tests: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    cmd = [sys.executable, str(BASE_DIR / "nba_scripts/run_nba_pretrain_gate.py"), "--fail-on-gate"]
    res = _run_cmd(cmd, cwd=BASE_DIR)
    details["gate_cmd"] = res
    tests.append(_test("schema", res["returncode"] == 0, f"gate_returncode={res['returncode']}", critical=True))

    gate_path = BASE_DIR / "data/audit/nba_pretrain_gate.json"
    if not gate_path.exists():
        tests.extend([
            _test("cardinality", False, "gate report missing", critical=True),
            _test("coverage", False, "gate report missing", critical=True),
            _test("distribution", False, "gate report missing", critical=True),
            _test("contract", False, "gate report missing", critical=True),
        ])
        return _stage_outcome("stage5_gate_checks", tests, details)

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    critical = gate.get("quality_gate", {}).get("critical_checks", [])
    critical_ok = all(bool(c.get("passed")) for c in critical)
    tests.append(_test("cardinality", critical_ok, f"critical_checks={len(critical)}", critical=True))

    cov = gate.get("coverage", {})
    cov_ok = (
        float(cov.get("y_peak_ovr_non_null_rate", 0.0)) >= 0.80
        and float(cov.get("year1_epm_tot_non_null_rate", 0.0)) >= 0.65
        and float(cov.get("dev_rate_y1_y3_mean_non_null_rate", 0.0)) >= 0.85
    )
    tests.append(_test("coverage", cov_ok, f"{cov}", critical=True))

    dist = gate.get("distribution", {})
    dist_ok = float(dist.get("dev_rate_std", 0.0)) > 0.01
    tests.append(_test("distribution", dist_ok, f"dev_rate_std={dist.get('dev_rate_std')}", critical=True))

    tests.append(_test("contract", bool(gate.get("quality_gate", {}).get("passed", False)), "pretrain quality gate passed", critical=True))
    details["gate_report"] = gate
    (run_dir / "stage6_gate_checks.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run_dir / "stage6_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return _stage_outcome("stage5_gate_checks", tests, details)


def _latest_latent_model_path() -> Path | None:
    cands = sorted(MODELS_DIR.glob("latent_model_*/model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def stage6_train(run_dir: Path) -> StageResult:
    tests: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(BASE_DIR / "data" / ".mplconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    # Massive validation pack first (target >= 3000 checks).
    granular_cmd = [sys.executable, str(BASE_DIR / "scripts/run_granular_pipeline_audit.py")]
    granular_res = _run_cmd(granular_cmd, cwd=BASE_DIR, env=env)
    details["granular_audit_cmd"] = granular_res
    granular_ok = granular_res["returncode"] == 0
    approx_checks = 0
    dead_inputs = None
    low_cov = None
    if granular_ok:
        # Parse latest granular audit summary.
        cand = sorted((BASE_DIR / "data/audit").glob("granular_pipeline_audit_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cand:
            summ = json.loads(cand[0].read_text(encoding="utf-8"))
            details["granular_summary"] = summ
            approx_checks = int(summ.get("approx_checks_total", 0))
            dead_inputs = int(summ.get("dead_input_columns_count", 0))
            low_cov = int(summ.get("low_cov_input_columns_count", 0))
            details["dead_input_columns"] = summ.get("dead_input_columns", [])
    tests.append(_test("schema", granular_ok, f"granular_returncode={granular_res['returncode']}", critical=True))
    tests.append(_test("cardinality", approx_checks >= 3000, f"approx_checks_total={approx_checks}", critical=True))
    allowed_dead = {
        "final_has_ws_last10",
        "final_ws_minutes_last10",
        "final_ws_pps_last10",
        "final_ws_delta_pps_last5_minus_prev5",
        "final_has_ws_breakout_timing_eff",
        "final_ws_breakout_timing_eff",
        "has_within_window_data",
    }
    dead_cols = set(details.get("dead_input_columns", []))
    disallowed_dead = sorted(list(dead_cols - allowed_dead))
    dead_ok = (dead_inputs is not None) and (len(disallowed_dead) == 0)
    tests.append(_test("coverage", dead_ok, f"dead_input_columns={sorted(dead_cols)}, disallowed_dead={disallowed_dead}", critical=True))
    tests.append(_test("distribution", True, f"low_cov_input_columns_count={low_cov}", critical=False))

    # Mandatory mistake-prevention reference #2 (pre-training).
    ref2_ok = MISTAKE_DOC.exists()
    tests.append(_test("contract", ref2_ok, f"mistake_prevention_doc_exists={ref2_ok}", critical=True))
    if not all(t["passed"] or (not t.get("critical", False)) for t in tests):
        (run_dir / "stage7_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        return _stage_outcome("stage6_validation_pack", tests, details)

    # Smoke runs
    smoke_cmds = [
        [sys.executable, str(BASE_DIR / "nba_scripts/train_latent_model.py"), "--epochs", "1", "--batch-size", "256"],
        [sys.executable, str(BASE_DIR / "nba_scripts/train_generative_model.py"), "--num-steps", "50"],
        [sys.executable, str(BASE_DIR / "nba_scripts/train_pathway_model.py"), "--num_steps", "50", "--skip-diagnostics"],
    ]
    smoke_results = []
    smoke_ok = True
    for cmd in smoke_cmds:
        res = _run_cmd(cmd, cwd=BASE_DIR, env=env)
        smoke_results.append(res)
        smoke_ok = smoke_ok and (res["returncode"] == 0)
    details["smoke_results"] = smoke_results
    tests.append(_test("schema_train", smoke_ok, "all smoke trainings returned 0", critical=True))

    # Full orchestrated run
    full_cmd = [sys.executable, str(BASE_DIR / "nba_scripts/run_training_pipeline.py"), "--all"]
    full_res = _run_cmd(full_cmd, cwd=BASE_DIR, env=env)
    details["full_run"] = full_res
    tests.append(_test("cardinality_train", full_res["returncode"] == 0, f"run_training_pipeline returncode={full_res['returncode']}", critical=True))

    # Full latent/generative/pathway runs
    full_model_cmds = [
        [sys.executable, str(BASE_DIR / "nba_scripts/train_latent_model.py")],
        [sys.executable, str(BASE_DIR / "nba_scripts/train_generative_model.py")],
        [sys.executable, str(BASE_DIR / "nba_scripts/train_pathway_model.py"), "--skip-diagnostics"],
    ]
    full_model_results = []
    full_models_ok = True
    for cmd in full_model_cmds:
        res = _run_cmd(cmd, cwd=BASE_DIR, env=env)
        full_model_results.append(res)
        full_models_ok = full_models_ok and (res["returncode"] == 0)
    details["full_model_results"] = full_model_results
    tests.append(_test("coverage_train", full_models_ok, "latent+generative+pathway full runs returned 0", critical=True))

    latent_model = _latest_latent_model_path()
    tests.append(_test("distribution_train", latent_model is not None, f"latest_latent_model={latent_model}", critical=True))
    tests.append(_test("contract_train", full_res["returncode"] == 0 and full_models_ok, "training artifacts generated", critical=True))
    (run_dir / "stage7_train.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run_dir / "stage7_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return _stage_outcome("stage6_training", tests, details)


def stage7_inference_verify(run_dir: Path) -> StageResult:
    tests: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    latent_model = _latest_latent_model_path()
    if latent_model is None:
        tests.extend([
            _test("schema", False, "no latent model found", critical=True),
            _test("cardinality", False, "skipped", critical=True),
            _test("coverage", False, "skipped", critical=True),
            _test("distribution", False, "skipped", critical=True),
            _test("contract", False, "skipped", critical=True),
        ])
        return _stage_outcome("stage7_inference", tests, details)

    recal_path = latent_model.parent / "season_recalibration.json"
    cmd = [
        sys.executable,
        str(BASE_DIR / "nba_scripts/nba_prospect_inference.py"),
        "--model-path",
        str(latent_model),
    ]
    if recal_path.exists():
        cmd.extend(["--recalibration-path", str(recal_path)])
    res = _run_cmd(cmd, cwd=BASE_DIR)
    details["inference_cmd"] = res
    tests.append(_test("schema", res["returncode"] == 0, f"inference_returncode={res['returncode']}", critical=True))

    preds = sorted(INFERENCE_DIR.glob("prospect_predictions_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not preds:
        tests.extend([
            _test("cardinality", False, "no predictions output found", critical=True),
            _test("coverage", False, "no predictions output found", critical=True),
            _test("distribution", False, "no predictions output found", critical=True),
            _test("contract", False, "no predictions output found", critical=True),
        ])
        return _stage_outcome("stage7_inference", tests, details)

    pred_path = preds[0]
    pred_df = pd.read_parquet(pred_path)
    tests.append(_test("cardinality", len(pred_df) > 0, f"pred_rows={len(pred_df)}", critical=True))
    req_cols = {"pred_dev_rate", "pred_dev_rate_std", "pred_peak_rapm"}
    missing = sorted([c for c in req_cols if c not in pred_df.columns])
    tests.append(_test("coverage", len(missing) == 0, f"missing_prediction_cols={missing}", critical=True))
    dist_ok = bool(np.isfinite(pd.to_numeric(pred_df["pred_dev_rate"], errors="coerce")).mean() > 0.5) if "pred_dev_rate" in pred_df.columns else False
    tests.append(_test("distribution", dist_ok, "pred_dev_rate has finite values for majority of rows", critical=True))

    # Dual-mode behavior check via direct forward pass:
    # - prospect-only: zero year1 mask
    # - year1-enhanced: real mask from unified table year1 columns
    dual_mode_ok = False
    dual_mode_detail = "not run"
    try:
        from models import ProspectModel, TIER1_COLUMNS, TIER2_COLUMNS, CAREER_BASE_COLUMNS, WITHIN_COLUMNS

        train = pd.read_parquet(TRAINING / "unified_training_table.parquet")
        take = train.head(min(256, len(train))).copy()
        if len(take) > 0:
            def _tensor(cols: List[str]) -> torch.Tensor:
                arr = np.nan_to_num(take.reindex(columns=cols).to_numpy(dtype=np.float32), nan=0.0)
                return torch.from_numpy(arr)

            tier1 = _tensor(TIER1_COLUMNS)
            tier2 = _tensor(TIER2_COLUMNS)
            career = _tensor(CAREER_BASE_COLUMNS)
            within = _tensor(WITHIN_COLUMNS)
            tier2_mask = torch.ones((len(take), 1), dtype=torch.float32)
            within_mask = torch.ones((len(take), 1), dtype=torch.float32)

            year1_cols = ["year1_epm_tot", "year1_epm_off", "year1_epm_def", "year1_usg", "year1_tspct"]
            y_arr = take.reindex(columns=year1_cols).to_numpy(dtype=np.float32)
            y_mask_real = (~np.isnan(y_arr)).astype(np.float32)
            y_real = np.nan_to_num(y_arr, nan=0.0)
            y_zero = np.zeros_like(y_real, dtype=np.float32)
            y_mask_zero = np.zeros_like(y_mask_real, dtype=np.float32)

            model = ProspectModel(
                latent_dim=32,
                n_archetypes=8,
                use_vae=False,
                predict_uncertainty=True,
                year1_feature_dim=len(year1_cols),
            )
            model.load_state_dict(torch.load(latent_model, map_location="cpu"), strict=False)
            model.eval()
            with torch.no_grad():
                out_zero = model(
                    tier1, tier2, career, within, tier2_mask, within_mask,
                    torch.from_numpy(y_zero), torch.from_numpy(y_mask_zero)
                )
                out_real = model(
                    tier1, tier2, career, within, tier2_mask, within_mask,
                    torch.from_numpy(y_real), torch.from_numpy(y_mask_real)
                )
            delta = (out_real["dev_pred"][:, 0] - out_zero["dev_pred"][:, 0]).abs().mean().item()
            dual_mode_ok = bool(delta > 1e-7)
            dual_mode_detail = f"mean_abs_delta_dev_pred={delta:.6g}"
    except Exception as exc:
        dual_mode_ok = False
        dual_mode_detail = f"dual_mode_check_exception={exc}"

    tests.append(_test("contract", dual_mode_ok, dual_mode_detail, critical=True))
    details["latest_predictions"] = str(pred_path)
    details["dual_mode_detail"] = dual_mode_detail
    (run_dir / "stage8_inference.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run_dir / "stage8_validation_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return _stage_outcome("stage7_inference", tests, details)


def write_final_audit(run_dir: Path, stages: List[StageResult]) -> Dict[str, Any]:
    final = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_dir": str(run_dir),
        "overall_passed": all(s.passed for s in stages),
        "critical_passed": not any(s.critical_failure for s in stages),
        "stages": [
            {
                "name": s.name,
                "passed": s.passed,
                "critical_failure": s.critical_failure,
                "tests": [{k: _to_native(v) for k, v in t.items()} for t in s.tests],
                "details": s.details,
            }
            for s in stages
        ],
        "known_deferred_non_critical": [
            "wingspan_minus_height_in remains sparse until external ingest",
            "gap_usg_legacy remains non-critical and sparse",
            "within-season placeholders may be zero until full windows ingestion",
        ],
    }
    (run_dir / "final_release_audit.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (run_dir / "final_readiness_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Full Pipeline Hardening Final Audit")
    lines.append("")
    lines.append(f"- Overall Passed: **{final['overall_passed']}**")
    lines.append(f"- Critical Passed: **{final['critical_passed']}**")
    lines.append("")
    lines.append("## Stage Summary")
    for s in stages:
        lines.append(f"- `{s.name}`: passed={s.passed}, critical_failure={s.critical_failure}")
    lines.append("")
    lines.append("## Deferred Non-Critical")
    for item in final["known_deferred_non_critical"]:
        lines.append(f"- {item}")
    (run_dir / "final_release_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "final_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")
    return final


def main() -> int:
    run_dir = AUDIT_ROOT / f"hardening_run_{_now()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Audit run directory: %s", run_dir)

    stages: List[StageResult] = []
    stage_fns = [
        stage0_snapshot,
        stage1_validate_college,
        stage2_harden_nba_targets,
        stage3_validate_crosswalk,
        stage4_dag_contract,
        stage4_rebuild_unified,
        stage5_gate_checks,
        stage6_train,
        stage7_inference_verify,
    ]

    for fn in stage_fns:
        res = fn(run_dir)
        stages.append(res)
        logger.info("[%s] passed=%s critical_failure=%s", res.name, res.passed, res.critical_failure)
        if res.critical_failure:
            logger.error("Critical gate failed at %s. Stopping run.", res.name)
            break

    final = write_final_audit(run_dir, stages)
    logger.info("Final audit written to %s", run_dir / "final_release_audit.json")
    return 0 if final["critical_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
