#!/usr/bin/env python3
"""
Contract check: no row with path_onoff_source in ['impact_or_onoff','proxy_from_box']
should have ctx_adj_onoff_net null (NaN poisoning guard).

Run after building pathway context and unified training table.
See mistake_prevention_retrospective #74.
"""
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SUP_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
PATHWAY_PATH = BASE / "data" / "college_feature_store" / "college_pathway_context_v2.parquet"
AUDIT_DIR = BASE / "data" / "audit"


def main() -> None:
    report = {"pass": False, "checked": [], "violations": 0, "detail": ""}
    if not SUP_PATH.exists():
        report["detail"] = f"Supervised table not found: {SUP_PATH}"
        _write_report(report)
        print(report["detail"])
        return
    df = pd.read_parquet(SUP_PATH, columns=["ctx_adj_onoff_net", "path_onoff_source"])
    df["path_onoff_source"] = df["path_onoff_source"].astype(str)
    has_source = df["path_onoff_source"].isin(["impact_or_onoff", "proxy_from_box"])
    ctx_null = df["ctx_adj_onoff_net"].isna()
    bad = has_source & ctx_null
    n_bad = int(bad.sum())
    report["checked"] = ["unified_training_table_supervised.parquet"]
    report["violations"] = n_bad
    if n_bad == 0:
        report["pass"] = True
        report["detail"] = "No rows with path_onoff_source in [impact_or_onoff, proxy_from_box] have ctx_adj_onoff_net null."
        print("PASS:", report["detail"])
    else:
        report["detail"] = f"Violation: {n_bad} rows have path_onoff_source in [impact_or_onoff, proxy_from_box] but ctx_adj_onoff_net is null."
        print("FAIL:", report["detail"])
    if PATHWAY_PATH.exists():
        pctx = pd.read_parquet(PATHWAY_PATH, columns=["ctx_adj_onoff_net", "path_onoff_source"])
        pctx["path_onoff_source"] = pctx["path_onoff_source"].astype(str)
        has_s = pctx["path_onoff_source"].isin(["impact_or_onoff", "proxy_from_box"])
        null_s = pctx["ctx_adj_onoff_net"].isna()
        n_pctx = int((has_s & null_s).sum())
        report["checked"].append("college_pathway_context_v2.parquet")
        report["pathway_violations"] = n_pctx
        if n_pctx > 0:
            report["pass"] = False
            print(f"Pathway parquet: {n_pctx} rows with source set but ctx_adj_onoff_net null.")
    _write_report(report)


def _write_report(report: dict) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / "pathway_context_nan_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
