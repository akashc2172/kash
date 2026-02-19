#!/usr/bin/env python3
"""
Build or refresh the master data quality tracking log from audit artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
AUDIT_DIR = BASE_DIR / "data" / "audit"
DOC_PATH = BASE_DIR / "docs" / "data_quality_master_log.md"


def _latest(glob_pat: str) -> Path | None:
    paths = sorted(AUDIT_DIR.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_stage(stage: dict[str, Any]) -> str:
    name = stage.get("name", "unknown")
    passed = stage.get("passed", False)
    critical = stage.get("critical_failure", False)
    mark = "PASS" if passed else "FAIL"
    out = [f"- `{name}`: **{mark}** (critical_failure={critical})"]
    failures = [t for t in stage.get("tests", []) if not t.get("passed", False)]
    for f in failures:
        out.append(
            f"  - `{f.get('name')}` (critical={f.get('critical', False)}): {f.get('detail', '')}"
        )
    return "\n".join(out)


def main() -> None:
    latest_hardening = _latest("hardening_run_*/final_release_audit.json")
    latest_granular = _latest("granular_pipeline_audit_*/summary.json")
    latest_gate = _latest("nba_pretrain_gate*.json")

    hard = _read_json(latest_hardening)
    gran = _read_json(latest_granular)
    gate = _read_json(latest_gate)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stage_lines = []
    for st in hard.get("stages", []):
        stage_lines.append(_fmt_stage(st))
    stage_block = "\n".join(stage_lines) if stage_lines else "- No hardening stage audit found."

    dead_cols = gran.get("dead_input_columns", [])
    low_cov = gran.get("low_cov_input_columns", [])

    checks_total = int(gran.get("approx_checks_total", 0))
    checks_inputs_targets = int(gran.get("approx_checks_inputs_targets", 0))
    checks_all_numeric = int(gran.get("approx_checks_all_numeric", 0))

    dead_block = "- None" if not dead_cols else "\n".join([f"- `{c}`" for c in dead_cols])
    low_block = "- None" if not low_cov else "\n".join([f"- `{c}`" for c in low_cov])

    text = f"""# Data Quality Master Log

Last updated: `{stamp}`

## Purpose
- Track every full pipeline validation and hardening run.
- Enforce strict visibility of data integrity before model decisions.
- Keep unresolved issues explicit until fixed.

## Latest Artifacts
- Hardening audit: `{latest_hardening}`  
- Granular audit: `{latest_granular}`  
- Pretrain gate: `{latest_gate}`

## Latest Hardening Stage Results
{stage_block}

## Latest Granular Coverage Snapshot
- `training_rows`: `{gran.get("training_rows", "n/a")}`
- `training_cols`: `{gran.get("training_cols", "n/a")}`
- `input_columns_checked`: `{gran.get("input_columns_checked", "n/a")}`
- `target_columns_checked`: `{gran.get("target_columns_checked", "n/a")}`
- `all_numeric_columns_checked`: `{gran.get("all_numeric_columns_checked", "n/a")}`
- `approx_checks_inputs_targets`: `{checks_inputs_targets}`
- `approx_checks_all_numeric`: `{checks_all_numeric}`
- `approx_scalar_checks_run`: `{checks_total}`
- `dead_input_columns_count`: `{gran.get("dead_input_columns_count", "n/a")}`
- `low_cov_input_columns_count`: `{gran.get("low_cov_input_columns_count", "n/a")}`
- `duplicate_nba_id_crosswalk`: `{gran.get("duplicate_nba_id_crosswalk", "n/a")}`
- `duplicate_athlete_id_crosswalk`: `{gran.get("duplicate_athlete_id_crosswalk", "n/a")}`
- `star_sanity_fail_count`: `{gran.get("star_sanity_fail_count", "n/a")}`

## Current Hard-Fail Policy
- Stop on duplicate target keys, missing contract columns, or gate failures.
- Non-critical failures remain listed here until resolved.

## Open Issues (Latest Run)
### Dead Inputs
{dead_block}

### Low-Coverage Inputs
{low_block}

## Notes
- Low coverage can be expected for optional branches (transfer/dev/spatial) depending on source availability.
- Ranking quality failures are tracked separately from wiring integrity; both must pass before declaring readiness.
"""

    DOC_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
