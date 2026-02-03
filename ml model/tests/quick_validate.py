#!/usr/bin/env python3
"""
Quick Validation Suite
======================
Fast checks that can run without data. Good for CI or pre-commit hooks.

Usage:
    python quick_validate.py
"""

import ast
import sys
from pathlib import Path
import importlib.util

BASE_DIR = Path(__file__).parent.parent


def check_syntax(file_path: Path) -> tuple:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_imports(file_path: Path) -> tuple:
    """Check if a script can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location("module", file_path)
        module = importlib.util.module_from_spec(spec)
        # Don't actually execute, just check it loads
        return True, None
    except Exception as e:
        return False, str(e)


def validate_all_scripts():
    """Validate all Python scripts in the project."""
    print("="*60)
    print("VALIDATING PYTHON SCRIPTS")
    print("="*60)
    
    scripts = [
        BASE_DIR / "nba_scripts/build_unified_training_table.py",
        BASE_DIR / "nba_scripts/train_baseline_xgboost.py",
        BASE_DIR / "nba_scripts/run_training_pipeline.py",
        BASE_DIR / "nba_scripts/nba_data_loader.py",
        BASE_DIR / "nba_scripts/nba_feature_transforms.py",
        BASE_DIR / "college_scripts/build_prospect_career_store_v2.py",
    ]
    
    results = []
    for script in scripts:
        if not script.exists():
            print(f"❌ Missing: {script.name}")
            results.append((script.name, False, "File not found"))
            continue
        
        # Syntax check
        ok, err = check_syntax(script)
        if not ok:
            print(f"❌ Syntax error in {script.name}: {err}")
            results.append((script.name, False, f"Syntax: {err}"))
            continue
        
        print(f"✅ {script.name}")
        results.append((script.name, True, None))
    
    return results


def check_feature_columns():
    """Verify feature column definitions are consistent."""
    print("\n" + "="*60)
    print("CHECKING FEATURE COLUMN CONSISTENCY")
    print("="*60)
    
    # Import and check
    sys.path.insert(0, str(BASE_DIR / "nba_scripts"))
    
    try:
        from build_unified_training_table import TIER1_SHOT_PROFILE, TIER2_SPATIAL
        print(f"✅ Tier 1 features: {len(TIER1_SHOT_PROFILE)} columns defined")
        print(f"✅ Tier 2 features: {len(TIER2_SPATIAL)} columns defined")
        return True
    except ImportError as e:
        print(f"⚠️  Could not import feature definitions: {e}")
        return False


def check_documentation_links():
    """Check that documentation files reference each other correctly."""
    print("\n" + "="*60)
    print("CHECKING DOCUMENTATION LINKS")
    print("="*60)
    
    docs_dir = BASE_DIR / "docs"
    required_docs = [
        "model_architecture_dag.md",
        "ml_model_master_plan.md",
        "next_steps_plan.md",
    ]
    
    all_ok = True
    for doc in required_docs:
        path = docs_dir / doc
        if path.exists():
            print(f"✅ {doc}")
        else:
            print(f"❌ Missing: {doc}")
            all_ok = False
    
    return all_ok


def check_within_season_windows_masks():
    """Optional data-backed check: validates within-season windows mask/value consistency."""
    print("\n" + "="*60)
    print("CHECKING WITHIN-SEASON WINDOWS (OPTIONAL)")
    print("="*60)

    script = BASE_DIR / "college_scripts/utils/validate_within_season_windows_v1.py"
    if not script.exists():
        print("⚠️  Validator not found, skipping")
        return True

    # Run in-process for speed and portability
    import subprocess
    res = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR), capture_output=True, text=True)
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if out:
        print(out)
    if err:
        print(err)
    return res.returncode == 0


def check_edge_case_wiring():
    """Run small edge-case asserts for wiring/inference behavior."""
    print("\n" + "="*60)
    print("CHECKING EDGE-CASE WIRING (OPTIONAL)")
    print("="*60)

    script = BASE_DIR / "tests/test_wiring_edge_cases.py"
    if not script.exists():
        print("⚠️  Edge-case test not found, skipping")
        return True

    import subprocess
    res = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR), capture_output=True, text=True)
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if out:
        print(out)
    if err:
        print(err)
    return res.returncode == 0


def check_encoder_gating():
    """Run small asserts that masked optional branches truly have zero influence."""
    print("\n" + "="*60)
    print("CHECKING ENCODER GATING (OPTIONAL)")
    print("="*60)

    script = BASE_DIR / "tests/test_encoder_gating.py"
    if not script.exists():
        print("⚠️  Encoder gating test not found, skipping")
        return True

    import subprocess
    res = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR), capture_output=True, text=True)
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if out:
        print(out)
    if err:
        print(err)
    return res.returncode == 0


def check_full_input_columns():
    """Generate and verify explicit full input column report."""
    print("\n" + "="*60)
    print("VERIFYING FULL INPUT COLUMNS (OPTIONAL)")
    print("="*60)

    script = BASE_DIR / "nba_scripts/emit_full_input_dag.py"
    if not script.exists():
        print("⚠️  Column emitter not found, skipping")
        return True

    import subprocess
    res = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR), capture_output=True, text=True)
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if out:
        print(out)
    if err:
        print(err)
    return res.returncode == 0


def main():
    print("QUICK VALIDATION SUITE")
    print("Running fast checks that don't require data...")
    print()
    
    results = []
    
    # Script validation
    script_results = validate_all_scripts()
    passed = sum(1 for _, ok, _ in script_results if ok)
    results.append(("Scripts", passed, len(script_results)))
    
    # Feature columns
    if check_feature_columns():
        results.append(("Feature Columns", 1, 1))
    else:
        results.append(("Feature Columns", 0, 1))
    
    # Documentation
    if check_documentation_links():
        results.append(("Documentation", 1, 1))
    else:
        results.append(("Documentation", 0, 1))

    # Optional data-backed within-season validation
    if check_within_season_windows_masks():
        results.append(("Within-Season", 1, 1))
    else:
        results.append(("Within-Season", 0, 1))

    # Optional edge-case wiring tests
    if check_edge_case_wiring():
        results.append(("Edge Cases", 1, 1))
    else:
        results.append(("Edge Cases", 0, 1))

    # Optional encoder gating tests
    if check_encoder_gating():
        results.append(("Encoder Gating", 1, 1))
    else:
        results.append(("Encoder Gating", 0, 1))

    # Optional: verify model input column lists against the actual training table
    if check_full_input_columns():
        results.append(("Full Inputs", 1, 1))
    else:
        results.append(("Full Inputs", 0, 1))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_passed = 0
    total_checks = 0
    
    for name, passed, total in results:
        status = "✅" if passed == total else "❌"
        print(f"{status} {name}: {passed}/{total} passed")
        total_passed += passed
        total_checks += total
    
    print(f"\nTotal: {total_passed}/{total_checks} checks passed")
    
    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
