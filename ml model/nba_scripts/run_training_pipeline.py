#!/usr/bin/env python3
"""
Training Pipeline Runner
========================
Orchestrates the full training pipeline from raw data to trained model.

Usage:
    python run_training_pipeline.py --check     # Check prerequisites
    python run_training_pipeline.py --gate      # Run pre-train QA gate
    python run_training_pipeline.py --build     # Build training table only
    python run_training_pipeline.py --train     # Train models only
    python run_training_pipeline.py --all       # Full pipeline

Pipeline Steps:
1. Check prerequisites (data files, packages)
2. Build unified training table (build_unified_training_table.py)
3. Train XGBoost baseline (train_baseline_xgboost.py)
"""

import sys
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent

# Required data files
REQUIRED_FILES = {
    'crosswalk': BASE_DIR / "data/warehouse_v2/dim_player_nba_college_crosswalk.parquet",
    'peak_rapm': BASE_DIR / "data/warehouse_v2/fact_player_peak_rapm.parquet",
    'year1_epm': BASE_DIR / "data/warehouse_v2/fact_player_year1_epm.parquet",
    'college_features': BASE_DIR / "data/college_feature_store/college_features_v1.parquet",
    'career_features': BASE_DIR / "data/college_feature_store/prospect_career_v1.parquet",
}

# Optional but recommended
OPTIONAL_FILES = {
    'gaps': BASE_DIR / "data/warehouse_v2/fact_player_nba_college_gaps.parquet",
    'career_long': BASE_DIR / "data/college_feature_store/prospect_career_long_v1.parquet",
    'within_season_windows': BASE_DIR / "data/college_feature_store/within_season_windows_v1.parquet",
}

# Required packages
REQUIRED_PACKAGES = [
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('xgboost', 'xgboost'),
    ('sklearn', 'scikit-learn'),
    ('scipy', 'scipy'),
]


def check_packages() -> bool:
    """Check if required packages are installed."""
    logger.info("Checking required packages...")
    all_ok = True
    
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            logger.info(f"  ✅ {import_name}")
        except ImportError:
            logger.error(f"  ❌ {import_name} - Install with: pip install {pip_name}")
            all_ok = False
    
    return all_ok


def check_data_files() -> dict:
    """Check if required data files exist."""
    logger.info("\nChecking required data files...")
    status = {'required': {}, 'optional': {}}
    
    for name, path in REQUIRED_FILES.items():
        exists = path.exists()
        status['required'][name] = exists
        if exists:
            logger.info(f"  ✅ {name}: {path}")
        else:
            logger.error(f"  ❌ {name}: {path}")
    
    logger.info("\nChecking optional data files...")
    for name, path in OPTIONAL_FILES.items():
        exists = path.exists()
        status['optional'][name] = exists
        if exists:
            logger.info(f"  ✅ {name}: {path}")
        else:
            logger.warning(f"  ⚠️ {name}: {path} (optional)")
    
    return status


def check_prerequisites() -> bool:
    """Run all prerequisite checks."""
    logger.info("=" * 60)
    logger.info("PREREQUISITE CHECK")
    logger.info("=" * 60)
    
    packages_ok = check_packages()
    data_status = check_data_files()
    
    required_ok = all(data_status['required'].values())
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Packages: {'✅ All OK' if packages_ok else '❌ Missing packages'}")
    logger.info(f"Required data: {'✅ All OK' if required_ok else '❌ Missing files'}")
    
    missing_required = [k for k, v in data_status['required'].items() if not v]
    if missing_required:
        logger.info(f"\n⚠️  Missing required files: {missing_required}")
        logger.info("\nTo generate missing data, run these scripts in order:")
        logger.info("  1. python -m cbd_pbp.cli ingest-season --season 2025")
        logger.info("  2. python college_scripts/build_college_feature_store_v1.py")
        logger.info("  3. python college_scripts/build_prospect_career_store_v2.py")
        logger.info("  4. python nba_scripts/build_nba_college_crosswalk.py")
        logger.info("  5. python nba_scripts/build_fact_nba_college_gaps.py")
    
    return packages_ok and required_ok


def build_training_table() -> bool:
    """Build the unified training table."""
    logger.info("=" * 60)
    logger.info("BUILDING UNIFIED TRAINING TABLE")
    logger.info("=" * 60)
    
    try:
        from build_unified_training_table import build_unified_training_table, save_training_table
        
        df = build_unified_training_table()
        if df.empty:
            logger.error("Failed to build training table!")
            return False
        
        save_training_table(df)
        logger.info(f"✅ Training table built: {len(df):,} rows")

        # Optional: build trajectory stub if long career file exists
        career_long_path = OPTIONAL_FILES['career_long']
        if career_long_path.exists():
            try:
                from build_trajectory_stub import build_trajectory_stub, save_trajectory_stub
                traj_df = build_trajectory_stub()
                save_trajectory_stub(traj_df)
            except Exception as e:
                logger.warning(f"Trajectory stub build failed: {e}")
        return True
        
    except Exception as e:
        logger.error(f"Error building training table: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_models() -> bool:
    """Train XGBoost baseline models."""
    logger.info("=" * 60)
    logger.info("TRAINING XGBOOST BASELINE MODELS")
    logger.info("=" * 60)
    
    try:
        from train_baseline_xgboost import main as train_main
        train_main()
        logger.info("✅ Training complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pretrain_gate(fail_on_gate: bool = True) -> bool:
    """Run pre-train readiness QA gate."""
    logger.info("=" * 60)
    logger.info("RUNNING PRE-TRAIN READINESS GATE")
    logger.info("=" * 60)

    try:
        from run_nba_pretrain_gate import build_gate_report, write_gate_report
        report = build_gate_report()
        write_gate_report(report)
        passed = bool(report.get("quality_gate", {}).get("passed", False))
        logger.info("Pre-train quality gate passed: %s", passed)
        if fail_on_gate and not passed:
            logger.error("Pre-train gate failed. See data/audit/nba_pretrain_gate.json")
            return False
        return True
    except Exception as e:
        logger.error(f"Error running pre-train gate: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline() -> bool:
    """Run the complete pipeline."""
    logger.info("=" * 60)
    logger.info("RUNNING FULL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        logger.error("\n❌ Prerequisites not met. Fix issues above before proceeding.")
        return False
    
    # Step 2: Build training table
    if not build_training_table():
        logger.error("\n❌ Failed to build training table.")
        return False

    # Step 3: Run pre-train gate
    if not run_pretrain_gate(fail_on_gate=True):
        logger.error("\n❌ Pre-train readiness gate failed.")
        return False
    
    # Step 4: Train models
    if not train_models():
        logger.error("\n❌ Failed to train models.")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline Runner")
    parser.add_argument('--check', action='store_true', help='Check prerequisites only')
    parser.add_argument('--gate', action='store_true', help='Run pre-train readiness gate only')
    parser.add_argument('--build', action='store_true', help='Build training table only')
    parser.add_argument('--train', action='store_true', help='Train models only')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    
    args = parser.parse_args()
    
    # Default to --check if no args
    if not any([args.check, args.gate, args.build, args.train, args.all]):
        args.check = True
    
    if args.check:
        check_prerequisites()
    
    if args.build:
        build_training_table()

    if args.gate:
        run_pretrain_gate(fail_on_gate=False)
    
    if args.train:
        train_models()
    
    if args.all:
        run_full_pipeline()


if __name__ == "__main__":
    main()
