#!/usr/bin/env python3
"""
Orchestration Script for Historical Data Pipeline
Runs cleaning, backfill, RApM calculation, and reporting.
"""
import subprocess
import sys
import os
import time

# Define steps
STEPS = [
    {
        "name": "Cleaning Historical PBP",
        "script": "college_scripts/utils/clean_historical_pbp_v2.py",
        "args": []
    },
    {
        "name": "Deriving Minutes/Turnovers (Backfill)",
        "script": "college_scripts/derive_minutes_from_historical_pbp.py",
        "args": ["--all"]
    },
    {
        "name": "Calculating Historical RApM",
        "script": "college_scripts/calculate_historical_rapm.py",
        "args": []
    },
    {
        "name": "Exporting RApM Rankings",
        "script": "college_scripts/export_rapm_rankings.py",
        "args": []
    }
]

def run_step(step):
    print(f"\n{'='*60}")
    print(f"Running Step: {step['name']}")
    print(f"Script: {step['script']}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    cmd = [sys.executable, step['script']] + step['args']
    
    try:
        # Run the command and stream output
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) # Run from ml model/ directory
        )
        escaped_time = time.time() - start_time
        print(f"\n✅ Step '{step['name']}' completed successfully in {escaped_time:.2f}s.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step '{step['name']}' failed with error code {e.returncode}.")
        return False
    except Exception as e:
        print(f"\n❌ Step '{step['name']}' failed with exception: {e}")
        return False

def main():
    print("🚀 Starting Historical Data Pipeline...")
    start_total = time.time()
    
    for step in STEPS:
        success = run_step(step)
        if not success:
            print("\n⛔ Pipeline stopped due to error.")
            sys.exit(1)
            
    total_time = time.time() - start_total
    print(f"\n✨ All steps completed successfully in {total_time:.2f}s!")

if __name__ == "__main__":
    main()
