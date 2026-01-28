#!/usr/bin/env python3
"""
Master Orchestrator: Runs Metadata first, then History.
Includes API call counter for safety monitoring.
"""
import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Quota tracking file
QUOTA_FILE = "api_call_count.txt"
QUOTA_LIMIT = 120000  # Stop at 120k to leave buffer

def get_call_count():
    if os.path.exists(QUOTA_FILE):
        with open(QUOTA_FILE) as f:
            return int(f.read().strip())
    return 0

def main():
    print("=" * 70)
    print("MASTER INGEST ORCHESTRATOR")
    print(f"Current API Call Count: {get_call_count()}")
    print(f"Safety Limit: {QUOTA_LIMIT}")
    print("=" * 70)
    
    # Step 1: Metadata (cheap, critical)
    print("\n[1/2] Running Metadata Ingest (Bio/Recruiting)...")
    result = subprocess.run(
        [sys.executable, "ingest_metadata.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print("WARNING: Metadata ingest had errors, but continuing...")
    
    # Check quota
    if get_call_count() >= QUOTA_LIMIT:
        print(f"STOPPING: Call count {get_call_count()} >= limit {QUOTA_LIMIT}")
        return
    
    # Step 2: History (expensive)
    print("\n[2/2] Running History Ingest (PBP 2020-2005)...")
    result = subprocess.run(
        [sys.executable, "ingest_history.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    print("\n" + "=" * 70)
    print(f"ORCHESTRATOR COMPLETE. Total API Calls: {get_call_count()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
