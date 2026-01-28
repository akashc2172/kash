import time
import os
import subprocess
import sys

# PID of the running ingest_history.py
TARGET_PID = 52745

def is_running(pid):
    """Check if PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def main():
    print(f"Orchestrator: Monitoring PID {TARGET_PID} (ingest_history.py)...")
    
    # Wait loop
    while is_running(TARGET_PID):
        time.sleep(30) # Check every 30s
    
    print(f"Orchestrator: PID {TARGET_PID} finished. Starting ingest_metadata.py...")
    
    # Log file for metadata
    with open("logs/ingest_metadata.log", "w") as f:
        subprocess.run(
            [sys.executable, "ingest_metadata.py"],
            stdout=f,
            stderr=subprocess.STDOUT
        )
    
    print("Orchestrator: ingest_metadata.py complete!")

if __name__ == "__main__":
    main()
