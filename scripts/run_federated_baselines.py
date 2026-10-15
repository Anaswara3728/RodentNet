import os
import sys
import time
import pandas as pd
from pathlib import Path

backend_path = str(Path(__file__).parent.parent / "backend")
sys.path.append(backend_path)
os.environ["PYTHONPATH"] = backend_path + os.pathsep + os.environ.get("PYTHONPATH", "")

from app.fl.simulation import run_simulation

def main():
    print("="*60)
    print("Running Federated Baselines (Strict Leak-Free Split)")
    print("="*60)
    
    tsv_path = "data/vdjdb.tsv"
    
    if not os.path.exists(tsv_path):
        print(f"Error: {tsv_path} not found. Please download VDJdb data first.")
        return

    results = []

    print("\n--- Running FedPer + FastKAN (Ours) ---")
    status = run_simulation(rounds=15, clients=3, task="epitope", vdjdb_path=tsv_path, fake_data=False)
    print(f"Status: {status.state}")
    
    # Normally we would extract metrics from status or flower history.
    # Since Flower simulation prints to stdout, the user will see it, 
    # but we can capture it by parsing or we just rely on the logs.
    print(f"Completed in {status.metrics.get('duration_seconds', 0)} seconds.")

if __name__ == "__main__":
    main()
