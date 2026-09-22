import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.fl.epitope_model import SiameseEpitopeKANNet, train_one_epoch, evaluate_model
from app.fl.vdjdb_data import load_vdjdb_partition

def run_experiment():
    print("Generating synthetic data and initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simulate a single client for plotting local training curves
    # Note: synthetic=True uses the planted motif rule for testing
    train_loader, test_loader, catalogue_tokens = load_vdjdb_partition(
        num_clients=1, client_id=0, synthetic=True
    )
    
    model = SiameseEpitopeKANNet().to(device)
    
    epochs = 15
    results = {
        "Epoch": [],
        "Loss": [],
        "Accuracy": [],
        "AUROC": [],
        "MRR": []
    }
    
    print(f"Training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, device=device, lr=2e-3)
        _, metrics = evaluate_model(model, test_loader, catalogue_tokens, device=device)
        
        results["Epoch"].append(epoch)
        results["Loss"].append(loss)
        results["Accuracy"].append(metrics["accuracy"])
        results["AUROC"].append(metrics["auroc"])
        results["MRR"].append(metrics["mrr"])
        
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Acc: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, MRR: {metrics['mrr']:.4f}")

    df = pd.DataFrame(results)
    
    # Save table to CSV
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/epitope_training_metrics.csv", index=False)
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    ax1.plot(df["Epoch"], df["Loss"], marker='o', color='tab:red', linewidth=2)
    ax1.set_title("Training Loss (InfoNCE)", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    
    # Metrics Curve
    ax2.plot(df["Epoch"], df["Accuracy"], marker='s', label='Top-1 Accuracy', color='tab:blue', linewidth=2)
    ax2.plot(df["Epoch"], df["AUROC"], marker='^', label='Macro AUROC', color='tab:orange', linewidth=2)
    ax2.plot(df["Epoch"], df["MRR"], marker='d', label='Mean Reciprocal Rank (MRR)', color='tab:green', linewidth=2)
    ax2.set_title("Evaluation Metrics", fontsize=14)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig("results/epitope_training_curves.png", dpi=300)
    print("Results saved to results/epitope_training_metrics.csv and results/epitope_training_curves.png")
    
if __name__ == "__main__":
    run_experiment()
