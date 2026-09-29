import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add backend to path (same convention as generate_presentation_results.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.fl.epitope_model import SiameseEpitopeKANNet, train_one_epoch, evaluate_model
from app.fl.vdjdb_data import load_vdjdb_partition


def run_experiment(tsv_path: str, epochs: int, out_suffix: str):
    print(f"Loading REAL VDJdb data from: {tsv_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # synthetic=False -> goes through _load_real() in vdjdb_data.py instead of
    # the planted-motif synthetic generator.
    train_loader, test_loader, catalogue_tokens = load_vdjdb_partition(
        num_clients=1, client_id=0, tsv_path=tsv_path, synthetic=False
    )

    # --- Sanity diagnostics: confirm real data actually loaded, not an
    # empty / mis-filtered result. If this prints a tiny number of
    # epitopes, stop and check your column names (see Step 2 in chat). ---
    n_test = len(test_loader.dataset)
    n_epitopes = catalogue_tokens.shape[0]
    print(f"Loaded dataset -> test examples: {n_test}, unique candidate epitopes: {n_epitopes}")
    if n_epitopes <= 5:
        print("WARNING: very few unique epitopes detected. This usually means the TSV's "
              "column names don't match what _load_real() expects (Species/CDR3/Epitope/"
              "vdjdb.score or Score), or the file path is wrong. Check before trusting results.")

    model = SiameseEpitopeKANNet().to(device)

    results = {"Epoch": [], "Loss": [], "Accuracy": [], "AUROC": [], "MRR": []}

    print(f"Training for {epochs} epochs on {device} against {n_epitopes} candidate epitopes...")
    print("Note: real VDJdb has far more records than the synthetic run (2,000 rows), so each "
          "epoch will contain many more batches and take noticeably longer. Consider a quick "
          "--epochs 1 dry run first to check timing before committing to a full run.")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, device=device, lr=2e-3)
        _, metrics = evaluate_model(model, test_loader, catalogue_tokens, device=device)

        results["Epoch"].append(epoch)
        results["Loss"].append(loss)
        results["Accuracy"].append(metrics["accuracy"])
        results["AUROC"].append(metrics["auroc"])
        results["MRR"].append(metrics["mrr"])

        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Acc: {metrics['accuracy']:.4f}, "
              f"AUROC: {metrics['auroc']:.4f}, MRR: {metrics['mrr']:.4f}")

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/epitope_training_metrics_{out_suffix}.csv"
    df.to_csv(csv_path, index=False)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["Epoch"], df["Loss"], marker='o', color='tab:red', linewidth=2)
    ax1.set_title("Training Loss (InfoNCE) --- REAL VDJdb data", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)

    ax2.plot(df["Epoch"], df["Accuracy"], marker='s', label='Top-1 Accuracy', color='tab:blue', linewidth=2)
    ax2.plot(df["Epoch"], df["AUROC"], marker='^', label='Macro AUROC', color='tab:orange', linewidth=2)
    ax2.plot(df["Epoch"], df["MRR"], marker='d', label='Mean Reciprocal Rank (MRR)', color='tab:green', linewidth=2)
    ax2.set_title(f"Evaluation Metrics --- REAL VDJdb ({n_epitopes} candidate epitopes)", fontsize=14)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    png_path = f"results/epitope_training_curves_{out_suffix}.png"
    plt.savefig(png_path, dpi=300)
    print(f"Saved: {csv_path} and {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SiameseEpitopeKANNet on REAL VDJdb data (not the synthetic planted-motif set)."
    )
    parser.add_argument("--tsv_path", type=str, required=True,
                         help="Path to the downloaded real VDJdb .tsv file")
    parser.add_argument("--epochs", type=int, default=15,
                         help="Number of local training epochs (default 15, matching the synthetic run)")
    parser.add_argument("--out_suffix", type=str, default="real",
                         help="Suffix for output files, e.g. results/epitope_training_curves_real.png")
    args = parser.parse_args()
    run_experiment(args.tsv_path, args.epochs, args.out_suffix)