"""
Architecture Ablation Script
==============================
Compares different architecture combinations on real VDJdb data
to prove each component of RodentNet adds value.

Experiments:
  1. FedPer + FastKAN  (Our full system — baseline)
  2. FedPer + MLP      (Replace KAN with standard MLP — proves KAN helps)
  3. No Personal Head  (FedAvg-style, average everything — proves FedPer helps)

Usage:
    python scripts/benchmark_ablation.py --tsv_path data/vdjdb.tsv --epochs 30
"""

import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.fl.epitope_model import (
    SiameseEpitopeKANNet,
    ResidueMotifEncoder,
    FastKAN,
    PersonalCalibration,
    info_nce_loss,
    train_one_epoch,
    evaluate_model,
)
from app.fl.vdjdb_data import load_vdjdb_partition


# ============================================================================
# Variant 1: MLP Head (replaces FastKAN with standard Linear layers)
# ============================================================================
class SiameseEpitopeMLP(nn.Module):
    """Same architecture as SiameseEpitopeKANNet but with a standard MLP
    instead of FastKAN. Used for ablation to prove KAN adds value."""

    def __init__(self, vocab_size=25, embed_dim=64, conv_channels=128, proj_dim=64):
        super().__init__()

        # SHARED — identical encoders
        self.features = nn.ModuleDict({
            'tcr': ResidueMotifEncoder(vocab_size, embed_dim, conv_channels),
            'epitope': ResidueMotifEncoder(vocab_size, embed_dim, conv_channels),
        })

        self.projection = nn.ModuleDict({
            'tcr': nn.Sequential(
                nn.Linear(conv_channels, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            ),
            'epitope': nn.Sequential(
                nn.Linear(conv_channels, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            ),
            # Standard MLP instead of FastKAN
            'kan': nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, 1),
            ),
        })

        # LOCAL
        self.personal_head = PersonalCalibration()

    def encode_tcr(self, tokens):
        feat = self.features['tcr'](tokens)
        proj = self.projection['tcr'](feat)
        return F.normalize(proj, p=2, dim=1)

    def encode_epitope(self, tokens):
        feat = self.features['epitope'](tokens)
        proj = self.projection['epitope'](feat)
        return F.normalize(proj, p=2, dim=1)

    def score(self, tcr_emb, epi_emb):
        x = torch.cat([tcr_emb, epi_emb], dim=-1)
        raw_score = self.projection['kan'](x).squeeze(-1)
        return self.personal_head(raw_score)

    def pairwise_logits(self, tcr_batch, epi_batch):
        B_tcr = tcr_batch.size(0)
        B_epi = epi_batch.size(0)
        tcr_exp = tcr_batch.unsqueeze(1).expand(B_tcr, B_epi, -1).reshape(-1, tcr_batch.size(1))
        epi_exp = epi_batch.unsqueeze(0).expand(B_tcr, B_epi, -1).reshape(-1, epi_batch.size(1))
        scores = self.score(tcr_exp, epi_exp)
        return scores.view(B_tcr, B_epi)

    def forward(self, tcr_tokens, epitope_tokens):
        tcr_emb = self.encode_tcr(tcr_tokens)
        epi_emb = self.encode_epitope(epitope_tokens)
        return self.score(tcr_emb, epi_emb)


# ============================================================================
# Variant 2: No Personal Head (FedAvg-style — everything averaged)
# ============================================================================
class SiameseEpitopeNoPersonal(nn.Module):
    """Same as SiameseEpitopeKANNet but the temperature head is removed.
    Simulates FedAvg where everything is shared. Proves FedPer adds value."""

    def __init__(self, vocab_size=25, embed_dim=64, conv_channels=128, proj_dim=64):
        super().__init__()

        self.features = nn.ModuleDict({
            'tcr': ResidueMotifEncoder(vocab_size, embed_dim, conv_channels),
            'epitope': ResidueMotifEncoder(vocab_size, embed_dim, conv_channels),
        })

        self.projection = nn.ModuleDict({
            'tcr': nn.Sequential(
                nn.Linear(conv_channels, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            ),
            'epitope': nn.Sequential(
                nn.Linear(conv_channels, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            ),
            'kan': FastKAN(proj_dim * 2, proj_dim, 1),
        })

        # NO personal_head — raw scores used directly (FedAvg style)

    def encode_tcr(self, tokens):
        feat = self.features['tcr'](tokens)
        proj = self.projection['tcr'](feat)
        return F.normalize(proj, p=2, dim=1)

    def encode_epitope(self, tokens):
        feat = self.features['epitope'](tokens)
        proj = self.projection['epitope'](feat)
        return F.normalize(proj, p=2, dim=1)

    def score(self, tcr_emb, epi_emb):
        x = torch.cat([tcr_emb, epi_emb], dim=-1)
        return self.projection['kan'](x).squeeze(-1)

    def pairwise_logits(self, tcr_batch, epi_batch):
        B_tcr = tcr_batch.size(0)
        B_epi = epi_batch.size(0)
        tcr_exp = tcr_batch.unsqueeze(1).expand(B_tcr, B_epi, -1).reshape(-1, tcr_batch.size(1))
        epi_exp = epi_batch.unsqueeze(0).expand(B_tcr, B_epi, -1).reshape(-1, epi_batch.size(1))
        scores = self.score(tcr_exp, epi_exp)
        return scores.view(B_tcr, B_epi)

    def forward(self, tcr_tokens, epitope_tokens):
        tcr_emb = self.encode_tcr(tcr_tokens)
        epi_emb = self.encode_epitope(epitope_tokens)
        return self.score(tcr_emb, epi_emb)


# ============================================================================
# Training runner (works for any model variant)
# ============================================================================
def run_single_experiment(model, train_loader, test_loader, catalogue_tokens,
                          device, epochs, label):
    """Train a model variant and return per-epoch metrics."""
    model = model.to(device)
    results = {"Epoch": [], "Loss": [], "Accuracy": [], "AUROC": [], "MRR": []}

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, device=device, lr=5e-4)
        _, metrics = evaluate_model(model, test_loader, catalogue_tokens, device=device)

        results["Epoch"].append(epoch)
        results["Loss"].append(loss)
        results["Accuracy"].append(metrics["accuracy"])
        results["AUROC"].append(metrics["auroc"])
        results["MRR"].append(metrics["mrr"])

        print(f"  [{label}] Epoch {epoch}/{epochs} - "
              f"Loss: {loss:.4f}, Acc: {metrics['accuracy']:.4f}, "
              f"AUROC: {metrics['auroc']:.4f}, MRR: {metrics['mrr']:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Architecture ablation on real VDJdb.")
    parser.add_argument("--tsv_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading real VDJdb data...")
    train_loader, test_loader, catalogue_tokens = load_vdjdb_partition(
        num_clients=1, client_id=0, tsv_path=args.tsv_path, synthetic=False,
    )
    n_epi = catalogue_tokens.shape[0]
    print(f"Loaded: {len(test_loader.dataset)} test examples, {n_epi} candidate epitopes\n")

    all_results = {}

    # --- Experiment 1: Our full system (FedPer + FastKAN) ---
    print("=" * 60)
    print("EXPERIMENT 1: FedPer + FastKAN (Ours)")
    print("=" * 60)
    model1 = SiameseEpitopeKANNet()
    all_results["FedPer+FastKAN (Ours)"] = run_single_experiment(
        model1, train_loader, test_loader, catalogue_tokens,
        device, args.epochs, "FedPer+FastKAN",
    )

    # --- Experiment 2: FedPer + MLP (KAN removed) ---
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: FedPer + MLP (No KAN)")
    print("=" * 60)
    model2 = SiameseEpitopeMLP()
    all_results["FedPer+MLP"] = run_single_experiment(
        model2, train_loader, test_loader, catalogue_tokens,
        device, args.epochs, "FedPer+MLP",
    )

    # --- Experiment 3: No Personal Head (FedAvg-style) ---
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: FedAvg + FastKAN (No Personal Head)")
    print("=" * 60)
    model3 = SiameseEpitopeNoPersonal()
    all_results["FedAvg+FastKAN"] = run_single_experiment(
        model3, train_loader, test_loader, catalogue_tokens,
        device, args.epochs, "FedAvg+FastKAN",
    )

    # --- Save all CSVs ---
    os.makedirs("results", exist_ok=True)
    for name, df in all_results.items():
        safe_name = name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
        df.to_csv(f"results/ablation_{safe_name}.csv", index=False)

    # --- Plot comparison ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"FedPer+FastKAN (Ours)": "tab:blue", "FedPer+MLP": "tab:orange", "FedAvg+FastKAN": "tab:red"}
    markers = {"FedPer+FastKAN (Ours)": "o", "FedPer+MLP": "s", "FedAvg+FastKAN": "^"}

    for name, df in all_results.items():
        c, m = colors[name], markers[name]
        axes[0].plot(df["Epoch"], df["Loss"], marker=m, color=c, label=name, linewidth=2, markersize=4)
        axes[1].plot(df["Epoch"], df["AUROC"], marker=m, color=c, label=name, linewidth=2, markersize=4)
        axes[2].plot(df["Epoch"], df["MRR"], marker=m, color=c, label=name, linewidth=2, markersize=4)

    axes[0].set_title("Training Loss (InfoNCE)", fontsize=14)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

    axes[1].set_title("Macro AUROC", fontsize=14)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUROC")
    axes[1].set_ylim(0.4, 1.0)

    axes[2].set_title("Mean Reciprocal Rank (MRR)", fontsize=14)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("MRR")

    for ax in axes:
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("results/ablation_comparison.png", dpi=300)
    print(f"\nSaved: results/ablation_comparison.png")

    # --- Print final summary ---
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY (Final Epoch)")
    print("=" * 70)
    print(f"{'Architecture':<28} {'Loss':>8} {'Acc':>8} {'AUROC':>8} {'MRR':>8}")
    print("-" * 70)
    for name, df in all_results.items():
        last = df.iloc[-1]
        print(f"{name:<28} {last['Loss']:>8.4f} {last['Accuracy']:>8.4f} "
              f"{last['AUROC']:>8.4f} {last['MRR']:>8.4f}")


if __name__ == "__main__":
    main()
