"""Rigorous Baseline Benchmark on Strict Unseen-TCR Split
=========================================================
Evaluates:
1. Centralized (Upper Bound)
2. Local-Only (Lower Bound)
3. FedAvg + FastKAN (Standard FL)
4. FedPer + FastKAN (Ours: Personalized FL)
Under identical random seeds (seed=42) on strict leak-free TCR splits.
"""

import sys
import os
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.fl.epitope_model import SiameseEpitopeKANNet, train_one_epoch, evaluate_model, info_nce_loss
from app.fl.vdjdb_data import VDJdbDataset, EpisodicBatchSampler


def get_leak_free_data(tsv_path="data/vdjdb.tsv", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = VDJdbDataset(tsv_path=tsv_path, synthetic=False)
    unique_tcrs = dataset.data['CDR3'].unique()
    np.random.shuffle(unique_tcrs)

    # 80/20 train/test split on TCRs (zero overlap)
    split_idx = int(0.8 * len(unique_tcrs))
    train_tcrs = set(unique_tcrs[:split_idx])
    test_tcrs = set(unique_tcrs[split_idx:])

    train_indices = [i for i in range(len(dataset)) if dataset.data.iloc[i]['CDR3'] in train_tcrs]
    test_indices = [i for i in range(len(dataset)) if dataset.data.iloc[i]['CDR3'] in test_tcrs]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Non-IID Hospital Partitions (3 simulated hospitals)
    hospital_subsets = []
    chunk_size = len(train_indices) // 3
    for h in range(3):
        h_idx = train_indices[h * chunk_size : (h + 1) * chunk_size]
        hospital_subsets.append(Subset(dataset, h_idx))

    return dataset, train_subset, test_subset, hospital_subsets


def run_baselines(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, train_subset, test_subset, hospital_subsets = get_leak_free_data()
    all_epi_tokens = torch.stack([dataset.epi_tokenizer.encode(e) for e in dataset.unique_epitopes]).to(device)

    # Loaders
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)

    results = {}

    # -------------------------------------------------------------
    # 1. Centralized Baseline (Upper Bound)
    # -------------------------------------------------------------
    print("\n[1/4] Running Centralized Baseline (Upper Bound)...")
    cent_model = SiameseEpitopeKANNet().to(device)
    cent_sampler = EpisodicBatchSampler(train_subset, n_ways=16, k_shots=16)
    cent_loader = DataLoader(train_subset, batch_sampler=cent_sampler)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(cent_model, cent_loader, device=device, lr=1e-3)
        print(f"  Centralized Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    _, cent_metrics = evaluate_model(cent_model, test_loader, all_epi_tokens, device=device)
    results["Centralized (Upper Bound)"] = {
        "AUROC": round(cent_metrics["auroc"], 4),
        "Top-1 Acc (%)": round(cent_metrics["accuracy"] * 100, 2),
        "MRR": round(cent_metrics["mrr"], 4),
        "Setting": "Centralized",
    }
    print(f"  Centralized Final -> AUROC: {cent_metrics['auroc']:.4f}, Acc: {cent_metrics['accuracy']*100:.2f}%, MRR: {cent_metrics['mrr']:.4f}")

    # -------------------------------------------------------------
    # 2. Local-Only Baseline (Lower Bound: Hospital 1 data only)
    # -------------------------------------------------------------
    print("\n[2/4] Running Local-Only Baseline (Lower Bound)...")
    local_model = SiameseEpitopeKANNet().to(device)
    local_sampler = EpisodicBatchSampler(hospital_subsets[0], n_ways=16, k_shots=16)
    local_loader = DataLoader(hospital_subsets[0], batch_sampler=local_sampler)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(local_model, local_loader, device=device, lr=1e-3)
        print(f"  Local Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    _, local_metrics = evaluate_model(local_model, test_loader, all_epi_tokens, device=device)
    results["Local-Only (Hospital 1)"] = {
        "AUROC": round(local_metrics["auroc"], 4),
        "Top-1 Acc (%)": round(local_metrics["accuracy"] * 100, 2),
        "MRR": round(local_metrics["mrr"], 4),
        "Setting": "Isolated Island",
    }
    print(f"  Local Final -> AUROC: {local_metrics['auroc']:.4f}, Acc: {local_metrics['accuracy']*100:.2f}%, MRR: {local_metrics['mrr']:.4f}")

    # -------------------------------------------------------------
    # 3. FedAvg + FastKAN (No personal head, standard FL)
    # -------------------------------------------------------------
    print("\n[3/4] Running FedAvg + FastKAN (Standard FL)...")
    fedavg_model = SiameseEpitopeKANNet().to(device)
    h_loaders = [
        DataLoader(h_sub, batch_sampler=EpisodicBatchSampler(h_sub, n_ways=16, k_shots=16))
        for h_sub in hospital_subsets
    ]

    for rnd in range(1, epochs + 1):
        # Local training per hospital
        client_models = [copy.deepcopy(fedavg_model) for _ in range(3)]
        for h_idx in range(3):
            train_one_epoch(client_models[h_idx], h_loaders[h_idx], device=device, lr=1e-3)

        # FedAvg: average all weights across clients
        avg_dict = fedavg_model.state_dict()
        for key in avg_dict:
            avg_dict[key] = torch.stack([client_models[h].state_dict()[key].float() for h in range(3)]).mean(dim=0)
        fedavg_model.load_state_dict(avg_dict)
        print(f"  FedAvg Round {rnd}/{epochs} aggregated")

    _, fedavg_metrics = evaluate_model(fedavg_model, test_loader, all_epi_tokens, device=device)
    results["FedAvg + FastKAN"] = {
        "AUROC": round(fedavg_metrics["auroc"], 4),
        "Top-1 Acc (%)": round(fedavg_metrics["accuracy"] * 100, 2),
        "MRR": round(fedavg_metrics["mrr"], 4),
        "Setting": "Federated (Standard)",
    }
    print(f"  FedAvg Final -> AUROC: {fedavg_metrics['auroc']:.4f}, Acc: {fedavg_metrics['accuracy']*100:.2f}%, MRR: {fedavg_metrics['mrr']:.4f}")

    # -------------------------------------------------------------
    # 4. FedPer + FastKAN (Ours: Personalized FL)
    # -------------------------------------------------------------
    print("\n[4/4] Running FedPer + FastKAN (Ours: Personalized FL)...")
    fedper_global = SiameseEpitopeKANNet().to(device)
    fedper_clients = [copy.deepcopy(fedper_global) for _ in range(3)]

    for rnd in range(1, epochs + 1):
        # 1. Each client updates shared parameters from global
        for h_idx in range(3):
            for name, param in fedper_clients[h_idx].named_parameters():
                if name.startswith(("features.", "projection.")):
                    param.data.copy_(fedper_global.state_dict()[name])

        # 2. Local training (personal_head stays local)
        for h_idx in range(3):
            train_one_epoch(fedper_clients[h_idx], h_loaders[h_idx], device=device, lr=1e-3)

        # 3. Server averages ONLY shared parameters (FedPer split)
        global_dict = fedper_global.state_dict()
        for name in global_dict:
            if name.startswith(("features.", "projection.")):
                global_dict[name] = torch.stack([fedper_clients[h].state_dict()[name].float() for h in range(3)]).mean(dim=0)
        fedper_global.load_state_dict(global_dict)
        print(f"  FedPer Round {rnd}/{epochs} aggregated (shared encoder only)")

    # Evaluate best personalized client
    _, fedper_metrics = evaluate_model(fedper_clients[0], test_loader, all_epi_tokens, device=device)
    results["FedPer + FastKAN (Ours)"] = {
        "AUROC": round(fedper_metrics["auroc"], 4),
        "Top-1 Acc (%)": round(fedper_metrics["accuracy"] * 100, 2),
        "MRR": round(fedper_metrics["mrr"], 4),
        "Setting": "Federated (Personalized)",
    }
    print(f"  FedPer Final -> AUROC: {fedper_metrics['auroc']:.4f}, Acc: {fedper_metrics['accuracy']*100:.2f}%, MRR: {fedper_metrics['mrr']:.4f}")

    # Summary Table
    print("\n" + "=" * 75)
    print("STRICT UNSEEN-TCR GENERALIZATION BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Method':<28} | {'Setting':<24} | {'AUROC':<7} | {'Top-1 Acc':<9} | {'MRR':<7}")
    print("-" * 75)
    for method, m in results.items():
        print(f"{method:<28} | {m['Setting']:<24} | {m['AUROC']:<7} | {m['Top-1 Acc (%)']:<8}% | {m['MRR']:<7}")
    print("=" * 75)

    return results


if __name__ == "__main__":
    run_baselines(epochs=5)
