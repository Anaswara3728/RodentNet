import os
import sys
import torch
from pathlib import Path
import pandas as pd

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from app.fl.epitope_model import SiameseEpitopeKANNet, train_one_epoch, evaluate_model
from app.fl.vdjdb_data import load_vdjdb_partition, VDJdbDataset, EpisodicBatchSampler
from torch.utils.data import DataLoader, Subset

def main():
    print("="*60)
    print("Running Centralized Baseline (Upper Bound)")
    print("="*60)
    
    tsv_path = "data/vdjdb.tsv"
    if not os.path.exists(tsv_path):
        print(f"Error: {tsv_path} not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load entire dataset
    dataset = VDJdbDataset(tsv_path=tsv_path, synthetic=False)
    unique_tcrs = dataset.data['CDR3'].unique()
    
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(unique_tcrs)
    split_idx = int(0.8 * len(unique_tcrs))
    
    train_tcrs = set(unique_tcrs[:split_idx])
    test_tcrs = set(unique_tcrs[split_idx:])
    
    train_idx = [i for i in range(len(dataset)) if dataset.data.iloc[i]['CDR3'] in train_tcrs]
    test_idx = [i for i in range(len(dataset)) if dataset.data.iloc[i]['CDR3'] in test_tcrs]
    
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    
    train_sampler = EpisodicBatchSampler(train_subset, n_ways=16, k_shots=16)
    train_loader = DataLoader(train_subset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    
    all_epi_tokens = torch.stack([dataset.epi_tokenizer.encode(epi) for epi in dataset.unique_epitopes]).to(device)

    model = SiameseEpitopeKANNet().to(device)
    epochs = 15
    
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}, Unique Epitopes: {len(dataset.unique_epitopes)}")
    
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, device=device, lr=5e-4)
        _, metrics = evaluate_model(model, test_loader, all_epi_tokens, device=device)
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Acc: {metrics['accuracy']:.4f} - AUROC: {metrics['auroc']:.4f} - MRR: {metrics['mrr']:.4f}")

if __name__ == "__main__":
    main()
