"""FashionMNIST Multi-Seed Utility Preservation Benchmark
======================================================
Runs 5 independent random seeds (42, 43, 44, 45, 46) comparing
Plain FedPer vs Secured FedPer (Authenticated ML-KEM-768 + 40-bit MPC).
Computes mean +/- standard deviation across seeds for each round,
proving whether the secured aggregation accuracy delta is statistically
distinguishable from zero.
"""

import sys
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.crypto.mpc import MPCSecureAggregator
from app.crypto.shamir import VECTOR_FIELD

class FedPerFashionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def extract_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x.view(x.size(0), -1)

    def forward(self, x):
        feat = self.extract_features(x)
        x = F.relu(self.fc1(feat))
        return self.fc2(x)

def get_shared_params(model):
    shared = []
    for name, param in model.named_parameters():
        if name.startswith(("conv1.", "bn1.", "conv2.", "bn2.")):
            shared.append(param.detach().cpu().numpy())
    return shared

def set_shared_params(model, params):
    idx = 0
    for name, param in model.named_parameters():
        if name.startswith(("conv1.", "bn1.", "conv2.", "bn2.")):
            param.data.copy_(torch.from_numpy(params[idx]).to(param.device))
            idx += 1

def get_fmnist_data(seed, num_clients=3, samples_per_client=3000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_full = datasets.FashionMNIST(root='data/fmnist', train=True, download=False, transform=transform)
    test_full = datasets.FashionMNIST(root='data/fmnist', train=False, download=False, transform=transform)
    
    client_loaders = []
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_full))
    
    for c_idx in range(num_clients):
        c_indices = indices[c_idx * samples_per_client : (c_idx + 1) * samples_per_client]
        loader = DataLoader(Subset(train_full, c_indices), batch_size=64, shuffle=True)
        client_loaders.append(loader)
        
    test_loader = DataLoader(test_full, batch_size=256, shuffle=False)
    return client_loaders, test_loader

def train_client(model, loader, device, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)

def run_single_experiment(seed, secured=False, rounds=4, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    client_loaders, test_loader = get_fmnist_data(seed=seed)
    global_model = FedPerFashionNet().to(device)
    clients = [copy.deepcopy(global_model) for _ in range(3)]
    
    roster = ["hosp-0", "hosp-1", "hosp-2"]
    shared_secrets = {
        "hosp-0": {"hosp-1": b"sec01" * 6 + b"00", "hosp-2": b"sec02" * 6 + b"00"},
        "hosp-1": {"hosp-0": b"sec01" * 6 + b"00", "hosp-2": b"sec12" * 6 + b"00"},
        "hosp-2": {"hosp-0": b"sec02" * 6 + b"00", "hosp-1": b"sec12" * 6 + b"00"},
    }
    
    round_accs = []
    
    for r in range(1, rounds + 1):
        global_params = get_shared_params(global_model)
        for c in clients:
            set_shared_params(c, global_params)
            
        for i, c in enumerate(clients):
            train_client(c, client_loaders[i], device, epochs=1)
            
        client_updates = [get_shared_params(c) for c in clients]
        weights = [len(loader.dataset) for loader in client_loaders]
        total_w = sum(weights)
        
        if not secured:
            avg_params = []
            for layer_idx in range(len(client_updates[0])):
                layer_sum = sum(client_updates[i][layer_idx] * weights[i] for i in range(3))
                avg_params.append(layer_sum / total_w)
        else:
            masked_updates = []
            for i, cid in enumerate(roster):
                agg = MPCSecureAggregator(field=VECTOR_FIELD, scale=1_000_000, shared_secrets=shared_secrets[cid])
                masked = agg.mask_weighted_update(client_updates[i], num_examples=weights[i], client_id=cid, roster=roster, round_id=f"r-{r}")
                masked_updates.append(masked)
            
            server_agg = MPCSecureAggregator(field=VECTOR_FIELD, scale=1_000_000)
            avg_params = server_agg.aggregate_masked_updates(masked_updates, total_examples=total_w)
            
        set_shared_params(global_model, avg_params)
        
        client_test_accs = [evaluate(c, test_loader, device) for c in clients]
        mean_acc = float(np.mean(client_test_accs)) * 100.0
        round_accs.append(mean_acc)
        
    return round_accs

def run_multi_seed_benchmark(seeds=(42, 43, 44, 45, 46), rounds=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80, flush=True)
    print("FASHION-MNIST 5-SEED UTILITY PRESERVATION BENCHMARK (10,000 TEST SAMPLES)", flush=True)
    print(f"Seeds: {seeds} | Rounds: {rounds} | Device: {device}", flush=True)
    print("=" * 80, flush=True)
    
    all_plain = []
    all_secured = []
    
    for s in seeds:
        print(f"\n--- Running Seed {s} ---", flush=True)
        p_accs = run_single_experiment(seed=s, secured=False, rounds=rounds, device=device)
        s_accs = run_single_experiment(seed=s, secured=True, rounds=rounds, device=device)
        all_plain.append(p_accs)
        all_secured.append(s_accs)
        deltas = [abs(p - sc) for p, sc in zip(p_accs, s_accs)]
        print(f"  Seed {s} Final Round: Plain = {p_accs[-1]:.2f}%, Secured = {s_accs[-1]:.2f}%, Delta = {deltas[-1]:.4f}%", flush=True)
        
    plain_arr = np.array(all_plain)     # shape: (num_seeds, rounds)
    secured_arr = np.array(all_secured) # shape: (num_seeds, rounds)
    delta_arr = np.abs(plain_arr - secured_arr)
    
    print("\n" + "=" * 80, flush=True)
    print("MULTI-SEED STATISTICAL SUMMARY (Mean +/- Std Dev across 5 seeds)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Round':<8} | {'Plain FedPer Acc (%)':<24} | {'Secured FedPer Acc (%)':<24} | {'Mean Delta (%)':<15}", flush=True)
    print("-" * 80, flush=True)
    for r in range(rounds):
        p_mean = plain_arr[:, r].mean()
        p_std = plain_arr[:, r].std()
        s_mean = secured_arr[:, r].mean()
        s_std = secured_arr[:, r].std()
        d_mean = delta_arr[:, r].mean()
        d_std = delta_arr[:, r].std()
        print(f"{r+1:<8} | {p_mean:.2f} +/- {p_std:.2f}%{' ':6} | {s_mean:.2f} +/- {s_std:.2f}%{' ':6} | {d_mean:.4f} +/- {d_std:.4f}%", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_multi_seed_benchmark()
