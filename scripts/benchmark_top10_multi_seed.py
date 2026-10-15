"""Top-10 Epitope Classification 5-Seed Benchmark
=================================================
Runs 5 independent random seeds (42, 43, 44, 45, 46) on the strict unseen-TCR split
with equal training budgets (10 epochs/rounds).
Evaluates:
- Random Guess (1/10 Chance)
- Majority Class Baseline (GILGFVFTL)
- Label-Shuffle Control (Negative Permutation Test)
- CDR3 1-NN Baseline (Levenshtein Edit Distance)
- 3-mer + Logistic Regression Baseline
- Centralized 1D-CNN + FastKAN
- Local-Only (Hospital 0)
- FedAvg + FastKAN
- FedAvg + Local Fine-Tuning
- FedPer + FastKAN (Global test evaluation & Local test evaluation)

Computes Mean +/- Standard Deviation across all 5 seeds.
Saves JSON summary to disk for complete reproducibility.
"""

import sys
import copy
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.fl.epitope_model import FastKAN

AA_VOCAB = {aa: idx + 1 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
MAX_CDR3_LEN = 30

def tokenize_cdr3(seq):
    tokens = [AA_VOCAB.get(c, 0) for c in seq[:MAX_CDR3_LEN]]
    if len(tokens) < MAX_CDR3_LEN:
        tokens.extend([0] * (MAX_CDR3_LEN - len(tokens)))
    return torch.tensor(tokens, dtype=torch.long)

def levenshtein_dist(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, subs))
        prev = curr
    return prev[-1]

class Top10TCRDataset(Dataset):
    def __init__(self, df, epi2idx):
        self.seqs = df['cdr3'].tolist()
        self.tokens = [tokenize_cdr3(s) for s in self.seqs]
        self.labels = torch.tensor([epi2idx[e] for e in df['antigen.epitope']], dtype=torch.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]

class Top10KANClassifier(nn.Module):
    """1D-CNN + FastKAN Feature Extractor with Linear Classification Head."""
    def __init__(self, num_classes=10, vocab_size=25, embed_dim=64, channels=128, kan_dim=64):
        super().__init__()
        # SHARED backbone
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.kan = FastKAN(channels, kan_dim, kan_dim)
        
        # PERSONAL head (local per hospital in FedPer)
        self.classifier = nn.Linear(kan_dim, num_classes)

    def extract_features(self, x):
        emb = self.embedding(x).transpose(1, 2)
        c1 = F.relu(self.bn1(self.conv1(emb)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        pooled = self.pool(c2).squeeze(-1)
        feat = self.kan(pooled)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)

def get_shared_params(model):
    shared = []
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            shared.append(param.detach().cpu().numpy())
    return shared

def set_shared_params(model, params):
    idx = 0
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            param.data.copy_(torch.from_numpy(params[idx]).to(param.device))
            idx += 1

def load_and_preprocess_top10(seed=42):
    raw_path = Path("data/vdjdb.tsv")
    df = pd.read_csv(raw_path, sep="\t", low_memory=False)
    df = df[df['species'] == 'HomoSapiens']
    df = df[df['gene'] == 'TRB']
    df = df[df['vdjdb.score'] >= 1]
    df = df.dropna(subset=['cdr3', 'antigen.epitope'])
    df = df[df['cdr3'].str.len() >= 5]
    
    top10_epis = df['antigen.epitope'].value_counts().nlargest(10).index.tolist()
    df = df[df['antigen.epitope'].isin(top10_epis)]
    
    cdr3_epis = df.groupby('cdr3')['antigen.epitope'].nunique()
    valid_cdr3s = cdr3_epis[cdr3_epis == 1].index
    df = df[df['cdr3'].isin(valid_cdr3s)].copy()
    
    epi2idx = {e: i for i, e in enumerate(top10_epis)}
    idx2epi = {i: e for e, i in epi2idx.items()}
    
    unique_cdr3s = df['cdr3'].unique()
    rng = np.random.default_rng(seed)
    unique_cdr3s = rng.permutation(unique_cdr3s)
    
    split_idx = int(0.8 * len(unique_cdr3s))
    train_cdr3s = set(unique_cdr3s[:split_idx])
    test_cdr3s = set(unique_cdr3s[split_idx:])
    
    train_df = df[df['cdr3'].isin(train_cdr3s)].copy().reset_index(drop=True)
    test_df = df[df['cdr3'].isin(test_cdr3s)].copy().reset_index(drop=True)
    
    hospital_train_dfs = [[] for _ in range(3)]
    hospital_test_dfs = [[] for _ in range(3)]
    
    for epi in top10_epis:
        sub_train = train_df[train_df['antigen.epitope'] == epi]
        if len(sub_train) > 0:
            props = rng.dirichlet([0.5, 0.5, 0.5])
            splits = [int(p * len(sub_train)) for p in np.cumsum(props)[:-1]]
            chunks = np.split(sub_train, splits)
            for h in range(3):
                hospital_train_dfs[h].append(chunks[h])
                
        sub_test = test_df[test_df['antigen.epitope'] == epi]
        if len(sub_test) > 0:
            props_t = rng.dirichlet([0.5, 0.5, 0.5])
            splits_t = [int(p * len(sub_test)) for p in np.cumsum(props_t)[:-1]]
            chunks_t = np.split(sub_test, splits_t)
            for h in range(3):
                hospital_test_dfs[h].append(chunks_t[h])
                
    hosp_train_dfs = [pd.concat(dfs).reset_index(drop=True) for dfs in hospital_train_dfs]
    hosp_test_dfs = [pd.concat(dfs).reset_index(drop=True) for dfs in hospital_test_dfs]
    
    return top10_epis, epi2idx, idx2epi, train_df, test_df, hosp_train_dfs, hosp_test_dfs

def evaluate_classifier(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    raw_acc = float((y_pred == y_true).mean() * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    try:
        macro_auroc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        macro_auroc = 0.5000
        
    return {
        "raw_acc": raw_acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc
    }

def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

def run_seed_evaluation(seed, epochs=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    
    top10_epis, epi2idx, idx2epi, train_df, test_df, hosp_train_dfs, hosp_test_dfs = load_and_preprocess_top10(seed=seed)
    
    train_dataset = Top10TCRDataset(train_df, epi2idx)
    test_dataset = Top10TCRDataset(test_df, epi2idx)
    hosp_train_datasets = [Top10TCRDataset(hdf, epi2idx) for hdf in hosp_train_dfs]
    hosp_test_datasets = [Top10TCRDataset(hdf, epi2idx) for hdf in hosp_test_dfs]
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    hosp_train_loaders = [DataLoader(ds, batch_size=128, shuffle=True) for ds in hosp_train_datasets]
    hosp_test_loaders = [DataLoader(ds, batch_size=256, shuffle=False) for ds in hosp_test_datasets]
    
    seed_res = {}
    
    # 1. Random Guess
    seed_res["Random Guess (1/10)"] = {"bal_acc": 10.0, "raw_acc": 10.0, "macro_f1": 0.1000, "macro_auroc": 0.5000}
    
    # 2. Majority Class
    maj_label = Counter(train_dataset.labels.numpy()).most_common(1)[0][0]
    y_test = test_dataset.labels.numpy()
    maj_preds = np.full_like(y_test, maj_label)
    seed_res["Majority Class"] = {
        "bal_acc": float(balanced_accuracy_score(y_test, maj_preds) * 100.0),
        "raw_acc": float((maj_preds == y_test).mean() * 100.0),
        "macro_f1": float(f1_score(y_test, maj_preds, average='macro', zero_division=0)),
        "macro_auroc": 0.5000
    }
    
    # 3. Label-Shuffle Control
    shuffled_dataset = copy.deepcopy(train_dataset)
    perm = torch.randperm(len(shuffled_dataset.labels))
    shuffled_dataset.labels = shuffled_dataset.labels[perm]
    shuffled_loader = DataLoader(shuffled_dataset, batch_size=128, shuffle=True)
    shuf_model = Top10KANClassifier().to(device)
    shuf_opt = torch.optim.Adam(shuf_model.parameters(), lr=1e-3)
    for _ in range(4):
        train_epoch(shuf_model, shuffled_loader, shuf_opt, device)
    seed_res["Label-Shuffle Control"] = evaluate_classifier(shuf_model, test_loader, device)
    
    # 4. 1-NN Levenshtein Distance
    # Stratified subset of training for tractable exact distance calculation
    train_sub = train_df.groupby('antigen.epitope', group_keys=False).apply(lambda g: g.sample(min(len(g), 80), random_state=seed)).reset_index(drop=True)
    train_seqs = train_sub['cdr3'].tolist()
    train_lbls = [epi2idx[e] for e in train_sub['antigen.epitope']]
    test_sub = test_df.sample(min(len(test_df), 500), random_state=seed).reset_index(drop=True)
    test_seqs = test_sub['cdr3'].tolist()
    test_lbls = [epi2idx[e] for e in test_sub['antigen.epitope']]
    
    nn_preds = []
    for t_seq in test_seqs:
        dists = [levenshtein_dist(t_seq, tr_seq) for tr_seq in train_seqs]
        nn_preds.append(train_lbls[np.argmin(dists)])
    nn_preds = np.array(nn_preds)
    test_lbls = np.array(test_lbls)
    seed_res["CDR3 1-NN (Levenshtein)"] = {
        "bal_acc": float(balanced_accuracy_score(test_lbls, nn_preds) * 100.0),
        "raw_acc": float((nn_preds == test_lbls).mean() * 100.0),
        "macro_f1": float(f1_score(test_lbls, nn_preds, average='macro', zero_division=0)),
        "macro_auroc": None
    }
    
    # 5. 3-mer + Logistic Regression
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X_tr = vectorizer.fit_transform(train_df['cdr3'])
    y_tr = [epi2idx[e] for e in train_df['antigen.epitope']]
    X_te = vectorizer.transform(test_df['cdr3'])
    y_te = [epi2idx[e] for e in test_df['antigen.epitope']]
    
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_tr, y_tr)
    lr_preds = clf.predict(X_te)
    lr_probs = clf.predict_proba(X_te)
    try:
        lr_auc = float(roc_auc_score(y_te, lr_probs, multi_class='ovr', average='macro'))
    except Exception:
        lr_auc = 0.5000
    seed_res["3-mer + Logistic Regression"] = {
        "bal_acc": float(balanced_accuracy_score(y_te, lr_preds) * 100.0),
        "raw_acc": float((lr_preds == y_te).mean() * 100.0),
        "macro_f1": float(f1_score(y_te, lr_preds, average='macro', zero_division=0)),
        "macro_auroc": lr_auc
    }
    
    # 6. Centralized FastKAN
    cent_model = Top10KANClassifier().to(device)
    cent_opt = torch.optim.Adam(cent_model.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_epoch(cent_model, train_loader, cent_opt, device)
    seed_res["Centralized FastKAN"] = evaluate_classifier(cent_model, test_loader, device)
    
    # 7. Local-Only (Hospital 0)
    local_model = Top10KANClassifier().to(device)
    local_opt = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_epoch(local_model, hosp_train_loaders[0], local_opt, device)
    seed_res["Local-Only (Hospital 0)"] = evaluate_classifier(local_model, test_loader, device)
    
    # 8. FedAvg + FastKAN
    fedavg_global = Top10KANClassifier().to(device)
    for rnd in range(epochs):
        client_models = [copy.deepcopy(fedavg_global) for _ in range(3)]
        for h in range(3):
            opt = torch.optim.Adam(client_models[h].parameters(), lr=1e-3)
            train_epoch(client_models[h], hosp_train_loaders[h], opt, device)
        avg_dict = fedavg_global.state_dict()
        for k in avg_dict:
            avg_dict[k] = torch.stack([client_models[h].state_dict()[k].float() for h in range(3)]).mean(dim=0)
        fedavg_global.load_state_dict(avg_dict)
    seed_res["FedAvg + FastKAN"] = evaluate_classifier(fedavg_global, test_loader, device)
    
    # 9. FedAvg + Fine-Tuning
    ft_model = copy.deepcopy(fedavg_global)
    ft_opt = torch.optim.Adam(ft_model.parameters(), lr=5e-4)
    for _ in range(2):
        train_epoch(ft_model, hosp_train_loaders[0], ft_opt, device)
    seed_res["FedAvg + Fine-Tuning"] = evaluate_classifier(ft_model, test_loader, device)
    
    # 10. FedPer + FastKAN
    fedper_global = Top10KANClassifier().to(device)
    fedper_clients = [copy.deepcopy(fedper_global) for _ in range(3)]
    for rnd in range(epochs):
        global_shared = get_shared_params(fedper_global)
        for h in range(3):
            set_shared_params(fedper_clients[h], global_shared)
        for h in range(3):
            opt = torch.optim.Adam(fedper_clients[h].parameters(), lr=1e-3)
            train_epoch(fedper_clients[h], hosp_train_loaders[h], opt, device)
        client_shared = [get_shared_params(fedper_clients[h]) for h in range(3)]
        avg_shared = []
        for l_idx in range(len(client_shared[0])):
            avg_shared.append(np.mean([client_shared[h][l_idx] for h in range(3)], axis=0))
        set_shared_params(fedper_global, avg_shared)
        
    # FedPer evaluated on pooled test set (client 0)
    seed_res["FedPer (Global Test)"] = evaluate_classifier(fedper_clients[0], test_loader, device)
    
    # FedPer evaluated on local matched test sets (mean of each client on its own skewed test partition)
    local_test_accs = [evaluate_classifier(fedper_clients[h], hosp_test_loaders[h], device) for h in range(3)]
    seed_res["FedPer (Local Test Matched)"] = {
        "bal_acc": float(np.mean([r["bal_acc"] for r in local_test_accs])),
        "raw_acc": float(np.mean([r["raw_acc"] for r in local_test_accs])),
        "macro_f1": float(np.mean([r["macro_f1"] for r in local_test_accs])),
        "macro_auroc": float(np.mean([r["macro_auroc"] for r in local_test_accs]))
    }
    
    return seed_res

def run_all_seeds(seeds=(42, 43, 44, 45, 46), epochs=10):
    print("=" * 85, flush=True)
    print("STARTING 5-SEED RIGOROUS TOP-10 EPITOPE BENCHMARK RUN", flush=True)
    print(f"Seeds: {seeds} | Equal Budget: {epochs} Epochs | Device: CPU", flush=True)
    print("=" * 85, flush=True)
    
    all_seed_results = []
    
    for s_idx, s in enumerate(seeds, start=1):
        t0 = time.perf_counter()
        print(f"\n>>> Running Seed {s_idx}/{len(seeds)} (Seed={s})...", flush=True)
        res = run_seed_evaluation(seed=s, epochs=epochs)
        all_seed_results.append(res)
        elapsed = time.perf_counter() - t0
        print(f"    Completed Seed {s} in {elapsed:.1f}s", flush=True)
        print(f"    FedAvg Bal Acc: {res['FedAvg + FastKAN']['bal_acc']:.2f}% | FedPer Bal Acc: {res['FedPer (Global Test)']['bal_acc']:.2f}%", flush=True)
        
    # Aggregate Mean +/- Std Dev
    methods = list(all_seed_results[0].keys())
    summary = {}
    
    print("\n" + "=" * 90, flush=True)
    print("FINAL 5-SEED AGGREGATED STATISTICAL SUMMARY (Mean +/- Std Dev)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Method / Baseline':<30} | {'Balanced Acc (%)':<18} | {'Raw Acc (%)':<18} | {'Macro F1':<15} | {'Macro AUROC':<15}", flush=True)
    print("-" * 90, flush=True)
    
    for m in methods:
        bal_accs = [r[m]["bal_acc"] for r in all_seed_results]
        raw_accs = [r[m]["raw_acc"] for r in all_seed_results]
        f1s = [r[m]["macro_f1"] for r in all_seed_results]
        aucs = [r[m]["macro_auroc"] for r in all_seed_results if r[m]["macro_auroc"] is not None]
        
        b_mean, b_std = np.mean(bal_accs), np.std(bal_accs)
        r_mean, r_std = np.mean(raw_accs), np.std(raw_accs)
        f_mean, f_std = np.mean(f1s), np.std(f1s)
        auc_str = f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}" if len(aucs) > 0 else "--"
        
        summary[m] = {
            "bal_acc_mean": float(b_mean), "bal_acc_std": float(b_std),
            "raw_acc_mean": float(r_mean), "raw_acc_std": float(r_std),
            "macro_f1_mean": float(f_mean), "macro_f1_std": float(f_std),
            "macro_auroc_mean": float(np.mean(aucs)) if len(aucs) > 0 else None,
            "macro_auroc_std": float(np.std(aucs)) if len(aucs) > 0 else None,
        }
        
        b_str = f"{b_mean:.2f} +/- {b_std:.2f}%"
        r_str = f"{r_mean:.2f} +/- {r_std:.2f}%"
        f_str = f"{f_mean:.4f} +/- {f_std:.4f}"
        print(f"{m:<30} | {b_str:<18} | {r_str:<18} | {f_str:<15} | {auc_str:<15}", flush=True)
        
    print("=" * 90, flush=True)
    
    # Save to disk
    out_path = Path("results/top10_5seeds_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved verified multi-seed summary to: {out_path}", flush=True)

if __name__ == "__main__":
    run_all_seeds()
