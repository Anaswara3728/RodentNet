"""Top-10 Epitope Classification Benchmark on Strict Unseen-TCR Split
====================================================================
Rigorous evaluation addressing all reviewer recommendations:
1. Filter to HomoSapiens TRB chain only
2. Focus on Top-10 epitopes (abundant data per class)
3. Drop ambiguous CDR3s (binding multiple top-10 epitopes)
4. Strict CDR3 split (80% train, 20% test; zero sequence overlap)
5. Comprehensive Baselines:
   - Majority Class Baseline (predicts most frequent epitope)
   - CDR3 1-NN Baseline (Levenshtein edit distance)
   - 3-mer (k-mer) counts + Logistic Regression
   - Label-Shuffle Control (proves model falls to chance under permutation)
6. Federated Settings (Dirichlet alpha=0.5 hospital partition):
   - Centralized (Upper Bound)
   - Local-Only (Hospital 0)
   - FedAvg
   - FedAvg + Local Fine-Tuning
   - FedPer (Ours: Shared Conv1D+FastKAN backbone + Local Linear Head)
7. Multi-Class Metrics:
   - Raw Accuracy (%)
   - Balanced Accuracy (%) [Macro Recall]
   - Macro F1
   - Macro AUROC (One-vs-Rest)
   - Per-Class Recall and Confusion Matrix
"""

import sys
import os
import copy
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.fl.epitope_model import FastKAN

# Amino acid vocabulary
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
            dels = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]

class Top10TCRDataset(Dataset):
    def __init__(self, df, epi2idx):
        self.df = df.reset_index(drop=True)
        self.epi2idx = epi2idx
        self.tokens = torch.stack([tokenize_cdr3(seq) for seq in self.df['cdr3']])
        self.labels = torch.tensor([self.epi2idx[e] for e in self.df['antigen.epitope']], dtype=torch.long)

    def __len__(self):
        return len(self.df)

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
        # x: (B, L)
        emb = self.embedding(x).transpose(1, 2) # (B, E, L)
        c1 = F.relu(self.bn1(self.conv1(emb)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        pooled = self.pool(c2).squeeze(-1) # (B, channels)
        feat = self.kan(pooled) # (B, kan_dim)
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

def load_and_preprocess_top10(tsv_path="data/vdjdb.tsv", seed=42):
    df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    # 1. HomoSapiens TRB chain filter
    df = df[(df['species'] == 'HomoSapiens') & (df['gene'] == 'TRB') & (df['vdjdb.score'] >= 1)].copy()
    df = df.dropna(subset=['cdr3', 'antigen.epitope'])
    
    # 2. Select Top-10 epitopes
    top10_counts = df['antigen.epitope'].value_counts().head(10)
    top10_epis = list(top10_counts.index)
    df = df[df['antigen.epitope'].isin(top10_epis)].copy()
    
    # 3. Drop ambiguous CDR3s (binding multiple epitopes)
    cdr3_epi_counts = df.groupby('cdr3')['antigen.epitope'].nunique()
    unambiguous_cdr3s = set(cdr3_epi_counts[cdr3_epi_counts == 1].index)
    df = df[df['cdr3'].isin(unambiguous_cdr3s)].copy()
    
    epi2idx = {e: i for i, e in enumerate(top10_epis)}
    idx2epi = {i: e for e, i in epi2idx.items()}
    
    # 4. Strict CDR3 split (80/20 on unique CDR3s)
    np.random.seed(seed)
    unique_cdr3s = np.array(list(unambiguous_cdr3s))
    np.random.shuffle(unique_cdr3s)
    
    split_idx = int(0.8 * len(unique_cdr3s))
    train_cdr3s = set(unique_cdr3s[:split_idx])
    test_cdr3s = set(unique_cdr3s[split_idx:])
    
    train_df = df[df['cdr3'].isin(train_cdr3s)].copy().reset_index(drop=True)
    test_df = df[df['cdr3'].isin(test_cdr3s)].copy().reset_index(drop=True)
    
    # 5. Dirichlet partition across 3 hospitals (alpha=0.5)
    hospital_train_dfs = [[] for _ in range(3)]
    for epi in top10_epis:
        sub = train_df[train_df['antigen.epitope'] == epi]
        if len(sub) == 0:
            continue
        props = np.random.dirichlet([0.5, 0.5, 0.5])
        props = np.cumsum(props)
        splits = [int(p * len(sub)) for p in props[:-1]]
        chunks = np.split(sub, splits)
        for h in range(3):
            hospital_train_dfs[h].append(chunks[h])
            
    hosp_dfs = [pd.concat(dfs).reset_index(drop=True) for dfs in hospital_train_dfs]
    
    return top10_epis, epi2idx, idx2epi, train_df, test_df, hosp_dfs

def evaluate_classifier(model, loader, device, num_classes=10):
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
    
    # One-vs-rest AUROC
    try:
        macro_auroc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        macro_auroc = 0.5000
        
    return {
        "raw_acc": raw_acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
        "y_true": y_true,
        "y_pred": y_pred
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

def run_benchmark(epochs=12, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80, flush=True)
    print("RUNNING TOP-10 TRB EPITOPE CLASSIFICATION BENCHMARK", flush=True)
    print(f"Device: {device} | Random Seed: {seed} | Epochs: {epochs}", flush=True)
    print("=" * 80, flush=True)
    
    top10_epis, epi2idx, idx2epi, train_df, test_df, hosp_dfs = load_and_preprocess_top10(seed=seed)
    
    print(f"Top 10 Epitopes: {top10_epis}", flush=True)
    print(f"Training Samples: {len(train_df)} ({train_df['cdr3'].nunique()} unique CDR3s)", flush=True)
    print(f"Testing Samples:  {len(test_df)} ({test_df['cdr3'].nunique()} unique CDR3s)", flush=True)
    for h in range(3):
        print(f"  Hospital {h}: {len(hosp_dfs[h])} samples", flush=True)
        
    train_dataset = Top10TCRDataset(train_df, epi2idx)
    test_dataset = Top10TCRDataset(test_df, epi2idx)
    hosp_datasets = [Top10TCRDataset(hdf, epi2idx) for hdf in hosp_dfs]
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    hosp_loaders = [DataLoader(hds, batch_size=128, shuffle=True) for hds in hosp_datasets]
    
    y_test = np.array([epi2idx[e] for e in test_df['antigen.epitope']])
    
    # -------------------------------------------------------------
    # Baseline 1: Majority Class Baseline
    # -------------------------------------------------------------
    print("\n--- Evaluating Baseline 1: Majority Class Rule ---", flush=True)
    majority_class = train_df['antigen.epitope'].value_counts().index[0]
    maj_idx = epi2idx[majority_class]
    y_maj_pred = np.full_like(y_test, maj_idx)
    maj_raw_acc = float((y_maj_pred == y_test).mean() * 100.0)
    maj_bal_acc = float(balanced_accuracy_score(y_test, y_maj_pred) * 100.0)
    maj_macro_f1 = float(f1_score(y_test, y_maj_pred, average='macro', zero_division=0))
    print(f"Majority Class ('{majority_class}'): Raw Acc = {maj_raw_acc:.2f}%, Bal Acc = {maj_bal_acc:.2f}%, F1 = {maj_macro_f1:.4f}", flush=True)

    # -------------------------------------------------------------
    # Baseline 2: CDR3 1-NN (Levenshtein Distance)
    # -------------------------------------------------------------
    print("\n--- Evaluating Baseline 2: CDR3 1-NN (Levenshtein) ---", flush=True)
    train_cdr3_list = list(train_df[['cdr3', 'antigen.epitope']].drop_duplicates('cdr3').itertuples(index=False))
    # Sample 500 test samples for fast edit distance calculation
    sample_test = test_df.sample(n=min(500, len(test_df)), random_state=seed)
    nn_preds = []
    nn_trues = []
    for t_seq, t_epi in zip(sample_test['cdr3'], sample_test['antigen.epitope']):
        best_dist = 999
        best_epi = None
        for tr_seq, tr_epi in train_cdr3_list:
            d = levenshtein_dist(t_seq, tr_seq)
            if d < best_dist:
                best_dist = d
                best_epi = tr_epi
                if d <= 1:
                    break
        nn_preds.append(epi2idx[best_epi])
        nn_trues.append(epi2idx[t_epi])
    nn_preds = np.array(nn_preds)
    nn_trues = np.array(nn_trues)
    nn_raw_acc = float((nn_preds == nn_trues).mean() * 100.0)
    nn_bal_acc = float(balanced_accuracy_score(nn_trues, nn_preds) * 100.0)
    nn_f1 = float(f1_score(nn_trues, nn_preds, average='macro', zero_division=0))
    print(f"CDR3 1-NN: Raw Acc = {nn_raw_acc:.2f}%, Bal Acc = {nn_bal_acc:.2f}%, F1 = {nn_f1:.4f}", flush=True)

    # -------------------------------------------------------------
    # Baseline 3: 3-mer (k-mer) counts + Logistic Regression
    # -------------------------------------------------------------
    print("\n--- Evaluating Baseline 3: 3-mer + Logistic Regression ---", flush=True)
    def kmer_tokenize(seq, k=3):
        return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])
    
    train_kmers = [kmer_tokenize(s) for s in train_df['cdr3']]
    test_kmers = [kmer_tokenize(s) for s in test_df['cdr3']]
    vectorizer = CountVectorizer()
    X_train_kmer = vectorizer.fit_transform(train_kmers)
    X_test_kmer = vectorizer.transform(test_kmers)
    y_train = np.array([epi2idx[e] for e in train_df['antigen.epitope']])
    
    lr = LogisticRegression(max_iter=500, random_state=seed, class_weight='balanced')
    lr.fit(X_train_kmer, y_train)
    y_lr_pred = lr.predict(X_test_kmer)
    lr_probs = lr.predict_proba(X_test_kmer)
    lr_raw_acc = float((y_lr_pred == y_test).mean() * 100.0)
    lr_bal_acc = float(balanced_accuracy_score(y_test, y_lr_pred) * 100.0)
    lr_f1 = float(f1_score(y_test, y_lr_pred, average='macro', zero_division=0))
    lr_auroc = float(roc_auc_score(y_test, lr_probs, multi_class='ovr', average='macro'))
    print(f"3-mer + LogReg: Raw Acc = {lr_raw_acc:.2f}%, Bal Acc = {lr_bal_acc:.2f}%, F1 = {lr_f1:.4f}, AUROC = {lr_auroc:.4f}", flush=True)

    # -------------------------------------------------------------
    # Baseline 4: Label-Shuffle Control (Permutation Test)
    # -------------------------------------------------------------
    print("\n--- Running Baseline 4: Label-Shuffle Control ---", flush=True)
    shuffled_dataset = copy.deepcopy(train_dataset)
    perm = torch.randperm(len(shuffled_dataset.labels))
    shuffled_dataset.labels = shuffled_dataset.labels[perm]
    shuffled_loader = DataLoader(shuffled_dataset, batch_size=128, shuffle=True)
    
    shuf_model = Top10KANClassifier().to(device)
    shuf_opt = torch.optim.Adam(shuf_model.parameters(), lr=1e-3)
    for _ in range(4):
        train_epoch(shuf_model, shuffled_loader, shuf_opt, device)
    shuf_res = evaluate_classifier(shuf_model, test_loader, device)
    print(f"Label-Shuffle Control: Bal Acc = {shuf_res['bal_acc']:.2f}% (Chance = 10.0%), AUROC = {shuf_res['macro_auroc']:.4f}", flush=True)

    results = {}
    
    # -------------------------------------------------------------
    # 1. Centralized 1D-CNN + FastKAN (Upper Bound)
    # -------------------------------------------------------------
    print("\n[1/5] Training Centralized 1D-CNN + FastKAN...", flush=True)
    cent_model = Top10KANClassifier().to(device)
    cent_opt = torch.optim.Adam(cent_model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        loss = train_epoch(cent_model, train_loader, cent_opt, device)
        if ep % 3 == 0 or ep == epochs:
            res = evaluate_classifier(cent_model, test_loader, device)
            print(f"  Centralized Epoch {ep}/{epochs} - Loss: {loss:.4f} | Bal Acc: {res['bal_acc']:.2f}% | F1: {res['macro_f1']:.4f}", flush=True)
    results["Centralized (Upper Bound)"] = evaluate_classifier(cent_model, test_loader, device)

    # -------------------------------------------------------------
    # 2. Local-Only (Hospital 0 Island)
    # -------------------------------------------------------------
    print("\n[2/5] Training Local-Only FastKAN (Hospital 0)...", flush=True)
    local_model = Top10KANClassifier().to(device)
    local_opt = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        loss = train_epoch(local_model, hosp_loaders[0], local_opt, device)
        if ep % 3 == 0 or ep == epochs:
            res = evaluate_classifier(local_model, test_loader, device)
            print(f"  Local Epoch {ep}/{epochs} - Loss: {loss:.4f} | Bal Acc: {res['bal_acc']:.2f}% | F1: {res['macro_f1']:.4f}", flush=True)
    results["Local-Only (Hospital 0)"] = evaluate_classifier(local_model, test_loader, device)

    # -------------------------------------------------------------
    # 3. FedAvg + FastKAN (Standard Federated Learning)
    # -------------------------------------------------------------
    print("\n[3/5] Training FedAvg + FastKAN...", flush=True)
    fedavg_global = Top10KANClassifier().to(device)
    for rnd in range(1, epochs + 1):
        client_models = [copy.deepcopy(fedavg_global) for _ in range(3)]
        for h in range(3):
            opt = torch.optim.Adam(client_models[h].parameters(), lr=1e-3)
            train_epoch(client_models[h], hosp_loaders[h], opt, device)
        # Average all weights
        avg_dict = fedavg_global.state_dict()
        for k in avg_dict:
            avg_dict[k] = torch.stack([client_models[h].state_dict()[k].float() for h in range(3)]).mean(dim=0)
        fedavg_global.load_state_dict(avg_dict)
        if rnd % 3 == 0 or rnd == epochs:
            res = evaluate_classifier(fedavg_global, test_loader, device)
            print(f"  FedAvg Round {rnd}/{epochs} - Bal Acc: {res['bal_acc']:.2f}% | F1: {res['macro_f1']:.4f}", flush=True)
    results["FedAvg + FastKAN"] = evaluate_classifier(fedavg_global, test_loader, device)

    # -------------------------------------------------------------
    # 4. FedAvg + Local Fine-Tuning
    # -------------------------------------------------------------
    print("\n[4/5] Evaluating FedAvg + Local Fine-Tuning...", flush=True)
    ft_model = copy.deepcopy(fedavg_global)
    ft_opt = torch.optim.Adam(ft_model.parameters(), lr=5e-4)
    for _ in range(2):
        train_epoch(ft_model, hosp_loaders[0], ft_opt, device)
    results["FedAvg + Fine-Tuning"] = evaluate_classifier(ft_model, test_loader, device)

    # -------------------------------------------------------------
    # 5. FedPer + FastKAN (Ours: Shared Conv1D+FastKAN + Local Head)
    # -------------------------------------------------------------
    print("\n[5/5] Training FedPer + FastKAN (Ours)...", flush=True)
    fedper_global = Top10KANClassifier().to(device)
    fedper_clients = [copy.deepcopy(fedper_global) for _ in range(3)]
    
    for rnd in range(1, epochs + 1):
        # 1. Distribute shared backbone
        global_shared = get_shared_params(fedper_global)
        for h in range(3):
            set_shared_params(fedper_clients[h], global_shared)
            
        # 2. Local client training
        for h in range(3):
            opt = torch.optim.Adam(fedper_clients[h].parameters(), lr=1e-3)
            train_epoch(fedper_clients[h], hosp_loaders[h], opt, device)
            
        # 3. Server aggregates ONLY shared backbone
        client_shared = [get_shared_params(fedper_clients[h]) for h in range(3)]
        avg_shared = []
        for l_idx in range(len(client_shared[0])):
            l_avg = np.mean([client_shared[h][l_idx] for h in range(3)], axis=0)
            avg_shared.append(l_avg)
        set_shared_params(fedper_global, avg_shared)
        
        if rnd % 3 == 0 or rnd == epochs:
            res = evaluate_classifier(fedper_clients[0], test_loader, device)
            print(f"  FedPer Round {rnd}/{epochs} - Bal Acc: {res['bal_acc']:.2f}% | F1: {res['macro_f1']:.4f}", flush=True)
            
    results["FedPer + FastKAN (Ours)"] = evaluate_classifier(fedper_clients[0], test_loader, device)

    # -------------------------------------------------------------
    # SUMMARY COMPARISON TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print("FINAL RIGOROUS TOP-10 EPITOPE CLASSIFICATION BENCHMARK TABLE", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Method / Baseline':<32} | {'Bal Acc (%)':<12} | {'Raw Acc (%)':<12} | {'Macro F1':<10} | {'Macro AUROC':<11}", flush=True)
    print("-" * 85, flush=True)
    print(f"{'Random Guess (1/10 Chance)':<32} | {'10.00%':<12} | {'10.00%':<12} | {'0.1000':<10} | {'0.5000':<11}", flush=True)
    print(f"{'Majority Class Baseline':<32} | {maj_bal_acc:<11.2f}% | {maj_raw_acc:<11.2f}% | {maj_macro_f1:<10.4f} | {'0.5000':<11}", flush=True)
    print(f"{'CDR3 1-NN (Levenshtein)':<32} | {nn_bal_acc:<11.2f}% | {nn_raw_acc:<11.2f}% | {nn_f1:<10.4f} | {'--':<11}", flush=True)
    print(f"{'3-mer + Logistic Regression':<32} | {lr_bal_acc:<11.2f}% | {lr_raw_acc:<11.2f}% | {lr_f1:<10.4f} | {lr_auroc:<11.4f}", flush=True)
    s_bal = shuf_res['bal_acc']
    s_raw = shuf_res['raw_acc']
    s_f1 = shuf_res['macro_f1']
    s_auc = shuf_res['macro_auroc']
    print(f"{'Label-Shuffle Control':<32} | {s_bal:<11.2f}% | {s_raw:<11.2f}% | {s_f1:<10.4f} | {s_auc:<11.4f}", flush=True)
    print("-" * 85, flush=True)
    for name, r in results.items():
        b_acc = r['bal_acc']
        r_acc = r['raw_acc']
        m_f1 = r['macro_f1']
        m_auc = r['macro_auroc']
        print(f"{name:<32} | {b_acc:<11.2f}% | {r_acc:<11.2f}% | {m_f1:<10.4f} | {m_auc:<11.4f}", flush=True)
    print("=" * 85, flush=True)
    
    # Per-Class Recall Table for FedPer
    print("\nPer-Class Recall for FedPer + FastKAN (Ours):", flush=True)
    cm = confusion_matrix(results["FedPer + FastKAN (Ours)"]["y_true"], results["FedPer + FastKAN (Ours)"]["y_pred"])
    class_recalls = cm.diagonal() / cm.sum(axis=1)
    for i, e in enumerate(top10_epis):
        print(f"  Class {i} [{e:<20}]: Recall = {class_recalls[i]*100:.2f}% ({cm.diagonal()[i]}/{cm.sum(axis=1)[i]})", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    run_benchmark(epochs=12, seed=42)
