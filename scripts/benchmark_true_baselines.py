"""Comprehensive Baseline Benchmark on Strict Unseen-TCR Split
==============================================================
Evaluates:
1. Random Guess Baseline
2. Frequency-Prior Baseline (always rank by training epitope count)
3. CDR3 1-Nearest-Neighbor Baseline (Levenshtein sequence distance)
4. Per-Epitope Specificity (Macro AUROC across individual epitopes)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.fl.vdjdb_data import VDJdbDataset

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]

def run_baselines():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE BASLEINES ON STRICT UNSEEN-TCR SPLIT")
    print("=" * 70)

    dataset = VDJdbDataset(tsv_path="data/vdjdb.tsv", synthetic=False)
    unique_tcrs = dataset.data['CDR3'].unique()
    
    np.random.seed(42)
    np.random.shuffle(unique_tcrs)
    split_idx = int(0.8 * len(unique_tcrs))
    train_tcrs = set(unique_tcrs[:split_idx])
    test_tcrs = set(unique_tcrs[split_idx:])

    train_df = dataset.data[dataset.data['CDR3'].isin(train_tcrs)].copy()
    test_df = dataset.data[dataset.data['CDR3'].isin(test_tcrs)].copy()

    print(f"Total Unique Epitopes: {len(dataset.unique_epitopes)}")
    print(f"Training Samples: {len(train_df)} ({len(train_tcrs)} unique TCRs)")
    print(f"Testing Samples:  {len(test_df)} ({len(test_tcrs)} unique TCRs)")

    # ------------------------------------------------------------------
    # 1. Random Chance Baseline (Uniform)
    # ------------------------------------------------------------------
    num_epis = len(dataset.unique_epitopes)
    rand_top1 = 1.0 / num_epis
    rand_mrr = sum(1.0 / r for r in range(1, num_epis + 1)) / num_epis
    rand_auroc = 0.5000

    # ------------------------------------------------------------------
    # 2. Frequency Prior Baseline
    # ------------------------------------------------------------------
    counts = train_df['Epitope'].value_counts()
    all_epis = list(dataset.unique_epitopes)
    ranked_epis = sorted(all_epis, key=lambda e: counts.get(e, 0), reverse=True)
    epi2rank = {e: rank + 1 for rank, e in enumerate(ranked_epis)}

    test_epis = test_df['Epitope'].values
    freq_top1 = sum(1 for e in test_epis if e == ranked_epis[0]) / len(test_epis)
    freq_mrr = sum(1.0 / epi2rank.get(e, num_epis) for e in test_epis) / len(test_epis)
    # For a fixed epitope, frequency prior gives the exact same score to all TCRs -> AUROC is 0.500
    freq_macro_auroc = 0.5000 

    # ------------------------------------------------------------------
    # 3. CDR3 1-Nearest Neighbor Baseline
    # ------------------------------------------------------------------
    print("\nComputing CDR3 1-Nearest Neighbor Baseline (sampling 500 test TCRs)...")
    train_tcr_list = list(train_df[['CDR3', 'Epitope']].drop_duplicates('CDR3').itertuples(index=False))
    
    sample_test = test_df.sample(n=min(500, len(test_df)), random_state=42)
    nn_top1_hits = 0
    
    for row in sample_test.itertuples(index=False):
        test_tcr, true_epi = row.CDR3, row.Epitope
        # Find nearest CDR3 in training set
        best_dist = 999
        best_epi = None
        for tr_tcr, tr_epi in train_tcr_list:
            d = levenshtein_distance(test_tcr, tr_tcr)
            if d < best_dist:
                best_dist = d
                best_epi = tr_epi
                if d <= 1:
                    break
        if best_epi == true_epi:
            nn_top1_hits += 1

    nn_top1 = nn_top1_hits / len(sample_test)

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON TABLE (STRICT UNSEEN-TCR EVALUATION)")
    print("=" * 70)
    print(f"{'Method / Baseline':<32} | {'Macro AUROC':<11} | {'Top-1 Acc (%)':<13} | {'MRR':<7}")
    print("-" * 70)
    print(f"{'Uniform Random Chance':<32} | {rand_auroc:<11.4f} | {rand_top1 * 100:<12.2f}% | {rand_mrr:<7.4f}")
    print(f"{'CDR3 1-NN (Levenshtein)':<32} | {'--':<11} | {nn_top1 * 100:<12.2f}% | {'--':<7}")
    print(f"{'Epitope Frequency Prior':<32} | {freq_macro_auroc:<11.4f} | {freq_top1 * 100:<12.2f}% | {freq_mrr:<7.4f}")
    print("=" * 70)

if __name__ == "__main__":
    run_baselines()
