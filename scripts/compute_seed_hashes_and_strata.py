"""
Computes exact per-seed dataset hashes, model initialization hashes,
and 10-seed oracle stratum-majority baselines.
"""

import sys
import os
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from collections import Counter

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from scripts.benchmark_authoritative_final import load_and_preprocess_top10, fast_nearest_neighbor, lev_dist, Top10KANClassifier

def main():
    data_path = REPO_ROOT / "data" / "vdjdb.tsv"
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    seeds = list(range(42, 52))
    print(f"Analyzing {len(seeds)} seeds: {seeds}")
    
    strata_majority_stats = {"d1": [], "d2": [], "d3": []}
    strata_counts = {"d1": [], "d2": [], "d3": [], "total": []}
    seed_provenance = []

    for seed in seeds:
        # Load and split exactly as in benchmark
        top10_epis, epi2idx, idx2epi, train_df, test_df, hosp_train_dfs, hosp_test_dfs, attrition = load_and_preprocess_top10(data_path, seed=seed)
        
        # 1. Dataset split hashes
        train_cdr3s_sorted = sorted(train_df['cdr3'].unique())
        test_cdr3s_sorted = sorted(test_df['cdr3'].unique())
        
        train_hash = hashlib.sha256("".join(train_cdr3s_sorted).encode('utf-8')).hexdigest()[:16]
        test_hash = hashlib.sha256("".join(test_cdr3s_sorted).encode('utf-8')).hexdigest()[:16]
        
        # 2. Model initialization hash
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = Top10KANClassifier()
        # Hash all model parameter bytes
        param_bytes = b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
        model_hash = hashlib.sha256(param_bytes).hexdigest()[:16]
        
        # 3. Stratum calculation
        train_records_by_len = {}
        for _, row in train_df.iterrows():
            seq = row['cdr3']
            lbl = epi2idx[row['antigen.epitope']]
            l = len(seq)
            if l not in train_records_by_len:
                train_records_by_len[l] = []
            train_records_by_len[l].append((seq, lbl))
            
        all_train_records = [(row['cdr3'], epi2idx[row['antigen.epitope']]) for _, row in train_df.iterrows()]
        
        min_dists = []
        for seq in test_df['cdr3']:
            d, _ = fast_nearest_neighbor(seq, train_records_by_len, all_train_records)
            min_dists.append(d)
        min_dists = np.array(min_dists)
        
        test_labels = np.array([epi2idx[e] for e in test_df['antigen.epitope']])
        
        d1_mask = (min_dists == 1)
        d2_mask = (min_dists == 2)
        d3_mask = (min_dists >= 3)
        
        # Compute majority baseline within each stratum
        def stratum_maj(mask):
            sub_lbls = test_labels[mask]
            if len(sub_lbls) == 0:
                return 0.0, None
            c = Counter(sub_lbls)
            most_common_lbl, most_common_cnt = c.most_common(1)[0]
            maj_acc = (most_common_cnt / len(sub_lbls)) * 100.0
            return maj_acc, idx2epi[most_common_lbl]
            
        maj_d1, top_epi_d1 = stratum_maj(d1_mask)
        maj_d2, top_epi_d2 = stratum_maj(d2_mask)
        maj_d3, top_epi_d3 = stratum_maj(d3_mask)
        
        strata_majority_stats["d1"].append(maj_d1)
        strata_majority_stats["d2"].append(maj_d2)
        strata_majority_stats["d3"].append(maj_d3)
        
        strata_counts["d1"].append(int(np.sum(d1_mask)))
        strata_counts["d2"].append(int(np.sum(d2_mask)))
        strata_counts["d3"].append(int(np.sum(d3_mask)))
        strata_counts["total"].append(len(min_dists))
        
        seed_provenance.append({
            "seed": seed,
            "train_cdr3_count": len(train_cdr3s_sorted),
            "test_cdr3_count": len(test_cdr3s_sorted),
            "train_hash": train_hash,
            "test_hash": test_hash,
            "model_init_hash": model_hash,
            "stratum_counts": {
                "d1": int(np.sum(d1_mask)),
                "d2": int(np.sum(d2_mask)),
                "d3": int(np.sum(d3_mask)),
                "total": len(min_dists)
            },
            "stratum_majority_acc": {
                "d1": maj_d1, "d1_top": top_epi_d1,
                "d2": maj_d2, "d2_top": top_epi_d2,
                "d3": maj_d3, "d3_top": top_epi_d3
            }
        })
        
        print(f"Seed {seed:2d}: Train Hash={train_hash}, Test Hash={test_hash}, Model Hash={model_hash}")
        print(f"         Counts: d1={np.sum(d1_mask):3d}, d2={np.sum(d2_mask):3d}, d3={np.sum(d3_mask):3d} (Total={len(min_dists)})")
        print(f"         Maj Baselines: d1={maj_d1:.2f}% ({top_epi_d1}), d2={maj_d2:.2f}% ({top_epi_d2}), d3={maj_d3:.2f}% ({top_epi_d3})")

    print("\n" + "="*80)
    print("10-SEED STRATUM MAJORITY BASELINES (Mean +/- Std):")
    print(f"  d = 1 (Near Homologs):     {np.mean(strata_majority_stats['d1']):.2f}% +/- {np.std(strata_majority_stats['d1']):.2f}%")
    print(f"  d = 2 (Intermediate):      {np.mean(strata_majority_stats['d2']):.2f}% +/- {np.std(strata_majority_stats['d2']):.2f}%")
    print(f"  d >= 3 (Distant Epitopes): {np.mean(strata_majority_stats['d3']):.2f}% +/- {np.std(strata_majority_stats['d3']):.2f}%")
    
    print("\n10-SEED STRATUM SAMPLE COUNTS & SHARES (Mean +/- Std):")
    n_tot = np.mean(strata_counts["total"])
    d1_m = np.mean(strata_counts["d1"])
    d2_m = np.mean(strata_counts["d2"])
    d3_m = np.mean(strata_counts["d3"])
    print(f"  Total test samples: {n_tot:.1f} +/- {np.std(strata_counts['total']):.1f}")
    print(f"  d = 1: {d1_m:.1f} +/- {np.std(strata_counts['d1']):.1f} ({d1_m/n_tot*100:.1f}%)")
    print(f"  d = 2: {d2_m:.1f} +/- {np.std(strata_counts['d2']):.1f} ({d2_m/n_tot*100:.1f}%)")
    print(f"  d >= 3: {d3_m:.1f} +/- {np.std(strata_counts['d3']):.1f} ({d3_m/n_tot*100:.1f}%)")
    print("="*80)

    # Save to json
    out_file = REPO_ROOT / "results" / "seed_provenance_and_strata_majority.json"
    summary_data = {
        "seeds": seeds,
        "stratum_majority_baselines": {
            "d1_mean": float(np.mean(strata_majority_stats["d1"])),
            "d1_std": float(np.std(strata_majority_stats["d1"])),
            "d2_mean": float(np.mean(strata_majority_stats["d2"])),
            "d2_std": float(np.std(strata_majority_stats["d2"])),
            "d3_mean": float(np.mean(strata_majority_stats["d3"])),
            "d3_std": float(np.std(strata_majority_stats["d3"])),
        },
        "stratum_counts_mean": {
            "total_mean": float(n_tot),
            "d1_count_mean": float(d1_m),
            "d1_share_pct": float(d1_m/n_tot*100),
            "d2_count_mean": float(d2_m),
            "d2_share_pct": float(d2_m/n_tot*100),
            "d3_count_mean": float(d3_m),
            "d3_share_pct": float(d3_m/n_tot*100),
        },
        "per_seed_provenance": seed_provenance
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved provenance and strata baselines to {out_file}")

if __name__ == "__main__":
    main()
