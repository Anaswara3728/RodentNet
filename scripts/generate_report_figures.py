import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Destination directories
dest_dirs = [
    Path(r"c:\Users\haris\Downloads\RodentNet-main\figures"),
    Path(r"c:\Users\haris\Downloads\RodentNet-main\RodentNet-main\figures"),
    Path(r"c:\Users\haris\Downloads\figures")
]

for d in dest_dirs:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ==============================================================================
# 1. System Topology & Cryptographic Architecture Diagram
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Central Server Box
server_box = FancyBboxPatch((2.8, 4.8), 4.4, 1.8, facecolor='#1f2937', edgecolor='#3b82f6', linewidth=2.5, boxstyle="round,pad=0.2")
ax.add_patch(server_box)
ax.text(5.0, 6.1, "CENTRAL AGGREGATION SERVER (FastAPI)", color='white', weight='bold', fontsize=12, ha='center')
ax.text(5.0, 5.65, r"$\bullet$ Verifies NIST ML-DSA-65 Signatures (3,293 B)", color='#93c5fd', fontsize=9.5, ha='center')
ax.text(5.0, 5.3, r"$\bullet$ Modular Sum: $\sum \widetilde{W}_i \equiv \sum W_i \ (\mathrm{mod}\ p)$,  $p = 2^{40}-87$", color='#93c5fd', fontsize=9.5, ha='center')
ax.text(5.0, 4.95, r"$\bullet$ Reconstructs Dropout Ephemeral Seeds $K_{i,j}^{(r)}$ via Shamir", color='#93c5fd', fontsize=9.5, ha='center')

# Hospital Nodes
hospitals = [
    {"name": "HOSPITAL A", "x": 0.5, "y": 0.6, "color": "#065f46", "border": "#10b981", "samples": "3,140 TCRs"},
    {"name": "HOSPITAL B", "x": 3.8, "y": 0.6, "color": "#1e3a8a", "border": "#3b82f6", "samples": "3,140 TCRs"},
    {"name": "HOSPITAL C", "x": 7.1, "y": 0.6, "color": "#701a75", "border": "#d946ef", "samples": "3,140 TCRs"},
]

for h in hospitals:
    h_box = FancyBboxPatch((h["x"], h["y"]), 2.4, 2.3, facecolor=h["color"], edgecolor=h["border"], linewidth=2, boxstyle="round,pad=0.15")
    ax.add_patch(h_box)
    ax.text(h["x"] + 1.2, h["y"] + 1.95, h["name"], color='white', weight='bold', fontsize=11, ha='center')
    ax.text(h["x"] + 1.2, h["y"] + 1.55, f"VDJdb Shard: {h['samples']}\nDirichlet Skew ($\\alpha=0.5$)", color='#e2e8f0', fontsize=8.5, ha='center')
    ax.text(h["x"] + 1.2, h["y"] + 0.95, "Local FastKAN Net\n+ 1D-CNN Motif Enc", color='#fef08a', fontsize=8.5, ha='center')
    ax.text(h["x"] + 1.2, h["y"] + 0.35, "ML-KEM-768 Keys\nML-DSA-65 Signer", color='#a7f3d0', fontsize=8, ha='center')

# Uplink / Downlink Arrows
ax.annotate("", xy=(3.6, 4.8), xytext=(1.7, 2.9), arrowprops=dict(arrowstyle="<->", color="#3b82f6", lw=2, mutation_scale=15))
ax.annotate("", xy=(5.0, 4.8), xytext=(5.0, 2.9), arrowprops=dict(arrowstyle="<->", color="#3b82f6", lw=2, mutation_scale=15))
ax.annotate("", xy=(6.4, 4.8), xytext=(8.3, 2.9), arrowprops=dict(arrowstyle="<->", color="#3b82f6", lw=2, mutation_scale=15))

# Annotations on links
ax.text(2.0, 4.0, "Masked Update\n+ ML-DSA Sig", color='#1e3a8a', fontsize=8, weight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.15", fc='#eff6ff', ec='#93c5fd', lw=1))
ax.text(5.0, 3.8, "Masked Update\n+ ML-DSA Sig", color='#1e3a8a', fontsize=8, weight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.15", fc='#eff6ff', ec='#93c5fd', lw=1))
ax.text(8.0, 4.0, "Masked Update\n+ ML-DSA Sig", color='#1e3a8a', fontsize=8, weight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.15", fc='#eff6ff', ec='#93c5fd', lw=1))

# Horizontal Pairwise Mesh
ax.annotate("", xy=(3.8, 1.7), xytext=(2.9, 1.7), arrowprops=dict(arrowstyle="<->", color="#f59e0b", lw=2, ls="--", mutation_scale=12))
ax.annotate("", xy=(7.1, 1.7), xytext=(6.2, 1.7), arrowprops=dict(arrowstyle="<->", color="#f59e0b", lw=2, ls="--", mutation_scale=12))
ax.text(3.35, 2.0, "ML-KEM Seed Agreement\n+ Shamir Round Shares", color='#b45309', fontsize=7.5, ha='center', weight='bold')
ax.text(6.65, 2.0, "ML-KEM Seed Agreement\n+ Shamir Round Shares", color='#b45309', fontsize=7.5, ha='center', weight='bold')

plt.tight_layout()
for d in dest_dirs:
    fig.savefig(d / "system_topology_architecture.png", bbox_inches='tight')
plt.close(fig)
print("[OK] system_topology_architecture.png created")

# ==============================================================================
# 2. Cryptographic Overheads & Classical vs Post-Quantum Benchmark
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2))

# Subplot 1: Payload sizes
labels = ['Key Agreement PK', 'Ciphertext', 'Digital Signature', 'Verify PK']
classical_bytes = [32, 32, 64, 32]
pq_bytes = [1184, 1088, 3293, 1952]

x = np.arange(len(labels))
width = 0.35

ax1.bar(x - width/2, classical_bytes, width, label='Classical (X25519/Ed25519)', color='#64748b')
ax1.bar(x + width/2, pq_bytes, width, label='Post-Quantum (ML-KEM/ML-DSA)', color='#2563eb')
ax1.set_ylabel('Payload Size (Bytes)', weight='bold')
ax1.set_title('Payload Size Comparison', weight='bold', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax1.set_yscale('log')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(fontsize=8)

# Subplot 2: Measured Latency
ops = ['Key Agreement', 'Signing', 'Verification']
class_lat = [0.033, 0.031, 0.090]
pq_lat = [9.36, 51.83, 11.22] # 3.93+5.43 for KEM

x2 = np.arange(len(ops))
ax2.bar(x2 - width/2, class_lat, width, label='Classical (ms)', color='#64748b')
ax2.bar(x2 + width/2, pq_lat, width, label='Post-Quantum (ms)', color='#059669')
ax2.set_ylabel('Measured Host CPU Time (ms)', weight='bold')
ax2.set_title('Cryptographic Operation Latency', weight='bold', fontsize=11)
ax2.set_xticks(x2)
ax2.set_xticklabels(ops, fontsize=9.5)
ax2.set_yscale('log')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(fontsize=8)

# Subplot 3: Network Bandwidth Proportion
pie_data = [1048576, 3293, 32]
pie_labels = [f'Model Weights\n(1.01 MB / 99.68%)', f'ML-DSA-65 Sig\n(3.29 KB / 0.31%)', f'SHA-256 Digest\n(32 B / 0.01%)']
pie_colors = ['#0284c7', '#f59e0b', '#dc2626']
ax3.pie(pie_data, labels=pie_labels, colors=pie_colors, startangle=140, 
        explode=(0, 0.15, 0.25), textprops={'fontsize': 8.5})
ax3.set_title('Per-Round Client Uplink Composition\n(<0.35% Cryptographic Cost)', weight='bold', fontsize=11)

plt.tight_layout()
for d in dest_dirs:
    fig.savefig(d / "crypto_overhead_comparison.png", bbox_inches='tight')
plt.close(fig)
print("[OK] crypto_overhead_comparison.png created")

# ==============================================================================
# 3. Distance Decay & Stratification Benchmark
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

strata = [r"$d = 1$" + "\n(Near Homologs\n750 test TCRs)", 
          r"$d = 2$" + "\n(Intermediate\n402 test TCRs)", 
          r"$d \geq 3$" + "\n(Distant Sequences\n603 test TCRs)"]
model_acc = [81.42, 48.41, 31.28]
majority_base = [62.26, 46.98, 36.22]
yerr = [10.00, 5.26, 4.78]

x = np.arange(len(strata))
width = 0.35

rects1 = ax.bar(x - width/2, model_acc, width, yerr=yerr, capsize=5, label='Secured FastKAN Model (10-Seed Mean)', color='#2563eb', edgecolor='#1d4ed8')
rects2 = ax.bar(x + width/2, majority_base, width, label='Oracle Stratum-Majority Baseline', color='#f97316', edgecolor='#ea580c')

ax.set_ylabel('Test Accuracy (%)', weight='bold', fontsize=11)
ax.set_title('Generalization Decays Monotonically Across Sequence Distance', weight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(strata, fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=9.5, loc='upper right')

# Value annotations
for r in rects1:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 14), textcoords="offset points",
                ha='center', va='bottom', weight='bold', color='#1e3a8a', fontsize=9.5)

for r in rects2:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', weight='bold', color='#9a3412', fontsize=9.5)

# Annotation callout
ax.annotate("+19.16 pp Gain\n(p = 0.0016, Paired t-test)", xy=(0 - width/2, 81.42), xytext=(0.35, 87),
            arrowprops=dict(facecolor='#1e3a8a', arrowstyle="->", lw=1.5),
            fontsize=8.5, weight='bold', color='#1e3a8a',
            bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#93c5fd"))

ax.annotate("Below Oracle Baseline\n(-4.94 pp, p = 0.0047)", xy=(2 - width/2, 31.28), xytext=(1.35, 45),
            arrowprops=dict(facecolor='#991b1b', arrowstyle="->", lw=1.5),
            fontsize=8.5, weight='bold', color='#991b1b',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#fca5a5"))

plt.tight_layout()
for d in dest_dirs:
    fig.savefig(d / "distance_decay_stratification.png", bbox_inches='tight')
plt.close(fig)
print("[OK] distance_decay_stratification.png created")


print("[DONE] All authoritative empirical figures generated successfully.")

