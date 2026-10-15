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

strata = [r"$d = 1$" + "\n(Near Homologs\n603 test TCRs)", 
          r"$d = 2$" + "\n(Intermediate\n407 test TCRs)", 
          r"$d \geq 3$" + "\n(Distant Sequences\n581 test TCRs)"]
model_acc = [77.43, 49.75, 29.79]
majority_base = [58.37, 51.35, 34.25]
yerr = [12.28, 4.88, 2.43]

x = np.arange(len(strata))
width = 0.35

rects1 = ax.bar(x - width/2, model_acc, width, yerr=yerr, capsize=5, label='Secured FastKAN Model (5-Seed Mean)', color='#2563eb', edgecolor='#1d4ed8')
rects2 = ax.bar(x + width/2, majority_base, width, label='Stratum Majority Antigen Baseline', color='#f97316', edgecolor='#ea580c')

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
ax.annotate("+19.06 pp Gain\n(Captures Shared Motifs)", xy=(0 - width/2, 77.43), xytext=(0.4, 85),
            arrowprops=dict(facecolor='#1e3a8a', arrowstyle="->", lw=1.5),
            fontsize=8.5, weight='bold', color='#1e3a8a',
            bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#93c5fd"))

ax.annotate("Drops Below Dominant Class\n(Cannot Extrapolate Without PLMs)", xy=(2 - width/2, 29.79), xytext=(1.4, 45),
            arrowprops=dict(facecolor='#991b1b', arrowstyle="->", lw=1.5),
            fontsize=8.5, weight='bold', color='#991b1b',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#fca5a5"))

plt.tight_layout()
for d in dest_dirs:
    fig.savefig(d / "distance_decay_stratification.png", bbox_inches='tight')
plt.close(fig)
print("[OK] distance_decay_stratification.png created")

# ==============================================================================
# 4. Operations Console UI Overview
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Console Background
bg = FancyBboxPatch((0.2, 0.2), 9.6, 5.6, facecolor='#0f172a', edgecolor='#334155', linewidth=2, boxstyle="round,pad=0.1")
ax.add_patch(bg)

# Top Bar
top_bar = FancyBboxPatch((0.2, 5.1), 9.6, 0.7, facecolor='#1e293b', edgecolor='#334155', linewidth=1, boxstyle="round,pad=0.05")
ax.add_patch(top_bar)
ax.text(0.5, 5.45, "RODENTNET SECURE FEDERATED OPERATIONS CONSOLE", color='#38bdf8', weight='bold', fontsize=11, va='center')
ax.text(8.5, 5.45, "[ STATUS: ALL SYSTEMS SECURE ]", color='#4ade80', weight='bold', fontsize=9, va='center')

# Left Panel: Enrolled Hospital Nodes
panel_left = FancyBboxPatch((0.5, 0.5), 4.3, 4.4, facecolor='#1e293b', edgecolor='#475569', linewidth=1, boxstyle="round,pad=0.1")
ax.add_patch(panel_left)
ax.text(0.7, 4.6, "ENROLLED CLINICAL NODES (NIST ML-DSA-65)", color='#f8fafc', weight='bold', fontsize=9.5)

nodes_info = [
    ("Hospital 0 (St. Jude Cohort)", "Verified Dilithium3 (3293B) | Kyber768 Encaps Active", "#22c55e"),
    ("Hospital 1 (Mayo Clinic Cohort)", "Verified Dilithium3 (3293B) | Kyber768 Encaps Active", "#22c55e"),
    ("Hospital 2 (Mount Sinai Cohort)", "Verified Dilithium3 (3293B) | Kyber768 Encaps Active", "#22c55e"),
]

y_pos = 3.9
for n, desc, status_c in nodes_info:
    item_bg = FancyBboxPatch((0.7, y_pos - 0.5), 3.9, 0.7, facecolor='#0f172a', edgecolor='#334155', linewidth=0.8, boxstyle="round,pad=0.05")
    ax.add_patch(item_bg)
    ax.text(0.9, y_pos, n, color='#f1f5f9', weight='bold', fontsize=8.5)
    ax.text(0.9, y_pos - 0.28, desc, color='#94a3b8', fontsize=7)
    ax.plot(4.3, y_pos - 0.05, 'o', color=status_c, markersize=7)
    y_pos -= 0.95

# Cryptographic Audit Metric box
crypto_box = FancyBboxPatch((0.7, 0.7), 3.9, 1.0, facecolor='#0f172a', edgecolor='#38bdf8', linewidth=1, boxstyle="round,pad=0.05")
ax.add_patch(crypto_box)
ax.text(0.9, 1.45, "Session Mask Cancellation Floor:", color='#94a3b8', fontsize=8)
ax.text(0.9, 1.15, "max abs diff (Δθ) = (2.37 ± 0.35) × 10⁻⁷  (Exact Cancellation)", color='#38bdf8', weight='bold', fontsize=8.5)
ax.text(0.9, 0.88, "Galois Field: p = 1,099,511,627,689 (40-bit Prime)", color='#a5f3fc', fontsize=7.5)

# Right Panel: Real-Time Simulation Status & Telemetry
panel_right = FancyBboxPatch((5.1, 0.5), 4.4, 4.4, facecolor='#1e293b', edgecolor='#475569', linewidth=1, boxstyle="round,pad=0.1")
ax.add_patch(panel_right)
ax.text(5.3, 4.6, "FEDERATED TRAINING TELEMETRY", color='#f8fafc', weight='bold', fontsize=9.5)

telemetry = [
    ("Federated Protocol", "Secured FedAvg + FastKAN Spline Layers"),
    ("Communication Round", "Round 10 of 10 Completed (Converged)"),
    ("Secured Balanced Accuracy", "38.70% (±4.49%)  [+4.84 pp Over Local-Only]"),
    ("Macro AUROC", "0.8162 (±0.0315)  [Zero Target Leakage]"),
    ("Cryptographic Uplink Cost", "3,325 Bytes / Round (<0.35% Network Cost)"),
    ("Dropout Resilience", "2-of-3 Shamir Ephemeral Seeds Verified"),
]

y_pos2 = 4.15
for k, v in telemetry:
    ax.text(5.3, y_pos2, f"• {k}:", color='#94a3b8', fontsize=8, weight='bold')
    ax.text(5.5, y_pos2 - 0.22, f"{v}", color='#f1f5f9', fontsize=8)
    y_pos2 -= 0.58

plt.tight_layout()
for d in dest_dirs:
    fig.savefig(d / "frontend_console_mock.png", bbox_inches='tight')
plt.close(fig)
print("[OK] frontend_console_mock.png created")
