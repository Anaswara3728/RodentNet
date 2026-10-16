import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle

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

fig, ax = plt.subplots(figsize=(14, 8.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.axis('off')

# Background container
bg = FancyBboxPatch((0.2, 0.2), 13.6, 8.1, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5, boxstyle="round,pad=0.1")
ax.add_patch(bg)

# Main Title Header
ax.text(7.0, 8.0, "RODENTNET: FULL METHODOLOGY & SECURE PROTOCOL PIPELINE", 
        color='#0f172a', weight='bold', fontsize=14, ha='center')
ax.text(7.0, 7.65, "Quantum-Resilient Federated Sequence Modeling: From Raw Repertoires to Zero-Leakage Global Consensus", 
        color='#475569', style='italic', fontsize=10, ha='center')

# Define 4 Sequential Stages (Horizontal Swimlanes / Columns)
stages = [
    {
        "title": "STAGE 1: DATA & AI ENGINE",
        "subtitle": "Clinical Repertoire & Local FastKAN",
        "x": 0.5, "w": 3.0,
        "color": "#0369a1", "bg": "#f0f9ff", "border": "#38bdf8",
        "blocks": [
            ("VDJdb Human TRB Repertoire", "9,420 records / 3,864 unique clonotypes\nFiltered to top-10 viral & tumor epitopes\nStrict 80/20 non-overlapping split"),
            ("Non-IID Dirichlet Sharding", "3 Hospital silos (H_A, H_B, H_C)\nDirichlet distribution (alpha = 0.5)\nSimulates natural clinical epitope skew"),
            ("Top10KANClassifier Architecture", "1D-CNN Motif Encoder (128 channels)\nFastKAN: Gaussian RBF Spline Basis\n187,466 parameters (1.26 MB update)")
        ]
    },
    {
        "title": "STAGE 2: POST-QUANTUM MPC",
        "subtitle": "Galois Blinding & PQC Authentication",
        "x": 3.8, "w": 3.0,
        "color": "#0f766e", "bg": "#f0fdfa", "border": "#2dd4bf",
        "blocks": [
            ("PQC Key Agreement & Handshake", "NIST FIPS 203: ML-KEM-768 pairwise keys\nNIST FIPS 204: ML-DSA-65 signatures\nSHAKE-256 ratchets for forward secrecy"),
            ("40-Bit Prime Galois Quantization", "Scale factor S = 10^6 on weights W_i\nMersenne-like prime: p = 2^40 - 87\nZero modular overflow: sum(W_i) << p/2"),
            ("Pairwise Zero-Sum Blinding", "Anti-symmetric masks: m_{i,j} = -m_{j,i}\nUnique self-blinding mask: b_i\nEphemeral seed Shamir split in GF(257)")
        ]
    },
    {
        "title": "STAGE 3: SECURE AGGREGATION",
        "subtitle": "Cloud Zero-Sum Noise Cancellation",
        "x": 7.1, "w": 3.0,
        "color": "#4338ca", "bg": "#eef2ff", "border": "#818cf8",
        "blocks": [
            ("Authenticated Masked Uplink", "Client uploads: x_tilde_i = x_i + b_i + sum(m_{i,j})\nSigned with 3,293 B ML-DSA-65 signature\nTotal uplink overhead <= 3.86% (47.5 KiB)"),
            ("Modular Cloud Summation", "Server computes: sum(x_tilde_i) mod p\nPairwise masks cancel IDENTICALLY to 0:\nsum_{i<j} m_{i,j} - sum_{j<i} m_{j,i} = 0 mod p"),
            ("Provable Utility Equivalence", "Zero floating-point corruption floor:\nmax |Delta theta| = (2.42 +/- 0.30) x 10^-7\nIdentical to plain FedAvg within scale S")
        ]
    },
    {
        "title": "STAGE 4: FAULT RECOVERY & CONSENSUS",
        "subtitle": "Lagrange Unmasking & Model Broadcast",
        "x": 10.4, "w": 3.0,
        "color": "#b45309", "bg": "#fffbeb", "border": "#fcd34d",
        "blocks": [
            ("Dynamic Dropout Recovery", "If client k drops out (up to 30% tolerance):\nSurviving peers upload Shamir shares of K_{k,j}\nServer reconstructs seeds in GF(257)"),
            ("Lagrange Interpolation Engine", "Evaluates exact polynomial at x = 0\nMeasured unmasking latency: 44.3x at N=50\nServer subtracts stranded masks safely"),
            ("Dequantization & Model Broadcast", "Exact global weights: W_bar = sum(x_i)/(|S|*S)\nMIA AUROC collapsed from 0.578 to 0.515\nGlobal model broadcast to hospital silos")
        ]
    }
]

# Draw Stages and Blocks
for stage in stages:
    sx = stage["x"]
    sw = stage["w"]
    
    # Stage Header
    header = FancyBboxPatch((sx, 6.7), sw, 0.7, facecolor=stage["color"], edgecolor='none', boxstyle="round,pad=0.05")
    ax.add_patch(header)
    ax.text(sx + sw/2, 7.18, stage["title"], color='white', weight='bold', fontsize=9.5, ha='center')
    ax.text(sx + sw/2, 6.85, stage["subtitle"], color='#e2e8f0', fontsize=7.5, ha='center')
    
    # Stage Container
    container = FancyBboxPatch((sx, 0.5), sw, 6.1, facecolor=stage["bg"], edgecolor=stage["border"], linewidth=1.5, boxstyle="round,pad=0.08")
    ax.add_patch(container)
    
    # Stage Blocks
    by_positions = [4.7, 2.7, 0.7]
    for idx, (b_title, b_desc) in enumerate(stage["blocks"]):
        by = by_positions[idx]
        b_box = FancyBboxPatch((sx + 0.15, by), sw - 0.3, 1.7, facecolor='white', edgecolor=stage["border"], linewidth=1.2, boxstyle="round,pad=0.06")
        ax.add_patch(b_box)
        
        # Step number badge
        badge = FancyBboxPatch((sx + 0.25, by + 1.25), 0.35, 0.35, facecolor=stage["color"], edgecolor='none', boxstyle="circle,pad=0.02")
        ax.add_patch(badge)
        step_num = f"{stages.index(stage)+1}.{idx+1}"
        ax.text(sx + 0.425, by + 1.38, str(idx+1), color='white', weight='bold', fontsize=7.5, ha='center', va='center')
        
        # Block title & description
        ax.text(sx + 0.7, by + 1.4, b_title, color='#0f172a', weight='bold', fontsize=8.2, va='center')
        ax.text(sx + 0.25, by + 0.65, b_desc, color='#334155', fontsize=7.2, va='center', linespacing=1.25)
        
        # Intra-stage connector arrows
        if idx < 2:
            ax.annotate("", xy=(sx + sw/2, by - 0.05), xytext=(sx + sw/2, by - 0.25),
                        arrowprops=dict(arrowstyle="->", color=stage["color"], lw=1.5, mutation_scale=10))

# Inter-Stage Major Transition Arrows
for i in range(3):
    ax.annotate("", xy=(stages[i+1]["x"] - 0.05, 3.6), xytext=(stages[i]["x"] + stages[i]["w"] + 0.05, 3.6),
                arrowprops=dict(arrowstyle="->", color="#0284c7", lw=3.0, mutation_scale=18))

# Global Iteration Loop (Feedback arrow from Stage 4 back to Stage 1)
ax.annotate("", xy=(1.5, 0.4), xytext=(12.0, 0.4),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=2.5, ls="--", mutation_scale=15))
ax.text(6.75, 0.28, "Round Complete: Broadcast Updated Global Weights W^(r+1) to Hospital Silos for Next Training Epoch", 
        color="#065f46", weight='bold', fontsize=8.5, ha='center',
        bbox=dict(boxstyle="round,pad=0.2", fc="#d1fae5", ec="#10b981", lw=1))

plt.tight_layout()
for d in dest_dirs:
    fig.savefig(d / "methodology_pipeline_flow.png", bbox_inches='tight')
plt.close(fig)
print("[OK] methodology_pipeline_flow.png successfully generated in all targets!")
