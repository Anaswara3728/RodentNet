# RodentNet: Privacy-Preserving Federated Learning with Post-Quantum Cryptography and Secure Multi-Party Computation

## 1. Problem Statement

The problem: Multiple institutions (hospitals/research labs) need to collaboratively train machine learning models on sensitive data (e.g., patient records, immunological data) without sharing raw data. Current federated learning systems have two critical vulnerabilities:
1. Model weight updates can be intercepted or tampered with during transit (integrity problem)
2. Individual model weights can leak private training data information (privacy problem)
3. Existing cryptographic defenses (RSA/ECDSA) will be broken by quantum computers (future-proofing problem)

RodentNet addresses all three by combining: FedPer for personalized federated learning, ML-DSA-65 (CRYSTALS-Dilithium) for quantum-resistant authentication, and MPC with pairwise masking for secure aggregation.

---

## 2. Objectives

1. Design and implement a federated learning framework using FedPer strategy
2. Implement post-quantum digital signatures using CRYSTALS-Dilithium (ML-DSA-65) for participant authentication
3. Develop Shamir Secret Sharing based threshold identity verification for multi-party key management
4. Implement secure multi-party computation (MPC) with pairwise additive masking for privacy-preserving model aggregation
5. Build a Siamese dual-encoder with FastKAN (RBF-KAN) interaction head for TCR-epitope binding prediction on VDJdb dataset
6. Develop non-IID data partitioning strategies that simulate real-world clinical data heterogeneity
7. Create a web-based operations console for FL simulation monitoring and cryptographic handshake management
8. Evaluate the system against ML metrics (accuracy, loss, InfoNCE), cryptographic overhead metrics, and compare with state-of-the-art approaches

---

## 3. Methodology Framework

### Phase 1: System Architecture Design
- Monorepo architecture with FastAPI backend and React frontend
- Modular separation: crypto layer, FL layer, API layer
- FedPer model split: shared layers (`features.*`, `projection.*`) vs personal head (`personal_head.*`)

### Phase 2: Post-Quantum Cryptography Implementation
- CRYSTALS-Dilithium (ML-DSA-65) digital signature scheme
- Threshold identity: Shamir secret sharing splits the signing key into N shares, T threshold needed to reconstruct
- Handshake protocol: Enroll → Challenge → Sign (reconstruct key from shares) → Verify → Admit
- Dev fallback: HMAC-SHA3-512 for development without pqcrypto C bindings

### Phase 3: Secure Multi-Party Computation
- Pairwise additive masking protocol using SHAKE-256 deterministic mask generation
- Fixed-point quantization (scale=1,000,000) over GF(2^31-1) for floating-point model weights
- Mask cancellation: Client A adds `mask_AB`, Client B subtracts `mask_AB` → server sum cancels masks

### Phase 4: Federated Learning Pipeline (CIFAR-10 Demo)
- `FedPerCifarNet`: 3-layer Conv2d backbone + projection layer (shared) + linear personal head (local)
- Flower framework for FL orchestration
- `MPCFedPerStrategy`: custom strategy integrating MPC aggregation + Dilithium verification
- `RodentNetClient`: NumPy Flower client with MPC masking in `fit()` method

### Phase 5: Epitope Classification Extension (VDJdb)
- `SiameseEpitopeKANNet`: dual-arm encoder for TCR and epitope amino acid sequences
- `ResidueMotifEncoder`: Embedding → 2× Conv1d + BatchNorm + ReLU → AdaptiveMaxPool1d
- FastKAN (RBF-KAN): fixed-grid RBF kernels (buffer, not learnable) → linear mixing weights (learnable, MPC-safe)
- InfoNCE contrastive loss (symmetric) over B×B pairwise logit matrix
- PersonalCalibration: learnable temperature scalar (local, not federated)
- VDJdb dataset: TSV parsing with CDR3/Epitope columns, non-IID partitioning by epitope group
- AATokenizer: 20 amino acids + PAD(0) + UNK(21), max_len truncation/padding

### Phase 6: Evaluation and Testing
- Unit tests for crypto protocols, model architecture, and parameter splitting
- Simulation with fake data for rapid smoke testing
- Metrics: accuracy, loss, InfoNCE convergence, crypto overhead

---

## 4. FedPer Clarity

- **What FedPer is and how it differs from FedAvg and FedProx**: FedPer (Federated Personalization) allows clients to collaboratively train a shared base model while keeping personal classification heads purely local. Unlike FedAvg (which averages the whole model) and FedProx (which adds a proximal term to penalize divergence from the global model), FedPer explicitly separates representation learning from task-specific classification.
- **Why FedPer was chosen**: It handles non-IID data effectively, enabling personalization for individual clients, and naturally accommodates different label spaces across diverse clinical or research environments.
- **How the split works**: `SHARED_PREFIXES = ('features.', 'projection.')` are federated and synchronized across clients. The `personal_head` parameters are never sent to the server.
- **Representation vs. Adaptation**: The shared layers learn generalizable representations from the collective dataset, while the personal head adapts specifically to the local data distribution of each client.
- **In the CIFAR-10 model**: The Conv2d blocks and the projection linear layers are shared. The final classification linear layer serves as the personal head.
- **In the Epitope model**: The `ResidueMotifEncoder`, projection layers, and FastKAN component are shared. The `PersonalCalibration` (temperature scalar) remains personal to each client.

---

## 5. Comparison Tables

### Table 1: ML Methods Comparison
| Method | Personalization | Non-IID Handling | Communication Cost | Privacy |
|--------|----------------|-----------------|-------------------|--------|
| FedAvg | None | Poor | Low | None |
| FedProx | Proximal term | Moderate | Low | None |
| FedBN | BatchNorm only | Moderate | Low | None |
| FedPer (Ours) | Full personal head | Strong | Medium | MPC + PQC |
| pFedMe | Meta-learning | Strong | High | None |

### Table 2: Cryptographic Methods Comparison
| Method | Key Size | Sig Size | Quantum-Safe | Speed | Standard |
|--------|----------|----------|-------------|-------|----------|
| RSA-2048 | 256 B | 256 B | ❌ No | Fast | FIPS 186 |
| ECDSA P-256 | 32 B | 64 B | ❌ No | Fast | FIPS 186 |
| FALCON-512 | 897 B | 666 B | ✅ Yes | Medium | NIST PQC |
| ML-DSA-65 / Dilithium3 (Ours) | 1952 B | 3293 B | ✅ Yes | Fast | NIST FIPS 204 |

### Table 3: State-of-the-Art Comparison
| System | FL Strategy | Crypto | Secure Aggregation | PQC | Dataset | Application |
|--------|------------|--------|-------------------|-----|---------|-------------|
| Google FL (2019) | FedAvg | TLS | SecAgg | ❌ | Mobile | Keyboard prediction |
| FATE (WeBank) | FedAvg/FedProx | RSA | Paillhe homomorphic | ❌ | Finance | Credit scoring |
| PySyft (OpenMined) | Various | RSA/ECDSA | SMPC | ❌ | General | Research |
| Flower (Beutel et al.) | Pluggable | None built-in | None built-in | ❌ | General | Research |
| RodentNet (Ours) | FedPer | ML-DSA-65 Threshold | Pairwise Shamir MPC | ✅ | CIFAR-10 / VDJdb | Healthcare/Immunology |

### Table 4: Dataset Comparison
| Dataset | Type | Samples | Classes | Input | Task | Non-IID |
|---------|------|---------|---------|-------|------|--------|
| CIFAR-10 | Images | 60,000 | 10 | 32×32×3 RGB | Classification | IID split |
| VDJdb | Sequences | ~100K+ | Variable | CDR3 + Epitope AA | Binding prediction | By epitope group |

### Table 5: Evaluation Metrics
| Metric | Domain | Description | Applied To |
|--------|--------|-------------|------------|
| Accuracy | ML | Correct predictions / total | CIFAR-10 |
| Cross-Entropy Loss | ML | Classification loss | CIFAR-10 |
| InfoNCE Loss | ML | Contrastive loss over B×B matrix | Epitope |
| Top-1 Accuracy | ML | Nearest-epitope classification | Epitope |
| MRR | ML | Mean Reciprocal Rank | Epitope |
| AUROC | ML | Area Under ROC | Epitope |
| Signature Verify Time | Crypto | Time for Dilithium verify() | Security |
| MPC Overhead | Crypto | Mask generation + encoding time | Privacy |
| Communication Cost | System | Bytes transmitted per round | Efficiency |

### Table 6: Ablation Study Design
| Configuration | MPC | PQC Auth | FedPer | Expected Impact |
|--------------|-----|----------|--------|----------------|
| Full System | ✅ | ✅ | ✅ | Baseline (best privacy + personalization) |
| No MPC | ❌ | ✅ | ✅ | No weight privacy; accuracy unchanged |
| No PQC | ✅ | ❌ | ✅ | Vulnerable to quantum attacks; accuracy unchanged |
| FedAvg (no FedPer) | ✅ | ✅ | ❌ | Reduced personalization under non-IID |
| No Privacy | ❌ | ❌ | ✅ | Maximum accuracy, no privacy |

---

## 6. System Architecture

```text
+-----------------------+         REST API         +---------------------------+
|    Frontend (React)   | <======================> |    Backend (FastAPI)      |
|    - Vite             |                          |    - API Layer            |
|    - Axios            |                          |                           |
+-----------------------+                          +---------------------------+
                                                               |
                                                               v
                                                   +---------------------------+
                                                   |         FL Layer          |
                                                   |  - Flower Strategy        |
                                                   |  - RodentNet Client       |
                                                   |  - Simulation             |
                                                   |  - PyTorch Models         |
                                                   +---------------------------+
                                                               |
                                                               v
                                                   +---------------------------+
                                                   |        Crypto Layer       |
                                                   |  - Dilithium (ML-DSA-65)  |
                                                   |  - Shamir Secret Sharing  |
                                                   |  - Pairwise Masking (MPC) |
                                                   +---------------------------+

Data Flow:
1. Client trains locally on private data.
2. Client signs weight updates using ML-DSA-65 (reconstructed via Shamir shares).
3. Client applies MPC pairwise masking to weights.
4. Server verifies signatures and aggregates masked weights (masks cancel out).
5. Server sends updated global weights back to clients.
```

---

## 7. Key Technologies Used

- **Python 3.10+, FastAPI, Uvicorn**: Modern, fast web framework for building the backend REST API, served via Uvicorn.
- **PyTorch, TorchVision**: Deep learning framework used for constructing the Siamese models, KAN networks, and handling CIFAR-10 data.
- **Flower (FL framework)**: A federated learning framework used for orchestrating the client-server communication and simulation.
- **pqcrypto (ML-DSA-65 / Dilithium3)**: Library for post-quantum cryptographic algorithms, used for quantum-safe digital signatures.
- **Shamir Secret Sharing**: A cryptographic algorithm used for threshold key management, splitting keys into N shares where T are required for reconstruction.
- **SHAKE-256**: An extendable-output function (XOF) used for deterministic and secure pairwise mask generation in the MPC protocol.
- **React, Vite, Axios**: Frontend stack for building a fast, responsive web-based operations console.
- **NumPy, Pandas**: Foundational libraries for numerical computation, dataset parsing (VDJdb TSV), and data manipulation.
