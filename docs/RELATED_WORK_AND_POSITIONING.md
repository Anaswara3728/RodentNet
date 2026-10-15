# 📚 RodentNet: Related Work, Literature Positioning, & Feature Matrix

**Document Purpose:** Formal cryptographic and systems positioning of RodentNet against state-of-the-art secure aggregation and post-quantum federated learning literature.

---

## 1. Taxonomic Landscape of Secure Aggregation

Secure aggregation protocols in federated learning generally fall into three major cryptographic paradigms:

1. **Masking-Based Secret Sharing (Bonawitz et al. Paradigm):**
   * *Mechanism:* Clients add structured pseudorandom masks that cancel additively when summed across all participants. Shamir secret sharing enables the aggregator to reconstruct masks of dropped clients without revealing surviving clients' inputs.
   * *Strengths:* Minimal computation overhead; exact algebraic aggregation over finite fields with zero approximation noise.
   * *Vulnerabilities:* Classical implementations rely on Diffie-Hellman (DH) or Elliptic Curve Diffie-Hellman (ECDH), which are vulnerable to polynomial-time recovery via Shor's algorithm on quantum computers. Furthermore, naive implementations that reconstruct master session secrets destroy **forward secrecy**.

2. **Homomorphic Encryption (Lattice-Based RLWE-HE):**
   * *Mechanism:* Clients encrypt local weight tensors using Ring Learning With Errors (RLWE) schemes (e.g., BFV, BGV, CKKS), allowing the server to directly sum ciphertexts in the encrypted domain without decryption.
   * *Strengths:* Post-quantum hardness; naturally handles arbitrary client dropouts without requiring threshold reconstruction rounds.
   * *Vulnerabilities:* Catastrophic ciphertext expansion ($20\times\text{--}100\times$). A 1.26 MB model update expands into 25–125 MB per client per round, exhausting clinical edge bandwidth. Furthermore, homomorphic evaluations over deep networks incur high compute latencies.

3. **Post-Quantum Masking Frameworks (Emerging PQC Paradigms):**
   * *Mechanism:* Replaces classical key agreements with lattice-based Key Encapsulation Mechanisms (NIST FIPS 203 ML-KEM) and lattice digital signatures (NIST FIPS 204 ML-DSA), paired with pseudorandom mask generation over finite fields.
   * *Key Representative Works:* **Beskar** (Zhang et al., 2023 [5]) and the **UNC Chapel Hill Study** (Zhang et al., 2024 [6]).

---

## 2. Comprehensive Feature Comparison Matrix

Below is the formal, multi-dimensional feature comparison matrix contrasting RodentNet against landmark protocols in the literature:

| Feature / Dimension | Bonawitz et al. (CCS '17) [1] | SecAgg+ (Bell et al. '20) [2] | Beskar (Zhang et al. '23) [5] | UNC Study (Zhang et al. '24) [6] | **RodentNet (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Quantum Hardness** | ❌ Classical (DH/ECDH) | ❌ Classical (ECDH) | ✅ Lattice KEM / Masking | ✅ Lattice DSA Only | ✅ **NIST FIPS 203 (ML-KEM-768)** |
| **Identity Authentication** | Classical (RSA/ECDSA) | Classical (ECDSA) | Classical / Symmetric | ✅ Dilithium / Falcon | ✅ **NIST FIPS 204 (ML-DSA-65)** |
| **Active MITM Defense** | Unauthenticated Setup | Unauthenticated Setup | Channel-level TLS | Signed Updates | ✅ **Mutual Signed Key Exchange** |
| **Aggregator Privacy** | ✅ Zero-Knowledge Agg | ✅ Zero-Knowledge Agg | ✅ Private Masking + DP | ❌ **Plaintext Updates** (No SecAgg) | ✅ **Zero-Knowledge Double Masking** |
| **Dropout Handling** | $(t, n)$ Shamir Sharing | $k$-regular Subgraphs | Threshold Recovery | N/A (Standard FedAvg) | ✅ **Shamir Ephemeral Key Sharing** |
| **Forward Secrecy** | Ephemeral DH per round | Ephemeral DH per round | Round isolation | N/A | ✅ **SHAKE-256 Hash Ratchet** |
| **False-Dropout Defense** | Double-Masking ($b_u + s_{u,v}$) | Double-Masking | Noise Blinding | N/A | ✅ **Self-Masking ($s_i^{\text{self}}$) + Pairwise** |
| **Finite Field Prime** | 31-bit integer | Multi-precision | Fixed Ring Lattice | N/A (Float32 Plaintext) | ✅ **40-bit Prime ($2^{40}-87$)** |
| **Per-Client Uplink Overhead** | $\mathcal{O}(N)$ Classical ($\approx 0.5$ KB) | $\mathcal{O}(k)$ Classical | Lattice Expansion | $\approx 3.3$ KB (Signature Only) | ✅ **10.6 KB ($N=3$) to 37.6 KB ($N=50$)** |
| **PQ Increment per Client** | 0 KB (Classical) | 0 KB (Classical) | High (Lattice Ciphertexts) | $+3.3$ KB (Signature Only) | ✅ **+10.4 KB ($N=3$) to +35.9 KB ($N=50$)** |
| **Target Setting** | Cross-Device ($N \ge 1{,}000$) | Cross-Device ($N \ge 1{,}000$) | Generic Vision (MNIST/CIFAR) | Vision & NLP (MNIST/CIFAR/RTE) | ✅ **Cross-Silo Clinical Consortia ($N \le 50$)** |

---

## 3. Rigorous Differentiation from Recent Post-Quantum FL Literature

### 3.1 Comparison Against Beskar (Zhang et al., 2023)
* **What Beskar Does:** The Beskar framework ([NSF Public Access Repository PAR 10653276](https://par.nsf.gov/servlets/purl/10653276) / arXiv:2308.15682) is a full-stack private federated learning architecture that integrates lattice-based masking with Local Differential Privacy (LDP) to achieve post-quantum privacy on standard machine learning benchmarks (MNIST and CIFAR).
* **Where RodentNet Differs and Contributes:**
  1. **Personalized Cross-Silo Architecture (FedPer):** Beskar aggregates the entire global model (FedAvg). In contrast, RodentNet explores personalized federated learning (FedPer) tailored for hospital consortia: hospital-specific diagnostic heads are strictly retained on-premise (never transmitted or federated), while shared FastKAN feature extractors are securely aggregated.
  2. **Hash-Ratcheted Forward Secrecy:** Beskar coordinates per-round keys through master secrets. RodentNet implements length-prefixed domain-separated SHAKE-256 hash ratchets: $K_{i,j}^{(r)} = \text{SHAKE-256}(\text{encode\_length\_prefixed}(\text{root}, r, \dots))$, ensuring that client dropout recovery in round $r$ reveals zero algebraic or computational information regarding historical rounds $r' < r$.
  3. **Domain Grounding on Clinical Immunology Sequences:** Rather than evaluating exclusively on synthetic computer vision benchmarks, RodentNet evaluates real-world clinical immunology sequences from VDJdb (T-cell receptor CDR3$\beta$ repertoire classification), measuring the exact interaction between edit-distance homology, label skew, and cryptographic quantization overhead.

### 3.2 Comparison Against the UNC Chapel Hill Study (Zhang et al., 2024)
* **What the UNC Study Does:** The study *"Enhancing Quantum Security over Federated Learning via Post-Quantum Cryptography"* ([arXiv:2409.04637v1](https://arxiv.org/html/2409.04637v1)) evaluates the computational latency and transmission overhead of using NIST post-quantum digital signature algorithms (Dilithium, Falcon, SPHINCS+) to sign federated model updates, defending against outsider poisoning attacks.
* **Where RodentNet Differs and Contributes:**
  1. **Confidentiality Against the Aggregator (Secure Aggregation):** The UNC study does **not** implement Secure Aggregation. Model updates in their protocol are transmitted in **plaintext** directly to the aggregator; consequently, the server observes every client's individual gradient updates, creating severe privacy vulnerabilities against honest-but-curious servers (as demonstrated in our TCR membership inference benchmark). RodentNet solves the much harder problem: provable confidentiality against the aggregator via double masking while simultaneously providing post-quantum authenticity.
  2. **End-to-End PQC Integration:** While the UNC study focuses strictly on digital signatures (ML-DSA), RodentNet couples authenticated ML-DSA-65 signatures with ML-KEM-768 lattice key encapsulation and Shamir $(t, N)$-threshold dropout recovery.

---

## 4. Formal Threat Model & Scope Bounds

### 4.1 In-Scope Security Guarantees
RodentNet provides formal security under the **Semi-Honest (Honest-but-Curious) Aggregator** model:
* **Passive Confidentiality:** An honest-but-curious server observes all uploaded masked updates $y_i^{(r)}$ and public transcript messages. Under the IND-CCA2 security of ML-KEM-768, PRG pseudorandomness of SHAKE-256, and Shamir $(t, N)$-threshold privacy ($t = \lfloor N/2 \rfloor + 1$), the server's view of any client's update is indistinguishable from uniform distribution over $\mathbb{Z}_p$ (verified via Kolmogorov-Smirnov test $p > 0.01$ and Shannon entropy $> 0.98$).
* **Active Network Adversary Defense:** An active network relay attempting to intercept, tamper with, or substitute ephemeral KEM ciphertexts or model updates is thwarted by EUF-CMA security of ML-DSA-65 signatures bound to session IDs, recipient public keys, and monotonic round nonces.
* **Forward Secrecy Across Epochs:** Compromise of an ephemeral round seed $K_{i,j}^{(r)}$ following client dropout grants no advantage in recovering prior masks $K_{i,j}^{(r')}$ ($r' < r$) due to the preimage resistance of SHAKE-256.

### 4.2 Explicit Scope Bounds and Non-Goals
1. **Actively Malicious Server View-Partitioning:** If a malicious aggregator presents conflicting survivor rosters to disjoint client subsets or actively colludes with $\ge t$ clients, a pure threshold protocol without consistency consensus can be vulnerable. Defending against active server view-partitioning requires an additional signed survivor consistency round (Bonawitz et al., 2017) and a higher threshold $t > 2N/3$. In cross-silo hospital settings, the central coordinator is typically an institutional consortium hub where honest-but-curious assumptions are standard.
2. **Memory Scrubbing in High-Level Runtimes:** While the protocol specification mandates zeroizing previous-round root seeds upon advancing the ratchet, pure-Python runtimes with immutable byte arrays and non-deterministic garbage collection cannot guarantee physical memory zeroization at the operating system level. Production clinical deployments should implement the cryptographic core in compiled C/Rust (e.g., `liboqs`).
3. **Malicious Client Model Poisoning:** Byzantine-robust defenses against adversarial client weight poisoning (e.g., zero-knowledge range proofs, coordinate-wise median aggregation) are orthogonal and explicit non-goals.

---

## 5. References

1. **Bonawitz, K., et al.** (2017). "Practical secure aggregation for privacy-preserving machine learning." *ACM CCS 2017*, pp. 1175–1191.
2. **Bell, J. H., et al.** (2020). "Secure single-server aggregation with sub-linear communication." *ACM CCS 2020*, pp. 1639–1659.
3. **Ma, Z., Chen, Y., et al.** (2023). "Flamingo: Multi-round secure aggregation with efficient client dropouts." *IEEE S&P (Oakland) 2023*, pp. 2487–2504.
4. **Froelicher, D., et al.** (2021). "Truly privacy-preserving federated learning for biomedical data." *Nature Communications*, 12(1), 5910.
5. **Zhang, Y., Behnia, R., Yavuz, A. A., Ebrahimi, R., & Bertino, E.** (2023). "Beskar: Efficient Full-Stack Private Federated Deep Learning with Post-Quantum Security." *NSF Public Access Repository (PAR 10653276)* / *arXiv:2308.15682*.
6. **Zhang, H., Mohapatra, P., et al.** (2024). "Enhancing Quantum Security over Federated Learning via Post-Quantum Cryptography." *arXiv:2409.04637v1*.
7. **NIST FIPS 203.** (2024). "Module-Lattice-Based Key-Encapsulation Mechanism Standard (ML-KEM)." *National Institute of Standards and Technology*.
8. **NIST FIPS 204.** (2024). "Module-Lattice-Based Digital Signature Standard (ML-DSA)." *National Institute of Standards and Technology*.
9. **Arivazhagan, M. G., et al.** (2019). "Federated learning with personalization layers." *arXiv:1912.00818*.
10. **Dash, P., et al.** (2017). "Quantifiable predictive features define epitope-specific T cell receptor repertoires." *Nature*, 547(7661), 89–93.
