# 🛡️ RodentNet: Formal Protocol Specification & Cryptographic Security Argument

**Protocol Designation:** $\Pi_{\text{PQ-SecAgg}}$ (Post-Quantum Secure Aggregation for Cross-Silo Consortia)  
**Standard Compliance:** NIST FIPS 203 (ML-KEM-768), NIST FIPS 204 (ML-DSA-65), NIST SP 800-185 (SHAKE-256)  
**Galois Field:** $\mathbb{Z}_p$, $p = 1{,}099{,}511{,}627{,}689 = 2^{40}-87$, Scale factor $S = 10^6$  

---

## 1. Formal Adversarial Threat Model

We define the security of $\Pi_{\text{PQ-SecAgg}}$ under the following standard cryptographic assumptions and adversarial bounds:

### 1.1 Threat Model Capabilities
1. **Semi-Honest (Honest-but-Curious) Central Aggregator $\mathcal{S}$:**  
   The aggregation server executes protocol routing and arithmetic aggregation faithfully according to the specification. However, $\mathcal{S}$ inspects all received masked update tensors, digital signatures, public keys, and execution transcripts to infer private client sequences or individual parameter updates $W_i$.
2. **Active Network Relay (Man-in-the-Middle $\mathcal{A}_{\text{net}}$):**  
   An active network adversary can intercept, reorder, substitute, or drop messages transmitted across public network channels between hospital nodes and the aggregator during key agreement or update transmission.
3. **Quantum-Capable Eavesdropper (Harvest-Now, Decrypt-Later $\mathcal{A}_{\text{quantum}}$):**  
   An adversary capable of recording all network traffic today and running Shor's algorithm on fault-tolerant quantum hardware in the future to solve discrete logarithm or integer factorization problems.
4. **Dynamic Client Disconnection (Dropout):**  
   Any subset $\mathcal{D} \subset \mathcal{U}$ of hospital clients ($|\mathcal{D}| \le N - t$) may disconnect at any point during local training or transmission without advance warning.
5. **Non-Colluding Honest Majority:**  
   At least $t = \lfloor N/2 \rfloor + 1$ hospital nodes remain honest and non-colluding.

### 1.2 Explicit Non-Goals & Out-of-Scope Attacks
* **Malicious Client Model Poisoning:** We assume hospital nodes in a vetted clinical consortium submit bona fide local training updates. Defending against arbitrary poisoned updates requires zero-knowledge range proofs over $\mathbb{Z}_p$ or secure multi-party median filtering, which is out of scope.
* **Inference from the Global Aggregate:** The aggregate sum $\sum_{i \in \mathcal{U} \setminus \mathcal{D}} W_i$ is intentionally revealed to all participants. Membership inference or attribute inference attacks on the aggregate itself require differential privacy noise injection (e.g., Gaussian mechanism over $\mathbb{Z}_p$), which is complementary but orthogonal to secure aggregation.

---

## 2. Cryptographic Primitives & Notation

* $\lambda = 192$: Target post-quantum security parameter (NIST Category 3).
* $\text{ML-DSA-65} = (\text{KeyGen}_D, \text{Sign}, \text{Verify})$: NIST FIPS 204 digital signature scheme with public keys $pk_D \in \{0,1\}^{1952}$, private keys $sk_D \in \{0,1\}^{4016}$, and signatures $\sigma \in \{0,1\}^{3293}$.
* $\text{ML-KEM-768} = (\text{KeyGen}_K, \text{Encaps}, \text{Decaps})$: NIST FIPS 203 key encapsulation mechanism with public keys $pk_K \in \{0,1\}^{1184}$, ciphertexts $ct \in \{0,1\}^{1088}$, and shared secrets $ss \in \{0,1\}^{32}$.
* $\text{SHAKE-256}(m, \ell)$: NIST SP 800-185 extendable-output function producing $\ell$-byte pseudorandom streams.
* $\text{Shamir}.\text{Split}(s, t, N)$ / $\text{Shamir}.\text{Reconstruct}(\{s_j\})$: $(t, N)$ threshold polynomial secret sharing over Galois Field $\text{GF}(257)$.
* $\mathbb{Z}_p$: Finite field of order $p = 1{,}099{,}511{,}627{,}689 = 2^{40}-87$.
* $\mathcal{U} = \{u_1, u_2, \dots, u_N\}$: Initial registered hospital client roster ($|\mathcal{U}| = N$).

---

## 3. Protocol Execution Flow ($\Pi_{\text{PQ-SecAgg}}$)

```
                            MESSAGE FLOW DIAGRAM
                            
      Client i                         Aggregator S                        Client j
         |                                  |                                  |
   [Phase 0: Setup]                         |                            [Phase 0: Setup]
   Generate (pk_D, sk_D)_i                  |                            Generate (pk_D, sk_D)_j
   Generate (pk_K, sk_K)_i                  |                            Generate (pk_K, sk_K)_j
         |------- Enroll(pk_D, pk_K)_i ---->|                                  |
         |                                  |<--- Enroll(pk_D, pk_K)_j --------|
         |                                  |                                  |
         |<------ Broadcast Roster U -------|------ Broadcast Roster U ------->|
         |                                  |                                  |
   [Phase 1: Key Agreement]                 |                            [Phase 1: Key Agreement]
   (ss_ij, ct_ij) <- Encaps(pk_K_j)         |                                  |
   sigma_ij <- Sign(ct_ij || pk_K_j || sid) |                                  |
   Derive K_ij^(r) <- SHAKE(ss_ij || r)     |                                  |
   Generate self seed b_i^(r)               |                                  |
   Split K_ij^(r) & b_i^(r) via Shamir      |                                  |
         |--- Relay(ct_ij, sigma_ij, sh_ij)->|                                 |
         |                                  |--- Relay(ct_ij, sigma_ij, sh_ij)->|
         |                                  |                            Verify sigma_ij
         |                                  |                            ss_ij <- Decaps(ct_ij)
         |                                  |                                  |
   [Phase 2: Masked Upload]                 |                            [Phase 2: Masked Upload]
   W_tilde_i = W_i + Mask(b_i) + sum(M_ij)  |                            W_tilde_j = W_j + ...
   sig_up_i <- Sign(SHA256(W_tilde_i) || r) |                                  |
         |---- (W_tilde_i, sig_up_i) ------>|                                  |
         |                                  |<--- (W_tilde_j, sig_up_j) -------|
         |                                  |                                  |
   [Phase 3: Dropout Detection]             |                            [Phase 3: Dropout Detection]
         |                                  | Identifies survivors U_surv      |
         |                                  | Identifies dropped D = U \ U_surv|
         |<-- Req: b_i (surv), K_dj (drop) -|-- Req: b_j (surv), K_di (drop) ->|
         |                                  |                                  |
         |--- Send shares of b_i & K_dj --->|                                  |
         |                                  |<--- Send shares of b_j & K_di ---|
         |                                  |                                  |
         |                            [Phase 4: Aggregation]                   |
         |                            Reconstruct b_i for survivors i in U_surv|
         |                            Reconstruct K_dj for dropped d in D      |
         |                            Subtract self-masks Mask(b_i)            |
         |                            Subtract stranded masks M_sd             |
         |                            Decode: W_global = sum(W_surv) / TotalEx |
         |<====== Broadcast Global Model W_global ============================>|
```

---

### Step-by-Step Mathematical Specification

#### Phase 0: Setup & Client Key Generation
Every hospital client $u_i \in \mathcal{U}$ independently generates its own post-quantum keypairs locally without any server-side dealer:
1. Long-term identity signing keypair: $(pk_{D, i}, sk_{D, i}) \leftarrow \text{KeyGen}_D()$.
2. Ephemeral key agreement keypair: $(pk_{K, i}, sk_{K, i}) \leftarrow \text{KeyGen}_K()$.
3. Client $u_i$ signs its KEM public key: $\sigma_i^{\text{init}} = \text{Sign}_{sk_{D, i}}(pk_{K, i} \parallel u_i \parallel \text{session\_id})$.
4. Client $u_i$ sends $(u_i, pk_{D, i}, pk_{K, i}, \sigma_i^{\text{init}})$ to Aggregator $\mathcal{S}$.
5. Aggregator $\mathcal{S}$ verifies $\sigma_i^{\text{init}}$ and broadcasts roster $\mathcal{U} = \{(u_i, pk_{D, i}, pk_{K, i})\}_{i=1}^N$.

#### Phase 1: Authenticated Pairwise Key Agreement & Share Distribution
For every peer $u_j \in \mathcal{U}$ with $j > i$:
1. Client $u_i$ encapsulates a pairwise root secret:
   $$(ss_{i,j}, ct_{i,j}) \leftarrow \text{Encaps}(pk_{K, j})$$
2. Client $u_i$ signs the ciphertext payload:
   $$\sigma_{i \to j} = \text{Sign}_{sk_{D, i}}(ct_{i,j} \parallel pk_{K, j} \parallel \text{session\_id} \parallel \text{round\_id})$$
3. Peer $u_j$ verifies $\sigma_{i \to j}$ under $pk_{D, i}$ and decapsulates:
   $$ss_{i,j} \leftarrow \text{Decaps}(sk_{K, j}, ct_{i,j})$$
4. **Domain-Separated Round Key Derivation:** Both $u_i$ and $u_j$ derive the ephemeral round key:
   $$K_{i,j}^{(r)} = \text{SHAKE-256}(\text{"rodentnet-v1\|pairwise-seed\|"} \parallel ss_{i,j} \parallel r \parallel \min(u_i, u_j) \parallel \max(u_i, u_j), 32)$$
5. Client $u_i$ generates an ephemeral private self-mask seed: $b_i^{(r)} \leftarrow \{0,1\}^{32}$.
6. Client $u_i$ generates Shamir shares over $\text{GF}(257)$ with threshold $t = \lfloor N/2 \rfloor + 1$:
   * Shares of $b_i^{(r)}$: $\{s_{i \to m}^{b}\}_{m=1}^N \leftarrow \text{Shamir}.\text{Split}(b_i^{(r)}, t, N)$.
   * Shares of $K_{i,j}^{(r)}$: $\{s_{i \to m}^{K_{i,j}}\}_{m=1}^N \leftarrow \text{Shamir}.\text{Split}(K_{i,j}^{(r)}, t, N)$.
7. Client $u_i$ encrypts share bundles under each peer's pairwise secret and routes them via $\mathcal{S}$.

#### Phase 2: Double-Masked Local Update Upload
1. Client $u_i$ quantizes and weights its local parameter vector:
   $$X_i = \lfloor W_i \cdot n_i \cdot S \rceil \bmod p$$
2. Client $u_i$ expands the local self-mask:
   $$M_i^{\text{self}} = \text{SHAKE-256}(\text{"rodentnet-v1\|mask-expand\|domain:self\|layer:"} \parallel b_i^{(r)}) \bmod p$$
3. Client $u_i$ expands pairwise masks for all peers $u_j \in \mathcal{U} \setminus \{u_i\}$:
   $$M_{i,j}^{(r)} = \text{SHAKE-256}(\text{"rodentnet-v1\|mask-expand\|domain:pairwise\|layer:"} \parallel K_{i,j}^{(r)}) \bmod p$$
4. Client $u_i$ computes its masked update tensor $\widetilde{X}_i$:
   $$\widetilde{X}_i = \left( X_i + M_i^{\text{self}} + \sum_{u_j > u_i} M_{i,j}^{(r)} - \sum_{u_j < u_i} M_{j,i}^{(r)} \right) \bmod p$$
5. Client $u_i$ signs the digest: $\sigma_i^{\text{up}} = \text{Sign}_{sk_{D, i}}(\text{SHA-256}(\widetilde{X}_i) \parallel r \parallel \text{session\_id})$.
6. Client $u_i$ uploads $(\widetilde{X}_i, \sigma_i^{\text{up}})$ to $\mathcal{S}$.

#### Phase 3: Dropout Detection & Unmasking Request
1. Aggregator $\mathcal{S}$ verifies signatures $\sigma_i^{\text{up}}$ and defines:
   * Active surviving cohort: $\mathcal{U}_{\text{surv}} \subseteq \mathcal{U}$ (where $|\mathcal{U}_{\text{surv}}| \ge t$).
   * Disconnected dropped cohort: $\mathcal{D} = \mathcal{U} \setminus \mathcal{U}_{\text{surv}}$.
2. $\mathcal{S}$ broadcasts $\mathcal{U}_{\text{surv}}$ and requests:
   * For surviving clients $u_i \in \mathcal{U}_{\text{surv}}$: shares of their self-mask seeds $s_{m \to i}^{b}$.
   * For dropped clients $u_d \in \mathcal{D}$: shares of their pairwise round seeds $s_{m \to d}^{K_{d,j}}$.

#### Phase 4: Server Aggregation & Exact Cancellation
1. $\mathcal{S}$ collects $\ge t$ shares and reconstructs:
   * Self-mask seeds $b_i^{(r)}$ for all surviving $u_i \in \mathcal{U}_{\text{surv}}$.
   * Ephemeral round keys $K_{d,s}^{(r)}$ for all dropped $u_d \in \mathcal{D}$ and survivors $u_s \in \mathcal{U}_{\text{surv}}$.
2. $\mathcal{S}$ sums all received masked updates over $\mathbb{Z}_p$:
   $$Y = \sum_{u_s \in \mathcal{U}_{\text{surv}}} \widetilde{X}_s \pmod p$$
3. Because pairwise masks between two surviving nodes cancel identically:
   $$\sum_{u_s \in \mathcal{U}_{\text{surv}}} \left( \sum_{u_{s'} > u_s} M_{s, s'}^{(r)} - \sum_{u_{s'} < u_s} M_{s', s}^{(r)} \right) \equiv 0 \pmod p$$
4. $\mathcal{S}$ subtracts the surviving self-masks and stranded masks:
   $$X_{\text{total}} = \left( Y - \sum_{u_s \in \mathcal{U}_{\text{surv}}} M_s^{\text{self}} - \sum_{u_s \in \mathcal{U}_{\text{surv}}} \sum_{u_d \in \mathcal{D}} \text{sign}(u_s, u_d) M_{s,d}^{(r)} \right) \bmod p$$
5. $\mathcal{S}$ decodes the global update:
   $$W_{\text{global}} = \frac{\text{decode}(X_{\text{total}})}{\sum_{u_s \in \mathcal{U}_{\text{surv}}} n_s}$$

---

## 4. Cryptographic Security Reductions & Arguments

We state four formal security lemmas reducing $\Pi_{\text{PQ-SecAgg}}$ to standard hardness properties:

### Lemma 1 (Confidentiality Against Eavesdroppers & Harvest-Now Adversaries)
*Assuming ML-KEM-768 is IND-CCA2 secure under the hardness of Module Learning With Errors (M-LWE) over polynomial ring $R_q = \mathbb{Z}_q[X]/(X^{256} + 1)$, no polynomial-time quantum or classical eavesdropper $\mathcal{A}$ can distinguish shared secret $ss_{i,j}$ from a uniform random 32-byte string.*
* **Proof Sketch:** Client $u_i$ encapsulates $ss_{i,j}$ using $pk_{K, j}$. Under the IND-CCA2 security of ML-KEM-768, any adversary $\mathcal{A}$ possessing public keys and transmitted ciphertext $ct_{i,j}$ has advantage $\text{Adv}_{\text{ML-KEM}}^{\text{IND-CCA2}}(\mathcal{A}) \le \text{negl}(\lambda)$. Because Shor's algorithm provides no sub-exponential speedup against lattice Module-LWE, the confidentiality holds against future quantum computers.

### Lemma 2 (Active MITM Resistance via Mutual Authentication)
*Assuming ML-DSA-65 is Existentially Unforgeable under Chosen Message Attacks (EUF-CMA) under the hardness of M-LWE and M-SIS, an active network adversary $\mathcal{A}_{\text{net}}$ cannot substitute ciphertexts or public keys without causing verification to abort with probability $1 - \text{negl}(\lambda)$.*
* **Proof Sketch:** Every KEM ciphertext $ct_{i,j}$ is bound by signature $\sigma_{i \to j} = \text{Sign}_{sk_{D, i}}(ct_{i,j} \parallel pk_{K, j} \parallel \text{session\_id} \parallel \text{round\_id})$. An adversary substituting rogue ciphertext $ct'$ must produce a valid signature $\sigma'$ under $pk_{D, i}$. This directly contradicts the EUF-CMA security of ML-DSA-65. Verified empirically in `test_1_active_mitm_rejection`.

### Lemma 3 (Forward Secrecy Across Training Rounds on Dropout)
*If client $u_d$ disconnects in round $r$ and the server reconstructs round seed $K_{d,j}^{(r)}$, the server's probability of distinguishing past masks $M_{d,j}^{(r')}$ for $r' < r$ from uniform random noise is bounded by the PRF/PRG security of SHAKE-256.*
* **Proof Sketch:** $K_{d,j}^{(r)} = \text{SHAKE-256}(\text{"prefix"} \parallel ss_{d,j} \parallel r)$. Inverting $K_{d,j}^{(r)}$ to recover root secret $ss_{d,j}$ requires finding a preimage for SHAKE-256, which requires $\mathcal{O}(2^{256})$ work. Because $r' \ne r$, round seed $K_{d,j}^{(r')}$ remains computationally independent from $K_{d,j}^{(r)}$. Verified empirically in `test_3_forward_secrecy_on_dropout`.

### Lemma 4 (False-Dropout Isolation Resistance via Self-Masking)
*A semi-honest aggregator $\mathcal{S}$ falsely declaring an active client $u_i$ as dropped cannot isolate or approximate $W_i$.*
* **Proof Sketch:** Client $u_i$'s upload is protected by $M_i^{\text{self}} = \text{SHAKE-256}(b_i^{(r)})$. If $\mathcal{S}$ falsely declares $u_i \in \mathcal{D}$, honest peers will refuse to release shares of $b_i^{(r)}$ (or client $u_i$ itself withholds $b_i^{(r)}$). Even if $\mathcal{S}$ reconstructs all pairwise masks $M_{i,j}^{(r)}$, the remaining tensor is blinded by uniform field mask $M_i^{\text{self}}$. By the PRG security of SHAKE-256, the server's view is statistically indistinguishable from uniform distribution over $\mathbb{Z}_p$. Verified empirically in `test_6_false_dropout_defense` and `test_4_statistical_uniformity`.

---

## 5. Formal Leakage Profile

Under $\Pi_{\text{PQ-SecAgg}}$, the aggregator $\mathcal{S}$ learns strictly:
1. The roster of enrolled clients $\mathcal{U}$ and their sample weights $n_i$.
2. The index set of surviving clients $\mathcal{U}_{\text{surv}}$ and dropped clients $\mathcal{D}$.
3. The true global weighted sum $X_{\text{total}} = \sum_{u_s \in \mathcal{U}_{\text{surv}}} n_s W_s$.

$\mathcal{S}$ learns **zero information** regarding:
* Any individual client's model weight vector $W_i$.
* Any intermediate gradient trajectory.
* Any pairwise root shared secret $ss_{i,j}$.
* Any historical mask from preceding rounds $r' < r$.
