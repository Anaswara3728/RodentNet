"""PQC & Authenticated Secure Aggregation Benchmarking Script
============================================================
Accurately benchmarks:
1. NIST PQC Digital Signatures (ML-DSA-65 / Dilithium3) vs Classical (RSA-2048, ECDSA P-256)
2. NIST PQC Key Encapsulation (ML-KEM-768 / Kyber768)
3. Scaled MPC Pairwise Masking Overhead across 3, 5, 10, and 20 clients
"""

import time
import os
import sys
import csv
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import numpy as np
from dilithium_py.dilithium import Dilithium3
from kyber_py.kyber import Kyber768
from app.crypto.shamir import VECTOR_FIELD, encode_signed_array, decode_signed_array
from hashlib import shake_256

N_ITER = 50


# ============================================================================
# 1. RSA-2048 Benchmark
# ============================================================================
def benchmark_rsa():
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    message = b"benchmark-payload-for-rodentnet-pqc-comparison"

    # KeyGen
    start = time.perf_counter()
    for _ in range(N_ITER):
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    keygen_ms = (time.perf_counter() - start) / N_ITER * 1000

    public_key = private_key.public_key()

    # Sign
    start = time.perf_counter()
    for _ in range(N_ITER):
        sig = private_key.sign(message, padding.PKCS1v15(), hashes.SHA256())
    sign_ms = (time.perf_counter() - start) / N_ITER * 1000

    # Verify
    start = time.perf_counter()
    for _ in range(N_ITER):
        public_key.verify(sig, message, padding.PKCS1v15(), hashes.SHA256())
    verify_ms = (time.perf_counter() - start) / N_ITER * 1000

    pk_bytes = len(public_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo))
    sig_bytes = len(sig)

    return {
        "Algorithm": "RSA-2048",
        "Type": "Signature",
        "Quantum-Safe": "No",
        "KeyGen (ms)": round(keygen_ms, 3),
        "Sign/Encaps (ms)": round(sign_ms, 3),
        "Verify/Decaps (ms)": round(verify_ms, 3),
        "Public Key (bytes)": pk_bytes,
        "Signature/CT (bytes)": sig_bytes,
    }


# ============================================================================
# 2. ECDSA P-256 Benchmark
# ============================================================================
def benchmark_ecdsa():
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    message = b"benchmark-payload-for-rodentnet-pqc-comparison"

    start = time.perf_counter()
    for _ in range(N_ITER):
        private_key = ec.generate_private_key(ec.SECP256R1())
    keygen_ms = (time.perf_counter() - start) / N_ITER * 1000

    public_key = private_key.public_key()

    start = time.perf_counter()
    for _ in range(N_ITER):
        sig = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    sign_ms = (time.perf_counter() - start) / N_ITER * 1000

    start = time.perf_counter()
    for _ in range(N_ITER):
        public_key.verify(sig, message, ec.ECDSA(hashes.SHA256()))
    verify_ms = (time.perf_counter() - start) / N_ITER * 1000

    pk_bytes = len(public_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo))
    sig_bytes = len(sig)

    return {
        "Algorithm": "ECDSA P-256",
        "Type": "Signature",
        "Quantum-Safe": "No",
        "KeyGen (ms)": round(keygen_ms, 3),
        "Sign/Encaps (ms)": round(sign_ms, 3),
        "Verify/Decaps (ms)": round(verify_ms, 3),
        "Public Key (bytes)": pk_bytes,
        "Signature/CT (bytes)": sig_bytes,
    }


# ============================================================================
# 3. ML-DSA-65 (Dilithium3) Benchmark
# ============================================================================
def benchmark_dilithium():
    message = b"benchmark-payload-for-rodentnet-pqc-comparison"

    start = time.perf_counter()
    for _ in range(N_ITER):
        pk, sk = Dilithium3.keygen()
    keygen_ms = (time.perf_counter() - start) / N_ITER * 1000

    start = time.perf_counter()
    for _ in range(N_ITER):
        sig = Dilithium3.sign(sk, message)
    sign_ms = (time.perf_counter() - start) / N_ITER * 1000

    start = time.perf_counter()
    for _ in range(N_ITER):
        valid = Dilithium3.verify(pk, message, sig)
    verify_ms = (time.perf_counter() - start) / N_ITER * 1000
    assert valid

    return {
        "Algorithm": "ML-DSA-65 (Dilithium3)",
        "Type": "Signature",
        "Quantum-Safe": "Yes",
        "KeyGen (ms)": round(keygen_ms, 3),
        "Sign/Encaps (ms)": round(sign_ms, 3),
        "Verify/Decaps (ms)": round(verify_ms, 3),
        "Public Key (bytes)": len(pk),
        "Signature/CT (bytes)": len(sig),
    }


# ============================================================================
# 4. ML-KEM-768 (Kyber768) Benchmark
# ============================================================================
def benchmark_kyber():
    start = time.perf_counter()
    for _ in range(N_ITER):
        pk, sk = Kyber768.keygen()
    keygen_ms = (time.perf_counter() - start) / N_ITER * 1000

    start = time.perf_counter()
    for _ in range(N_ITER):
        ss, ct = Kyber768.encaps(pk)
    encaps_ms = (time.perf_counter() - start) / N_ITER * 1000

    start = time.perf_counter()
    for _ in range(N_ITER):
        ss_rec = Kyber768.decaps(sk, ct)
    decaps_ms = (time.perf_counter() - start) / N_ITER * 1000
    assert ss == ss_rec

    return {
        "Algorithm": "ML-KEM-768 (Kyber768)",
        "Type": "Key Exchange",
        "Quantum-Safe": "Yes",
        "KeyGen (ms)": round(keygen_ms, 3),
        "Sign/Encaps (ms)": round(encaps_ms, 3),
        "Verify/Decaps (ms)": round(decaps_ms, 3),
        "Public Key (bytes)": len(pk),
        "Signature/CT (bytes)": len(ct),
    }


# ============================================================================
# 5. Scaled MPC Overhead (3, 5, 10, 20 clients)
# ============================================================================
def benchmark_scaled_mpc():
    from app.fl.epitope_model import SiameseEpitopeKANNet
    from app.fl.model import get_shared_parameters

    model = SiameseEpitopeKANNet()
    params = get_shared_parameters(model)
    total_params = sum(p.size for p in params)

    client_scales = [3, 5, 10, 20]
    results = []

    for num_clients in client_scales:
        roster = [f"hospital-{i}" for i in range(num_clients)]
        # Pairwise keys for client 0
        shared_secrets = {peer: os.urandom(32) for peer in roster if peer != roster[0]}

        # Plain aggregation baseline
        start = time.perf_counter()
        for _ in range(20):
            _ = [p * 100.0 for p in params]
        plain_ms = (time.perf_counter() - start) / 20 * 1000

        # Mask derivation & application
        start = time.perf_counter()
        for _ in range(20):
            masks = [np.zeros_like(layer, dtype=np.int64) for layer in params]
            for peer in roster[1:]:
                sign = 1 if roster[0] < peer else -1
                seed = shake_256(shared_secrets[peer] + b"round-1").digest(32)
                for l_idx, layer in enumerate(params):
                    stream = shake_256(seed + f"layer:{l_idx}".encode()).digest(layer.size * 8)
                    m = np.frombuffer(stream, dtype=np.uint64, count=layer.size).astype(np.int64).reshape(layer.shape)
                    masks[l_idx] = np.mod(masks[l_idx] + sign * m, VECTOR_FIELD)
        masked_ms = (time.perf_counter() - start) / 20 * 1000

        results.append({
            "clients": num_clients,
            "total_params": total_params,
            "plain_ms": round(plain_ms, 3),
            "masked_ms": round(masked_ms, 3),
            "overhead_ms": round(masked_ms - plain_ms, 3),
        })

    return results


if __name__ == "__main__":
    print("=" * 75)
    print("RodentNet Post-Quantum Cryptography & MPC Benchmark")
    print(f"Iterations per crypto measurement: {N_ITER}")
    print("=" * 75)

    pqc_results = [
        benchmark_rsa(),
        benchmark_ecdsa(),
        benchmark_dilithium(),
        benchmark_kyber(),
    ]

    print("\n--- CRYPTOGRAPHIC ALGORITHM BENCHMARK ---")
    headers = list(pqc_results[0].keys())
    col_w = [max(len(str(r[h])) for r in pqc_results + [{h: h}]) for h in headers]
    print(" | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)))
    print("-" * (sum(col_w) + 3 * len(headers)))
    for r in pqc_results:
        print(" | ".join(str(r[h]).ljust(col_w[i]) for i, h in enumerate(headers)))

    print("\n--- SCALED MPC AGGREGATION OVERHEAD ---")
    mpc_results = benchmark_scaled_mpc()
    headers_mpc = list(mpc_results[0].keys())
    col_w_mpc = [max(len(str(r[h])) for r in mpc_results + [{h: h}]) for h in headers_mpc]
    print(" | ".join(h.ljust(col_w_mpc[i]) for i, h in enumerate(headers_mpc)))
    print("-" * (sum(col_w_mpc) + 3 * len(headers_mpc)))
    for r in mpc_results:
        print(" | ".join(str(r[h]).ljust(col_w_mpc[i]) for i, h in enumerate(headers_mpc)))

    os.makedirs("results", exist_ok=True)
    with open("results/pqc_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(pqc_results)

    with open("results/mpc_overhead.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers_mpc)
        w.writeheader()
        w.writerows(mpc_results)

    print("\nSaved genuine benchmarks to results/pqc_benchmark.csv and results/mpc_overhead.csv!")
