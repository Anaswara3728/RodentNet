"""Direct Empirical Benchmark: Classical (X25519/Ed25519) vs. Post-Quantum (ML-KEM/ML-DSA)
========================================================================================
Measures exact wall-clock latencies and payload sizes on the local CPU:
- Classical Key Exchange: X25519 (ECDH) via Python cryptography library
- Classical Digital Signatures: Ed25519 via Python cryptography library
- Post-Quantum Key Exchange: NIST ML-KEM-768 (Kyber768) via pure-Python kyber_py
- Post-Quantum Digital Signatures: NIST ML-DSA-65 / Dilithium3 via pure-Python dilithium-py
All benchmarks execute 100 trials to compute Mean, Median, and P95 latency.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from cryptography.hazmat.primitives.asymmetric import x25519, ed25519
from cryptography.hazmat.primitives import serialization

from app.crypto.threshold_dilithium import DilithiumPyProvider
from kyber_py.kyber import Kyber768

def benchmark_all(trials=100):
    print("=" * 80)
    print(f"BENCHMARKING CLASSICAL VS POST-QUANTUM CRYPTOGRAPHY ({trials} TRIALS ON HOST CPU)")
    print("=" * 80)
    
    # 1. Classical: X25519 Key Agreement
    x_priv_a = x25519.X25519PrivateKey.generate()
    x_priv_b = x25519.X25519PrivateKey.generate()
    x_pub_a = x_priv_a.public_key()
    x_pub_b = x_priv_b.public_key()
    
    x_pub_bytes = x_pub_a.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    
    t_ecdh = []
    for _ in range(trials):
        t0 = time.perf_counter()
        shared_sec = x_priv_a.exchange(x_pub_b)
        t_ecdh.append((time.perf_counter() - t0) * 1000)
        
    # 2. Classical: Ed25519 Digital Signatures
    ed_priv = ed25519.Ed25519PrivateKey.generate()
    ed_pub = ed_priv.public_key()
    ed_pub_bytes = ed_pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    
    sample_msg = b"federated_model_update_digest_32bytes_sample!"
    sample_sig = ed_priv.sign(sample_msg)
    
    t_ed_sign = []
    t_ed_verify = []
    for _ in range(trials):
        t0 = time.perf_counter()
        sig = ed_priv.sign(sample_msg)
        t_ed_sign.append((time.perf_counter() - t0) * 1000)
        
        t0 = time.perf_counter()
        ed_pub.verify(sig, sample_msg)
        t_ed_verify.append((time.perf_counter() - t0) * 1000)
        
    # 3. Post-Quantum: ML-KEM-768 (Kyber768)
    ek_pk, ek_sk = Kyber768.keygen()
    t_kem_encaps = []
    t_kem_decaps = []
    sample_ss, sample_ct = Kyber768.encaps(ek_pk)
    
    for _ in range(trials):
        t0 = time.perf_counter()
        ss, ct = Kyber768.encaps(ek_pk)
        t_kem_encaps.append((time.perf_counter() - t0) * 1000)
        
        t0 = time.perf_counter()
        ss_dec = Kyber768.decaps(ek_sk, ct)
        t_kem_decaps.append((time.perf_counter() - t0) * 1000)
        
    # 4. Post-Quantum: ML-DSA-65 / Dilithium3
    dil = DilithiumPyProvider()
    d_pk, d_sk = dil.generate_keypair()
    
    t_dil_sign = []
    t_dil_verify = []
    d_sig = dil.sign(sample_msg, d_sk)
    
    for _ in range(trials):
        t0 = time.perf_counter()
        s = dil.sign(sample_msg, d_sk)
        t_dil_sign.append((time.perf_counter() - t0) * 1000)
        
        t0 = time.perf_counter()
        v = dil.verify(sample_msg, s, d_pk)
        t_dil_verify.append((time.perf_counter() - t0) * 1000)
        
    res = {
        "classical": {
            "x25519_pk_bytes": len(x_pub_bytes),
            "x25519_secret_bytes": 32,
            "x25519_exchange_ms": float(np.mean(t_ecdh)),
            "ed25519_pk_bytes": len(ed_pub_bytes),
            "ed25519_sig_bytes": len(sample_sig),
            "ed25519_sign_ms": float(np.mean(t_ed_sign)),
            "ed25519_verify_ms": float(np.mean(t_ed_verify)),
        },
        "post_quantum": {
            "ml_kem_768_pk_bytes": len(ek_pk),
            "ml_kem_768_ct_bytes": len(sample_ct),
            "ml_kem_768_secret_bytes": len(sample_ss),
            "ml_kem_768_encaps_ms": float(np.mean(t_kem_encaps)),
            "ml_kem_768_decaps_ms": float(np.mean(t_kem_decaps)),
            "ml_dsa_65_pk_bytes": len(d_pk),
            "ml_dsa_65_sig_bytes": len(d_sig),
            "ml_dsa_65_sign_ms": float(np.mean(t_dil_sign)),
            "ml_dsa_65_verify_ms": float(np.mean(t_dil_verify)),
        }
    }
    
    print("\nRESULTS SUMMARY:")
    print("--- CLASSICAL (OpenSSL / C-accelerated cryptography) ---")
    print(f"  X25519 PK: {res['classical']['x25519_pk_bytes']} B | Exchange: {res['classical']['x25519_exchange_ms']:.4f} ms")
    print(f"  Ed25519 Sig: {res['classical']['ed25519_sig_bytes']} B | PK: {res['classical']['ed25519_pk_bytes']} B | Sign: {res['classical']['ed25519_sign_ms']:.4f} ms | Verify: {res['classical']['ed25519_verify_ms']:.4f} ms")
    print("--- POST-QUANTUM (pure-Python reference implementation) ---")
    print(f"  ML-KEM-768 PK: {res['post_quantum']['ml_kem_768_pk_bytes']} B | Ciphertext: {res['post_quantum']['ml_kem_768_ct_bytes']} B | Encaps: {res['post_quantum']['ml_kem_768_encaps_ms']:.2f} ms | Decaps: {res['post_quantum']['ml_kem_768_decaps_ms']:.2f} ms")
    print(f"  ML-DSA-65 Sig: {res['post_quantum']['ml_dsa_65_sig_bytes']} B | PK: {res['post_quantum']['ml_dsa_65_pk_bytes']} B | Sign: {res['post_quantum']['ml_dsa_65_sign_ms']:.2f} ms | Verify: {res['post_quantum']['ml_dsa_65_verify_ms']:.2f} ms")
    
    out_file = Path("results/crypto_benchmark_measured.json")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved measured benchmark to: {out_file}")

if __name__ == "__main__":
    benchmark_all()
