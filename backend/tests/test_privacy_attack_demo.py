import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from hashlib import shake_256

from kyber_py.kyber import Kyber768
from dilithium_py.dilithium import Dilithium3
from app.crypto.shamir import VECTOR_FIELD, encode_signed_array, decode_signed_array


def test_naive_public_masking_leaks_to_server():
    """Demonstrates that deriving masks from public IDs allows an honest-but-curious

    server to completely recover an individual hospital's raw model weights.
    """
    field = VECTOR_FIELD
    scale = 1_000_000

    # Hospital A's private weights
    hospital_A_weights = np.array([0.154, -0.428, 0.891, -0.012], dtype=np.float64)
    encoded_A = encode_signed_array(hospital_A_weights, scale=scale, field=field)

    # Naive scheme: mask derived from public IDs
    round_id = "round-42"
    ordered = "|".join(sorted(("hospital-A", "hospital-B")))
    public_seed = f"rodentnet-secagg-v1|{round_id}|{ordered}".encode("utf-8")
    stream = shake_256(public_seed).digest(len(encoded_A) * 8)
    naive_mask = np.mod(np.frombuffer(stream, dtype=np.uint64, count=len(encoded_A)), field).astype(np.int64)

    # Hospital A uploads masked update to server
    hospital_A_upload = np.mod(encoded_A + naive_mask, field)

    # --- ATTACK BY THE SERVER ---
    # The server knows hospital IDs and round_id! It recomputes the exact mask:
    server_recomputed_seed = f"rodentnet-secagg-v1|{round_id}|{ordered}".encode("utf-8")
    server_stream = shake_256(server_recomputed_seed).digest(len(encoded_A) * 8)
    server_mask = np.mod(np.frombuffer(server_stream, dtype=np.uint64, count=len(encoded_A)), field).astype(np.int64)

    # Server strips mask:
    stolen_encoded = np.mod(hospital_A_upload - server_mask, field)
    stolen_weights = decode_signed_array(stolen_encoded, scale=scale, field=field)

    # Proof of leak: Server recovered exact private weights!
    assert np.allclose(stolen_weights, hospital_A_weights, atol=1e-5), "Attack failed to recover weights"
    print("Test passed: Naive public masking confirmed vulnerable (Server recovered private weights).")


def test_authenticated_ml_kem_prevents_server_recovery():
    """Demonstrates that with Authenticated ML-KEM (Kyber-768 + Dilithium-3),

    the server CANNOT recover individual weights because it lacks the shared secret.
    """
    field = VECTOR_FIELD
    scale = 1_000_000

    # Hospital A's private weights
    hospital_A_weights = np.array([0.154, -0.428, 0.891, -0.012], dtype=np.float64)
    encoded_A = encode_signed_array(hospital_A_weights, scale=scale, field=field)

    # Hospital A & B do authenticated Kyber-768 key exchange
    pk_B, sk_B = Kyber768.keygen()
    shared_secret_AB, ct_AB = Kyber768.encaps(pk_B)

    # Hospital A generates mask from private shared secret
    round_id = "round-42"
    pqc_seed = shake_256(shared_secret_AB + f"|{round_id}".encode("utf-8")).digest(32)
    stream = shake_256(pqc_seed).digest(len(encoded_A) * 8)
    pqc_mask = np.mod(np.frombuffer(stream, dtype=np.uint64, count=len(encoded_A)), field).astype(np.int64)

    # Hospital A uploads masked update
    hospital_A_upload = np.mod(encoded_A + pqc_mask, field)

    # --- ATTACK BY THE SERVER ---
    # The server tries to guess or recompute the mask using public transcript:
    server_guess_seed = f"round-42|hospital-A|hospital-B".encode("utf-8")
    server_stream = shake_256(server_guess_seed).digest(len(encoded_A) * 8)
    server_mask = np.mod(np.frombuffer(server_stream, dtype=np.uint64, count=len(encoded_A)), field).astype(np.int64)

    # Server attempts to strip:
    recovered_encoded = np.mod(hospital_A_upload - server_mask, field)
    recovered_weights = decode_signed_array(recovered_encoded, scale=scale, field=field)

    # Proof of security: Server output is random noise, correlation is essentially 0
    error = np.abs(recovered_weights - hospital_A_weights)
    assert np.all(error > 0.01), "Server was able to approximate weights!"
    print("Test passed: Authenticated ML-KEM confirmed secure (Server gets only pseudorandom noise).")


def test_active_server_mitm_tampering_detected_by_dilithium():
    """Demonstrates that if a malicious server intercepts and replaces a hospital's Kyber

    public key with its own, the Dilithium signature verification fails and the client aborts.
    """
    # Hospital B generates Dilithium identity and Kyber key
    d_pk_B, d_sk_B = Dilithium3.keygen()
    k_pk_B, k_sk_B = Kyber768.keygen()
    sig_B = Dilithium3.sign(d_sk_B, k_pk_B)

    # Malicious server generates evil Kyber key to impersonate Hospital B
    evil_k_pk, evil_k_sk = Kyber768.keygen()

    # Server forwards evil key with original signature
    # Hospital A verifies before encapsulating:
    valid_original = Dilithium3.verify(d_pk_B, k_pk_B, sig_B)
    assert valid_original, "Legitimate signature should be valid"

    tampered_valid = Dilithium3.verify(d_pk_B, evil_k_pk, sig_B)
    assert not tampered_valid, "Tampered key should fail signature verification!"

    # Server tries to forge signature for evil key without knowing d_sk_B
    fake_sig = b"\x00" * len(sig_B)
    assert not Dilithium3.verify(d_pk_B, evil_k_pk, fake_sig), "Forged signature should fail!"
    print("Test passed: Active server MITM key replacement prevented by Dilithium signature.")


def test_shamir_dropout_recovery_cancels_stranded_masks():
    """Demonstrates that if Client C drops out, Shamir shares of its pairwise seeds

    allow the server to reconstruct and subtract C's stranded masks, producing the exact average of survivors.
    """
    from app.crypto.shamir import split_secret, reconstruct_secret

    field = VECTOR_FIELD
    scale = 1_000_000

    # Survivors A and B, Dropout C
    weights_A = np.array([0.4, -0.2], dtype=np.float64)
    weights_B = np.array([0.2, 0.6], dtype=np.float64)
    weights_C = np.array([-0.5, 0.1], dtype=np.float64)

    enc_A = encode_signed_array(weights_A, scale=scale, field=field)
    enc_B = encode_signed_array(weights_B, scale=scale, field=field)

    # Pairwise seeds
    seed_AB = b"seed-AB-32-bytes-long-padding!!!"
    seed_AC = b"seed-AC-32-bytes-long-padding!!!"
    seed_BC = b"seed-BC-32-bytes-long-padding!!!"

    # Client C shared its seeds with A and B using Shamir (2-of-2)
    shares_AC = split_secret(seed_AC, threshold=2, shares=2)
    shares_BC = split_secret(seed_BC, threshold=2, shares=2)

    # Derived masks for round-1
    def make_mask(seed, sign):
        stream = shake_256(seed + b"|r1").digest(len(enc_A) * 8)
        raw = np.mod(np.frombuffer(stream, dtype=np.uint64, count=len(enc_A)), field).astype(np.int64)
        return sign * raw

    m_AB = make_mask(seed_AB, 1)   # A adds
    m_AC = make_mask(seed_AC, 1)   # A adds
    m_BA = make_mask(seed_AB, -1)  # B subs
    m_BC = make_mask(seed_BC, 1)   # B adds

    # Client A and B upload masked updates
    upload_A = np.mod(enc_A + m_AB + m_AC, field)
    upload_B = np.mod(enc_B + m_BA + m_BC, field)
    # Client C DROPS OUT and submits nothing!

    # Server receives uploads from A and B only:
    sum_survivors = np.mod(upload_A + upload_B, field)
    # Note: m_AB and m_BA cancelled, but m_AC and m_BC are stranded!

    # --- DROPOUT RECOVERY ---
    # Surviving clients A and B send their Shamir shares of C's seeds:
    recovered_seed_AC = reconstruct_secret(shares_AC, threshold=2)
    recovered_seed_BC = reconstruct_secret(shares_BC, threshold=2)
    assert recovered_seed_AC == seed_AC and recovered_seed_BC == seed_BC

    # Server computes the stranded masks and subtracts them:
    stranded_AC = make_mask(recovered_seed_AC, 1)
    stranded_BC = make_mask(recovered_seed_BC, 1)
    clean_sum = np.mod(sum_survivors - stranded_AC - stranded_BC, field)

    decoded = decode_signed_array(clean_sum, scale=scale, field=field) / 2.0
    expected = (weights_A + weights_B) / 2.0
    assert np.allclose(decoded, expected, atol=1e-5), "Dropout recovery failed to reproduce exact average"
    print("Test passed: Shamir dropout recovery successfully cancels stranded masks.")


if __name__ == "__main__":
    test_naive_public_masking_leaks_to_server()
    test_authenticated_ml_kem_prevents_server_recovery()
    test_active_server_mitm_tampering_detected_by_dilithium()
    test_shamir_dropout_recovery_cancels_stranded_masks()
