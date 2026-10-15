import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from app.crypto.mpc import MPCSecureAggregator
from app.crypto.shamir import VECTOR_FIELD

def test_mpc_zero_utility_loss():
    """Proves that MPC pairwise masking and quantization preserve model weights exactly,

    introducing zero loss in model utility up to floating-point quantization tolerance (1e-5).
    """
    scale = 1_000_000
    field = VECTOR_FIELD
    roster = ["hospital-1", "hospital-2", "hospital-3"]

    # Simulated client weight updates (e.g. gradients or parameters from training)
    np.random.seed(42)
    client_updates = [
        [np.random.randn(32, 16).astype(np.float64), np.random.randn(16).astype(np.float64)]
        for _ in range(3)
    ]
    examples = [150, 200, 250]
    total_examples = sum(examples)

    # 1. Plain Unmasked FedAvg (Ground Truth Aggregation)
    plain_aggregated = []
    for layer_idx in range(len(client_updates[0])):
        weighted_sum = sum(client_updates[i][layer_idx] * examples[i] for i in range(3))
        plain_aggregated.append(weighted_sum / total_examples)

    # 2. MPC Masked Aggregation (With Authenticated Key Agreement & Pairwise Masks)
    shared_secrets = {
        "hospital-1": {"hospital-2": b"secret12" * 4, "hospital-3": b"secret13" * 4},
        "hospital-2": {"hospital-1": b"secret12" * 4, "hospital-3": b"secret23" * 4},
        "hospital-3": {"hospital-1": b"secret13" * 4, "hospital-2": b"secret23" * 4},
    }

    masked_uploads = []
    for i, cid in enumerate(roster):
        agg = MPCSecureAggregator(field=field, scale=scale, shared_secrets=shared_secrets[cid])
        masked = agg.mask_weighted_update(
            client_updates[i],
            num_examples=examples[i],
            client_id=cid,
            roster=roster,
            round_id="round-1",
        )
        masked_uploads.append(masked)

    # Server sums masked updates
    server_agg = MPCSecureAggregator(field=field, scale=scale)
    mpc_aggregated = server_agg.aggregate_masked_updates(masked_uploads, total_examples=total_examples)

    # 3. Mathematical Verification of Zero Utility Loss
    for layer_idx in range(len(plain_aggregated)):
        max_diff = np.max(np.abs(plain_aggregated[layer_idx] - mpc_aggregated[layer_idx]))
        assert max_diff < 1e-5, f"Utility degraded! Max difference: {max_diff}"
        print(f"Layer {layer_idx}: Max quantization difference = {max_diff:.8f} (Zero utility loss confirmed)")

if __name__ == "__main__":
    test_mpc_zero_utility_loss()
