import numpy as np

from app.crypto.mpc import MPCSecureAggregator
from app.crypto.shamir import reconstruct_array, reconstruct_secret, split_array, split_secret
from app.crypto.threshold_dilithium import (
    DevMLDSA65Provider,
    IdentityRegistry,
    ThresholdDilithiumIdentity,
)


def test_shamir_secret_reconstructs_with_threshold_subset() -> None:
    secret = b"rodentnet-threshold-dilithium-key"
    shares = split_secret(secret, threshold=3, shares=5)

    reconstructed = reconstruct_secret([shares[0], shares[2], shares[4]], threshold=3)

    assert reconstructed == secret


def test_shamir_array_reconstructs_quantized_tensor() -> None:
    tensor = np.array([[0.25, -0.125], [0.0, 1.75]], dtype=np.float64)
    shares = split_array(tensor, threshold=2, shares=4, scale=1_000_000)

    reconstructed = reconstruct_array([shares[1], shares[3]], threshold=2)

    np.testing.assert_allclose(reconstructed, tensor, atol=1e-6)


def test_mpc_pairwise_masks_cancel_to_weighted_average() -> None:
    aggregator = MPCSecureAggregator(scale=1_000_000)
    roster = ["hospital-a", "hospital-b", "hospital-c"]
    updates = [
        [np.array([0.2, -0.1, 0.3], dtype=np.float64)],
        [np.array([-0.4, 0.5, 0.1], dtype=np.float64)],
        [np.array([0.6, 0.0, -0.2], dtype=np.float64)],
    ]
    examples = [5, 3, 2]

    masked = [
        aggregator.mask_weighted_update(
            update,
            num_examples=count,
            client_id=client_id,
            roster=roster,
            round_id="round-7",
        )
        for update, count, client_id in zip(updates, examples, roster)
    ]

    aggregate = aggregator.aggregate_masked_updates(masked, total_examples=sum(examples))
    expected = sum(update[0] * count for update, count in zip(updates, examples)) / sum(examples)

    np.testing.assert_allclose(aggregate[0], expected, atol=1e-6)


def test_threshold_identity_verifies_and_rejects_replay() -> None:
    provider = DevMLDSA65Provider()
    identity = ThresholdDilithiumIdentity.create(
        client_id="hospital-a",
        threshold=2,
        shares=3,
        provider=provider,
    )
    registry = IdentityRegistry(provider=provider)
    registry.register_client(
        client_id=identity.client_id,
        public_key=identity.public_key,
        display_name="Hospital A",
        threshold=identity.threshold,
    )

    transcript = registry.issue_challenge(
        client_id=identity.client_id,
        round_id="round-1",
        model_digest="fedper-shared-bootstrap",
    )
    signature = identity.sign(transcript, share_ids=[1, 3])

    assert registry.verify(transcript=transcript, signature=signature)
    assert not registry.verify(transcript=transcript, signature=signature)
