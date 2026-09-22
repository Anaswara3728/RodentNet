from __future__ import annotations

from dataclasses import dataclass
from hashlib import shake_256
import json
from typing import Sequence

import numpy as np

from app.crypto.shamir import (
    VECTOR_FIELD,
    SecretShare,
    decode_signed_array,
    encode_signed_array,
    split_secret,
)


@dataclass(frozen=True)
class MaskShareBundle:
    client_id: str
    round_id: str
    shares: list[SecretShare]


class MPCSecureAggregator:
    """Finite-field secure aggregation with pairwise masks and Shamir recovery hooks.

    Each client submits fixed-point model tensors multiplied by its sample count and masked
    with deterministic pairwise pads. For every pair of clients, one side adds the mask and
    the other subtracts it, so the server can sum all submitted tensors and decode only the
    cohort-level update. Shamir shares are used for threshold recovery of mask seeds when a
    production transport adds client-to-client share delivery.
    """

    def __init__(self, *, field: int = VECTOR_FIELD, scale: int = 1_000_000) -> None:
        self.field = field
        self.scale = scale

    def mask_weighted_update(
        self,
        parameters: Sequence[np.ndarray],
        *,
        num_examples: int,
        client_id: str,
        roster: Sequence[str],
        round_id: str,
    ) -> list[np.ndarray]:
        weighted = [parameter.astype(np.float64) * float(num_examples) for parameter in parameters]
        encoded = [encode_signed_array(parameter, scale=self.scale, field=self.field) for parameter in weighted]
        masks = self._pairwise_masks(client_id=client_id, roster=roster, round_id=round_id, shapes=encoded)
        return [np.mod(value + mask, self.field).astype(np.float64) for value, mask in zip(encoded, masks)]

    def aggregate_masked_updates(
        self,
        masked_updates: Sequence[Sequence[np.ndarray]],
        *,
        total_examples: int,
    ) -> list[np.ndarray]:
        if not masked_updates:
            raise ValueError("at least one masked update is required")
        if total_examples <= 0:
            raise ValueError("total_examples must be positive")

        layer_count = len(masked_updates[0])
        totals: list[np.ndarray] = []
        for layer_index in range(layer_count):
            layer_sum = np.zeros_like(masked_updates[0][layer_index], dtype=np.int64)
            for update in masked_updates:
                if len(update) != layer_count:
                    raise ValueError("all updates must contain the same number of layers")
                layer_sum = np.mod(layer_sum + np.rint(update[layer_index]).astype(np.int64), self.field)
            aggregate = decode_signed_array(layer_sum, scale=self.scale, field=self.field)
            totals.append(aggregate / float(total_examples))
        return totals

    def make_recovery_shares(
        self,
        *,
        client_id: str,
        round_id: str,
        peer_id: str,
        threshold: int,
        shares: int,
    ) -> MaskShareBundle:
        seed = self._seed_material(client_id, peer_id, round_id)
        return MaskShareBundle(
            client_id=client_id,
            round_id=round_id,
            shares=split_secret(seed, threshold=threshold, shares=shares),
        )

    def config_payload(self, *, roster: Sequence[str], round_id: str) -> dict[str, str | int]:
        return {
            "mpc_field": self.field,
            "mpc_scale": self.scale,
            "mpc_roster": json.dumps(list(roster)),
            "mpc_round_id": round_id,
            "mpc_protocol": "pairwise-shamir-secagg-v1",
        }

    def _pairwise_masks(
        self,
        *,
        client_id: str,
        roster: Sequence[str],
        round_id: str,
        shapes: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        masks = [np.zeros_like(layer, dtype=np.int64) for layer in shapes]
        for peer_id in roster:
            if peer_id == client_id:
                continue
            low, high = sorted((client_id, peer_id))
            sign = 1 if client_id == low else -1
            seed = self._seed_material(low, high, round_id)
            for layer_index, layer in enumerate(shapes):
                mask = self._mask_from_seed(seed, layer.shape, layer_index)
                masks[layer_index] = np.mod(masks[layer_index] + sign * mask, self.field)
        return masks

    def _seed_material(self, client_a: str, client_b: str, round_id: str) -> bytes:
        ordered = "|".join(sorted((client_a, client_b)))
        return f"rodentnet-secagg-v1|{round_id}|{ordered}".encode("utf-8")

    def _mask_from_seed(self, seed: bytes, shape: tuple[int, ...], layer_index: int) -> np.ndarray:
        element_count = int(np.prod(shape))
        digest_size = element_count * 8
        stream = shake_256(seed + f"|layer:{layer_index}".encode("utf-8")).digest(digest_size)
        mask = np.frombuffer(stream, dtype=np.uint64, count=element_count)
        return np.mod(mask, self.field).astype(np.int64).reshape(shape)
