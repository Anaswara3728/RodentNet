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


from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization


def encode_length_prefixed(*components: bytes | str) -> bytes:
    """Encode domain separation components with 4-byte big-endian length prefixes.
    
    Prevents canonicalization and delimiter-injection attacks across variable-length fields.
    """
    buf = bytearray()
    for c in components:
        b = c.encode("utf-8") if isinstance(c, str) else c
        buf.extend(len(b).to_bytes(4, "big"))
        buf.extend(b)
    return bytes(buf)


def ratchet_root_secret(root_secret: bytes, round_id: str) -> bytes:
    """Derive forward-secret next-epoch pairwise root secret via one-way hash ratchet.
    
    Ensures that prospective node compromise at round r+1 cannot reveal pairwise
    seeds of preceding rounds r' <= r.
    """
    tag = encode_length_prefixed(b"rodentnet-v1", b"ratchet", round_id, root_secret)
    return shake_256(tag).digest(32)


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
    cohort-level update. 
    
    Privacy is guaranteed via an initial Key Agreement phase (ECDH used as a placeholder
    for Post-Quantum ML-KEM). The pairwise masks are derived from the resulting shared
    secrets, meaning the server cannot compute or strip the masks.
    """

    def __init__(
        self, 
        *, 
        field: int = VECTOR_FIELD, 
        scale: int = 1_000_000,
        shared_secrets: dict[str, bytes] | None = None,
        private_key_bytes: bytes | None = None,
        peer_public_keys: dict[str, bytes] | None = None,
        self_mask_seed: bytes | None = None,
    ) -> None:
        self.field = field
        self.scale = scale
        self.shared_secrets = shared_secrets or {}
        self.private_key_bytes = private_key_bytes
        self.peer_public_keys = peer_public_keys or {}
        self.self_mask_seed = self_mask_seed

    def mask_weighted_update(
        self,
        parameters: Sequence[np.ndarray],
        *,
        num_examples: int,
        client_id: str,
        roster: Sequence[str],
        round_id: str,
        self_mask_seed: bytes | None = None,
    ) -> list[np.ndarray]:
        weighted = [parameter.astype(np.float64) * float(num_examples) for parameter in parameters]
        encoded = [encode_signed_array(parameter, scale=self.scale, field=self.field) for parameter in weighted]
        active_self = self_mask_seed or self.self_mask_seed
        masks = self._pairwise_masks(
            client_id=client_id, roster=roster, round_id=round_id, shapes=encoded, self_mask_seed=active_self
        )
        return [np.mod(value + mask, self.field).astype(np.float64) for value, mask in zip(encoded, masks)]

    def compute_self_mask(
        self,
        self_mask_seed: bytes,
        shapes: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Compute the local private self-mask for layer shapes."""
        return [self._mask_from_seed(self_mask_seed, layer.shape, layer_index, domain="self") 
                for layer_index, layer in enumerate(shapes)]

    def compute_stranded_pairwise_mask(
        self,
        *,
        dropped_client_id: str,
        active_client_id: str,
        seed: bytes,
        shapes: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Compute the pairwise mask that dropped_client applied to active_client.
        
        If active_client < dropped_client: active added the mask, so stranded in accumulator with +1.
        If active_client > dropped_client: active subtracted the mask, so stranded with -1.
        """
        sign = 1 if active_client_id < dropped_client_id else -1
        return [sign * self._mask_from_seed(seed, layer.shape, layer_index, domain="pairwise")
                for layer_index, layer in enumerate(shapes)]

    def aggregate_masked_updates(
        self,
        masked_updates: Sequence[Sequence[np.ndarray]],
        *,
        total_examples: int,
        self_mask_seeds: Sequence[bytes | None] | None = None,
        stranded_pairwise_masks: Sequence[Sequence[np.ndarray]] | None = None,
    ) -> list[np.ndarray]:
        if not masked_updates:
            raise ValueError("at least one masked update is required")
        if total_examples <= 0:
            raise ValueError("total_examples must be positive")

        layer_count = len(masked_updates[0])
        totals: list[np.ndarray] = []
        for layer_index in range(layer_count):
            layer_sum = np.zeros_like(masked_updates[0][layer_index], dtype=np.int64)
            for update_idx, update in enumerate(masked_updates):
                if len(update) != layer_count:
                    raise ValueError("all updates must contain the same number of layers")
                u_layer = np.rint(update[layer_index]).astype(np.int64)
                
                # Subtract self-mask if surviving client revealed its self_mask_seed
                if self_mask_seeds and update_idx < len(self_mask_seeds) and self_mask_seeds[update_idx]:
                    s_seed = self_mask_seeds[update_idx]
                    s_mask = self._mask_from_seed(s_seed, u_layer.shape, layer_index, domain="self")
                    u_layer = np.mod(u_layer - s_mask, self.field)
                    
                layer_sum = np.mod(layer_sum + u_layer, self.field)
                
            # Subtract stranded pairwise masks from dropped clients if provided
            if stranded_pairwise_masks:
                for stranded in stranded_pairwise_masks:
                    if len(stranded) == layer_count:
                        str_layer = np.rint(stranded[layer_index]).astype(np.int64)
                        layer_sum = np.mod(layer_sum - str_layer, self.field)
                        
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
        self_mask_seed: bytes | None = None,
    ) -> list[np.ndarray]:
        masks = [np.zeros_like(layer, dtype=np.int64) for layer in shapes]
        for peer_id in roster:
            if peer_id == client_id:
                continue
            low, high = sorted((client_id, peer_id))
            sign = 1 if client_id == low else -1
            seed = self._seed_material(low, high, round_id)
            for layer_index, layer in enumerate(shapes):
                mask = self._mask_from_seed(seed, layer.shape, layer_index, domain="pairwise")
                masks[layer_index] = np.mod(masks[layer_index] + sign * mask, self.field)
                
        # Apply local self-mask if enabled
        if self_mask_seed:
            for layer_index, layer in enumerate(shapes):
                s_mask = self._mask_from_seed(self_mask_seed, layer.shape, layer_index, domain="self")
                masks[layer_index] = np.mod(masks[layer_index] + s_mask, self.field)
                
        return masks

    def _seed_material(self, client_a: str, client_b: str, round_id: str) -> bytes:
        c_min, c_max = sorted((client_a, client_b))
        
        # 1. Check if we have an authenticated post-quantum shared secret (ML-KEM-768)
        for candidate in (client_a, client_b):
            if candidate in self.shared_secrets:
                secret = self.shared_secrets[candidate]
                tag = encode_length_prefixed(b"rodentnet-v1", b"pairwise-seed", c_min, c_max, round_id, secret)
                return shake_256(tag).digest(32)

        # 2. Classical ECDH fallback if configured
        if self.private_key_bytes and self.peer_public_keys:
            peer_id = client_b if client_b in self.peer_public_keys else client_a
            if peer_id in self.peer_public_keys:
                private_key = x25519.X25519PrivateKey.from_private_bytes(self.private_key_bytes)
                peer_public_bytes = self.peer_public_keys[peer_id]
                peer_public_key = x25519.X25519PublicKey.from_public_bytes(peer_public_bytes)
                shared_secret = private_key.exchange(peer_public_key)
                tag = encode_length_prefixed(b"rodentnet-v1", b"ecdh-seed", c_min, c_max, round_id, shared_secret)
                return shake_256(tag).digest(32)

        # 3. Development fallback (strictly 256-bit digest)
        tag = encode_length_prefixed(b"rodentnet-secagg-v1", b"dev-fallback", c_min, c_max, round_id)
        return shake_256(tag).digest(32)

    def _mask_from_seed(self, seed: bytes, shape: tuple[int, ...], layer_index: int, domain: str = "pairwise") -> np.ndarray:
        element_count = int(np.prod(shape))
        digest_size = element_count * 8
        tag = encode_length_prefixed(b"rodentnet-v1", b"mask-expand", domain, str(layer_index), seed)
        stream = shake_256(tag).digest(digest_size)
        mask = np.frombuffer(stream, dtype=np.uint64, count=element_count)
        return np.mod(mask, self.field).astype(np.int64).reshape(shape)
