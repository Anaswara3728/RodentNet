from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Iterable, Sequence

import numpy as np

BYTE_FIELD = 257
VECTOR_FIELD = 2_147_483_647


@dataclass(frozen=True)
class SecretShare:
    index: int
    values: tuple[int, ...]
    field: int = BYTE_FIELD

    def to_json(self) -> dict[str, int | list[int]]:
        return {"index": self.index, "values": list(self.values), "field": self.field}

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "SecretShare":
        return cls(
            index=int(payload["index"]),
            values=tuple(int(value) for value in payload["values"]),  # type: ignore[index]
            field=int(payload.get("field", BYTE_FIELD)),
        )


@dataclass(frozen=True)
class ArrayShare:
    index: int
    values: np.ndarray
    scale: int
    field: int = VECTOR_FIELD


def _validate_threshold(threshold: int, shares: int) -> None:
    if threshold < 2:
        raise ValueError("threshold must be at least 2")
    if shares < threshold:
        raise ValueError("shares must be greater than or equal to threshold")


def _eval_polynomial(coefficients: Sequence[int], x_value: int, field: int) -> int:
    total = 0
    power = 1
    for coefficient in coefficients:
        total = (total + coefficient * power) % field
        power = (power * x_value) % field
    return total


def _lagrange_at_zero(points: Sequence[tuple[int, int]], field: int) -> int:
    total = 0
    for j, (x_j, y_j) in enumerate(points):
        numerator = 1
        denominator = 1
        for m, (x_m, _) in enumerate(points):
            if m == j:
                continue
            numerator = (numerator * (-x_m)) % field
            denominator = (denominator * (x_j - x_m)) % field
        total = (total + y_j * numerator * pow(denominator, -1, field)) % field
    return total


def split_secret(secret: bytes, threshold: int, shares: int, field: int = BYTE_FIELD) -> list[SecretShare]:
    """Split arbitrary bytes with Shamir secret sharing over GF(257)."""

    _validate_threshold(threshold, shares)
    if field <= 256:
        raise ValueError("field must be greater than the maximum byte value")

    share_values: list[list[int]] = [[] for _ in range(shares)]
    for byte in secret:
        coefficients = [byte, *[secrets.randbelow(field) for _ in range(threshold - 1)]]
        for share_index in range(1, shares + 1):
            share_values[share_index - 1].append(_eval_polynomial(coefficients, share_index, field))

    return [
        SecretShare(index=share_index + 1, values=tuple(values), field=field)
        for share_index, values in enumerate(share_values)
    ]


def reconstruct_secret(shares: Sequence[SecretShare], threshold: int | None = None) -> bytes:
    if not shares:
        raise ValueError("at least one share is required")

    threshold = threshold or len(shares)
    selected = list(shares[:threshold])
    if len(selected) < threshold:
        raise ValueError("not enough shares to reconstruct the secret")

    field = selected[0].field
    value_count = len(selected[0].values)
    if any(share.field != field for share in selected):
        raise ValueError("all shares must use the same field")
    if any(len(share.values) != value_count for share in selected):
        raise ValueError("all shares must contain the same number of values")

    output = bytearray()
    for value_index in range(value_count):
        point_values = [(share.index, share.values[value_index]) for share in selected]
        byte_value = _lagrange_at_zero(point_values, field)
        if byte_value > 255:
            raise ValueError("reconstructed value is outside byte range")
        output.append(byte_value)
    return bytes(output)


def encode_signed_array(values: np.ndarray, scale: int = 1_000_000, field: int = VECTOR_FIELD) -> np.ndarray:
    scaled = np.rint(values.astype(np.float64) * scale).astype(np.int64)
    return np.mod(scaled, field).astype(np.int64)


def decode_signed_array(values: np.ndarray, scale: int = 1_000_000, field: int = VECTOR_FIELD) -> np.ndarray:
    residues = np.mod(np.rint(values).astype(np.int64), field)
    signed = np.where(residues > field // 2, residues - field, residues)
    return signed.astype(np.float64) / scale


def split_array(
    values: np.ndarray,
    threshold: int,
    shares: int,
    *,
    scale: int = 1_000_000,
    field: int = VECTOR_FIELD,
) -> list[ArrayShare]:
    """Shamir-share a floating point tensor after fixed-point quantization."""

    _validate_threshold(threshold, shares)
    encoded = encode_signed_array(values, scale=scale, field=field).reshape(-1)
    share_values = [np.zeros(encoded.shape, dtype=np.int64) for _ in range(shares)]

    for value_index, encoded_value in enumerate(encoded.tolist()):
        coefficients = [encoded_value, *[secrets.randbelow(field) for _ in range(threshold - 1)]]
        for share_index in range(1, shares + 1):
            share_values[share_index - 1][value_index] = _eval_polynomial(
                coefficients, share_index, field
            )

    return [
        ArrayShare(index=share_index + 1, values=share.reshape(values.shape), scale=scale, field=field)
        for share_index, share in enumerate(share_values)
    ]


def reconstruct_array(shares: Sequence[ArrayShare], threshold: int | None = None) -> np.ndarray:
    if not shares:
        raise ValueError("at least one share is required")

    threshold = threshold or len(shares)
    selected = list(shares[:threshold])
    if len(selected) < threshold:
        raise ValueError("not enough shares to reconstruct the array")

    field = selected[0].field
    scale = selected[0].scale
    shape = selected[0].values.shape
    if any(share.field != field or share.scale != scale for share in selected):
        raise ValueError("all shares must use the same field and scale")
    if any(share.values.shape != shape for share in selected):
        raise ValueError("all shares must have the same shape")

    flat_shares = [np.rint(share.values.reshape(-1)).astype(np.int64) for share in selected]
    reconstructed = np.zeros(flat_shares[0].shape, dtype=np.int64)
    for value_index in range(reconstructed.size):
        points = [(share.index, int(flat[value_index])) for share, flat in zip(selected, flat_shares)]
        reconstructed[value_index] = _lagrange_at_zero(points, field)

    return decode_signed_array(reconstructed.reshape(shape), scale=scale, field=field)


def sum_share_vectors(shares_by_client: Iterable[Sequence[ArrayShare]], field: int = VECTOR_FIELD) -> list[ArrayShare]:
    """Sum same-index Shamir shares so only the aggregate polynomial can be opened."""

    client_shares = [list(client_share_set) for client_share_set in shares_by_client]
    if not client_shares:
        return []

    share_count = len(client_shares[0])
    summed: list[ArrayShare] = []
    for share_position in range(share_count):
        matching = [client_share_set[share_position] for client_share_set in client_shares]
        first = matching[0]
        total = np.zeros_like(first.values, dtype=np.int64)
        for share in matching:
            if share.index != first.index:
                raise ValueError("share indexes must align before summing")
            total = np.mod(total + np.rint(share.values).astype(np.int64), field)
        summed.append(ArrayShare(index=first.index, values=total, scale=first.scale, field=first.field))
    return summed
