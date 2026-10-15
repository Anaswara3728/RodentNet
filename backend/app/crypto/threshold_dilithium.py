from __future__ import annotations

from dataclasses import dataclass, field
import base64
import hmac
import json
import secrets
import time
from hashlib import sha3_512
from typing import Protocol, Sequence

from app.crypto.shamir import SecretShare, reconstruct_secret, split_secret


class SignatureProvider(Protocol):
    algorithm: str

    def generate_keypair(self) -> tuple[bytes, bytes]:
        ...

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        ...

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        ...


class DilithiumPyProvider:
    """Genuine NIST ML-DSA-65 / Dilithium3 provider using dilithium-py."""

    algorithm = "ML-DSA-65"

    def __init__(self) -> None:
        from dilithium_py.dilithium import Dilithium3
        self._dilithium = Dilithium3

    def generate_keypair(self) -> tuple[bytes, bytes]:
        pk, sk = self._dilithium.keygen()
        return bytes(pk), bytes(sk)

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        return bytes(self._dilithium.sign(secret_key, message))

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        try:
            return bool(self._dilithium.verify(public_key, message, signature))
        except Exception:
            return False


class PqcryptoMLDSA65Provider:
    """Adapter for pqcrypto ML-DSA-65 or its Dilithium3 predecessor name."""

    algorithm = "ML-DSA-65"

    def __init__(self) -> None:
        try:
            from pqcrypto.sign import ml_dsa_65 as module  # type: ignore
        except Exception:
            from pqcrypto.sign import dilithium3 as module  # type: ignore

        self._module = module

    def generate_keypair(self) -> tuple[bytes, bytes]:
        public_key, secret_key = self._module.generate_keypair()
        return bytes(public_key), bytes(secret_key)

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        return bytes(self._module.sign(message, secret_key))

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        try:
            verified = self._module.verify(signature, message, public_key)
            return verified is None or bool(verified)
        except TypeError:
            opened = self._module.open(signature, public_key)
            return bytes(opened) == message
        except Exception:
            return False


class DevMLDSA65Provider:
    """Deterministic local-only stand-in for development machines without PQC bindings.

    This class keeps the handshake and threshold reconstruction path runnable in CI and on
    laptops. Deployments should install `dilithium-py` or `pqcrypto` to use genuine ML-DSA-65.
    """

    algorithm = "ML-DSA-65-DEV-HMAC"

    def generate_keypair(self) -> tuple[bytes, bytes]:
        secret_key = secrets.token_bytes(64)
        return secret_key, secret_key

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        return hmac.new(secret_key, message, sha3_512).digest()

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        expected = hmac.new(public_key, message, sha3_512).digest()
        return hmac.compare_digest(signature, expected)


def load_signature_provider(*, allow_dev_fallback: bool = True) -> SignatureProvider:
    try:
        return DilithiumPyProvider()
    except Exception:
        pass

    try:
        return PqcryptoMLDSA65Provider()
    except Exception:
        pass

    if not allow_dev_fallback:
        raise RuntimeError("No genuine PQC signature provider available and dev fallback disallowed")
    return DevMLDSA65Provider()


def b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))


@dataclass(frozen=True)
class HandshakeTranscript:
    client_id: str
    nonce: str
    round_id: str
    model_digest: str
    issued_at: int = field(default_factory=lambda: int(time.time()))
    role: str = "hospital-client"
    protocol: str = "flower-handshake-v1"
    algorithm: str = "ML-DSA-65"

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def to_json(self) -> dict[str, str | int]:
        return {
            "algorithm": self.algorithm,
            "client_id": self.client_id,
            "issued_at": self.issued_at,
            "model_digest": self.model_digest,
            "nonce": self.nonce,
            "protocol": self.protocol,
            "role": self.role,
            "round_id": self.round_id,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "HandshakeTranscript":
        return cls(
            client_id=str(payload["client_id"]),
            nonce=str(payload["nonce"]),
            round_id=str(payload["round_id"]),
            model_digest=str(payload["model_digest"]),
            issued_at=int(payload["issued_at"]),
            role=str(payload.get("role", "hospital-client")),
            protocol=str(payload.get("protocol", "flower-handshake-v1")),
            algorithm=str(payload.get("algorithm", "ML-DSA-65")),
        )


@dataclass
class ThresholdDilithiumIdentity:
    client_id: str
    public_key: bytes
    key_shares: list[SecretShare]
    threshold: int
    provider: SignatureProvider

    @classmethod
    def create(
        cls,
        *,
        client_id: str,
        threshold: int = 2,
        shares: int = 3,
        provider: SignatureProvider | None = None,
    ) -> "ThresholdDilithiumIdentity":
        provider = provider or load_signature_provider()
        public_key, secret_key = provider.generate_keypair()
        key_shares = split_secret(secret_key, threshold=threshold, shares=shares)
        return cls(
            client_id=client_id,
            public_key=public_key,
            key_shares=key_shares,
            threshold=threshold,
            provider=provider,
        )

    def sign(self, transcript: HandshakeTranscript, share_ids: Sequence[int] | None = None) -> bytes:
        selected = self._select_shares(share_ids)
        secret_key = reconstruct_secret(selected, threshold=self.threshold)
        return self.provider.sign(transcript.canonical_bytes(), secret_key)

    def public_key_b64(self) -> str:
        return b64encode(self.public_key)

    def shares_json(self) -> list[dict[str, int | list[int]]]:
        return [share.to_json() for share in self.key_shares]

    def _select_shares(self, share_ids: Sequence[int] | None) -> list[SecretShare]:
        if share_ids is None:
            return self.key_shares[: self.threshold]
        wanted = set(share_ids)
        selected = [share for share in self.key_shares if share.index in wanted]
        if len(selected) < self.threshold:
            raise ValueError("not enough threshold shares selected")
        return selected[: self.threshold]


@dataclass(frozen=True)
class RegisteredClient:
    client_id: str
    public_key: bytes
    display_name: str
    threshold: int
    algorithm: str


class IdentityRegistry:
    """In-memory authority for PQC client admission and Flower handshake checks."""

    def __init__(self, provider: SignatureProvider | None = None) -> None:
        self.provider = provider or load_signature_provider()
        self._clients: dict[str, RegisteredClient] = {}
        self._challenges: dict[str, HandshakeTranscript] = {}

    def register_client(
        self,
        *,
        client_id: str,
        public_key: bytes,
        display_name: str,
        threshold: int,
        algorithm: str | None = None,
    ) -> RegisteredClient:
        registered = RegisteredClient(
            client_id=client_id,
            public_key=public_key,
            display_name=display_name,
            threshold=threshold,
            algorithm=algorithm or self.provider.algorithm,
        )
        self._clients[client_id] = registered
        return registered

    def issue_challenge(
        self,
        *,
        client_id: str,
        round_id: str,
        model_digest: str,
    ) -> HandshakeTranscript:
        if client_id not in self._clients:
            raise KeyError(f"client is not enrolled: {client_id}")
        transcript = HandshakeTranscript(
            client_id=client_id,
            nonce=b64encode(secrets.token_bytes(24)),
            round_id=round_id,
            model_digest=model_digest,
            algorithm=self._clients[client_id].algorithm,
        )
        self._challenges[transcript.nonce] = transcript
        return transcript

    def verify(
        self,
        *,
        transcript: HandshakeTranscript,
        signature: bytes,
        consume: bool = True,
    ) -> bool:
        registered = self._clients.get(transcript.client_id)
        if registered is None:
            return False
        issued = self._challenges.get(transcript.nonce)
        if issued is None or issued.to_json() != transcript.to_json():
            return False
        if not self.provider.verify(transcript.canonical_bytes(), signature, registered.public_key):
            return False
        if consume:
            self._challenges.pop(transcript.nonce, None)
        return True

    def verify_metrics(self, metrics: dict[str, object], *, consume: bool = True) -> bool:
        try:
            transcript = HandshakeTranscript.from_json(
                json.loads(str(metrics["rodentnet_auth_transcript"]))
            )
            signature = b64decode(str(metrics["rodentnet_auth_signature"]))
        except Exception:
            return False
        return self.verify(transcript=transcript, signature=signature, consume=consume)

    def clients(self) -> list[RegisteredClient]:
        return sorted(self._clients.values(), key=lambda client: client.client_id)
