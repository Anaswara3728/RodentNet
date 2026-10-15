from __future__ import annotations

import json
from threading import Thread

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.state import state
from app.config import get_settings
from app.crypto.threshold_dilithium import (
    HandshakeTranscript,
    ThresholdDilithiumIdentity,
    b64decode,
    b64encode,
)
from app.crypto.shamir import SecretShare, reconstruct_secret
from app.fl.simulation import run_simulation

router = APIRouter(prefix="/api")


class EnrollClientRequest(BaseModel):
    client_id: str = Field(min_length=3)
    display_name: str = Field(min_length=1)
    public_key_b64: str
    threshold: int = Field(default=2, ge=2)
    algorithm: str | None = None


class ChallengeRequest(BaseModel):
    client_id: str
    round_id: str = "round-0"
    model_digest: str = "bootstrap"


class VerifyHandshakeRequest(BaseModel):
    transcript: dict[str, str | int]
    signature_b64: str


class DevSignRequest(BaseModel):
    transcript: dict[str, str | int]
    shares: list[dict[str, int | list[int]]]
    threshold: int = Field(default=2, ge=2)


class DevIdentityRequest(BaseModel):
    client_id: str = "hospital-1"
    threshold: int = Field(default=2, ge=2)
    shares: int = Field(default=3, ge=2)
    display_name: str = "Demo Hospital"


class SimulationRequest(BaseModel):
    rounds: int = Field(default=1, ge=1, le=50)
    clients: int = Field(default=3, ge=2, le=20)
    fake_data: bool = True
    task: str = Field(default="cifar", pattern="^(cifar|epitope)$")
    vdjdb_path: str | None = None


@router.get("/status")
def status() -> dict[str, object]:
    settings = get_settings()
    return {
        "service": "rodentnet",
        "pqc_provider": state.registry.provider.algorithm,
        "registered_clients": [client.__dict__ | {"public_key": b64encode(client.public_key)} for client in state.registry.clients()],
        "simulation": state.simulation.__dict__,
        "audit": state.audit_json(),
        "defaults": {
            "rounds": settings.default_rounds,
            "clients": settings.default_clients,
            "threshold": settings.default_threshold,
            "mpc_field": settings.mpc_field,
            "mpc_scale": settings.mpc_scale,
        },
    }


@router.post("/clients/enroll")
def enroll_client(request: EnrollClientRequest) -> dict[str, object]:
    try:
        public_key = b64decode(request.public_key_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="public_key_b64 is not valid base64") from exc

    registered = state.registry.register_client(
        client_id=request.client_id,
        display_name=request.display_name,
        public_key=public_key,
        threshold=request.threshold,
        algorithm=request.algorithm,
    )
    state.audit("client.enrolled", f"{registered.client_id} enrolled with {registered.algorithm}")
    return {
        "client_id": registered.client_id,
        "display_name": registered.display_name,
        "threshold": registered.threshold,
        "algorithm": registered.algorithm,
    }


@router.post("/dev/identity")
def create_dev_identity(request: DevIdentityRequest) -> dict[str, object]:
    """DEV SIMULATION HELPER ONLY.
    
    WARNING: In production federated deployment, hospital clients generate their own
    Dilithium3 keypairs strictly on-premise and enroll solely via POST /clients/enroll.
    This route exists solely for single-host browser UI demonstrations.
    """
    if request.shares < request.threshold:
        raise HTTPException(status_code=400, detail="shares must be greater than or equal to threshold")

    identity = ThresholdDilithiumIdentity.create(
        client_id=request.client_id,
        threshold=request.threshold,
        shares=request.shares,
        provider=state.registry.provider,
    )
    state.registry.register_client(
        client_id=request.client_id,
        display_name=request.display_name,
        public_key=identity.public_key,
        threshold=request.threshold,
        algorithm=identity.provider.algorithm,
    )
    state.audit("client.dev_identity", f"{request.client_id} development identity generated")
    return {
        "client_id": request.client_id,
        "public_key_b64": identity.public_key_b64(),
        "threshold": request.threshold,
        "shares": identity.shares_json(),
        "algorithm": identity.provider.algorithm,
    }


@router.post("/handshake/challenge")
def issue_challenge(request: ChallengeRequest) -> dict[str, object]:
    try:
        transcript = state.registry.issue_challenge(
            client_id=request.client_id,
            round_id=request.round_id,
            model_digest=request.model_digest,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    state.audit("handshake.challenge", f"{request.client_id} challenge for {request.round_id}")
    return {
        "transcript": transcript.to_json(),
        "canonical": transcript.canonical_bytes().decode("utf-8"),
    }


@router.post("/handshake/verify")
def verify_handshake(request: VerifyHandshakeRequest) -> dict[str, object]:
    try:
        transcript = HandshakeTranscript.from_json(dict(request.transcript))
        signature = b64decode(request.signature_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid transcript or signature") from exc

    accepted = state.registry.verify(transcript=transcript, signature=signature)
    state.audit(
        "handshake.accepted" if accepted else "handshake.rejected",
        f"{transcript.client_id} for {transcript.round_id}",
    )
    return {"accepted": accepted, "client_id": transcript.client_id, "round_id": transcript.round_id}


@router.post("/dev/sign")
def dev_sign(request: DevSignRequest) -> dict[str, object]:
    """DEV SIMULATION HELPER ONLY.
    
    WARNING: In production federated deployment, hospital clients execute threshold
    reconstruction and Dilithium3 signing strictly on-device. The central aggregation
    server never receives shares or private keys. This route exists solely for
    single-host browser UI demonstrations.
    """
    try:
        transcript = HandshakeTranscript.from_json(dict(request.transcript))
        shares = [SecretShare.from_json(share) for share in request.shares[: request.threshold]]
        secret_key = reconstruct_secret(shares, threshold=request.threshold)
        signature = state.registry.provider.sign(transcript.canonical_bytes(), secret_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not threshold-sign transcript: {exc}") from exc

    state.audit("client.dev_sign", f"{transcript.client_id} signed {transcript.round_id}")
    return {"signature_b64": b64encode(signature), "algorithm": state.registry.provider.algorithm}


@router.post("/simulation/start")
def start_simulation(request: SimulationRequest) -> dict[str, object]:
    if state.simulation.state == "running":
        raise HTTPException(status_code=409, detail="simulation already running")

    state.simulation.state = "running"
    state.simulation.rounds = request.rounds
    state.simulation.clients = request.clients
    state.simulation.message = "Simulation is running."
    state.audit("simulation.started", f"{request.clients} clients for {request.rounds} rounds (task={request.task})")

    def runner() -> None:
        state.simulation = run_simulation(
            rounds=request.rounds,
            clients=request.clients,
            fake_data=request.fake_data,
            task=request.task,
            vdjdb_path=request.vdjdb_path,
        )
        state.audit("simulation.finished", state.simulation.message)

    Thread(target=runner, daemon=True).start()
    return {"state": "running", "request": json.loads(request.model_dump_json())}
