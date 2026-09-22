from __future__ import annotations

from app.config import get_settings
from app.crypto.threshold_dilithium import IdentityRegistry
from app.fl.model import FedPerCifarNet, get_shared_parameters
from app.fl.strategy import MPCFedPerStrategy

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters
except Exception:  # pragma: no cover
    fl = None  # type: ignore[assignment]
    ndarrays_to_parameters = None  # type: ignore[assignment]


def start_flower_server(
    *,
    identity_registry: IdentityRegistry,
    rounds: int | None = None,
    address: str | None = None,
    require_identity: bool = True,
) -> None:
    if fl is None or ndarrays_to_parameters is None:
        raise RuntimeError("Flower is not installed. Install backend dependencies first.")

    settings = get_settings()
    model = FedPerCifarNet()
    strategy = MPCFedPerStrategy(
        identity_registry=identity_registry,
        require_identity=require_identity,
        initial_parameters=ndarrays_to_parameters(get_shared_parameters(model)),
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )
    fl.server.start_server(
        server_address=address or settings.flower_address,
        config=fl.server.ServerConfig(num_rounds=rounds or settings.default_rounds),
        strategy=strategy,
    )
