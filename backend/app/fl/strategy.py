from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from app.crypto.mpc import MPCSecureAggregator
from app.crypto.threshold_dilithium import IdentityRegistry

try:
    from flwr.common import FitIns, Metrics, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.client_manager import ClientManager
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg

    FLOWER_AVAILABLE = True
except Exception:  # pragma: no cover - imported by lightweight tests without Flower
    FitIns = Metrics = Parameters = Scalar = ClientManager = ClientProxy = Any  # type: ignore
    FLOWER_AVAILABLE = False

    class FedAvg:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.initial_parameters = kwargs.get("initial_parameters")

        def configure_fit(self, *args: object, **kwargs: object) -> list[object]:
            return []


@dataclass(frozen=True)
class AggregationAudit:
    accepted_clients: int
    rejected_clients: int
    total_examples: int
    protocol: str

    def as_metrics(self) -> dict[str, bool | int | str]:
        return {
            "accepted_clients": self.accepted_clients,
            "rejected_clients": self.rejected_clients,
            "total_examples": self.total_examples,
            "mpc_protocol": self.protocol,
            "identity_verified": self.rejected_clients == 0,
        }


class MPCFedPerStrategy(FedAvg):
    """Flower FedPer strategy with threshold-authenticated secure aggregation.

    Clients keep `personal_head` local and send only shared tensors. Submitted tensors are
    fixed-point, sample-weighted, and pairwise masked; this strategy sums the masks away and
    decodes the global shared representation without reading individual hospital updates.
    """

    def __init__(
        self,
        *,
        mpc: MPCSecureAggregator | None = None,
        identity_registry: IdentityRegistry | None = None,
        require_identity: bool = True,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        **fedavg_kwargs: object,
    ) -> None:
        self.mpc = mpc or MPCSecureAggregator()
        self.identity_registry = identity_registry
        self.require_identity = require_identity
        self._user_fit_config = on_fit_config_fn

        def fit_config(server_round: int) -> dict[str, Scalar]:
            config: dict[str, Scalar] = {}
            if self._user_fit_config is not None:
                config.update(self._user_fit_config(server_round))
            config["local_epochs"] = int(config.get("local_epochs", 1))
            return config

        super().__init__(on_fit_config_fn=fit_config, **fedavg_kwargs)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        instructions = super().configure_fit(server_round, parameters, client_manager)
        roster = [client.cid for client, _ in instructions]
        round_id = f"round-{server_round}"
        secure_config = self.mpc.config_payload(roster=roster, round_id=round_id)
        for client, fit_ins in instructions:
            fit_ins.config.update(secure_config)
            fit_ins.config["client_id"] = client.cid
        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, Any]],
        failures: list[BaseException] | list[tuple[ClientProxy, BaseException]],
    ) -> tuple[Parameters | None, Metrics]:
        if not results:
            return None, {"accepted_clients": 0, "rejected_clients": 0, "failures": len(failures)}

        accepted_updates: list[list[np.ndarray]] = []
        accepted_examples = 0
        rejected = 0

        for _, fit_res in results:
            metrics = dict(getattr(fit_res, "metrics", {}) or {})
            if self.require_identity and self.identity_registry is not None:
                if not self.identity_registry.verify_metrics(metrics):
                    rejected += 1
                    continue

            accepted_updates.append(parameters_to_ndarrays(fit_res.parameters))
            accepted_examples += int(fit_res.num_examples)

        if not accepted_updates or accepted_examples <= 0:
            return None, {
                "accepted_clients": 0,
                "rejected_clients": rejected,
                "failures": len(failures),
                "mpc_protocol": "pairwise-shamir-secagg-v1",
            }

        aggregate = self.mpc.aggregate_masked_updates(
            accepted_updates,
            total_examples=accepted_examples,
        )
        audit = AggregationAudit(
            accepted_clients=len(accepted_updates),
            rejected_clients=rejected,
            total_examples=accepted_examples,
            protocol="pairwise-shamir-secagg-v1",
        )
        return ndarrays_to_parameters(aggregate), audit.as_metrics()
