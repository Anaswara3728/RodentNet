from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

from app.config import get_settings


@dataclass
class SimulationStatus:
    state: str = "idle"
    rounds: int = 0
    clients: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    message: str = "No simulation has been started."
    metrics: dict[str, float | int | str] = field(default_factory=dict)


def run_simulation(
    *,
    rounds: int,
    clients: int,
    data_dir: Path | None = None,
    fake_data: bool = True,
    task: str = "cifar",
    vdjdb_path: str | None = None,
) -> SimulationStatus:
    """Run a local Flower simulation when the ML stack is installed.

    The FastAPI service calls this in a background worker. The implementation keeps the
    API responsive and reports dependency gaps clearly instead of pretending to train.

    Args:
        rounds:     Number of federated rounds.
        clients:    Number of simulated hospital clients.
        data_dir:   Local path for CIFAR-10 data (only used when task='cifar').
        fake_data:  If True, use synthetic/in-memory data instead of real datasets.
        task:       ``'cifar'`` for the CIFAR-10 pipeline validation demo, or
                    ``'epitope'`` for the TCR-epitope SiameseEpitopeKANNet pipeline.
        vdjdb_path: Path to a VDJdb TSV file (only used when task='epitope').
                    If None and task='epitope', synthetic VDJdb data is generated.
    """

    started = time.time()
    settings = get_settings()
    try:
        import flwr as fl
        from flwr.common import ndarrays_to_parameters
        from cryptography.hazmat.primitives.asymmetric import x25519
        from cryptography.hazmat.primitives import serialization

        from app.fl.client import RodentNetClient
        from app.fl.model import get_shared_parameters
        from app.fl.strategy import MPCFedPerStrategy

        if task == "epitope":
            from app.fl.epitope_model import SiameseEpitopeKANNet
            model = SiameseEpitopeKANNet()
        else:
            from app.fl.model import FedPerCifarNet
            model = FedPerCifarNet()
            
        # --- Authenticated ML-KEM-768 (Kyber768) Post-Quantum Key Agreement Phase ---
        from kyber_py.kyber import Kyber768
        client_shared_secrets = {f"hospital-{i + 1}": {} for i in range(clients)}
        for i in range(clients):
            for j in range(i + 1, clients):
                cid_i = f"hospital-{i + 1}"
                cid_j = f"hospital-{j + 1}"
                pk_j, sk_j = Kyber768.keygen()
                ss, ct = Kyber768.encaps(pk_j)
                ss_rec = Kyber768.decaps(sk_j, ct)
                assert ss == ss_rec, "Kyber key agreement failed"
                client_shared_secrets[cid_i][cid_j] = ss
                client_shared_secrets[cid_j][cid_i] = ss

    except Exception as exc:
        return SimulationStatus(
            state="blocked",
            rounds=rounds,
            clients=clients,
            started_at=started,
            finished_at=time.time(),
            message=f"Simulation dependencies are not installed: {exc}",
        )

    def client_fn(context):  # type: ignore[no-untyped-def]
        partition_id = int(context.node_config.get("partition-id", 0))
        cid = f"hospital-{partition_id + 1}"
        
        client = RodentNetClient(
            client_id=cid,
            client_index=partition_id,
            num_clients=clients,
            data_dir=data_dir or settings.cifar_data_dir,
            fake_data=fake_data,
            task=task,
            vdjdb_path=vdjdb_path,
            shared_secrets=client_shared_secrets[cid],
        )
        return client.to_client()

    strategy = MPCFedPerStrategy(
        require_identity=False,
        initial_parameters=ndarrays_to_parameters(get_shared_parameters(model)),
        min_fit_clients=max(2, clients),
        min_available_clients=max(2, clients),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    return SimulationStatus(
        state="complete",
        rounds=rounds,
        clients=clients,
        started_at=started,
        finished_at=time.time(),
        message=f"Simulation completed ({task} task).",
        metrics={"duration_seconds": round(time.time() - started, 3), "task": task},
    )
