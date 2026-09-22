from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from app.crypto.mpc import MPCSecureAggregator
from app.crypto.threshold_dilithium import (
    HandshakeTranscript,
    ThresholdDilithiumIdentity,
    b64encode,
)
from app.fl.data import load_cifar10_partition
from app.fl.model import (
    FedPerCifarNet,
    evaluate_model,
    get_shared_parameters,
    set_shared_parameters,
    train_one_epoch,
)
from app.fl.epitope_model import (
    SiameseEpitopeKANNet,
    train_one_epoch as epitope_train_one_epoch,
    evaluate_model as epitope_evaluate_model,
)
from app.fl.vdjdb_data import load_vdjdb_partition

try:
    import flwr as fl
except Exception:  # pragma: no cover - exercised on machines without Flower installed
    fl = None  # type: ignore[assignment]


BaseNumPyClient = fl.client.NumPyClient if fl is not None else object


def auth_metrics(identity: ThresholdDilithiumIdentity, transcript: HandshakeTranscript) -> dict[str, str]:
    signature = identity.sign(transcript)
    return {
        "rodentnet_auth_client_id": identity.client_id,
        "rodentnet_auth_transcript": json.dumps(transcript.to_json(), sort_keys=True),
        "rodentnet_auth_signature": b64encode(signature),
        "rodentnet_auth_algorithm": identity.provider.algorithm,
    }


class RodentNetClient(BaseNumPyClient):
    """Flower NumPyClient implementing FedPer local personalization and MPC masking.

    Supports two tasks:
    - ``task='cifar'``    (default): CIFAR-10 image classification demo.
    - ``task='epitope'``: TCR-epitope binding prediction using VDJdb + SiameseEpitopeKANNet.
    """

    def __init__(
        self,
        *,
        client_id: str,
        client_index: int,
        num_clients: int,
        data_dir: Path,
        identity: ThresholdDilithiumIdentity | None = None,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        fake_data: bool = False,
        task: str = "cifar",
        vdjdb_path: str | None = None,
    ) -> None:
        self.client_id = client_id
        self.identity = identity
        self.learning_rate = learning_rate
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.catalogue_tokens: torch.Tensor | None = None

        if task == "epitope":
            self.model = SiameseEpitopeKANNet().to(self.device)
            self.train_loader, self.test_loader, self.catalogue_tokens = load_vdjdb_partition(
                num_clients=num_clients,
                client_id=client_index,
                tsv_path=vdjdb_path,
                synthetic=(vdjdb_path is None or fake_data),
            )
        else:
            # Default: CIFAR-10 demo
            self.model = FedPerCifarNet().to(self.device)
            self.train_loader, self.test_loader = load_cifar10_partition(
                data_dir=data_dir,
                client_index=client_index,
                num_clients=num_clients,
                batch_size=batch_size,
                fake_data=fake_data,
            )

    def get_parameters(self, config: dict[str, object] | None = None) -> list[np.ndarray]:
        return get_shared_parameters(self.model)

    def fit(
        self,
        parameters: Sequence[np.ndarray],
        config: dict[str, object],
    ) -> tuple[list[np.ndarray], int, dict[str, object]]:
        set_shared_parameters(self.model, parameters)
        epochs = int(config.get("local_epochs", 1))
        loss = 0.0

        for _ in range(epochs):
            if self.task == "epitope":
                loss = epitope_train_one_epoch(
                    self.model,
                    self.train_loader,
                    device=self.device,
                    lr=self.learning_rate,
                )
            else:
                loss = train_one_epoch(
                    self.model,
                    self.train_loader,
                    device=self.device,
                    learning_rate=self.learning_rate,
                )

        shared_parameters = get_shared_parameters(self.model)
        examples = len(self.train_loader.dataset)  # type: ignore[arg-type]
        metrics: dict[str, object] = {
            "client_id": self.client_id,
            "task": self.task,
            "fedper_personal_layers": "personal_head",
            "train_loss": loss,
            "num_examples": examples,
        }

        transcript_payload = config.get("auth_transcript")
        if self.identity is not None and isinstance(transcript_payload, str):
            transcript = HandshakeTranscript.from_json(json.loads(transcript_payload))
            metrics.update(auth_metrics(self.identity, transcript))

        if "mpc_roster" in config:
            roster = json.loads(str(config["mpc_roster"]))
            round_id = str(config["mpc_round_id"])
            aggregator = MPCSecureAggregator(
                field=int(config["mpc_field"]),
                scale=int(config["mpc_scale"]),
            )
            shared_parameters = aggregator.mask_weighted_update(
                shared_parameters,
                num_examples=examples,
                client_id=self.client_id,
                roster=roster,
                round_id=round_id,
            )
            metrics["mpc_masked"] = True
            metrics["mpc_protocol"] = str(config.get("mpc_protocol", "pairwise-shamir-secagg-v1"))

        return shared_parameters, examples, metrics

    def evaluate(
        self,
        parameters: Sequence[np.ndarray],
        config: dict[str, object] | None = None,
    ) -> tuple[float, int, dict[str, object]]:
        set_shared_parameters(self.model, parameters)

        if self.task == "epitope":
            if self.catalogue_tokens is None:
                return 0.0, 0, {"error": "catalogue_tokens not loaded", "client_id": self.client_id}
            loss, accuracy = epitope_evaluate_model(
                self.model,
                self.test_loader,
                catalogue_tokens=self.catalogue_tokens,
                device=self.device,
            )
        else:
            loss, accuracy = evaluate_model(self.model, self.test_loader, device=self.device)

        examples = len(self.test_loader.dataset)  # type: ignore[arg-type]
        return loss, examples, {"accuracy": accuracy, "task": self.task, "client_id": self.client_id}


def make_flower_client(**kwargs: object):
    if fl is None:
        raise RuntimeError("Flower is not installed. Install backend dependencies first.")
    return RodentNetClient(**kwargs).to_client()
