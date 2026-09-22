from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

from app.config import get_settings
from app.crypto.threshold_dilithium import IdentityRegistry, load_signature_provider
from app.fl.simulation import SimulationStatus


@dataclass
class AuditEvent:
    event: str
    detail: str
    created_at: float = field(default_factory=time.time)


class RodentNetState:
    def __init__(self) -> None:
        settings = get_settings()
        provider = load_signature_provider(allow_dev_fallback=settings.allow_dev_pqc_fallback)
        self.registry = IdentityRegistry(provider=provider)
        self.simulation = SimulationStatus()
        self.audit_events: list[AuditEvent] = []

    def audit(self, event: str, detail: str) -> None:
        self.audit_events.insert(0, AuditEvent(event=event, detail=detail))
        del self.audit_events[50:]

    def audit_json(self) -> list[dict[str, float | str]]:
        return [asdict(event) for event in self.audit_events]


state = RodentNetState()
