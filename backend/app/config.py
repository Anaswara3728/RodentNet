from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by the API, Flower server, and simulation code."""

    model_config = SettingsConfigDict(env_prefix="RODENTNET_", env_file=".env", extra="ignore")

    api_title: str = "RodentNet Privacy FL Gateway"
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    runtime_dir: Path = Path("runtime")
    flower_address: str = "0.0.0.0:8080"
    default_rounds: int = 3
    default_clients: int = 3
    default_threshold: int = 2
    mpc_field: int = 2_147_483_647
    mpc_scale: int = 1_000_000
    cifar_data_dir: Path = Path("data/cifar10")
    allow_dev_pqc_fallback: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
