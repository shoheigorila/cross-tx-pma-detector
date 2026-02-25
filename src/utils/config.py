"""Configuration loader for the project."""

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_key = obj[2:-1]
        return os.environ.get(env_key, "")
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def load_config(chain: str = "ethereum") -> dict:
    """Load merged config: default.yaml + chain-specific yaml."""
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    chain_path = CONFIG_DIR / "chains" / f"{chain}.yaml"
    if chain_path.exists():
        with open(chain_path) as f:
            chain_cfg = yaml.safe_load(f)
        cfg["chain"] = chain_cfg.get("chain", {})
        cfg["known_contracts"] = chain_cfg.get("known_contracts", {})
        cfg["events"] = chain_cfg.get("events", {})

    return _resolve_env_vars(cfg)
