from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    project_dir = config_path.parent.parent
    if config_path.parent.name in {"prep", "training", "inference"}:
        project_dir = config_path.parent.parent.parent
    return config, project_dir


def resolve_path(value: str | Path, project_dir: Path) -> Path:
    raw = str(value)
    if raw.startswith("~projects/"):
        raw = f"~/{raw[1:]}"
    path = Path(os.path.expanduser(raw))
    return path if path.is_absolute() else (project_dir / path).resolve()


def require_list(config: dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = config.get(key, default)
    if isinstance(value, (str, int)):
        return [str(value)]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a value or YAML list")
    return [str(item) for item in value]


def selector(config: dict[str, Any], key: str, default: str = "all") -> list[str]:
    values = [item.strip() for item in require_list(config, key, [default])]
    if not values or any(not item for item in values):
        raise ValueError(f"{key} must not be empty")
    if len(values) > 1 and any(item.casefold() == "all" for item in values):
        raise ValueError(f"{key}: 'all' must be used by itself")
    if len({item.casefold() for item in values}) != len(values):
        raise ValueError(f"{key} contains duplicate selectors")
    return values


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M-%S")


def model_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

