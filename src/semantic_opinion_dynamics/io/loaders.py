from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from semantic_opinion_dynamics.contracts import NetworkSpec, PersonasSpec, RunConfig


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_run_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(_read_yaml(path))


def load_personas_yaml(path: Path) -> PersonasSpec:
    return PersonasSpec.model_validate(_read_yaml(path))


def load_network_json(path: Path) -> NetworkSpec:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return NetworkSpec.model_validate(data)


def dump_yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
