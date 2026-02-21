from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from semantic_opinion_dynamics.contracts import RunConfig, RunMetadata, StepLog
from semantic_opinion_dynamics.io.loaders import dump_yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)[:60]
    return f"{ts}_{safe}" if safe else ts


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


@dataclass(frozen=True)
class RunPaths:
    root: Path
    steps_dir: Path

    meta_path: Path
    config_used_path: Path
    network_used_path: Path
    personas_used_path: Path


def init_run_dir(output_dir: Path, run_id: str) -> RunPaths:
    root = output_dir / run_id
    steps_dir = root / "steps"
    steps_dir.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        root=root,
        steps_dir=steps_dir,
        meta_path=root / "run_meta.json",
        config_used_path=root / "config_used.yaml",
        network_used_path=root / "network_used.json",
        personas_used_path=root / "personas_used.yaml",
    )


def save_run_metadata(paths: RunPaths, meta: RunMetadata) -> None:
    _atomic_write_json(paths.meta_path, meta.model_dump(mode="json"))


def save_inputs_snapshot(
    paths: RunPaths,
    cfg: RunConfig,
    network_dict: Dict[str, Any],
    personas_dict: Dict[str, Any],
) -> None:
    _atomic_write_text(paths.config_used_path, dump_yaml(cfg.model_dump(mode="json")))
    _atomic_write_json(paths.network_used_path, network_dict)
    _atomic_write_text(paths.personas_used_path, dump_yaml(personas_dict))


def save_step(paths: RunPaths, step: StepLog) -> Path:
    out = paths.steps_dir / f"step_{step.t:04d}.json"
    _atomic_write_json(out, step.model_dump(mode="json"))
    return out
