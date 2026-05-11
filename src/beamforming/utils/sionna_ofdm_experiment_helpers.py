"""Helpers for optional Sionna OFDM experiment orchestration."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def apply_quick_overrides(
    config: dict[str, Any],
    *,
    epochs: int = 5,
    batch_size: int = 32,
    num_batches_per_epoch: int = 10,
    num_val_batches: int = 2,
) -> dict[str, Any]:
    payload = clone_config(config)
    payload.setdefault("training", {})
    payload.setdefault("dataset", {})
    payload["training"]["epochs"] = min(int(payload["training"].get("epochs", epochs)), epochs)
    payload["dataset"]["batch_size"] = min(int(payload["dataset"].get("batch_size", batch_size)), batch_size)
    payload["dataset"]["num_batches_per_epoch"] = min(
        int(payload["dataset"].get("num_batches_per_epoch", num_batches_per_epoch)),
        num_batches_per_epoch,
    )
    payload["dataset"]["num_val_batches"] = min(
        int(payload["dataset"].get("num_val_batches", num_val_batches)),
        num_val_batches,
    )
    return payload


def override_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    payload = clone_config(config)
    payload.setdefault("training", {})
    payload["training"]["seed"] = int(seed)
    return payload


def update_nested_dict(payload: dict[str, Any], updates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = clone_config(payload)
    for section, section_updates in updates.items():
        result.setdefault(section, {})
        result[section].update(section_updates)
    return result


def run_python_command(args: list[str], cwd: str | Path) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=str(cwd))

