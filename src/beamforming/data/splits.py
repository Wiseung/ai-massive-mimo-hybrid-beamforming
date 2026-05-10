"""Reusable dataset split helpers for reproducible experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset


def make_dataset_split(
    num_samples: int,
    *,
    split_mode: str,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict[str, Any]:
    """Build train/val/test indices for random or contiguous splits."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    indices = torch.arange(num_samples, dtype=torch.long)
    if split_mode == "random":
        generator = torch.Generator().manual_seed(seed)
        indices = indices[torch.randperm(num_samples, generator=generator)]
    elif split_mode != "contiguous":
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    train_end = max(1, int(num_samples * train_fraction))
    val_end = max(train_end + 1, int(num_samples * (train_fraction + val_fraction)))
    val_end = min(val_end, num_samples)
    if val_end >= num_samples:
        val_end = num_samples - 1
    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    if not val_indices:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1]
    if not test_indices:
        test_indices = val_indices[-1:]
        val_indices = val_indices[:-1]
    return {
        "split_mode": split_mode,
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "test_fraction": float(1.0 - train_fraction - val_fraction),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def save_dataset_split(
    path: str | Path,
    split_payload: dict[str, Any],
    *,
    dataset_metadata: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**split_payload, "dataset_metadata": dataset_metadata or {}}
    torch.save(payload, out_path)
    return out_path


def load_dataset_split(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"train_indices", "val_indices", "test_indices"}
    if not required.issubset(payload):
        raise ValueError(f"Split file {path} is missing required keys: {sorted(required - set(payload))}")
    return payload


def subset_from_split(dataset, split_payload: dict[str, Any], split_name: str) -> Subset:
    key = f"{split_name}_indices"
    if key not in split_payload:
        raise KeyError(f"Split payload does not contain {key}")
    return Subset(dataset, list(split_payload[key]))


def split_counts(split_payload: dict[str, Any]) -> dict[str, int]:
    return {
        "train": len(split_payload["train_indices"]),
        "val": len(split_payload["val_indices"]),
        "test": len(split_payload["test_indices"]),
    }
