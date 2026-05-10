"""Unified dataset wrappers for synthetic and loaded channel tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from beamforming.metrics.sum_rate import noise_variance_from_snr
from beamforming.utils.complex_ops import complex_to_real


@dataclass
class ChannelDatasetConfig:
    snr_db: list[float]
    num_users: int
    num_bs_ant: int
    num_rf_chains: int
    num_samples: int
    metadata: dict[str, Any]


class ChannelDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for narrowband or wideband channel tensors."""

    def __init__(self, channels: torch.Tensor, snr_db: torch.Tensor | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.channels = channels.to(torch.complex64)
        self.snr_db = snr_db.float() if snr_db is not None else torch.zeros(channels.size(0), dtype=torch.float32)
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return self.channels.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        channel = self.channels[index]
        snr_db = self.snr_db[index]
        return {
            "channel": channel,
            "channel_real": complex_to_real(channel.unsqueeze(0)).squeeze(0),
            "snr_db": snr_db,
            "noise_var": noise_variance_from_snr(snr_db),
            "index": torch.tensor(index, dtype=torch.long),
        }


def save_channel_dataset(
    path: str | Path,
    channels: torch.Tensor,
    snr_db: list[float],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "channels": channels.cpu(),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "metadata": metadata,
        },
        path,
    )


def load_channel_dataset(path: str | Path) -> ChannelDataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    channels = payload["channels"]
    metadata = payload.get("metadata", {})
    snr_values = payload.get("snr_db", torch.zeros(channels.size(0), dtype=torch.float32))
    if snr_values.ndim == 1 and snr_values.numel() != channels.size(0):
        repeats = (channels.size(0) + snr_values.numel() - 1) // snr_values.numel()
        snr_values = snr_values.repeat(repeats)[: channels.size(0)]
    return ChannelDataset(channels=channels, snr_db=snr_values, metadata=metadata)


def split_dataset(dataset: ChannelDataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    """Create deterministic train/validation splits from dataset indices."""
    total_len = len(dataset)
    val_len = max(1, int(total_len * val_fraction))
    train_len = total_len - val_len
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_len, generator=generator)
    train_indices = perm[:train_len].tolist()
    val_indices = perm[train_len:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
