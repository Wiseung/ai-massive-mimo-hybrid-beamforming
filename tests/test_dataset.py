from __future__ import annotations

import torch

from beamforming.data.dataset import ChannelDataset
from beamforming.data.synthetic import rayleigh_narrowband_channel, sparse_geometric_mmwave_channel


def test_rayleigh_channel_shape() -> None:
    channels = rayleigh_narrowband_channel(8, 4, 16)
    assert channels.shape == (8, 4, 16)
    assert torch.is_complex(channels)


def test_mmwave_channel_shape() -> None:
    channels = sparse_geometric_mmwave_channel(6, 3, 8, 2)
    assert channels.shape == (6, 3, 8)
    assert torch.is_complex(channels)


def test_channel_dataset_item() -> None:
    channels = rayleigh_narrowband_channel(4, 2, 8)
    dataset = ChannelDataset(channels, torch.tensor([0.0, 5.0, 10.0, 15.0]))
    item = dataset[0]
    assert item["channel"].shape == (2, 8)
    assert item["channel_real"].shape == (2, 2, 8)
