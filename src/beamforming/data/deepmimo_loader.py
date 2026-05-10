"""DeepMIMO dataset integration with graceful missing-dependency handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from beamforming.data.dataset import ChannelDataset


def _import_deepmimo():
    try:
        import DeepMIMO  # type: ignore

        return DeepMIMO
    except Exception:
        try:
            import deepmimo  # type: ignore

            return deepmimo
        except Exception as exc:
            raise ImportError(
                "DeepMIMO is not installed. Install the optional dependency and place the dataset locally before running DeepMIMO experiments."
            ) from exc


def load_deepmimo_dataset(
    scenario_path: str | Path,
    bs_idx: int = 0,
    user_slice: tuple[int, int] | None = None,
    num_users: int = 4,
    num_bs_ant: int | None = None,
    num_paths: int | None = None,
    subcarrier_idx: int | None = None,
) -> ChannelDataset:
    """Load a local DeepMIMO scenario into the unified ChannelDataset interface."""
    scenario_path = Path(scenario_path)
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"DeepMIMO scenario path does not exist: {scenario_path}. DeepMIMO experiments have not been run because the dataset is not present locally."
        )
    if scenario_path.suffix in {".pt", ".pth"}:
        payload = torch.load(scenario_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "channels" in payload:
            channels = payload["channels"]
            metadata = payload.get("metadata", {})
        else:
            channels = payload
            metadata = {}
        return ChannelDataset(channels=_adapt_channels(channels, num_users=num_users, subcarrier_idx=subcarrier_idx), metadata=metadata)

    if scenario_path.suffix == ".npy":
        array = np.load(scenario_path)
        channels = torch.from_numpy(array)
        return ChannelDataset(channels=_adapt_channels(channels, num_users=num_users, subcarrier_idx=subcarrier_idx), metadata={})

    dm = _import_deepmimo()
    if not hasattr(dm, "load"):
        raise RuntimeError("Unsupported DeepMIMO package interface detected: missing load().")
    dataset = dm.load(str(scenario_path))
    if hasattr(dataset, "compute_channels"):
        channel_params = None
        if hasattr(dm, "ChannelParameters"):
            try:
                channel_params = dm.ChannelParameters()
                if num_paths is not None and hasattr(channel_params, "num_paths"):
                    channel_params.num_paths = num_paths
                if subcarrier_idx is not None and hasattr(channel_params, "subcarrier_idx"):
                    channel_params.subcarrier_idx = [subcarrier_idx]
                if num_bs_ant is not None and hasattr(channel_params, "tx_ant"):
                    channel_params.tx_ant = num_bs_ant
            except Exception:
                channel_params = None
        raw = dataset.compute_channels(channel_params) if channel_params is not None else dataset.compute_channels()
    else:
        raw = dataset

    channels = _extract_bs_channels(raw, bs_idx=bs_idx)
    channels = _adapt_channels(channels, num_users=num_users, subcarrier_idx=subcarrier_idx, num_bs_ant=num_bs_ant)
    metadata = {
        "source": "deepmimo",
        "scenario_path": str(scenario_path),
        "bs_idx": bs_idx,
        "user_slice": user_slice,
        "num_users": num_users,
        "num_bs_ant": num_bs_ant or channels.size(-1),
        "num_paths": num_paths,
        "subcarrier_idx": subcarrier_idx,
    }
    if user_slice is not None:
        start, end = user_slice
        channels = channels[start:end]
    return ChannelDataset(channels=channels, metadata=metadata)


def deepmimo_not_available_metadata() -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": "DeepMIMO experiments have not been run because the dataset is not present locally.",
    }


def _extract_bs_channels(raw: Any, bs_idx: int) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw
    if isinstance(raw, np.ndarray):
        return torch.from_numpy(raw)
    if isinstance(raw, dict):
        if "channels" in raw:
            return _extract_bs_channels(raw["channels"], bs_idx)
        if "channel" in raw:
            return _extract_bs_channels(raw["channel"], bs_idx)
        if bs_idx in raw:
            return _extract_bs_channels(raw[bs_idx], bs_idx)
    if isinstance(raw, list):
        return _extract_bs_channels(raw[bs_idx], bs_idx)
    if hasattr(raw, "channel"):
        return _extract_bs_channels(raw.channel, bs_idx)
    raise RuntimeError("Unable to extract channel tensor from DeepMIMO object.")


def _adapt_channels(
    channels: torch.Tensor,
    num_users: int,
    subcarrier_idx: int | None = None,
    num_bs_ant: int | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(channels)
    if not torch.is_complex(tensor):
        tensor = torch.complex(tensor.float(), torch.zeros_like(tensor.float()))
    if tensor.ndim == 4:
        if subcarrier_idx is not None and tensor.size(-1) > 1:
            tensor = tensor[..., subcarrier_idx]
        else:
            tensor = tensor[..., 0]
    if tensor.ndim == 3 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 2:
        raise RuntimeError(f"Unsupported DeepMIMO channel shape after adaptation: {tuple(tensor.shape)}")
    if num_bs_ant is not None:
        tensor = tensor[:, :num_bs_ant]
    usable = (tensor.size(0) // num_users) * num_users
    tensor = tensor[:usable]
    return tensor.reshape(-1, num_users, tensor.size(-1)).to(torch.complex64)
