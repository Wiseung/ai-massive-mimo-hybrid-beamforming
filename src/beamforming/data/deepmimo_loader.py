"""DeepMIMO v4 and legacy dataset integration with graceful errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from beamforming.data.dataset import ChannelDataset


def _import_deepmimo():
    try:
        import deepmimo as dm  # type: ignore

        return dm
    except Exception as stable_exc:
        try:
            import DeepMIMO as legacy_dm  # type: ignore

            return legacy_dm
        except Exception as legacy_exc:
            raise ImportError(
                "DeepMIMO is not installed. Install it with: pip install deepmimo"
            ) from legacy_exc if legacy_exc else stable_exc


def load_deepmimo_dataset(
    scenario_path: str | Path | None = None,
    *,
    scenario: str | None = None,
    download: bool = False,
    bs_idx: int = 0,
    user_slice: tuple[int, int] | None = None,
    num_users: int = 4,
    num_bs_ant: int | None = None,
    num_paths: int | None = None,
    num_subcarriers: int | None = None,
    subcarrier_idx: int | None = None,
    narrowband: bool = True,
) -> ChannelDataset:
    """Load DeepMIMO v4 or legacy output into the project dataset contract."""
    if scenario is None and scenario_path is None:
        raise ValueError("Provide either scenario or scenario_path.")

    if scenario_path is not None:
        path = Path(scenario_path)
        if path.suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            channels = payload["channels"] if isinstance(payload, dict) and "channels" in payload else payload
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            return ChannelDataset(_adapt_loaded_tensor(channels, num_users=num_users, narrowband=narrowband), metadata=metadata)
        if path.suffix == ".npy":
            channels = np.load(path)
            return ChannelDataset(_adapt_loaded_tensor(torch.from_numpy(channels), num_users=num_users, narrowband=narrowband), metadata={})

    dm = _import_deepmimo()

    if hasattr(dm, "load") and scenario is not None:
        if download and hasattr(dm, "download"):
            dm.download(scenario)
        dataset = dm.load(scenario)
        if not hasattr(dataset, "compute_channels"):
            raise RuntimeError("Unsupported DeepMIMO v4 dataset object: missing compute_channels().")
        raw = dataset.compute_channels()
        channels = _adapt_v4_channels(
            raw,
            bs_idx=bs_idx,
            num_users=num_users,
            num_bs_ant=num_bs_ant,
            num_subcarriers=num_subcarriers,
            narrowband=narrowband,
            subcarrier_idx=subcarrier_idx,
        )
        metadata = {
            "source": "deepmimo_v4",
            "scenario": scenario,
            "bs_idx": bs_idx,
            "num_users": num_users,
            "num_bs_ant": num_bs_ant or channels.shape[-1],
            "num_rf_chains": min(num_users, 4),
            "num_subcarriers": 1 if narrowband else channels.shape[-2],
            "narrowband": narrowband,
        }
        if user_slice is not None:
            start, end = user_slice
            channels = channels[start:end]
        return ChannelDataset(channels=channels, metadata=metadata)

    if scenario_path is None:
        raise FileNotFoundError(
            "DeepMIMO v4 scenario could not be loaded and no legacy scenario path was provided."
        )
    legacy_path = Path(scenario_path)
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"DeepMIMO scenario path does not exist: {legacy_path}. If you want v4 auto-download, pass --scenario <name> --download."
        )
    if hasattr(dm, "load"):
        dataset = dm.load(str(legacy_path))
    else:
        raise RuntimeError("Unsupported DeepMIMO legacy interface for the provided path.")
    if hasattr(dataset, "compute_channels"):
        raw = dataset.compute_channels()
    else:
        raw = dataset
    channels = _adapt_legacy_channels(raw, num_users=num_users, num_bs_ant=num_bs_ant, narrowband=narrowband, subcarrier_idx=subcarrier_idx)
    metadata = {
        "source": "deepmimo_legacy",
        "scenario_path": str(legacy_path),
        "num_users": num_users,
        "num_bs_ant": num_bs_ant or channels.shape[-1],
        "num_rf_chains": min(num_users, 4),
        "narrowband": narrowband,
    }
    return ChannelDataset(channels=channels, metadata=metadata)


def _adapt_loaded_tensor(channels: torch.Tensor, num_users: int, narrowband: bool) -> torch.Tensor:
    tensor = torch.as_tensor(channels)
    if not torch.is_complex(tensor):
        tensor = torch.complex(tensor.float(), torch.zeros_like(tensor.float()))
    if tensor.ndim == 3:
        return tensor.to(torch.complex64)
    if tensor.ndim == 4 and not narrowband:
        return tensor.to(torch.complex64)
    if tensor.ndim == 4 and narrowband:
        return tensor.mean(dim=-2).to(torch.complex64)
    raise RuntimeError(f"Unsupported saved tensor shape: {tuple(tensor.shape)}")


def _adapt_v4_channels(
    raw: Any,
    *,
    bs_idx: int,
    num_users: int,
    num_bs_ant: int | None,
    num_subcarriers: int | None,
    narrowband: bool,
    subcarrier_idx: int | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(raw)
    if not torch.is_complex(tensor):
        tensor = torch.complex(tensor.float(), torch.zeros_like(tensor.float()))

    if tensor.ndim == 5:
        tensor = tensor[:, bs_idx]
    if tensor.ndim != 4:
        raise RuntimeError(f"Unsupported DeepMIMO v4 channel shape: {tuple(tensor.shape)}")

    num_ue, n_rx, n_tx, n_sub = tensor.shape
    if n_rx > 1:
        tensor = tensor.mean(dim=1)
    else:
        tensor = tensor[:, 0]
    if num_bs_ant is not None:
        tensor = tensor[:, :num_bs_ant, :]
        n_tx = tensor.size(1)
    if num_subcarriers is not None:
        tensor = tensor[:, :, :num_subcarriers]
        n_sub = tensor.size(-1)

    usable = (num_ue // num_users) * num_users
    tensor = tensor[:usable]
    if narrowband:
        if subcarrier_idx is not None:
            tensor = tensor[:, :, subcarrier_idx]
        else:
            tensor = tensor.mean(dim=-1)
        tensor = tensor.reshape(-1, num_users, tensor.size(-1))
        return tensor.to(torch.complex64)

    tensor = tensor.permute(0, 2, 1)  # [n_ue, n_sub, n_tx]
    tensor = tensor.reshape(-1, num_users, tensor.size(1), tensor.size(2))
    return tensor.to(torch.complex64)


def _adapt_legacy_channels(
    raw: Any,
    *,
    num_users: int,
    num_bs_ant: int | None,
    narrowband: bool,
    subcarrier_idx: int | None,
) -> torch.Tensor:
    tensor = _extract_legacy_tensor(raw)
    if tensor.ndim == 4:
        if subcarrier_idx is not None:
            tensor = tensor[..., subcarrier_idx]
        else:
            tensor = tensor[..., 0]
    if tensor.ndim == 3 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 2:
        raise RuntimeError(f"Unsupported DeepMIMO legacy channel shape after adaptation: {tuple(tensor.shape)}")
    if num_bs_ant is not None:
        tensor = tensor[:, :num_bs_ant]
    usable = (tensor.size(0) // num_users) * num_users
    tensor = tensor[:usable]
    return tensor.reshape(-1, num_users, tensor.size(-1)).to(torch.complex64)


def _extract_legacy_tensor(raw: Any) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw if torch.is_complex(raw) else torch.complex(raw.float(), torch.zeros_like(raw.float()))
    if isinstance(raw, np.ndarray):
        arr = torch.from_numpy(raw)
        return arr if torch.is_complex(arr) else torch.complex(arr.float(), torch.zeros_like(arr.float()))
    if isinstance(raw, dict):
        for key in ("channels", "channel"):
            if key in raw:
                return _extract_legacy_tensor(raw[key])
        first_key = next(iter(raw))
        return _extract_legacy_tensor(raw[first_key])
    if isinstance(raw, list):
        return _extract_legacy_tensor(raw[0])
    if hasattr(raw, "channel"):
        return _extract_legacy_tensor(raw.channel)
    raise RuntimeError("Unable to extract legacy DeepMIMO tensor.")
