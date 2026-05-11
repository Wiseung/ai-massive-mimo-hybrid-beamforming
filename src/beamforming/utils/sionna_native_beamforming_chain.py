"""Helpers for inserting project precoders into the Sionna-native OFDM chain."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import torch

from beamforming.baselines.common import get_digital_precoder
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device


def build_frequency_domain_channel(
    batch_size: int,
    num_subcarriers: int,
    num_users: int,
    num_bs_ant: int,
    device: torch.device,
    resource_grid: Any | None = None,
    noise_var: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return H_f with shape ``(B, Nsc, K, Nt)``.

    Args:
        batch_size: Batch size ``B``.
        num_subcarriers: Number of effective frequency bins ``Nsc``.
        num_users: Number of users / streams ``K``.
        num_bs_ant: Number of BS antennas ``Nt``.
        device: Torch device for the returned tensor.
        resource_grid: Optional Sionna ``ResourceGrid`` used to attempt native channel extraction.
        noise_var: Optional noise variance for the Sionna channel call path.
    """
    sionna_device = resolve_sionna_device(device)
    OFDMChannel, _, _ = load_component("OFDMChannel")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
    if resource_grid is not None and OFDMChannel is not None and RayleighBlockFading is not None:
        try:
            channel_model = RayleighBlockFading(
                num_rx=num_users,
                num_rx_ant=1,
                num_tx=1,
                num_tx_ant=num_bs_ant,
                device=sionna_device,
            )
            channel = OFDMChannel(channel_model, resource_grid, return_channel=True, device=sionna_device)
            dummy_x = torch.zeros(
                batch_size,
                1,
                num_bs_ant,
                resource_grid.num_ofdm_symbols,
                resource_grid.fft_size,
                dtype=torch.complex64,
                device=device,
            )
            _, h_freq = channel(dummy_x, no=torch.full((batch_size, num_users, 1), float(noise_var or 0.0), device=device))
            h_freq = h_freq.squeeze(2).squeeze(3).squeeze(3).permute(0, 3, 1, 2).contiguous()
            return h_freq, {
                "used_sionna_channel_tensor": True,
                "fallback_used": False,
                "notes": ["Extracted frequency-domain channel tensor from real Sionna OFDMChannel."],
            }
        except Exception as exc:  # pragma: no cover - optional runtime path
            pass

    real = torch.randn(batch_size, num_subcarriers, num_users, num_bs_ant, device=device)
    imag = torch.randn(batch_size, num_subcarriers, num_users, num_bs_ant, device=device)
    h_f = (real + 1j * imag) / math.sqrt(2.0)
    return h_f.to(torch.complex64), {
        "used_sionna_channel_tensor": False,
        "fallback_used": True,
        "notes": ["Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable."],
    }


def compute_project_precoder_per_subcarrier(
    method: str,
    channel_f: torch.Tensor,
    noise_var: float | torch.Tensor,
) -> torch.Tensor:
    """Compute project precoders with output shape ``(B, Nsc, Nt, K)``."""
    precoders = []
    for sc in range(channel_f.size(1)):
        precoders.append(get_digital_precoder(method, channel_f[:, sc, :, :], noise_var=noise_var))
    return torch.stack(precoders, dim=1)


def describe_tensor(
    name: str,
    tensor: torch.Tensor | None,
    semantic_axes: list[str] | None = None,
    expected: str | None = None,
) -> dict[str, Any]:
    """Return a serializable tensor trace record."""
    if tensor is None:
        return {
            "tensor_name": name,
            "shape": None,
            "dtype": None,
            "device": None,
            "semantic_axes": semantic_axes or [],
            "expected": expected,
        }
    return {
        "tensor_name": name,
        "shape": [int(x) for x in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "semantic_axes": semantic_axes or [],
        "expected": expected,
    }


def build_pilot_aware_multiuser_resource_grid(
    num_users: int,
    num_effective_subcarriers: int,
    device: torch.device,
    *,
    num_ofdm_symbols: int = 2,
    num_tx: int = 1,
    num_guard_carriers: tuple[int, int] = (1, 1),
    dc_null: bool = True,
    pilot_pattern: str = "kronecker",
    pilot_ofdm_symbol_indices: list[int] | None = None,
    subcarrier_spacing: float = 15_000.0,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    """Build a pilot-aware Sionna ResourceGrid and StreamManagement for MU downlink.

    Args:
        num_users: Number of logical user streams ``K``.
        num_effective_subcarriers: Number of effective subcarriers to expose to
            the project-side ``H_f`` / ``F_f`` interface.
        device: Torch device used only to resolve a Sionna-compatible device string.
        num_ofdm_symbols: Total OFDM symbols. Must exceed the number of pilot OFDM symbols
            to ensure ``num_data_symbols > 0``.
        num_tx: Number of Sionna transmitters. The current project-side beamformed downlink
            path uses ``num_tx=1`` and ``num_streams_per_tx=K``.

    Returns:
        ``(resource_grid, stream_management, meta)`` where ``meta`` records explicit
        fallback reasons if the requested configuration cannot support a pilot-aware
        multi-user beamformed receiver path.
    """
    ResourceGrid, _, _ = load_component("ResourceGrid")
    StreamManagement, _, _ = load_component("StreamManagement")
    sionna_device = resolve_sionna_device(device)
    pilot_ofdm_symbol_indices = [0] if pilot_ofdm_symbol_indices is None else pilot_ofdm_symbol_indices
    num_streams_per_tx = num_users if num_tx == 1 else 1
    total_streams = num_tx * num_streams_per_tx
    meta: dict[str, Any] = {
        "fallback_used": False,
        "fallback_reason": "",
        "num_users": int(num_users),
        "num_tx": int(num_tx),
        "num_streams_per_tx": int(num_streams_per_tx),
    }
    if ResourceGrid is None or StreamManagement is None:
        meta.update({"fallback_used": True, "fallback_reason": "resource_grid_or_stream_management_unavailable"})
        return None, None, meta
    if num_ofdm_symbols <= len(pilot_ofdm_symbol_indices):
        meta.update(
            {
                "fallback_used": True,
                "fallback_reason": "num_ofdm_symbols_must_exceed_number_of_pilot_symbols_to_leave_data_re",
            }
        )
        return None, None, meta
    if num_effective_subcarriers % total_streams != 0:
        meta.update(
            {
                "fallback_used": True,
                "fallback_reason": "num_effective_subcarriers_must_be_multiple_of_num_tx_times_num_streams_per_tx_for_kronecker_pilots",
            }
        )
        return None, None, meta

    fft_size = num_effective_subcarriers + int(dc_null) + int(num_guard_carriers[0]) + int(num_guard_carriers[1])
    resource_grid = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=subcarrier_spacing,
        num_tx=num_tx,
        num_streams_per_tx=num_streams_per_tx,
        num_guard_carriers=num_guard_carriers,
        dc_null=dc_null,
        pilot_pattern=pilot_pattern,
        pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        device=sionna_device,
    )
    rx_tx_association = np.ones((num_users, num_tx), dtype=int)
    stream_management = StreamManagement(rx_tx_association, num_streams_per_tx=num_streams_per_tx)
    meta.update(
        {
            "fft_size": int(resource_grid.fft_size),
            "num_data_symbols": int(resource_grid.num_data_symbols),
            "num_pilot_symbols": int(resource_grid.num_pilot_symbols),
            "effective_subcarrier_ind": [int(x) for x in resource_grid.effective_subcarrier_ind],
            "rx_tx_association": rx_tx_association.tolist(),
        }
    )
    return resource_grid, stream_management, meta


def apply_precoder_to_resource_grid(stream_symbols: torch.Tensor, precoder_f: torch.Tensor) -> torch.Tensor:
    """Apply per-subcarrier precoding.

    stream_symbols: (B, Nsc, K)
    precoder_f: (B, Nsc, Nt, K)
    returns: (B, Nt, Nsc)
    """
    tx = torch.matmul(precoder_f, stream_symbols.unsqueeze(-1)).squeeze(-1)
    return tx.transpose(1, 2).contiguous()


def project_symbols_to_sionna_grid(
    stream_symbols: torch.Tensor,
    num_ofdm_symbols: int = 1,
) -> tuple[torch.Tensor, dict[str, str | bool]]:
    """Bridge project symbols to a Sionna-style grid tensor.

    Args:
        stream_symbols: ``(B, Nsc, K)`` project user-symbol tensor.
        num_ofdm_symbols: Number of OFDM symbols. Currently only ``1`` is supported natively.

    Returns:
        grid: ``(B, 1, K, num_ofdm_symbols, Nsc)`` if supported, otherwise a fallback reshaping result.
        meta: Explicit bridge status with fallback reason when needed.
    """
    if num_ofdm_symbols != 1:
        return (
            stream_symbols.transpose(1, 2).unsqueeze(1),
            {"fallback_used": True, "fallback_reason": "only_num_ofdm_symbols_eq_1_is_supported_in_project_symbol_bridge"},
        )
    return (
        stream_symbols.transpose(1, 2).unsqueeze(1).unsqueeze(-2).contiguous(),
        {"fallback_used": False, "fallback_reason": ""},
    )


def map_project_streams_to_sionna_rg(
    stream_symbols: torch.Tensor,
    resource_grid: Any,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Map project user-stream symbols to a pilot-aware Sionna ResourceGrid.

    Args:
        stream_symbols: ``(B, Nsc, K)`` where ``Nsc`` must match
            ``resource_grid.num_data_symbols`` for the current one-data-OFDM-symbol bridge.
        resource_grid: Sionna ``ResourceGrid`` configured with
            ``num_tx=1`` and ``num_streams_per_tx=K``.

    Returns:
        ``x_rg`` with shape ``(B, 1, K, num_ofdm_symbols, fft_size)`` and explicit
        fallback metadata if the bridge cannot be formed.
    """
    ResourceGridMapper, _, _ = load_component("ResourceGridMapper")
    if ResourceGridMapper is None:
        return None, {"fallback_used": True, "fallback_reason": "ResourceGridMapper_unavailable"}
    if stream_symbols.ndim != 3:
        return None, {"fallback_used": True, "fallback_reason": f"expected_project_stream_symbols_rank_3_got_{stream_symbols.ndim}"}
    batch_size, num_subcarriers, num_users = stream_symbols.shape
    if int(resource_grid.num_streams_per_tx) != int(num_users):
        return None, {
            "fallback_used": True,
            "fallback_reason": "resource_grid_num_streams_per_tx_must_match_project_num_users",
        }
    if int(resource_grid.num_data_symbols) != int(num_subcarriers):
        return None, {
            "fallback_used": True,
            "fallback_reason": "resource_grid_num_data_symbols_must_match_project_num_subcarriers_for_current_bridge",
            "resource_grid_num_data_symbols": int(resource_grid.num_data_symbols),
            "project_num_subcarriers": int(num_subcarriers),
        }
    mapper = ResourceGridMapper(resource_grid, device=resolve_sionna_device(stream_symbols.device))
    mapped = mapper(stream_symbols.permute(0, 2, 1).unsqueeze(1).contiguous())
    return mapped, {
        "fallback_used": False,
        "fallback_reason": "",
        "mapped_shape": [int(x) for x in mapped.shape],
        "batch_size": int(batch_size),
        "num_users": int(num_users),
    }


def project_precoded_grid_to_sionna_tx(
    tx_precoded: torch.Tensor,
    num_ofdm_symbols: int = 1,
) -> tuple[torch.Tensor, dict[str, str | bool]]:
    """Bridge project precoded antenna-domain symbols to a Sionna OFDMChannel input.

    Args:
        tx_precoded: ``(B, Nt, Nsc)`` from ``apply_precoder_to_resource_grid``.
        num_ofdm_symbols: Number of OFDM symbols. Currently only ``1`` is supported.

    Returns:
        tx_grid: ``(B, 1, Nt, num_ofdm_symbols, Nsc)`` if supported.
        meta: Explicit bridge status with fallback reason when needed.
    """
    if num_ofdm_symbols != 1:
        return (
            tx_precoded.unsqueeze(1),
            {"fallback_used": True, "fallback_reason": "only_num_ofdm_symbols_eq_1_is_supported_in_precoded_tx_bridge"},
        )
    return (
        tx_precoded.unsqueeze(1).unsqueeze(-2).contiguous(),
        {"fallback_used": False, "fallback_reason": ""},
    )


def extract_effective_channel_from_sionna(
    resource_grid: Any,
    batch_size: int,
    num_users: int,
    num_bs_ant: int,
    device: torch.device,
    *,
    noise_var: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    """Extract a project-side effective channel from Sionna OFDMChannel.

    Returns:
        ``(H_f, h_freq_full, meta)`` where
        ``H_f`` has shape ``(B, Nsc, K, Nt)`` and ``h_freq_full`` has shape
        ``(B, K, 1, 1, Nt, num_ofdm_symbols, fft_size)`` when successful.
    """
    OFDMChannel, _, _ = load_component("OFDMChannel")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
    meta: dict[str, Any] = {
        "fallback_used": False,
        "fallback_reason": "",
        "used_native_channel_extraction": False,
        "selected_data_symbol_index": None,
    }
    if OFDMChannel is None or RayleighBlockFading is None:
        meta.update({"fallback_used": True, "fallback_reason": "OFDMChannel_or_RayleighBlockFading_unavailable"})
        return None, None, meta
    try:
        sionna_device = resolve_sionna_device(device)
        channel_model = RayleighBlockFading(
            num_rx=num_users,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=num_bs_ant,
            device=sionna_device,
        )
        channel = OFDMChannel(channel_model, resource_grid, return_channel=True, device=sionna_device)
        dummy_x = torch.zeros(
            batch_size,
            1,
            num_bs_ant,
            resource_grid.num_ofdm_symbols,
            resource_grid.fft_size,
            dtype=torch.complex64,
            device=device,
        )
        noise = torch.full((batch_size, num_users, 1), float(noise_var), dtype=torch.float32, device=device)
        _, h_freq_full = channel(dummy_x, no=noise)
        pilot_indices = set(getattr(resource_grid.pilot_pattern, "_pilot_ofdm_symbol_indices", []) or [])
        data_symbol_indices = [idx for idx in range(int(resource_grid.num_ofdm_symbols)) if idx not in pilot_indices]
        if not data_symbol_indices:
            meta.update({"fallback_used": True, "fallback_reason": "resource_grid_has_no_data_ofdm_symbol"})
            return None, h_freq_full, meta
        data_symbol_index = data_symbol_indices[0]
        effective_subcarrier_ind = torch.as_tensor(resource_grid.effective_subcarrier_ind, device=device)
        h_sel = h_freq_full[:, :, 0, 0, :, :, :]  # [B, K, Nt, S, F]
        h_perm = h_sel.permute(0, 3, 4, 1, 2).contiguous()  # [B, S, F, K, Nt]
        h_f = h_perm[:, data_symbol_index, effective_subcarrier_ind, :, :].contiguous()
        meta.update(
            {
                "used_native_channel_extraction": True,
                "selected_data_symbol_index": int(data_symbol_index),
                "full_channel_shape": [int(x) for x in h_freq_full.shape],
                "effective_subcarrier_ind": [int(x) for x in effective_subcarrier_ind.tolist()],
            }
        )
        return h_f, h_freq_full, meta
    except Exception as exc:  # pragma: no cover - optional runtime path
        meta.update({"fallback_used": True, "fallback_reason": f"{type(exc).__name__}: {exc}"})
        return None, None, meta


def apply_project_precoder_to_sionna_grid(
    x_rg: torch.Tensor,
    precoder_f: torch.Tensor,
    resource_grid: Any,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Apply project precoders to a pilot-aware Sionna stream grid.

    Args:
        x_rg: ``(B, 1, K, num_ofdm_symbols, fft_size)`` stream-domain Sionna grid.
        precoder_f: ``(B, Nsc, Nt, K)`` project precoders on effective subcarriers.
        resource_grid: ResourceGrid whose effective subcarrier indices define ``Nsc``.

    Returns:
        Antenna-domain Sionna transmit grid ``(B, 1, Nt, num_ofdm_symbols, fft_size)``
        plus explicit fallback metadata.
    """
    if x_rg.ndim != 5:
        return None, {"fallback_used": True, "fallback_reason": f"expected_x_rg_rank_5_got_{x_rg.ndim}"}
    if precoder_f.ndim != 4:
        return None, {"fallback_used": True, "fallback_reason": f"expected_precoder_rank_4_got_{precoder_f.ndim}"}
    effective_subcarrier_ind = torch.as_tensor(resource_grid.effective_subcarrier_ind, device=x_rg.device)
    if int(precoder_f.size(1)) != int(effective_subcarrier_ind.numel()):
        return None, {
            "fallback_used": True,
            "fallback_reason": "precoder_num_subcarriers_must_match_number_of_effective_subcarriers",
            "precoder_num_subcarriers": int(precoder_f.size(1)),
            "effective_subcarriers": int(effective_subcarrier_ind.numel()),
        }
    if int(x_rg.size(2)) != int(precoder_f.size(-1)):
        return None, {
            "fallback_used": True,
            "fallback_reason": "stream_dimension_mismatch_between_x_rg_and_precoder",
            "x_rg_num_streams": int(x_rg.size(2)),
            "precoder_num_streams": int(precoder_f.size(-1)),
        }
    x_eff = x_rg[..., effective_subcarrier_ind]  # [B, 1, K, S, Nsc]
    x_stream = x_eff[:, 0, :, :, :].permute(0, 2, 3, 1).contiguous()  # [B, S, Nsc, K]
    tx_ant_per_symbol = []
    for symbol_idx in range(x_stream.size(1)):
        stream_slice = x_stream[:, symbol_idx, :, :]  # [B, Nsc, K]
        tx_ant_per_symbol.append(torch.matmul(precoder_f, stream_slice.unsqueeze(-1)).squeeze(-1))
    tx_ant = torch.stack(tx_ant_per_symbol, dim=3).permute(0, 2, 3, 1).contiguous()  # [B, Nt, S, Nsc]
    tx_grid = torch.zeros(
        x_rg.size(0),
        1,
        precoder_f.size(2),
        x_rg.size(3),
        x_rg.size(4),
        dtype=x_rg.dtype,
        device=x_rg.device,
    )
    tx_grid[..., effective_subcarrier_ind] = tx_ant.unsqueeze(1)
    return tx_grid, {
        "fallback_used": False,
        "fallback_reason": "",
        "effective_subcarrier_ind": [int(x) for x in effective_subcarrier_ind.tolist()],
        "tx_grid_shape": [int(x) for x in tx_grid.shape],
    }


def sionna_rx_to_project_symbols(
    x_hat: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, str | bool]]:
    """Bridge Sionna equalizer output back to project symbol layout.

    Args:
        x_hat: Expected Sionna equalizer output ``(B, 1, K, Nsc_total)`` for the currently supported path.

    Returns:
        project_symbols: ``(B, Nsc_total, K)`` if supported.
        meta: Explicit bridge status with fallback reason when needed.
    """
    if x_hat.ndim != 4:
        return x_hat, {"fallback_used": True, "fallback_reason": f"unexpected_equalizer_rank_{x_hat.ndim}"}
    return x_hat.squeeze(1).transpose(1, 2).contiguous(), {"fallback_used": False, "fallback_reason": ""}


def compute_project_metrics_from_sionna_rx(
    project_rx_symbols: torch.Tensor,
    ref_symbols: torch.Tensor,
) -> dict[str, float]:
    """Compute simple project-side metrics from a bridged Sionna receiver output.

    Args:
        project_rx_symbols: ``(B, Nsc, K)``
        ref_symbols: ``(B, Nsc, K)``
    """
    mse = torch.mean(torch.abs(project_rx_symbols - ref_symbols) ** 2).item()
    eff_sinr = torch.mean(torch.abs(ref_symbols) ** 2) / torch.mean(torch.abs(project_rx_symbols - ref_symbols) ** 2).clamp_min(1e-12)
    return {
        "symbol_mse": float(mse),
        "effective_sinr_db": float(10.0 * torch.log10(eff_sinr).item()),
    }


def validate_sionna_receiver_shapes(
    y: torch.Tensor,
    h_hat: torch.Tensor,
    err_var: torch.Tensor,
    stream_management: Any,
    resource_grid: Any,
) -> dict[str, Any]:
    """Validate whether Sionna receiver tensors are internally consistent."""
    num_rx = int(stream_management.num_rx)
    num_tx = int(stream_management.num_tx)
    num_streams_per_tx = int(stream_management.num_streams_per_tx)
    expected_h_hat_shape = [
        int(y.size(0)),
        num_rx,
        int(y.size(2)),
        num_tx,
        num_streams_per_tx,
        int(resource_grid.num_ofdm_symbols),
        len(resource_grid.effective_subcarrier_ind),
    ]
    result = {
        "valid": True,
        "resource_grid_num_data_symbols": int(resource_grid.num_data_symbols),
        "stream_management_num_desired_streams": int(len(stream_management.detection_desired_ind)),
        "expected_h_hat_shape": expected_h_hat_shape,
        "observed_y_shape": [int(x) for x in y.shape],
        "observed_h_hat_shape": [int(x) for x in h_hat.shape],
        "observed_err_var_shape": [int(x) for x in err_var.shape],
        "reason": "",
    }
    if int(resource_grid.num_data_symbols) <= 0:
        result.update({"valid": False, "reason": "resource_grid_has_zero_data_symbols"})
        return result
    if len(stream_management.detection_desired_ind) == 0:
        result.update({"valid": False, "reason": "stream_management_has_no_desired_streams"})
        return result
    if list(h_hat.shape) != expected_h_hat_shape:
        result.update({"valid": False, "reason": "h_hat_shape_mismatch"})
        return result
    if list(err_var.shape) != expected_h_hat_shape:
        result.update({"valid": False, "reason": "err_var_shape_mismatch"})
        return result
    return result


def evaluate_ofdm_beamforming_outputs(
    channel_f: torch.Tensor,
    precoder_f: torch.Tensor,
    stream_symbols: torch.Tensor,
    noise_var: float,
) -> dict[str, float]:
    """Compute proxy metrics for the beamformed OFDM chain."""
    heff = torch.matmul(channel_f, precoder_f)
    diag = torch.diagonal(heff, dim1=-2, dim2=-1)
    signal_power = torch.abs(diag) ** 2
    total_power = torch.abs(heff) ** 2
    interference_power = total_power.sum(dim=-1) - signal_power
    sinr = signal_power / (interference_power + noise_var)
    noisy_streams = diag * stream_symbols
    mse = torch.mean(torch.abs(noisy_streams - stream_symbols) ** 2).item()
    power_norm = torch.mean((torch.abs(precoder_f) ** 2).sum(dim=(-2, -1))).item()
    power_violation = torch.mean(torch.abs((torch.abs(precoder_f) ** 2).sum(dim=(-2, -1)) - 1.0)).item()
    sinr_mean = sinr.mean().clamp_min(1e-12)
    sum_rate = torch.log2(1.0 + sinr).sum(dim=-1).mean().item()
    return {
        "ber_if_available": None,
        "symbol_mse": float(mse),
        "effective_sinr_db": float(10.0 * torch.log10(sinr_mean).item()),
        "approximate_sum_rate": float(sum_rate),
        "power_norm": float(power_norm),
        "power_violation": float(power_violation),
    }


def time_function(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = perf_counter()
    out = fn(*args, **kwargs)
    return out, (perf_counter() - start) * 1000.0
