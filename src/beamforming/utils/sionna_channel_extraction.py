"""Utilities for extracting project-side H_f from Sionna channel tensors."""

from __future__ import annotations

from typing import Any

import torch


def convert_sionna_h_to_project_h_f(
    sionna_channel: torch.Tensor,
    *,
    effective_subcarrier_ind: list[int] | torch.Tensor,
    data_symbol_index: int,
    num_users: int,
    num_bs_ant: int,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Convert a Sionna OFDM channel tensor to project ``H_f=(B,Nsc,K,Nt)``.

    Expected Sionna channel axes:
    ``(batch, rx, rx_ant, tx, tx_ant, ofdm_symbol, fft_bin)``.

    Project axes:
    ``(batch, subcarrier, user, bs_ant)``.

    The current bridge assumes MU downlink semantics:
    - ``rx`` indexes logical users ``K``
    - ``rx_ant=1``
    - ``tx=1``
    - ``tx_ant`` indexes BS antennas ``Nt``
    - one data OFDM symbol is selected via ``data_symbol_index``
    - effective subcarriers are selected via ``effective_subcarrier_ind``
    """
    meta: dict[str, Any] = {
        "input_shape": [int(x) for x in sionna_channel.shape],
        "assumed_sionna_axes": ["batch", "rx", "rx_ant", "tx", "tx_ant", "ofdm_symbol", "fft_bin"],
        "target_project_axes": ["batch", "subcarrier", "user", "bs_ant"],
        "fallback_reason": "",
        "selected_data_symbol_index": int(data_symbol_index),
    }
    if sionna_channel.ndim != 7:
        meta["fallback_reason"] = f"expected_rank_7_sionna_channel_got_{sionna_channel.ndim}"
        return None, meta
    if not torch.is_complex(sionna_channel):
        meta["fallback_reason"] = "sionna_channel_is_not_complex"
        return None, meta

    eff_idx = torch.as_tensor(effective_subcarrier_ind, device=sionna_channel.device, dtype=torch.long)
    if eff_idx.ndim != 1:
        meta["fallback_reason"] = "effective_subcarrier_ind_must_be_1d"
        return None, meta
    if int(sionna_channel.size(1)) != int(num_users):
        meta["fallback_reason"] = "rx_dimension_does_not_match_num_users"
        return None, meta
    if int(sionna_channel.size(2)) != 1:
        meta["fallback_reason"] = "rx_ant_dimension_must_be_1_for_current_bridge"
        return None, meta
    if int(sionna_channel.size(3)) != 1:
        meta["fallback_reason"] = "tx_dimension_must_be_1_for_current_bridge"
        return None, meta
    if int(sionna_channel.size(4)) != int(num_bs_ant):
        meta["fallback_reason"] = "tx_ant_dimension_does_not_match_num_bs_ant"
        return None, meta
    if not (0 <= int(data_symbol_index) < int(sionna_channel.size(5))):
        meta["fallback_reason"] = "data_symbol_index_out_of_range"
        return None, meta
    if eff_idx.numel() == 0:
        meta["fallback_reason"] = "effective_subcarrier_ind_empty"
        return None, meta
    if int(torch.max(eff_idx).item()) >= int(sionna_channel.size(6)):
        meta["fallback_reason"] = "effective_subcarrier_ind_out_of_range"
        return None, meta

    h_sel = sionna_channel[:, :, 0, 0, :, :, :]  # [B, K, Nt, S, F]
    h_perm = h_sel.permute(0, 3, 4, 1, 2).contiguous()  # [B, S, F, K, Nt]
    h_f = h_perm[:, int(data_symbol_index), eff_idx, :, :].contiguous()
    meta["output_shape"] = [int(x) for x in h_f.shape]
    return h_f, meta


def validate_extracted_h_f(h_f: torch.Tensor | None) -> dict[str, Any]:
    """Validate an extracted project-side ``H_f`` tensor."""
    result: dict[str, Any] = {
        "valid": False,
        "shape": None,
        "dtype": None,
        "is_complex": False,
        "all_finite": False,
        "norm_mean": None,
        "norm_std": None,
        "subcarrier_norm_std_mean": None,
        "fallback_reason": "",
    }
    if h_f is None:
        result["fallback_reason"] = "h_f_is_none"
        return result
    result["shape"] = [int(x) for x in h_f.shape]
    result["dtype"] = str(h_f.dtype)
    result["is_complex"] = bool(torch.is_complex(h_f))
    if h_f.ndim != 4:
        result["fallback_reason"] = f"expected_rank_4_h_f_got_{h_f.ndim}"
        return result
    if not torch.is_complex(h_f):
        result["fallback_reason"] = "h_f_is_not_complex"
        return result
    all_finite = bool(torch.isfinite(h_f.real).all() and torch.isfinite(h_f.imag).all())
    result["all_finite"] = all_finite
    if not all_finite:
        result["fallback_reason"] = "h_f_contains_non_finite_values"
        return result
    norms = torch.linalg.vector_norm(h_f, dim=(-2, -1))
    result["norm_mean"] = float(norms.mean().item())
    result["norm_std"] = float(norms.std(unbiased=False).item())
    per_subcarrier_norm = torch.linalg.vector_norm(h_f, dim=(-2, -1))
    result["subcarrier_norm_std_mean"] = float(per_subcarrier_norm.std(dim=1, unbiased=False).mean().item())
    result["valid"] = True
    return result


def compare_extracted_h_f_with_synthetic_reference(
    extracted_h_f: torch.Tensor | None,
    reference_h_f: torch.Tensor | None,
) -> dict[str, Any]:
    """Compare extracted ``H_f`` against a synthetic reference statistically."""
    result: dict[str, Any] = {
        "comparison_valid": False,
        "shape_match": False,
        "mean_norm_ratio": None,
        "rank_mean_extracted": None,
        "rank_mean_reference": None,
        "fallback_reason": "",
    }
    if extracted_h_f is None or reference_h_f is None:
        result["fallback_reason"] = "missing_extracted_or_reference_h_f"
        return result
    result["shape_match"] = list(extracted_h_f.shape) == list(reference_h_f.shape)
    if not result["shape_match"]:
        result["fallback_reason"] = "shape_mismatch"
        return result
    ex_norm = torch.linalg.vector_norm(extracted_h_f, dim=(-2, -1)).mean()
    ref_norm = torch.linalg.vector_norm(reference_h_f, dim=(-2, -1)).mean().clamp_min(1e-12)
    result["mean_norm_ratio"] = float((ex_norm / ref_norm).item())
    ex_rank = torch.linalg.matrix_rank(extracted_h_f).float().mean()
    ref_rank = torch.linalg.matrix_rank(reference_h_f).float().mean()
    result["rank_mean_extracted"] = float(ex_rank.item())
    result["rank_mean_reference"] = float(ref_rank.item())
    result["comparison_valid"] = True
    return result


def extract_h_f_from_sionna_channel(
    sionna_channel: torch.Tensor | None,
    *,
    resource_grid: Any,
    num_users: int,
    num_bs_ant: int,
) -> tuple[torch.Tensor | None, dict[str, Any], bool, str]:
    """Extract project ``H_f=(B,Nsc,K,Nt)`` from a Sionna channel tensor.

    Args:
        sionna_channel: Sionna channel tensor, expected rank 7.
        resource_grid: Sionna ``ResourceGrid`` providing effective subcarriers and pilot layout.
        num_users: Target project user/stream count ``K``.
        num_bs_ant: Target BS antenna count ``Nt``.

    Returns:
        ``(h_f, metadata, extraction_success, fallback_reason)``.
    """
    metadata: dict[str, Any] = {
        "resource_grid_num_ofdm_symbols": int(resource_grid.num_ofdm_symbols),
        "resource_grid_num_data_symbols": int(resource_grid.num_data_symbols),
        "resource_grid_effective_subcarrier_ind": [int(x) for x in resource_grid.effective_subcarrier_ind],
        "shape_assumptions": {
            "sionna_axes": ["batch", "rx", "rx_ant", "tx", "tx_ant", "ofdm_symbol", "fft_bin"],
            "project_axes": ["batch", "subcarrier", "user", "bs_ant"],
            "rx_maps_to_user": True,
            "tx_ant_maps_to_bs_ant": True,
            "single_tx_required": True,
            "single_rx_ant_required": True,
        },
    }
    if sionna_channel is None:
        fallback_reason = "sionna_channel_tensor_missing"
        metadata["fallback_reason"] = fallback_reason
        return None, metadata, False, fallback_reason

    pilot_indices = set(getattr(resource_grid.pilot_pattern, "_pilot_ofdm_symbol_indices", []) or [])
    data_symbol_indices = [idx for idx in range(int(resource_grid.num_ofdm_symbols)) if idx not in pilot_indices]
    if not data_symbol_indices:
        fallback_reason = "resource_grid_has_no_data_ofdm_symbol"
        metadata["fallback_reason"] = fallback_reason
        return None, metadata, False, fallback_reason
    data_symbol_index = int(data_symbol_indices[0])
    converted, convert_meta = convert_sionna_h_to_project_h_f(
        sionna_channel,
        effective_subcarrier_ind=list(resource_grid.effective_subcarrier_ind),
        data_symbol_index=data_symbol_index,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
    )
    metadata["conversion"] = convert_meta
    if converted is None:
        fallback_reason = str(convert_meta.get("fallback_reason", "conversion_failed"))
        metadata["fallback_reason"] = fallback_reason
        return None, metadata, False, fallback_reason

    validation = validate_extracted_h_f(converted)
    metadata["validation"] = validation
    if not validation["valid"]:
        fallback_reason = str(validation.get("fallback_reason", "validation_failed"))
        metadata["fallback_reason"] = fallback_reason
        return None, metadata, False, fallback_reason
    metadata["selected_data_symbol_index"] = data_symbol_index
    metadata["fallback_reason"] = ""
    return converted, metadata, True, ""
