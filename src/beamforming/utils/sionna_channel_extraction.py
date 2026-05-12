"""Utilities for extracting project-side H_f from Sionna channel tensors."""

from __future__ import annotations

from typing import Any

import torch

from beamforming.utils.csi_interface import ExtractedCSI, SharedSionnaOFDMBatch


def _pilot_ofdm_symbol_indices(resource_grid: Any) -> list[int]:
    return [int(x) for x in (getattr(resource_grid, "_pilot_ofdm_symbol_indices", []) or [])]


def resolve_resource_grid_data_symbol_indices(resource_grid: Any) -> list[int]:
    """Return OFDM symbol indices that are not reserved for pilots."""
    pilot_indices = set(_pilot_ofdm_symbol_indices(resource_grid))
    return [idx for idx in range(int(resource_grid.num_ofdm_symbols)) if idx not in pilot_indices]


def resolve_selected_ofdm_symbol_indices(
    resource_grid: Any,
    selected_ofdm_symbol: str | int = "first_data",
) -> tuple[list[int], dict[str, Any], str]:
    """Resolve requested OFDM-symbol selection to explicit data-bearing indices."""
    data_symbol_indices = resolve_resource_grid_data_symbol_indices(resource_grid)
    meta: dict[str, Any] = {
        "requested_selected_ofdm_symbol": selected_ofdm_symbol,
        "data_symbol_indices": [int(x) for x in data_symbol_indices],
        "selected_data_symbol_indices": [],
        "selection_mode": str(selected_ofdm_symbol),
    }
    if not data_symbol_indices:
        return [], meta, "resource_grid_has_no_data_ofdm_symbol"
    if isinstance(selected_ofdm_symbol, int):
        if selected_ofdm_symbol not in data_symbol_indices:
            return [], meta, "selected_ofdm_symbol_is_not_data_bearing"
        meta["selected_data_symbol_indices"] = [int(selected_ofdm_symbol)]
        meta["selection_mode"] = "explicit_index"
        return [int(selected_ofdm_symbol)], meta, ""
    if selected_ofdm_symbol == "first_data":
        out = [int(data_symbol_indices[0])]
    elif selected_ofdm_symbol == "last_data":
        out = [int(data_symbol_indices[-1])]
    elif selected_ofdm_symbol == "all_data_average":
        out = [int(x) for x in data_symbol_indices]
    else:
        return [], meta, f"unsupported_selected_ofdm_symbol_{selected_ofdm_symbol}"
    meta["selected_data_symbol_indices"] = out
    return out, meta, ""


def resolve_selected_effective_subcarrier_indices(
    resource_grid: Any,
    effective_subcarriers: str | list[int] | torch.Tensor = "all_effective",
) -> tuple[list[int], dict[str, Any], str]:
    """Resolve requested effective-subcarrier selection to explicit FFT-bin indices."""
    all_effective = [int(x) for x in resource_grid.effective_subcarrier_ind]
    meta: dict[str, Any] = {
        "requested_effective_subcarriers": effective_subcarriers,
        "all_effective_subcarrier_indices": all_effective,
        "selected_effective_subcarrier_indices": [],
        "selection_mode": str(effective_subcarriers),
    }
    if isinstance(effective_subcarriers, torch.Tensor):
        selected = [int(x) for x in effective_subcarriers.detach().cpu().tolist()]
        meta["selection_mode"] = "tensor_indices"
        meta["selected_effective_subcarrier_indices"] = selected
        return selected, meta, ""
    if isinstance(effective_subcarriers, list):
        selected = [int(x) for x in effective_subcarriers]
        meta["selection_mode"] = "list_indices"
        meta["selected_effective_subcarrier_indices"] = selected
        return selected, meta, ""
    if effective_subcarriers == "all_effective":
        meta["selected_effective_subcarrier_indices"] = all_effective
        return all_effective, meta, ""
    if effective_subcarriers.startswith("center_"):
        try:
            requested = int(effective_subcarriers.split("_", 1)[1])
        except ValueError:
            return [], meta, f"unsupported_effective_subcarriers_{effective_subcarriers}"
        if requested <= 0:
            return [], meta, "center_subcarrier_count_must_be_positive"
        if requested > len(all_effective):
            return [], meta, "requested_center_subcarrier_count_exceeds_available_effective_subcarriers"
        start = (len(all_effective) - requested) // 2
        selected = all_effective[start : start + requested]
        meta["selected_effective_subcarrier_indices"] = selected
        return selected, meta, ""
    return [], meta, f"unsupported_effective_subcarriers_{effective_subcarriers}"


def summarize_h_f_matrix_stats(h_f: torch.Tensor | None) -> dict[str, Any]:
    """Return norm/rank/condition-number statistics for ``H_f=(B,Nsc,K,Nt)``."""
    stats: dict[str, Any] = {
        "available": False,
        "rank_mean": None,
        "rank_min": None,
        "rank_max": None,
        "condition_number_mean": None,
        "condition_number_max": None,
        "fro_norm_mean": None,
        "fro_norm_std": None,
    }
    if h_f is None or h_f.ndim != 4 or not torch.is_complex(h_f):
        return stats
    ranks = torch.linalg.matrix_rank(h_f).float()
    sv = torch.linalg.svdvals(h_f)
    sigma_max = sv[..., 0]
    sigma_min = sv[..., -1].clamp_min(1e-12)
    cond = sigma_max / sigma_min
    norms = torch.linalg.vector_norm(h_f, dim=(-2, -1))
    stats.update(
        {
            "available": True,
            "rank_mean": float(ranks.mean().item()),
            "rank_min": int(ranks.min().item()),
            "rank_max": int(ranks.max().item()),
            "condition_number_mean": float(cond.mean().item()),
            "condition_number_max": float(cond.max().item()),
            "fro_norm_mean": float(norms.mean().item()),
            "fro_norm_std": float(norms.std(unbiased=False).item()),
        }
    )
    return stats


def build_csi_from_h_f(
    h_f: torch.Tensor,
    *,
    source: str,
    source_component: str,
    selected_ofdm_symbol: int,
    effective_subcarrier_indices: list[int],
    num_users: int,
    num_bs_ant: int,
    project_h_f_assisted: bool,
    extracted_h_f_used: bool,
    full_native_only: bool,
    metadata: dict[str, Any],
) -> ExtractedCSI:
    """Create a validated ``ExtractedCSI`` object from a project-side ``H_f`` tensor."""
    csi = ExtractedCSI(
        h_f=h_f.contiguous(),
        source=source,
        source_component=source_component,
        axes={"B": 0, "Nsc": 1, "K": 2, "Nt": 3},
        shape={
            "B": int(h_f.size(0)),
            "Nsc": int(h_f.size(1)),
            "K": int(h_f.size(2)),
            "Nt": int(h_f.size(3)),
        },
        selected_ofdm_symbol=int(selected_ofdm_symbol),
        effective_subcarrier_indices=[int(x) for x in effective_subcarrier_indices],
        num_users=int(num_users),
        num_bs_ant=int(num_bs_ant),
        num_subcarriers=int(h_f.size(1)),
        project_h_f_assisted=bool(project_h_f_assisted),
        extracted_h_f_used=bool(extracted_h_f_used),
        full_native_only=bool(full_native_only),
        metadata=dict(metadata),
    )
    validation = csi.validate(raise_on_error=False)
    csi.metadata.setdefault("validation", validation)
    return csi


def convert_sionna_h_to_project_h_f(
    sionna_channel: torch.Tensor,
    *,
    effective_subcarrier_ind: list[int] | torch.Tensor,
    data_symbol_indices: list[int] | torch.Tensor,
    num_users: int,
    num_bs_ant: int,
    normalize_channel: bool = False,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Convert a Sionna OFDM channel tensor to project ``H_f=(B,Nsc,K,Nt)``.

    Expected Sionna channel axes:
    ``(batch, rx, rx_ant, tx, tx_ant, ofdm_symbol, fft_bin)``.

    Project axes:
    ``(batch, subcarrier, user, bs_ant)``.
    """
    meta: dict[str, Any] = {
        "input_shape": [int(x) for x in sionna_channel.shape],
        "assumed_sionna_axes": ["batch", "rx", "rx_ant", "tx", "tx_ant", "ofdm_symbol", "fft_bin"],
        "target_project_axes": ["batch", "subcarrier", "user", "bs_ant"],
        "target_project_axes_map": {"B": 0, "Nsc": 1, "K": 2, "Nt": 3},
        "original_axes": {
            "batch": 0,
            "rx": 1,
            "rx_ant": 2,
            "tx": 3,
            "tx_ant": 4,
            "ofdm_symbol": 5,
            "fft_bin": 6,
        },
        "fallback_reason": "",
        "selected_data_symbol_indices": [int(x) for x in torch.as_tensor(data_symbol_indices, dtype=torch.long).tolist()],
        "normalize_channel": bool(normalize_channel),
    }
    if sionna_channel.ndim != 7:
        meta["fallback_reason"] = f"expected_rank_7_sionna_channel_got_{sionna_channel.ndim}"
        return None, meta
    if not torch.is_complex(sionna_channel):
        meta["fallback_reason"] = "sionna_channel_is_not_complex"
        return None, meta

    eff_idx = torch.as_tensor(effective_subcarrier_ind, device=sionna_channel.device, dtype=torch.long)
    data_idx = torch.as_tensor(data_symbol_indices, device=sionna_channel.device, dtype=torch.long)
    if eff_idx.ndim != 1:
        meta["fallback_reason"] = "effective_subcarrier_ind_must_be_1d"
        return None, meta
    if data_idx.ndim != 1:
        meta["fallback_reason"] = "data_symbol_indices_must_be_1d"
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
    if data_idx.numel() == 0:
        meta["fallback_reason"] = "data_symbol_indices_empty"
        return None, meta
    if int(torch.min(data_idx).item()) < 0 or int(torch.max(data_idx).item()) >= int(sionna_channel.size(5)):
        meta["fallback_reason"] = "data_symbol_indices_out_of_range"
        return None, meta
    if eff_idx.numel() == 0:
        meta["fallback_reason"] = "effective_subcarrier_ind_empty"
        return None, meta
    if int(torch.min(eff_idx).item()) < 0 or int(torch.max(eff_idx).item()) >= int(sionna_channel.size(6)):
        meta["fallback_reason"] = "effective_subcarrier_ind_out_of_range"
        return None, meta

    h_sel = sionna_channel[:, :, 0, 0, :, :, :]
    h_perm = h_sel.permute(0, 3, 4, 1, 2).contiguous()
    h_pick = h_perm[:, data_idx, :, :, :]
    h_pick = h_pick[:, :, eff_idx, :, :]
    if h_pick.size(1) == 1:
        h_f = h_pick[:, 0, :, :, :].contiguous()
    else:
        h_f = h_pick.mean(dim=1).contiguous()
    if normalize_channel:
        h_norm = torch.linalg.vector_norm(h_f, dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        h_f = h_f / h_norm
        meta["normalization_rule"] = "per_batch_per_subcarrier_fro_norm_to_one"
    meta["output_shape"] = [int(x) for x in h_f.shape]
    meta["selected_effective_subcarrier_count"] = int(eff_idx.numel())
    return h_f, meta


def validate_extracted_h_f(h_f: torch.Tensor | ExtractedCSI | None) -> dict[str, Any]:
    """Validate an extracted project-side ``H_f`` tensor or ``ExtractedCSI``."""
    if isinstance(h_f, ExtractedCSI):
        result = h_f.validate(raise_on_error=False)
        result["matrix_stats"] = summarize_h_f_matrix_stats(h_f.h_f)
        return result

    result: dict[str, Any] = {
        "valid": False,
        "shape": None,
        "dtype": None,
        "is_complex": False,
        "all_finite": False,
        "norm_mean": None,
        "norm_std": None,
        "subcarrier_norm_std_mean": None,
        "matrix_stats": None,
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
    result["subcarrier_norm_std_mean"] = float(norms.std(dim=1, unbiased=False).mean().item())
    result["matrix_stats"] = summarize_h_f_matrix_stats(h_f)
    result["valid"] = True
    return result


def compare_extracted_h_f_with_synthetic_reference(
    extracted_h_f: torch.Tensor | ExtractedCSI | None,
    reference_h_f: torch.Tensor | ExtractedCSI | None,
) -> dict[str, Any]:
    """Compare extracted ``H_f`` against a synthetic reference statistically."""
    extracted_tensor = extracted_h_f.h_f if isinstance(extracted_h_f, ExtractedCSI) else extracted_h_f
    reference_tensor = reference_h_f.h_f if isinstance(reference_h_f, ExtractedCSI) else reference_h_f
    result: dict[str, Any] = {
        "comparison_valid": False,
        "shape_match": False,
        "mean_norm_ratio": None,
        "rank_mean_extracted": None,
        "rank_mean_reference": None,
        "fallback_reason": "",
    }
    if extracted_tensor is None or reference_tensor is None:
        result["fallback_reason"] = "missing_extracted_or_reference_h_f"
        return result
    result["shape_match"] = list(extracted_tensor.shape) == list(reference_tensor.shape)
    if not result["shape_match"]:
        result["fallback_reason"] = "shape_mismatch"
        return result
    ex_norm = torch.linalg.vector_norm(extracted_tensor, dim=(-2, -1)).mean()
    ref_norm = torch.linalg.vector_norm(reference_tensor, dim=(-2, -1)).mean().clamp_min(1e-12)
    result["mean_norm_ratio"] = float((ex_norm / ref_norm).item())
    ex_rank = torch.linalg.matrix_rank(extracted_tensor).float().mean()
    ref_rank = torch.linalg.matrix_rank(reference_tensor).float().mean()
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
    selected_ofdm_symbol: str | int = "first_data",
    effective_subcarriers: str | list[int] | torch.Tensor = "all_effective",
    normalize_channel: bool = False,
    return_csi: bool = False,
) -> tuple[torch.Tensor | ExtractedCSI | None, dict[str, Any], bool, str]:
    """Extract project ``H_f=(B,Nsc,K,Nt)`` from a Sionna channel tensor."""
    metadata: dict[str, Any] = {
        "resource_grid_num_ofdm_symbols": int(resource_grid.num_ofdm_symbols),
        "resource_grid_num_data_symbols": int(resource_grid.num_data_symbols),
        "resource_grid_effective_subcarrier_ind": [int(x) for x in resource_grid.effective_subcarrier_ind],
        "resource_grid_pilot_ofdm_symbol_indices": _pilot_ofdm_symbol_indices(resource_grid),
        "shape_assumptions": {
            "sionna_axes": ["batch", "rx", "rx_ant", "tx", "tx_ant", "ofdm_symbol", "fft_bin"],
            "project_axes": ["batch", "subcarrier", "user", "bs_ant"],
            "rx_maps_to_user": True,
            "tx_ant_maps_to_bs_ant": True,
            "single_tx_required": True,
            "single_rx_ant_required": True,
        },
        "csi_interface_used": bool(return_csi),
    }
    if sionna_channel is None:
        fallback_reason = "sionna_channel_tensor_missing"
        metadata["fallback_reason"] = fallback_reason
        return None, metadata, False, fallback_reason

    data_symbol_indices, symbol_meta, symbol_reason = resolve_selected_ofdm_symbol_indices(
        resource_grid,
        selected_ofdm_symbol=selected_ofdm_symbol,
    )
    metadata["selected_ofdm_symbol"] = symbol_meta
    if symbol_reason:
        fallback_reason = symbol_reason
        metadata["fallback_reason"] = fallback_reason
        return None, metadata, False, fallback_reason

    selected_effective_subcarrier_ind, subcarrier_meta, subcarrier_reason = resolve_selected_effective_subcarrier_indices(
        resource_grid,
        effective_subcarriers=effective_subcarriers,
    )
    metadata["selected_effective_subcarriers"] = subcarrier_meta
    if subcarrier_reason:
        fallback_reason = subcarrier_reason
        metadata["fallback_reason"] = fallback_reason
        return None, metadata, False, fallback_reason

    converted, convert_meta = convert_sionna_h_to_project_h_f(
        sionna_channel,
        effective_subcarrier_ind=selected_effective_subcarrier_ind,
        data_symbol_indices=data_symbol_indices,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        normalize_channel=normalize_channel,
    )
    metadata["conversion"] = convert_meta
    metadata["original_sionna_h_shape"] = [int(x) for x in sionna_channel.shape]
    metadata["original_axes"] = convert_meta.get("original_axes")
    metadata["selected_data_symbol_indices"] = [int(x) for x in data_symbol_indices]
    metadata["selected_data_symbol_index"] = int(data_symbol_indices[0]) if len(data_symbol_indices) == 1 else None
    metadata["selected_effective_subcarrier_indices"] = [int(x) for x in selected_effective_subcarrier_ind]
    metadata["selected_effective_subcarrier_count"] = int(len(selected_effective_subcarrier_ind))
    metadata["normalize_channel"] = bool(normalize_channel)
    metadata["effective_subcarrier_ind"] = [int(x) for x in selected_effective_subcarrier_ind]
    metadata["selected_data_symbol"] = int(data_symbol_indices[0]) if len(data_symbol_indices) == 1 else None
    metadata["pilot_symbol_indices"] = _pilot_ofdm_symbol_indices(resource_grid)

    if converted is None:
        fallback_reason = str(convert_meta.get("fallback_reason", "conversion_failed"))
        metadata["fallback_reason"] = fallback_reason
        metadata["extraction_success"] = False
        return None, metadata, False, fallback_reason

    csi = build_csi_from_h_f(
        converted,
        source="sionna_ofdm_channel",
        source_component="OFDMChannel(return_channel=True)",
        selected_ofdm_symbol=int(data_symbol_indices[0]) if len(data_symbol_indices) == 1 else int(data_symbol_indices[0]),
        effective_subcarrier_indices=selected_effective_subcarrier_ind,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        project_h_f_assisted=False,
        extracted_h_f_used=True,
        full_native_only=False,
        metadata={
            "original_sionna_h_shape": [int(x) for x in sionna_channel.shape],
            "original_axes": convert_meta.get("original_axes"),
            "selected_data_symbol": int(data_symbol_indices[0]) if len(data_symbol_indices) == 1 else None,
            "selected_data_symbol_indices": [int(x) for x in data_symbol_indices],
            "pilot_symbol_indices": _pilot_ofdm_symbol_indices(resource_grid),
            "effective_subcarrier_ind": [int(x) for x in selected_effective_subcarrier_ind],
            "extraction_success": True,
            "fallback_reason": "",
            "conversion_meta": convert_meta,
            "resource_grid_num_ofdm_symbols": int(resource_grid.num_ofdm_symbols),
            "resource_grid_num_data_symbols": int(resource_grid.num_data_symbols),
        },
    )
    validation = validate_extracted_h_f(csi)
    metadata["validation"] = validation
    metadata["csi_summary"] = csi.summary_dict()
    metadata["extraction_success"] = bool(validation["valid"])
    if not validation["valid"]:
        fallback_reason = str(validation.get("fallback_reason", "validation_failed"))
        metadata["fallback_reason"] = fallback_reason
        return None, metadata, False, fallback_reason
    metadata["fallback_reason"] = ""
    return (csi if return_csi else converted), metadata, True, ""


def create_shared_sionna_ofdm_batch(
    *,
    batch_size: int,
    snr_db: float,
    resource_grid: Any,
    stream_management: Any,
    sionna_channel_tensor: torch.Tensor,
    num_users: int,
    num_bs_ant: int,
    selected_ofdm_symbol: str | int = "first_data",
    effective_subcarriers: str | list[int] | torch.Tensor = "all_effective",
    normalize_channel: bool = False,
    seed: int = 0,
    device: torch.device | None = None,
) -> SharedSionnaOFDMBatch:
    """Create a deterministic shared batch reused by raw-H and CSI-backed paths."""
    if device is None:
        device = sionna_channel_tensor.device if sionna_channel_tensor is not None else torch.device("cpu")
    bits = torch.randint(0, 2, (batch_size, len(resource_grid.effective_subcarrier_ind), num_users, 2), device=device)
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    symbols = ((real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=device))).to(torch.complex64)
    noise_var = float(10.0 ** (-float(snr_db) / 10.0))
    csi_or_h_f, meta, success, fallback_reason = extract_h_f_from_sionna_channel(
        sionna_channel_tensor,
        resource_grid=resource_grid,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        selected_ofdm_symbol=selected_ofdm_symbol,
        effective_subcarriers=effective_subcarriers,
        normalize_channel=normalize_channel,
        return_csi=True,
    )
    if not success or csi_or_h_f is None:
        raise RuntimeError(f"Failed to create shared Sionna OFDM batch: {fallback_reason}")
    csi = csi_or_h_f
    extracted_h_f = csi.to_project_h_f()
    rx_noise_real = torch.randn(
        batch_size,
        num_users,
        1,
        int(resource_grid.num_ofdm_symbols),
        int(resource_grid.fft_size),
        device=device,
    )
    rx_noise_imag = torch.randn(
        batch_size,
        num_users,
        1,
        int(resource_grid.num_ofdm_symbols),
        int(resource_grid.fft_size),
        device=device,
    )
    rx_noise_grid = (
        (rx_noise_real + 1j * rx_noise_imag)
        * torch.sqrt(torch.tensor(noise_var / 2.0, dtype=torch.float32, device=device))
    ).to(torch.complex64)
    selected_symbol = int(csi.selected_ofdm_symbol)
    batch = SharedSionnaOFDMBatch(
        bits=bits.to(torch.int64),
        symbols=symbols,
        resource_grid=resource_grid,
        stream_management=stream_management,
        sionna_channel_tensor=sionna_channel_tensor.contiguous(),
        extracted_h_f=extracted_h_f,
        csi=csi,
        rx_noise_grid=rx_noise_grid,
        noise_var=noise_var,
        snr_db=float(snr_db),
        seed=int(seed),
        selected_ofdm_symbol=selected_symbol,
        effective_subcarrier_indices=[int(x) for x in csi.effective_subcarrier_indices],
        metadata={
            "seed": int(seed),
            "batch_size": int(batch_size),
            "snr_db": float(snr_db),
            "selected_ofdm_symbol": int(selected_symbol),
            "effective_subcarrier_indices": [int(x) for x in csi.effective_subcarrier_indices],
            "original_sionna_h_shape": [int(x) for x in sionna_channel_tensor.shape],
            "extracted_h_f_shape": [int(x) for x in extracted_h_f.shape],
            "resource_grid_num_ofdm_symbols": int(resource_grid.num_ofdm_symbols),
            "resource_grid_fft_size": int(resource_grid.fft_size),
            "stream_management_present": stream_management is not None,
            "csi_summary": csi.summary_dict(),
        },
    )
    return batch
