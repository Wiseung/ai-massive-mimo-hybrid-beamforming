"""Optional bridge helpers between Sionna native precoder APIs and PrecoderOutput."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from beamforming.utils.complex_ops import normalize_power
from beamforming.utils.csi_interface import ExtractedCSI, as_project_h_f, summarize_csi_input
from beamforming.utils.precoder_interface import PrecoderOutput, build_precoder_output, compare_precoder_outputs
from beamforming.utils.sionna_native_beamforming_chain import compute_project_precoder_per_subcarrier, summarize_receiver_config
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device


def _fallback(
    reason: str,
    *,
    csi_summary: dict[str, Any] | None = None,
    shape_assumptions: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "success": False,
        "fallback_used": True,
        "fallback_reason": reason,
        "shape_assumptions": shape_assumptions or {},
        "csi_summary": csi_summary,
    }
    if extra:
        payload.update(extra)
    return payload


def map_extracted_csi_to_sionna_precoder_inputs(
    csi: ExtractedCSI,
    *,
    device: torch.device,
    num_ofdm_symbols: int | None = None,
    fft_size: int | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Map ``ExtractedCSI`` to a minimal Sionna ``RZFPrecoder`` input contract."""

    if not isinstance(csi, ExtractedCSI):
        return _fallback(f"expected_ExtractedCSI_input_got_{type(csi).__name__}")

    h_f = csi.to_project_h_f()
    batch_size, num_subcarriers, num_users, num_bs_ant = h_f.shape
    num_ofdm_symbols = int(num_ofdm_symbols or num_users)
    fft_size = int(fft_size or num_subcarriers)
    shape_assumptions = {
        "project_h_f_shape": [int(x) for x in h_f.shape],
        "project_axes": {"B": 0, "Nsc": 1, "K": 2, "Nt": 3},
        "sionna_rzf_x_shape": [batch_size, 1, num_users, num_ofdm_symbols, fft_size],
        "sionna_rzf_h_shape": [batch_size, num_users, 1, 1, num_bs_ant, num_ofdm_symbols, fft_size],
        "one_tx_assumed": True,
        "one_rx_ant_per_user_assumed": True,
        "num_streams_per_tx_equals_num_users": True,
        "ofdm_symbol_axis_replicated_from_single_extracted_symbol": True,
        "fft_axis_matches_extracted_effective_subcarriers_only": fft_size == num_subcarriers,
    }
    if fft_size != num_subcarriers:
        return _fallback(
            "fft_size_must_match_extracted_num_subcarriers_for_current_probe",
            csi_summary=csi.summary_dict(),
            shape_assumptions=shape_assumptions,
        )

    ResourceGrid, _, rg_error = load_component("ResourceGrid")
    StreamManagement, _, sm_error = load_component("StreamManagement")
    if ResourceGrid is None or StreamManagement is None:
        return _fallback(
            rg_error or sm_error or "resource_grid_or_stream_management_unavailable",
            csi_summary=csi.summary_dict(),
            shape_assumptions=shape_assumptions,
        )
    resource_grid = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=15_000.0,
        num_tx=1,
        num_streams_per_tx=num_users,
        num_guard_carriers=(0, 0),
        dc_null=False,
        pilot_pattern=None,
        device=resolve_sionna_device(device),
    )
    stream_management = StreamManagement(np.ones((num_users, 1), dtype=int), num_streams_per_tx=num_users)
    rg_meta = {
        "probe_only_resource_grid": True,
        "num_ofdm_symbols": int(num_ofdm_symbols),
        "fft_size": int(fft_size),
        "num_tx": 1,
        "num_streams_per_tx": int(num_users),
        "pilot_pattern": None,
        "dc_null": False,
        "num_data_symbols": int(resource_grid.num_data_symbols),
    }

    x = torch.zeros(
        batch_size,
        1,
        num_users,
        num_ofdm_symbols,
        fft_size,
        dtype=torch.complex64,
        device=device,
    )
    stream_eye = torch.eye(num_users, dtype=torch.complex64, device=device)
    for stream_idx in range(num_users):
        x[:, 0, :, stream_idx, :] = stream_eye[:, stream_idx].view(1, num_users, 1)

    h = h_f.permute(0, 2, 3, 1).unsqueeze(2).unsqueeze(3).unsqueeze(5).repeat(1, 1, 1, 1, 1, num_ofdm_symbols, 1)
    h = h.contiguous().to(torch.complex64)
    alpha_value = float(alpha) if alpha is not None else float(num_users)

    return {
        "success": True,
        "fallback_used": False,
        "fallback_reason": "",
        "csi_summary": csi.summary_dict(),
        "project_h_f": h_f,
        "resource_grid": resource_grid,
        "stream_management": stream_management,
        "x": x,
        "h": h,
        "alpha": alpha_value,
        "shape_assumptions": shape_assumptions,
        "receiver_config": summarize_receiver_config(
            resource_grid,
            stream_management,
            selected_ofdm_symbol=csi.selected_ofdm_symbol,
            effective_subcarrier_indices=csi.effective_subcarrier_indices,
        ),
        "resource_grid_meta": rg_meta,
        "stream_management_requirements": {
            "rx_tx_association": np.ones((num_users, 1), dtype=int).tolist(),
            "num_streams_per_tx": int(num_users),
            "num_tx": 1,
            "num_rx": int(num_users),
        },
    }


def map_sionna_precoder_output_to_precoder_output(
    sionna_output: torch.Tensor,
    *,
    input_csi: ExtractedCSI,
    source_component: str = "RZFPrecoder",
    metadata: dict[str, Any] | None = None,
) -> tuple[PrecoderOutput | None, dict[str, Any]]:
    """Convert Sionna native precoder output to ``PrecoderOutput`` if shape-compatible."""

    csi_summary = summarize_csi_input(input_csi)
    shape_assumptions = {
        "expected_sionna_output_rank": 5,
        "expected_sionna_axes": ["B", "num_tx", "Nt", "num_ofdm_symbols", "fft_size"],
        "target_project_axes": ["B", "Nsc", "Nt", "K"],
        "mapping_rule": "tx=0 then permute to (B, fft_size, Nt, num_ofdm_symbols)",
        "requires_num_ofdm_symbols_equals_num_users": True,
        "requires_fft_size_equals_num_subcarriers": True,
    }
    if sionna_output.ndim != 5:
        return None, _fallback(
            f"expected_sionna_precoder_output_rank_5_got_{sionna_output.ndim}",
            csi_summary=csi_summary,
            shape_assumptions=shape_assumptions,
        )
    if int(sionna_output.size(1)) != 1:
        return None, _fallback(
            "sionna_precoder_output_num_tx_must_equal_1_for_current_bridge",
            csi_summary=csi_summary,
            shape_assumptions=shape_assumptions,
            extra={"observed_shape": [int(x) for x in sionna_output.shape]},
        )
    if int(sionna_output.size(3)) != int(input_csi.num_users):
        return None, _fallback(
            "sionna_precoder_output_num_ofdm_symbols_must_equal_num_users_for_current_bridge",
            csi_summary=csi_summary,
            shape_assumptions=shape_assumptions,
            extra={
                "observed_num_ofdm_symbols": int(sionna_output.size(3)),
                "num_users": int(input_csi.num_users),
            },
        )
    if int(sionna_output.size(4)) != int(input_csi.num_subcarriers):
        return None, _fallback(
            "sionna_precoder_output_fft_size_must_equal_num_subcarriers_for_current_bridge",
            csi_summary=csi_summary,
            shape_assumptions=shape_assumptions,
            extra={
                "observed_fft_size": int(sionna_output.size(4)),
                "num_subcarriers": int(input_csi.num_subcarriers),
            },
        )

    raw_f_f = sionna_output[:, 0].permute(0, 3, 1, 2).contiguous()
    normalized_f_f = normalize_power(raw_f_f, target_power=1.0)
    combined_metadata = dict(metadata or {})
    combined_metadata.update(
        {
            "source_component": source_component,
            "shape_assumptions": shape_assumptions,
            "raw_sionna_output_shape": [int(x) for x in sionna_output.shape],
            "converted_project_f_f_shape": [int(x) for x in normalized_f_f.shape],
            "native_project_power_before_normalization": float(
                ((torch.abs(raw_f_f) ** 2).sum(dim=(-2, -1))).mean().item()
            ),
            "native_project_power_after_normalization": float(
                ((torch.abs(normalized_f_f) ** 2).sum(dim=(-2, -1))).mean().item()
            ),
            "input_csi_source": csi_summary.get("source"),
            "input_h_f_shape": csi_summary.get("h_f_shape"),
            "fallback_reason": "",
        }
    )
    precoder = build_precoder_output(
        f_f=normalized_f_f,
        source="sionna_rzf_precoder",
        method="sionna_rzf_precoder",
        input_csi=input_csi,
        project_side_precoder=False,
        sionna_native_precoder=True,
        teacher_used_during_inference=False,
        power_normalized=True,
        full_native_only=False,
        metadata=combined_metadata,
    )
    return precoder, {
        "success": True,
        "fallback_used": False,
        "fallback_reason": "",
        "shape_assumptions": shape_assumptions,
        "power_norm_before_normalization": combined_metadata["native_project_power_before_normalization"],
        "power_norm_after_normalization": combined_metadata["native_project_power_after_normalization"],
    }


def compare_sionna_precoder_output_with_project_precoder_output(
    csi_input: ExtractedCSI | torch.Tensor | dict[str, Any],
    *,
    project_method: str = "rzf",
    project_noise_var: float | torch.Tensor | None = None,
    sionna_precoder_output: PrecoderOutput | torch.Tensor | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare one Sionna-native precoder path against the project RZF path."""

    h_f, csi_meta = as_project_h_f(csi_input)
    csi_summary = summarize_csi_input(csi_input)
    if project_noise_var is None:
        raise ValueError("project_noise_var_is_required_for_precoder_comparison")
    project_output = compute_project_precoder_per_subcarrier(
        project_method,
        csi_input if isinstance(csi_input, (ExtractedCSI, dict)) else h_f,
        project_noise_var,
        return_precoder_output=True,
    )
    summary: dict[str, Any] = {
        "project_method": f"project_{project_method}",
        "project_shape": [int(x) for x in project_output.f_f.shape],
        "project_power_norm": project_output.power_norm,
        "project_csi_interface_used": csi_summary.get("csi_interface_used"),
        "project_input_type": csi_meta.get("input_type"),
        "sionna_summary": None,
        "shape_compatible": False,
        "power_norm_close": False,
        "max_abs_diff": None,
        "comparison_valid": False,
        "fallback_reason": "",
    }
    if sionna_precoder_output is None:
        summary["fallback_reason"] = "missing_sionna_precoder_output"
        return summary

    comparison = compare_precoder_outputs(project_output, sionna_precoder_output)
    sionna_summary = (
        sionna_precoder_output.summary_dict()
        if isinstance(sionna_precoder_output, PrecoderOutput)
        else {
            "input_summary": comparison["b"],
        }
    )
    project_power_mean = float((project_output.power_norm or {}).get("mean", 0.0))
    sionna_power_mean = 0.0
    if isinstance(sionna_precoder_output, PrecoderOutput):
        sionna_power_mean = float((sionna_precoder_output.power_norm or {}).get("mean", 0.0))
    summary.update(
        {
            "sionna_summary": sionna_summary,
            "shape_compatible": bool(comparison["same_shape"]),
            "power_norm_close": abs(project_power_mean - sionna_power_mean) <= 1e-4,
            "max_abs_diff": comparison["max_abs_diff"],
            "comparison_valid": bool(comparison["same_shape"]),
            "fallback_reason": "" if comparison["same_shape"] else "shape_mismatch_between_project_and_sionna_precoder_output",
            "comparison": comparison,
        }
    )
    return summary


def run_sionna_rzf_precoder_probe(
    csi: ExtractedCSI,
    *,
    project_noise_var: float,
    device: torch.device,
) -> dict[str, Any]:
    """Probe Sionna ``RZFPrecoder`` on one ``ExtractedCSI`` object."""

    RZFPrecoder, _, rzf_error = load_component("RZFPrecoder")
    payload: dict[str, Any] = {
        "sionna_rzf_available": RZFPrecoder is not None,
        "sionna_rzf_callable": False,
        "sionna_precoder_success": False,
        "converted_to_precoder_output": False,
        "native_receiver_success_if_attempted": False,
        "probe_only": True,
        "fallback_used": True,
        "fallback_reason": "",
        "recommended_next_step": "keep_project_side_precoder_output",
        "input_csi_summary": csi.summary_dict(),
    }
    if RZFPrecoder is None:
        payload["fallback_reason"] = rzf_error or "RZFPrecoder_unavailable"
        return payload

    mapped = map_extracted_csi_to_sionna_precoder_inputs(csi, device=device, alpha=float(csi.num_users * project_noise_var))
    payload["shape_mapping"] = mapped
    if not mapped.get("success", False):
        payload["fallback_reason"] = str(mapped.get("fallback_reason", "mapping_failed"))
        payload["recommended_next_step"] = "adapter_bridge_only"
        return payload

    try:
        precoder = RZFPrecoder(
            mapped["resource_grid"],
            mapped["stream_management"],
            return_effective_channel=True,
            device=resolve_sionna_device(device),
        )
        sionna_output, h_eff = precoder(mapped["x"], mapped["h"], alpha=mapped["alpha"])
        payload["sionna_rzf_callable"] = True
        payload["sionna_precoder_success"] = True
        payload["probe_only"] = False
        payload["sionna_output_shape"] = [int(x) for x in sionna_output.shape]
        payload["effective_channel_shape"] = [int(x) for x in h_eff.shape]
    except Exception as exc:  # pragma: no cover - optional runtime path
        payload["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        payload["recommended_next_step"] = "adapter_bridge_only"
        return payload

    converted, convert_meta = map_sionna_precoder_output_to_precoder_output(
        sionna_output,
        input_csi=csi,
        metadata={
            "effective_channel_shape": payload.get("effective_channel_shape"),
            "mapping_receiver_config": mapped.get("receiver_config"),
        },
    )
    payload["conversion"] = convert_meta
    if converted is None:
        payload["fallback_reason"] = str(convert_meta.get("fallback_reason", "conversion_failed"))
        payload["recommended_next_step"] = "adapter_bridge_only"
        return payload

    comparison = compare_sionna_precoder_output_with_project_precoder_output(
        csi,
        project_method="rzf",
        project_noise_var=project_noise_var,
        sionna_precoder_output=converted,
    )
    payload.update(
        {
            "converted_to_precoder_output": True,
            "fallback_used": False,
            "fallback_reason": "",
            "sionna_precoder_output": converted,
            "comparison": comparison,
            "recommended_next_step": "adapter_bridge_then_optional_native_receiver_probe",
        }
    )
    return payload
