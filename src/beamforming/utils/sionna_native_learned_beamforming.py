"""Learned beamformer adapters for the Sionna-native OFDM receiver chain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from beamforming.models.factory import build_model
from beamforming.utils.csi_interface import ExtractedCSI, as_project_h_f
from beamforming.utils.precoder_interface import PrecoderOutput, as_project_f_f, build_precoder_output
from beamforming.utils.sionna_native_beamforming_chain import (
    apply_project_precoder_to_sionna_grid,
    build_pilot_aware_multiuser_resource_grid,
    compute_project_metrics_from_sionna_rx,
    create_shared_sionna_ofdm_batch_from_generator,
    describe_tensor,
    extract_effective_channel_from_sionna,
    map_project_streams_to_sionna_rg,
    summarize_receiver_config,
    sionna_rx_to_project_symbols,
    validate_sionna_receiver_shapes,
)
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device
from beamforming.utils.sionna_ofdm_training import run_model_forward


DEFAULT_LEARNED_CHECKPOINTS = {
    "learned_residual_rzf": "outputs/runs/sionna_ofdm_residual_rzf/best.pt",
    "learned_residual_wmmse_distill": "outputs/runs/sionna_ofdm_residual_wmmse_distill/best.pt",
    "learned_unfolded_lite": "outputs/runs/sionna_ofdm_unfolded_lite/best.pt",
}


@dataclass
class LoadedLearnedBeamformer:
    method_name: str
    checkpoint_path: Path
    model_name: str
    model: torch.nn.Module
    config: dict[str, Any]
    checkpoint: dict[str, Any]


@dataclass
class NativeReceiverContext:
    bits: torch.Tensor
    stream_symbols: torch.Tensor
    resource_grid: Any
    stream_management: Any
    h_f: torch.Tensor
    csi: ExtractedCSI | None
    h_full: torch.Tensor
    noise_var: float
    snr_db: float
    device: torch.device
    context_meta: dict[str, Any]


@dataclass
class SharedSionnaChannelBundle:
    resource_grid: Any
    stream_management: Any
    h_full: torch.Tensor
    h_f: torch.Tensor
    csi: ExtractedCSI | None
    bits: torch.Tensor | None
    stream_symbols: torch.Tensor | None
    shared_batch_meta: dict[str, Any] | None
    noise_var: float
    bundle_meta: dict[str, Any]


def default_checkpoint_path(method_name: str, repo_root: Path) -> Path:
    if method_name not in DEFAULT_LEARNED_CHECKPOINTS:
        raise KeyError(f"Unsupported learned method: {method_name}")
    return repo_root / DEFAULT_LEARNED_CHECKPOINTS[method_name]


def load_learned_beamformer_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    method_name: str | None = None,
) -> LoadedLearnedBeamformer:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_cfg = dict(config["model"])
    dataset_cfg = config["dataset"]
    data_cfg = {
        "num_users": int(dataset_cfg["num_users"]),
        "num_bs_ant": int(dataset_cfg["num_bs_ant"]),
        "num_rf_chains": int(dataset_cfg.get("num_users", dataset_cfg["num_bs_ant"])),
    }
    model = build_model(model_cfg, data_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return LoadedLearnedBeamformer(
        method_name=method_name or str(model_cfg["name"]),
        checkpoint_path=checkpoint_path,
        model_name=str(model_cfg["name"]),
        model=model,
        config=config,
        checkpoint=checkpoint,
    )


def clone_native_receiver_context(
    context: NativeReceiverContext,
    *,
    h_f: torch.Tensor | None = None,
    csi: ExtractedCSI | None = None,
    h_full: torch.Tensor | None = None,
    context_meta_updates: dict[str, Any] | None = None,
) -> NativeReceiverContext:
    """Clone ``NativeReceiverContext`` while replacing shared channel tensors."""
    merged_meta = dict(context.context_meta)
    if context_meta_updates:
        merged_meta.update(context_meta_updates)
    return replace(
        context,
        h_f=context.h_f if h_f is None else h_f,
        csi=context.csi if csi is None else csi,
        h_full=context.h_full if h_full is None else h_full,
        context_meta=merged_meta,
    )


def infer_learned_precoder(
    bundle: LoadedLearnedBeamformer,
    h_f: ExtractedCSI | torch.Tensor | dict[str, Any],
    snr_db: torch.Tensor,
    *,
    native_receiver_path: bool,
    return_precoder_output: bool = False,
) -> tuple[torch.Tensor | PrecoderOutput, dict[str, Any], float]:
    h_f_tensor, csi_input_meta = as_project_h_f(h_f)
    start = perf_counter()
    with torch.no_grad():
        outputs = run_model_forward(bundle.model, h_f_tensor, snr_db)
    runtime_ms = (perf_counter() - start) * 1000.0
    precoder = outputs["precoder"]
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    meta = {
        "method_name": bundle.method_name,
        "model_name": outputs.get("model_name", bundle.model_name),
        "checkpoint_path": str(bundle.checkpoint_path),
        "teacher_used_during_inference": bool(outputs.get("teacher_used_during_inference", False)),
        "teacher_used_during_training": bool(outputs.get("teacher_used_during_training", False)),
        "uses_project_h_f_input": True,
        "native_receiver_path": bool(native_receiver_path),
        "input_type": csi_input_meta.get("input_type"),
        "csi_interface_used": bool(csi_input_meta.get("csi_interface_used")),
        "project_h_f_assisted": csi_input_meta.get("project_h_f_assisted"),
        "extracted_h_f_used": csi_input_meta.get("extracted_h_f_used"),
        "full_native_only": csi_input_meta.get("full_native_only"),
        "source": csi_input_meta.get("source"),
        "source_component": csi_input_meta.get("source_component"),
        "csi_input_summary": {
            "input_type": csi_input_meta.get("input_type"),
            "tensor_signature": csi_input_meta.get("tensor_signature"),
            "validation": csi_input_meta.get("validation"),
            "source": csi_input_meta.get("source"),
            "source_component": csi_input_meta.get("source_component"),
        },
        "init_method": outputs.get("init_method"),
        "num_layers": outputs.get("num_layers"),
        "power_norm": float(power.mean().item()),
        "power_violation": float(torch.mean(torch.abs(power - 1.0)).item()),
        "all_finite": bool(torch.isfinite(precoder.real).all() and torch.isfinite(precoder.imag).all()),
        "inference_inputs": outputs.get("inference_inputs"),
    }
    if return_precoder_output:
        precoder_output = build_precoder_output(
            f_f=precoder,
            source=bundle.method_name,
            method=bundle.method_name,
            input_csi=h_f if isinstance(h_f, (ExtractedCSI, dict)) else {"h_f": h_f_tensor, **csi_input_meta},
            project_side_precoder=True,
            sionna_native_precoder=False,
            teacher_used_during_inference=bool(meta["teacher_used_during_inference"]),
            power_normalized=True,
            checkpoint_path=str(bundle.checkpoint_path),
            skipped_missing_checkpoint=False,
            fallback_reason="",
            full_native_only=False,
            metadata={"inference_inputs": outputs.get("inference_inputs")},
        )
        meta["precoder_interface_used"] = True
        meta["precoder_summary"] = precoder_output.summary_dict()
        return precoder_output, meta, runtime_ms
    meta["precoder_interface_used"] = False
    return precoder, meta, runtime_ms


def generate_shared_sionna_channel_bundle(
    *,
    batch_size: int,
    num_subcarriers: int,
    num_users: int,
    num_bs_ant: int,
    noise_var: float,
    device: torch.device,
    selected_ofdm_symbol: str | int = "first_data",
    effective_subcarriers: str | list[int] = "all_effective",
    normalize_channel: bool = False,
    seed: int = 0,
) -> SharedSionnaChannelBundle:
    """Generate one shared Sionna channel realization and its extracted ``H_f``."""
    resource_grid, stream_management, rg_meta = build_pilot_aware_multiuser_resource_grid(
        num_users=num_users,
        num_effective_subcarriers=num_subcarriers,
        num_ofdm_symbols=2,
        device=device,
    )
    if resource_grid is None or stream_management is None:
        raise RuntimeError(f"Failed to build pilot-aware native receiver context: {rg_meta.get('fallback_reason')}")
    shared_batch = create_shared_sionna_ofdm_batch_from_generator(
        batch_size=batch_size,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        snr_db=float(-10.0 * torch.log10(torch.tensor(noise_var)).item()),
        device=device,
        resource_grid=resource_grid,
        stream_management=stream_management,
        selected_ofdm_symbol=selected_ofdm_symbol,
        effective_subcarriers=effective_subcarriers,
        normalize_channel=normalize_channel,
        seed=seed,
    )
    h_f = shared_batch.extracted_h_f
    h_full = shared_batch.sionna_channel_tensor
    csi = shared_batch.csi
    h_meta = {
        "fallback_used": False,
        "fallback_reason": "",
        "used_native_channel_extraction": True,
        "selected_data_symbol_index": int(shared_batch.selected_ofdm_symbol),
        "full_channel_shape": [int(x) for x in h_full.shape],
        "effective_subcarrier_ind": [int(x) for x in resource_grid.effective_subcarrier_ind],
        "extraction_meta": {
            "selected_data_symbol_indices": list(shared_batch.csi.metadata.get("selected_data_symbol_indices", [shared_batch.selected_ofdm_symbol])),
            "selected_effective_subcarrier_indices": list(shared_batch.effective_subcarrier_indices),
            "selected_data_symbol_index": int(shared_batch.selected_ofdm_symbol),
            "selected_effective_subcarrier_count": int(len(shared_batch.effective_subcarrier_indices)),
            "original_sionna_h_shape": [int(x) for x in h_full.shape],
            "original_axes": shared_batch.csi.metadata.get("original_axes"),
            "conversion_meta": shared_batch.csi.metadata.get("conversion_meta", {}),
        },
        "csi_summary": shared_batch.csi.summary_dict(),
        "shared_batch_summary": shared_batch.summary_dict(),
    }
    if h_f is None or h_full is None:
        raise RuntimeError(f"Failed to extract native H_f from Sionna channel: {h_meta.get('fallback_reason')}")
    return SharedSionnaChannelBundle(
        resource_grid=resource_grid,
        stream_management=stream_management,
        h_full=h_full,
        h_f=h_f,
        csi=csi,
        bits=shared_batch.bits,
        stream_symbols=shared_batch.symbols,
        shared_batch_meta=shared_batch.summary_dict(),
        noise_var=float(noise_var),
        bundle_meta={
            "resource_grid_meta": rg_meta,
            "channel_meta": h_meta,
            "csi_interface_used": csi is not None,
            "csi_summary": csi.summary_dict() if csi is not None else None,
            "shared_batch_summary": shared_batch.summary_dict(),
            "shared_rx_noise_grid": shared_batch.rx_noise_grid,
            "native_receiver_path": True,
            "synthetic_channel_level_only": True,
            "project_h_f_assisted": bool(not h_meta.get("used_native_channel_extraction", False)),
        },
    )


def build_native_receiver_context(
    *,
    batch_size: int,
    num_subcarriers: int,
    num_users: int,
    num_bs_ant: int,
    snr_db: float,
    device: torch.device,
    channel_bundle: SharedSionnaChannelBundle | None = None,
) -> NativeReceiverContext:
    bits = torch.randint(0, 2, (batch_size, num_subcarriers, num_users, 2), device=device)
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    stream_symbols = ((real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=device))).to(torch.complex64)
    noise_var = float(10.0 ** (-snr_db / 10.0))

    if channel_bundle is None:
        channel_bundle = generate_shared_sionna_channel_bundle(
            batch_size=batch_size,
            num_subcarriers=num_subcarriers,
            num_users=num_users,
            num_bs_ant=num_bs_ant,
            noise_var=noise_var,
            device=device,
            seed=0,
        )
    if channel_bundle.bits is not None and channel_bundle.stream_symbols is not None:
        bits = channel_bundle.bits.to(torch.int64)
        stream_symbols = channel_bundle.stream_symbols.to(torch.complex64)
    else:
        bits = torch.randint(0, 2, (batch_size, num_subcarriers, num_users, 2), device=device)
        real = 1.0 - 2.0 * bits[..., 0].float()
        imag = 1.0 - 2.0 * bits[..., 1].float()
        stream_symbols = ((real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=device))).to(torch.complex64)
    return NativeReceiverContext(
        bits=bits.to(torch.int64),
        stream_symbols=stream_symbols,
        resource_grid=channel_bundle.resource_grid,
        stream_management=channel_bundle.stream_management,
        h_f=channel_bundle.h_f,
        csi=channel_bundle.csi,
        h_full=channel_bundle.h_full,
        noise_var=noise_var,
        snr_db=float(snr_db),
        device=device,
        context_meta=dict(channel_bundle.bundle_meta),
    )


def summarize_native_receiver_context(context: NativeReceiverContext) -> dict[str, Any]:
    """Return a serializable summary for same-batch equivalence checks."""

    csi = context.csi
    selected_ofdm_symbol = csi.selected_ofdm_symbol if csi is not None else None
    effective_subcarrier_indices = csi.effective_subcarrier_indices if csi is not None else None
    return {
        "noise_var": float(context.noise_var),
        "snr_db": float(context.snr_db),
        "bits_shape": [int(x) for x in context.bits.shape],
        "stream_symbols_shape": [int(x) for x in context.stream_symbols.shape],
        "h_f_shape": [int(x) for x in context.h_f.shape],
        "h_full_shape": [int(x) for x in context.h_full.shape],
        "receiver_config": summarize_receiver_config(
            context.resource_grid,
            context.stream_management,
            selected_ofdm_symbol=selected_ofdm_symbol,
            effective_subcarrier_indices=effective_subcarrier_indices,
        ),
        "shared_rx_noise_grid_present": context.context_meta.get("shared_rx_noise_grid") is not None,
    }


def run_native_receiver_with_precoder(
    *,
    method: str,
    method_type: str,
    precoder_f: PrecoderOutput | torch.Tensor | dict[str, Any],
    context: NativeReceiverContext,
    runtime_ms: float,
    checkpoint_path: str | None,
    teacher_used_during_inference: bool,
    trace_shapes: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    ApplyOFDMChannel, _, _ = load_component("ApplyOFDMChannel")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    Demapper, _, _ = load_component("Demapper")

    precoder_tensor, precoder_meta = as_project_f_f(precoder_f, validate=False)
    teacher_flag = bool(teacher_used_during_inference or precoder_meta.get("teacher_used_during_inference", False))
    row: dict[str, Any] = {
        "method": method,
        "method_type": method_type,
        "checkpoint_path": checkpoint_path,
        "precoder_interface_used": bool(precoder_meta.get("precoder_interface_used")),
        "precoder_input_type": precoder_meta.get("input_type"),
        "precoder_source": precoder_meta.get("source"),
        "project_side_precoder": precoder_meta.get("project_side_precoder"),
        "sionna_native_precoder": precoder_meta.get("sionna_native_precoder"),
        "full_native_only": precoder_meta.get("full_native_only", False),
        "power_normalized": precoder_meta.get("power_normalized"),
        "relationship_status": precoder_meta.get("relationship_status"),
        "strict_equivalence_claim_allowed": precoder_meta.get("strict_equivalence_claim_allowed"),
        "native_receiver_success": False,
        "used_sionna_resource_grid": True,
        "used_sionna_channel": False,
        "used_sionna_estimator": False,
        "used_sionna_equalizer": False,
        "used_sionna_demapper": False,
        "teacher_used_during_inference": teacher_flag,
        "fallback_used": True,
        "fallback_stage": "",
        "fallback_reason": "",
        "ber_if_available": None,
        "symbol_mse": float("nan"),
        "effective_sinr_db": float("nan"),
        "approximate_sum_rate": float("nan"),
        "power_norm": float(torch.mean((torch.abs(precoder_tensor) ** 2).sum(dim=(-2, -1))).item()),
        "runtime_ms": float(runtime_ms),
    }
    trace: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "native_receiver_attempted": True,
        "native_receiver_success": False,
        "native_failure_stage": "",
        "native_failure_reason": "",
    }

    x_rg, rg_bridge_meta = map_project_streams_to_sionna_rg(context.stream_symbols, context.resource_grid)
    if x_rg is None:
        row["fallback_stage"] = "resource_grid_mapper"
        row["fallback_reason"] = str(rg_bridge_meta.get("fallback_reason", "resource_grid_mapping_failed"))
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        return row, trace, meta

    tx_grid, tx_meta = apply_project_precoder_to_sionna_grid(x_rg, precoder_f, context.resource_grid)
    if tx_grid is None:
        row["fallback_stage"] = "precoder_bridge"
        row["fallback_reason"] = str(tx_meta.get("fallback_reason", "precoder_bridge_failed"))
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        return row, trace, meta
    if context.context_meta.get("force_precoder_matrix_receiver_failure", False):
        row["fallback_stage"] = "forced_receiver_failure"
        row["fallback_reason"] = "forced_receiver_failure_for_contract_matrix"
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        return row, trace, meta

    if trace_shapes:
        trace.extend(
            [
                describe_tensor("stream_symbols", context.stream_symbols, ["batch", "effective_subcarrier", "user"]),
                describe_tensor("x_rg", x_rg, ["batch", "num_tx", "num_streams", "ofdm_symbol", "fft_bin"]),
                describe_tensor("H_f", context.h_f, ["batch", "effective_subcarrier", "user", "bs_ant"]),
                describe_tensor("F_f", precoder_tensor, ["batch", "effective_subcarrier", "bs_ant", "user"]),
                describe_tensor("tx_grid", tx_grid, ["batch", "num_tx", "num_tx_ant", "ofdm_symbol", "fft_bin"]),
            ]
        )

    try:
        rx_grid = ApplyOFDMChannel(device=resolve_sionna_device(context.device))(tx_grid, context.h_full)
        noise_grid = context.context_meta.get("shared_rx_noise_grid")
        if noise_grid is not None:
            rx_grid = rx_grid + noise_grid.to(rx_grid.device)
        else:
            noise = torch.full(
                (context.stream_symbols.size(0), context.stream_symbols.size(2), 1),
                context.noise_var,
                dtype=torch.float32,
                device=context.device,
            )
            rx_grid = rx_grid + (
                (torch.randn_like(rx_grid.real) + 1j * torch.randn_like(rx_grid.real))
                * torch.sqrt(torch.tensor(context.noise_var / 2.0, dtype=torch.float32, device=context.device))
            ).to(torch.complex64)
    except Exception as exc:
        row["fallback_stage"] = "channel_apply"
        row["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        return row, trace, meta
    if trace_shapes:
        trace.append(describe_tensor("rx_grid", rx_grid, ["batch", "num_rx", "num_rx_ant", "ofdm_symbol", "fft_bin"]))

    try:
        estimator = LSChannelEstimator(context.resource_grid, device=resolve_sionna_device(context.device))
        estimator_noise = torch.full(
            (context.stream_symbols.size(0), context.stream_symbols.size(2), 1),
            context.noise_var,
            dtype=torch.float32,
            device=context.device,
        )
        h_hat, err_var = estimator(rx_grid, estimator_noise)
    except Exception as exc:
        row["fallback_stage"] = "estimator"
        row["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        return row, trace, meta

    validation = validate_sionna_receiver_shapes(rx_grid, h_hat, err_var, context.stream_management, context.resource_grid)
    if trace_shapes:
        trace.extend(
            [
                describe_tensor("h_hat", h_hat, ["batch", "num_rx", "num_rx_ant", "num_tx", "num_streams_per_tx", "ofdm_symbol", "effective_subcarrier"]),
                describe_tensor("err_var", err_var, ["batch", "num_rx", "num_rx_ant", "num_tx", "num_streams_per_tx", "ofdm_symbol", "effective_subcarrier"]),
            ]
        )
    if not validation["valid"]:
        row["fallback_stage"] = "shape_validation"
        row["fallback_reason"] = str(validation["reason"])
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        meta["shape_validation"] = validation
        return row, trace, meta

    try:
        equalizer = LMMSEEqualizer(context.resource_grid, context.stream_management, device=resolve_sionna_device(context.device))
        equalizer_noise = torch.full(
            (context.stream_symbols.size(0), context.stream_symbols.size(2), 1),
            context.noise_var,
            dtype=torch.float32,
            device=context.device,
        )
        x_hat, no_eff = equalizer(rx_grid, h_hat, err_var, equalizer_noise)
    except Exception as exc:
        row["fallback_stage"] = "equalizer"
        row["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        return row, trace, meta
    if trace_shapes:
        trace.extend(
            [
                describe_tensor("x_hat", x_hat, ["batch", "num_tx", "num_streams", "data_symbols"]),
                describe_tensor("no_eff", no_eff, ["batch", "num_tx", "num_streams", "data_symbols"]),
            ]
        )

    try:
        demapper = Demapper("app", "qam", 2, hard_out=True, device=resolve_sionna_device(context.device))
        hard_bits = demapper(x_hat, no_eff)
    except Exception as exc:
        row["fallback_stage"] = "demapper"
        row["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        return row, trace, meta
    if trace_shapes:
        trace.append(describe_tensor("hard_bits", hard_bits, ["batch", "num_tx", "num_streams", "coded_bits"]))

    project_rx, bridge_meta = sionna_rx_to_project_symbols(x_hat)
    if bridge_meta["fallback_used"]:
        row["fallback_stage"] = "project_rx_bridge"
        row["fallback_reason"] = str(bridge_meta["fallback_reason"])
        meta["native_failure_stage"] = row["fallback_stage"]
        meta["native_failure_reason"] = row["fallback_reason"]
        return row, trace, meta

    rx_metrics = compute_project_metrics_from_sionna_rx(project_rx, context.stream_symbols)
    bit_ref = context.bits.permute(0, 2, 1, 3).reshape_as(hard_bits)
    row.update(
        {
            "native_receiver_success": True,
            "used_sionna_channel": True,
            "used_sionna_estimator": True,
            "used_sionna_equalizer": True,
            "used_sionna_demapper": True,
            "fallback_used": False,
            "fallback_stage": "",
            "fallback_reason": "",
            "ber_if_available": float((hard_bits.to(torch.int64) != bit_ref.to(torch.int64)).float().mean().item()),
            "symbol_mse": float(rx_metrics["symbol_mse"]),
            "effective_sinr_db": float(rx_metrics["effective_sinr_db"]),
            "approximate_sum_rate": float(
                context.stream_symbols.size(2)
                * torch.log2(torch.tensor(1.0 + (10.0 ** (rx_metrics["effective_sinr_db"] / 10.0)), device=context.device)).item()
            ),
        }
    )
    meta["native_receiver_success"] = True
    meta["shape_validation"] = validation
    return row, trace, meta
