#!/usr/bin/env python
"""Trace tensor shapes for minimal and beamformed Sionna receiver paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import (
    apply_project_precoder_to_sionna_grid,
    build_pilot_aware_multiuser_resource_grid,
    compute_project_precoder_per_subcarrier,
    describe_tensor,
    extract_effective_channel_from_sionna,
    map_project_streams_to_sionna_rg,
    validate_sionna_receiver_shapes,
)
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _qpsk_symbols(batch_size: int, num_subcarriers: int, num_users: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(0, 2, (batch_size, num_subcarriers, num_users, 2), device=device)
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    symbols = (real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=device))
    return bits.to(torch.int64), symbols.to(torch.complex64)


def _trace_minimal_success(device: torch.device) -> dict[str, Any]:
    sionna_device = resolve_sionna_device(device)
    ResourceGrid, _, _ = load_component("ResourceGrid")
    ResourceGridMapper, _, _ = load_component("ResourceGridMapper")
    OFDMChannel, _, _ = load_component("OFDMChannel")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    Demapper, _, _ = load_component("Demapper")
    BinarySource, _, _ = load_component("BinarySource")
    Mapper, _, _ = load_component("Mapper")
    StreamManagement, _, _ = load_component("StreamManagement")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")

    rg = ResourceGrid(
        num_ofdm_symbols=4,
        fft_size=16,
        subcarrier_spacing=15_000.0,
        num_tx=1,
        num_streams_per_tx=1,
        num_guard_carriers=(1, 1),
        dc_null=True,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[0],
        device=sionna_device,
    )
    sm = StreamManagement(np.array([[1]]), num_streams_per_tx=1)
    source = BinarySource(device=sionna_device)
    mapper = Mapper("qam", 2, device=sionna_device)
    rg_mapper = ResourceGridMapper(rg, device=sionna_device)
    channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1, device=sionna_device)
    channel = OFDMChannel(channel_model, rg, return_channel=True, device=sionna_device)
    estimator = LSChannelEstimator(rg, device=sionna_device)
    equalizer = LMMSEEqualizer(rg, sm, device=sionna_device)
    demapper = Demapper("app", "qam", 2, hard_out=True, device=sionna_device)

    traces: list[dict[str, Any]] = []
    noise = torch.full((32, 1, 1), 0.1, dtype=torch.float32, device=device)
    bits = source([32, 1, 1, int(rg.num_data_symbols * 2)])
    tx_symbols = mapper(bits)
    tx_grid = rg_mapper(tx_symbols)
    rx_grid, h_freq = channel(tx_grid, no=noise)
    h_hat, err_var = estimator(rx_grid, noise)
    x_hat, no_eff = equalizer(rx_grid, h_hat, err_var, noise)
    hard_bits = demapper(x_hat, torch.full((32, 1, 1, 1), 0.1, dtype=torch.float32, device=device))

    traces.extend(
        [
            describe_tensor("bits", bits, ["batch", "num_tx", "num_streams", "coded_bits"]),
            describe_tensor("tx_symbols", tx_symbols, ["batch", "num_tx", "num_streams", "data_symbols"]),
            describe_tensor("tx_grid", tx_grid, ["batch", "num_tx", "num_streams", "ofdm_symbol", "fft_bin"]),
            describe_tensor("rx_grid", rx_grid, ["batch", "num_rx", "num_rx_ant", "ofdm_symbol", "fft_bin"]),
            describe_tensor(
                "h_freq",
                h_freq,
                ["batch", "num_rx", "num_rx_ant", "num_tx", "num_tx_ant", "ofdm_symbol", "fft_bin"],
            ),
            describe_tensor(
                "h_hat",
                h_hat,
                ["batch", "num_rx", "num_rx_ant", "num_tx", "num_streams_per_tx", "ofdm_symbol", "effective_subcarrier"],
            ),
            describe_tensor(
                "err_var",
                err_var,
                ["batch", "num_rx", "num_rx_ant", "num_tx", "num_streams_per_tx", "ofdm_symbol", "effective_subcarrier"],
            ),
            describe_tensor("x_hat", x_hat, ["batch", "num_tx", "num_streams", "data_symbols"]),
            describe_tensor("no_eff", no_eff, ["batch", "num_tx", "num_streams", "data_symbols"]),
            describe_tensor("hard_bits", hard_bits, ["batch", "num_tx", "num_streams", "coded_bits"]),
        ]
    )
    return {
        "path_name": "minimal_receiver_success_path",
        "success": True,
        "resource_grid": {
            "num_ofdm_symbols": int(rg.num_ofdm_symbols),
            "fft_size": int(rg.fft_size),
            "num_data_symbols": int(rg.num_data_symbols),
            "num_pilot_symbols": int(rg.num_pilot_symbols),
            "effective_subcarrier_ind": [int(x) for x in rg.effective_subcarrier_ind],
        },
        "stream_management": {
            "num_rx": int(sm.num_rx),
            "num_tx": int(sm.num_tx),
            "num_streams_per_tx": int(sm.num_streams_per_tx),
            "detection_desired_ind": sm.detection_desired_ind.tolist(),
        },
        "tensor_trace": traces,
    }


def _trace_beamformed_paths(device: torch.device) -> dict[str, Any]:
    sionna_device = resolve_sionna_device(device)
    ApplyOFDMChannel, _, _ = load_component("ApplyOFDMChannel")
    OFDMChannel, _, _ = load_component("OFDMChannel")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    Demapper, _, _ = load_component("Demapper")

    batch_size = 16
    num_users = 4
    num_bs_ant = 16
    num_effective_subcarriers = 16
    noise_var = 10.0 ** (-10.0 / 10.0)

    failing_rg, failing_sm, failing_meta = build_pilot_aware_multiuser_resource_grid(
        num_users=num_users,
        num_effective_subcarriers=num_effective_subcarriers,
        num_ofdm_symbols=1,
        device=device,
    )
    success_rg, success_sm, success_meta = build_pilot_aware_multiuser_resource_grid(
        num_users=num_users,
        num_effective_subcarriers=num_effective_subcarriers,
        num_ofdm_symbols=2,
        device=device,
    )
    bits, stream_symbols = _qpsk_symbols(batch_size, num_effective_subcarriers, num_users, device)
    failing_trace: list[dict[str, Any]] = []
    failing_summary: dict[str, Any] = {
        "path_name": "beamformed_receiver_previous_failure_path",
        "success": False,
        "resource_grid_meta": failing_meta,
        "stream_management": None,
        "failure_stage": "",
        "exception_type": "",
        "exception_message": "",
        "zero_dimension_origin": "",
        "tensor_trace": failing_trace,
    }
    if failing_rg is None or failing_sm is None:
        failing_summary["failure_stage"] = "resource_grid_construction"
        failing_summary["exception_type"] = "LogicalShapeFailure"
        failing_summary["exception_message"] = str(failing_meta.get("fallback_reason", "invalid_pilot_only_grid"))
        failing_summary["zero_dimension_origin"] = (
            "num_data_symbols=0 because num_ofdm_symbols=1 and pilot_ofdm_symbol_indices=[0] leave no data OFDM symbol"
        )
    if failing_rg is not None and failing_sm is not None and OFDMChannel is not None and LSChannelEstimator is not None and LMMSEEqualizer is not None:
        h_f_fail, _, h_meta_fail = extract_effective_channel_from_sionna(
            failing_rg,
            batch_size=batch_size,
            num_users=num_users,
            num_bs_ant=num_bs_ant,
            device=device,
            noise_var=noise_var,
        )
        if h_f_fail is not None:
            failing_trace.append(describe_tensor("H_f_failing", h_f_fail, ["batch", "effective_subcarrier", "user", "bs_ant"]))
            precoder_f_fail = compute_project_precoder_per_subcarrier("rzf", h_f_fail, noise_var)
            failing_trace.append(describe_tensor("F_f_failing", precoder_f_fail, ["batch", "effective_subcarrier", "bs_ant", "user"]))
            x_rg_fail, rg_meta_fail = map_project_streams_to_sionna_rg(stream_symbols, failing_rg)
            if x_rg_fail is not None:
                failing_trace.append(describe_tensor("x_rg_failing", x_rg_fail, ["batch", "num_tx", "num_streams", "ofdm_symbol", "fft_bin"]))
                tx_grid_fail, tx_meta_fail = apply_project_precoder_to_sionna_grid(x_rg_fail, precoder_f_fail, failing_rg)
                if tx_grid_fail is not None:
                    failing_trace.append(describe_tensor("tx_grid_failing", tx_grid_fail, ["batch", "num_tx", "num_tx_ant", "ofdm_symbol", "fft_bin"]))
                    channel_model = load_component("RayleighBlockFading")[0](num_rx=num_users, num_rx_ant=1, num_tx=1, num_tx_ant=num_bs_ant, device=sionna_device)
                    channel = OFDMChannel(channel_model, failing_rg, return_channel=True, device=sionna_device)
                    noise = torch.full((batch_size, num_users, 1), noise_var, dtype=torch.float32, device=device)
                    rx_grid_fail, _ = channel(tx_grid_fail, no=noise)
                    estimator = LSChannelEstimator(failing_rg, device=sionna_device)
                    h_hat_fail, err_var_fail = estimator(rx_grid_fail, noise)
                    failing_trace.extend(
                        [
                            describe_tensor("rx_grid_failing", rx_grid_fail, ["batch", "num_rx", "num_rx_ant", "ofdm_symbol", "fft_bin"]),
                            describe_tensor(
                                "h_hat_failing",
                                h_hat_fail,
                                ["batch", "num_rx", "num_rx_ant", "num_tx", "num_streams_per_tx", "ofdm_symbol", "effective_subcarrier"],
                            ),
                            describe_tensor(
                                "err_var_failing",
                                err_var_fail,
                                ["batch", "num_rx", "num_rx_ant", "num_tx", "num_streams_per_tx", "ofdm_symbol", "effective_subcarrier"],
                            ),
                        ]
                    )
                    validation_fail = validate_sionna_receiver_shapes(rx_grid_fail, h_hat_fail, err_var_fail, failing_sm, failing_rg)
                    failing_summary["stream_management"] = {
                        "num_rx": int(failing_sm.num_rx),
                        "num_tx": int(failing_sm.num_tx),
                        "num_streams_per_tx": int(failing_sm.num_streams_per_tx),
                        "detection_desired_ind": failing_sm.detection_desired_ind.tolist(),
                    }
                    failing_summary["shape_validation"] = validation_fail
                    try:
                        equalizer = LMMSEEqualizer(failing_rg, failing_sm, device=sionna_device)
                        x_hat_fail, no_eff_fail = equalizer(rx_grid_fail, h_hat_fail, err_var_fail, noise)
                        failing_trace.extend(
                            [
                                describe_tensor("x_hat_failing", x_hat_fail, ["batch", "num_tx", "num_streams", "data_symbols"]),
                                describe_tensor("no_eff_failing", no_eff_fail, ["batch", "num_tx", "num_streams", "data_symbols"]),
                            ]
                        )
                        if x_hat_fail.size(-1) == 0:
                            failing_summary["failure_stage"] = "equalizer_output"
                            failing_summary["exception_type"] = "LogicalShapeFailure"
                            failing_summary["exception_message"] = "equalizer produced zero data symbols"
                            failing_summary["zero_dimension_origin"] = "num_data_symbols=0 because num_ofdm_symbols=1 and pilot_ofdm_symbol_indices=[0] consumed the only OFDM symbol"
                    except Exception as exc:
                        failing_summary["failure_stage"] = "equalizer"
                        failing_summary["exception_type"] = type(exc).__name__
                        failing_summary["exception_message"] = str(exc)
                        failing_summary["zero_dimension_origin"] = "receiver equalizer attempted to reshape with num_data_symbols=0 after a pilot-only grid"
        failing_summary["channel_meta"] = h_meta_fail

    success_trace: list[dict[str, Any]] = []
    success_summary: dict[str, Any] = {
        "path_name": "beamformed_receiver_native_retry_path",
        "success": False,
        "resource_grid_meta": success_meta,
        "stream_management": None,
        "failure_stage": "",
        "exception_type": "",
        "exception_message": "",
        "tensor_trace": success_trace,
    }
    if success_rg is not None and success_sm is not None and ApplyOFDMChannel is not None and OFDMChannel is not None:
        h_f, h_full, h_meta = extract_effective_channel_from_sionna(
            success_rg,
            batch_size=batch_size,
            num_users=num_users,
            num_bs_ant=num_bs_ant,
            device=device,
            noise_var=noise_var,
        )
        success_summary["stream_management"] = {
            "num_rx": int(success_sm.num_rx),
            "num_tx": int(success_sm.num_tx),
            "num_streams_per_tx": int(success_sm.num_streams_per_tx),
            "detection_desired_ind": success_sm.detection_desired_ind.tolist(),
        }
        success_summary["channel_meta"] = h_meta
        if h_f is not None and h_full is not None:
            success_trace.append(describe_tensor("H_f", h_f, ["batch", "effective_subcarrier", "user", "bs_ant"]))
            precoder_f = compute_project_precoder_per_subcarrier("rzf", h_f, noise_var)
            success_trace.append(describe_tensor("F_f", precoder_f, ["batch", "effective_subcarrier", "bs_ant", "user"]))
            x_rg, rg_meta = map_project_streams_to_sionna_rg(stream_symbols, success_rg)
            success_summary["resource_grid_bridge_meta"] = rg_meta
            if x_rg is not None:
                success_trace.append(describe_tensor("x_rg", x_rg, ["batch", "num_tx", "num_streams", "ofdm_symbol", "fft_bin"]))
                tx_grid, tx_meta = apply_project_precoder_to_sionna_grid(x_rg, precoder_f, success_rg)
                success_summary["tx_bridge_meta"] = tx_meta
                if tx_grid is not None:
                    success_trace.append(describe_tensor("tx_grid", tx_grid, ["batch", "num_tx", "num_tx_ant", "ofdm_symbol", "fft_bin"]))
                    apply_channel = ApplyOFDMChannel(device=sionna_device)
                    noise = torch.full((batch_size, num_users, 1), noise_var, dtype=torch.float32, device=device)
                    rx_grid = apply_channel(tx_grid, h_full, no=noise)
                    success_trace.append(describe_tensor("rx_grid", rx_grid, ["batch", "num_rx", "num_rx_ant", "ofdm_symbol", "fft_bin"]))
                    estimator = load_component("LSChannelEstimator")[0](success_rg, device=sionna_device)
                    h_hat, err_var = estimator(rx_grid, noise)
                    success_trace.extend(
                        [
                            describe_tensor(
                                "h_hat",
                                h_hat,
                                ["batch", "num_rx", "num_rx_ant", "num_tx", "num_streams_per_tx", "ofdm_symbol", "effective_subcarrier"],
                            ),
                            describe_tensor(
                                "err_var",
                                err_var,
                                ["batch", "num_rx", "num_rx_ant", "num_tx", "num_streams_per_tx", "ofdm_symbol", "effective_subcarrier"],
                            ),
                        ]
                    )
                    validation = validate_sionna_receiver_shapes(rx_grid, h_hat, err_var, success_sm, success_rg)
                    success_summary["shape_validation"] = validation
                    equalizer = load_component("LMMSEEqualizer")[0](success_rg, success_sm, device=sionna_device)
                    x_hat, no_eff = equalizer(rx_grid, h_hat, err_var, noise)
                    success_trace.extend(
                        [
                            describe_tensor("x_hat", x_hat, ["batch", "num_tx", "num_streams", "data_symbols"]),
                            describe_tensor("no_eff", no_eff, ["batch", "num_tx", "num_streams", "data_symbols"]),
                        ]
                    )
                    hard_bits = load_component("Demapper")[0]("app", "qam", 2, hard_out=True, device=sionna_device)(x_hat, no_eff)
                    success_trace.append(describe_tensor("hard_bits", hard_bits, ["batch", "num_tx", "num_streams", "coded_bits"]))
                    success_summary.update(
                        {
                            "success": True,
                            "ber": float((hard_bits.to(torch.int64) != bits.permute(0, 2, 1, 3).reshape_as(hard_bits)).float().mean().item()),
                            "symbol_mse": float(torch.mean(torch.abs(x_hat - x_rg[:, :, :, 1, success_rg.effective_subcarrier_ind]) ** 2).item()),
                        }
                    )

    return {"failing_path": failing_summary, "native_retry_path": success_summary}


def _to_markdown(payload: dict[str, Any]) -> list[str]:
    minimal = payload["minimal_success_path"]
    failing = payload["beamformed_paths"]["failing_path"]
    retry = payload["beamformed_paths"]["native_retry_path"]
    lines = [
        "# Beamformed Receiver Shape Trace",
        "",
        "## Summary",
        "",
        f"1. minimal success path y/h_hat/err_var/x_hat/no_eff shapes are: "
        f"`{minimal['tensor_trace'][3]['shape']}` / `{minimal['tensor_trace'][5]['shape']}` / "
        f"`{minimal['tensor_trace'][6]['shape']}` / `{minimal['tensor_trace'][7]['shape']}` / `{minimal['tensor_trace'][8]['shape']}`",
        f"2. previous beamformed path failed at stage `{failing.get('failure_stage', '')}`",
        f"3. shape `[16,1,1,0]` came from `{failing.get('zero_dimension_origin', '')}`",
        "4. the 0 dimension is a `num_data_symbols` problem caused by a pilot-only grid, not a random CUDA issue",
        "5. recommended fix is `num_tx=1, num_streams_per_tx=K`, `rx_tx_association=ones(K,1)`, and at least one non-pilot OFDM symbol",
        "",
        "## Native Retry",
        f"- success: `{retry['success']}`",
        f"- BER: `{retry.get('ber')}`",
        f"- symbol_mse: `{retry.get('symbol_mse')}`",
    ]
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload: dict[str, Any] = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "device": str(device),
        "minimal_success_path": {},
        "beamformed_paths": {},
        "summary": {},
    }
    if env["sionna_import_ok"]:
        payload["minimal_success_path"] = _trace_minimal_success(device)
        payload["beamformed_paths"] = _trace_beamformed_paths(device)
        payload["summary"] = {
            "minimal_shapes": {
                "y": payload["minimal_success_path"]["tensor_trace"][3]["shape"],
                "h_hat": payload["minimal_success_path"]["tensor_trace"][5]["shape"],
                "err_var": payload["minimal_success_path"]["tensor_trace"][6]["shape"],
                "x_hat": payload["minimal_success_path"]["tensor_trace"][7]["shape"],
                "no_eff": payload["minimal_success_path"]["tensor_trace"][8]["shape"],
            },
            "beamformed_failure_stage": payload["beamformed_paths"]["failing_path"]["failure_stage"],
            "beamformed_failure_reason": payload["beamformed_paths"]["failing_path"]["exception_message"],
            "beamformed_zero_dimension_origin": payload["beamformed_paths"]["failing_path"]["zero_dimension_origin"],
            "native_retry_success": payload["beamformed_paths"]["native_retry_path"]["success"],
            "native_retry_ber": payload["beamformed_paths"]["native_retry_path"].get("ber"),
            "native_retry_symbol_mse": payload["beamformed_paths"]["native_retry_path"].get("symbol_mse"),
            "recommended_fix": "Use a pilot-aware downlink MU ResourceGrid with num_tx=1, num_streams_per_tx=K, rx_tx_association=ones(K,1), and num_ofdm_symbols > len(pilot_ofdm_symbol_indices).",
        }
    write_json(out_path, payload)
    write_markdown(md_path, _to_markdown(payload))
    print(f"Saved beamformed receiver shape trace to {out_path}")


if __name__ == "__main__":
    main()
