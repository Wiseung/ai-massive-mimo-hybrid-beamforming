#!/usr/bin/env python
"""Insert project beamforming into a Sionna-native OFDM chain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import (
    apply_project_precoder_to_sionna_grid,
    build_frequency_domain_channel,
    build_pilot_aware_multiuser_resource_grid,
    compute_project_metrics_from_sionna_rx,
    compute_project_precoder_per_subcarrier,
    describe_tensor,
    evaluate_ofdm_beamforming_outputs,
    extract_effective_channel_from_sionna,
    map_project_streams_to_sionna_rg,
    sionna_rx_to_project_symbols,
    time_function,
    validate_sionna_receiver_shapes,
)
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--enable-receiver-chain", action="store_true")
    parser.add_argument("--receiver-mode", choices=["proxy", "native", "auto"], default="proxy")
    parser.add_argument("--trace-shapes", action="store_true")
    return parser.parse_args()


def _qpsk_symbols(batch_size: int, num_subcarriers: int, num_users: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(0, 2, (batch_size, num_subcarriers, num_users, 2), device=device)
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    symbols = (real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=device))
    return bits.to(torch.int64), symbols.to(torch.complex64)


def _identity_precoder(batch_size: int, num_subcarriers: int, num_users: int, num_bs_ant: int, device: torch.device) -> torch.Tensor:
    precoder = torch.zeros(batch_size, num_subcarriers, num_bs_ant, num_users, dtype=torch.complex64, device=device)
    eye_k = torch.eye(num_users, dtype=torch.complex64, device=device)
    precoder[:, :, :num_users, :] = eye_k.unsqueeze(0).unsqueeze(0)
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return precoder / torch.sqrt(power)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_md(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Sionna Native OFDM Beamforming Chain",
        "",
        f"- Demo status: `{summary['demo_status']}`",
        f"- receiver_mode: `{summary['receiver_mode']}`",
        f"- native_receiver_attempted: `{summary['native_receiver_attempted']}`",
        f"- native_receiver_success: `{summary['native_receiver_success']}`",
        f"- Used Sionna ResourceGrid: `{summary['used_sionna_resource_grid']}`",
        f"- Used Sionna channel: `{summary['used_sionna_channel']}`",
        f"- Used Sionna estimator: `{summary['used_sionna_estimator']}`",
        f"- Used Sionna equalizer: `{summary['used_sionna_equalizer']}`",
        f"- Used Sionna demapper: `{summary['used_sionna_demapper']}`",
        f"- Fallback used: `{summary['fallback_used']}`",
        f"- shape_trace_path: `{summary.get('shape_trace_path')}`",
        "",
        "| Method | Native OK | BER | Symbol MSE | Effective SINR dB | Approx Sum Rate | Fallback | Stage | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"| {row['method']} | {row['native_receiver_success']} | {row['ber_if_available']} | {row['symbol_mse']:.6f} | "
            f"{row['effective_sinr_db']:.6f} | {row['approximate_sum_rate']:.6f} | {row['fallback_used']} | "
            f"{row['fallback_stage']} | {row['fallback_reason']} |"
        )
    lines.extend(["", "## Notes"])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return lines


def _native_receiver_attempt(
    method: str,
    bits: torch.Tensor,
    stream_symbols: torch.Tensor,
    precoder_f: torch.Tensor,
    noise_var: float,
    device: torch.device,
    trace_shapes: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """Attempt a real Sionna-native beamformed receiver path."""
    ApplyOFDMChannel, _, _ = load_component("ApplyOFDMChannel")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    Demapper, _, _ = load_component("Demapper")
    trace: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "native_receiver_attempted": True,
        "native_receiver_success": False,
        "native_failure_stage": "",
        "native_failure_reason": "",
        "channel_extraction_mode": "",
    }
    rg, sm, rg_meta = build_pilot_aware_multiuser_resource_grid(
        num_users=stream_symbols.size(2),
        num_effective_subcarriers=stream_symbols.size(1),
        num_ofdm_symbols=2,
        device=device,
    )
    meta["resource_grid_meta"] = rg_meta
    if rg is None or sm is None:
        meta["native_failure_stage"] = "resource_grid"
        meta["native_failure_reason"] = str(rg_meta.get("fallback_reason", "resource_grid_build_failed"))
        return None, trace, meta

    h_f, h_full, h_meta = extract_effective_channel_from_sionna(
        rg,
        batch_size=stream_symbols.size(0),
        num_users=stream_symbols.size(2),
        num_bs_ant=precoder_f.size(2),
        device=device,
        noise_var=noise_var,
    )
    meta["channel_extraction_mode"] = "native" if h_meta.get("used_native_channel_extraction") else "fallback"
    meta["channel_meta"] = h_meta
    if h_f is None or h_full is None:
        meta["native_failure_stage"] = "channel_extraction"
        meta["native_failure_reason"] = str(h_meta.get("fallback_reason", "channel_extraction_failed"))
        return None, trace, meta

    x_rg, rg_bridge_meta = map_project_streams_to_sionna_rg(stream_symbols, rg)
    meta["resource_grid_bridge_meta"] = rg_bridge_meta
    if x_rg is None:
        meta["native_failure_stage"] = "resource_grid_mapper"
        meta["native_failure_reason"] = str(rg_bridge_meta.get("fallback_reason", "resource_grid_mapping_failed"))
        return None, trace, meta
    if trace_shapes:
        trace.extend(
            [
                describe_tensor("stream_symbols", stream_symbols, ["batch", "effective_subcarrier", "user"]),
                describe_tensor("x_rg", x_rg, ["batch", "num_tx", "num_streams", "ofdm_symbol", "fft_bin"]),
                describe_tensor("H_f", h_f, ["batch", "effective_subcarrier", "user", "bs_ant"]),
                describe_tensor("F_f", precoder_f, ["batch", "effective_subcarrier", "bs_ant", "user"]),
            ]
        )

    tx_grid, tx_meta = apply_project_precoder_to_sionna_grid(x_rg, precoder_f, rg)
    meta["tx_bridge_meta"] = tx_meta
    if tx_grid is None:
        meta["native_failure_stage"] = "precoder_bridge"
        meta["native_failure_reason"] = str(tx_meta.get("fallback_reason", "precoder_bridge_failed"))
        return None, trace, meta
    if trace_shapes:
        trace.append(describe_tensor("tx_grid", tx_grid, ["batch", "num_tx", "num_tx_ant", "ofdm_symbol", "fft_bin"]))

    noise = torch.full((stream_symbols.size(0), stream_symbols.size(2), 1), noise_var, dtype=torch.float32, device=device)
    try:
        rx_grid = ApplyOFDMChannel(device=resolve_sionna_device(device))(tx_grid, h_full, no=noise)
    except Exception as exc:
        meta["native_failure_stage"] = "channel_apply"
        meta["native_failure_reason"] = f"{type(exc).__name__}: {exc}"
        return None, trace, meta
    if trace_shapes:
        trace.append(describe_tensor("rx_grid", rx_grid, ["batch", "num_rx", "num_rx_ant", "ofdm_symbol", "fft_bin"]))

    try:
        estimator = LSChannelEstimator(rg, device=resolve_sionna_device(device))
        h_hat, err_var = estimator(rx_grid, noise)
    except Exception as exc:
        meta["native_failure_stage"] = "estimator"
        meta["native_failure_reason"] = f"{type(exc).__name__}: {exc}"
        return None, trace, meta
    if trace_shapes:
        trace.extend(
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

    validation = validate_sionna_receiver_shapes(rx_grid, h_hat, err_var, sm, rg)
    meta["shape_validation"] = validation
    if not validation["valid"]:
        meta["native_failure_stage"] = "shape_validation"
        meta["native_failure_reason"] = validation["reason"]
        return None, trace, meta

    try:
        equalizer = LMMSEEqualizer(rg, sm, device=resolve_sionna_device(device))
        x_hat, no_eff = equalizer(rx_grid, h_hat, err_var, noise)
    except Exception as exc:
        meta["native_failure_stage"] = "equalizer"
        meta["native_failure_reason"] = f"{type(exc).__name__}: {exc}"
        return None, trace, meta
    if trace_shapes:
        trace.extend(
            [
                describe_tensor("x_hat", x_hat, ["batch", "num_tx", "num_streams", "data_symbols"]),
                describe_tensor("no_eff", no_eff, ["batch", "num_tx", "num_streams", "data_symbols"]),
            ]
        )

    try:
        demapper = Demapper("app", "qam", 2, hard_out=True, device=resolve_sionna_device(device))
        hard_bits = demapper(x_hat, no_eff)
    except Exception as exc:
        meta["native_failure_stage"] = "demapper"
        meta["native_failure_reason"] = f"{type(exc).__name__}: {exc}"
        return None, trace, meta
    if trace_shapes:
        trace.append(describe_tensor("hard_bits", hard_bits, ["batch", "num_tx", "num_streams", "coded_bits"]))

    project_rx, bridge_meta = sionna_rx_to_project_symbols(x_hat)
    if bridge_meta["fallback_used"]:
        meta["native_failure_stage"] = "project_rx_bridge"
        meta["native_failure_reason"] = str(bridge_meta["fallback_reason"])
        return None, trace, meta
    project_ref = stream_symbols
    rx_metrics = compute_project_metrics_from_sionna_rx(project_rx, project_ref)
    bit_ref = bits.permute(0, 2, 1, 3).reshape_as(hard_bits)
    result = {
        "method": method,
        "used_sionna_resource_grid": True,
        "used_sionna_channel": True,
        "used_sionna_estimator": True,
        "used_sionna_equalizer": True,
        "used_sionna_demapper": True,
        "native_receiver_success": True,
        "fallback_used": False,
        "fallback_stage": "",
        "fallback_reason": "",
        "ber_if_available": float((hard_bits.to(torch.int64) != bit_ref.to(torch.int64)).float().mean().item()),
        "symbol_mse": rx_metrics["symbol_mse"],
        "effective_sinr_db": rx_metrics["effective_sinr_db"],
        "approximate_sum_rate": float(stream_symbols.size(2) * torch.log2(torch.tensor(1.0 + (10.0 ** (rx_metrics["effective_sinr_db"] / 10.0)), device=device)).item()),
        "power_norm": float(torch.mean((torch.abs(precoder_f) ** 2).sum(dim=(-2, -1))).item()),
        "power_violation": float(torch.mean(torch.abs((torch.abs(precoder_f) ** 2).sum(dim=(-2, -1)) - 1.0)).item()),
        "runtime_ms": 0.0,
    }
    meta["native_receiver_success"] = True
    return result, trace, meta


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_name = "beamforming_receiver_chain_v2_metrics.csv" if args.enable_receiver_chain else "beamforming_chain_metrics.csv"
    csv_path = out_path.with_name(csv_name)
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary: dict[str, Any] = {
        "demo_scope": "experimental_sionna_native_ofdm_beamforming_receiver_chain_v2" if args.enable_receiver_chain else "experimental_sionna_native_ofdm_beamforming_chain",
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "device": str(device),
        "demo_status": "skipped",
        "used_sionna_resource_grid": False,
        "used_sionna_channel": False,
        "used_sionna_estimator": False,
        "used_sionna_equalizer": False,
        "used_sionna_demapper": False,
        "receiver_chain_enabled": bool(args.enable_receiver_chain),
        "receiver_mode": args.receiver_mode,
        "native_receiver_attempted": False,
        "native_receiver_success": False,
        "native_failure_stage": "",
        "native_failure_reason": "",
        "fallback_used": False,
        "shape_trace_path": None,
        "methods_successful_under_native_receiver": [],
        "methods_fallback_only": [],
        "notes": [],
        "metrics": [],
    }

    if not env["sionna_import_ok"]:
        summary["notes"] = ["Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`."]
        write_json(out_path, summary)
        write_markdown(md_path, _to_md(summary))
        print(f"Saved Sionna native OFDM beamforming summary to {out_path}")
        return

    batch_size = 16
    num_subcarriers = 16
    num_users = 4
    num_bs_ant = 16
    noise_var = 10.0 ** (-10.0 / 10.0)
    methods = ["no_precoding", "project_rzf", "project_wmmse_iter_2", "project_wmmse_iter_5"] if args.enable_receiver_chain else ["no_precoding", "project_rzf", "project_wmmse_iter_1", "project_wmmse_iter_2", "project_wmmse_iter_5"]

    bits, stream_symbols = _qpsk_symbols(batch_size, num_subcarriers, num_users, device)
    h_f, h_meta = build_frequency_domain_channel(
        batch_size=batch_size,
        num_subcarriers=num_subcarriers,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        device=device,
        resource_grid=None,
        noise_var=noise_var,
    )
    summary["used_sionna_resource_grid"] = True
    summary["used_sionna_channel"] = bool(h_meta["used_sionna_channel_tensor"])
    summary["fallback_used"] = bool(h_meta["fallback_used"])
    summary["notes"].extend(h_meta["notes"])

    shape_trace_payload: dict[str, Any] = {"methods": []}
    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == "no_precoding":
            precoder_f = _identity_precoder(batch_size, num_subcarriers, num_users, num_bs_ant, device)
            runtime_ms = 0.0
        else:
            precoder_f, runtime_ms = time_function(compute_project_precoder_per_subcarrier, method.removeprefix("project_"), h_f, noise_var)

        proxy_metrics = evaluate_ofdm_beamforming_outputs(h_f, precoder_f, stream_symbols, noise_var)
        row = {
            "method": method,
            "used_sionna_resource_grid": True,
            "used_sionna_channel": False,
            "used_sionna_estimator": False,
            "used_sionna_equalizer": False,
            "used_sionna_demapper": False,
            "native_receiver_success": False,
            "fallback_used": True,
            "fallback_stage": "",
            "fallback_reason": "",
            "ber_if_available": proxy_metrics["ber_if_available"],
            "symbol_mse": proxy_metrics["symbol_mse"],
            "effective_sinr_db": proxy_metrics["effective_sinr_db"],
            "approximate_sum_rate": proxy_metrics["approximate_sum_rate"],
            "power_norm": proxy_metrics["power_norm"],
            "power_violation": proxy_metrics["power_violation"],
            "runtime_ms": runtime_ms,
        }

        should_try_native = args.enable_receiver_chain and args.receiver_mode in {"native", "auto"}
        if args.receiver_mode == "proxy":
            row["fallback_stage"] = "receiver_mode"
            row["fallback_reason"] = "receiver_mode_proxy_kept_project_side_metrics"
        elif should_try_native:
            native_result, native_trace, native_meta = _native_receiver_attempt(
                method=method,
                bits=bits,
                stream_symbols=stream_symbols,
                precoder_f=precoder_f,
                noise_var=noise_var,
                device=device,
                trace_shapes=args.trace_shapes,
            )
            summary["native_receiver_attempted"] = True
            if args.trace_shapes:
                shape_trace_payload["methods"].append({"method": method, "trace": native_trace, "meta": native_meta})
            if native_result is not None:
                row.update(native_result)
                row["runtime_ms"] = runtime_ms
                summary["methods_successful_under_native_receiver"].append(method)
            else:
                row["fallback_stage"] = native_meta.get("native_failure_stage", "native_receiver")
                row["fallback_reason"] = native_meta.get("native_failure_reason", "native_receiver_failed")
                if args.receiver_mode == "native":
                    row["fallback_used"] = False
                summary["methods_fallback_only"].append(method)
                if not summary["native_failure_stage"]:
                    summary["native_failure_stage"] = row["fallback_stage"]
                    summary["native_failure_reason"] = row["fallback_reason"]
        else:
            row["fallback_stage"] = "receiver_chain"
            row["fallback_reason"] = "receiver_chain_not_enabled"

        rows.append(row)

    if args.trace_shapes:
        shape_trace_path = out_path.with_name("beamforming_receiver_shape_trace_runtime.json")
        write_json(shape_trace_path, shape_trace_payload)
        summary["shape_trace_path"] = str(shape_trace_path)

    summary.update(
        {
            "demo_status": "ok",
            "used_sionna_channel": any(row["used_sionna_channel"] for row in rows),
            "used_sionna_estimator": any(row["used_sionna_estimator"] for row in rows),
            "used_sionna_equalizer": any(row["used_sionna_equalizer"] for row in rows),
            "used_sionna_demapper": any(row["used_sionna_demapper"] for row in rows),
            "native_receiver_success": any(row["native_receiver_success"] for row in rows),
            "fallback_used": summary["fallback_used"] or any(row["fallback_used"] for row in rows),
            "metrics": rows,
            "notes": summary["notes"]
            + [
                "Pilot-aware native beamformed receiver mode uses num_tx=1, num_streams_per_tx=K, rx_tx_association=ones(K,1), and at least one non-pilot OFDM symbol.",
                "receiver-mode=proxy keeps project-side proxy metrics only.",
                "receiver-mode=native requires a real Sionna receiver path and records exact failure stage/reason if it fails.",
                "receiver-mode=auto attempts the native receiver first and falls back to proxy metrics if needed.",
            ],
        }
    )
    _write_csv(csv_path, rows)
    write_json(out_path, summary)
    write_markdown(md_path, _to_md(summary))
    print(f"Saved Sionna native OFDM beamforming summary to {out_path}")


if __name__ == "__main__":
    main()
