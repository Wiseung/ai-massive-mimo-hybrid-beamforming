#!/usr/bin/env python
"""Insert beamforming into the Sionna-native OFDM chain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import (
    apply_precoder_to_resource_grid,
    build_frequency_domain_channel,
    compute_project_precoder_per_subcarrier,
    evaluate_ofdm_beamforming_outputs,
    time_function,
)
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--enable-receiver-chain", action="store_true")
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


def _receiver_chain_grid(
    bits: torch.Tensor,
    tx_precoded: torch.Tensor,
    noise_var: float,
    rg: Any,
    sm: Any,
    device: torch.device,
    sionna_device: str | None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Try a real Sionna receiver chain for the beamformed path.

    Current expected shapes:
    - bits: (B, Nsc, K, 2)
    - tx_precoded: (B, Nt, Nsc)
    - rg: pilot-enabled ResourceGrid with num_ofdm_symbols=1
    """
    OFDMChannel, _, _ = load_component("OFDMChannel")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    Demapper, _, _ = load_component("Demapper")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
    if not all([OFDMChannel, LSChannelEstimator, LMMSEEqualizer, Demapper, RayleighBlockFading]):
        return None, "receiver_components", "one_or_more_receiver_components_unavailable"
    try:
        batch_size = bits.size(0)
        num_users = bits.size(2)
        num_bs_ant = tx_precoded.size(1)
        channel_model = RayleighBlockFading(num_rx=num_users, num_rx_ant=1, num_tx=1, num_tx_ant=num_bs_ant, device=sionna_device)
        channel = OFDMChannel(channel_model, rg, return_channel=True, device=sionna_device)
        tx_grid = tx_precoded.unsqueeze(1).unsqueeze(-2)
        rx_grid, _ = channel(tx_grid, no=torch.full((batch_size, num_users, 1), noise_var, dtype=torch.float32, device=device))
        estimator = LSChannelEstimator(rg, device=sionna_device)
        h_hat, err_var = estimator(rx_grid, torch.full((batch_size, num_users, 1), noise_var, dtype=torch.float32, device=device))
    except Exception as exc:  # pragma: no cover
        return None, "receiver_channel_or_estimator", f"{type(exc).__name__}: {exc}"
    try:
        equalizer = LMMSEEqualizer(rg, sm, device=sionna_device)
        x_hat, no_eff = equalizer(rx_grid, h_hat, err_var, torch.full((batch_size, num_users, 1), noise_var, dtype=torch.float32, device=device))
    except Exception as exc:  # pragma: no cover
        return None, "receiver_equalizer", f"{type(exc).__name__}: {exc}"
    try:
        demapper = Demapper("app", "qam", 2, hard_out=True, device=sionna_device)
        hard_bits = demapper(x_hat, torch.full((batch_size, 1, num_users, 1), noise_var, dtype=torch.float32, device=device))
        ber = float((hard_bits.to(torch.int64) != bits.transpose(1, 2).reshape_as(hard_bits)).float().mean().item())
        return {
            "used_sionna_estimator": True,
            "used_sionna_equalizer": True,
            "used_sionna_demapper": True,
            "ber_if_available": ber,
            "x_hat": x_hat,
            "no_eff_mean": float(no_eff.mean().item()),
        }, None, None
    except Exception as exc:  # pragma: no cover
        return None, "receiver_demapper", f"{type(exc).__name__}: {exc}"


def _md(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Sionna Native OFDM Beamforming Chain",
        "",
        f"- Demo status: `{summary['demo_status']}`",
        f"- Used Sionna ResourceGrid: `{summary['used_sionna_resource_grid']}`",
        f"- Used Sionna channel: `{summary['used_sionna_channel']}`",
        f"- Used Sionna estimator: `{summary['used_sionna_estimator']}`",
        f"- Used Sionna equalizer: `{summary['used_sionna_equalizer']}`",
        f"- Used Sionna demapper: `{summary['used_sionna_demapper']}`",
        f"- Fallback used: `{summary['fallback_used']}`",
        "",
        "| Method | BER | Symbol MSE | Effective SINR dB | Approx Sum Rate | Fallback | Reason |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"| {row['method']} | {row['ber_if_available']} | {row['symbol_mse']:.6f} | {row['effective_sinr_db']:.6f} | "
            f"{row['approximate_sum_rate']:.6f} | {row['fallback_used']} | {row['fallback_reason']} |"
        )
    lines.extend(["", "## Notes"])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_name("beamforming_receiver_chain_metrics.csv" if args.enable_receiver_chain else "beamforming_chain_metrics.csv")
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sionna_device = resolve_sionna_device(device)

    summary: dict[str, Any] = {
        "demo_scope": "experimental_sionna_native_ofdm_beamforming_receiver_chain" if args.enable_receiver_chain else "experimental_sionna_native_ofdm_beamforming_chain",
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "device": str(device),
        "demo_status": "skipped",
        "used_sionna_resource_grid": False,
        "used_sionna_channel": False,
        "used_sionna_estimator": False,
        "used_sionna_equalizer": False,
        "used_sionna_demapper": False,
        "fallback_used": False,
        "notes": [],
        "receiver_chain_enabled": bool(args.enable_receiver_chain),
        "metrics": [],
    }

    if not env["sionna_import_ok"]:
        summary["notes"] = ["Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`."]
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved Sionna native OFDM beamforming summary to {out_path}")
        return

    ResourceGrid, _, _ = load_component("ResourceGrid")
    OFDMChannel, _, _ = load_component("OFDMChannel")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    Demapper, _, _ = load_component("Demapper")
    RZFPrecoder, _, _ = load_component("RZFPrecoder")
    StreamManagement, _, _ = load_component("StreamManagement")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")

    batch_size = 16
    num_subcarriers = 16
    num_users = 4
    num_bs_ant = 16
    noise_var = 10.0 ** (-10.0 / 10.0)
    methods = ["no_precoding", "project_rzf", "project_wmmse_iter_2", "project_wmmse_iter_5"] if args.enable_receiver_chain else ["no_precoding", "project_rzf", "project_wmmse_iter_1", "project_wmmse_iter_2", "project_wmmse_iter_5"]

    rg = ResourceGrid(
        num_ofdm_symbols=1,
        fft_size=num_subcarriers,
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
    summary["used_sionna_resource_grid"] = True

    bits, stream_symbols = _qpsk_symbols(batch_size, num_subcarriers, num_users, device)
    h_f, h_meta = build_frequency_domain_channel(
        batch_size=batch_size,
        num_subcarriers=num_subcarriers,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        device=device,
        resource_grid=rg,
        noise_var=noise_var,
    )
    summary["used_sionna_channel"] = bool(h_meta["used_sionna_channel_tensor"])
    summary["fallback_used"] = bool(h_meta["fallback_used"])
    summary["notes"].extend(h_meta["notes"])

    if RZFPrecoder is not None:
        methods.append("sionna_rzf_precoder")

    rows: list[dict[str, Any]] = []
    for method in methods:
        fallback_reason = ""
        local_fallback = False
        used_sionna_estimator = False
        used_sionna_equalizer = False
        used_sionna_demapper = False
        if method == "no_precoding":
            precoder_f = _identity_precoder(batch_size, num_subcarriers, num_users, num_bs_ant, device)
            runtime_ms = 0.0
            fallback_reason = "reference_only_identity_precoder"
        elif method.startswith("project_"):
            project_method = method.removeprefix("project_")
            precoder_f, runtime_ms = time_function(compute_project_precoder_per_subcarrier, project_method, h_f, noise_var)
        else:
            fallback_reason = "sionna_rzf_shape_probe_only"
            local_fallback = True
            runtime_ms = 0.0
            try:
                precoder = RZFPrecoder(rg, sm, return_effective_channel=True, device=sionna_device)
                x = torch.randn(batch_size, 1, num_users, 1, num_subcarriers, dtype=torch.complex64, device=device)
                h = torch.randn(batch_size, 1, num_users, 1, num_bs_ant, 1, num_subcarriers, dtype=torch.complex64, device=device)
                _, _ = precoder(x, h)
                fallback_reason = "sionna_rzf_callable_but_not_used_for_project_H_f_layout"
            except Exception as exc:  # pragma: no cover
                fallback_reason = f"sionna_rzf_incompatible: {type(exc).__name__}: {exc}"
            rows.append(
                {
                    "method": method,
                    "used_sionna_resource_grid": True,
                    "used_sionna_channel": summary["used_sionna_channel"],
                    "used_sionna_estimator": False,
                    "used_sionna_equalizer": False,
                    "used_sionna_demapper": False,
                    "fallback_used": True,
                    "fallback_reason": fallback_reason,
                    "ber_if_available": None,
                    "symbol_mse": float("nan"),
                    "effective_sinr_db": float("nan"),
                    "approximate_sum_rate": float("nan"),
                    "power_norm": float("nan"),
                    "power_violation": float("nan"),
                    "runtime_ms": runtime_ms,
                }
            )
            continue

        tx_precoded = apply_precoder_to_resource_grid(stream_symbols, precoder_f)
        metrics = evaluate_ofdm_beamforming_outputs(h_f, precoder_f, stream_symbols, noise_var)
        if args.enable_receiver_chain:
            receiver_result, fallback_stage, receiver_reason = _receiver_chain_grid(bits, tx_precoded, noise_var, rg, sm, device, sionna_device)
            if receiver_result is not None:
                used_sionna_estimator = True
                used_sionna_equalizer = True
                used_sionna_demapper = True
                metrics["ber_if_available"] = receiver_result["ber_if_available"]
            else:
                local_fallback = True
                fallback_reason = receiver_reason or "receiver_chain_unknown_failure"
        else:
            local_fallback = True
            fallback_reason = "receiver_chain_not_enabled"

        rows.append(
            {
                "method": method,
                "used_sionna_resource_grid": True,
                "used_sionna_channel": summary["used_sionna_channel"],
                "used_sionna_estimator": used_sionna_estimator,
                "used_sionna_equalizer": used_sionna_equalizer,
                "used_sionna_demapper": used_sionna_demapper,
                "fallback_used": local_fallback,
                "fallback_stage": "receiver_chain" if local_fallback else "",
                "fallback_reason": fallback_reason,
                "ber_if_available": metrics["ber_if_available"],
                "symbol_mse": metrics["symbol_mse"],
                "effective_sinr_db": metrics["effective_sinr_db"],
                "approximate_sum_rate": metrics["approximate_sum_rate"],
                "power_norm": metrics["power_norm"],
                "power_violation": metrics["power_violation"],
                "runtime_ms": runtime_ms,
            }
        )

    summary.update(
        {
            "demo_status": "ok",
            "used_sionna_estimator": any(row["used_sionna_estimator"] for row in rows),
            "used_sionna_equalizer": any(row["used_sionna_equalizer"] for row in rows),
            "used_sionna_demapper": any(row["used_sionna_demapper"] for row in rows),
            "fallback_used": summary["fallback_used"] or any(row["fallback_used"] for row in rows),
            "metrics": rows,
            "notes": summary["notes"]
            + [
                "Pilot-enabled ResourceGrid is required when --enable-receiver-chain is used.",
                "Project frequency-domain precoders are the current clean insertion path because they match H_f=(B,Nsc,K,Nt) directly.",
                "Optional Sionna RZFPrecoder is audited separately and recorded only as a shape-checked reference path.",
            ],
        }
    )
    _write_csv(csv_path, rows)
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved Sionna native OFDM beamforming summary to {out_path}")


if __name__ == "__main__":
    main()
