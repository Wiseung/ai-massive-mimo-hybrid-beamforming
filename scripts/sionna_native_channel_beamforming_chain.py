#!/usr/bin/env python
"""Use extracted Sionna channel tensors to drive project precoders and learned beamformers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_channel_extraction import extract_h_f_from_sionna_channel
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import build_pilot_aware_multiuser_resource_grid, compute_project_precoder_per_subcarrier, resolve_sionna_device
from beamforming.utils.sionna_native_chain import load_component, write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import (
    build_native_receiver_context,
    default_checkpoint_path,
    infer_learned_precoder,
    load_learned_beamformer_checkpoint,
    run_native_receiver_with_precoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--receiver-mode", choices=["proxy", "native", "auto"], default="auto")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Native Channel Beamforming Summary",
        "",
        f"- extraction_success: `{summary['extraction_success']}`",
        f"- project_h_f_assisted: `{summary['project_h_f_assisted']}`",
        f"- native_receiver_success: `{summary['native_receiver_success']}`",
        "",
        "| Method | Extracted H_f | Native OK | Teacher Inference | Sum Rate | Fallback | Reason |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"| {row['method']} | {row['extracted_h_f_used']} | {row['native_receiver_success']} | {row['teacher_used_during_inference']} | "
            f"{row['approximate_sum_rate']:.6f} | {row['fallback_used']} | {row['fallback_reason']} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_name("native_channel_beamforming_metrics.csv")
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sionna_device = resolve_sionna_device(device)
    repo_root = Path(__file__).resolve().parents[1]

    summary: dict[str, Any] = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "receiver_mode": args.receiver_mode,
        "extraction_success": False,
        "project_h_f_assisted": True,
        "native_receiver_success": False,
        "notes": [],
        "metrics": [],
    }
    if not env["sionna_import_ok"]:
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved native-channel beamforming summary to {out_path}")
        return

    context = build_native_receiver_context(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
    )
    rg = context.resource_grid
    OFDMChannel, _, _ = load_component("OFDMChannel")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
    extracted_h_f = None
    if OFDMChannel is not None and RayleighBlockFading is not None:
        channel_model = RayleighBlockFading(num_rx=4, num_rx_ant=1, num_tx=1, num_tx_ant=16, device=sionna_device)
        channel = OFDMChannel(channel_model, rg, return_channel=True, device=sionna_device)
        dummy_x = torch.zeros(16, 1, 16, rg.num_ofdm_symbols, rg.fft_size, dtype=torch.complex64, device=device)
        _, h = channel(dummy_x, no=torch.full((16, 4, 1), context.noise_var, dtype=torch.float32, device=device))
        extracted_h_f, extraction_meta, extraction_success, fallback_reason = extract_h_f_from_sionna_channel(
            h,
            resource_grid=rg,
            num_users=4,
            num_bs_ant=16,
        )
        summary["extraction_success"] = bool(extraction_success)
        summary["project_h_f_assisted"] = not extraction_success
        summary["extraction_meta"] = extraction_meta
        if not extraction_success:
            summary["notes"].append(f"Fell back to project-assisted H_f: {fallback_reason}")

    h_f_for_methods = extracted_h_f if extracted_h_f is not None else context.h_f
    methods = [
        ("project_rzf_from_extracted_h", "analytic"),
        ("project_wmmse_iter_5_from_extracted_h", "analytic"),
        ("learned_residual_rzf_from_extracted_h", "learned"),
        ("learned_residual_wmmse_distill_from_extracted_h", "learned"),
    ]
    rows: list[dict[str, Any]] = []
    for method, method_type in methods:
        if method_type == "analytic":
            base = method.replace("_from_extracted_h", "").replace("project_", "")
            precoder = compute_project_precoder_per_subcarrier(base, h_f_for_methods, context.noise_var)
            runtime_ms = 0.0
            teacher_flag = False
            checkpoint_path = None
        else:
            learned_name = method.replace("_from_extracted_h", "")
            ckpt = default_checkpoint_path(learned_name, repo_root)
            if not ckpt.exists():
                rows.append(
                    {
                        "method": method,
                        "extraction_success": summary["extraction_success"],
                        "project_h_f_assisted": not bool(extracted_h_f is not None),
                        "extracted_h_f_used": bool(extracted_h_f is not None),
                        "fallback_used": True,
                        "fallback_reason": "skipped_missing_checkpoint",
                        "native_receiver_success": False,
                        "teacher_used_during_inference": False,
                        "ber_if_available": None,
                        "symbol_mse": float("nan"),
                        "effective_sinr_db": float("nan"),
                        "approximate_sum_rate": float("nan"),
                        "power_norm": float("nan"),
                        "runtime_ms": 0.0,
                    }
                )
                continue
            bundle = load_learned_beamformer_checkpoint(ckpt, device, method_name=learned_name)
            snr_tensor = torch.full((h_f_for_methods.size(0),), context.snr_db, dtype=torch.float32, device=device)
            precoder, infer_meta, runtime_ms = infer_learned_precoder(bundle, h_f_for_methods, snr_tensor, native_receiver_path=True)
            teacher_flag = bool(infer_meta["teacher_used_during_inference"])
            checkpoint_path = str(ckpt)

        method_context = context
        if extracted_h_f is not None:
            method_context = type(context)(
                bits=context.bits,
                stream_symbols=context.stream_symbols,
                resource_grid=context.resource_grid,
                stream_management=context.stream_management,
                h_f=extracted_h_f,
                h_full=context.h_full,
                noise_var=context.noise_var,
                snr_db=context.snr_db,
                device=context.device,
                context_meta={**context.context_meta, "project_h_f_assisted": False, "extracted_h_f_used": True},
            )
        row, _, _ = run_native_receiver_with_precoder(
            method=method,
            method_type=method_type,
            precoder_f=precoder,
            context=method_context,
            runtime_ms=runtime_ms,
            checkpoint_path=checkpoint_path,
            teacher_used_during_inference=teacher_flag,
            trace_shapes=False,
        )
        row["extraction_success"] = summary["extraction_success"]
        row["project_h_f_assisted"] = not bool(extracted_h_f is not None)
        row["extracted_h_f_used"] = bool(extracted_h_f is not None)
        rows.append(row)

    summary["metrics"] = rows
    summary["native_receiver_success"] = any(bool(row["native_receiver_success"]) for row in rows)
    _write_csv(csv_path, rows)
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved native-channel beamforming summary to {out_path}")


if __name__ == "__main__":
    main()
