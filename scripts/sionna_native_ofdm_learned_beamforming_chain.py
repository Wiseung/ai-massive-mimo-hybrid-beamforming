#!/usr/bin/env python
"""Evaluate learned beamformers inside the Sionna-native receiver chain."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import compute_project_precoder_per_subcarrier, evaluate_ofdm_beamforming_outputs, time_function
from beamforming.utils.sionna_native_chain import write_json, write_markdown
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
    parser.add_argument("--trace-shapes", action="store_true")
    return parser.parse_args()


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
        "# Sionna Native Learned Beamforming Chain",
        "",
        f"- Demo status: `{summary['demo_status']}`",
        f"- native_receiver_attempted: `{summary['native_receiver_attempted']}`",
        f"- methods_successful_under_native_receiver: `{summary['methods_successful_under_native_receiver']}`",
        f"- methods_skipped_missing_checkpoint: `{summary['methods_skipped_missing_checkpoint']}`",
        "",
        "| Method | Type | Native OK | Teacher Inference | BER | MSE | SINR dB | Sum Rate | Fallback | Reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"| {row['method']} | {row['method_type']} | {row['native_receiver_success']} | {row['teacher_used_during_inference']} | "
            f"{row['ber_if_available']} | {row['symbol_mse']:.6f} | {row['effective_sinr_db']:.6f} | {row['approximate_sum_rate']:.6f} | "
            f"{row['fallback_used']} | {row['fallback_reason']} |"
        )
    lines.extend(["", "## Notes"])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_name("learned_beamforming_receiver_metrics.csv")
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[1]

    summary: dict[str, Any] = {
        "demo_scope": "experimental_sionna_native_ofdm_learned_beamforming_chain",
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "device": str(device),
        "receiver_mode": args.receiver_mode,
        "demo_status": "skipped",
        "native_receiver_attempted": False,
        "native_receiver_success": False,
        "methods_successful_under_native_receiver": [],
        "methods_fallback_only": [],
        "methods_skipped_missing_checkpoint": [],
        "used_sionna_resource_grid": False,
        "used_sionna_channel": False,
        "used_sionna_estimator": False,
        "used_sionna_equalizer": False,
        "used_sionna_demapper": False,
        "shape_trace_path": None,
        "notes": [],
        "metrics": [],
    }
    if not env["sionna_import_ok"]:
        summary["notes"] = ["Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`."]
        write_json(out_path, summary)
        write_markdown(md_path, _to_md(summary))
        print(f"Saved native learned beamforming summary to {out_path}")
        return

    context = build_native_receiver_context(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
    )
    summary["used_sionna_resource_grid"] = True
    summary["notes"].append("Receiver chain is a real Sionna-native path, but precoder/H_f remains project-assisted.")
    summary["notes"].append("This is still a synthetic/channel-level benchmark, not a full native-only or production e2e system.")

    methods = [
        ("no_precoding", "reference"),
        ("project_rzf", "analytic"),
        ("project_wmmse_iter_2", "analytic"),
        ("project_wmmse_iter_5", "analytic"),
        ("learned_residual_rzf", "learned"),
        ("learned_residual_wmmse_distill", "learned"),
    ]
    rows: list[dict[str, Any]] = []
    trace_payload: dict[str, Any] = {"methods": []}
    for method, method_type in methods:
        if method == "no_precoding":
            precoder_f = _identity_precoder(context.h_f.size(0), context.h_f.size(1), context.h_f.size(2), context.h_f.size(3), device)
            runtime_ms = 0.0
            teacher_flag = False
            checkpoint_path = None
        elif method.startswith("project_"):
            precoder_f, runtime_ms = time_function(compute_project_precoder_per_subcarrier, method.removeprefix("project_"), context.h_f, context.noise_var)
            teacher_flag = False
            checkpoint_path = None
        else:
            ckpt = default_checkpoint_path(method, repo_root)
            if not ckpt.exists():
                summary["methods_skipped_missing_checkpoint"].append(method)
                rows.append(
                    {
                        "method": method,
                        "method_type": method_type,
                        "checkpoint_path": str(ckpt),
                        "native_receiver_success": False,
                        "used_sionna_resource_grid": True,
                        "used_sionna_channel": False,
                        "used_sionna_estimator": False,
                        "used_sionna_equalizer": False,
                        "used_sionna_demapper": False,
                        "teacher_used_during_inference": False,
                        "fallback_used": True,
                        "fallback_stage": "checkpoint",
                        "fallback_reason": "skipped_missing_checkpoint",
                        "ber_if_available": None,
                        "symbol_mse": float("nan"),
                        "effective_sinr_db": float("nan"),
                        "approximate_sum_rate": float("nan"),
                        "power_norm": float("nan"),
                        "runtime_ms": 0.0,
                    }
                )
                continue
            bundle = load_learned_beamformer_checkpoint(ckpt, device, method_name=method)
            snr_tensor = torch.full((context.h_f.size(0),), context.snr_db, dtype=torch.float32, device=device)
            precoder_f, infer_meta, runtime_ms = infer_learned_precoder(bundle, context.h_f, snr_tensor, native_receiver_path=True)
            teacher_flag = bool(infer_meta["teacher_used_during_inference"])
            checkpoint_path = str(ckpt)

        native_result, native_trace, native_meta = run_native_receiver_with_precoder(
            method=method,
            method_type=method_type,
            precoder_f=precoder_f,
            context=context,
            runtime_ms=runtime_ms,
            checkpoint_path=checkpoint_path,
            teacher_used_during_inference=teacher_flag,
            trace_shapes=args.trace_shapes,
        )
        if args.trace_shapes:
            trace_payload["methods"].append({"method": method, "trace": native_trace, "meta": native_meta})
        if native_result["native_receiver_success"]:
            summary["methods_successful_under_native_receiver"].append(method)
        else:
            summary["methods_fallback_only"].append(method)
        summary["native_receiver_attempted"] = True
        rows.append(native_result)

    if args.trace_shapes:
        shape_trace_path = out_path.with_name("learned_beamforming_receiver_trace.json")
        write_json(shape_trace_path, trace_payload)
        summary["shape_trace_path"] = str(shape_trace_path)

    summary.update(
        {
            "demo_status": "ok",
            "native_receiver_success": any(row["native_receiver_success"] for row in rows),
            "used_sionna_channel": any(row["used_sionna_channel"] for row in rows),
            "used_sionna_estimator": any(row["used_sionna_estimator"] for row in rows),
            "used_sionna_equalizer": any(row["used_sionna_equalizer"] for row in rows),
            "used_sionna_demapper": any(row["used_sionna_demapper"] for row in rows),
            "metrics": rows,
        }
    )
    _write_csv(csv_path, rows)
    write_json(out_path, summary)
    write_markdown(md_path, _to_md(summary))
    print(f"Saved native learned beamforming summary to {out_path}")


if __name__ == "__main__":
    main()
