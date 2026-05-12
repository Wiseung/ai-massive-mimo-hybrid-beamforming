#!/usr/bin/env python
"""Run the native-channel-assisted beamforming chain through the CSI interface."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.csi_interface import as_project_h_f, summarize_csi_input
from beamforming.utils.precoder_interface import summarize_precoder_input
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import compute_project_precoder_per_subcarrier
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import (
    build_native_receiver_context,
    clone_native_receiver_context,
    default_checkpoint_path,
    infer_learned_precoder,
    load_learned_beamformer_checkpoint,
    run_native_receiver_with_precoder,
)
from beamforming.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--receiver-mode", choices=["proxy", "native", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw-f-f",
        action="store_true",
        help="Use the legacy raw F_f tensor path instead of the preferred PrecoderOutput interface.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# CSI-backed Beamforming Summary",
        "",
        f"- csi_interface_used: `{summary['csi_interface_used']}`",
        f"- csi_source: `{summary['csi_source']}`",
        f"- project_h_f_assisted: `{summary['project_h_f_assisted']}`",
        f"- extracted_h_f_used: `{summary['extracted_h_f_used']}`",
        f"- full_native_only: `{summary['full_native_only']}`",
        f"- native_receiver_success: `{summary['native_receiver_success']}`",
        f"- precoder_interface_used: `{summary['precoder_interface_used']}`",
        "",
        "| Method | Precoder Input | Native OK | Teacher Inference | Sum Rate | Fallback | Reason |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in summary["metrics"]:
        sum_rate = row["approximate_sum_rate"]
        sum_rate_text = "nan" if sum_rate != sum_rate else f"{sum_rate:.6f}"
        lines.append(
            f"| {row['method']} | {row.get('precoder_input_type')} | {row['native_receiver_success']} | {row['teacher_used_during_inference']} | "
            f"{sum_rate_text} | {row['fallback_used']} | {row['fallback_reason']} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_name("csi_backed_beamforming_metrics.csv")
    env = collect_sionna_env_info()
    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    summary: dict[str, Any] = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "receiver_mode": args.receiver_mode,
        "seed": int(args.seed),
        "csi_interface_used": True,
        "precoder_interface_supported": True,
        "precoder_interface_requested": not bool(args.raw_f_f),
        "precoder_interface_used": False,
        "csi_source": "sionna_ofdm_channel",
        "project_h_f_assisted": False,
        "extracted_h_f_used": True,
        "full_native_only": False,
        "native_receiver_success": False,
        "teacher_used_during_inference": False,
        "methods_evaluated": [],
        "skipped_missing_checkpoint": [],
        "csi_summary": None,
        "input_type": None,
        "csi_input_summary": None,
        "metrics": [],
    }
    use_precoder_output = not bool(args.raw_f_f)
    if not env["sionna_import_ok"]:
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved CSI-backed beamforming summary to {out_path}")
        return

    context = build_native_receiver_context(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
    )
    if context.csi is None:
        summary["project_h_f_assisted"] = True
        summary["extracted_h_f_used"] = False
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved CSI-backed beamforming summary to {out_path}")
        return

    csi = context.csi
    summary["csi_summary"] = csi.summary_dict()
    csi_input = csi
    h_f, csi_input_meta = as_project_h_f(csi_input)
    csi_input_summary = summarize_csi_input(csi_input)
    summary["input_type"] = csi_input_summary["input_type"]
    summary["csi_input_summary"] = csi_input_summary

    methods = [
        ("project_rzf", "analytic"),
        ("project_wmmse_iter_5", "analytic"),
        ("learned_residual_rzf", "learned"),
        ("learned_residual_wmmse_distill", "learned"),
    ]
    rows: list[dict[str, Any]] = []
    for method, method_type in methods:
        summary["methods_evaluated"].append(method)
        checkpoint_path = None
        teacher_flag = False
        runtime_ms = 0.0
        if method_type == "analytic":
            precoder = compute_project_precoder_per_subcarrier(
                method.removeprefix("project_"),
                csi_input,
                context.noise_var,
                return_precoder_output=use_precoder_output,
            )
        else:
            ckpt = default_checkpoint_path(method, repo_root)
            if not ckpt.exists():
                summary["skipped_missing_checkpoint"].append(method)
                rows.append(
                    {
                        "method": method,
                        "method_type": method_type,
                        "checkpoint_path": None,
                        "input_type": csi_input_summary["input_type"],
                        "csi_interface_used": True,
                        "extraction_success": True,
                        "project_h_f_assisted": False,
                        "extracted_h_f_used": True,
                        "full_native_only": False,
                        "native_receiver_success": False,
                        "teacher_used_during_inference": False,
                        "precoder_interface_used": bool(use_precoder_output),
                        "precoder_input_type": "PrecoderOutput" if use_precoder_output else "raw_f_f",
                        "precoder_source": method,
                        "project_side_precoder": True,
                        "sionna_native_precoder": False,
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
            snr_tensor = torch.full((h_f.size(0),), context.snr_db, dtype=torch.float32, device=device)
            precoder, infer_meta, runtime_ms = infer_learned_precoder(
                bundle,
                csi_input,
                snr_tensor,
                native_receiver_path=True,
                return_precoder_output=use_precoder_output,
            )
            checkpoint_path = str(ckpt)
            teacher_flag = bool(infer_meta["teacher_used_during_inference"])

        method_context = clone_native_receiver_context(
            context,
            h_f=h_f,
            csi=csi,
            h_full=context.h_full,
            context_meta_updates={
                "project_h_f_assisted": False,
                "extracted_h_f_used": True,
                "csi_interface_used": True,
                "csi_summary": csi.summary_dict(),
            },
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
        row["csi_interface_used"] = True
        row["input_type"] = csi_input_summary["input_type"]
        row["extraction_success"] = True
        row["project_h_f_assisted"] = False
        row["extracted_h_f_used"] = True
        row["full_native_only"] = False
        row["precoder_summary"] = summarize_precoder_input(precoder)
        rows.append(row)

    summary["metrics"] = rows
    summary["native_receiver_success"] = any(bool(row["native_receiver_success"]) for row in rows)
    summary["teacher_used_during_inference"] = any(bool(row["teacher_used_during_inference"]) for row in rows)
    summary["precoder_interface_used"] = bool(rows) and all(bool(row.get("precoder_interface_used")) for row in rows if row["fallback_reason"] != "skipped_missing_checkpoint")
    _write_csv(csv_path, rows)
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved CSI-backed beamforming summary to {out_path}")


if __name__ == "__main__":
    main()
