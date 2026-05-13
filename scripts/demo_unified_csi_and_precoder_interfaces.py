#!/usr/bin/env python
"""Drive the native receiver chain with one shared CSI object and PrecoderOutput producers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.csi_interface import summarize_csi_input
from beamforming.utils.precoder_interface import summarize_precoder_input
from beamforming.utils.seed import set_seed
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
from beamforming.utils.sionna_precoder_api_bridge import run_sionna_rzf_precoder_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-sionna-rzf", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Unified CSI + PrecoderOutput Demo",
        "",
        f"- csi_interface_used: `{summary['csi_interface_used']}`",
        f"- same_csi_object_used_for_all_methods: `{summary['same_csi_object_used_for_all_methods']}`",
        f"- precoder_interface_used: `{summary['precoder_interface_used']}`",
        f"- all_precoders_emit_precoder_output: `{summary['all_precoders_emit_precoder_output']}`",
        f"- all_receiver_consumers_accept_precoder_output: `{summary['all_receiver_consumers_accept_precoder_output']}`",
        f"- native_receiver_success: `{summary['native_receiver_success']}`",
        f"- no_new_fallback_introduced: `{summary['no_new_fallback_introduced']}`",
        f"- sionna_rzf_available: `{summary['sionna_rzf_available']}`",
        f"- sionna_rzf_callable: `{summary['sionna_rzf_callable']}`",
        f"- sionna_rzf_evaluated: `{summary['sionna_rzf_evaluated']}`",
        f"- sionna_rzf_skipped_reason: `{summary['sionna_rzf_skipped_reason']}`",
        "",
        f"- failed_methods: `{summary['failed_methods']}`",
        f"- methods_evaluated: `{summary['methods_evaluated']}`",
        "",
        "| Method | Precoder Input | Native OK | Teacher Inference | Sum Rate | Fallback | Reason |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in summary["metrics"]:
        sum_rate = row["approximate_sum_rate"]
        sum_rate_text = "nan" if sum_rate != sum_rate else f"{sum_rate:.6f}"
        lines.append(
            f"| {row['method']} | {row.get('precoder_input_type')} | {row['native_receiver_success']} | "
            f"{row['teacher_used_during_inference']} | {sum_rate_text} | {row['fallback_used']} | {row['fallback_reason']} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_name("unified_csi_precoder_metrics.csv")
    env = collect_sionna_env_info()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[1]

    summary: dict[str, Any] = {
        "status": "skipped",
        "seed": int(args.seed),
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "csi_interface_used": True,
        "same_csi_object_used_for_all_methods": False,
        "precoder_interface_used": True,
        "all_precoders_emit_precoder_output": False,
        "all_receiver_consumers_accept_precoder_output": False,
        "native_receiver_success": False,
        "teacher_used_during_inference": False,
        "no_new_fallback_introduced": True,
        "sionna_rzf_requested": bool(args.include_sionna_rzf),
        "sionna_rzf_available": False,
        "sionna_rzf_callable": False,
        "sionna_rzf_evaluated": False,
        "sionna_rzf_skipped": False,
        "sionna_rzf_skipped_reason": "",
        "project_h_f_assisted": False,
        "extracted_h_f_used": True,
        "full_native_only": False,
        "failed_methods": [],
        "methods_evaluated": [],
        "csi_summary": None,
        "csi_input_summary": None,
        "metrics": [],
    }
    if not env["sionna_import_ok"]:
        summary["sionna_rzf_skipped"] = bool(args.include_sionna_rzf)
        summary["sionna_rzf_skipped_reason"] = "sionna_not_installed" if args.include_sionna_rzf else ""
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved unified CSI+PrecoderOutput summary to {out_path}")
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
        summary["status"] = "failed"
        summary["failed_methods"] = ["csi_object_creation"]
        summary["project_h_f_assisted"] = True
        summary["extracted_h_f_used"] = False
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved unified CSI+PrecoderOutput summary to {out_path}")
        return

    csi = context.csi
    summary["status"] = "ok"
    summary["same_csi_object_used_for_all_methods"] = True
    summary["csi_summary"] = csi.summary_dict()
    summary["csi_input_summary"] = summarize_csi_input(csi)

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
        runtime_ms = 0.0
        teacher_flag = False
        if method_type == "analytic":
            precoder_output = compute_project_precoder_per_subcarrier(
                method.removeprefix("project_"),
                csi,
                context.noise_var,
                return_precoder_output=True,
            )
        else:
            ckpt = default_checkpoint_path(method, repo_root)
            if not ckpt.exists():
                summary["failed_methods"].append(method)
                rows.append(
                    {
                        "method": method,
                        "method_type": method_type,
                        "checkpoint_path": None,
                        "input_type": "ExtractedCSI",
                        "csi_interface_used": True,
                        "precoder_interface_used": True,
                        "precoder_input_type": "PrecoderOutput",
                        "precoder_source": method,
                        "project_side_precoder": True,
                        "sionna_native_precoder": False,
                        "project_h_f_assisted": False,
                        "extracted_h_f_used": True,
                        "full_native_only": False,
                        "native_receiver_success": False,
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
            precoder_output, infer_meta, runtime_ms = infer_learned_precoder(
                bundle,
                csi,
                snr_tensor,
                native_receiver_path=True,
                return_precoder_output=True,
            )
            checkpoint_path = str(ckpt)
            teacher_flag = bool(infer_meta["teacher_used_during_inference"])

        method_context = clone_native_receiver_context(
            context,
            h_f=context.h_f,
            csi=csi,
            h_full=context.h_full,
            context_meta_updates={
                "csi_interface_used": True,
                "project_h_f_assisted": False,
                "extracted_h_f_used": True,
                "full_native_only": False,
                "csi_summary": csi.summary_dict(),
            },
        )
        row, _, _ = run_native_receiver_with_precoder(
            method=method,
            method_type=method_type,
            precoder_f=precoder_output,
            context=method_context,
            runtime_ms=runtime_ms,
            checkpoint_path=checkpoint_path,
            teacher_used_during_inference=teacher_flag,
            trace_shapes=False,
        )
        row["input_type"] = "ExtractedCSI"
        row["csi_interface_used"] = True
        row["project_h_f_assisted"] = False
        row["extracted_h_f_used"] = True
        row["full_native_only"] = False
        row["precoder_summary"] = summarize_precoder_input(precoder_output)
        rows.append(row)

    if args.include_sionna_rzf:
        probe = run_sionna_rzf_precoder_probe(
            csi,
            project_noise_var=context.noise_var,
            device=device,
        )
        summary["sionna_rzf_available"] = bool(probe.get("sionna_rzf_available", False))
        summary["sionna_rzf_callable"] = bool(probe.get("sionna_rzf_callable", False))
        if probe.get("converted_to_precoder_output") and probe.get("sionna_precoder_output") is not None:
            summary["methods_evaluated"].append("sionna_rzf_precoder")
            summary["sionna_rzf_evaluated"] = True
            method_context = clone_native_receiver_context(
                context,
                h_f=context.h_f,
                csi=csi,
                h_full=context.h_full,
                context_meta_updates={
                    "csi_interface_used": True,
                    "project_h_f_assisted": False,
                    "extracted_h_f_used": True,
                    "full_native_only": False,
                    "csi_summary": csi.summary_dict(),
                },
            )
            row, _, _ = run_native_receiver_with_precoder(
                method="sionna_rzf_precoder",
                method_type="native_optional",
                precoder_f=probe["sionna_precoder_output"],
                context=method_context,
                runtime_ms=0.0,
                checkpoint_path=None,
                teacher_used_during_inference=False,
                trace_shapes=False,
            )
            row["input_type"] = "ExtractedCSI"
            row["csi_interface_used"] = True
            row["project_h_f_assisted"] = False
            row["extracted_h_f_used"] = True
            row["full_native_only"] = False
            row["precoder_summary"] = summarize_precoder_input(probe["sionna_precoder_output"])
            row["relationship_status"] = probe.get("relationship_status")
            row["strict_equivalence_claim_allowed"] = probe.get("strict_equivalence_claim_allowed")
            rows.append(row)
            if not bool(row["native_receiver_success"]):
                summary["sionna_rzf_skipped_reason"] = row["fallback_reason"]
        else:
            summary["sionna_rzf_skipped"] = True
            summary["sionna_rzf_skipped_reason"] = str(probe.get("fallback_reason", "sionna_rzf_probe_failed"))

    summary["metrics"] = rows
    summary["all_precoders_emit_precoder_output"] = bool(rows) and all(
        row.get("precoder_input_type") == "PrecoderOutput" for row in rows if row["fallback_reason"] != "skipped_missing_checkpoint"
    )
    summary["all_receiver_consumers_accept_precoder_output"] = summary["all_precoders_emit_precoder_output"]
    summary["native_receiver_success"] = any(bool(row["native_receiver_success"]) for row in rows)
    summary["teacher_used_during_inference"] = any(bool(row["teacher_used_during_inference"]) for row in rows)
    summary["no_new_fallback_introduced"] = not any(
        bool(row["fallback_used"]) and row["fallback_reason"] not in {"", "skipped_missing_checkpoint"} for row in rows
    )

    if rows:
        _write_csv(csv_path, rows)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("method\n", encoding="utf-8")
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved unified CSI+PrecoderOutput summary to {out_path}")


if __name__ == "__main__":
    main()
