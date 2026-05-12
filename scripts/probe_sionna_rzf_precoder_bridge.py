#!/usr/bin/env python
"""Probe Sionna RZFPrecoder against the current ExtractedCSI/PrecoderOutput bridge."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.precoder_interface import summarize_precoder_input
from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import build_native_receiver_context, clone_native_receiver_context, run_native_receiver_with_precoder
from beamforming.utils.sionna_precoder_api_bridge import run_sionna_rzf_precoder_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md(summary: dict[str, Any]) -> list[str]:
    return [
        "# Sionna RZFPrecoder Bridge Probe",
        "",
        f"- sionna_rzf_available: `{summary['sionna_rzf_available']}`",
        f"- sionna_rzf_callable: `{summary['sionna_rzf_callable']}`",
        f"- extracted_csi_used: `{summary['extracted_csi_used']}`",
        f"- sionna_precoder_success: `{summary['sionna_precoder_success']}`",
        f"- converted_to_precoder_output: `{summary['converted_to_precoder_output']}`",
        f"- shape_compatible: `{summary['shape_compatible']}`",
        f"- native_receiver_success_if_attempted: `{summary['native_receiver_success_if_attempted']}`",
        f"- fallback_used: `{summary['fallback_used']}`",
        f"- fallback_reason: `{summary['fallback_reason']}`",
        f"- recommended_next_step: `{summary['recommended_next_step']}`",
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_name("rzf_precoder_probe_metrics.csv")
    env = collect_sionna_env_info()
    summary: dict[str, Any] = {
        "status": "skipped",
        "seed": int(args.seed),
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "sionna_rzf_available": False,
        "sionna_rzf_callable": False,
        "extracted_csi_used": False,
        "sionna_precoder_success": False,
        "sionna_output_shape": None,
        "converted_to_precoder_output": False,
        "project_rzf_output_shape": None,
        "shape_compatible": False,
        "power_norm_project": None,
        "power_norm_sionna": None,
        "max_abs_diff_if_comparable": None,
        "native_receiver_success_if_attempted": False,
        "fallback_used": True,
        "fallback_reason": "",
        "probe_only": True,
        "recommended_next_step": "keep_project_side_precoder_output",
        "rows": [],
    }
    if not env["sionna_import_ok"]:
        summary["fallback_reason"] = "sionna_not_installed"
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved Sionna RZFPrecoder probe summary to {out_path}")
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = build_native_receiver_context(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
    )
    csi = context.csi
    if csi is None:
        summary["status"] = "failed"
        summary["fallback_reason"] = "failed_to_create_extracted_csi"
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved Sionna RZFPrecoder probe summary to {out_path}")
        return

    probe = run_sionna_rzf_precoder_probe(
        csi,
        project_noise_var=context.noise_var,
        device=device,
    )
    summary.update(
        {
            "status": "ok" if probe.get("sionna_precoder_success") else "failed",
            "sionna_rzf_available": bool(probe.get("sionna_rzf_available")),
            "sionna_rzf_callable": bool(probe.get("sionna_rzf_callable")),
            "extracted_csi_used": True,
            "sionna_precoder_success": bool(probe.get("sionna_precoder_success")),
            "sionna_output_shape": probe.get("sionna_output_shape"),
            "converted_to_precoder_output": bool(probe.get("converted_to_precoder_output")),
            "fallback_used": bool(probe.get("fallback_used", True)),
            "fallback_reason": str(probe.get("fallback_reason", "")),
            "probe_only": bool(probe.get("probe_only", True)),
            "recommended_next_step": str(probe.get("recommended_next_step", "keep_project_side_precoder_output")),
        }
    )

    comparison = probe.get("comparison") or {}
    summary["project_rzf_output_shape"] = comparison.get("project_shape")
    summary["shape_compatible"] = bool(comparison.get("shape_compatible", False))
    summary["power_norm_project"] = (comparison.get("project_power_norm") or {}).get("mean") if comparison.get("project_power_norm") else None
    sionna_summary = comparison.get("sionna_summary") or {}
    summary["power_norm_sionna"] = (sionna_summary.get("power_norm") or {}).get("mean") if isinstance(sionna_summary.get("power_norm"), dict) else None
    summary["max_abs_diff_if_comparable"] = comparison.get("max_abs_diff")

    rows: list[dict[str, Any]] = []
    if probe.get("converted_to_precoder_output") and probe.get("sionna_precoder_output") is not None:
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
            method_type="native_probe",
            precoder_f=probe["sionna_precoder_output"],
            context=method_context,
            runtime_ms=0.0,
            checkpoint_path=None,
            teacher_used_during_inference=False,
            trace_shapes=False,
        )
        row["precoder_summary"] = summarize_precoder_input(probe["sionna_precoder_output"])
        rows.append(row)
        summary["native_receiver_success_if_attempted"] = bool(row["native_receiver_success"])

    summary["rows"] = rows
    if rows:
        _write_csv(csv_path, rows)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("method\n", encoding="utf-8")
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved Sionna RZFPrecoder probe summary to {out_path}")


if __name__ == "__main__":
    main()
