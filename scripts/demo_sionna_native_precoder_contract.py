#!/usr/bin/env python
"""Run a contract-aware demo for the optional Sionna native precoder bridge."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.precoder_interface import summarize_precoder_input
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import build_native_receiver_context, clone_native_receiver_context, run_native_receiver_with_precoder
from beamforming.utils.sionna_precoder_api_bridge import run_sionna_rzf_precoder_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md(summary: dict[str, Any]) -> list[str]:
    return [
        "# Sionna Native Precoder Contract Demo",
        "",
        f"- contract_valid: `{summary['contract_valid']}`",
        f"- sionna_import_ok: `{summary['sionna_import_ok']}`",
        f"- sionna_rzf_available: `{summary['sionna_rzf_available']}`",
        f"- sionna_rzf_callable: `{summary['sionna_rzf_callable']}`",
        f"- adapter_success: `{summary['adapter_success']}`",
        f"- converted_to_precoder_output: `{summary['converted_to_precoder_output']}`",
        f"- native_receiver_success: `{summary['native_receiver_success']}`",
        f"- sionna_native_precoder: `{summary['sionna_native_precoder']}`",
        f"- project_side_precoder: `{summary['project_side_precoder']}`",
        f"- full_native_only: `{summary['full_native_only']}`",
        f"- relationship_status: `{summary['relationship_status']}`",
        f"- strict_equivalence_claim_allowed: `{summary['strict_equivalence_claim_allowed']}`",
        f"- skip_policy_exercised: `{summary['skip_policy_exercised']}`",
        f"- fallback_used: `{summary['fallback_used']}`",
        f"- fallback_reason: `{summary['fallback_reason']}`",
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_name("native_precoder_contract_demo_metrics.csv")
    env = collect_sionna_env_info()
    summary: dict[str, Any] = {
        "status": "skipped",
        "contract_valid": False,
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "sionna_rzf_available": False,
        "sionna_rzf_callable": False,
        "adapter_success": False,
        "converted_to_precoder_output": False,
        "native_receiver_success": False,
        "sionna_native_precoder": False,
        "project_side_precoder": False,
        "full_native_only": False,
        "relationship_status": "not_evaluated",
        "strict_equivalence_claim_allowed": False,
        "skip_policy_exercised": False,
        "fallback_used": True,
        "fallback_reason": "sionna_not_installed",
        "rows": [],
    }
    if not env["sionna_import_ok"]:
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("method\n", encoding="utf-8")
        print(f"Saved native precoder contract demo to {out_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = build_native_receiver_context(
        batch_size=8,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
    )
    if context.csi is None:
        summary["status"] = "failed"
        summary["fallback_reason"] = "failed_to_create_extracted_csi"
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("method\n", encoding="utf-8")
        print(f"Saved native precoder contract demo to {out_path}")
        return

    probe = run_sionna_rzf_precoder_probe(context.csi, project_noise_var=context.noise_var, device=device)
    contract_payload = probe.get("contract", {})
    rows: list[dict[str, Any]] = []
    if probe.get("converted_to_precoder_output") and probe.get("sionna_precoder_output") is not None:
        method_context = clone_native_receiver_context(
            context,
            h_f=context.h_f,
            csi=context.csi,
            h_full=context.h_full,
            context_meta_updates={"csi_summary": context.csi.summary_dict()},
        )
        row, _, _ = run_native_receiver_with_precoder(
            method="sionna_rzf_precoder",
            method_type="native_contract_demo",
            precoder_f=probe["sionna_precoder_output"],
            context=method_context,
            runtime_ms=0.0,
            checkpoint_path=None,
            teacher_used_during_inference=False,
            trace_shapes=False,
        )
        row["precoder_summary"] = summarize_precoder_input(probe["sionna_precoder_output"])
        rows.append(row)
        summary["native_receiver_success"] = bool(row["native_receiver_success"])

    summary.update(
        {
            "status": "ok",
            "contract_valid": bool(contract_payload.get("validation", {}).get("valid", False)),
            "sionna_rzf_available": bool(probe.get("sionna_rzf_available", False)),
            "sionna_rzf_callable": bool(probe.get("sionna_rzf_callable", False)),
            "adapter_success": bool(probe.get("shape_mapping", {}).get("success", False)),
            "converted_to_precoder_output": bool(probe.get("converted_to_precoder_output", False)),
            "sionna_native_precoder": bool(contract_payload.get("sionna_native_precoder", False)),
            "project_side_precoder": bool(contract_payload.get("project_side_precoder", False)),
            "full_native_only": bool(contract_payload.get("full_native_only", False)),
            "relationship_status": contract_payload.get("relationship_to_project_rzf", "not_evaluated"),
            "strict_equivalence_claim_allowed": bool(contract_payload.get("strict_equivalence_claim_allowed", False)),
            "skip_policy_exercised": bool(probe.get("sionna_rzf_skipped", False)),
            "fallback_used": bool(probe.get("fallback_used", True)),
            "fallback_reason": str(probe.get("fallback_reason", "")),
            "rows": rows,
        }
    )
    if rows:
        _write_csv(csv_path, rows)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("method\n", encoding="utf-8")
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved native precoder contract demo to {out_path}")


if __name__ == "__main__":
    main()
