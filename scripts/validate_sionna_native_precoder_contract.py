#!/usr/bin/env python
"""Validate the current Sionna native precoder contract and skip/fallback semantics."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import build_native_receiver_context
from beamforming.utils.sionna_precoder_api_bridge import build_sionna_rzf_precoder_contract, run_sionna_rzf_precoder_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _md(summary: dict[str, object]) -> list[str]:
    return [
        "# Sionna Native Precoder Contract Validation",
        "",
        f"- sionna_import_ok: `{summary['sionna_import_ok']}`",
        f"- contract_valid: `{summary['contract_valid']}`",
        f"- sionna_rzf_available: `{summary['sionna_rzf_available']}`",
        f"- sionna_rzf_callable: `{summary['sionna_rzf_callable']}`",
        f"- converted_to_precoder_output: `{summary['converted_to_precoder_output']}`",
        f"- relationship_status: `{summary['relationship_status']}`",
        f"- strict_equivalence_claim_allowed: `{summary['strict_equivalence_claim_allowed']}`",
        f"- full_native_only: `{summary['full_native_only']}`",
        f"- fallback_used: `{summary['fallback_used']}`",
        f"- fallback_reason: `{summary['fallback_reason']}`",
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    summary: dict[str, object] = {
        "status": "skipped",
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "contract_valid": False,
        "sionna_rzf_available": False,
        "sionna_rzf_callable": False,
        "converted_to_precoder_output": False,
        "relationship_status": "not_evaluated",
        "strict_equivalence_claim_allowed": False,
        "semantic_compatibility_passed": False,
        "sionna_native_precoder": False,
        "project_side_precoder": False,
        "full_native_only": False,
        "fallback_used": True,
        "fallback_reason": "sionna_not_installed",
        "contract": None,
    }
    if not env["sionna_import_ok"]:
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved native precoder contract validation to {out_path}")
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
        print(f"Saved native precoder contract validation to {out_path}")
        return

    probe = run_sionna_rzf_precoder_probe(context.csi, project_noise_var=context.noise_var, device=device)
    contract_payload = probe.get("contract") or build_sionna_rzf_precoder_contract(
        sionna_version=env["sionna_version"],
        callable=False,
        converted_precoder_output_shape=None,
        relationship_to_project_rzf="not_evaluated",
        strict_equivalence_claim_allowed=False,
        semantic_compatibility_passed=False,
        project_side_precoder=False,
        sionna_native_precoder=False,
        full_native_only=False,
    ).summary_dict()

    summary.update(
        {
            "status": "ok",
            "contract_valid": bool(contract_payload.get("validation", {}).get("valid", False)),
            "sionna_rzf_available": bool(probe.get("sionna_rzf_available", False)),
            "sionna_rzf_callable": bool(probe.get("sionna_rzf_callable", False)),
            "converted_to_precoder_output": bool(probe.get("converted_to_precoder_output", False)),
            "relationship_status": contract_payload.get("relationship_to_project_rzf"),
            "strict_equivalence_claim_allowed": bool(contract_payload.get("strict_equivalence_claim_allowed", False)),
            "semantic_compatibility_passed": bool(contract_payload.get("semantic_compatibility_passed", False)),
            "sionna_native_precoder": bool(contract_payload.get("sionna_native_precoder", False)),
            "project_side_precoder": bool(contract_payload.get("project_side_precoder", False)),
            "full_native_only": bool(contract_payload.get("full_native_only", False)),
            "fallback_used": bool(probe.get("fallback_used", True)),
            "fallback_reason": str(probe.get("fallback_reason", "")),
            "contract": contract_payload,
        }
    )
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved native precoder contract validation to {out_path}")


if __name__ == "__main__":
    main()
