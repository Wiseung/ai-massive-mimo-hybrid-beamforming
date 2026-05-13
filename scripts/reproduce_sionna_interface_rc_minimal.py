#!/usr/bin/env python
"""Minimal end-to-end reproduction for the interface-first Sionna bridge RC."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.precoder_interface import build_precoder_output
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import compute_project_precoder_per_subcarrier
from beamforming.utils.sionna_native_learned_beamforming import build_native_receiver_context, clone_native_receiver_context, run_native_receiver_with_precoder
from beamforming.utils.sionna_ofdm_experiment_helpers import load_json
from beamforming.utils.sionna_precoder_api_bridge import run_sionna_rzf_precoder_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=str(cwd))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = collect_sionna_env_info()
    repo_root = Path(__file__).resolve().parents[1]
    debug_dir = repo_root / "outputs/repro/debug/sionna_interface_rc_minimal"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not env["sionna_import_ok"]:
        payload = {
            "status": "skipped",
            "sionna_import_ok": False,
            "extracted_csi_created": False,
            "csi_validation_passed": False,
            "project_precoder_output_created": False,
            "sionna_rzf_available": False,
            "sionna_rzf_callable": False,
            "sionna_precoder_output_created": False,
            "native_receiver_success": False,
            "contract_valid": False,
            "contract_matrix_passed": False,
            "relationship_status": "not_evaluated",
            "strict_equivalence_claim_allowed": False,
            "full_native_only": False,
            "sionna_rt_used": False,
            "ray_tracing_used": False,
            "fiveg_full_stack_used": False,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved interface RC minimal summary to {out_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = build_native_receiver_context(
        batch_size=4,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
    )
    csi = context.csi
    if csi is None:
        payload = {
            "status": "failed",
            "sionna_import_ok": True,
            "extracted_csi_created": False,
            "csi_validation_passed": False,
            "project_precoder_output_created": False,
            "sionna_rzf_available": False,
            "sionna_rzf_callable": False,
            "sionna_precoder_output_created": False,
            "native_receiver_success": False,
            "contract_valid": False,
            "contract_matrix_passed": False,
            "relationship_status": "not_evaluated",
            "strict_equivalence_claim_allowed": False,
            "full_native_only": False,
            "sionna_rt_used": False,
            "ray_tracing_used": False,
            "fiveg_full_stack_used": False,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved interface RC minimal summary to {out_path}")
        return

    project_precoder_output = compute_project_precoder_per_subcarrier("rzf", csi, context.noise_var, return_precoder_output=True)
    method_context = clone_native_receiver_context(
        context,
        h_f=context.h_f,
        csi=csi,
        h_full=context.h_full,
        context_meta_updates={"csi_summary": csi.summary_dict()},
    )
    project_row, _, _ = run_native_receiver_with_precoder(
        method="project_rzf",
        method_type="analytic",
        precoder_f=project_precoder_output,
        context=method_context,
        runtime_ms=0.0,
        checkpoint_path=None,
        teacher_used_during_inference=False,
        trace_shapes=False,
    )

    probe = run_sionna_rzf_precoder_probe(csi, project_noise_var=context.noise_var, device=device)

    contract_validation_out = debug_dir / "native_precoder_contract_validation.json"
    contract_matrix_out = debug_dir / "native_precoder_contract_matrix.json"
    _run(["scripts/validate_sionna_native_precoder_contract.py", "--out", str(contract_validation_out)], cwd=repo_root)
    _run(["scripts/test_sionna_native_precoder_contract_matrix.py", "--out", str(contract_matrix_out)], cwd=repo_root)
    contract_validation = load_json(contract_validation_out)
    contract_matrix = load_json(contract_matrix_out)

    payload = {
        "status": "ok",
        "sionna_import_ok": True,
        "sionna_version": env["sionna_version"],
        "extracted_csi_created": True,
        "csi_validation_passed": bool(csi.summary_dict().get("validation", {}).get("valid", True)),
        "project_precoder_output_created": True,
        "sionna_rzf_available": bool(probe.get("sionna_rzf_available", False)),
        "sionna_rzf_callable": bool(probe.get("sionna_rzf_callable", False)),
        "sionna_precoder_output_created": bool(probe.get("converted_to_precoder_output", False)),
        "native_receiver_success": bool(project_row.get("native_receiver_success", False))
        and bool(probe.get("native_receiver_success_if_attempted", False) or probe.get("converted_to_precoder_output", False)),
        "contract_valid": bool(contract_validation.get("contract_valid", False)),
        "contract_matrix_passed": bool(contract_matrix.get("all_scenarios_contract_compliant", False)),
        "relationship_status": str(probe.get("relationship_status", "not_evaluated")),
        "strict_equivalence_claim_allowed": bool(probe.get("strict_equivalence_claim_allowed", False)),
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved interface RC minimal summary to {out_path}")


if __name__ == "__main__":
    main()
