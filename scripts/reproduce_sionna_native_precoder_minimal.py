#!/usr/bin/env python
"""Run a very small Sionna native precoder bridge reproduction check."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_ofdm_experiment_helpers import load_json
from beamforming.utils.sionna_native_learned_beamforming import build_native_receiver_context
from beamforming.utils.sionna_precoder_api_bridge import evaluate_project_vs_sionna_rzf_same_realization


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
    if not env["sionna_import_ok"]:
        payload = {
            "status": "skipped",
            "sionna_import_ok": False,
            "sionna_rzf_available": False,
            "sionna_rzf_callable": False,
            "converted_to_precoder_output": False,
            "native_receiver_success": False,
            "sionna_native_precoder": False,
            "project_side_precoder": True,
            "relationship_status": "not_evaluated",
            "strict_equivalence_claim_allowed": False,
            "full_native_only": False,
            "reason": "Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`.",
            "sionna_rt_used": False,
            "ray_tracing_used": False,
            "fiveg_full_stack_used": False,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved minimal Sionna native precoder summary to {out_path}")
        return

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "outputs/repro/debug/sionna_native_precoder_minimal"
    base_dir.mkdir(parents=True, exist_ok=True)
    same_realization_out = base_dir / "sionna_rzf_same_realization.json"

    _run(["scripts/validate_sionna_rzf_same_realization.py", "--out", str(same_realization_out), "--seed", "0"], cwd=repo_root)
    same_realization = load_json(same_realization_out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = build_native_receiver_context(
        batch_size=4,
        num_subcarriers=8,
        num_users=4,
        num_bs_ant=8,
        snr_db=10.0,
        device=device,
    )
    evaluation = evaluate_project_vs_sionna_rzf_same_realization(context=context, device=device)

    payload = {
        "status": "ok",
        "sionna_import_ok": True,
        "sionna_version": env["sionna_version"],
        "sionna_rzf_available": bool(evaluation.get("sionna_rzf_available", False)),
        "sionna_rzf_callable": bool(evaluation.get("sionna_rzf_callable", False)),
        "converted_to_precoder_output": bool(evaluation.get("converted_to_precoder_output", False)),
        "native_receiver_success": bool(evaluation.get("native_receiver_success_sionna", False)),
        "sionna_native_precoder": bool(evaluation.get("converted_to_precoder_output", False)),
        "project_side_precoder": False if evaluation.get("converted_to_precoder_output", False) else True,
        "relationship_status": evaluation.get("relationship_status"),
        "strict_equivalence_claim_allowed": bool(evaluation.get("strict_equivalence_claim_allowed", False)),
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
        "notes": [
            "This is a reviewer-oriented minimal reproduction check for the optional Sionna native precoder bridge.",
            "It does not download data and does not train any model.",
            "The supported interpretation remains optional native method bridge, not full native-only and not strict project_rzf equivalence.",
        ],
        "same_realization_reference": {
            "relationship_status": same_realization.get("relationship_status"),
            "strict_equivalence_claim_allowed": same_realization.get("strict_equivalence_claim_allowed"),
            "semantic_compatibility_passed": same_realization.get("semantic_compatibility_passed"),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved minimal Sionna native precoder summary to {out_path}")


if __name__ == "__main__":
    main()
