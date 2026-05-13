#!/usr/bin/env python
"""Exercise contract, skip, and fallback semantics for the Sionna native precoder bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import build_native_receiver_context, clone_native_receiver_context, run_native_receiver_with_precoder
from beamforming.utils.sionna_precoder_api_bridge import run_sionna_rzf_precoder_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _md(payload: dict[str, object]) -> list[str]:
    return [
        "# Sionna Native Precoder Contract Matrix",
        "",
        f"- all_scenarios_contract_compliant: `{payload['all_scenarios_contract_compliant']}`",
        f"- aliasing_project_rzf_detected: `{payload['aliasing_project_rzf_detected']}`",
        f"- all_failure_scenarios_have_reason: `{payload['all_failure_scenarios_have_reason']}`",
        f"- successful_scenario_relationship_status: `{payload['successful_scenario_relationship_status']}`",
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    payload: dict[str, object] = {
        "status": "skipped",
        "sionna_import_ok": env["sionna_import_ok"],
        "scenarios": [],
        "all_scenarios_contract_compliant": False,
        "aliasing_project_rzf_detected": False,
        "all_failure_scenarios_have_reason": False,
        "successful_scenario_relationship_status": "not_evaluated",
    }
    if not env["sionna_import_ok"]:
        write_json(out_path, payload)
        write_markdown(md_path, _md(payload))
        print(f"Saved native precoder contract matrix to {out_path}")
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
    if context.csi is None:
        payload["status"] = "failed"
        write_json(out_path, payload)
        write_markdown(md_path, _md(payload))
        print(f"Saved native precoder contract matrix to {out_path}")
        return

    scenarios = [
        ("normal_available", {}),
        ("force_missing_sionna", {"force_missing_sionna": True}),
        ("force_rzf_unavailable", {"force_rzf_unavailable": True}),
        ("force_adapter_failure", {"force_adapter_failure": True}),
        ("force_receiver_failure", {}),
    ]
    rows = []
    for name, kwargs in scenarios:
        probe = run_sionna_rzf_precoder_probe(context.csi, project_noise_var=context.noise_var, device=device, **kwargs)
        native_receiver_success = False
        fallback_reason = str(probe.get("fallback_reason", ""))
        if name == "force_receiver_failure" and probe.get("converted_to_precoder_output") and probe.get("sionna_precoder_output") is not None:
            bad_context = clone_native_receiver_context(
                context,
                h_f=context.h_f,
                csi=context.csi,
                h_full=context.h_full,
                context_meta_updates={"shared_rx_noise_grid": None, "force_precoder_matrix_receiver_failure": True},
            )
            row, _, _ = run_native_receiver_with_precoder(
                method="sionna_rzf_precoder",
                method_type="native_contract_matrix",
                precoder_f=probe["sionna_precoder_output"],
                context=bad_context,
                runtime_ms=0.0,
                checkpoint_path=None,
                teacher_used_during_inference=False,
                trace_shapes=False,
            )
            native_receiver_success = bool(row["native_receiver_success"])
            if not native_receiver_success:
                fallback_reason = row["fallback_reason"] or "forced_receiver_failure"
            else:
                native_receiver_success = False
                fallback_reason = "forced_receiver_failure_not_triggered"
        rows.append(
            {
                "scenario": name,
                "sionna_rzf_available": bool(probe.get("sionna_rzf_available", False)),
                "sionna_rzf_callable": bool(probe.get("sionna_rzf_callable", False)),
                "converted_to_precoder_output": bool(probe.get("converted_to_precoder_output", False)),
                "sionna_rzf_skipped": bool(probe.get("sionna_rzf_skipped", False)),
                "fallback_used": bool(probe.get("fallback_used", True)),
                "fallback_reason": fallback_reason,
                "relationship_status": probe.get("relationship_status", "not_evaluated"),
                "strict_equivalence_claim_allowed": bool(probe.get("strict_equivalence_claim_allowed", False)),
                "project_rzf_aliasing_detected": probe.get("precoder_source") == "project_rzf",
                "contract_valid": bool((probe.get("contract") or {}).get("validation", {}).get("valid", False)),
                "native_receiver_success": native_receiver_success if name == "force_receiver_failure" else bool(probe.get("converted_to_precoder_output", False)),
            }
        )

    payload.update(
        {
            "status": "ok",
            "scenarios": rows,
            "all_scenarios_contract_compliant": all(bool(row["contract_valid"]) for row in rows),
            "aliasing_project_rzf_detected": any(bool(row["project_rzf_aliasing_detected"]) for row in rows),
            "all_failure_scenarios_have_reason": all(
                (not row["fallback_used"]) or bool(row["fallback_reason"])
                for row in rows
            ),
            "successful_scenario_relationship_status": next(
                row["relationship_status"] for row in rows if row["scenario"] == "normal_available"
            ),
        }
    )
    write_json(out_path, payload)
    write_markdown(md_path, _md(payload))
    print(f"Saved native precoder contract matrix to {out_path}")


if __name__ == "__main__":
    main()
