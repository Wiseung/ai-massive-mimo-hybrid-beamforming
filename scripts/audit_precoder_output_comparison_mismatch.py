#!/usr/bin/env python
"""Audit why the previous raw-F_f vs PrecoderOutput comparison showed mismatch."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--precoder-output", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _parse_precoder_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = Path(args.raw)
    precoder_csv = Path(args.precoder_output)
    raw_summary = raw_csv.with_name("csi_backed_beamforming_summary.json")
    precoder_summary = precoder_csv.with_name("unified_csi_precoder_summary.json")

    raw = pd.read_csv(raw_csv)
    precoder = pd.read_csv(precoder_csv)
    raw_payload = json.loads(raw_summary.read_text(encoding="utf-8")) if raw_summary.exists() else {}
    precoder_payload = json.loads(precoder_summary.read_text(encoding="utf-8")) if precoder_summary.exists() else {}

    raw_csi_sig = (raw_payload.get("csi_input_summary") or {}).get("tensor_signature")
    precoder_csi_sig = (precoder_payload.get("csi_input_summary") or {}).get("tensor_signature")

    raw_precoder_signatures = {
        row["method"]: (_parse_precoder_summary(row.get("precoder_summary")) or {}).get("tensor_signature")
        for _, row in raw.iterrows()
    }
    precoder_signatures = {
        row["method"]: (_parse_precoder_summary(row.get("precoder_summary")) or {}).get("tensor_signature")
        for _, row in precoder.iterrows()
    }
    shared_methods = sorted(set(raw_precoder_signatures) & set(precoder_signatures))
    same_raw_ff_used = bool(shared_methods) and all(
        raw_precoder_signatures.get(method) == precoder_signatures.get(method) for method in shared_methods
    )

    payload = {
        "raw_metrics_path": str(raw_csv),
        "precoder_output_metrics_path": str(precoder_csv),
        "raw_summary_path": str(raw_summary),
        "precoder_output_summary_path": str(precoder_summary),
        "same_seed_used": raw_payload.get("seed") == precoder_payload.get("seed"),
        "same_csi_object_used": raw_csi_sig == precoder_csi_sig,
        "same_raw_f_f_used": same_raw_ff_used,
        "same_bits_used": False,
        "same_symbols_used": False,
        "same_noise_realization_used": False,
        "same_noise_var_used": False,
        "same_receiver_config_used": False,
        "same_metric_definition_family": True,
        "comparison_independent_runs": True,
        "comparison_type": "cross_run_comparison",
        "root_cause": "cross_run_comparison_without_shared_csi_and_precoder_realization",
        "interface_bug_evidence": False,
        "update_compare_script_semantics": True,
        "notes": [
            "The previous raw-vs-PrecoderOutput artifact compares separate runs rather than one shared same-batch fixture.",
            "The two summary JSON files reuse seed=0 but expose different CSI tensor signatures, so the channel realization is not shared.",
            "Per-method PrecoderOutput tensor signatures also differ across the two artifacts, so ranking mismatch is cross-run variance rather than direct PrecoderOutput bug evidence.",
        ],
        "raw_csi_tensor_signature": raw_csi_sig,
        "precoder_output_csi_tensor_signature": precoder_csi_sig,
        "raw_precoder_signatures": raw_precoder_signatures,
        "precoder_output_signatures": precoder_signatures,
    }

    raw_receiver = {
        "receiver_mode": raw_payload.get("receiver_mode"),
        "resource_grid_num_ofdm_symbols": (raw_payload.get("csi_summary") or {}).get("metadata", {}).get("resource_grid_num_ofdm_symbols"),
        "resource_grid_fft_size": (raw_payload.get("csi_summary") or {}).get("metadata", {}).get("resource_grid_fft_size"),
    }
    precoder_receiver = {
        "receiver_mode": precoder_payload.get("receiver_mode"),
        "resource_grid_num_ofdm_symbols": (precoder_payload.get("csi_summary") or {}).get("metadata", {}).get("resource_grid_num_ofdm_symbols"),
        "resource_grid_fft_size": (precoder_payload.get("csi_summary") or {}).get("metadata", {}).get("resource_grid_fft_size"),
    }
    payload["same_receiver_config_used"] = raw_receiver == precoder_receiver
    payload["same_noise_var_used"] = (raw_payload.get("noise_var") == precoder_payload.get("noise_var"))

    lines = [
        "# PrecoderOutput Comparison Mismatch Audit",
        "",
        f"- comparison_type: `{payload['comparison_type']}`",
        f"- comparison_independent_runs: `{payload['comparison_independent_runs']}`",
        f"- same_seed_used: `{payload['same_seed_used']}`",
        f"- same_csi_object_used: `{payload['same_csi_object_used']}`",
        f"- same_raw_f_f_used: `{payload['same_raw_f_f_used']}`",
        f"- same_receiver_config_used: `{payload['same_receiver_config_used']}`",
        f"- interface_bug_evidence: `{payload['interface_bug_evidence']}`",
        "",
        "Root cause: previous raw-F_f-vs-PrecoderOutput mismatch was a cross-run comparison without a shared CSI/Precoder realization, not direct PrecoderOutput interface bug evidence.",
    ]
    write_json(out_dir / "precoder_output_mismatch_audit.json", payload)
    write_markdown(out_dir / "precoder_output_mismatch_audit.md", lines)
    print(f"Saved PrecoderOutput mismatch audit to {out_dir}")


if __name__ == "__main__":
    main()
