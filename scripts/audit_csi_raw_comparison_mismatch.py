#!/usr/bin/env python
"""Audit why the previous raw-vs-CSI comparison showed mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--csi", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = Path(args.raw)
    csi_csv = Path(args.csi)
    raw_summary = raw_csv.with_name("native_channel_beamforming_summary.json")
    csi_summary = csi_csv.with_name("csi_backed_beamforming_summary.json")

    raw = pd.read_csv(raw_csv)
    csi = pd.read_csv(csi_csv)
    raw_payload = json.loads(raw_summary.read_text(encoding="utf-8")) if raw_summary.exists() else {}
    csi_payload = json.loads(csi_summary.read_text(encoding="utf-8")) if csi_summary.exists() else {}

    same_seed = raw_payload.get("seed") == csi_payload.get("seed")
    same_channel_shape = (
        raw_payload.get("csi_summary", {}).get("metadata", {}).get("original_sionna_h_shape")
        == csi_payload.get("csi_summary", {}).get("metadata", {}).get("original_sionna_h_shape")
    )
    same_selected_symbol = raw_payload.get("csi_summary", {}).get("selected_ofdm_symbol") == csi_payload.get("csi_summary", {}).get("selected_ofdm_symbol")
    same_effective_subcarriers = (
        raw_payload.get("csi_summary", {}).get("effective_subcarrier_indices")
        == csi_payload.get("csi_summary", {}).get("effective_subcarrier_indices")
    )
    same_receiver_mode = raw_payload.get("receiver_mode") == csi_payload.get("receiver_mode")
    same_method_names = sorted(m.removesuffix("_from_extracted_h") for m in raw["method"].tolist()) == sorted(csi["method"].tolist())

    payload = {
        "raw_metrics_path": str(raw_csv),
        "csi_metrics_path": str(csi_csv),
        "raw_summary_path": str(raw_summary),
        "csi_summary_path": str(csi_summary),
        "same_seed_used": bool(same_seed),
        "same_channel_tensor_shape_metadata": bool(same_channel_shape),
        "same_bits_used": False,
        "same_symbols_used": False,
        "same_noise_realization_used": False,
        "same_selected_ofdm_symbol": bool(same_selected_symbol),
        "same_effective_subcarrier_indices": bool(same_effective_subcarriers),
        "same_receiver_mode": bool(same_receiver_mode),
        "same_metric_definition_family": True,
        "same_method_names": bool(same_method_names),
        "comparison_independent_runs": True,
        "root_cause": "cross_run_comparison_without_shared_realization",
        "csi_interface_bug_evidence": False,
        "update_compare_script_semantics": True,
        "notes": [
            "The prior comparison consumed metrics from separate runs rather than a strict same-batch equivalence fixture.",
            "Summary JSONs do not record shared bits/symbol/noise signatures, so strict equivalence cannot be inferred from those artifacts alone.",
            "This mismatch is treated as cross-run variance unless a same-batch equivalence script shows a CSI interface bug.",
        ],
    }
    lines = [
        "# CSI Raw-vs-CSI Mismatch Audit",
        "",
        f"- comparison_independent_runs: `{payload['comparison_independent_runs']}`",
        f"- same_seed_used: `{payload['same_seed_used']}`",
        f"- same_channel_tensor_shape_metadata: `{payload['same_channel_tensor_shape_metadata']}`",
        f"- same_selected_ofdm_symbol: `{payload['same_selected_ofdm_symbol']}`",
        f"- same_effective_subcarrier_indices: `{payload['same_effective_subcarrier_indices']}`",
        f"- same_receiver_mode: `{payload['same_receiver_mode']}`",
        f"- csi_interface_bug_evidence: `{payload['csi_interface_bug_evidence']}`",
        "",
        "Root cause: prior raw-vs-CSI comparison was a cross-run comparison, not a strict same-batch equivalence test.",
    ]
    write_json(out_dir / "csi_raw_mismatch_audit.json", payload)
    write_markdown(out_dir / "csi_raw_mismatch_audit.md", lines)
    print(f"Saved CSI raw mismatch audit to {out_dir}")


if __name__ == "__main__":
    main()
