#!/usr/bin/env python
"""Compare project RZF with the optional Sionna RZF native bridge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--same-realization", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--unified", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    same_realization = _load_json(Path(args.same_realization))
    alignment_df = pd.read_csv(args.alignment)
    unified_df = pd.read_csv(args.unified) if Path(args.unified).exists() and Path(args.unified).stat().st_size > 0 else pd.DataFrame()

    sionna_rows = unified_df[unified_df["method"] == "sionna_rzf_precoder"] if "method" in unified_df.columns else pd.DataFrame()
    sionna_optional_in_demo = not sionna_rows.empty
    native_receiver_success_in_demo = bool(sionna_rows["native_receiver_success"].all()) if sionna_optional_in_demo else False
    payload: dict[str, Any] = {
        "sionna_rzf_callable": bool(same_realization.get("sionna_rzf_callable", False)),
        "converted_to_precoder_output": bool(same_realization.get("converted_to_precoder_output", False)),
        "native_receiver_success_same_realization": bool(same_realization.get("native_receiver_success_sionna", False)),
        "native_receiver_success_in_demo": native_receiver_success_in_demo,
        "semantic_compatibility_passed": bool(same_realization.get("semantic_compatibility_passed", False)),
        "strict_equivalence_claim_allowed": bool(same_realization.get("strict_equivalence_claim_allowed", False)),
        "relationship_status": same_realization.get("relationship_status", "unknown"),
        "power_norm_project": same_realization.get("power_norm_project"),
        "power_norm_sionna": same_realization.get("power_norm_sionna"),
        "max_abs_diff_f_f_if_comparable": same_realization.get("max_abs_diff_f_f_if_comparable"),
        "mean_abs_diff_sum_rate": float(alignment_df["abs_diff_sum_rate"].mean()) if "abs_diff_sum_rate" in alignment_df.columns else None,
        "mean_abs_diff_symbol_mse": float(alignment_df["abs_diff_symbol_mse"].mean()) if "abs_diff_symbol_mse" in alignment_df.columns else None,
        "mean_abs_diff_sinr_db": float(alignment_df["abs_diff_sinr_db"].mean()) if "abs_diff_sinr_db" in alignment_df.columns else None,
        "callable_all_runs": bool(alignment_df["sionna_rzf_callable"].all()) if "sionna_rzf_callable" in alignment_df.columns else False,
        "converted_all_runs": bool(alignment_df["converted_to_precoder_output"].all()) if "converted_to_precoder_output" in alignment_df.columns else False,
        "native_receiver_success_all_runs": bool(alignment_df["native_receiver_success_sionna"].all()) if "native_receiver_success_sionna" in alignment_df.columns else False,
        "sionna_optional_method_in_unified_demo": sionna_optional_in_demo,
        "sionna_native_precoder_true_now": bool(same_realization.get("converted_to_precoder_output", False)),
        "project_rzf_strict_equivalence_allowed": bool(same_realization.get("strict_equivalence_claim_allowed", False)),
        "full_native_only": False,
        "recommended_next_step": (
            "release_hardening"
            if bool(same_realization.get("semantic_compatibility_passed", False))
            else "continue_adapter_debugging"
        ),
    }

    comparison_df = pd.DataFrame(
        [
            {
                "method": "project_rzf_vs_sionna_rzf_precoder",
                "sionna_rzf_callable": payload["sionna_rzf_callable"],
                "converted_to_precoder_output": payload["converted_to_precoder_output"],
                "native_receiver_success_same_realization": payload["native_receiver_success_same_realization"],
                "native_receiver_success_in_demo": payload["native_receiver_success_in_demo"],
                "semantic_compatibility_passed": payload["semantic_compatibility_passed"],
                "strict_equivalence_claim_allowed": payload["strict_equivalence_claim_allowed"],
                "relationship_status": payload["relationship_status"],
                "power_norm_project": payload["power_norm_project"],
                "power_norm_sionna": payload["power_norm_sionna"],
                "max_abs_diff_f_f_if_comparable": payload["max_abs_diff_f_f_if_comparable"],
                "mean_abs_diff_sum_rate": payload["mean_abs_diff_sum_rate"],
                "mean_abs_diff_symbol_mse": payload["mean_abs_diff_symbol_mse"],
                "mean_abs_diff_sinr_db": payload["mean_abs_diff_sinr_db"],
                "sionna_native_precoder_true_now": payload["sionna_native_precoder_true_now"],
                "project_rzf_strict_equivalence_allowed": payload["project_rzf_strict_equivalence_allowed"],
                "full_native_only": False,
                "recommended_next_step": payload["recommended_next_step"],
            }
        ]
    )
    comparison_csv = out_dir / "project_vs_sionna_precoder_comparison_v2.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    lines = [
        "# Project vs Sionna Native Precoder Comparison v2",
        "",
        f"1. Sionna RZFPrecoder callable: `{payload['sionna_rzf_callable']}`",
        f"2. converted to PrecoderOutput: `{payload['converted_to_precoder_output']}`",
        f"3. enters native receiver path: `same_realization={payload['native_receiver_success_same_realization']}`, `unified_demo={payload['native_receiver_success_in_demo']}`",
        f"4. semantic compatibility: `{payload['semantic_compatibility_passed']}` with relationship `{payload['relationship_status']}`",
        f"5. sionna_native_precoder=true allowed: `{payload['sionna_native_precoder_true_now']}`",
        f"6. project_rzf strict equivalence allowed: `{payload['project_rzf_strict_equivalence_allowed']}`",
        "7. full native-only benchmark: `False`",
        f"8. recommended next step: `{payload['recommended_next_step']}`",
        "",
        "Current interpretation:",
        "- same-realization validation is the only valid place to judge strict numerical equivalence",
        "- quick alignment sweep is used to judge stability of the semantic gap across seeds and SNRs",
        "- if semantic_compatibility_passed=true, `sionna_rzf_precoder` can be kept as an optional native method behind explicit Sionna availability checks",
        "- this still does not justify a full native-only benchmark claim",
    ]
    write_markdown(out_dir / "project_vs_sionna_precoder_comparison_v2.md", lines)
    print(f"Saved project-vs-Sionna precoder comparison to {out_dir}")


if __name__ == "__main__":
    main()
