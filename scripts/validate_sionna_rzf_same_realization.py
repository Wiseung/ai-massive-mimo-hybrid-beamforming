#!/usr/bin/env python
"""Validate project RZF and Sionna RZFPrecoder on one shared realization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import (
    build_native_receiver_context,
    generate_shared_sionna_channel_bundle,
)
from beamforming.utils.sionna_precoder_api_bridge import (
    STRICT_EQUIVALENCE_TOL,
    evaluate_project_vs_sionna_rzf_same_realization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=10.0)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md(summary: dict[str, Any]) -> list[str]:
    return [
        "# Sionna RZF Same-realization Validation",
        "",
        f"- comparison_type: `{summary['comparison_type']}`",
        f"- same_realization_comparison: `{summary['same_realization_comparison']}`",
        f"- same_csi_object_used: `{summary['same_csi_object_used']}`",
        f"- same_symbols_used: `{summary['same_symbols_used']}`",
        f"- same_receiver_config_used: `{summary['same_receiver_config_used']}`",
        f"- same_noise_config_used: `{summary['same_noise_config_used']}`",
        f"- converted_to_precoder_output: `{summary['converted_to_precoder_output']}`",
        f"- native_receiver_success_project: `{summary['native_receiver_success_project']}`",
        f"- native_receiver_success_sionna: `{summary['native_receiver_success_sionna']}`",
        f"- relationship_status: `{summary['relationship_status']}`",
        f"- semantic_compatibility_passed: `{summary['semantic_compatibility_passed']}`",
        f"- strict_equivalence_claim_allowed: `{summary['strict_equivalence_claim_allowed']}`",
        f"- suggested_for_optional_method_list: `{summary['suggested_for_optional_method_list']}`",
        "",
        f"- max_abs_diff_f_f_if_comparable: `{summary['max_abs_diff_f_f_if_comparable']}`",
        f"- abs_diff_sum_rate: `{summary['abs_diff_sum_rate']}`",
        f"- abs_diff_symbol_mse: `{summary['abs_diff_symbol_mse']}`",
        f"- abs_diff_sinr_db: `{summary['abs_diff_sinr_db']}`",
        f"- difference_primary_axis: `{summary['difference_primary_axis']}`",
        "",
        f"- conclusion: `{summary['conclusion']}`",
        f"- notes: `{summary['notes']}`",
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_suffix(".csv")
    env = collect_sionna_env_info()
    summary: dict[str, Any] = {
        "status": "skipped",
        "seed": int(args.seed),
        "snr_db": float(args.snr_db),
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "comparison_type": "same_realization_comparison",
        "same_realization_comparison": False,
        "same_csi_object_used": False,
        "same_symbols_used": False,
        "same_receiver_config_used": False,
        "same_noise_config_used": False,
        "sionna_rzf_available": False,
        "sionna_rzf_callable": False,
        "converted_to_precoder_output": False,
        "native_receiver_success_project": False,
        "native_receiver_success_sionna": False,
        "relationship_status": "incompatible",
        "semantic_compatibility_passed": False,
        "strict_equivalence_claim_allowed": False,
        "suggested_for_optional_method_list": False,
        "project_f_f_shape": None,
        "sionna_precoder_output_shape": None,
        "converted_precoder_output_shape": None,
        "power_norm_project": None,
        "power_norm_sionna": None,
        "power_norm_gap": None,
        "max_abs_diff_f_f_if_comparable": None,
        "project_sum_rate": None,
        "sionna_sum_rate": None,
        "abs_diff_sum_rate": None,
        "rel_diff_sum_rate": None,
        "project_symbol_mse": None,
        "sionna_symbol_mse": None,
        "abs_diff_symbol_mse": None,
        "project_sinr_db": None,
        "sionna_sinr_db": None,
        "abs_diff_sinr_db": None,
        "difference_primary_axis": "",
        "conclusion": "",
        "notes": [],
        "rows": [],
        "strict_tolerance": STRICT_EQUIVALENCE_TOL,
    }
    if not env["sionna_import_ok"]:
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("method\n", encoding="utf-8")
        print(f"Saved Sionna same-realization validation to {out_path}")
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = generate_shared_sionna_channel_bundle(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        noise_var=float(10.0 ** (-float(args.snr_db) / 10.0)),
        device=device,
        seed=args.seed,
    )
    context = build_native_receiver_context(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=float(args.snr_db),
        device=device,
        channel_bundle=bundle,
    )
    result = evaluate_project_vs_sionna_rzf_same_realization(
        context=context,
        device=device,
        strict_tolerance=STRICT_EQUIVALENCE_TOL,
    )

    row = {
        "method": "project_rzf_vs_sionna_rzf_precoder",
        "same_csi_object_used": bool(result["same_csi_object_used"]),
        "same_symbols_used": bool(result["same_symbols_used"]),
        "same_receiver_config_used": bool(result["same_receiver_config_used"]),
        "same_noise_config_used": bool(result["same_noise_config_used"]),
        "project_f_f_shape": result["project_f_f_shape"],
        "sionna_precoder_output_shape": result["sionna_precoder_output_shape"],
        "converted_precoder_output_shape": result["converted_precoder_output_shape"],
        "power_norm_project": result["power_norm_project"],
        "power_norm_sionna": result["power_norm_sionna"],
        "max_abs_diff_f_f_if_comparable": result["max_abs_diff_f_f_if_comparable"],
        "project_sum_rate": result["project_sum_rate"],
        "sionna_sum_rate": result["sionna_sum_rate"],
        "abs_diff_sum_rate": result["abs_diff_sum_rate"],
        "rel_diff_sum_rate": result["rel_diff_sum_rate"],
        "project_symbol_mse": result["project_symbol_mse"],
        "sionna_symbol_mse": result["sionna_symbol_mse"],
        "abs_diff_symbol_mse": result["abs_diff_symbol_mse"],
        "project_sinr_db": result["project_sinr_db"],
        "sionna_sinr_db": result["sionna_sinr_db"],
        "abs_diff_sinr_db": result["abs_diff_sinr_db"],
        "strict_equivalence_claim_allowed": bool(result["strict_equivalence_claim_allowed"]),
        "semantic_compatibility_passed": bool(result["semantic_compatibility_passed"]),
    }
    summary.update(result)
    summary["status"] = "ok"
    summary["same_realization_comparison"] = True
    summary["suggested_for_optional_method_list"] = bool(
        result["semantic_compatibility_passed"] and result["native_receiver_success_sionna"]
    )
    if result["strict_equivalence_claim_allowed"]:
        summary["conclusion"] = "strict_equivalent_under_shared_realization"
    elif result["semantic_compatibility_passed"]:
        summary["conclusion"] = "close_but_different_under_shared_realization"
    else:
        summary["conclusion"] = "incompatible_under_shared_realization"
    summary["notes"] = [
        "This artifact is a same-realization comparison with one shared ExtractedCSI object, one shared symbol batch, and one shared native receiver configuration.",
        "strict_equivalence_claim_allowed only becomes true if F_f, sum-rate, symbol MSE, and SINR all match within tolerance.",
    ]
    summary["rows"] = [row]

    _write_csv(csv_path, [row])
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved Sionna same-realization validation to {out_path}")


if __name__ == "__main__":
    main()
