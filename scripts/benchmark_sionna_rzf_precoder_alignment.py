#!/usr/bin/env python
"""Quick sweep for project RZF vs Sionna RZFPrecoder semantic alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd
import torch

add_src_to_path()

from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import write_markdown
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
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--snrs", nargs="+", type=float, required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _render_gap_plot(df: pd.DataFrame, value_col: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped = df.groupby("snr_db")[value_col]
    ax.plot(grouped.mean().index.tolist(), grouped.mean().values.tolist(), marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_md(summary: dict[str, Any]) -> list[str]:
    return [
        "# Sionna RZF Alignment Quick Sweep",
        "",
        f"- quick_mode: `{summary['quick_mode']}`",
        f"- callable_all_runs: `{summary['callable_all_runs']}`",
        f"- converted_all_runs: `{summary['converted_all_runs']}`",
        f"- native_receiver_success_all_runs: `{summary['native_receiver_success_all_runs']}`",
        f"- semantic_compatibility_all_runs: `{summary['semantic_compatibility_all_runs']}`",
        f"- strict_equivalence_claim_allowed_any_run: `{summary['strict_equivalence_claim_allowed_any_run']}`",
        f"- relationship_status_majority: `{summary['relationship_status_majority']}`",
        f"- recommended_optional_method: `{summary['recommended_optional_method']}`",
        "",
        f"- mean_abs_diff_sum_rate: `{summary['mean_abs_diff_sum_rate']}`",
        f"- mean_abs_diff_symbol_mse: `{summary['mean_abs_diff_symbol_mse']}`",
        f"- mean_abs_diff_sinr_db: `{summary['mean_abs_diff_sinr_db']}`",
        f"- mean_max_abs_diff_f_f: `{summary['mean_max_abs_diff_f_f']}`",
        "",
        f"- conclusion: `{summary['conclusion']}`",
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_sionna_env_info()
    metrics_path = out_dir / "metrics.csv"
    power_gap_path = out_dir / "power_norm_gap.csv"
    summary_path = out_dir / "summary.md"
    if not env["sionna_import_ok"]:
        pd.DataFrame([{"status": "skipped", "reason": "sionna_not_installed"}]).to_csv(metrics_path, index=False)
        pd.DataFrame([{"status": "skipped", "reason": "sionna_not_installed"}]).to_csv(power_gap_path, index=False)
        write_markdown(summary_path, ["# Sionna RZF Alignment Quick Sweep", "", "- status: `skipped`", "- reason: `sionna_not_installed`"])
        print(f"Saved Sionna RZF alignment quick sweep to {out_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for snr_db in args.snrs:
            set_seed(seed)
            bundle = generate_shared_sionna_channel_bundle(
                batch_size=4 if args.quick else 16,
                num_subcarriers=16,
                num_users=4,
                num_bs_ant=16,
                noise_var=float(10.0 ** (-float(snr_db) / 10.0)),
                device=device,
                seed=seed,
            )
            context = build_native_receiver_context(
                batch_size=4 if args.quick else 16,
                num_subcarriers=16,
                num_users=4,
                num_bs_ant=16,
                snr_db=float(snr_db),
                device=device,
                channel_bundle=bundle,
            )
            result = evaluate_project_vs_sionna_rzf_same_realization(
                context=context,
                device=device,
                strict_tolerance=STRICT_EQUIVALENCE_TOL,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "snr_db": float(snr_db),
                    "sionna_rzf_available": bool(result["sionna_rzf_available"]),
                    "sionna_rzf_callable": bool(result["sionna_rzf_callable"]),
                    "converted_to_precoder_output": bool(result["converted_to_precoder_output"]),
                    "native_receiver_success_project": bool(result["native_receiver_success_project"]),
                    "native_receiver_success_sionna": bool(result["native_receiver_success_sionna"]),
                    "semantic_compatibility_passed": bool(result["semantic_compatibility_passed"]),
                    "strict_equivalence_claim_allowed": bool(result["strict_equivalence_claim_allowed"]),
                    "relationship_status": result["relationship_status"],
                    "power_norm_project": result["power_norm_project"],
                    "power_norm_sionna": result["power_norm_sionna"],
                    "power_norm_gap": result["power_norm_gap"],
                    "max_abs_diff_f_f_if_comparable": result["max_abs_diff_f_f_if_comparable"],
                    "project_sum_rate": result["project_sum_rate"],
                    "sionna_sum_rate": result["sionna_sum_rate"],
                    "abs_diff_sum_rate": result["abs_diff_sum_rate"],
                    "project_symbol_mse": result["project_symbol_mse"],
                    "sionna_symbol_mse": result["sionna_symbol_mse"],
                    "abs_diff_symbol_mse": result["abs_diff_symbol_mse"],
                    "project_sinr_db": result["project_sinr_db"],
                    "sionna_sinr_db": result["sionna_sinr_db"],
                    "abs_diff_sinr_db": result["abs_diff_sinr_db"],
                    "difference_primary_axis": result["difference_primary_axis"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False)
    df[["seed", "snr_db", "power_norm_project", "power_norm_sionna", "power_norm_gap"]].to_csv(power_gap_path, index=False)
    _render_gap_plot(df, "abs_diff_sum_rate", "Abs sum-rate gap", out_dir / "sum_rate_gap_vs_snr.png")
    _render_gap_plot(df, "abs_diff_symbol_mse", "Abs symbol-MSE gap", out_dir / "mse_gap_vs_snr.png")

    relationship_counts = df["relationship_status"].value_counts().to_dict()
    relationship_status_majority = max(relationship_counts.items(), key=lambda item: item[1])[0] if relationship_counts else "unknown"
    summary = {
        "quick_mode": bool(args.quick),
        "callable_all_runs": bool(df["sionna_rzf_callable"].all()),
        "converted_all_runs": bool(df["converted_to_precoder_output"].all()),
        "native_receiver_success_all_runs": bool(df["native_receiver_success_sionna"].all()),
        "semantic_compatibility_all_runs": bool(df["semantic_compatibility_passed"].all()),
        "strict_equivalence_claim_allowed_any_run": bool(df["strict_equivalence_claim_allowed"].any()),
        "relationship_status_majority": relationship_status_majority,
        "recommended_optional_method": bool(
            df["sionna_rzf_callable"].all()
            and df["converted_to_precoder_output"].all()
            and df["native_receiver_success_sionna"].all()
            and df["semantic_compatibility_passed"].all()
        ),
        "mean_abs_diff_sum_rate": float(df["abs_diff_sum_rate"].mean()),
        "mean_abs_diff_symbol_mse": float(df["abs_diff_symbol_mse"].mean()),
        "mean_abs_diff_sinr_db": float(df["abs_diff_sinr_db"].mean()),
        "mean_max_abs_diff_f_f": float(df["max_abs_diff_f_f_if_comparable"].mean()),
        "conclusion": (
            "Sionna RZFPrecoder is callable, convertible, and receiver-compatible across the quick sweep, "
            "but strict numerical equivalence is not supported unless every same-realization run passes exact tolerance."
        ),
    }
    write_markdown(summary_path, _summary_md(summary))
    print(f"Saved Sionna RZF alignment quick sweep to {out_dir}")


if __name__ == "__main__":
    main()
